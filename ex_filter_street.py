import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset

from convert_48k_to_16k import convert_audio


def read_street_labels(street_file: Path):
    labels = []
    with street_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                labels.append(parts[1].strip())
    if not labels:
        raise ValueError(f"No labels found in {street_file}")
    return labels


def read_label_id_map(labels_file: Path):
    id_to_name = {}
    with labels_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            label_id, label_name = line.split("\t", 1)
            id_to_name[label_id] = label_name
    if not id_to_name:
        raise ValueError(f"No labels found in {labels_file}")
    return id_to_name


def load_split(dataset_root: Path, split: str):
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' under {data_dir}")
    ds = load_dataset("parquet", data_files={split: [str(p) for p in parquet_files]}, split=split)
    return ds.cast_column("audio", Audio(decode=False))


def build_sample_index(sample, allowed_label_ids):
    events = sample.get("events", [])
    labels = set(sample.get("labels", []))
    human_labels = set(sample.get("human_labels", []))
    event_labels = set()
    for event in events or []:
        if not isinstance(event, dict):
            continue
        for key in ("label", "event_label", "event_name", "human_label", "name"):
            value = event.get(key)
            if value:
                event_labels.add(value)
    labels = {label for label in labels if label in allowed_label_ids}
    human_labels = {label for label in human_labels if label in allowed_label_ids}
    event_labels = {label for label in event_labels if label in allowed_label_ids}
    return sorted(labels | human_labels | event_labels)


def write_split(dataset, out_audio_dir: Path, allowed_label_ids, id_to_name, split_name: str, target_sr: int):
    out_json = {}
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    matched = 0
    for sample in dataset:
        video_id = sample["video_id"]
        labels = build_sample_index(sample, allowed_label_ids)
        if not labels:
            continue
        matched += 1
        wav_name = f"{video_id}.wav"
        audio = sample["audio"]
        if not isinstance(audio, dict):
            raise TypeError(f"{video_id}: expected audio dict, got {type(audio).__name__}")
        if "array" in audio:
            convert_audio(audio["array"], int(audio.get("sampling_rate", target_sr)), out_audio_dir / wav_name, target_sr)
        elif "bytes" in audio and audio["bytes"] is not None:
            import io
            import soundfile as sf

            raw_audio, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            convert_audio(raw_audio, sr, out_audio_dir / wav_name, target_sr)
        elif "path" in audio and audio["path"] and Path(audio["path"]).exists():
            from convert_48k_to_16k import convert_file

            convert_file(Path(audio["path"]), out_audio_dir / wav_name, target_sr)
        else:
            raise TypeError(f"{video_id}: unsupported audio payload keys={list(audio.keys())}")
        events = []
        raw_events = sample.get("events", [])
        for event in raw_events or []:
            if not isinstance(event, dict):
                continue
            label = event.get("label") or event.get("event_label") or event.get("event_name") or event.get("human_label") or event.get("name")
            if label not in allowed_label_ids:
                continue
            start = event.get("start", event.get("onset", 0.0))
            end = event.get("end", event.get("offset", None))
            if end is None:
                continue
            events.append({
                "label": id_to_name[label],
                "start": float(start),
                "end": float(end),
            })
        if not events:
            # Fall back to a clip-level annotation if the row exposes only labels.
            for label in labels:
                events.append({
                    "label": id_to_name[label],
                    "start": 0.0,
                    "end": 10.0,
                })
        if events:
            out_json[wav_name] = events
    print(f"{split_name}: matched {matched} clips, wrote {len(out_json)} files to {out_audio_dir}")
    return out_json


def main():
    parser = argparse.ArgumentParser(description="Filter AudioSet Strong Balanced into HEAR format.")
    parser.add_argument("--source_root", type=str, default="/data/AudioSet-Strong-Balanced")
    parser.add_argument("--output_root", type=str, default="/data/hear_datasets/tasks/audio_set_strong_street")
    parser.add_argument("--street_file", type=str, default="/data/AudioSet-Strong-Balanced/street.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--src_sr", type=int, default=48000)
    parser.add_argument("--dst_sr", type=int, default=16000)
    args = parser.parse_args()

    random.seed(args.seed)

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    audio_root = output_root / str(args.dst_sr)
    train_dir = audio_root / "train"
    valid_dir = audio_root / "valid"
    test_dir = audio_root / "test"
    output_root.mkdir(parents=True, exist_ok=True)

    id_to_name = read_label_id_map(Path(args.source_root) / "labels.txt")
    allowed_labels = read_street_labels(Path(args.street_file))
    allowed_label_ids = {label_id for label_id, label_name in id_to_name.items() if label_name in allowed_labels}
    print(f"street labels: {len(allowed_labels)}")
    print(f"matched label ids: {len(allowed_label_ids)}")
    label_map = pd.DataFrame({"idx": range(len(allowed_labels)), "label": allowed_labels})
    label_map.to_csv(output_root / "labelvocabulary.csv", index=False)

    task_metadata = {
        "task_name": "audio_set_strong_street",
        "version": "hear2021",
        "embedding_type": "event",
        "prediction_type": "multilabel",
        "split_mode": "trainvaltest",
        "sample_duration": 10.0,
        "evaluation": ["event_onset_200ms_fms", "segment_1s_er"],
        "default_mode": "full",
        "split_percentage": {"valid": int(args.valid_ratio * 100), "test": 20, "train": int((1 - args.valid_ratio - 0.2) * 100)},
        "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
        "tmp_dir": "_workdir",
        "mode": "full",
        "splits": ["train", "valid", "test"],
        "download_urls": [],
    }
    (output_root / "task_metadata.json").write_text(json.dumps(task_metadata, indent=1))

    train_ds = load_split(source_root, "train")
    test_ds = load_split(source_root, "test")

    train_items = list(train_ds)
    random.shuffle(train_items)
    n_valid = max(1, int(len(train_items) * args.valid_ratio))
    valid_items = train_items[:n_valid]
    train_items = train_items[n_valid:]

    train_json = write_split(train_items, train_dir, allowed_label_ids, id_to_name, "train", args.dst_sr)
    valid_json = write_split(valid_items, valid_dir, allowed_label_ids, id_to_name, "valid", args.dst_sr)
    test_json = write_split(test_ds, test_dir, allowed_label_ids, id_to_name, "test", args.dst_sr)

    (output_root / "train.json").write_text(json.dumps(train_json, indent=1))
    (output_root / "valid.json").write_text(json.dumps(valid_json, indent=1))
    (output_root / "test.json").write_text(json.dumps(test_json, indent=1))
    print(f"Done: {output_root}")


if __name__ == "__main__":
    main()
