import argparse
import csv
import io
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


DEFAULT_AUDIOSET_PATH = Path("/data/AudioSet-Strong-Balanced")
DEFAULT_OUTPUT_PATH = Path("/data/hear_datasets/tasks/audio_set_strong_street")
DCASE_SPLIT_COUNTS = {"train": 44, "valid": 14, "test": 14}
DEFAULT_NO_CROP_LABELS = ("Gunshot, gunfire", "Breaking")


@dataclass
class EventClip:
    audio: np.ndarray
    sample_rate: int
    label: str
    duration_s: float
    source_id: str


@dataclass
class BackgroundClip:
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    source_id: str


def load_street_classes(path: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    mid_to_label = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Expected '<mid>\\t<label>' in {path}, got: {line!r}")
            mid, label = parts
            mid_to_label[mid] = label
    if len(mid_to_label) != 11:
        raise ValueError(f"Expected 11 classes in {path}, found {len(mid_to_label)}")
    return mid_to_label, {label: idx for idx, label in enumerate(mid_to_label.values())}


def require_reader():
    try:
        import datasets  # type: ignore

        return "datasets", datasets
    except ImportError:
        pass

    try:
        import pyarrow.parquet as pq  # type: ignore

        return "pyarrow", pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading /data/AudioSet-Strong-Balanced parquet files requires either "
            "`datasets` or `pyarrow`. Install one in this environment, e.g. "
            "`pip install datasets` or `pip install pyarrow`."
        ) from exc


def parquet_files(source_path: Path, split: str) -> List[Path]:
    files = sorted(source_path.joinpath("data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet files found under {source_path / 'data'}")
    return files


def iter_rows_with_datasets(source_path: Path, split: str, datasets_module) -> Iterable[dict]:
    files = [str(p) for p in parquet_files(source_path, split)]
    dataset = datasets_module.load_dataset(
        "parquet",
        data_files={split: files},
        split=split,
        streaming=True,
    )
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", datasets_module.Audio(decode=False))
    yield from dataset


def iter_rows_with_pyarrow(source_path: Path, split: str, pq_module) -> Iterable[dict]:
    for path in parquet_files(source_path, split):
        table = pq_module.read_table(path)
        for row in table.to_pylist():
            yield row


def row_audio_to_array(audio_obj) -> Tuple[np.ndarray, int]:
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            audio = np.asarray(audio_obj["array"], dtype=np.float32)
            sample_rate = int(audio_obj["sampling_rate"])
            return to_mono(audio), sample_rate
        if "bytes" in audio_obj and audio_obj["bytes"] is not None:
            data, sample_rate = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32")
            return to_mono(data), int(sample_rate)
        if "path" in audio_obj and audio_obj["path"]:
            data, sample_rate = sf.read(audio_obj["path"], dtype="float32")
            return to_mono(data), int(sample_rate)
    raise ValueError(f"Unsupported audio field format: {type(audio_obj)}")


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32, copy=False)


def resample_if_needed(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
        from math import gcd

        div = gcd(source_sr, target_sr)
        return resample_poly(audio, target_sr // div, source_sr // div).astype(np.float32)
    except ImportError:
        import librosa

        return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr).astype(np.float32)


def event_list(row: dict) -> List[dict]:
    events = row.get("events") or []
    if isinstance(events, dict):
        starts = events.get("start", [])
        ends = events.get("end", [])
        names = events.get("event_name", [])
        return [{"start": s, "end": e, "event_name": n} for s, e, n in zip(starts, ends, names)]
    return list(events)


def event_label_values(event: dict) -> List[str]:
    values = []
    for key in ("event_name", "label", "event_label", "human_label", "name"):
        value = event.get(key)
        if value:
            values.append(str(value))
    return values


def useful_events_for_target(events: List[dict], human_labels: set, clip_duration_s: float) -> List[dict]:
    ignore_names = {
        "Background noise",
        "Music",
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Hubbub, speech noise, speech babble",
        "Silence",
        "Inside, small room",
        "Outside, urban or manmade",
        "Outside, rural or natural",
    }
    useful = []
    for event in events:
        name = str(event.get("event_name", ""))
        start = float(event.get("start", 0.0))
        end = float(event.get("end", 0.0))
        if end <= start:
            continue
        if name in ignore_names:
            continue
        if end - start >= 0.95 * clip_duration_s and name not in human_labels:
            continue
        useful.append(event)
    return useful


def collect_event_clips(
    source_path: Path,
    split: str,
    mid_to_label: Dict[str, str],
    target_sr: int,
    max_clips_per_class: int,
    min_event_s: float,
    max_event_s: float,
    no_crop_labels: set,
) -> Dict[str, List[EventClip]]:
    reader_name, reader_module = require_reader()
    if reader_name == "datasets":
        rows = iter_rows_with_datasets(source_path, split, reader_module)
        total_rows = None
    else:
        rows = iter_rows_with_pyarrow(source_path, split, reader_module)
        total_rows = None

    clips_by_label: Dict[str, List[EventClip]] = defaultdict(list)
    wanted_mids = set(mid_to_label)
    label_name_to_label = {label: label for label in mid_to_label.values()}

    progress = tqdm(
        rows,
        total=total_rows,
        desc=f"Collecting {split} source clips",
        unit="row",
        dynamic_ncols=True,
    )
    for row in progress:
        row_labels = set(row.get("labels") or [])
        matching_mids = row_labels.intersection(wanted_mids)
        if not matching_mids:
            progress.set_postfix_str(source_clip_summary(clips_by_label, mid_to_label))
            continue

        events = event_list(row)
        audio, sr = row_audio_to_array(row["audio"])
        audio = resample_if_needed(audio, sr, target_sr)
        source_id = str(row.get("video_id", "unknown"))
        clip_duration_s = len(audio) / target_sr

        matching_events = []
        for event in events:
            event_name = str(event.get("event_name", ""))
            if event_name in wanted_mids:
                matching_events.append((event, mid_to_label[event_name]))
            elif event_name in label_name_to_label:
                matching_events.append((event, label_name_to_label[event_name]))

        # AudioSet strong labels are often more specific human names than the row-level MID.
        # If a row has exactly one requested target MID, use its useful strong intervals.
        if not matching_events and len(matching_mids) == 1:
            target_label = mid_to_label[next(iter(matching_mids))]
            human_labels = set(row.get("human_labels") or [])
            useful_events = useful_events_for_target(events, human_labels, clip_duration_s)
            if not useful_events:
                useful_events = [{"start": 0.0, "end": min(clip_duration_s, max_event_s)}]
            matching_events = [(event, target_label) for event in useful_events]

        for event, label in matching_events:
            if len(clips_by_label[label]) >= max_clips_per_class:
                continue

            start_s = max(0.0, float(event["start"]))
            end_s = min(float(event["end"]), len(audio) / target_sr)
            duration_s = end_s - start_s
            if duration_s <= 0:
                continue

            if label not in no_crop_labels and duration_s > max_event_s:
                center = 0.5 * (start_s + end_s)
                start_s = max(0.0, center - max_event_s / 2)
                end_s = min(len(audio) / target_sr, start_s + max_event_s)
                duration_s = end_s - start_s
            if label not in no_crop_labels and duration_s < min_event_s:
                pad = 0.5 * (min_event_s - duration_s)
                start_s = max(0.0, start_s - pad)
                end_s = min(len(audio) / target_sr, end_s + pad)

            start = int(round(start_s * target_sr))
            end = int(round(end_s * target_sr))
            clip = audio[start:end]
            if clip.size == 0:
                continue
            clip = normalize_event(clip)
            clips_by_label[label].append(EventClip(clip, target_sr, label, len(clip) / target_sr, source_id))

        if all(len(clips_by_label[label]) >= max_clips_per_class for label in mid_to_label.values()):
            break
        progress.set_postfix_str(source_clip_summary(clips_by_label, mid_to_label))

    missing = [label for label in mid_to_label.values() if not clips_by_label[label]]
    if missing:
        raise RuntimeError(f"No source clips found for labels: {missing}")
    return clips_by_label


def collect_background_clips(
    source_path: Path,
    split: str,
    mid_to_label: Dict[str, str],
    target_sr: int,
    max_clips: int,
) -> List[BackgroundClip]:
    if max_clips <= 0:
        return []

    reader_name, reader_module = require_reader()
    if reader_name == "datasets":
        rows = iter_rows_with_datasets(source_path, split, reader_module)
    else:
        rows = iter_rows_with_pyarrow(source_path, split, reader_module)

    wanted_mids = set(mid_to_label)
    wanted_names = set(mid_to_label.values())
    clips: List[BackgroundClip] = []

    progress = tqdm(
        rows,
        desc=f"Collecting {split} background clips",
        unit="row",
        dynamic_ncols=True,
    )
    for row in progress:
        row_labels = set(row.get("labels") or [])
        human_labels = set(row.get("human_labels") or [])
        if row_labels.intersection(wanted_mids) or human_labels.intersection(wanted_names):
            continue

        has_target_event = False
        for event in event_list(row):
            event_labels = set(event_label_values(event))
            if event_labels.intersection(wanted_mids) or event_labels.intersection(wanted_names):
                has_target_event = True
                break
        if has_target_event:
            continue

        audio, sr = row_audio_to_array(row["audio"])
        audio = resample_if_needed(audio, sr, target_sr)
        if audio.size == 0:
            continue

        source_id = str(row.get("video_id", "unknown"))
        audio = normalize_background(audio)
        clips.append(BackgroundClip(audio, target_sr, len(audio) / target_sr, source_id))
        progress.set_postfix_str(f"clips={len(clips)}/{max_clips}")
        if len(clips) >= max_clips:
            break

    if len(clips) < max_clips:
        raise RuntimeError(f"Only found {len(clips)} background clips for {split}, requested {max_clips}")
    return clips


def source_clip_summary(clips_by_label: Dict[str, List[EventClip]], mid_to_label: Dict[str, str]) -> str:
    counts = [len(clips_by_label[label]) for label in mid_to_label.values()]
    return f"clips={sum(counts)} min_class={min(counts) if counts else 0} max_class={max(counts) if counts else 0}"


def normalize_event(audio: np.ndarray, peak: float = 0.85) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    audio = audio - float(np.mean(audio))
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1e-6:
        audio = audio / max_abs * peak
    return audio


def normalize_background(audio: np.ndarray, peak: float = 0.85) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    audio = audio - float(np.mean(audio))
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > peak and max_abs > 1e-6:
        audio = audio / max_abs * peak
    return audio


def synthesize_background_clip(
    background_clips: List[BackgroundClip],
    rng: random.Random,
    sample_rate: int,
    clip_len_s: float,
) -> np.ndarray:
    total_samples = int(round(clip_len_s * sample_rate))
    output = np.zeros(total_samples, dtype=np.float32)
    cursor = 0

    while cursor < total_samples:
        clip = rng.choice(background_clips).audio
        if clip.size == 0:
            continue
        if clip.size > total_samples - cursor:
            max_start = max(0, clip.size - (total_samples - cursor))
            start = rng.randint(0, max_start) if max_start > 0 else 0
            chunk = clip[start:start + (total_samples - cursor)]
        else:
            chunk = clip
        end = min(total_samples, cursor + len(chunk))
        output[cursor:end] = chunk[: end - cursor]
        cursor = end

    gain = 10 ** (rng.uniform(-9.0, -1.0) / 20.0)
    output *= gain
    max_abs = float(np.max(np.abs(output))) if output.size else 0.0
    if max_abs > 0.99:
        output = output / max_abs * 0.99
    return output


def synthesize_clip(
    clips_by_label: Dict[str, List[EventClip]],
    rng: random.Random,
    sample_rate: int,
    clip_len_s: float,
    min_events: int,
    max_events: int,
    min_gap_s: float,
    max_gap_s: float,
) -> Tuple[np.ndarray, List[dict]]:
    total_samples = int(round(clip_len_s * sample_rate))
    output = np.zeros(total_samples, dtype=np.float32)
    labels = list(clips_by_label)
    events = []
    cursor_s = rng.uniform(0.25, max_gap_s)
    n_events = rng.randint(min_events, max_events)

    for _ in range(n_events):
        label = rng.choice(labels)
        event_clip = rng.choice(clips_by_label[label])
        event_audio = event_clip.audio
        duration_s = len(event_audio) / sample_rate
        if cursor_s + duration_s >= clip_len_s:
            break

        start = int(round(cursor_s * sample_rate))
        end = min(total_samples, start + len(event_audio))
        event_audio = event_audio[: end - start]

        gain = 10 ** (rng.uniform(-6.0, 0.0) / 20.0)
        output[start:end] += event_audio * gain
        events.append(
            {
                "label": label,
                "start": round(start / sample_rate * 1000.0, 3),
                "end": round(end / sample_rate * 1000.0, 3),
            }
        )
        cursor_s = end / sample_rate + rng.uniform(min_gap_s, max_gap_s)

    max_abs = float(np.max(np.abs(output))) if output.size else 0.0
    if max_abs > 0.99:
        output = output / max_abs * 0.99
    return output, events


def write_label_vocab(output_path: Path, label_to_idx: Dict[str, int]) -> None:
    with output_path.joinpath("labelvocabulary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label"])
        for label, idx in label_to_idx.items():
            writer.writerow([idx, label])


def write_task_metadata(output_path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "description": "Synthetic DCASE2016 Task 2 style dataset from selected AudioSet strong street classes.",
        "sample_rate": args.sample_rate,
        "clip_length_seconds": args.clip_len,
        "source_path": str(args.source_path),
        "class_file": str(args.class_file),
    }
    output_path.joinpath("task_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def generate_split(
    split: str,
    count: int,
    clips_by_label: Dict[str, List[EventClip]],
    background_clips: List[BackgroundClip],
    output_path: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, List[dict]]:
    audio_dir = output_path.joinpath(str(args.sample_rate), split)
    audio_dir.mkdir(parents=True, exist_ok=True)
    annotations = {}

    for idx in range(count):
        audio, events = synthesize_clip(
            clips_by_label=clips_by_label,
            rng=rng,
            sample_rate=args.sample_rate,
            clip_len_s=args.clip_len,
            min_events=args.min_events,
            max_events=args.max_events,
            min_gap_s=args.min_gap,
            max_gap_s=args.max_gap,
        )
        filename = f"{split}_{idx:06d}.wav"
        sf.write(audio_dir / filename, audio, args.sample_rate)
        annotations[filename] = events

    background_count = int(round(count * args.background_ratio))
    if background_count > 0 and not background_clips:
        raise RuntimeError(f"Cannot generate {split} background negatives without background source clips")
    for idx in range(background_count):
        audio = synthesize_background_clip(
            background_clips=background_clips,
            rng=rng,
            sample_rate=args.sample_rate,
            clip_len_s=args.clip_len,
        )
        filename = f"{split}_background_{idx:06d}.wav"
        sf.write(audio_dir / filename, audio, args.sample_rate)
        annotations[filename] = []

    output_path.joinpath(f"{split}.json").write_text(json.dumps(annotations, indent=2) + "\n")
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a DCASE2016 Task 2 style 11-class street SED dataset from AudioSet-Strong-Balanced."
    )
    parser.add_argument("--source_path", type=Path, default=DEFAULT_AUDIOSET_PATH)
    parser.add_argument("--class_file", type=Path, default=DEFAULT_AUDIOSET_PATH / "street.txt")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_len", type=float, default=120.0)
    parser.add_argument("--train_count", type=int, default=DCASE_SPLIT_COUNTS["train"])
    parser.add_argument("--valid_count", type=int, default=DCASE_SPLIT_COUNTS["valid"])
    parser.add_argument("--test_count", type=int, default=DCASE_SPLIT_COUNTS["test"])
    parser.add_argument("--min_events", type=int, default=24)
    parser.add_argument("--max_events", type=int, default=38)
    parser.add_argument("--min_gap", type=float, default=0.25)
    parser.add_argument("--max_gap", type=float, default=4.0)
    parser.add_argument("--min_event_s", type=float, default=0.8)
    parser.add_argument("--max_event_s", type=float, default=5.0)
    parser.add_argument(
        "--background_ratio",
        type=float,
        default=0.25,
        help="Additional pure-background negative clips per split, relative to the positive clip count.",
    )
    parser.add_argument(
        "--max_background_source_clips",
        type=int,
        default=500,
        help="Maximum non-target AudioSet clips to collect per source split for background negatives.",
    )
    parser.add_argument(
        "--no_crop_labels",
        nargs="*",
        default=list(DEFAULT_NO_CROP_LABELS),
        help="Labels whose source event intervals should not be expanded or truncated.",
    )
    parser.add_argument("--max_source_clips_per_class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_path before writing. Without this, generation refuses to overwrite existing data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.background_ratio < 0:
        raise ValueError("--background_ratio must be >= 0")
    if args.background_ratio > 0 and args.max_background_source_clips <= 0:
        raise ValueError("--max_background_source_clips must be > 0 when --background_ratio is > 0")
    if not args.source_path.is_dir():
        raise FileNotFoundError(f"AudioSet source_path does not exist: {args.source_path}")
    if not args.class_file.is_file():
        raise FileNotFoundError(f"Class file does not exist: {args.class_file}")

    if args.output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_path)
    args.output_path.mkdir(parents=True)

    mid_to_label, label_to_idx = load_street_classes(args.class_file)
    no_crop_labels = set(args.no_crop_labels)
    write_label_vocab(args.output_path, label_to_idx)
    write_task_metadata(args.output_path, args)

    print(f"Collecting train/valid source events from AudioSet train split: {args.source_path}")
    train_source = collect_event_clips(
        args.source_path,
        "train",
        mid_to_label,
        args.sample_rate,
        args.max_source_clips_per_class,
        args.min_event_s,
        args.max_event_s,
        no_crop_labels,
    )
    print({label: len(clips) for label, clips in train_source.items()})
    train_background = collect_background_clips(
        args.source_path,
        "train",
        mid_to_label,
        args.sample_rate,
        args.max_background_source_clips if args.background_ratio > 0 else 0,
    )
    if train_background:
        print(f"Collected {len(train_background)} train/valid background clips")

    print(f"Collecting test source events from AudioSet test split: {args.source_path}")
    test_source = collect_event_clips(
        args.source_path,
        "test",
        mid_to_label,
        args.sample_rate,
        args.max_source_clips_per_class,
        args.min_event_s,
        args.max_event_s,
        no_crop_labels,
    )
    print({label: len(clips) for label, clips in test_source.items()})
    test_background = collect_background_clips(
        args.source_path,
        "test",
        mid_to_label,
        args.sample_rate,
        args.max_background_source_clips if args.background_ratio > 0 else 0,
    )
    if test_background:
        print(f"Collected {len(test_background)} test background clips")

    split_counts = {
        "train": args.train_count,
        "valid": args.valid_count,
        "test": args.test_count,
    }
    generate_split("train", split_counts["train"], train_source, train_background, args.output_path, args, rng)
    generate_split("valid", split_counts["valid"], train_source, train_background, args.output_path, args, rng)
    generate_split("test", split_counts["test"], test_source, test_background, args.output_path, args, rng)

    print(f"Wrote DCASE-style dataset to {args.output_path}")
    print(f"Train with: python ex_dcase2016task2.py --task_path={args.output_path} --model_name=ATST-F --pretrained=strong --lr_decay=0.95 --batch_size 32")


if __name__ == "__main__":
    main()
