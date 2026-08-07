#!/usr/bin/env python3
"""Convert an aug_street.py dataset to and from Label Studio.

Forward conversion creates a Label Studio task JSON and labeling config.  Reverse
conversion applies reviewed annotations to a copy of the original HEAR-compatible
dataset.  Label Studio uses seconds; aug_street.py JSON files use milliseconds.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4


SPLITS = ("train", "valid", "test")
LABEL_COLORS = (
    "#E53935",
    "#43A047",
    "#1E88E5",
    "#FB8C00",
    "#00ACC1",
    "#8E24AA",
    "#FDD835",
    "#6D4C41",
    "#D81B60",
    "#546E7A",
    "#7CB342",
    "#3949AB",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an aug_street.py HEAR subset to/from Label Studio."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Original HEAR-compatible street subset.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Convert a Label Studio export back to a HEAR-compatible subset.",
    )
    parser.add_argument(
        "--labelstudio-export",
        type=Path,
        help="Label Studio JSON or CSV export (required with --reverse).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Reverse output dataset directory (required with --reverse).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Forward task JSON (default: DATASET_DIR/label_studio_import.json).",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        help="Forward labeling config (default: DATASET_DIR/label_studio_config.xml).",
    )
    parser.add_argument(
        "--audio-prefix",
        help=(
            "Forward audio URL prefix. The default is "
            "'/data/local-files/?d=DATASET_NAME'."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        help="Audio directory name; defaults to task_metadata.json sample_rate.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        help="Forward only the selected split; may be repeated.",
    )
    parser.add_argument("--from-name", default="label")
    parser.add_argument("--to-name", default="audio")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument(
        "--drop-unreviewed",
        action="store_true",
        help=(
            "In reverse mode, clear annotations for source tasks absent from the "
            "export. By default their original annotations are retained."
        ),
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help=(
            "In reverse mode, include only tasks present in the Label Studio "
            "export and copy only their WAV files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing reverse output directory.",
    )
    args = parser.parse_args()
    if args.reverse:
        if args.labelstudio_export is None or args.output_dir is None:
            parser.error("--reverse requires --labelstudio-export and --output-dir")
        if args.output_json is not None or args.output_config is not None:
            parser.error("--output-json/--output-config are only for forward conversion")
    elif args.labelstudio_export is not None or args.output_dir is not None:
        parser.error("--labelstudio-export/--output-dir require --reverse")
    return args


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_metadata(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "task_metadata.json"
    return load_json(path) if path.exists() else {}


def resolve_sample_rate(dataset_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    metadata = read_metadata(dataset_dir)
    if "sample_rate" in metadata:
        return int(metadata["sample_rate"])
    numeric_dirs = sorted(
        int(path.name)
        for path in dataset_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if len(numeric_dirs) == 1:
        return numeric_dirs[0]
    raise ValueError("Cannot determine sample rate; pass --sample-rate")


def read_label_vocab(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "labelvocabulary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"idx", "label"}.issubset(rows[0]):
        raise ValueError(f"Expected idx,label columns in {path}")
    return [
        row["label"].strip()
        for row in sorted(rows, key=lambda row: int(row["idx"]))
        if row["label"].strip()
    ]


def read_source_annotations(dataset_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    annotations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing source annotations: {path}")
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in {path}")
        annotations[split] = payload
    return annotations


def build_audio_reference(prefix: str, sample_rate: int, split: str, filename: str) -> str:
    relative = f"{sample_rate}/{split}/{filename}"
    if "?d=" in prefix:
        base, root = prefix.split("?d=", 1)
        root = root.strip("/")
        joined = "/".join(part for part in (root, relative) if part)
        return f"{base}?d={quote(joined, safe='/')}"
    return f"{prefix.rstrip('/')}/{quote(relative, safe='/')}"


def validate_source_event(event: dict[str, Any], filename: str) -> tuple[float, float, str]:
    try:
        start_ms = float(event["start"])
        end_ms = float(event["end"])
        label = str(event["label"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid event in {filename}: {event!r}") from exc
    if start_ms < 0 or end_ms <= start_ms or not label:
        raise ValueError(f"Invalid event in {filename}: {event!r}")
    return start_ms, end_ms, label


def make_tasks(
    dataset_dir: Path,
    annotations: dict[str, dict[str, list[dict[str, Any]]]],
    labels: list[str],
    sample_rate: int,
    splits: tuple[str, ...],
    audio_prefix: str,
    from_name: str,
    to_name: str,
    audio_field: str,
) -> list[dict[str, Any]]:
    known_labels = set(labels)
    tasks: list[dict[str, Any]] = []
    for split in splits:
        audio_dir = dataset_dir / str(sample_rate) / split
        for filename, events in annotations[split].items():
            audio_path = audio_dir / filename
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio file: {audio_path}")
            results = []
            for event in events:
                start_ms, end_ms, label = validate_source_event(event, filename)
                if label not in known_labels:
                    raise ValueError(
                        f"{filename} uses {label!r}, absent from labelvocabulary.csv"
                    )
                results.append(
                    {
                        "id": uuid4().hex[:10],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start_ms / 1000.0,
                            "end": end_ms / 1000.0,
                            "channel": 0,
                            "labels": [label],
                        },
                    }
                )
            tasks.append(
                {
                    "data": {
                        audio_field: build_audio_reference(
                            audio_prefix, sample_rate, split, filename
                        ),
                        "filename": filename,
                        "split": split,
                        "num_events": len(results),
                    },
                    "predictions": [{"model_version": "aug_street", "result": results}],
                }
            )
    return tasks


def build_label_config(
    labels: list[str], from_name: str, to_name: str, audio_field: str
) -> str:
    lines = [
        "<View>",
        f'  <Labels name="{html.escape(from_name, quote=True)}" '
        f'toName="{html.escape(to_name, quote=True)}" zoom="true">',
    ]
    for index, label in enumerate(labels):
        lines.append(
            f'    <Label value="{html.escape(label, quote=True)}" '
            f'background="{LABEL_COLORS[index % len(LABEL_COLORS)]}"/>'
        )
    lines.extend(
        [
            "  </Labels>",
            f'  <Audio name="{html.escape(to_name, quote=True)}" '
            f'value="${html.escape(audio_field, quote=True)}"/>',
            "</View>",
        ]
    )
    return "\n".join(lines) + "\n"


def forward(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    sample_rate = resolve_sample_rate(dataset_dir, args.sample_rate)
    labels = read_label_vocab(dataset_dir)
    annotations = read_source_annotations(dataset_dir)
    splits = tuple(args.split or SPLITS)
    audio_prefix = args.audio_prefix or f"/data/local-files/?d={dataset_dir.name}"
    tasks = make_tasks(
        dataset_dir,
        annotations,
        labels,
        sample_rate,
        splits,
        audio_prefix,
        args.from_name,
        args.to_name,
        args.audio_field,
    )
    output_json = args.output_json or dataset_dir / "label_studio_import.json"
    output_config = args.output_config or dataset_dir / "label_studio_config.xml"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    output_config.write_text(
        build_label_config(
            labels, args.from_name, args.to_name, args.audio_field
        ),
        encoding="utf-8",
    )
    print(f"dataset_dir={dataset_dir}")
    print(f"sample_rate={sample_rate}")
    print(f"splits={','.join(splits)}")
    print(f"tasks={len(tasks)}")
    print(f"labels={len(labels)}")
    print(f"output_json={output_json}")
    print(f"output_config={output_config}")


def task_identity(data: dict[str, Any], audio_field: str) -> tuple[str, str]:
    split = str(data.get("split", "")).strip()
    filename = str(data.get("filename", "")).strip()
    if not filename:
        audio_ref = str(data.get(audio_field, "")).split("?", 1)[0]
        if "?d=" in str(data.get(audio_field, "")):
            audio_ref = str(data[audio_field]).split("?d=", 1)[1]
        filename = Path(audio_ref).name
    if not split and filename:
        prefix = filename.split("_", 1)[0]
        if prefix in SPLITS:
            split = prefix
    if split not in SPLITS or not filename:
        raise ValueError(f"Cannot identify split/filename from task data: {data!r}")
    return split, filename


def value_to_event(value: dict[str, Any], identity: tuple[str, str]) -> dict[str, Any]:
    labels = value.get("labels") or []
    if not labels:
        raise ValueError(f"Annotation has no label for {identity}: {value!r}")
    start_s = float(value["start"])
    end_s = float(value["end"])
    label = str(labels[0]).strip()
    if start_s < 0 or end_s <= start_s or not label:
        raise ValueError(f"Invalid annotation for {identity}: {value!r}")
    return {
        "label": label,
        "start": round(start_s * 1000.0, 3),
        "end": round(end_s * 1000.0, 3),
    }


def result_items_to_events(
    items: list[dict[str, Any]], identity: tuple[str, str]
) -> list[dict[str, Any]]:
    events = []
    for item in items:
        if item.get("type") not in (None, "labels"):
            continue
        value = item.get("value", item)
        if not isinstance(value, dict) or not (value.get("labels") or []):
            continue
        events.append(value_to_event(value, identity))
    return add_overlap(events)


def add_overlap(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events.sort(key=lambda event: (event["start"], event["end"], event["label"]))
    completed: list[dict[str, Any]] = []
    for event in events:
        event["overlap"] = any(
            event["start"] < previous["end"] and event["end"] > previous["start"]
            for previous in completed
        )
        completed.append(event)
    return completed


def latest_annotation(task: dict[str, Any]) -> list[dict[str, Any]] | None:
    annotations = task.get("annotations")
    if isinstance(annotations, list) and annotations:
        usable = [
            annotation
            for annotation in annotations
            if not annotation.get("was_cancelled", False)
        ]
        if usable:
            return usable[-1].get("result") or []
        return None
    # This also permits round-trip testing directly from the generated import.
    predictions = task.get("predictions")
    if isinstance(predictions, list) and predictions:
        return predictions[-1].get("result") or []
    return None


def read_json_export(
    path: Path, audio_field: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    payload = load_json(path)
    if isinstance(payload, dict) and "tasks" in payload:
        payload = payload["tasks"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a Label Studio task list in {path}")
    reviewed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in payload:
        data = task.get("data") or {}
        identity = task_identity(data, audio_field)
        results = latest_annotation(task)
        if results is not None:
            reviewed[identity] = result_items_to_events(results, identity)
    return reviewed


def parse_csv_result(raw: str, identity: tuple[str, str]) -> list[dict[str, Any]]:
    if not raw.strip():
        return []
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("result", [payload])
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in CSV annotation for {identity}")
    return result_items_to_events(payload, identity)


def read_csv_export(
    path: Path, audio_field: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(handle, dialect=dialect))
    reviewed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        identity = task_identity(row, audio_field)
        raw = row.get("label")
        if raw is None:
            raw = row.get("annotations", row.get("result", ""))
        reviewed[identity] = parse_csv_result(raw or "", identity)
    return reviewed


def read_export(
    path: Path, audio_field: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if path.suffix.lower() == ".csv":
        return read_csv_export(path, audio_field)
    return read_json_export(path, audio_field)


def validate_reviewed_tasks(
    reviewed: dict[tuple[str, str], list[dict[str, Any]]],
    source: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    source_keys = {
        (split, filename)
        for split, split_annotations in source.items()
        for filename in split_annotations
    }
    unknown = sorted(set(reviewed) - source_keys)
    if unknown:
        preview = ", ".join(f"{split}/{name}" for split, name in unknown[:5])
        raise ValueError(f"Export contains tasks absent from the source dataset: {preview}")


def merged_annotations(
    source: dict[str, dict[str, list[dict[str, Any]]]],
    reviewed: dict[tuple[str, str], list[dict[str, Any]]],
    drop_unreviewed: bool,
    reviewed_only: bool = False,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in SPLITS:
        merged[split] = {}
        for filename, original in source[split].items():
            key = (split, filename)
            if key in reviewed:
                merged[split][filename] = reviewed[key]
            elif reviewed_only:
                continue
            elif drop_unreviewed:
                merged[split][filename] = []
            else:
                merged[split][filename] = original
    return merged


def prepare_output(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if source_dir == output_dir:
        raise ValueError("--output-dir must differ from dataset_dir")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def write_vocab(path: Path, labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "label"])
        writer.writerows(enumerate(labels))


def reverse(args: argparse.Namespace) -> None:
    source_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    export_path = args.labelstudio_export.resolve()
    sample_rate = resolve_sample_rate(source_dir, args.sample_rate)
    source = read_source_annotations(source_dir)
    reviewed = read_export(export_path, args.audio_field)
    validate_reviewed_tasks(reviewed, source)
    annotations = merged_annotations(
        source,
        reviewed,
        args.drop_unreviewed,
        args.reviewed_only,
    )
    labels = read_label_vocab(source_dir)
    reviewed_labels = {
        event["label"]
        for split_annotations in annotations.values()
        for events in split_annotations.values()
        for event in events
    }
    labels.extend(sorted(reviewed_labels - set(labels)))

    prepare_output(source_dir, output_dir, args.overwrite)
    write_vocab(output_dir / "labelvocabulary.csv", labels)
    metadata_path = source_dir / "task_metadata.json"
    if metadata_path.exists():
        shutil.copy2(metadata_path, output_dir / metadata_path.name)
    for split in SPLITS:
        (output_dir / f"{split}.json").write_text(
            json.dumps(annotations[split], indent=2) + "\n", encoding="utf-8"
        )
        source_audio = source_dir / str(sample_rate) / split
        output_audio = output_dir / str(sample_rate) / split
        if args.reviewed_only:
            output_audio.mkdir(parents=True, exist_ok=True)
            for filename in annotations[split]:
                source_wav = source_audio / filename
                if not source_wav.is_file():
                    raise FileNotFoundError(f"Missing source audio: {source_wav}")
                shutil.copy2(source_wav, output_audio / filename)
        else:
            output_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_audio, output_audio)

    event_count = sum(
        len(events)
        for split_annotations in annotations.values()
        for events in split_annotations.values()
    )
    print(f"source_dataset={source_dir}")
    print(f"labelstudio_export={export_path}")
    print(f"reviewed_tasks={len(reviewed)}")
    print(f"reviewed_only={args.reviewed_only}")
    print(f"events={event_count}")
    print(f"output_dir={output_dir}")


def main() -> None:
    args = parse_args()
    if args.reverse:
        reverse(args)
    else:
        forward(args)


if __name__ == "__main__":
    main()
