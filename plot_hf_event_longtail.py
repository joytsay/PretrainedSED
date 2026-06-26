#!/usr/bin/env python3
"""Summarize event-label counts from AudioSet-Strong parquet shards.

This script scans parquet or Arrow shards such as:
    /data/audioset-humans-reprocessed/data

It writes:
    <out_dir>/event_label_counts.tsv
    <out_dir>/event_label_counts.png
    <out_dir>/event_label.tsv

The TSV contains one row per class with the resolved MID, display name,
event count, clip count, and first source that exposed the label.
"""

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from tqdm import tqdm


DEFAULT_SOURCE_DIR = Path("/data/audioset-humans-reprocessed/data")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LABEL_MAP_PATHS = [
    REPO_ROOT / "ontology/mid_to_display_name.tsv",
    REPO_ROOT / "hf_dataset_gen/metadata/class_labels_indices.csv",
    REPO_ROOT / "hf_dataset_gen/metadata/class_labels_indices_strong.csv",
    REPO_ROOT / "ontology/ontology.json",
]
LABEL_NAME_ALIAS_DROP_WORDS = {"glass", "unmodified", "wild"}
LABEL_NAME_ALIAS_REPLACEMENTS = {
    "ducks": "duck",
}

ROW_FIELD_CANDIDATES = {
    "segment_id": ["segment_id", "segment", "clip_id", "id", "ytid", "video_id"],
    "start": ["start", "onset", "start_time", "start_time_seconds"],
    "end": ["end", "offset", "end_time", "end_time_seconds"],
    "duration": ["duration", "audio_length"],
    "events": ["events", "event", "annotations", "segments"],
    "labels": ["mid", "m_id", "mids", "label", "labels", "event_label", "event_labels"],
    "human_labels": ["human_labels", "event_name", "human_label", "name"],
}

EVENT_FIELD_CANDIDATES = {
    "mid": ["mid", "m_id", "label", "event_label"],
    "name": ["event_name", "human_label", "name", "label"],
    "start": ["start", "onset", "start_time", "start_time_seconds"],
    "end": ["end", "offset", "end_time", "end_time_seconds"],
    "duration": ["duration", "audio_length"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count sound-event labels across AudioSet-Strong parquet or Arrow shards."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing parquet or Arrow shards.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for the TSV and plot. Defaults to <source-dir parent>/event_stats.",
    )
    parser.add_argument(
        "--top-k-labels",
        type=int,
        default=25,
        help="Annotate the top-K labels on the plot.",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Optional two-column TSV of MID/display-name classes for an additional filtered plot.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help=(
            "Optional MID/display-name CSV or TSV used to resolve MID-only events. "
            "Defaults to bundled AudioSet metadata when present."
        ),
    )
    return parser.parse_args()


def first_present(mapping: dict, names: list[str]):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def looks_like_mid(value: str) -> bool:
    return value.startswith(("/m/", "/g/", "/t/"))


def sniff_delimiter(path: Path) -> str:
    with path.open(newline="") as handle:
        sample = handle.read(4096)
    if "\t" in sample and sample.count("\t") >= sample.count(","):
        return "\t"
    return ","


def read_label_map(path: Path) -> dict[str, str]:
    if path.suffix == ".json":
        return read_ontology_label_map(path)

    label_map: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=sniff_delimiter(path))
        header: list[str] | None = None
        for row_index, row in enumerate(reader):
            if not row:
                continue
            normalized = [value.strip() for value in row]
            lowered = [value.lower() for value in normalized]
            if row_index == 0 and (
                {"mid", "display_name"}.issubset(set(lowered))
                or {"m_id", "labels"}.issubset(set(lowered))
                or {"m_id", "human_labels"}.issubset(set(lowered))
            ):
                header = lowered
                continue
            if header is not None:
                values = dict(zip(header, normalized))
                mid = first_present(values, ["mid", "m_id", "label_mid"])
                display_name = first_present(values, ["display_name", "labels", "human_labels", "event_label"])
            elif len(row) >= 3 and looks_like_mid(normalized[1]):
                mid = normalized[1]
                display_name = normalized[2]
            elif len(row) >= 2:
                mid = normalized[0]
                display_name = normalized[1]
            else:
                continue
            mid = (mid or "").strip()
            display_name = (display_name or "").strip()
            if looks_like_mid(display_name) and not looks_like_mid(mid):
                mid, display_name = display_name, mid
            if not looks_like_mid(mid) and looks_like_mid(normalized[0]):
                mid = normalized[0]
            if mid:
                label_map[mid] = display_name
    if not label_map:
        raise ValueError(f"No classes found in {path}")
    return label_map


def read_ontology_label_map(path: Path) -> dict[str, str]:
    with path.open() as handle:
        ontology = json.load(handle)
    label_map: dict[str, str] = {}
    for item in ontology:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        display_name = str(item.get("name") or "").strip()
        if mid:
            label_map[mid] = display_name
    if not label_map:
        raise ValueError(f"No classes found in {path}")
    return label_map


def load_default_label_map() -> dict[str, str]:
    label_map: dict[str, str] = {}
    for path in DEFAULT_LABEL_MAP_PATHS:
        if path.exists():
            label_map.update(read_label_map(path))
    return label_map


def normalize_label_name_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def label_name_alias_key(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    words = [LABEL_NAME_ALIAS_REPLACEMENTS.get(word, word) for word in words]
    words = [word for word in words if word not in LABEL_NAME_ALIAS_DROP_WORDS]
    return " ".join(words)


def label_name_lookup_keys(value: str) -> list[str]:
    keys = [
        value,
        normalize_label_name_key(value),
        label_name_alias_key(value),
    ]
    parts = [part.strip() for part in value.split(",") if part.strip()]
    for end in range(1, len(parts)):
        prefix = ", ".join(parts[:end])
        keys.append(normalize_label_name_key(prefix))
        keys.append(label_name_alias_key(prefix))
    return [key for key in keys if key]


def build_label_name_map(label_map: dict[str, str]) -> dict[str, str]:
    label_name_map: dict[str, str] = {}
    for mid, display_name in label_map.items():
        for key in label_name_lookup_keys(display_name):
            label_name_map.setdefault(key, mid)
    return label_name_map


def lookup_mid_by_name(label_name: str, label_name_map: dict[str, str]) -> str:
    if not label_name:
        return ""
    for key in label_name_lookup_keys(label_name):
        if key in label_name_map:
            return label_name_map[key]
    return ""


def maybe_iterable(value) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))


def normalize_label_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        nested = first_present(value, ROW_FIELD_CANDIDATES["labels"])
        return normalize_label_list(nested)
    if maybe_iterable(value):
        labels: list[str] = []
        for item in value:
            labels.extend(normalize_label_list(item))
        return labels
    return [str(value)]


def normalize_event_list(value) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if maybe_iterable(value):
        return list(value)
    return []


def as_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return None
    return None


def normalize_event_item(value) -> dict | None:
    item = as_plain_dict(value)
    if item is not None:
        return item
    return None


def coerce_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_label_pair(label_mid, label_name) -> tuple[str, str]:
    label_mid_text = str(label_mid).strip() if label_mid is not None else ""
    label_name_text = str(label_name).strip() if label_name is not None else ""
    if looks_like_mid(label_name_text):
        if not label_mid_text or not looks_like_mid(label_mid_text):
            label_mid_text = label_name_text
        label_name_text = ""
    if label_mid_text == label_name_text and looks_like_mid(label_mid_text):
        label_name_text = ""
    return label_mid_text, label_name_text


def serialize_tsv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def pick_present_columns(path: Path, candidates: list[str]) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(path).schema_arrow
    elif path.suffix == ".arrow":
        import pyarrow.ipc as ipc

        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            schema = reader.schema
    else:
        return []
    available = set(schema.names)
    return [name for name in candidates if name in available]


def load_parquet_columns(path: Path, candidates: list[str]) -> pd.DataFrame:
    columns = pick_present_columns(path, candidates)
    if not columns:
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".arrow":
        import pyarrow.ipc as ipc

        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            table = reader.read_all()
        selected = [name for name in columns if name in table.column_names]
        if not selected:
            return pd.DataFrame()
        return table.select(selected).to_pandas()
    return pd.DataFrame()


def extract_segment_id(row: dict, parquet_path: Path, index: int) -> str:
    value = first_present(row, ROW_FIELD_CANDIDATES["segment_id"])
    if value is None:
        return f"{parquet_path.stem}_{index:08d}"
    return Path(str(value)).stem


def iter_row_labels(row: dict):
    default_start = coerce_float(first_present(row, ROW_FIELD_CANDIDATES["start"]), 0.0)
    default_end = coerce_float(first_present(row, ROW_FIELD_CANDIDATES["end"]))
    if default_end is None:
        default_duration = coerce_float(first_present(row, ROW_FIELD_CANDIDATES["duration"]), 10.0)
        default_end = default_start + default_duration

    raw_events = normalize_event_list(first_present(row, ROW_FIELD_CANDIDATES["events"]))
    if raw_events:
        for raw_event in raw_events:
            event = normalize_event_item(raw_event)
            if event is None:
                continue
            label_mid = first_present(event, EVENT_FIELD_CANDIDATES["mid"])
            label_name = first_present(event, EVENT_FIELD_CANDIDATES["name"])
            if label_mid is None and label_name is None:
                continue
            label_mid, label_name = normalize_label_pair(label_mid, label_name)
            start = coerce_float(first_present(event, EVENT_FIELD_CANDIDATES["start"]), default_start)
            end = coerce_float(first_present(event, EVENT_FIELD_CANDIDATES["end"]))
            if end is None:
                duration = coerce_float(first_present(event, EVENT_FIELD_CANDIDATES["duration"]))
                if duration is not None and start is not None:
                    end = start + duration
            if end is None:
                end = default_end
            duration_seconds = max(0.0, (end or 0.0) - (start or 0.0))
            yield (
                label_mid,
                label_name,
                "events",
                duration_seconds,
            )
        return

    mids = normalize_label_list(first_present(row, ROW_FIELD_CANDIDATES["labels"]))
    names = normalize_label_list(first_present(row, ROW_FIELD_CANDIDATES["human_labels"]))
    duration_seconds = max(0.0, (default_end or 0.0) - (default_start or 0.0))
    if mids or names:
        limit = max(len(mids), len(names))
        for idx in range(limit):
            label_mid = mids[idx].strip() if idx < len(mids) else ""
            label_name = names[idx].strip() if idx < len(names) else ""
            label_mid, label_name = normalize_label_pair(label_mid, label_name)
            if label_mid or label_name:
                yield (label_mid, label_name, "row_labels", duration_seconds)


def canonical_key(label_mid: str, label_name: str) -> str:
    if label_mid:
        return f"mid:{label_mid}"
    if label_name:
        return f"name:{label_name}"
    return ""


def collect_counts(source_dir: Path, label_map: dict[str, str] | None = None):
    label_map = label_map or {}
    label_name_map = build_label_name_map(label_map)
    shard_paths = sorted(source_dir.glob("*.parquet")) + sorted(source_dir.glob("*.arrow"))
    if not shard_paths:
        raise FileNotFoundError(f"No parquet or Arrow shards found in {source_dir}")

    event_seconds: Counter[str] = Counter()
    clip_sets: dict[str, set[str]] = {}
    label_rows: dict[str, dict[str, str]] = {}
    missing_rows: list[dict[str, str]] = []

    candidates = (
        ROW_FIELD_CANDIDATES["segment_id"]
        + ROW_FIELD_CANDIDATES["start"]
        + ROW_FIELD_CANDIDATES["end"]
        + ROW_FIELD_CANDIDATES["duration"]
        + ROW_FIELD_CANDIDATES["events"]
        + ROW_FIELD_CANDIDATES["labels"]
        + ROW_FIELD_CANDIDATES["human_labels"]
    )

    for parquet_path in tqdm(shard_paths, desc="Shards", unit="shard"):
        frame = load_parquet_columns(parquet_path, candidates)
        for index, row in enumerate(frame.to_dict(orient="records"), start=1):
            segment_id = extract_segment_id(row, parquet_path, index)
            row_has_missing_label = False
            for label_mid, label_name, source, duration_seconds in iter_row_labels(row):
                label_mid = label_mid or lookup_mid_by_name(label_name, label_name_map)
                resolved_label_name = label_name or label_map.get(label_mid, "")
                if not label_mid or not resolved_label_name or looks_like_mid(resolved_label_name):
                    row_has_missing_label = True
                key = canonical_key(label_mid, label_name)
                if not key:
                    continue
                event_seconds[key] += duration_seconds
                clip_sets.setdefault(key, set()).add(segment_id)
                existing = label_rows.get(key)
                if existing is None or (not existing["event_label"] and label_name):
                    label_rows[key] = {
                        "label_mid": label_mid,
                        "event_label": resolved_label_name or label_mid,
                        "source": source,
                    }
            if row_has_missing_label:
                labels = first_present(row, ROW_FIELD_CANDIDATES["labels"])
                human_labels = first_present(row, ROW_FIELD_CANDIDATES["human_labels"])
                missing_rows.append(
                    {
                        "video_id": serialize_tsv_value(first_present(row, ROW_FIELD_CANDIDATES["segment_id"])),
                        "labels": serialize_tsv_value(labels),
                        "human_labels": serialize_tsv_value(human_labels),
                        "events": serialize_tsv_value(first_present(row, ROW_FIELD_CANDIDATES["events"])),
                    }
                )

    rows = []
    for key, total_seconds in event_seconds.items():
        meta = label_rows[key]
        rows.append(
            {
                "label_mid": meta["label_mid"],
                "event_label": meta["event_label"],
                "event_seconds": total_seconds,
                "clip_count": len(clip_sets.get(key, set())),
                "source": meta["source"],
            }
        )
    rows.sort(key=lambda row: (-row["event_seconds"], row["event_label"], row["label_mid"]))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows, missing_rows


def fill_missing_label_fields(rows: list[dict], label_map: dict[str, str]) -> None:
    label_name_map = build_label_name_map(label_map)
    for row in rows:
        if not row["event_label"] or looks_like_mid(row["event_label"]):
            row["event_label"] = label_map.get(row["label_mid"], row["event_label"])
        if not row["label_mid"]:
            row["label_mid"] = lookup_mid_by_name(row["event_label"], label_name_map)


def write_tsv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["rank", "label_mid", "event_label", "event_seconds", "clip_count"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_event_label_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["event_name", "labels"])
        for row in rows:
            writer.writerow([row["event_label"], row["label_mid"]])


def write_events_missing_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "video_id",
        "labels",
        "human_labels",
        "events",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_title_name(source_dir: Path) -> str:
    source_dir = source_dir.resolve()
    if source_dir.name == "data" and source_dir.parent.name:
        return source_dir.parent.name
    return source_dir.name


def write_plot(
    path: Path,
    rows: list[dict],
    top_k_labels: int,
    source_dir: Path,
    title_suffix: str = "event label long tail",
    annotate_rank_milestones: bool = False,
) -> None:
    ranks = [row["rank"] for row in rows]
    counts_minutes = [row["event_seconds"] / 60.0 for row in rows]

    plt.figure(figsize=(12, 7))
    plt.plot(ranks, counts_minutes, linewidth=2, color="#1f4e79")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Class rank")
    plt.ylabel("Event minutes")
    plt.title(f"{plot_title_name(source_dir)} {title_suffix}")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    for row in rows[: max(0, top_k_labels)]:
        plt.annotate(
            row["event_label"],
            (row["rank"], row["event_seconds"] / 60.0),
            textcoords="offset points",
            xytext=(0, 0),
            fontsize=8,
            rotation=0,
            ha="center",
            va="center",
        )

    if annotate_rank_milestones:
        by_rank = {row["rank"]: row for row in rows}
        for rank in range(25, min(len(rows), 400) + 1, 25):
            row = by_rank.get(rank)
            if row is None:
                continue
            plt.annotate(
                row["event_label"],
                (row["rank"], row["event_seconds"] / 60.0),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=8,
                rotation=90,
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def filter_rows_by_label_map(rows: list[dict], label_map: dict[str, str]) -> list[dict]:
    label_names = set(label_map.values())
    filtered = [
        row for row in rows
        if (row["label_mid"] and row["label_mid"] in label_map) or row["event_label"] in label_names
    ]
    filtered = [row.copy() for row in filtered]
    filtered.sort(key=lambda row: (-row["event_seconds"], row["event_label"], row["label_mid"]))
    for index, row in enumerate(filtered, start=1):
        row["rank"] = index
    return filtered


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.source_dir.resolve().parent / "event_stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = read_label_map(args.label_map) if args.label_map is not None else load_default_label_map()
    rows, missing_rows = collect_counts(args.source_dir, label_map)
    if not rows:
        raise SystemExit(f"No labels found under {args.source_dir}")
    fill_missing_label_fields(rows, label_map)

    tsv_path = out_dir / "event_label_counts.tsv"
    plot_path = out_dir / "event_label_counts.png"
    event_label_path = out_dir / "event_label.tsv"
    events_missing_path = out_dir / "events_missing.tsv"
    write_tsv(tsv_path, rows)
    write_event_label_tsv(event_label_path, rows)
    write_events_missing_tsv(events_missing_path, missing_rows)
    write_plot(
        plot_path,
        rows,
        min(args.top_k_labels, 5),
        args.source_dir,
        annotate_rank_milestones=True,
    )

    filtered_plot_path = None
    if args.tsv is not None:
        label_map = read_label_map(args.tsv)
        filtered_rows = filter_rows_by_label_map(rows, label_map)
        if filtered_rows:
            filtered_plot_path = out_dir / f"event_label_counts__{args.tsv.stem}.png"
            write_plot(
                filtered_plot_path,
                filtered_rows,
                args.top_k_labels,
                args.source_dir,
                title_suffix=f"{args.tsv.stem} event label long tail",
            )

    print(f"source_dir={args.source_dir}")
    print(f"out_dir={out_dir}")
    print(f"classes={len(rows)}")
    print(f"tsv={tsv_path}")
    print(f"event_label={event_label_path}")
    print(f"events_missing={events_missing_path}")
    print(f"plot={plot_path}")
    if filtered_plot_path is not None:
        print(f"filtered_plot={filtered_plot_path}")
    print(f"top_class={rows[0]['event_label']}")
    print(f"top_class_seconds={rows[0]['event_seconds']}")


if __name__ == "__main__":
    main()
