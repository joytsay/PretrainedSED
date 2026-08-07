import argparse
import csv
import io
import json
import random
import shlex
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm


DEFAULT_AUDIOSET_PATH = Path("/data/AudioSet-Strong-Balanced")
DEFAULT_OUTPUT_PATH = Path("/data/hear_datasets/tasks/audio_set_strong_street")
DEFAULT_MID_TO_DISPLAY_NAME = Path(__file__).with_name("mid_to_display_name.tsv")
DCASE_SPLIT_COUNTS = {"train": 44, "valid": 14, "test": 14}
DEFAULT_NO_CROP_LABELS = (
    "Gunshot, gunfire",
    "Firecracker",
    "Explosion",
    "Shatter",
    "Breaking",
)


@dataclass
class EventClip:
    audio: np.ndarray
    sample_rate: int
    label: str
    duration_s: float
    source_id: str
    source_start_s: float
    source_end_s: float


@dataclass
class BackgroundClip:
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    source_id: str


def load_street_classes(path: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    mid_to_label = {}
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if {"mid", "label"}.issubset(df.columns):
            rows = df[["mid", "label"]].itertuples(index=False, name=None)
        elif {"idx", "label"}.issubset(df.columns):
            raise ValueError(
                f"{path} is a label vocabulary, not a class mapping. "
                "Pass a class map with `mid,label` columns or use the default street.txt."
            )
        else:
            raise ValueError(f"Unsupported CSV format in {path}. Expected columns: mid,label")
        for mid, label in rows:
            mid_to_label[str(mid).strip()] = str(label).strip()
    else:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"Expected '<mid>\\t<label>' in {path}, got: {line!r}")
                mid, label = (part.strip() for part in parts)
                mid_to_label[mid] = label
    # Multiple AudioSet MIDs may intentionally map to one canonical task label.
    # Preserve first-seen order while assigning contiguous output indices.
    unique_labels = list(dict.fromkeys(mid_to_label.values()))
    return mid_to_label, {label: idx for idx, label in enumerate(unique_labels)}


def load_mid_to_display_name(path: Path, wanted_mids: set[str]) -> Dict[str, str]:
    mid_to_display_name: Dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Expected '<mid>\\t<display name>' in {path}:{line_number}, got {line!r}"
                )
            mid, display_name = (part.strip() for part in parts)
            if mid in wanted_mids:
                mid_to_display_name[mid] = display_name
    missing = sorted(wanted_mids - set(mid_to_display_name))
    if missing:
        raise ValueError(f"Missing MIDs in {path}: {', '.join(missing)}")
    return mid_to_display_name


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


def matching_target_events(
    events: List[dict],
    matching_mids: set[str],
    mid_to_label: Dict[str, str],
    mid_to_display_name: Dict[str, str],
) -> List[Tuple[dict, str]]:
    """Return only strong intervals explicitly matching a requested AudioSet MID."""
    names_to_labels: Dict[str, set[str]] = defaultdict(set)
    for mid in matching_mids:
        canonical_label = mid_to_label[mid]
        names_to_labels[mid].add(canonical_label)
        names_to_labels[mid_to_display_name[mid]].add(canonical_label)
        names_to_labels[canonical_label].add(canonical_label)

    matches: List[Tuple[dict, str]] = []
    seen: set[Tuple[int, str]] = set()
    for event_index, event in enumerate(events):
        event_names = event_label_values(event)
        for event_name in event_names:
            for canonical_label in names_to_labels.get(event_name, ()):
                key = (event_index, canonical_label)
                if key not in seen:
                    matches.append((event, canonical_label))
                    seen.add(key)
    return matches


def collect_event_clips(
    source_path: Path,
    split: str,
    mid_to_label: Dict[str, str],
    target_sr: int,
    max_sources_per_class: int,
    min_event_s: float,
    max_event_s: float,
    no_crop_labels: set,
    mid_to_display_name: Optional[Dict[str, str]] = None,
) -> Dict[str, List[EventClip]]:
    if mid_to_display_name is None:
        mid_to_display_name = load_mid_to_display_name(
            DEFAULT_MID_TO_DISPLAY_NAME,
            set(mid_to_label),
        )
    reader_name, reader_module = require_reader()
    if reader_name == "datasets":
        rows = iter_rows_with_datasets(source_path, split, reader_module)
        total_rows = None
    else:
        rows = iter_rows_with_pyarrow(source_path, split, reader_module)
        total_rows = None

    clips_by_label: Dict[str, List[EventClip]] = defaultdict(list)
    source_ids_by_label: Dict[str, set] = defaultdict(set)
    seen_intervals_by_label: Dict[str, set[Tuple[str, int, int]]] = defaultdict(set)
    wanted_mids = set(mid_to_label)

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
            progress.set_postfix_str(
                source_clip_summary(clips_by_label, source_ids_by_label, mid_to_label)
            )
            continue

        events = event_list(row)
        matching_events = matching_target_events(
            events,
            matching_mids,
            mid_to_label,
            mid_to_display_name,
        )
        if not matching_events:
            continue

        audio, sr = row_audio_to_array(row["audio"])
        audio = resample_if_needed(audio, sr, target_sr)
        source_id = str(row.get("video_id", "unknown"))

        for event, label in matching_events:
            is_new_source = source_id not in source_ids_by_label[label]
            if is_new_source and len(source_ids_by_label[label]) >= max_sources_per_class:
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
            # Preserve the complete annotated interval for protected impact
            # classes, but still add context around very short annotations.
            if duration_s < min_event_s:
                pad = 0.5 * (min_event_s - duration_s)
                start_s = max(0.0, start_s - pad)
                end_s = min(len(audio) / target_sr, end_s + pad)

            start = int(round(start_s * target_sr))
            end = int(round(end_s * target_sr))
            interval_key = (source_id, start, end)
            if interval_key in seen_intervals_by_label[label]:
                continue
            clip = audio[start:end]
            if clip.size == 0:
                continue
            clip = normalize_event(clip)
            clips_by_label[label].append(
                EventClip(
                    clip,
                    target_sr,
                    label,
                    len(clip) / target_sr,
                    source_id,
                    start / target_sr,
                    end / target_sr,
                )
            )
            seen_intervals_by_label[label].add(interval_key)
            source_ids_by_label[label].add(source_id)

        if all(
            len(source_ids_by_label[label]) >= max_sources_per_class
            for label in set(mid_to_label.values())
        ):
            break
        progress.set_postfix_str(
            source_clip_summary(clips_by_label, source_ids_by_label, mid_to_label)
        )

    missing = sorted({label for label in mid_to_label.values() if not clips_by_label[label]})
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


def source_clip_summary(
    clips_by_label: Dict[str, List[EventClip]],
    source_ids_by_label: Dict[str, set],
    mid_to_label: Dict[str, str],
) -> str:
    labels = set(mid_to_label.values())
    clip_counts = [len(clips_by_label[label]) for label in labels]
    source_counts = [len(source_ids_by_label[label]) for label in labels]
    return (
        f"clips={sum(clip_counts)} sources={sum(source_counts)} "
        f"min_sources={min(source_counts) if source_counts else 0} "
        f"max_sources={max(source_counts) if source_counts else 0}"
    )


def split_event_clips_by_source(
    clips_by_label: Dict[str, List[EventClip]],
    valid_fraction: float,
    rng: random.Random,
) -> Tuple[Dict[str, List[EventClip]], Dict[str, List[EventClip]]]:
    """Split globally by source ID so no recording appears in train and valid."""
    label_sources = {
        label: {clip.source_id for clip in clips}
        for label, clips in clips_by_label.items()
    }
    # A rare class may have only one source recording. Keep that source in
    # training; requiring a positive validation example would either fail the
    # whole generation or leak the recording across the split.
    protected_train_sources = {
        source_id
        for sources in label_sources.values()
        if len(sources) < 2
        for source_id in sources
    }

    valid_source_ids: set[str] = set()
    for label in sorted(label_sources):
        sources = sorted(label_sources[label])
        rng.shuffle(sources)
        target_count = max(1, min(len(sources) - 1, round(len(sources) * valid_fraction)))
        already_selected = len(set(sources).intersection(valid_source_ids))
        if already_selected < target_count:
            candidates = [
                source_id
                for source_id in sources
                if source_id not in valid_source_ids
                and source_id not in protected_train_sources
            ]
            valid_source_ids.update(candidates[: target_count - already_selected])

    source_to_labels: Dict[str, set[str]] = defaultdict(set)
    for label, sources in label_sources.items():
        for source_id in sources:
            source_to_labels[source_id].add(label)

    # A multi-label recording selected for another class can accidentally move
    # every source of a small class into validation. Move a safe source back.
    for label in sorted(label_sources):
        if label_sources[label] - valid_source_ids:
            continue
        candidates = sorted(label_sources[label].intersection(valid_source_ids))
        safe_source = next(
            (
                source_id
                for source_id in candidates
                if all(
                    len(label_sources[other_label].intersection(valid_source_ids)) > 1
                    for other_label in source_to_labels[source_id]
                )
            ),
            None,
        )
        if safe_source is None:
            raise RuntimeError(f"Could not create a source-disjoint split for class {label!r}")
        valid_source_ids.remove(safe_source)

    train: Dict[str, List[EventClip]] = {}
    valid: Dict[str, List[EventClip]] = {}
    for label, clips in clips_by_label.items():
        train[label] = [clip for clip in clips if clip.source_id not in valid_source_ids]
        valid[label] = [clip for clip in clips if clip.source_id in valid_source_ids]
        if not train[label]:
            raise RuntimeError(f"Source split left class {label!r} empty")
        if len(label_sources[label]) >= 2 and not valid[label]:
            raise RuntimeError(f"Source split left class {label!r} without validation data")
    return train, valid


def split_background_clips_by_source(
    clips: List[BackgroundClip],
    valid_fraction: float,
    rng: random.Random,
) -> Tuple[List[BackgroundClip], List[BackgroundClip]]:
    source_ids = sorted({clip.source_id for clip in clips})
    if len(source_ids) < 2:
        raise RuntimeError("Need at least two background sources for train/valid")
    rng.shuffle(source_ids)
    valid_count = max(1, min(len(source_ids) - 1, round(len(source_ids) * valid_fraction)))
    valid_source_ids = set(source_ids[:valid_count])
    train = [clip for clip in clips if clip.source_id not in valid_source_ids]
    valid = [clip for clip in clips if clip.source_id in valid_source_ids]
    return train, valid


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


def shuffled_unique_source_clips(clips: List[EventClip], rng: random.Random) -> List[EventClip]:
    """Return one random event interval per source recording, in random order."""
    clips_by_source: Dict[str, List[EventClip]] = defaultdict(list)
    for clip in clips:
        clips_by_source[clip.source_id].append(clip)

    source_ids = list(clips_by_source)
    rng.shuffle(source_ids)
    return [rng.choice(clips_by_source[source_id]) for source_id in source_ids]


def make_event_clip_queues(
    clips_by_label: Dict[str, List[EventClip]],
    rng: random.Random,
) -> Dict[str, List[EventClip]]:
    """Build split-wide queues in which each source interval can be consumed once."""
    queues = {label: list(clips) for label, clips in clips_by_label.items()}
    for queue in queues.values():
        rng.shuffle(queue)
    return queues


def pop_available_event_clip(
    queue: List[EventClip],
    used_source_ids: set[str],
    max_samples: Optional[int] = None,
) -> Optional[EventClip]:
    """Pop one unused full interval without discarding ineligible intervals."""
    for index in range(len(queue) - 1, -1, -1):
        clip = queue[index]
        if clip.source_id in used_source_ids:
            continue
        if max_samples is not None and len(clip.audio) > max_samples:
            continue
        return queue.pop(index)
    return None


def event_clip_key(clip: EventClip) -> Tuple[str, float, float]:
    return (clip.source_id, clip.source_start_s, clip.source_end_s)


def draw_balanced_class_clips(
    label: str,
    count: int,
    clips_by_label: Dict[str, List[EventClip]],
    clip_queues: Dict[str, List[EventClip]],
    previous_wav_keys: Dict[str, set[Tuple[str, float, float]]],
    rng: random.Random,
) -> List[EventClip]:
    """Draw a class-balanced sequence, avoiding the preceding WAV when possible."""
    originals = clips_by_label[label]
    if not originals:
        raise RuntimeError(f"No source intervals available for class {label!r}")

    selected: List[EventClip] = []
    selected_keys: set[Tuple[str, float, float]] = set()
    while len(selected) < count:
        if not clip_queues[label]:
            clip_queues[label] = list(originals)
            rng.shuffle(clip_queues[label])

        candidates = [
            index
            for index, clip in enumerate(clip_queues[label])
            if event_clip_key(clip) not in selected_keys
            and event_clip_key(clip) not in previous_wav_keys.get(label, set())
        ]
        if not candidates:
            candidates = [
                index
                for index, clip in enumerate(clip_queues[label])
                if event_clip_key(clip) not in selected_keys
            ]
        if not candidates:
            # The class has fewer unique intervals than requested occurrences.
            # Recycle only after all available alternatives have been used.
            clip_queues[label] = list(originals)
            rng.shuffle(clip_queues[label])
            candidates = list(range(len(clip_queues[label])))

        index = rng.choice(candidates)
        clip = clip_queues[label].pop(index)
        selected.append(clip)
        selected_keys.add(event_clip_key(clip))
    previous_wav_keys[label] = {event_clip_key(clip) for clip in selected}
    return selected


def vary_event_audio(
    audio: np.ndarray,
    rng: random.Random,
    min_speed: float = 0.92,
    max_speed: float = 1.08,
) -> Tuple[np.ndarray, float]:
    """Apply a small speed/pitch variation so recycled sources are not identical."""
    speed = rng.uniform(min_speed, max_speed)
    output_length = max(1, int(round(len(audio) / speed)))
    if output_length == len(audio):
        return audio.copy(), speed
    source_positions = np.linspace(0.0, max(0, len(audio) - 1), output_length)
    varied = np.interp(
        source_positions,
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
    return varied, speed


def random_balanced_start(
    duration_s: float,
    clip_len_s: float,
    label: str,
    events: List[dict],
    overlap: float,
    min_gap_s: float,
    rng: random.Random,
) -> float:
    max_start = max(0.0, clip_len_s - duration_s)
    other_class_events = [
        event
        for event in events
        if event["label"] != label and event["label"] != "Gunshot, gunfire"
    ]
    if other_class_events and rng.random() < overlap:
        target = rng.choice(other_class_events)
        target_start = target["start"] / 1000.0
        target_end = target["end"] / 1000.0
        low = max(0.0, target_start - duration_s * 0.75)
        high = min(max_start, target_end - duration_s * 0.25)
        if high >= low:
            return rng.uniform(low, high)

    # Non-overlap placements are deliberately spaced apart. Random retries
    # retain variation without allowing unrelated events to bunch together.
    for _ in range(200):
        candidate = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
        candidate_end = candidate + duration_s
        if all(
            candidate_end + min_gap_s <= event["start"] / 1000.0
            or candidate >= event["end"] / 1000.0 + min_gap_s
            for event in events
        ):
            return candidate
    return rng.uniform(0.0, max_start) if max_start > 0 else 0.0


def random_nonoverlap_start(
    duration_s: float,
    clip_len_s: float,
    events: List[dict],
    min_gap_s: float,
    rng: random.Random,
) -> float:
    """Find a placement that does not touch any existing event."""
    max_start = max(0.0, clip_len_s - duration_s)

    def valid(candidate: float) -> bool:
        candidate_end = candidate + duration_s
        return all(
            candidate_end + min_gap_s <= event["start"] / 1000.0
            or candidate >= event["end"] / 1000.0 + min_gap_s
            for event in events
        )

    for _ in range(300):
        candidate = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
        if valid(candidate):
            return candidate

    boundaries = [0.0]
    boundaries.extend(event["end"] / 1000.0 + min_gap_s for event in events)
    for candidate in sorted(boundaries):
        if candidate <= max_start and valid(candidate):
            return candidate
    # A fully packed clip has no legal gap; retain the no-crop guarantee and
    # place at the latest possible position rather than truncating the shot.
    return max_start


def synthesize_balanced_clip(
    clips_by_label: Dict[str, List[EventClip]],
    clip_queues: Dict[str, List[EventClip]],
    previous_wav_keys: Dict[str, set[Tuple[str, float, float]]],
    background_clips: List[BackgroundClip],
    rng: random.Random,
    sample_rate: int,
    clip_len_s: float,
    events_per_class: int,
    overlap: float,
    balanced_min_gap_s: float,
    gunshot_min_gap_s: float,
    gunshot_max_gap_s: float,
    background_bed_db: float,
) -> Tuple[np.ndarray, List[dict]]:
    """Create one WAV with exactly the same event count for every class."""
    total_samples = int(round(clip_len_s * sample_rate))
    output = np.zeros(total_samples, dtype=np.float32)
    if background_clips:
        background = synthesize_background_clip(
            background_clips,
            rng,
            sample_rate,
            clip_len_s,
        )
        output += background * (10 ** (background_bed_db / 20.0))

    # Rare classes can legitimately have no source in validation after a
    # source-disjoint split. Keep them in the vocabulary, but do not fabricate
    # validation audio for an empty split-specific pool.
    labels = [label for label, clips in clips_by_label.items() if clips]
    rng.shuffle(labels)
    if "Gunshot, gunfire" in labels:
        labels.remove("Gunshot, gunfire")
        labels.insert(0, "Gunshot, gunfire")
    events: List[dict] = []
    for label in labels:
        selection_count = 1 if label == "Gunshot, gunfire" else events_per_class
        selected = draw_balanced_class_clips(
            label,
            selection_count,
            clips_by_label,
            clip_queues,
            previous_wav_keys,
            rng,
        )
        # A gunshot burst deliberately repeats one shot inside this WAV. The
        # selected base changes between WAVs whenever alternatives exist.
        if label == "Gunshot, gunfire":
            selected = [selected[0]] * events_per_class

        burst_cursor_s: Optional[float] = None
        for event_clip in selected:
            event_audio, speed = vary_event_audio(event_clip.audio, rng)
            duration_s = len(event_audio) / sample_rate
            if duration_s > clip_len_s:
                raise RuntimeError(
                    f"Source interval for {label!r} is {duration_s:.2f}s, longer than "
                    f"the requested {clip_len_s:.2f}s WAV."
                )

            if label == "Gunshot, gunfire" and burst_cursor_s is not None:
                start_s = burst_cursor_s + rng.uniform(
                    gunshot_min_gap_s,
                    gunshot_max_gap_s,
                )
                if start_s + duration_s > clip_len_s:
                    burst_cursor_s = None
                elif any(
                    start_s < existing["end"] / 1000.0
                    and start_s + duration_s > existing["start"] / 1000.0
                    and existing["label"] != label
                    for existing in events
                ):
                    start_s = random_nonoverlap_start(
                        duration_s,
                        clip_len_s,
                        events,
                        balanced_min_gap_s,
                        rng,
                    )
            if label != "Gunshot, gunfire" or burst_cursor_s is None:
                if label == "Gunshot, gunfire":
                    start_s = random_nonoverlap_start(
                        duration_s,
                        clip_len_s,
                        events,
                        balanced_min_gap_s,
                        rng,
                    )
                else:
                    start_s = random_balanced_start(
                        duration_s,
                        clip_len_s,
                        label,
                        events,
                        overlap,
                        balanced_min_gap_s,
                        rng,
                    )

            start = int(round(start_s * sample_rate))
            end = start + len(event_audio)
            gain_db = rng.uniform(-6.0, 0.0)
            output[start:end] += event_audio * (10 ** (gain_db / 20.0))
            start_ms = round(start / sample_rate * 1000.0, 3)
            end_ms = round(end / sample_rate * 1000.0, 3)
            events.append(
                {
                    "label": label,
                    "start": start_ms,
                    "end": end_ms,
                    "overlap": any(
                        start_ms < existing["end"] and end_ms > existing["start"]
                        for existing in events
                    ),
                    "source_id": event_clip.source_id,
                    "source_start": round(event_clip.source_start_s * 1000.0, 3),
                    "source_end": round(event_clip.source_end_s * 1000.0, 3),
                    "augmentation_speed": round(speed, 6),
                    "gain_db": round(gain_db, 3),
                }
            )
            if label == "Gunshot, gunfire":
                burst_cursor_s = end / sample_rate

    max_abs = float(np.max(np.abs(output))) if output.size else 0.0
    if max_abs > 0.99:
        output = output / max_abs * 0.99
    return output, sorted(events, key=lambda event: event["start"])


def synthesize_background_clip(
    background_clips: List[BackgroundClip],
    rng: random.Random,
    sample_rate: int,
    clip_len_s: float,
) -> np.ndarray:
    total_samples = int(round(clip_len_s * sample_rate))
    output = np.zeros(total_samples, dtype=np.float32)
    cursor = 0

    backgrounds_by_source: Dict[str, List[BackgroundClip]] = defaultdict(list)
    for clip in background_clips:
        backgrounds_by_source[clip.source_id].append(clip)
    source_ids = list(backgrounds_by_source)
    rng.shuffle(source_ids)
    unique_backgrounds = [rng.choice(backgrounds_by_source[source_id]) for source_id in source_ids]
    for background_clip in unique_backgrounds:
        if cursor >= total_samples:
            break
        clip = background_clip.audio
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
    overlap: float,
    fill_clip: bool,
    clip_queues: Optional[Dict[str, List[EventClip]]] = None,
) -> Tuple[np.ndarray, List[dict]]:
    total_samples = int(round(clip_len_s * sample_rate))
    output = np.zeros(total_samples, dtype=np.float32)
    labels = list(clips_by_label)
    if clip_queues is None:
        clip_queues = make_event_clip_queues(clips_by_label, rng)
    used_source_ids = set()
    events = []
    cursor_s = rng.uniform(0.25, max_gap_s)
    n_events = rng.randint(min_events, max_events)
    placed_events = 0

    while placed_events < n_events or (fill_clip and cursor_s < clip_len_s - min_gap_s):
        available_labels = [
            label for label in labels
            if any(
                clip.source_id not in used_source_ids
                and len(clip.audio) <= int(round((clip_len_s - cursor_s) * sample_rate))
                for clip in clip_queues[label]
            )
        ]
        if not available_labels:
            break
        label = rng.choice(available_labels)
        event_clip = pop_available_event_clip(
            clip_queues[label],
            used_source_ids,
            int(round((clip_len_s - cursor_s) * sample_rate)),
        )
        if event_clip is None:
            continue
        used_source_ids.add(event_clip.source_id)
        event_audio = event_clip.audio
        duration_s = len(event_audio) / sample_rate
        if cursor_s >= clip_len_s:
            break

        # Overlaps are useful between different classes, but same-class overlaps
        # make the target timeline ambiguous. Push the event past any existing
        # event of the same label before rendering it.
        while True:
            candidate_end_s = min(clip_len_s, cursor_s + duration_s)
            candidate_start_ms = round(cursor_s * 1000.0, 3)
            candidate_end_ms = round(candidate_end_s * 1000.0, 3)
            same_label_overlaps = [
                event for event in events
                if event["label"] == label
                and candidate_start_ms < event["end"]
                and candidate_end_ms > event["start"]
            ]
            if not same_label_overlaps:
                break
            cursor_s = max(event["end"] for event in same_label_overlaps) / 1000.0 + min_gap_s
            if cursor_s >= clip_len_s:
                break
        if cursor_s >= clip_len_s:
            break

        if cursor_s + duration_s > clip_len_s:
            # Never keep only the beginning of an event: the audible impulse
            # can occur later in the annotated source interval.
            used_source_ids.remove(event_clip.source_id)
            clip_queues[label].append(event_clip)
            break

        start = int(round(cursor_s * sample_rate))
        end = min(total_samples, start + len(event_audio))
        event_audio = event_audio[: end - start]
        if event_audio.size == 0:
            break

        gain = 10 ** (rng.uniform(-6.0, 0.0) / 20.0)
        output[start:end] += event_audio * gain
        start_ms = round(start / sample_rate * 1000.0, 3)
        end_ms = round(end / sample_rate * 1000.0, 3)
        overlaps_existing = any(start_ms < event["end"] and end_ms > event["start"] for event in events)
        events.append(
            {
                "label": label,
                "start": start_ms,
                "end": end_ms,
                "overlap": overlaps_existing,
                "source_id": event_clip.source_id,
                "source_start": round(event_clip.source_start_s * 1000.0, 3),
                "source_end": round(event_clip.source_end_s * 1000.0, 3),
            }
        )
        event_end_s = end / sample_rate
        if events and rng.random() < overlap:
            max_overlap_s = min(duration_s * 0.75, max(0.0, event_end_s - cursor_s))
            cursor_s = max(0.0, event_end_s - rng.uniform(0.0, max_overlap_s))
        else:
            cursor_s = event_end_s + rng.uniform(min_gap_s, max_gap_s)
        placed_events += 1

    max_abs = float(np.max(np.abs(output))) if output.size else 0.0
    if max_abs > 0.99:
        output = output / max_abs * 0.99
    return output, events


def build_class_audio(
    clips: List[EventClip],
    rng: random.Random,
    sample_rate: int,
    duration_s: float,
    snippet_gap_s: float,
    snippet_gap_jitter_s: float,
    used_source_ids: Optional[set] = None,
    clip_queue: Optional[List[EventClip]] = None,
) -> Tuple[np.ndarray, List[dict]]:
    target_samples = max(1, int(round(duration_s * sample_rate)))
    pieces = []
    segments: List[dict] = []
    total = 0
    used_source_ids = used_source_ids if used_source_ids is not None else set()
    if clip_queue is None:
        clip_queue = list(clips)
        rng.shuffle(clip_queue)
    while total < target_samples:
        remaining = target_samples - total
        event_clip = pop_available_event_clip(
            clip_queue,
            used_source_ids,
            remaining,
        )
        if event_clip is None:
            break
        used_source_ids.add(event_clip.source_id)
        clip = event_clip.audio
        if clip.size == 0:
            continue
        chunk_start = total / sample_rate
        chunk_end = (total + len(clip)) / sample_rate
        pieces.append(clip)
        segments.append(
            {
                "start": round(chunk_start, 3),
                "end": round(chunk_end, 3),
                "source_id": event_clip.source_id,
                "source_start": round(event_clip.source_start_s * 1000.0, 3),
                "source_end": round(event_clip.source_end_s * 1000.0, 3),
            }
        )
        total += len(clip)
        if total >= target_samples:
            break
        gap_s = snippet_gap_s
        if snippet_gap_jitter_s > 0:
            gap_s = max(0.0, gap_s + rng.uniform(-snippet_gap_jitter_s, snippet_gap_jitter_s))
        gap_samples = min(target_samples - total, int(round(gap_s * sample_rate)))
        if gap_samples > 0:
            pieces.append(np.zeros(gap_samples, dtype=np.float32))
            total += gap_samples
    if total < target_samples:
        pieces.append(np.zeros(target_samples - total, dtype=np.float32))
    audio = np.concatenate(pieces)[:target_samples].astype(np.float32, copy=False)
    return audio, segments


def synthesize_equal_class_time_clip(
    clips_by_label: Dict[str, List[EventClip]],
    rng: random.Random,
    sample_rate: int,
    clip_len_s: float,
    overlap: float,
    class_show_time_s: Optional[float],
    equal_class_gap_s: float,
    equal_class_overlap_ratio: float,
    class_snippet_gap_s: float,
    class_snippet_gap_jitter_s: float,
    clip_queues: Optional[Dict[str, List[EventClip]]] = None,
) -> Tuple[np.ndarray, List[dict]]:
    total_samples = int(round(clip_len_s * sample_rate))
    output = np.zeros(total_samples, dtype=np.float32)
    labels = list(clips_by_label)
    rng.shuffle(labels)
    events = []

    if not labels:
        return output, events

    duration_s = class_show_time_s
    if duration_s is None:
        expected_overlaps = max(0, len(labels) - 1) * overlap
        expected_gaps = max(0, len(labels) - 1) - expected_overlaps
        denominator = len(labels) - expected_overlaps * equal_class_overlap_ratio
        duration_s = (clip_len_s - expected_gaps * equal_class_gap_s) / max(1.0, denominator)
    duration_s = min(duration_s, clip_len_s)
    start_s = 0.0
    used_source_ids = set()
    if clip_queues is None:
        clip_queues = make_event_clip_queues(clips_by_label, rng)

    for idx, label in enumerate(labels):
        start = int(round(start_s * sample_rate))
        end = min(total_samples, start + int(round(duration_s * sample_rate)))
        if end <= start:
            continue

        event_audio, class_segments = build_class_audio(
            clips_by_label[label],
            rng,
            sample_rate,
            (end - start) / sample_rate,
            class_snippet_gap_s,
            class_snippet_gap_jitter_s,
            used_source_ids,
            clip_queues[label],
        )
        if not class_segments:
            raise RuntimeError(
                f"No complete unique source interval remains for class {label!r}; "
                "reduce the split count or collect more source intervals."
            )
        gain = 10 ** (rng.uniform(-6.0, 0.0) / 20.0)
        output[start:end] += event_audio[: end - start] * gain

        for segment in class_segments:
            start_ms = round((start / sample_rate + segment["start"]) * 1000.0, 3)
            end_ms = round((start / sample_rate + segment["end"]) * 1000.0, 3)
            overlaps_existing = any(start_ms < event["end"] and end_ms > event["start"] for event in events)
            events.append(
                {
                    "label": label,
                    "start": start_ms,
                    "end": end_ms,
                    "overlap": overlaps_existing,
                    "source_id": segment["source_id"],
                    "source_start": segment["source_start"],
                    "source_end": segment["source_end"],
                }
            )
        event_end_s = end / sample_rate
        if idx < len(labels) - 1:
            if rng.random() < overlap:
                start_s = max(0.0, event_end_s - duration_s * equal_class_overlap_ratio)
            else:
                start_s = event_end_s + equal_class_gap_s

    max_abs = float(np.max(np.abs(output))) if output.size else 0.0
    if max_abs > 0.99:
        output = output / max_abs * 0.99
    return output, sorted(events, key=lambda event: event["start"])


def write_label_vocab(output_path: Path, label_to_idx: Dict[str, int]) -> None:
    with output_path.joinpath("labelvocabulary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label"])
        for label, idx in sorted(label_to_idx.items(), key=lambda item: item[1]):
            writer.writerow([idx, label])


def write_task_metadata(output_path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "description": "Synthetic DCASE2016 Task 2 style dataset from selected AudioSet strong street classes.",
        "sample_rate": args.sample_rate,
        "clip_length_seconds": args.clip_len,
        "source_path": str(args.source_path),
        "class_file": str(args.class_file),
        "mid_to_display_name": str(args.mid_to_display_name),
        "valid_source_fraction": args.valid_source_fraction,
    }
    output_path.joinpath("task_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_command_metadata(output_path: Path, args: argparse.Namespace) -> None:
    """Record the exact invocation and every effective argument value."""

    def json_value(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        return value

    payload = {
        "command": shlex.join(["python", *sys.argv]),
        "argv": list(sys.argv),
        "arguments": {
            key: json_value(value)
            for key, value in sorted(vars(args).items())
        },
        "python_executable": sys.executable,
        "script": str(Path(sys.argv[0]).resolve()),
        "working_directory": str(Path.cwd()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.joinpath("cmd.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


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
    clip_queues = make_event_clip_queues(clips_by_label, rng)
    previous_wav_keys: Dict[str, set[Tuple[str, float, float]]] = defaultdict(set)
    generated_positive_count = 0
    events_per_class = int(getattr(args, "events_per_class", 0))

    for idx in range(count):
        if events_per_class > 0:
            audio, events = synthesize_balanced_clip(
                clips_by_label=clips_by_label,
                clip_queues=clip_queues,
                previous_wav_keys=previous_wav_keys,
                background_clips=background_clips,
                rng=rng,
                sample_rate=args.sample_rate,
                clip_len_s=args.clip_len,
                events_per_class=events_per_class,
                overlap=args.overlap,
                balanced_min_gap_s=args.balanced_min_gap,
                gunshot_min_gap_s=args.gunshot_min_gap,
                gunshot_max_gap_s=args.gunshot_max_gap,
                background_bed_db=args.background_bed_db,
            )
        elif args.equal_class_show_time:
            empty_labels = [label for label, queue in clip_queues.items() if not queue]
            if empty_labels:
                print(
                    f"Stopped {split} after {generated_positive_count}/{count} positive WAVs: "
                    "no unique intervals remain for " + ", ".join(sorted(empty_labels))
                )
                break
            audio, events = synthesize_equal_class_time_clip(
                clips_by_label=clips_by_label,
                rng=rng,
                sample_rate=args.sample_rate,
                clip_len_s=args.clip_len,
                overlap=args.overlap,
                class_show_time_s=args.class_show_time,
                equal_class_gap_s=args.equal_class_gap,
                equal_class_overlap_ratio=args.equal_class_overlap_ratio,
                class_snippet_gap_s=args.class_snippet_gap,
                class_snippet_gap_jitter_s=args.class_snippet_gap_jitter,
                clip_queues=clip_queues,
            )
        else:
            remaining_intervals = sum(len(queue) for queue in clip_queues.values())
            remaining_sources = len(
                {
                    clip.source_id
                    for queue in clip_queues.values()
                    for clip in queue
                }
            )
            available_events = min(remaining_intervals, remaining_sources)
            if available_events == 0:
                print(
                    f"Stopped {split} after {generated_positive_count}/{count} positive WAVs: "
                    "all unique source intervals have been consumed."
                )
                break
            effective_min_events = min(args.min_events, available_events)
            effective_max_events = min(args.max_events, available_events)
            audio, events = synthesize_clip(
                clips_by_label=clips_by_label,
                rng=rng,
                sample_rate=args.sample_rate,
                clip_len_s=args.clip_len,
                min_events=effective_min_events,
                max_events=effective_max_events,
                min_gap_s=args.min_gap,
                max_gap_s=args.max_gap,
                overlap=args.overlap,
                fill_clip=args.fill_clip,
                clip_queues=clip_queues,
            )
        if not events:
            print(
                f"Stopped {split} after {generated_positive_count}/{count} positive WAVs: "
                "the remaining complete intervals could not be placed without cropping."
            )
            break
        filename = f"{split}_{idx:06d}.wav"
        sf.write(audio_dir / filename, audio, args.sample_rate)
        annotations[filename] = events
        generated_positive_count += 1

    background_count = int(round(generated_positive_count * args.background_ratio))
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
    parser.add_argument(
        "--mid_to_display_name",
        type=Path,
        default=DEFAULT_MID_TO_DISPLAY_NAME,
        help="AudioSet MID-to-original-display-name TSV used for strict strong-event matching.",
    )
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_len", type=float, default=120.0)
    parser.add_argument("--train_count", type=int, default=DCASE_SPLIT_COUNTS["train"])
    parser.add_argument("--valid_count", type=int, default=DCASE_SPLIT_COUNTS["valid"])
    parser.add_argument("--test_count", type=int, default=DCASE_SPLIT_COUNTS["test"])
    parser.add_argument("--min_events", type=int, default=24)
    parser.add_argument("--max_events", type=int, default=38)
    parser.add_argument(
        "--events_per_class",
        type=int,
        default=3,
        help=(
            "Balanced mode: place exactly this many events from every class in "
            "every positive WAV. Set to 0 to use the legacy random scheduler."
        ),
    )
    parser.add_argument("--gunshot_min_gap", type=float, default=0.15)
    parser.add_argument("--gunshot_max_gap", type=float, default=0.8)
    parser.add_argument(
        "--balanced_min_gap",
        type=float,
        default=0.5,
        help="Minimum spacing between balanced events that are not intentionally overlapped.",
    )
    parser.add_argument(
        "--background_bed_db",
        type=float,
        default=-30.0,
        help="Gain applied to a continuous background bed in balanced positive WAVs.",
    )
    parser.add_argument("--min_gap", type=float, default=0.5)
    parser.add_argument("--max_gap", type=float, default=4.0)
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Probability of placing an event over an event from another class.",
    )
    parser.add_argument(
        "--no_fill_clip",
        action="store_false",
        dest="fill_clip",
        help="Disable adding extra events to avoid long silent tails.",
    )
    parser.set_defaults(fill_clip=True)
    parser.add_argument(
        "--equal_class_show_time",
        action="store_true",
        help="Give every class equal total show time in each positive clip by concatenating short source events.",
    )
    parser.add_argument(
        "--class_show_time",
        type=float,
        default=None,
        help="Seconds per class for --equal_class_show_time. Defaults to a clip-length-derived value.",
    )
    parser.add_argument(
        "--equal_class_gap",
        type=float,
        default=0.2,
        help="Silent gap in seconds for non-overlapping transitions in --equal_class_show_time mode.",
    )
    parser.add_argument(
        "--equal_class_overlap_ratio",
        type=float,
        default=0.25,
        help="Fraction of class duration to overlap on overlapping transitions in --equal_class_show_time mode.",
    )
    parser.add_argument(
        "--class_snippet_gap",
        type=float,
        default=0.12,
        help="Silent gap between concatenated source snippets inside a class region.",
    )
    parser.add_argument(
        "--class_snippet_gap_jitter",
        type=float,
        default=0.08,
        help="Random jitter added to --class_snippet_gap for a less mechanical pattern.",
    )
    parser.add_argument("--min_event_s", type=float, default=0.3)
    parser.add_argument("--max_event_s", type=float, default=3.0)
    parser.add_argument(
        "--background_ratio",
        type=float,
        default=0.1,
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
        help=(
            "Labels whose complete source event intervals must never be truncated. "
            "Intervals shorter than --min_event_s are still padded with context."
        ),
    )
    parser.add_argument("--max_source_clips_per_class", type=int, default=500)
    parser.add_argument(
        "--valid_source_fraction",
        type=float,
        default=0.2,
        help="Fraction of AudioSet train source recordings reserved exclusively for validation.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Generate one train clip and write an eval_aug preview instead of the full dataset.",
    )
    parser.add_argument(
        "--eval_out_dir",
        type=Path,
        default=Path("eval_aug"),
        help="Output directory for --eval preview PNG and ground-truth CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_path before writing. Without this, generation refuses to overwrite existing data.",
    )
    return parser.parse_args()


def write_eval_preview(output_path: Path, sample_rate: int, eval_out_dir: Path) -> None:
    from eval_aug import events_to_dataframe, read_label_names, write_plot

    split = "train"
    filename = "train_000000.wav"
    audio_path = output_path / str(sample_rate) / split / filename
    annotations = json.load(output_path.joinpath(f"{split}.json").open())
    events = annotations[filename]
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    eval_out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{Path(filename).stem}"
    plot_path = eval_out_dir / f"{stem}.png"
    csv_path = eval_out_dir / f"{stem}_ground_truth.csv"

    write_plot(plot_path, audio, sr, filename, events, read_label_names(output_path))
    events_to_dataframe(events).to_csv(csv_path, index=False)

    print(f"audio={audio_path}")
    print(f"events={len(events)}")
    print(f"plot={plot_path}")
    print(f"ground_truth={csv_path}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.background_ratio < 0:
        raise ValueError("--background_ratio must be >= 0")
    if not 0 <= args.overlap <= 1:
        raise ValueError("--overlap must be between 0 and 1")
    if args.events_per_class < 0:
        raise ValueError("--events_per_class must be >= 0")
    if args.gunshot_min_gap < 0 or args.gunshot_max_gap < args.gunshot_min_gap:
        raise ValueError(
            "--gunshot_min_gap must be >= 0 and --gunshot_max_gap must be >= it"
        )
    if args.balanced_min_gap < 0:
        raise ValueError("--balanced_min_gap must be >= 0")
    if args.background_bed_db > 0:
        raise ValueError("--background_bed_db must be <= 0")
    if not 0 < args.valid_source_fraction < 1:
        raise ValueError("--valid_source_fraction must be between 0 and 1")
    if args.class_show_time is not None and args.class_show_time <= 0:
        raise ValueError("--class_show_time must be > 0")
    if args.equal_class_gap < 0:
        raise ValueError("--equal_class_gap must be >= 0")
    if not 0 <= args.equal_class_overlap_ratio < 1:
        raise ValueError("--equal_class_overlap_ratio must be >= 0 and < 1")
    if args.class_snippet_gap < 0:
        raise ValueError("--class_snippet_gap must be >= 0")
    if args.class_snippet_gap_jitter < 0:
        raise ValueError("--class_snippet_gap_jitter must be >= 0")
    if args.background_ratio > 0 and args.max_background_source_clips <= 0:
        raise ValueError("--max_background_source_clips must be > 0 when --background_ratio is > 0")
    if not args.source_path.is_dir():
        raise FileNotFoundError(f"AudioSet source_path does not exist: {args.source_path}")
    if not args.class_file.is_file():
        raise FileNotFoundError(f"Class file does not exist: {args.class_file}")
    if not args.mid_to_display_name.is_file():
        raise FileNotFoundError(
            f"MID-to-display-name TSV does not exist: {args.mid_to_display_name}"
        )

    if args.output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_path)
    args.output_path.mkdir(parents=True)

    mid_to_label, label_to_idx = load_street_classes(args.class_file)
    mid_to_display_name = load_mid_to_display_name(
        args.mid_to_display_name,
        set(mid_to_label),
    )
    no_crop_labels = set(args.no_crop_labels)
    write_label_vocab(args.output_path, label_to_idx)
    write_task_metadata(args.output_path, args)
    write_command_metadata(args.output_path, args)

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
        mid_to_display_name=mid_to_display_name,
    )
    print({label: len(clips) for label, clips in train_source.items()})
    train_background = collect_background_clips(
        args.source_path,
        "train",
        mid_to_label,
        args.sample_rate,
        0 if args.eval else args.max_background_source_clips if args.background_ratio > 0 else 0,
    )
    if train_background:
        print(f"Collected {len(train_background)} train/valid background clips")

    if args.eval:
        generate_split("train", 1, train_source, [], args.output_path, args, rng)
        args.output_path.joinpath("valid.json").write_text("{}\n")
        args.output_path.joinpath("test.json").write_text("{}\n")
        args.output_path.joinpath(str(args.sample_rate), "valid").mkdir(parents=True, exist_ok=True)
        args.output_path.joinpath(str(args.sample_rate), "test").mkdir(parents=True, exist_ok=True)
        write_eval_preview(args.output_path, args.sample_rate, args.eval_out_dir)
        return

    split_rng = random.Random(args.seed + 1)
    train_source, valid_source = split_event_clips_by_source(
        train_source,
        args.valid_source_fraction,
        split_rng,
    )
    if train_background:
        train_background, valid_background = split_background_clips_by_source(
            train_background,
            args.valid_source_fraction,
            split_rng,
        )
    else:
        valid_background = []
    print(
        "Source-disjoint train clips:",
        {label: len(clips) for label, clips in train_source.items()},
    )
    print(
        "Source-disjoint valid clips:",
        {label: len(clips) for label, clips in valid_source.items()},
    )

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
        mid_to_display_name=mid_to_display_name,
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
    generate_split("valid", split_counts["valid"], valid_source, valid_background, args.output_path, args, rng)
    generate_split("test", split_counts["test"], test_source, test_background, args.output_path, args, rng)

    print(f"Wrote DCASE-style dataset to {args.output_path}")
    print(
        f"Train with: python ex_dcase2016task2.py --task_path={args.output_path} "
        "--model_name=ATST-F --pretrained=strong --lr_decay=0.95 --batch_size 32 "
        f"--n_classes {len(label_to_idx)} --experiment_name AS_GROUP "
        "--wavmix_p 0 --mixup_p 0"
    )


if __name__ == "__main__":
    main()
