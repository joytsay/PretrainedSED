#!/usr/bin/env python3
"""Render a video with ATST strong-label overlays.

The output video includes:
- top-left confidence bars for classes from a target TSV
- bottom-center mel spectrogram
- bottom-center SED timeline heatmap with a moving playhead
"""

import argparse
import copy
import csv
import gc
import html
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGET_SAMPLE_RATE = 16000
SEGMENT_DURATION = 10.0
SEGMENT_SAMPLES = int(SEGMENT_DURATION * TARGET_SAMPLE_RATE)
MODEL_FRAME_RATE = 100 / 4
MEL_HOP_SAMPLES = 160
DEFAULT_CKPT = REPO_ROOT / "resources" / "ATST-F_strong_1.pt"
DEFAULT_LABEL_TSV = REPO_ROOT / "mid_street_surveillance_10.tsv"
DEFAULT_MID_TO_DISPLAY = REPO_ROOT / "mid_to_display_name.tsv"
DEFAULT_COMMON_LABELS = REPO_ROOT / "common_labels.txt"

# Gradio comparison labels. Change these two names to update the buttons,
# result players, progress messages, and checkpoint lookup keys together.
BASELINE = "ATST-F_strong_1"
ABLATION = "e3jatho"
GRADIO_MODELS = [BASELINE, ABLATION]

GRADIO_CHECKPOINTS = {
    BASELINE: {
        "checkpoint": REPO_ROOT / "resources" / "ATST-F_strong_1.pt",
        "label_vocab": REPO_ROOT / "mid_to_display_name.tsv",
        "n_classes": 447,
        "top_k": 13,
    },
    ABLATION: {
        "checkpoint": REPO_ROOT / "PTSED" / "e3jathho" / "checkpoints" / "epoch=299-step=5400.ckpt",
        "label_vocab": REPO_ROOT / "PTSED" / "e3jathho" / "labelvocabulary.csv",
        "n_classes": 13,
        "top_k": 13,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SED overlay video.")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Input video (required unless --gradio is used).",
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--label-tsv", type=Path, default=None, help="Optional subset of labels to render.")
    parser.add_argument(
        "--label-vocab",
        type=Path,
        default=None,
        help=(
            "Complete CSV/TSV model output vocabulary to use for overlay names and class order, "
            "usually labelvocabulary.csv."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--task", type=str, default="street", choices=["street", "dcase"])
    parser.add_argument("--task-path", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="ATST-F", choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"])
    parser.add_argument("--pretrained", type=str, default="strong", choices=["scratch", "ssl", "weak", "strong"])
    parser.add_argument("--seq-model-type", type=str, default="none", choices=["none", "rnn"])
    parser.add_argument("--n-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=11)
    parser.add_argument(
        "--pos-x",
        choices=["left", "right"],
        default="right",
        help="Horizontal position of the confidence legend.",
    )
    parser.add_argument(
        "--pos-y",
        choices=["top", "bottom"],
        default="top",
        help="Vertical position of the confidence legend.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.012,
        help="Font size as a fraction of frame width for the confidence overlay.",
    )
    parser.add_argument("--gradio", action="store_true", help="Launch the Gradio web interface.")
    parser.add_argument("--videos-dir", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Ask Gradio to create a public share URL.")
    return parser.parse_args()


PARSED_ARGS = parse_args() if __name__ == "__main__" else None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torchaudio
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from data_util.audioset_classes import as_strong_train_classes  # noqa: E402
from eval_test import (  # noqa: E402
    build_config,
    get_task_spec,
    infer_n_classes_from_checkpoint,
    load_model_from_checkpoint,
)


def resolve_binary(env_name: str, default_name: str, fallback_paths: list[str]) -> str:
    env_value = os.environ.get(env_name)
    if env_value:
        candidate = Path(env_value)
        if candidate.exists():
            return str(candidate)
    for candidate in fallback_paths:
        path = Path(candidate)
        if path.exists():
            return str(path)
    resolved = shutil.which(default_name)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Could not locate {default_name}; set {env_name} explicitly.")


FFMPEG_BIN: str | None = None
FFPROBE_BIN: str | None = None


def ffmpeg_bin() -> str:
    global FFMPEG_BIN
    if FFMPEG_BIN is None:
        FFMPEG_BIN = resolve_binary("FFMPEG_BIN", "ffmpeg", ["/usr/bin/ffmpeg", "/opt/conda/bin/ffmpeg"])
    return FFMPEG_BIN


def ffprobe_bin() -> str:
    global FFPROBE_BIN
    if FFPROBE_BIN is None:
        FFPROBE_BIN = resolve_binary("FFPROBE_BIN", "ffprobe", ["/usr/bin/ffprobe", "/opt/conda/bin/ffprobe"])
    return FFPROBE_BIN


class RenderSedModel(nn.Module):
    def __init__(self, model: nn.Module, num_labels: int):
        super().__init__()
        self.model = model
        self.num_labels = num_labels

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(wav))


def load_render_model(args: argparse.Namespace) -> nn.Module:
    task_spec = get_task_spec(args.task)
    if args.task_path is None:
        args.task_path = task_spec["default_task_path"]
    if args.n_classes is None:
        inferred_n_classes = infer_n_classes_from_checkpoint(args.ckpt)
        args.n_classes = inferred_n_classes if inferred_n_classes is not None else task_spec["default_n_classes"]
    if args.experiment_name is None:
        args.experiment_name = task_spec["default_experiment_name"]
    cfg = build_config(args)
    try:
        model = load_model_from_checkpoint(task_spec["plmodule"], args.ckpt, cfg)
    except RuntimeError as exc:
        if "failed finding central directory" in str(exc).lower():
            raise RuntimeError(
                f"Could not read checkpoint archive {args.ckpt}. The file looks truncated or invalid."
            ) from exc
        raise
    return RenderSedModel(model, args.n_classes)


def read_label_tsv(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if sample:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            except csv.Error:
                first_line = sample.splitlines()[0] if sample.splitlines() else ""
                dialect = csv.excel_tab if "\t" in first_line else csv.excel
        else:
            dialect = csv.excel_tab
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames and {"idx", "label"}.issubset(set(reader.fieldnames)):
            for row in reader:
                idx = str(row.get("idx", "")).strip()
                label = str(row.get("label", "")).strip()
                if idx and label:
                    rows.append((idx, label))
        else:
            handle.seek(0)
            for row in csv.reader(handle, dialect=dialect):
                if len(row) >= 2 and row[0].strip():
                    rows.append((row[0].strip(), row[1].strip()))
    if not rows:
        raise ValueError(f"No labels found in {path}")
    return rows


def class_names_from_label_vocab(path: Path) -> list[str]:
    """Read a complete, index-ordered model output vocabulary."""
    rows = read_label_tsv(path)
    indexed_names = []
    for raw_index, name in rows:
        try:
            index = int(raw_index)
        except ValueError:
            # AudioSet's MID-to-display-name file is not itself ordered like the
            # model head. Use it to validate and name the known 447-output order.
            mapped_names = {display_name for _, display_name in rows}
            model_names = pretrained_sed_class_names()
            missing_names = [name for name in model_names if name not in mapped_names]
            if missing_names:
                raise ValueError(
                    f"MID mapping {path} is missing {len(missing_names)} model labels: "
                    + ", ".join(missing_names[:5])
                )
            return model_names
        indexed_names.append((index, name))

    indexed_names.sort(key=lambda item: item[0])
    actual_indices = [index for index, _ in indexed_names]
    expected_indices = list(range(len(indexed_names)))
    if actual_indices != expected_indices:
        raise ValueError(
            f"Label vocabulary {path} must contain each idx from 0 through "
            f"{len(indexed_names) - 1} exactly once; found {actual_indices}."
        )
    return [name for _, name in indexed_names]


def selected_indices_for_tsv(
    label_rows: list[tuple[str, str]],
    model_class_names: list[str],
    num_labels: int,
):
    model_class_names = model_class_names[:num_labels]
    name_to_index = {name.lower(): idx for idx, name in enumerate(model_class_names)}
    indices = []
    names = []
    mids = []
    for label_id, name in label_rows:
        index = None
        if label_id.isdigit():
            candidate = int(label_id)
            if 0 <= candidate < num_labels:
                index = candidate
        if index is None:
            index = name_to_index.get(name.lower())
        if index is None:
            index = name_to_index.get(label_id.lower())
        if index is None:
            raise ValueError(f"Requested label is not in the model output vocabulary: {label_id}, {name}")
        indices.append(index)
        names.append(name or model_class_names[index])
        mids.append(label_id)
    if not indices:
        raise ValueError("No requested labels could be mapped into the model output.")
    return indices, mids, names


def read_mid_to_display_name(path: Path) -> dict[str, str]:
    mid_to_name = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2 and row[0].strip():
                mid_to_name[row[0].strip()] = row[1].strip()
    return mid_to_name


def pretrained_sed_class_names() -> list[str]:
    return list(as_strong_train_classes)


def all_model_class_names(mid_to_display_path: Path, num_labels: int) -> list[str]:
    model_labels = pretrained_sed_class_names()
    if len(model_labels) < num_labels:
        raise ValueError(
            f"PretrainedSED strong label list has {len(model_labels)} labels but the model exposes {num_labels} outputs"
        )
    # PretrainedSED already exposes human-readable class names here.
    # The MID-to-display mapping is only needed when label_tsv supplies MIDs.
    return list(model_labels[:num_labels])


def load_common_label_names(common_labels_path: Path) -> list[str]:
    with common_labels_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def is_pretrained_audioset_ckpt(ckpt_path: Path) -> bool:
    return Path(ckpt_path).name in {"ATST-F_strong_1.pt", "BEATs_strong_1.pt"}


def ffprobe_video(path: Path) -> dict:
    cmd = [
        ffprobe_bin(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]

    def parse_rate(text: str) -> float:
        num, den = text.split("/")
        return float(num) / float(den)

    fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "25/1")
    duration = float(stream.get("duration") or 0.0)
    width = int(stream["width"])
    height = int(stream["height"])
    nb_frames = stream.get("nb_frames")
    frame_count = int(nb_frames) if nb_frames not in (None, "N/A") else max(1, int(round(duration * fps)))
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
    }


def extract_audio(video_path: Path, wav_path: Path):
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    # Force mono so the model sees a single-channel waveform even if the source is stereo.
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
    return wav


def run_inference(model: nn.Module, wav: torch.Tensor, device: str) -> torch.Tensor:
    model = model.to(device).eval()
    wav = wav.to(device)
    waveform_len = wav.shape[1]
    num_chunks = waveform_len // SEGMENT_SAMPLES + int(waveform_len % SEGMENT_SAMPLES != 0)
    chunk_preds = []
    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * SEGMENT_SAMPLES
            end_idx = min((chunk_idx + 1) * SEGMENT_SAMPLES, waveform_len)
            waveform_chunk = wav[:, start_idx:end_idx]
            valid_samples = waveform_chunk.shape[1]
            if valid_samples < SEGMENT_SAMPLES:
                waveform_chunk = torch.nn.functional.pad(
                    waveform_chunk,
                    (0, SEGMENT_SAMPLES - valid_samples),
                )
            chunk_preds.append(model(waveform_chunk))
    predictions = torch.cat(chunk_preds, dim=2)
    expected_frames = max(1, int(math.ceil((waveform_len / TARGET_SAMPLE_RATE) * MODEL_FRAME_RATE)))
    return predictions[:, :, :expected_frames].detach().cpu()


def compute_mel_image(wav: torch.Tensor, panel_width: int, mel_height: int) -> np.ndarray:
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    mel = torchaudio.transforms.MelSpectrogram(
        TARGET_SAMPLE_RATE,
        f_min=60,
        f_max=7800,
        hop_length=MEL_HOP_SAMPLES,
        win_length=1024,
        n_fft=1024,
        n_mels=64,
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel)[0].numpy()
    mel_db = (mel_db - mel_db.min()) / max(1e-6, mel_db.max() - mel_db.min())

    fig = plt.figure(figsize=(panel_width / 100, mel_height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return image


def compute_timeline_image(scores: np.ndarray, class_names: list[str], panel_width: int, timeline_height: int) -> np.ndarray:
    fig = plt.figure(figsize=(panel_width / 100, timeline_height / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.imshow(scores, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white", length=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return image


def load_font(size: int):
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    return ImageFont.truetype(font_path, size=size)


def font_text_height(draw: ImageDraw.ImageDraw, font) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1]


def compose_frame(
    frame_rgb: np.ndarray,
    current_scores: np.ndarray,
    class_names: list[str],
    current_time: float,
    duration: float,
    mel_image: np.ndarray,
    timeline_image: np.ndarray,
    panel_width: int,
    mel_height: int,
    timeline_height: int,
    top_k: int,
    font,
    pos_x: str = "left",
    pos_y: str = "top",
) -> np.ndarray:
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    entries = [
        item
        for item in zip(class_names, current_scores.tolist())
    ]
    entries.sort(key=lambda item: item[1], reverse=True)
    entries = entries[:top_k]

    bar_width = 260
    text_height = font_text_height(draw, font)
    score_bbox = draw.textbbox((0, 0), "0.0000 (0.00%)", font=font)
    score_height = score_bbox[3] - score_bbox[1]
    row_height = max(text_height, score_height) + 10
    label_widths = [draw.textbbox((0, 0), label, font=font)[2] for label, _ in entries]
    score_widths = [
        draw.textbbox((0, 0), f"{score:.4f} ({score * 100:.2f}%)", font=font)[2]
        for _, score in entries
    ]
    label_column_width = max(label_widths) if label_widths else 0
    score_column_width = max(score_widths) if score_widths else 0
    row_width = label_column_width + 18 + score_column_width
    block_width = max(bar_width, row_width)
    block_height = len(entries) * row_height + 8
    left = 24 if pos_x == "left" else width - block_width - 36
    top = 24 if pos_y == "top" else height - block_height - 12
    draw.rectangle(
        (left - 12, top - 12, left - 12 + block_width + 24, top - 12 + block_height),
        fill=(0, 0, 0),
    )
    for idx, (label, score) in enumerate(entries):
        row_top = top + idx * row_height
        row_mid = row_top + row_height / 2
        draw.text((left, row_top), label, font=font, fill=(255, 255, 255))
        # Probabilities can be very small for sparse SED labels; show both raw and percent.
        score_text = f"{score:.4f} ({score * 100:.2f}%)"
        score_x = left + label_column_width + 18
        score_text_y = row_top
        draw.text(
            (score_x, score_text_y),
            score_text,
            font=font,
            fill=(255, 255, 255),
        )
        bar_y = row_top + max(text_height, score_height) + 2
        draw.rectangle((left, bar_y, left + block_width, bar_y + 6), fill=(40, 40, 40))
        draw.rectangle(
            (left, bar_y, left + int(block_width * max(0.0, min(1.0, score))), bar_y + 6),
            fill=(0, 200, 140),
        )

    panel_x = 24
    panel_y = height - (mel_height + timeline_height) - 24
    mel_panel = Image.fromarray(mel_image)
    timeline_panel = Image.fromarray(timeline_image)
    image.paste(mel_panel, (panel_x, panel_y))
    image.paste(timeline_panel, (panel_x, panel_y + mel_height))

    cursor_x = panel_x + int((current_time / max(duration, 1e-6)) * panel_width)
    draw.line((cursor_x, panel_y, cursor_x, panel_y + mel_height + timeline_height), fill=(255, 40, 40), width=3)
    return np.asarray(image)


def open_video_reader(video_path: Path, width: int, height: int):
    cmd = [
        ffmpeg_bin(),
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def open_video_writer(output_path: Path, width: int, height: int, fps: float, audio_path: Path):
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def prepare_render(args: argparse.Namespace) -> tuple[nn.Module, list[int], list[str]]:
    """Load the model and resolve the ordered labels once."""
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    task_spec = get_task_spec(args.task)
    if args.task_path is None:
        args.task_path = task_spec["default_task_path"]
    if args.label_vocab is not None:
        task_class_names = class_names_from_label_vocab(args.label_vocab)
        vocabulary_description = str(args.label_vocab)
    elif (Path(args.task_path) / "labelvocabulary.csv").exists():
        label_vocab, _ = task_spec["label_vocab_nlabels"](Path(args.task_path))
        label_vocab = label_vocab.sort_values("idx")
        task_class_names = list(label_vocab["label"].astype(str))
        vocabulary_description = str(Path(args.task_path) / "labelvocabulary.csv")
    elif is_pretrained_audioset_ckpt(args.ckpt):
        task_class_names = pretrained_sed_class_names()
        vocabulary_description = "the built-in AudioSet Strong vocabulary"
    else:
        raise FileNotFoundError(
            f"No label vocabulary found under {args.task_path}. Pass --label-vocab for this checkpoint."
        )

    model = load_render_model(args)
    if len(task_class_names) != model.num_labels:
        raise ValueError(
            f"Class count mismatch: vocabulary {vocabulary_description} has "
            f"{len(task_class_names)} labels, but the model exposes {model.num_labels} outputs."
        )
    if args.label_tsv is None:
        class_indices = list(range(model.num_labels))
        class_names = task_class_names
    else:
        label_rows = read_label_tsv(args.label_tsv)
        class_indices, _, class_names = selected_indices_for_tsv(
            label_rows,
            task_class_names,
            model.num_labels,
        )
    return model, class_indices, class_names


def render_video(
    args: argparse.Namespace,
    model: nn.Module,
    class_indices: list[int],
    class_names: list[str],
    video_path: Path,
    output_path: Path,
    progress=None,
) -> Path:
    """Run SED inference and render a browser-compatible overlay video."""
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def report(fraction: float, description: str) -> None:
        if progress is not None:
            progress(fraction, desc=description)

    with tempfile.TemporaryDirectory(prefix="sed_overlay_") as tmp_dir:
        audio_path = Path(tmp_dir) / "audio.wav"
        report(0.01, "Extracting audio")
        extract_audio(video_path, audio_path)
        wav = load_audio(audio_path)
        report(0.05, "Running sound-event detection")
        predictions = run_inference(model, wav, args.device)[0, class_indices].numpy()
        print("selected_labels=" + ", ".join(
            f"{index}:{name}" for index, name in zip(class_indices, class_names)
        ))
        print(
            f"score_stats=min={float(predictions.min()):.4f} "
            f"mean={float(predictions.mean()):.4f} max={float(predictions.max()):.4f}"
        )

        meta = ffprobe_video(video_path)
        frame_width = meta["width"]
        frame_height = meta["height"]
        fps = meta["fps"]
        duration = max(meta["duration"], wav.shape[-1] / TARGET_SAMPLE_RATE)
        frame_count = meta["frame_count"]
        score_frames = predictions.shape[1]
        panel_width = max(1, int(frame_width * 0.25))
        mel_height = max(1, int(panel_width * 60 / 260))
        timeline_height = max(1, int(panel_width * 75 / 260))
        mel_image = compute_mel_image(wav, panel_width, mel_height)
        timeline_image = compute_timeline_image(predictions, class_names, panel_width, timeline_height)

        font_size = max(10, int(frame_width * args.font_scale))
        font = load_font(font_size)

        report(0.12, "Preparing overlays")
        reader = open_video_reader(video_path, frame_width, frame_height)
        writer = open_video_writer(output_path, frame_width, frame_height, fps, audio_path)
        reader_stderr = b""
        writer_stderr = b""
        try:
            frame_bytes = frame_width * frame_height * 3
            for frame_index in range(frame_count):
                raw = reader.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(frame_height, frame_width, 3).copy()
                current_time = frame_index / fps
                score_index = min(score_frames - 1, max(0, int(current_time * MODEL_FRAME_RATE)))
                current_scores = predictions[:, score_index]
                composed = compose_frame(
                    frame,
                    current_scores,
                    class_names,
                    current_time,
                    duration,
                    mel_image,
                    timeline_image,
                    panel_width,
                    mel_height,
                    timeline_height,
                    args.top_k,
                    font,
                    args.pos_x,
                    args.pos_y,
                )
                try:
                    writer.stdin.write(composed.tobytes())
                except BrokenPipeError as exc:
                    if reader.stdout:
                        reader.stdout.close()
                    if writer.stdin:
                        writer.stdin.close()
                    reader_stderr = reader.stderr.read() if reader.stderr else b""
                    writer_stderr = writer.stderr.read() if writer.stderr else b""
                    raise RuntimeError(
                        "ffmpeg video writer exited early while receiving frames:\n"
                        + writer_stderr.decode("utf-8", errors="replace")
                    ) from exc
                if frame_index % max(1, int(fps)) == 0:
                    report(0.15 + 0.84 * (frame_index + 1) / frame_count, "Rendering video")
        finally:
            if reader.stdout:
                reader.stdout.close()
            reader_stderr = reader.stderr.read() if reader.stderr else b""
            if reader.stderr:
                reader.stderr.close()
            reader.wait()
            if writer.stdin:
                writer.stdin.close()
            writer_stderr = writer.stderr.read() if writer.stderr else b""
            if writer.stderr:
                writer.stderr.close()
            writer.wait()

        if reader.returncode not in (0, None):
            raise RuntimeError(
                "ffmpeg video reader failed:\n"
                + reader_stderr.decode("utf-8", errors="replace")
            )
        if writer.returncode not in (0, None):
            raise RuntimeError(
                "ffmpeg video writer failed:\n"
                + writer_stderr.decode("utf-8", errors="replace")
            )
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output was not created: {output_path}")

    report(1.0, "Done")
    print(f"video={video_path}")
    print(f"ckpt={args.ckpt}")
    print(f"label_vocab={args.label_vocab}")
    print(f"label_tsv={args.label_tsv}")
    print(f"classes={len(class_names)}")
    print(f"output={output_path}")
    return output_path


def discover_example_videos(videos_dir: Path) -> list[Path]:
    extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    if not videos_dir.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in videos_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: path.name.lower(),
    )


def label_badges_html(label_vocab_path: Path, ordered_labels: list[str] | None = None) -> str:
    labels = ordered_labels or [name for _, name in read_label_tsv(label_vocab_path)]
    badges = []
    for index, label in enumerate(labels):
        key = str(index + 1)
        hue = (len(label) * 137.508) % 360
        color = f"hsl({hue:.2f}, 70%, 52%)"
        key_markup = (
            f'<kbd style="border:0;background:transparent;color:var(--body-text-color-subdued);'
            f'font:13px ui-monospace,monospace;">{html.escape(key)}</kbd>'
            if key
            else ""
        )
        badges.append(
            f"""
            <span style="display:inline-flex;align-items:center;gap:12px;
                         padding:7px 10px;border-left:4px solid {color};
                         border-radius:4px;background:var(--background-fill-secondary);
                         color:var(--body-text-color);font-size:15px;line-height:1;">
                <span>{html.escape(label)}</span>
                {key_markup}
            </span>
            """
        )
    return (
        "<div style=\"display:flex;flex-wrap:wrap;gap:8px;padding:12px;"
        "height:84px;overflow-y:auto;margin-bottom:8px;border-radius:8px;"
        "background:var(--background-fill-primary);\">"
        + "".join(badges)
        + "</div>"
    )


def launch_gradio(args: argparse.Namespace) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed. Run: pip install gradio") from exc

    examples = discover_example_videos(args.videos_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_name, ablation_name = GRADIO_MODELS
    model_cache = {"name": None, "bundle": None}

    def get_renderer(checkpoint_name: str):
        if checkpoint_name not in GRADIO_CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint selection: {checkpoint_name}")
        if model_cache["name"] == checkpoint_name:
            return model_cache["bundle"]

        old_bundle = model_cache["bundle"]
        if old_bundle is not None:
            old_bundle[1].to("cpu")
            model_cache["name"] = None
            model_cache["bundle"] = None
            del old_bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        preset = GRADIO_CHECKPOINTS[checkpoint_name]
        checkpoint_path = Path(preset["checkpoint"])
        label_vocab_path = Path(preset["label_vocab"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        if not label_vocab_path.is_file():
            raise FileNotFoundError(f"Label vocabulary does not exist: {label_vocab_path}")

        render_args = copy.copy(args)
        render_args.ckpt = checkpoint_path
        render_args.label_vocab = label_vocab_path
        render_args.n_classes = int(preset["n_classes"])
        render_args.top_k = int(preset["top_k"])
        model, class_indices, class_names = prepare_render(render_args)
        bundle = (render_args, model, class_indices, class_names)
        model_cache["name"] = checkpoint_name
        model_cache["bundle"] = bundle
        return bundle

    def render_preset(video, checkpoint_name, progress):
        if not video:
            raise ValueError("Choose or upload a video first.")
        render_args, model, class_indices, class_names = get_renderer(checkpoint_name)
        source = Path(video)
        output = output_dir / f"{source.stem}__{render_args.ckpt.stem}__sed_overlay.mp4"
        return str(
            render_video(
                render_args,
                model,
                class_indices,
                class_names,
                source,
                output,
                progress,
            )
        )

    def render_baseline_ui(video, progress=gr.Progress()):
        try:
            return render_preset(video, baseline_name, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def render_ablation_ui(video, progress=gr.Progress()):
        try:
            return render_preset(video, ablation_name, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def render_both_ui(video, progress=gr.Progress()):
        if not video:
            raise gr.Error("Choose or upload a video first.")
        try:
            baseline_progress = lambda fraction, desc=None: progress(
                fraction * 0.5,
                desc=f"{baseline_name}: {desc or 'working'}",
            )
            ablation_progress = lambda fraction, desc=None: progress(
                0.5 + fraction * 0.5,
                desc=f"{ablation_name}: {desc or 'working'}",
            )
            baseline_output = render_preset(video, baseline_name, baseline_progress)
            ablation_output = render_preset(video, ablation_name, ablation_progress)
            return baseline_output, ablation_output
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="PretrainedSED Video Overlay") as demo:
        gr.Markdown(
            "# Sound Event Detection (SED) Video Overlay\n"
            "Choose one of the repository videos below or upload your own, then render the SED overlay."
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(
                    label="Input video",
                    sources=["upload"],
                    include_audio=True,
                )
            with gr.Column(scale=1):
                if examples:
                    gr.Examples(
                        examples=[[str(path)] for path in examples],
                        inputs=input_video,
                        label=f"Example videos from {args.videos_dir.resolve()}",
                        examples_per_page=20,
                    )
                else:
                    gr.Markdown(f"No example videos found in `{args.videos_dir.resolve()}`.")
        with gr.Row():
            render_both_button = gr.Button("1. Render both", variant="primary")
            render_baseline_button = gr.Button(f"2. Render baseline {baseline_name}")
            render_ablation_button = gr.Button(f"3. Render ablation {ablation_name}")
        with gr.Row():
            with gr.Column(scale=1):
                baseline_label_path = Path(GRADIO_CHECKPOINTS[baseline_name]["label_vocab"])
                gr.Markdown(f"### {baseline_name} : {len(read_label_tsv(baseline_label_path))} labels")
                gr.HTML(label_badges_html(
                    baseline_label_path,
                    pretrained_sed_class_names(),
                ))
            with gr.Column(scale=1):
                ablation_label_path = Path(GRADIO_CHECKPOINTS[ablation_name]["label_vocab"])
                gr.Markdown(f"### {ablation_name} : {len(read_label_tsv(ablation_label_path))} labels")
                gr.HTML(label_badges_html(ablation_label_path))
        with gr.Row():
            baseline_output_video = gr.Video(
                label=f"Baseline {baseline_name} result",
                format="mp4",
                interactive=False,
                elem_id="baseline-output-video",
            )
            ablation_output_video = gr.Video(
                label=f"Ablation {ablation_name} result",
                format="mp4",
                interactive=False,
                elem_id="ablation-output-video",
            )
        with gr.Row():
            match_ablation_button = gr.Button("Match time to ablation")
            match_baseline_button = gr.Button("Match time to baseline")
        match_baseline_button.click(
            fn=None,
            queue=False,
            js="""
            () => {
                const baseline = document.querySelector('#baseline-output-video video');
                const ablation = document.querySelector('#ablation-output-video video');
                if (!baseline || !ablation) {
                    alert('Render both videos before matching their playheads.');
                    return [];
                }
                ablation.currentTime = baseline.currentTime;
                return [];
            }
            """,
        )
        match_ablation_button.click(
            fn=None,
            queue=False,
            js="""
            () => {
                const baseline = document.querySelector('#baseline-output-video video');
                const ablation = document.querySelector('#ablation-output-video video');
                if (!baseline || !ablation) {
                    alert('Render both videos before matching their playheads.');
                    return [];
                }
                baseline.currentTime = ablation.currentTime;
                return [];
            }
            """,
        )
        with gr.Row():
            play_pause_both_button = gr.Button("▶️ Play", elem_id="toggle-playback-button")
            restart_both_button = gr.Button("⏮️ Replay")
            backward_one_second_button = gr.Button("⏪ Backward 1 sec")
            forward_one_second_button = gr.Button("Forward 1 sec ⏩")
        play_pause_both_button.click(
            fn=None,
            queue=False,
            js="""
            () => {
                const baseline = document.querySelector('#baseline-output-video video');
                const ablation = document.querySelector('#ablation-output-video video');
                if (!baseline || !ablation) {
                    alert('Render both videos before controlling playback.');
                    return [];
                }
                const container = document.querySelector('#toggle-playback-button');
                const button = container?.matches('button')
                    ? container
                    : container?.querySelector('button');
                const shouldPause = !baseline.paused || !ablation.paused;
                if (shouldPause) {
                    baseline.pause();
                    ablation.pause();
                    baseline.muted = false;
                    if (button) button.textContent = '▶️ Play';
                } else {
                    baseline.muted = true;
                    ablation.muted = false;
                    const restoreBaselineAudio = () => {
                        baseline.muted = false;
                    };
                    ablation.addEventListener('pause', restoreBaselineAudio, {once: true});
                    ablation.addEventListener('ended', restoreBaselineAudio, {once: true});
                    baseline.play();
                    ablation.play();
                    if (button) button.textContent = '⏸️ Pause';
                }
                return [];
            }
            """,
        )
        restart_both_button.click(
            fn=None,
            queue=False,
            js="""
            () => {
                const baseline = document.querySelector('#baseline-output-video video');
                const ablation = document.querySelector('#ablation-output-video video');
                if (!baseline || !ablation) {
                    alert('Render both videos before controlling playback.');
                    return [];
                }
                baseline.currentTime = 0;
                ablation.currentTime = 0;
                return [];
            }
            """,
        )
        backward_one_second_button.click(
            fn=None,
            queue=False,
            js="""
            () => {
                const baseline = document.querySelector('#baseline-output-video video');
                const ablation = document.querySelector('#ablation-output-video video');
                if (!baseline || !ablation) {
                    alert('Render both videos before controlling playback.');
                    return [];
                }
                const targetTime = Math.max(0, baseline.currentTime - 1);
                baseline.currentTime = targetTime;
                ablation.currentTime = targetTime;
                return [];
            }
            """,
        )
        forward_one_second_button.click(
            fn=None,
            queue=False,
            js="""
            () => {
                const baseline = document.querySelector('#baseline-output-video video');
                const ablation = document.querySelector('#ablation-output-video video');
                if (!baseline || !ablation) {
                    alert('Render both videos before controlling playback.');
                    return [];
                }
                const durations = [baseline.duration, ablation.duration].filter(Number.isFinite);
                const lastTime = durations.length ? Math.min(...durations) : Infinity;
                const targetTime = Math.min(lastTime, baseline.currentTime + 1);
                baseline.currentTime = targetTime;
                ablation.currentTime = targetTime;
                return [];
            }
            """,
        )
        render_both_button.click(
            render_both_ui,
            inputs=input_video,
            outputs=[baseline_output_video, ablation_output_video],
        )
        render_baseline_button.click(
            render_baseline_ui,
            inputs=input_video,
            outputs=baseline_output_video,
        )
        render_ablation_button.click(
            render_ablation_ui,
            inputs=input_video,
            outputs=ablation_output_video,
        )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(args.videos_dir.resolve()), str(output_dir)],
    )


def main() -> None:
    args = PARSED_ARGS or parse_args()
    if args.gradio:
        launch_gradio(args)
        return
    if args.video is None:
        raise SystemExit("--video is required unless --gradio is used")
    model, class_indices, class_names = prepare_render(args)
    output_path = (
        args.out
        or args.video.with_name(f"{args.video.stem}__{args.ckpt.stem}__sed_overlay.mp4")
    )
    render_video(args, model, class_indices, class_names, args.video, output_path)


if __name__ == "__main__":
    main()
