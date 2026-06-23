import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from data_util.audioset_street import label_vocab_nlabels as street_label_vocab_nlabels
from data_util.dcase2016task2 import label_vocab_nlabels as dcase_label_vocab_nlabels
from ex_audioset_street import PLModule
from ex_dcase2016task2 import PLModule as DcasePLModule
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder


TARGET_SAMPLE_RATE = 16000
SEGMENT_DURATION = 10.0
SEGMENT_SAMPLES = int(SEGMENT_DURATION * TARGET_SAMPLE_RATE)
MODEL_FRAME_RATE = 100 / 4


def class_colors(n):
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab20")
    return [palette(i % palette.N) for i in range(n)]


def build_config(args):
    return argparse.Namespace(
        task_path=args.task_path,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_devices=args.num_devices,
        precision=args.precision,
        check_val_every_n_epoch=10,
        model_name=args.model_name,
        pretrained=args.pretrained,
        seq_model_type=None if args.seq_model_type == "none" else args.seq_model_type,
        n_classes=args.n_classes,
        n_epochs=300,
        wavmix_p=0.5,
        freq_warp_p=0.0,
        filter_augment_p=0.0,
        frame_shift_range=0.0,
        mixup_p=0.5,
        mixstyle_p=0.0,
        max_time_mask_size=0.0,
        no_adamw=False,
        weight_decay=0.001,
        transformer_frozen=False,
        schedule_mode="cos",
        max_lr=1.06e-4,
        transformer_lr=None,
        lr_decay=1.0,
        lr_end=1e-7,
        warmup_steps=100,
    )


def iter_wavs(audio_root):
    audio_root = Path(audio_root)
    return sorted(audio_root.rglob("*.wav"))


def load_ground_truth_map(task_path, split_name="valid"):
    task_path = Path(task_path)
    split_json = task_path / f"{split_name}.json"
    if not split_json.exists():
        return {}
    with split_json.open() as handle:
        data = json.load(handle)
    return {f"{key}.wav": value for key, value in data.items()}


def get_task_spec(task_name):
    task_name = task_name.lower()
    if task_name == "street":
        return {
            "label_vocab_nlabels": street_label_vocab_nlabels,
            "plmodule": PLModule,
            "default_task_path": Path("/data/hear_datasets/tasks/audio_set_strong_street"),
            "default_audio_root": Path("/data/hear_datasets/tasks/audio_set_strong_street/16000/valid"),
            "default_n_classes": 11,
            "default_experiment_name": "AudioSetStreet_Val",
        }
    if task_name == "dcase":
        return {
            "label_vocab_nlabels": dcase_label_vocab_nlabels,
            "plmodule": DcasePLModule,
            "default_task_path": Path("/data/hear_datasets/tasks/dcase2016_task2-hear2021-full"),
            "default_audio_root": None,
            "default_n_classes": 11,
            "default_experiment_name": "DCASE2016Task2_Val",
        }
    raise ValueError(f"Unknown task '{task_name}'. Expected 'street' or 'dcase'.")


def infer_n_classes_from_checkpoint(ckpt_path: Path):
    """
    Infer the output class count from the saved checkpoint before building
    the Lightning module config.
    """
    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        return None

    state_dict = checkpoint.get("state_dict", {})
    for key in ("strong_loss.pos_weight", "model.strong_head.bias", "model.weak_head.bias"):
        tensor = state_dict.get(key)
        if tensor is None:
            continue
        if key == "strong_loss.pos_weight" and tensor.ndim >= 2:
            return int(tensor.shape[1])
        if tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def load_model_from_checkpoint(model_cls, ckpt_path: Path, cfg):
    """
    Load a Lightning module while ignoring stale loss-buffer shapes in older
    checkpoints.
    """
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model = model_cls(config=cfg)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if "strong_loss.pos_weight" in state_dict:
        current = model.state_dict().get("strong_loss.pos_weight")
        saved = state_dict["strong_loss.pos_weight"]
        if current is not None and tuple(current.shape) != tuple(saved.shape):
            state_dict = dict(state_dict)
            state_dict.pop("strong_loss.pos_weight", None)
    model.load_state_dict(state_dict, strict=False)
    return model


def remap_dcase_scores_to_audioset(scores, dcase_labels):
    audioset_to_idx = {label: idx for idx, label in enumerate([
        "Throat clearing",
        "Cough",
        "Door",
        "Drawer open or close",
        "Computer keyboard",
        "Keys jangling",
        "Knock",
        "Laughter",
        "Paper rustling",
        "Telephone",
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
    ])}
    dcase_to_source = {
        "clearthroat": ["Throat clearing"],
        "cough": ["Cough"],
        "doorslam": ["Door"],
        "drawer": ["Drawer open or close"],
        "keyboard": ["Computer keyboard"],
        "keys": ["Keys jangling"],
        "knock": ["Knock"],
        "laughter": ["Laughter"],
        "pageturn": ["Paper rustling"],
        "phone": ["Telephone"],
        "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking"],
    }
    remapped = []
    for label in dcase_labels:
        source_indices = [audioset_to_idx[src] for src in dcase_to_source.get(label, []) if src in audioset_to_idx]
        if not source_indices:
            raise KeyError(f"Could not map DCASE label '{label}'")
        remapped.append(scores[:, source_indices].max(axis=1))
    return np.stack(remapped, axis=1)


def events_to_frame_mask(events, frame_times, label_names):
    if not events:
        return np.zeros(len(frame_times), dtype=bool)
    mask = np.zeros(len(frame_times), dtype=bool)
    for event in events:
        if event.get("label") not in label_names:
            continue
        start = float(event["start"])
        end = float(event["end"])
        mask |= (frame_times >= start) & (frame_times < end)
    return mask


def write_frame_plot(path, filename, scores, gt_mask, class_names, threshold=0.5):
    import matplotlib.pyplot as plt

    times = np.arange(scores.shape[0]) / MODEL_FRAME_RATE
    plt.figure(figsize=(10.24, 3.2), dpi=100)
    colors = class_colors(len(class_names))
    for class_idx, class_name in enumerate(class_names):
        plt.plot(times, scores[:, class_idx], linewidth=1.2, color=colors[class_idx], label=class_name)
    plt.fill_between(times, 0, 1, where=gt_mask.tolist(), alpha=0.18, step="mid", color="lightblue")
    plt.axhline(threshold, color="black", linewidth=1, linestyle="--", label="threshold")
    plt.ylim(0, 1)
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.title(filename)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_audio_video(path, audio_path, scores, gt_mask, class_names, fps=10):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    times = np.arange(scores.shape[0]) / MODEL_FRAME_RATE
    colors = class_colors(len(class_names))
    fig, ax = plt.subplots(figsize=(10.24, 3.2), dpi=100)
    for class_idx, class_name in enumerate(class_names):
        ax.plot(times, scores[:, class_idx], linewidth=1.1, color=colors[class_idx], label=class_name)
    ax.fill_between(times, 0, 1, where=gt_mask.tolist(), alpha=0.18, step="mid", color="lightblue")
    playhead = ax.axvline(0.0, color="red", linewidth=1.5)
    ax.set_xlim(0, max(float(times[-1]) if len(times) else 0.0, 0.1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_title(Path(audio_path).name)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()

    def update(frame_idx):
        current_time = frame_idx / fps
        playhead.set_xdata([current_time, current_time])
        return [playhead]

    total_frames = max(1, int(np.ceil(times[-1] * fps)) + 1 if len(times) else 1)
    animation = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=True)

    tmp_video_path = Path(path).with_suffix(".tmp.mp4")
    last_error = None
    for codec in ("mpeg4", "libxvid", "libx264"):
        try:
            writer = FFMpegWriter(
                fps=fps,
                codec=codec,
                bitrate=1800,
                extra_args=["-vcodec", codec, "-pix_fmt", "yuv420p"],
            )
            progress = tqdm(total=total_frames, desc=f"Video {Path(audio_path).stem}", unit="frame", leave=False)

            def progress_update(frame_idx):
                progress.update(1)
                return update(frame_idx)

            animation = FuncAnimation(fig, progress_update, frames=total_frames, interval=1000 / fps, blit=True)
            animation.save(tmp_video_path, writer=writer, dpi=100)
            progress.close()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        plt.close(fig)
        tmp_video_path.unlink(missing_ok=True)
        raise last_error
    plt.close(fig)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        tmp_video_path,
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    tmp_video_path.unlink(missing_ok=True)


def forward_strong_logits(model, waveform):
    return model.forward(waveform)


def get_ground_truth_path(task_path, split_name):
    task_path = Path(task_path)
    split_json = task_path / f"{split_name}.json"
    if not split_json.exists():
        return {}
    with split_json.open() as handle:
        data = json.load(handle)
    return {f"{key}.wav": value for key, value in data.items()}


def decode_and_plot(
    audio_files,
    model,
    encoder,
    gt_map,
    out_dir,
    label_names,
    thresholds,
    median_filter,
    split_name,
    make_video=False,
    video_fps=10,
):
    device = next(model.parameters()).device
    all_predictions = defaultdict(pd.DataFrame)
    for audio_path in tqdm(audio_files, desc=f"Evaluating {split_name} wavs", unit="file"):
        tqdm.write(f"Processing {audio_path.name}")
        waveform, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        waveform = torch.from_numpy(waveform[None, :]).to(device)
        waveform_len = waveform.shape[1]
        num_chunks = waveform_len // SEGMENT_SAMPLES + int(waveform_len % SEGMENT_SAMPLES != 0)
        chunk_preds = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * SEGMENT_SAMPLES
            end_idx = min((chunk_idx + 1) * SEGMENT_SAMPLES, waveform_len)
            waveform_chunk = waveform[:, start_idx:end_idx]
            if waveform_chunk.shape[1] < SEGMENT_SAMPLES:
                waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, SEGMENT_SAMPLES - waveform_chunk.shape[1]))
            with torch.no_grad():
                chunk_preds.append(torch.sigmoid(forward_strong_logits(model, waveform_chunk)).detach().cpu())

        y_strong = torch.cat(chunk_preds, dim=2)
        print(
            f"  score stats: min={float(y_strong.min()):.4f} "
            f"mean={float(y_strong.mean()):.4f} max={float(y_strong.max()):.4f}"
        )
        _, _, decoded_predictions = batched_decode_preds(
            y_strong.float(),
            [str(audio_path)],
            encoder,
            median_filter=median_filter,
            thresholds=tuple(thresholds),
        )

        audio_id = audio_path.stem
        for threshold, df in decoded_predictions.items():
            if not df.empty:
                df = df.copy()
                df["threshold"] = threshold
            all_predictions[threshold] = pd.concat([all_predictions[threshold], df], ignore_index=True)

        frame_scores = y_strong.squeeze(0).transpose(0, 1).cpu().numpy()
        frame_times = np.arange(frame_scores.shape[0]) / MODEL_FRAME_RATE
        gt_mask = events_to_frame_mask(gt_map.get(audio_path.name, []), frame_times, set(label_names))
        write_frame_plot(out_dir / f"{audio_id}.png", audio_path.name, frame_scores, gt_mask, label_names)
        if make_video:
            tqdm.write(f"Rendering video {audio_path.name}", file=sys.stderr)
            write_audio_video(out_dir / f"{audio_id}.mp4", audio_path, frame_scores, gt_mask, label_names, fps=video_fps)

        print(f"{audio_path.name}:")
        for threshold, df in decoded_predictions.items():
            print(f"  threshold={threshold}: {len(df)} events")

    for threshold, df in all_predictions.items():
        df.to_csv(out_dir / f"predictions_{threshold}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Validate a fine-tuned street or DCASE checkpoint.")
    parser.add_argument("--task", type=str, default="street", choices=["street", "dcase"])
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--task_path", type=Path, default=None)
    parser.add_argument("--audio_root", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("eval_test"))
    parser.add_argument("--model_name", type=str, default="ATST-F", choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"])
    parser.add_argument("--pretrained", type=str, default="strong", choices=["scratch", "ssl", "weak", "strong"])
    parser.add_argument("--seq_model_type", type=str, default="none", choices=["none", "rnn"])
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.01, 0.05, 0.1, 0.2))
    parser.add_argument("--median_filter", type=int, default=3)
    parser.add_argument("--test_num", type=int, default=1, help="Limit evaluation to the first N wavs.")
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--no_video", action="store_false", dest="make_video")
    parser.add_argument("--video_fps", type=int, default=5)
    parser.set_defaults(make_video=True)
    args = parser.parse_args()

    task_spec = get_task_spec(args.task)
    if args.task_path is None:
        args.task_path = task_spec["default_task_path"]
    if args.audio_root is None:
        if task_spec["default_audio_root"] is None:
            raise ValueError("--audio_root is required for dcase")
        args.audio_root = task_spec["default_audio_root"]
    if args.n_classes is None:
        inferred_n_classes = infer_n_classes_from_checkpoint(args.ckpt_path)
        args.n_classes = inferred_n_classes if inferred_n_classes is not None else task_spec["default_n_classes"]
    if args.experiment_name is None:
        args.experiment_name = task_spec["default_experiment_name"]

    cfg = build_config(args)
    model_cls = task_spec["plmodule"]
    model = load_model_from_checkpoint(model_cls, args.ckpt_path, cfg)
    print(f"Loaded checkpoint via Lightning: {args.ckpt_path}")
    model.eval()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = iter_wavs(args.audio_root)
    if not audio_files:
        raise FileNotFoundError(f"No wav files found under {args.audio_root}")
    if args.test_num is not None:
        audio_files = audio_files[:args.test_num]

    label_vocab, _ = task_spec["label_vocab_nlabels"](Path(args.task_path))
    label_names = list(label_vocab["label"].astype(str))
    if len(label_names) != args.n_classes:
        raise ValueError(
            f"Class count mismatch: task vocabulary has {len(label_names)} labels, "
            f"but the model is configured for {args.n_classes} classes."
        )
    encoder = ManyHotEncoder(label_names, audio_len=SEGMENT_DURATION, fs=TARGET_SAMPLE_RATE)
    split_name = "valid"
    gt_map = get_ground_truth_path(args.task_path, split_name=split_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    decode_and_plot(
        audio_files=audio_files,
        model=model,
        encoder=encoder,
        gt_map=gt_map,
        out_dir=out_dir,
        label_names=label_names,
        thresholds=args.thresholds,
        median_filter=args.median_filter,
        split_name=split_name,
        make_video=args.make_video,
        video_fps=args.video_fps,
    )

    print(f"Wrote plots and predictions to {out_dir}")


if __name__ == "__main__":
    main()
