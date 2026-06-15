import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import tempfile
import sys

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from data_util.dcase2016task2 import label_vocab_nlabels
from data_util.audioset_classes import as_strong_train_classes
from ex_dcase2016task2 import PLModule
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper


TARGET_SAMPLE_RATE = 16000
SEGMENT_DURATION = 10.0
SEGMENT_SAMPLES = int(SEGMENT_DURATION * TARGET_SAMPLE_RATE)
MODEL_FRAME_RATE = 100 / 4
MAX_PLOT_CLASSES = 12

ATST_DCASE_TO_AUDIOSSET = {
    "clearthroat": ["Throat clearing"],
    "cough": ["Cough"],
    "doorslam": ["Door"],
    "drawer": ["Drawer open or close"],
    "keyboard": ["Computer keyboard"],
    "keys": ["Keys jangling"],
    "knock": ["Knock"],
    "laughter": ["Laughter"],
    "pageturn": ["Paper rustling"],
    "phone": ["Telephone", "Telephone bell ringing", "Telephone dialing, DTMF"],
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking"],
}


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


def load_ground_truth_map(task_path):
    task_path = Path(task_path)
    test_json = task_path / "test.json"
    if not test_json.exists():
        return {}
    with test_json.open() as handle:
        data = json.load(handle)
    return {f"{key}.wav": value for key, value in data.items()}


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
            animation.save(tmp_video_path, writer=writer, dpi=100)
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


def remap_audioset_scores_to_dcase(scores, audioset_labels, dcase_labels):
    audioset_to_idx = {label: idx for idx, label in enumerate(audioset_labels)}
    remapped = []
    missing = []
    for label in dcase_labels:
        source_labels = ATST_DCASE_TO_AUDIOSSET.get(label, [])
        source_indices = [audioset_to_idx[src] for src in source_labels if src in audioset_to_idx]
        if not source_indices:
            missing.append(label)
            continue
        remapped.append(scores[:, source_indices].max(axis=1))
    if missing:
        raise KeyError(f"Could not map DCASE labels to AudioSet classes: {missing}")
    return np.stack(remapped, axis=1)


def forward_strong_logits(model, waveform):
    if isinstance(model, PLModule):
        return model.forward(waveform)
    mel = model.mel_forward(waveform)
    strong, _ = model(mel)
    return strong


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned DCASE2016 Task 2 checkpoint.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--task_path", type=str, required=True)
    parser.add_argument("--audio_root", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("eval_dcase2016task2_ckpt"))
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--no_video", action="store_false", dest="make_video")
    parser.add_argument("--video_fps", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="ATST-F",
                        choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"])
    parser.add_argument("--pretrained", type=str, default="strong", choices=["scratch", "ssl", "weak", "strong"])
    parser.add_argument("--seq_model_type", type=str, default="none", choices=["none", "rnn"])
    parser.add_argument("--n_classes", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--experiment_name", type=str, default="DCASE2016Task2_Eval")
    parser.set_defaults(make_video=True)
    args = parser.parse_args()

    cfg = build_config(args)
    use_audioset_projection = args.ckpt_path.name == "ATST-F_strong_1.pt"
    if use_audioset_projection:
        model = PredictionsWrapper(
            ATSTWrapper(),
            checkpoint="ATST-F_strong_1",
            n_classes_strong=len(as_strong_train_classes),
            n_classes_weak=len(as_strong_train_classes),
        )
        print("Using AudioSet projection from ATST-F_strong_1.pt")
    else:
        model = PLModule(cfg)
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        if state_dict and not next(iter(state_dict)).startswith("model.model."):
            state_dict = {f"model.{key}": value for key, value in state_dict.items()}
        model_state = model.state_dict()
        filtered_state_dict = {}
        dropped = []
        for key, value in state_dict.items():
            if key not in model_state:
                dropped.append(key)
                continue
            if model_state[key].shape != value.shape:
                dropped.append(f"{key} (ckpt {tuple(value.shape)} != model {tuple(model_state[key].shape)})")
                continue
            filtered_state_dict[key] = value

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if dropped:
            print("Dropped incompatible keys:")
            for key in dropped:
                print("  ", key)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    model.eval()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_root = args.audio_root if args.audio_root is not None else Path(cfg.task_path) / str(TARGET_SAMPLE_RATE) / "test"
    audio_files = iter_wavs(audio_root)
    if not audio_files:
        raise FileNotFoundError(f"No wav files found under {audio_root}")

    label_vocab, _ = label_vocab_nlabels(Path(cfg.task_path))
    label_names = list(label_vocab["label"].astype(str))
    encoder = ManyHotEncoder(label_names, audio_len=SEGMENT_DURATION, fs=TARGET_SAMPLE_RATE)
    gt_map = load_ground_truth_map(cfg.task_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    all_predictions = defaultdict(pd.DataFrame)
    for audio_path in tqdm(audio_files, desc="Evaluating wavs", unit="file"):
        tqdm.write(f"Processing {audio_path.name}")
        waveform, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        waveform = torch.from_numpy(waveform[None, :]).to(device)
        waveform_len = waveform.shape[1]
        num_chunks = waveform_len // SEGMENT_SAMPLES + int(waveform_len % SEGMENT_SAMPLES != 0)
        chunk_preds = []
        chunk_bar = tqdm(total=num_chunks, desc=f"Chunks {audio_path.stem}", unit="chunk", leave=False, dynamic_ncols=True)
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * SEGMENT_SAMPLES
            end_idx = min((chunk_idx + 1) * SEGMENT_SAMPLES, waveform_len)
            waveform_chunk = waveform[:, start_idx:end_idx]
            if waveform_chunk.shape[1] < SEGMENT_SAMPLES:
                waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, SEGMENT_SAMPLES - waveform_chunk.shape[1]))
            with torch.no_grad():
                chunk_preds.append(torch.sigmoid(forward_strong_logits(model, waveform_chunk)).detach().cpu())
            chunk_bar.update(1)
            chunk_bar.set_postfix_str(f"{chunk_idx + 1}/{num_chunks}")
        chunk_bar.close()

        y_strong = torch.cat(chunk_preds, dim=2)
        y_strong_for_dcase = y_strong
        if use_audioset_projection:
            y_strong_for_dcase = torch.from_numpy(
                remap_audioset_scores_to_dcase(
                    y_strong.squeeze(0).transpose(0, 1).cpu().numpy(),
                    as_strong_train_classes,
                    label_names,
                )
            ).unsqueeze(0).transpose(1, 2)

        _, _, decoded_predictions = batched_decode_preds(
            y_strong_for_dcase.float(),
            [str(audio_path)],
            encoder,
            median_filter=9,
            thresholds=(0.1, 0.2, 0.5),
        )

        audio_id = audio_path.stem
        for threshold, df in decoded_predictions.items():
            if not df.empty:
                df = df.copy()
                df["threshold"] = threshold
            all_predictions[threshold] = pd.concat([all_predictions[threshold], df], ignore_index=True)

        frame_scores = y_strong_for_dcase.squeeze(0).transpose(0, 1).cpu().numpy()
        frame_times = np.arange(frame_scores.shape[0]) / MODEL_FRAME_RATE
        gt_mask = events_to_frame_mask(gt_map.get(audio_path.name, []), frame_times, set(label_names))
        plot_path = out_dir / f"{audio_id}.png"
        write_frame_plot(plot_path, audio_path.name, frame_scores, gt_mask, label_names)
        if args.make_video:
            tqdm.write(f"Rendering video {audio_path.name}", file=sys.stderr)
            video_path = out_dir / f"{audio_id}.mp4"
            write_audio_video(video_path, audio_path, frame_scores, gt_mask, label_names, fps=args.video_fps)

        print(f"{audio_path.name}:")
        for threshold, df in decoded_predictions.items():
            print(f"  threshold={threshold}: {len(df)} events")

    for threshold, df in all_predictions.items():
        df.to_csv(out_dir / f"predictions_{threshold}.csv", index=False)

    print(f"Wrote plots and predictions to {out_dir}")


if __name__ == "__main__":
    main()
