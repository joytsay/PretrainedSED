#!/usr/bin/env python3
"""
Convert a HEAR-style DCASE 2016 Task 2 dataset from 48 kHz WAVs to 16 kHz WAVs.

Usage:
    python scripts/convert_48k_to_16k.py \
        --task_path /path/to/dcase2016_task2-hear2021-full

This reads:
    <task_path>/48000/**.wav

and writes:
    <task_path>/16000/**.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def convert_file(src: Path, dst: Path, target_sr: int) -> None:
    audio, sr = sf.read(str(src), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, target_sr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_path", type=Path, required=True)
    parser.add_argument("--src_sr", type=int, default=48000)
    parser.add_argument("--dst_sr", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = args.task_path / str(args.src_sr)
    dst_root = args.task_path / str(args.dst_sr)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    for src_file in sorted(src_root.rglob("*")):
        if not src_file.is_file() or src_file.suffix.lower() != ".wav":
            continue

        rel_path = src_file.relative_to(src_root)
        dst_file = dst_root / rel_path
        if dst_file.exists() and not args.overwrite:
            continue
        convert_file(src_file, dst_file, args.dst_sr)

    print(f"Done. Wrote 16 kHz WAVs under {dst_root}")


if __name__ == "__main__":
    main()
