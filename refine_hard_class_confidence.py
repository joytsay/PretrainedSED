"""Refine the better v1 model for higher positive-frame confidence."""
import argparse
import shlex
import subprocess
from pathlib import Path


TARGET_CLASSES = [
    "Gunshot, gunfire",
    "Shatter",
    "Breaking",
    "Explosion",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    dataset = args.input_root / "dataset"
    source_checkpoint = args.input_root / "checkpoint" / "last.ckpt"
    output_checkpoint_dir = args.output_root / "checkpoint"
    current_checkpoint = output_checkpoint_dir / "last.ckpt"
    if not (dataset / "train.json").is_file():
        raise FileNotFoundError(f"Missing v1 dataset: {dataset}")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Missing v1 checkpoint: {source_checkpoint}")

    train = [
        "python", "ex_audioset_strong.py",
        "--task_path", str(dataset),
        "--model_name", "ATST-F",
        "--pretrained", "strong",
        "--n_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_lr", "7e-6",
        "--warmup_steps", "50",
        "--positive_crop_p", "0.9",
        "--confidence_classes", *TARGET_CLASSES,
        "--confidence_pos_weight", "4.0",
        "--positive_confidence_target", "0.7",
        "--positive_confidence_loss_weight", "0.5",
        "--wavmix_p", "0",
        "--mixup_p", "0",
        "--filter_augment_p", "0.2",
        "--freq_warp_p", "0.2",
        "--mixstyle_p", "0",
        "--frame_shift_range", "0.125",
        "--check_val_every_n_epoch", "2",
        "--experiment_name", "street_hard_impacts_confidence_refine",
        "--checkpoint_dir", str(output_checkpoint_dir),
    ]
    if current_checkpoint.is_file():
        train += ["--resume_from_checkpoint", str(current_checkpoint)]
    else:
        train += ["--init_from_checkpoint", str(source_checkpoint)]

    print(shlex.join(train))
    if args.run:
        subprocess.run(train, check=True)


if __name__ == "__main__":
    main()
