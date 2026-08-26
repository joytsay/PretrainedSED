"""Refine impact scores without treating related impact sounds as negatives."""
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
AMBIGUOUS_IMPACTS = TARGET_CLASSES + ["Machine gun", "Firecracker"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--input_checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset_root / "dataset"
    if not (dataset / "train.json").is_file():
        raise FileNotFoundError(f"Missing dataset: {dataset}")
    if not args.input_checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.input_checkpoint}")

    checkpoint_dir = args.output_root / "checkpoint"
    current_checkpoint = checkpoint_dir / "last.ckpt"
    train = [
        "python", "ex_audioset_strong.py",
        "--task_path", str(dataset),
        "--model_name", "ATST-F",
        "--pretrained", "strong",
        "--n_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_lr", "4e-6",
        "--warmup_steps", "40",
        "--positive_crop_p", "0.9",
        "--confidence_classes", *TARGET_CLASSES,
        "--ambiguous_impact_classes", *AMBIGUOUS_IMPACTS,
        "--confidence_pos_weight", "3.0",
        "--positive_confidence_target", "0.7",
        "--positive_confidence_loss_weight", "0.5",
        "--confidence_margin_top_fraction", "0.25",
        "--wavmix_p", "0",
        "--mixup_p", "0",
        "--filter_augment_p", "0.15",
        "--freq_warp_p", "0.15",
        "--mixstyle_p", "0",
        "--frame_shift_range", "0.125",
        "--check_val_every_n_epoch", "2",
        "--experiment_name", "street_ambiguous_impact_refine",
        "--checkpoint_dir", str(checkpoint_dir),
    ]
    if current_checkpoint.is_file():
        train += ["--resume_from_checkpoint", str(current_checkpoint)]
    else:
        train += ["--init_from_checkpoint", str(args.input_checkpoint)]

    print(shlex.join(train))
    if args.run:
        subprocess.run(train, check=True)


if __name__ == "__main__":
    main()
