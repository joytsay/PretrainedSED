"""Consolidate a focused checkpoint with balanced refinement on all street classes."""
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


def read_classes(path: Path) -> list[str]:
    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Expected '<mid>\\t<label>', got: {line!r}")
        classes.append(parts[1].strip())
    return classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--class_file", type=Path, default=Path("street_bark.txt"))
    parser.add_argument("--source_path", type=Path, default=Path("/data/AudioSet-Strong-Balanced"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    classes = read_classes(args.class_file)
    if len(classes) != 13:
        raise ValueError(f"Expected 13 classes in {args.class_file}, found {len(classes)}")
    if not args.input_checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.input_checkpoint}")

    dataset = args.output_root / "dataset"
    checkpoint_dir = args.output_root / "checkpoint"
    current_checkpoint = checkpoint_dir / "last.ckpt"
    generate = [
        "python", "aug_street.py",
        "--class_file", str(args.class_file),
        "--source_path", str(args.source_path),
        "--output_path", str(dataset),
        "--overwrite",
        "--clip_len", "120",
        "--train_count", "132",
        "--valid_count", "14",
        "--test_count", "14",
        "--pool_source_splits",
        "--test_source_fraction", "0.2",
        "--events_per_class", "3",
        "--focus_classes", *classes,
        "--focus_min_events", "3",
        "--focus_max_events", "5",
        "--overlap", "0.3",
        "--balanced_min_gap", "0.2",
        "--gunshot_min_gap", "0.12",
        "--gunshot_max_gap", "0.9",
        "--gunshot_unique_sources",
        "--gunshot_burst_probability", "0.35",
        "--event_gain_db_min", "-14",
        "--event_gain_db_max", "3",
        "--event_speed_min", "0.88",
        "--event_speed_max", "1.12",
        "--background_bed_db_min", "-24",
        "--background_bed_db_max", "-12",
        "--background_ratio", "0.25",
        "--max_event_s", "3",
    ]
    train = [
        "python", "ex_audioset_strong.py",
        "--task_path", str(dataset),
        "--model_name", "ATST-F",
        "--pretrained", "strong",
        "--n_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_lr", "3e-6",
        "--warmup_steps", "75",
        "--positive_crop_p", "0.8",
        "--confidence_classes", *TARGET_CLASSES,
        "--ambiguous_impact_classes", *AMBIGUOUS_IMPACTS,
        "--confidence_pos_weight", "2.5",
        "--positive_confidence_target", "0.65",
        "--positive_confidence_loss_weight", "0.25",
        "--confidence_margin_top_fraction", "0.25",
        "--wavmix_p", "0",
        "--mixup_p", "0",
        "--filter_augment_p", "0.15",
        "--freq_warp_p", "0.15",
        "--mixstyle_p", "0",
        "--frame_shift_range", "0.125",
        "--check_val_every_n_epoch", "3",
        "--experiment_name", "street_all_13_consolidation",
        "--checkpoint_dir", str(checkpoint_dir),
    ]
    if current_checkpoint.is_file():
        train += ["--resume_from_checkpoint", str(current_checkpoint)]
    else:
        train += ["--init_from_checkpoint", str(args.input_checkpoint)]

    if args.run:
        if (dataset / "train.json").is_file():
            print(f"Reusing existing dataset: {dataset}")
        else:
            print(shlex.join(generate))
            subprocess.run(generate, check=True)
        print(shlex.join(train))
        subprocess.run(train, check=True)
    else:
        print(shlex.join(generate))
        print(shlex.join(train))


if __name__ == "__main__":
    main()
