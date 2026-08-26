"""Generate and train a joint hard-class street SED task."""
import argparse
import shlex
import subprocess
from pathlib import Path


HARD_CLASSES = [
    "Gunshot, gunfire",
    "Shatter",
    "Breaking",
    "Explosion",
]
AUXILIARY_IMPACTS = ["Machine gun", "Firecracker"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--class_file", type=Path, default=Path("street_bark.txt"))
    parser.add_argument("--source_path", type=Path, default=Path("/data/AudioSet-Strong-Balanced"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    dataset = args.output_root / "dataset"
    checkpoint_dir = args.output_root / "checkpoint"
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
        "--focus_classes", *HARD_CLASSES,
        "--confuser_classes", *AUXILIARY_IMPACTS,
        "--co_label_focus_events",
        "--focus_min_events", "8",
        "--focus_max_events", "12",
        "--confuser_min_events", "4",
        "--confuser_max_events", "7",
        "--context_min_events", "0",
        "--context_max_events", "1",
        "--overlap", "0.35",
        "--balanced_min_gap", "0.15",
        "--gunshot_min_gap", "0.12",
        "--gunshot_max_gap", "0.9",
        "--gunshot_unique_sources",
        "--gunshot_burst_probability", "0.35",
        "--event_gain_db_min", "-16",
        "--event_gain_db_max", "3",
        "--event_speed_min", "0.85",
        "--event_speed_max", "1.15",
        "--background_bed_db_min", "-22",
        "--background_bed_db_max", "-10",
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
        "--max_lr", "2e-5",
        "--warmup_steps", "200",
        "--positive_crop_p", "0.75",
        "--wavmix_p", "0",
        "--mixup_p", "0",
        "--filter_augment_p", "0.3",
        "--freq_warp_p", "0.3",
        "--mixstyle_p", "0.1",
        "--frame_shift_range", "0.25",
        "--check_val_every_n_epoch", "5",
        "--experiment_name", "street_high_recall_impacts_joint",
        "--checkpoint_dir", str(checkpoint_dir),
    ]

    if args.run:
        if (dataset / "train.json").is_file():
            print(f"Reusing existing dataset: {dataset}")
        else:
            print(shlex.join(generate))
            subprocess.run(generate, check=True)
        checkpoint = checkpoint_dir / "last.ckpt"
        if checkpoint.is_file():
            train += ["--resume_from_checkpoint", str(checkpoint)]
        print(shlex.join(train))
        subprocess.run(train, check=True)
    else:
        print(shlex.join(generate))
        print(shlex.join(train))


if __name__ == "__main__":
    main()
