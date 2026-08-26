"""Prepare the 13-pass street SED curriculum.

Stage 1 uses the normal balanced recipe.  Stages 2..13 resume the previous
checkpoint and add one extra occurrence of one class per generated clip.
Commands are printed (and optionally executed) so long-running jobs remain
under the user's control.
"""
import argparse
import subprocess
from pathlib import Path


def checkpoint_is_complete(path: Path, max_epochs: int) -> bool:
    """Return whether Lightning's last checkpoint reached max_epochs."""
    if not path.is_file():
        return False
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return int(checkpoint.get("epoch", -1)) + 1 >= max_epochs
    except Exception as exc:
        print(f"Could not inspect {path}: {exc}; reruming from it.")
        return False


def read_classes(path: Path):
    return [line.split("\t", 1)[1].strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class_file", type=Path, default=Path("street_bark.txt"))
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--source_path", type=Path, default=Path("/data/AudioSet-Strong-Balanced"))
    p.add_argument("--train_epochs", type=int, default=5)
    p.add_argument("--run", action="store_true", help="Run each generation and training command.")
    args = p.parse_args()

    classes = read_classes(args.class_file)
    if len(classes) != 13:
        raise ValueError(f"Expected 13 classes, found {len(classes)}")
    # Every stage is single-class audio, while the dataset vocabulary remains
    # all 13 classes so checkpoints remain shape-compatible.
    stages = [(i + 1, c) for i, c in enumerate(classes)]
    if len(stages) != 13:
        raise AssertionError(stages)

    previous_ckpt = None
    for stage, target in stages:
        dataset = args.output_root / f"stage_{stage:02d}_dataset"
        ckpt_dir = args.output_root / f"stage_{stage:02d}_checkpoint"
        aug = ["python", "aug_street.py", "--class_file", str(args.class_file),
               "--source_path", str(args.source_path), "--output_path", str(dataset),
               "--overwrite", "--clip_len", "120", "--events_per_class", "3",
               "--overlap", "0.5", "--balanced_min_gap", "0.5",
               "--gunshot_min_gap", "0.15", "--gunshot_max_gap", "0.8",
               "--background_bed_db", "-30", "--background_ratio", "0.1",
               "--max_event_s", "3", "--active_class", target]
        train = ["python", "ex_audioset_strong.py", "--task_path", str(dataset),
                 "--model_name", "ATST-F", "--pretrained", "strong", "--n_epochs",
                 # Lightning treats max_epochs as an absolute epoch number
                 # when resuming, so each stage adds another train_epochs.
                 str(stage * args.train_epochs), "--batch_size", "32", "--wavmix_p", "0",
                 "--mixup_p", "0", "--experiment_name", f"street_stage_{stage:02d}",
                 "--checkpoint_dir", str(ckpt_dir)]
        if args.run:
            if (dataset / "train.json").is_file():
                print(f"Reusing existing dataset: {dataset}")
            else:
                subprocess.run(aug, check=True)
            current_ckpt = ckpt_dir / "last.ckpt"
            max_epochs = stage * args.train_epochs
            if checkpoint_is_complete(current_ckpt, max_epochs):
                print(f"Stage {stage} already complete: {current_ckpt}")
            else:
                if current_ckpt.is_file():
                    train += ["--resume_from_checkpoint", str(current_ckpt)]
                elif previous_ckpt:
                    train += ["--resume_from_checkpoint", str(previous_ckpt)]
                print(" ".join(train))
                subprocess.run(train, check=True)
        else:
            if previous_ckpt:
                train += ["--resume_from_checkpoint", str(previous_ckpt)]
            print(" ".join(aug))
            print(" ".join(train))
        previous_ckpt = ckpt_dir / "last.ckpt"


if __name__ == "__main__":
    main()
