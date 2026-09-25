#!/usr/bin/env python3
"""Refresh the Stage B baseball pipeline after metadata updates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.baseball.example.yaml"),
        help="Baseball preprocessing config to use for feature extraction, fitting, and evaluation.",
    )
    parser.add_argument(
        "--skip-overrides",
        action="store_true",
        help="Do not apply handedness overrides before rebuilding downstream artifacts.",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Do not rebuild the baseball pitching train/test split CSVs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run held-out baseball evaluation after fitting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print("$", " ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    python_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    config_path = str(args.config)

    commands: list[list[str]] = []
    if not args.skip_overrides:
        commands.append([python_bin, "src/preprocessing/apply_baseball_handedness_overrides.py"])
    if not args.skip_splits:
        commands.append([python_bin, "src/preprocessing/build_baseball_pitching_splits.py", "--pool-size", "24", "--test-size", "6"])

    commands.extend(
        [
            [python_bin, "src/preprocessing/process_baseball_standardized_pose.py", "--split", "train", "--task-type", "pitching"],
            [python_bin, "src/preprocessing/process_baseball_standardized_pose.py", "--split", "test", "--task-type", "pitching"],
            [python_bin, "src/preprocessing/extract_baseball_features.py", "--config", config_path, "--split", "train"],
            [python_bin, "src/preprocessing/extract_baseball_features.py", "--config", config_path, "--split", "test"],
            [python_bin, "src/preprocessing/fit_baseball_sindy.py", "--config", config_path, "--split", "train", "--task-types", "pitching"],
        ]
    )
    if not args.skip_eval:
        commands.append(
            [
                python_bin,
                "src/preprocessing/evaluate_baseball_sindy.py",
                "--config",
                config_path,
                "--train-split",
                "train",
                "--eval-split",
                "test",
                "--task-types",
                "pitching",
            ]
        )

    for command in commands:
        run_command(command, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
