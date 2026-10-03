#!/usr/bin/env python3
"""Run MediaPipe export and Stage B import for a CSV of baseball pose jobs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-csv",
        type=Path,
        required=True,
        help="Pose-job CSV relative to project root or absolute path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/pose_landmarker_full.task"),
        help="MediaPipe pose landmarker model path.",
    )
    parser.add_argument(
        "--external-pose-root",
        type=Path,
        default=Path("data/baseball/external_pose"),
        help="Directory for raw MediaPipe exports.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of jobs to process.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose standardized-pose output already exists.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    jobs_path = resolve_path(args.jobs_csv)
    model_path = resolve_path(args.model_path)
    external_pose_root = resolve_path(args.external_pose_root)
    external_pose_root.mkdir(parents=True, exist_ok=True)

    jobs = read_jobs(jobs_path)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    python_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    export_script = PROJECT_ROOT / "src" / "preprocessing" / "export_mediapipe_pose.py"
    template_script = PROJECT_ROOT / "src" / "preprocessing" / "create_baseball_pose_example.py"
    populate_script = PROJECT_ROOT / "src" / "preprocessing" / "populate_baseball_pose_payload.py"

    completed_jobs = 0
    failed_jobs: list[tuple[str, str]] = []
    for job in jobs:
        clip_id = job["clip_id"]
        standardized_pose_path = resolve_path(Path(job["pose_output_path"]))
        if args.skip_existing and standardized_pose_path.exists():
            print(f"skip existing {clip_id}")
            continue

        external_pose_path = external_pose_root / f"{clip_id}_mediapipe_pose.npz"
        try:
            run_command(
                [
                    python_bin,
                    str(template_script),
                    "--clip-id",
                    clip_id,
                    "--pose-jobs-csv",
                    str(jobs_path.relative_to(PROJECT_ROOT)),
                ]
            )
            run_command([python_bin, str(export_script), "--input-video", job["local_media_path"], "--output-path", str(external_pose_path), "--model-path", str(model_path)])
            run_command(
                [
                    python_bin,
                    str(populate_script),
                    "--clip-id",
                    clip_id,
                    "--input-path",
                    str(external_pose_path),
                    "--pose-source",
                    "mediapipe_pose",
                    "--pose-jobs-csv",
                    str(jobs_path.relative_to(PROJECT_ROOT)),
                ]
            )
        except RuntimeError as exc:
            print(f"failed {clip_id}: {exc}", file=sys.stderr)
            failed_jobs.append((clip_id, str(exc)))
            continue
        completed_jobs += 1
        print(f"completed {clip_id}")

    print(f"batch completed: {completed_jobs}")
    if failed_jobs:
        print(f"batch failed: {len(failed_jobs)}")
        for clip_id, message in failed_jobs:
            print(f"  {clip_id}: {message}")


if __name__ == "__main__":
    main()
