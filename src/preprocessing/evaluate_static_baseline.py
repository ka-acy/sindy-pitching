#!/usr/bin/env python3
"""Score a frozen (zero-derivative) model on the fair core-position metric.

This is the missing control for `core_position_recon_r2`. A frozen model holds the
first observed state constant for the whole rollout, so it encodes no dynamics at
all. Any fitted SINDy model must beat this baseline for its reconstruction score to
mean anything: `core_position_recon_r2` is measured against the per-window mean, and
over short rollouts the core positions move very little, so a do-nothing predictor
already scores well. Reported alongside the fitted models in the comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.evaluate_baseball_sindy import core_position_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.baseball.unaligned.handedness.yaml"),
        help="YAML config path (supplies feature_subdir, reconstruction_frames, recon_horizons).",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        help="Baseball split to evaluate on.",
    )
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=("pitching",),
        help="Task types to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/evaluation/baseball/static_baseline/summary.csv",
        help="Summary CSV path relative to the project root.",
    )
    parser.add_argument(
        "--per-clip-csv",
        default="results/evaluation/baseball/static_baseline/per_clip.csv",
        help="Per-clip scores. Needed to compare the control against a fitted model on the "
             "SAME clips, which is a paired comparison and far tighter than comparing summaries.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]

    recon_frames = int(eval_cfg["reconstruction_frames"])
    horizons = [int(h) for h in eval_cfg.get("recon_horizons", [5, 10, 20, 40])]

    feature_root = (
        PROJECT_ROOT
        / dataset_cfg["processed_root"]
        / args.eval_split
        / eval_cfg["feature_subdir"]
    )
    manifest_rows = read_csv_rows(feature_root / "manifest.csv")

    summary_rows: list[dict[str, Any]] = []
    for task_type in args.task_types:
        task_rows = [row for row in manifest_rows if row["task_type"] == task_type]
        core_r2s: list[float] = []
        core_nrmses: list[float] = []
        per_clip: list[dict[str, Any]] = []
        horizon_r2s: dict[int, list[float]] = {h: [] for h in horizons}

        for row in task_rows:
            obj = np.load(PROJECT_ROOT / row["filepath"])
            X = np.asarray(obj["X"], dtype=np.float64)
            feature_names = obj["feature_names"].tolist()

            frame_count = min(X.shape[0], recon_frames)
            X_recon = X[:frame_count]
            # Frozen model: dx/dt == 0, so the state never leaves its first observation.
            sim = np.repeat(X_recon[0:1], frame_count, axis=0)

            core_r2, core_nrmse = core_position_metrics(X_recon, sim, feature_names)
            core_r2s.append(core_r2)
            core_nrmses.append(core_nrmse)
            per_clip.append({"task_type": task_type, "clip_id": row["clip_id"],
                             "core_position_recon_r2": core_r2,
                             "core_position_recon_nrmse": core_nrmse})
            for h in horizons:
                if frame_count >= h:
                    h_r2, _ = core_position_metrics(X_recon[:h], sim[:h], feature_names)
                    horizon_r2s[h].append(h_r2)

        summary_row: dict[str, Any] = {
            "task_type": task_type,
            "num_clips": len(core_r2s),
            "reconstruction_frames": recon_frames,
            "mean_core_position_recon_r2": float(np.mean(core_r2s)) if core_r2s else "",
            "mean_core_position_recon_nrmse": float(np.mean(core_nrmses)) if core_nrmses else "",
        }
        for h in horizons:
            values = horizon_r2s[h]
            summary_row[f"mean_core_recon_r2_h{h}"] = float(np.mean(values)) if values else ""
        summary_rows.append(summary_row)

    fieldnames = [
        "task_type",
        "num_clips",
        "reconstruction_frames",
        "mean_core_position_recon_r2",
        "mean_core_position_recon_nrmse",
    ] + [f"mean_core_recon_r2_h{h}" for h in horizons]

    output_path = PROJECT_ROOT / args.output_csv
    write_csv(output_path, fieldnames, summary_rows)
    if per_clip:
        write_csv(PROJECT_ROOT / args.per_clip_csv,
                  ["task_type", "clip_id", "core_position_recon_r2", "core_position_recon_nrmse"],
                  per_clip)
    print(json.dumps(summary_rows, indent=2))
    print(f"output: {output_path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
