#!/usr/bin/env python3
"""Run and compare multiple Stage B baseball configs."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "config/preprocessing.baseball.unaligned.yaml",
            "config/preprocessing.baseball.aligned.yaml",
        ],
        help="Baseball config paths relative to the project root.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/evaluation/baseball/config_comparison.csv",
        help="Comparison CSV path relative to the project root.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def read_csv(path: Path) -> list[dict[str, str]]:
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
    python_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        config_name = Path(config_path).stem
        run_command([python_bin, "src/preprocessing/extract_baseball_features.py", "--config", config_path, "--split", "train"])
        run_command([python_bin, "src/preprocessing/extract_baseball_features.py", "--config", config_path, "--split", "test"])
        run_command([python_bin, "src/preprocessing/fit_baseball_sindy.py", "--config", config_path, "--split", "train", "--task-types", "pitching"])
        run_command([python_bin, "src/preprocessing/evaluate_baseball_sindy.py", "--config", config_path, "--train-split", "train", "--eval-split", "test", "--task-types", "pitching"])

        train_summary = read_csv(PROJECT_ROOT / "results" / "equations" / "baseball" / "train" / "summary.csv")[0]
        eval_summary = read_csv(PROJECT_ROOT / "results" / "evaluation" / "baseball" / "train_on_test" / "summary.csv")[0]
        rows.append(
            {
                "config_name": config_name,
                "config_path": config_path,
                "train_num_terms": train_summary["num_equation_terms"],
                "train_mean_trial_rmse": train_summary["mean_trial_rmse"],
                "test_mean_derivative_r2": eval_summary["mean_derivative_r2"],
                "test_mean_derivative_rmse": eval_summary["mean_derivative_rmse"],
                "test_stable_simulation_fraction": eval_summary["stable_simulation_fraction"],
                "test_mean_reconstruction_r2": eval_summary["mean_reconstruction_r2"],
                "test_mean_reconstruction_rmse": eval_summary["mean_reconstruction_rmse"],
                "test_core_position_recon_r2": eval_summary.get("mean_core_position_recon_r2", ""),
                "test_core_position_recon_nrmse": eval_summary.get("mean_core_position_recon_nrmse", ""),
            }
        )

    write_csv(
        PROJECT_ROOT / args.output_csv,
        [
            "config_name",
            "config_path",
            "train_num_terms",
            "train_mean_trial_rmse",
            "test_mean_derivative_r2",
            "test_mean_derivative_rmse",
            "test_stable_simulation_fraction",
            "test_mean_reconstruction_r2",
            "test_mean_reconstruction_rmse",
            "test_core_position_recon_r2",
            "test_core_position_recon_nrmse",
        ],
        rows,
    )
    print(f"compared configs: {len(rows)}")
    print(f"output: {(PROJECT_ROOT / args.output_csv).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
