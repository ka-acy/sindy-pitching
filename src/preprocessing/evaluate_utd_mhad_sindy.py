#!/usr/bin/env python3
"""Evaluate trained UTD-MHAD PySINDy models on feature trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.sindy_utils import (
    inverse_transform_derivative,
    inverse_transform_trajectory,
    transform_derivatives,
    transform_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.utd_mhad.example.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--split",
        choices=("smoke_test", "proxy_subset"),
        default="proxy_subset",
        help="Feature/model split to evaluate.",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_feature_payload(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    obj = np.load(path)
    X = np.asarray(obj["X"], dtype=np.float64)
    X_dot = np.asarray(obj["X_dot"], dtype=np.float64)
    feature_names = obj["feature_names"].tolist()
    metadata = json.loads(str(obj["metadata"]))
    return X, X_dot, feature_names, metadata


def load_model(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def derivative_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def reconstruct_trial(
    model,
    X: np.ndarray,
    dt: float,
    reconstruction_frames: int,
    scaler: dict[str, Any] | None,
) -> tuple[np.ndarray | None, str | None, int]:
    frame_count = min(X.shape[0], reconstruction_frames)
    X = X[:frame_count]
    try:
        scaled = transform_trajectories([X], scaler)[0] if scaler is not None else X
        sim = np.zeros_like(X)
        sim[0] = X[0]
        sim_scaled = np.zeros_like(scaled)
        sim_scaled[0] = scaled[0]
        for idx in range(1, frame_count):
            x_dot = np.asarray(model.predict(sim_scaled[idx - 1 : idx]), dtype=np.float64)
            sim_scaled[idx] = sim_scaled[idx - 1] + dt * x_dot[0]
        sim = inverse_transform_trajectory(sim_scaled, scaler) if scaler is not None else sim_scaled
    except Exception as exc:  # pragma: no cover - defensive around stiff integrator failures
        return None, f"{type(exc).__name__}: {exc}", frame_count

    if sim.shape != X.shape:
        return None, f"shape_mismatch:{sim.shape}", frame_count
    if not np.isfinite(sim).all():
        return None, "non_finite_simulation", frame_count
    if np.max(np.abs(sim)) > 1e6:
        return None, "simulation_magnitude_exceeded", frame_count
    return sim, None, frame_count


def plot_reconstruction(
    path: Path,
    t: np.ndarray,
    X: np.ndarray,
    sim: np.ndarray,
    feature_names: list[str],
    selected_names: list[str],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = [feature_names.index(name) for name in selected_names if name in feature_names]
    if not indices:
        return

    fig, axes = plt.subplots(len(indices), 1, figsize=(9, 3 * len(indices)), sharex=True)
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices, strict=True):
        ax.plot(t, X[:, idx], label="observed", linewidth=1.8)
        ax.plot(t, sim[:, idx], label="simulated", linewidth=1.4, linestyle="--")
        ax.set_ylabel(feature_names[idx])
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config((project_root / args.config).resolve())

    eval_cfg = config["evaluation"]
    feature_root = project_root / config["dataset"]["processed_root"] / args.split / eval_cfg["feature_subdir"]
    equation_root = project_root / eval_cfg["model_results_root"] / args.split
    evaluation_root = project_root / eval_cfg["evaluation_root"] / args.split
    plot_root = project_root / eval_cfg["plot_root"] / args.split

    manifest_rows = read_csv_rows(feature_root / "manifest.csv")
    labels = sorted({row["subset_label"] for row in manifest_rows})
    by_label = {label: [row for row in manifest_rows if row["subset_label"] == label] for label in labels}
    models = {}
    scalers: dict[str, dict[str, Any] | None] = {}
    for label in labels:
        model_path = equation_root / label / "model.pkl"
        if not model_path.exists():
            continue
        models[label] = load_model(model_path)
        scaler_payload = load_json(equation_root / label / "scaler.json")
        scalers[label] = scaler_payload["feature_scaler"] if scaler_payload.get("scale_features") else None
    labels = sorted(models)

    derivative_summary_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []

    for train_label, model in models.items():
        for eval_label, rows in by_label.items():
            Xs: list[np.ndarray] = []
            X_dots: list[np.ndarray] = []
            for row in rows:
                X, X_dot, _, _ = load_feature_payload(project_root / row["filepath"])
                Xs.append(X)
                X_dots.append(X_dot)

            dt = float(rows[0]["dt"])
            scaler = scalers.get(train_label)
            scaled_Xs = transform_trajectories(Xs, scaler) if scaler is not None else Xs
            scaled_X_dots = transform_derivatives(X_dots, scaler) if scaler is not None else X_dots
            global_r2 = float(model.score(scaled_Xs, t=dt, x_dot=scaled_X_dots))
            rmses = []
            trial_r2s = []
            for X, X_dot, X_scaled in zip(Xs, X_dots, scaled_Xs, strict=True):
                pred = model.predict(X_scaled)
                pred = inverse_transform_derivative(pred, scaler) if scaler is not None else pred
                r2, rmse = derivative_metrics(X_dot, pred)
                trial_r2s.append(r2)
                rmses.append(rmse)
            transfer_rows.append(
                {
                    "train_label": train_label,
                    "eval_label": eval_label,
                    "num_trials": len(rows),
                    "global_r2": global_r2,
                    "mean_trial_r2": float(np.mean(trial_r2s)),
                    "mean_trial_rmse": float(np.mean(rmses)),
                }
            )

        plot_budget = int(eval_cfg["max_plot_trials_per_label"])
        plotted = 0
        for row in by_label[train_label]:
            X, X_dot, feature_names, metadata = load_feature_payload(project_root / row["filepath"])
            dt = float(row["dt"])
            scaler = scalers.get(train_label)
            X_scaled = transform_trajectories([X], scaler)[0] if scaler is not None else X
            pred = model.predict(X_scaled)
            pred = inverse_transform_derivative(pred, scaler) if scaler is not None else pred
            deriv_r2, deriv_rmse = derivative_metrics(X_dot, pred)

            sim, error, recon_frames = reconstruct_trial(
                model,
                X,
                dt,
                int(eval_cfg["reconstruction_frames"]),
                scaler,
            )
            recon_r2 = ""
            recon_rmse = ""
            stable = False
            if sim is not None:
                stable = True
                X_recon = X[:recon_frames]
                recon_r2, recon_rmse = derivative_metrics(X_recon, sim)
                if plotted < plot_budget:
                    t = np.arange(recon_frames, dtype=np.float64) * dt
                    plot_reconstruction(
                        plot_root / train_label / f"{Path(row['filepath']).stem}_reconstruction.png",
                        t,
                        X_recon,
                        sim,
                        feature_names,
                        list(eval_cfg["plot_feature_names"]),
                        title=f"{train_label} reconstruction: {Path(row['filepath']).stem}",
                    )
                    plotted += 1

            reconstruction_rows.append(
                {
                    "train_label": train_label,
                    "eval_label": row["subset_label"],
                    "subject_id": metadata["subject_id"],
                    "action_id": metadata["action_id"],
                    "trial_id": metadata["trial_id"],
                    "filepath": row["filepath"],
                    "num_frames": X.shape[0],
                    "reconstruction_frames": recon_frames,
                    "derivative_r2": deriv_r2,
                    "derivative_rmse": deriv_rmse,
                    "reconstruction_r2": recon_r2,
                    "reconstruction_rmse": recon_rmse,
                    "stable_simulation": stable,
                    "simulation_error": error or "",
                }
            )

        own_rows = [row for row in reconstruction_rows if row["train_label"] == train_label]
        stable_fraction = float(np.mean([1.0 if row["stable_simulation"] else 0.0 for row in own_rows]))
        recon_rmse_values = [float(row["reconstruction_rmse"]) for row in own_rows if row["reconstruction_rmse"] != ""]
        recon_r2_values = [float(row["reconstruction_r2"]) for row in own_rows if row["reconstruction_r2"] != ""]
        derivative_summary_rows.append(
            {
                "subset_label": train_label,
                "num_trials": len(own_rows),
                "stable_simulation_fraction": stable_fraction,
                "mean_reconstruction_r2": float(np.mean(recon_r2_values)) if recon_r2_values else "",
                "mean_reconstruction_rmse": float(np.mean(recon_rmse_values)) if recon_rmse_values else "",
            }
        )

    write_csv(
        evaluation_root / "transfer_metrics.csv",
        ["train_label", "eval_label", "num_trials", "global_r2", "mean_trial_r2", "mean_trial_rmse"],
        transfer_rows,
    )
    write_csv(
        evaluation_root / "reconstruction_metrics.csv",
        [
            "train_label",
            "eval_label",
            "subject_id",
            "action_id",
            "trial_id",
            "filepath",
            "num_frames",
            "reconstruction_frames",
            "derivative_r2",
            "derivative_rmse",
            "reconstruction_r2",
            "reconstruction_rmse",
            "stable_simulation",
            "simulation_error",
        ],
        reconstruction_rows,
    )
    write_csv(
        evaluation_root / "summary.csv",
        [
            "subset_label",
            "num_trials",
            "stable_simulation_fraction",
            "mean_reconstruction_r2",
            "mean_reconstruction_rmse",
        ],
        derivative_summary_rows,
    )
    write_json(
        evaluation_root / "config.json",
        {
            "split": args.split,
            "evaluation": eval_cfg,
            "labels": labels,
        },
    )

    print(f"evaluated models: {len(labels)}")
    print(f"evaluation root: {evaluation_root.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
