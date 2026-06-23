#!/usr/bin/env python3
"""Evaluate trained baseball PySINDy models on held-out baseball feature trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
import sys
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
    transform_trajectories,
)

CORE_POSITIONS = [
    "shoulder_rotation_angle",
    "hip_rotation_angle",
    "dominant_elbow_flexion_angle",
    "trail_elbow_flexion_angle",
    "dominant_knee_flexion_angle",
    "trail_knee_flexion_angle",
    "lead_knee_lift",
    "stride_ankle_separation",
]


def core_position_metrics(
    X_true: np.ndarray, sim: np.ndarray, feature_names: list[str]
) -> tuple[float, float]:
    """Fair, state-comparable reconstruction metrics over the shared core positions.

    Returns (uniform-average R2, mean normalized RMSE) computed only on the
    CORE_POSITIONS columns present in feature_names. NRMSE normalizes each
    column's RMSE by that column's std so the value is unit-free and
    comparable across different state-vector definitions.
    """
    core_idx = [feature_names.index(name) for name in CORE_POSITIONS if name in feature_names]
    if not core_idx:
        return float("nan"), float("nan")
    true_c = X_true[:, core_idx]
    sim_c = sim[:, core_idx]
    r2 = float(r2_score(true_c, sim_c, multioutput="uniform_average"))
    per_col_rmse = np.sqrt(np.mean((sim_c - true_c) ** 2, axis=0))
    per_col_std = np.std(true_c, axis=0) + 1e-8
    nrmse = float(np.mean(per_col_rmse / per_col_std))
    return r2, nrmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.baseball.example.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Baseball split used to fit the model.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        help="Baseball split used for held-out evaluation.",
    )
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=("pitching",),
        help="Task types to evaluate.",
    )
    parser.add_argument(
        "--group-by",
        default="",
        help="Optional metadata field used to evaluate subgroup-specific models, e.g. resolved_handedness.",
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


def reconstruct_trial(model, X: np.ndarray, dt: float, reconstruction_frames: int, scaler: dict[str, Any] | None, integrator: str = "euler"):
    frame_count = min(X.shape[0], reconstruction_frames)
    X = X[:frame_count]
    try:
        scaled = transform_trajectories([X], scaler)[0] if scaler is not None else X
        sim_scaled = np.zeros_like(scaled)
        sim_scaled[0] = scaled[0]
        if integrator == "rk4":
            def f(state: np.ndarray) -> np.ndarray:
                return np.asarray(model.predict(state[None, :]), dtype=np.float64)[0]
            for idx in range(1, frame_count):
                x = sim_scaled[idx - 1]
                k1 = f(x)
                k2 = f(x + 0.5 * dt * k1)
                k3 = f(x + 0.5 * dt * k2)
                k4 = f(x + dt * k3)
                sim_scaled[idx] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            for idx in range(1, frame_count):
                x_dot = np.asarray(model.predict(sim_scaled[idx - 1 : idx]), dtype=np.float64)
                sim_scaled[idx] = sim_scaled[idx - 1] + dt * x_dot[0]
        sim = inverse_transform_trajectory(sim_scaled, scaler) if scaler is not None else sim_scaled
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", frame_count

    if sim.shape != X.shape:
        return None, f"shape_mismatch:{sim.shape}", frame_count
    if not np.isfinite(sim).all():
        return None, "non_finite_simulation", frame_count
    if np.max(np.abs(sim)) > 1e6:
        return None, "simulation_magnitude_exceeded", frame_count
    return sim, None, frame_count


def plot_reconstruction(path: Path, t: np.ndarray, X: np.ndarray, sim: np.ndarray, feature_names: list[str], selected_names: list[str], title: str) -> None:
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
    config = load_config((PROJECT_ROOT / args.config).resolve())
    eval_cfg = config["evaluation"]
    integrator = str(eval_cfg.get("integrator", "euler"))

    feature_root = PROJECT_ROOT / "data" / "baseball" / "processed" / args.eval_split / eval_cfg["feature_subdir"]
    equation_root = PROJECT_ROOT / "results" / "equations" / "baseball" / args.train_split
    evaluation_root = PROJECT_ROOT / "results" / "evaluation" / "baseball" / f"{args.train_split}_on_{args.eval_split}"
    plot_root = PROJECT_ROOT / "results" / "plots" / "baseball" / f"{args.train_split}_on_{args.eval_split}"
    if args.group_by:
        evaluation_root = PROJECT_ROOT / "results" / "evaluation" / "baseball" / f"{args.train_split}_on_{args.eval_split}_by_{args.group_by}"
        plot_root = PROJECT_ROOT / "results" / "plots" / "baseball" / f"{args.train_split}_on_{args.eval_split}_by_{args.group_by}"

    manifest_rows = read_csv_rows(feature_root / "manifest.csv")
    reconstruction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for task_type in args.task_types:
        task_rows = [row for row in manifest_rows if row["task_type"] == task_type]
        if not task_rows:
            continue
        loaded_items = []
        for row in task_rows:
            X, X_dot, feature_names, metadata = load_feature_payload(PROJECT_ROOT / row["filepath"])
            loaded_items.append((row, X, X_dot, feature_names, metadata))

        grouped_items: dict[str, list[tuple[dict[str, str], np.ndarray, np.ndarray, list[str], dict[str, Any]]]] = {}
        if args.group_by:
            for item in loaded_items:
                group_value = str(item[4].get(args.group_by, "")).strip() or "unknown"
                grouped_items.setdefault(group_value, []).append(item)
        else:
            grouped_items["all"] = loaded_items

        for group_value, items in grouped_items.items():
            if not items:
                continue
            model_root = equation_root / task_type
            if args.group_by:
                model_root = model_root / args.group_by / group_value
            model = load_model(model_root / "model.pkl")
            scaler_payload = load_json(model_root / "scaler.json")
            scaler = scaler_payload["feature_scaler"] if scaler_payload.get("scale_features") else None

            deriv_r2s: list[float] = []
            deriv_rmses: list[float] = []
            recon_r2s: list[float] = []
            recon_rmses: list[float] = []
            core_r2s: list[float] = []
            core_nrmses: list[float] = []
            stable_flags: list[float] = []
            plotted = 0

            for row, X, X_dot, feature_names, metadata in items:
                dt = float(row["dt"])
                X_scaled = transform_trajectories([X], scaler)[0] if scaler is not None else X
                pred = model.predict(X_scaled)
                pred = inverse_transform_derivative(pred, scaler) if scaler is not None else pred
                deriv_r2, deriv_rmse = derivative_metrics(X_dot, pred)

                sim, error, recon_frames = reconstruct_trial(
                    model, X, dt, int(eval_cfg["reconstruction_frames"]), scaler, integrator=integrator
                )
                recon_r2 = ""
                recon_rmse = ""
                core_position_recon_r2 = ""
                core_position_recon_nrmse = ""
                stable = False
                if sim is not None:
                    stable = True
                    X_recon = X[:recon_frames]
                    recon_r2, recon_rmse = derivative_metrics(X_recon, sim)
                    recon_r2s.append(float(recon_r2))
                    recon_rmses.append(float(recon_rmse))
                    core_r2, core_nrmse = core_position_metrics(X_recon, sim, feature_names)
                    core_r2s.append(core_r2)
                    core_nrmses.append(core_nrmse)
                    core_position_recon_r2 = core_r2
                    core_position_recon_nrmse = core_nrmse
                    if plotted < int(eval_cfg["max_plot_trials_per_label"]):
                        t = np.arange(recon_frames, dtype=np.float64) * dt
                        plot_path = plot_root / task_type
                        if args.group_by:
                            plot_path = plot_path / args.group_by / group_value
                        plot_reconstruction(
                            plot_path / f"{Path(row['filepath']).stem}_reconstruction.png",
                            t,
                            X_recon,
                            sim,
                            feature_names,
                            list(eval_cfg["plot_feature_names"]),
                            title=f"{task_type} {group_value} reconstruction: {Path(row['filepath']).stem}",
                        )
                        plotted += 1

                deriv_r2s.append(deriv_r2)
                deriv_rmses.append(deriv_rmse)
                stable_flags.append(1.0 if stable else 0.0)
                reconstruction_rows.append(
                    {
                        "task_type": task_type,
                        "group_by": args.group_by,
                        "group_value": group_value,
                        "clip_id": metadata.get("clip_id", row["clip_id"]),
                        "filepath": row["filepath"],
                        "num_frames": X.shape[0],
                        "reconstruction_frames": recon_frames,
                        "derivative_r2": deriv_r2,
                        "derivative_rmse": deriv_rmse,
                        "reconstruction_r2": recon_r2,
                        "reconstruction_rmse": recon_rmse,
                        "core_position_recon_r2": core_position_recon_r2,
                        "core_position_recon_nrmse": core_position_recon_nrmse,
                        "stable_simulation": stable,
                        "simulation_error": error or "",
                    }
                )

            summary_rows.append(
                {
                    "task_type": task_type,
                    "group_by": args.group_by,
                    "group_value": group_value,
                    "num_clips": len(items),
                    "mean_derivative_r2": float(np.mean(deriv_r2s)) if deriv_r2s else "",
                    "mean_derivative_rmse": float(np.mean(deriv_rmses)) if deriv_rmses else "",
                    "stable_simulation_fraction": float(np.mean(stable_flags)) if stable_flags else "",
                    "mean_reconstruction_r2": float(np.mean(recon_r2s)) if recon_r2s else "",
                    "mean_reconstruction_rmse": float(np.mean(recon_rmses)) if recon_rmses else "",
                    "mean_core_position_recon_r2": float(np.mean(core_r2s)) if core_r2s else "",
                    "mean_core_position_recon_nrmse": float(np.mean(core_nrmses)) if core_nrmses else "",
                }
            )

    write_csv(
        evaluation_root / "reconstruction_metrics.csv",
        [
            "task_type",
            "group_by",
            "group_value",
            "clip_id",
            "filepath",
            "num_frames",
            "reconstruction_frames",
            "derivative_r2",
            "derivative_rmse",
            "reconstruction_r2",
            "reconstruction_rmse",
            "core_position_recon_r2",
            "core_position_recon_nrmse",
            "stable_simulation",
            "simulation_error",
        ],
        reconstruction_rows,
    )
    write_csv(
        evaluation_root / "summary.csv",
        [
            "task_type",
            "group_by",
            "group_value",
            "num_clips",
            "mean_derivative_r2",
            "mean_derivative_rmse",
            "stable_simulation_fraction",
            "mean_reconstruction_r2",
            "mean_reconstruction_rmse",
            "mean_core_position_recon_r2",
            "mean_core_position_recon_nrmse",
        ],
        summary_rows,
    )
    write_json(
        evaluation_root / "config.json",
        {
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "task_types": list(args.task_types),
            "group_by": args.group_by,
            "evaluation": eval_cfg,
        },
    )

    print(f"evaluated task types: {len(summary_rows)}")
    print(f"evaluation root: {evaluation_root.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
