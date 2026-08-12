#!/usr/bin/env python3
"""Multi-step forecasting comparison on the fair core-position metric.

Derivative R2 is a poor final arbiter: a state with no velocity information cannot
express dx/dt at all, so derivative scores mostly measure whether the representation
encodes a derivative rather than whether it captures pitching dynamics. This script
compares multi-step forecasts under the same metric used for SINDy reconstruction
(`core_position_metrics`), against controls of increasing strength:

  frozen             hold the last seed frame constant                (no dynamics)
  constant_velocity  linear extrapolation from the seed velocity      (no dynamics)
  mean_pitch         phase-warped mean training trajectory, offset to the seed
                     (a template: "every pitch looks like the average pitch")
  mean_pitch_raw     the same template with no offset alignment
  delay_linear       ridge next-state map on d delay frames, rolled out autoregressively
                     (the representation that lifted derivative R2 to ~0.69)

Fairness: every method receives the same `--seed-frames` observed frames and is scored on
exactly the same window, frames [seed_frames, min(T, reconstruction_frames)). The template
methods additionally use the test clip's total duration to compute phase, and delay_linear
is fit on training clips only; both advantages are noted in the output.

A method is only worth pursuing if it beats BOTH no-dynamics controls and the template.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.evaluate_baseball_sindy import core_position_metrics

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TEMPLATE_POINTS = 101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/preprocessing.baseball.unaligned.handedness.yaml"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--task-type", default="pitching")
    parser.add_argument("--seed-frames", type=int, default=3, help="Observed frames given to every method.")
    parser.add_argument("--delays", type=int, default=3, help="Delay depth for delay_linear (must be <= seed-frames).")
    parser.add_argument("--output-csv", default="results/evaluation/baseball/forecast_baselines/summary.csv")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_clips(config: dict[str, Any], split: str, task_type: str) -> list[dict[str, Any]]:
    feature_root = (
        PROJECT_ROOT / config["dataset"]["processed_root"] / split / config["evaluation"]["feature_subdir"]
    )
    clips = []
    for row in read_csv_rows(feature_root / "manifest.csv"):
        if row["task_type"] != task_type:
            continue
        obj = np.load(PROJECT_ROOT / row["filepath"])
        clips.append(
            {
                "clip_id": row["clip_id"],
                "X": np.asarray(obj["X"], dtype=np.float64),
                "X_dot": np.asarray(obj["X_dot"], dtype=np.float64),
                "feature_names": obj["feature_names"].tolist(),
                "dt": float(row["dt"]),
            }
        )
    return clips


def build_template(clips: list[dict[str, Any]]) -> np.ndarray:
    """Phase-warp every training clip to TEMPLATE_POINTS and average -> (TEMPLATE_POINTS, n_feat)."""
    grid = np.linspace(0.0, 1.0, TEMPLATE_POINTS)
    stack = []
    for c in clips:
        X = c["X"]
        n = X.shape[0]
        if n < 2:
            continue
        src = np.linspace(0.0, 1.0, n)
        stack.append(np.column_stack([np.interp(grid, src, X[:, j]) for j in range(X.shape[1])]))
    return np.mean(np.stack(stack, axis=0), axis=0)


def sample_template(template: np.ndarray, clip_len: int, upto: int) -> np.ndarray:
    """Template values at the phases of frames [0, upto) of a clip of length clip_len."""
    grid = np.linspace(0.0, 1.0, template.shape[0])
    phases = (np.arange(upto, dtype=np.float64) / max(clip_len - 1, 1)).clip(0.0, 1.0)
    return np.column_stack([np.interp(phases, grid, template[:, j]) for j in range(template.shape[1])])


def fit_delay_map(clips: list[dict[str, Any]], depth: int) -> tuple[Any, Any]:
    """Ridge map [x(t), x(t-1), ..., x(t-depth+1)] -> x(t+1), fit on training clips."""
    designs, targets = [], []
    for c in clips:
        X = c["X"]
        n = X.shape[0]
        if n < depth + 1:
            continue
        for t in range(depth - 1, n - 1):
            designs.append(np.concatenate([X[t - k] for k in range(depth)]))
            targets.append(X[t + 1])
    D, Y = np.asarray(designs), np.asarray(targets)
    scaler = StandardScaler().fit(D)
    model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(D), Y)
    return scaler, model


def rollout_delay(scaler: Any, model: Any, X: np.ndarray, depth: int, upto: int) -> np.ndarray:
    """Autoregressive rollout seeded with the first `depth` observed frames."""
    history = [X[i].copy() for i in range(depth)]
    out = np.zeros((upto, X.shape[1]))
    out[:depth] = X[:depth]
    for t in range(depth, upto):
        design = np.concatenate([history[-1 - k] for k in range(depth)])[None, :]
        nxt = np.asarray(model.predict(scaler.transform(design)), dtype=np.float64)[0]
        out[t] = nxt
        history.append(nxt)
    return out


def teacher_forced_delay(scaler: Any, model: Any, X: np.ndarray, depth: int, upto: int) -> np.ndarray:
    """One-step-ahead predictions using TRUE history at every step (no compounding).

    Verification control for `rollout_delay`: if this scores well while the autoregressive
    rollout does not, the divergence is genuine error compounding rather than a broken
    rollout implementation.
    """
    out = np.zeros((upto, X.shape[1]))
    out[:depth] = X[:depth]
    for t in range(depth, upto):
        design = np.concatenate([X[t - 1 - k] for k in range(depth)])[None, :]
        out[t] = np.asarray(model.predict(scaler.transform(design)), dtype=np.float64)[0]
    return out


def persistence_teacher_forced(X: np.ndarray, depth: int, upto: int) -> np.ndarray:
    """One-step control: predict x(t) = x(t-1). No model at all.

    At ~60 fps consecutive frames are nearly identical, so persistence is a very strong
    one-step baseline. Any teacher-forced model must beat THIS, not the free-running
    frozen control, because one-step prediction and free-running rollout are different
    tasks and their scores are not comparable.
    """
    out = np.zeros((upto, X.shape[1]))
    out[:depth] = X[:depth]
    for t in range(depth, upto):
        out[t] = X[t - 1]
    return out


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    eval_cfg = config["evaluation"]
    recon_frames = int(eval_cfg["reconstruction_frames"])
    horizons = [int(h) for h in eval_cfg.get("recon_horizons", [5, 10, 20, 40])]
    seed = args.seed_frames
    depth = min(args.delays, seed)

    train_clips = load_clips(config, args.train_split, args.task_type)
    test_clips = load_clips(config, args.eval_split, args.task_type)
    if not train_clips or not test_clips:
        raise SystemExit("missing clips; check --config / splits / --task-type")

    template = build_template(train_clips)
    scaler, delay_model = fit_delay_map(train_clips, depth)
    print(f"train clips: {len(train_clips)}  test clips: {len(test_clips)}")
    print(f"seed frames: {seed}  delay depth: {depth}  scored window: [{seed}, min(T,{recon_frames}))\n")

    methods = ["frozen", "constant_velocity", "mean_pitch", "mean_pitch_raw", "delay_linear", "delay_linear_teacher_forced", "persistence_teacher_forced"]
    acc: dict[str, dict[str, list[float]]] = {
        m: {"r2": [], "nrmse": [], **{f"h{h}": [] for h in horizons}} for m in methods
    }

    for c in test_clips:
        X, X_dot, names, dt = c["X"], c["X_dot"], c["feature_names"], c["dt"]
        n = X.shape[0]
        upto = min(n, recon_frames)
        if upto <= seed + 1:
            continue

        anchor = X[seed - 1]
        steps = np.arange(upto, dtype=np.float64) - (seed - 1)

        preds: dict[str, np.ndarray] = {}
        preds["frozen"] = np.repeat(anchor[None, :], upto, axis=0)
        preds["constant_velocity"] = anchor[None, :] + steps[:, None] * dt * X_dot[seed - 1][None, :]
        raw_template = sample_template(template, n, upto)
        preds["mean_pitch_raw"] = raw_template
        preds["mean_pitch"] = raw_template + (anchor - raw_template[seed - 1])[None, :]
        preds["delay_linear"] = rollout_delay(scaler, delay_model, X, depth, upto)
        preds["delay_linear_teacher_forced"] = teacher_forced_delay(scaler, delay_model, X, depth, upto)
        preds["persistence_teacher_forced"] = persistence_teacher_forced(X, depth, upto)

        for m, sim in preds.items():
            true_w, sim_w = X[seed:upto], sim[seed:upto]
            r2, nrmse = core_position_metrics(true_w, sim_w, names)
            acc[m]["r2"].append(r2)
            acc[m]["nrmse"].append(nrmse)
            for h in horizons:
                end = min(seed + h, upto)
                if end > seed + 1:
                    h_r2, _ = core_position_metrics(X[seed:end], sim[seed:end], names)
                    acc[m][f"h{h}"].append(h_r2)

    rows = []
    header = f"{'method':20s} {'core R2':>10s} {'nrmse':>8s}" + "".join(f"{'h'+str(h):>10s}" for h in horizons)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = {
            "method": m,
            "num_clips": len(acc[m]["r2"]),
            "seed_frames": seed,
            "core_position_recon_r2": float(np.mean(acc[m]["r2"])) if acc[m]["r2"] else "",
            "core_position_recon_nrmse": float(np.mean(acc[m]["nrmse"])) if acc[m]["nrmse"] else "",
        }
        for h in horizons:
            vals = acc[m][f"h{h}"]
            row[f"core_recon_r2_h{h}"] = float(np.mean(vals)) if vals else ""
        rows.append(row)
        line = f"{m:20s} {row['core_position_recon_r2']:>10.4f} {row['core_position_recon_nrmse']:>8.4f}"
        line += "".join(f"{row[f'core_recon_r2_h{h}']:>10.4f}" for h in horizons)
        print(line)

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\noutput: {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
