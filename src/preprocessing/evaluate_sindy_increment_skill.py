#!/usr/bin/env python3
"""Score the FITTED SINDy models on short-horizon increment prediction.

Closes the gap flagged in the technical paper: `diagnose_increment_skill.py` showed that
the pitching feature data supports short-horizon prediction (0.39 uniform skill over
persistence), but it demonstrated that with a delay-embedded ridge model, not with the
sparse STLSQ equations this pipeline actually produces. So the paper could claim only that
the *data* carries short-horizon structure, not that the pipeline's own artifacts capture it.

This script evaluates the fitted SINDy model directly on the same task. The model predicts
dx/dt, so its one-step increment prediction is

    delta_hat(t) = dt * f(x(t))

scored against the true forward increment delta(t) = x(t+1) - x(t) via skill over
persistence (zero increment), reported under both pooled and uniform aggregation:

    skill = 1 - SSE(model) / SSE(zero-increment)

Interpretation:
  - positive uniform skill  -> the pipeline's sparse equations DO capture short-horizon
                              structure, and the positive result belongs to the pipeline;
  - skill <= 0              -> the data carries structure the sparse models do not capture,
                              and the paper's claim must stay narrower.

Caveat recorded in the output: the model is fit against Savitzky-Golay centered derivatives
while the target here is a forward difference, so a small systematic mismatch is expected.
The `savgol_derivative_reference` row quantifies it by scoring dt * X_dot (the derivative
the model was trained to reproduce) on the same task -- that is the ceiling any model fit
to these derivative targets could reach, and the model should be judged against it.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.evaluate_baseball_sindy import CORE_POSITIONS
from src.preprocessing.sindy_utils import inverse_transform_derivative, transform_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/preprocessing.baseball.unaligned.handedness.yaml"))
    parser.add_argument(
        "--model-root",
        default="results/equations/baseball/train/pitching",
        help="Directory holding model.pkl and scaler.json for the fitted model.",
    )
    parser.add_argument("--model-label", default="", help="Label for the output row; defaults to the model-root name.")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--task-type", default="pitching")
    parser.add_argument(
        "--column-set",
        default="core_positions",
        choices=("core_positions", "rates"),
        help="Which state columns to score. For a second-order state, position-increment "
             "prediction is trivial because velocity is an input column, so `rates` "
             "(acceleration prediction) is the meaningful task.",
    )
    parser.add_argument("--output-csv", default="results/analysis/sindy_increment_skill.csv")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_clips(config: dict[str, Any], split: str, task_type: str) -> list[dict[str, Any]]:
    root = PROJECT_ROOT / config["dataset"]["processed_root"] / split / config["evaluation"]["feature_subdir"]
    with (root / "manifest.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["task_type"] == task_type]
    clips = []
    for row in rows:
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


def skill_pooled(true_d: np.ndarray, pred_d: np.ndarray) -> float:
    sse_zero = float(np.sum(true_d**2))
    return 1.0 - float(np.sum((true_d - pred_d) ** 2)) / sse_zero if sse_zero > 0 else float("nan")


def skill_uniform(true_d: np.ndarray, pred_d: np.ndarray) -> float:
    num = np.sum((true_d - pred_d) ** 2, axis=0)
    den = np.sum(true_d**2, axis=0)
    keep = den > 0
    return float(np.mean(1.0 - num[keep] / den[keep])) if keep.any() else float("nan")


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    clips = load_clips(config, args.eval_split, args.task_type)
    if not clips:
        raise SystemExit("no clips loaded")

    model_root = PROJECT_ROOT / args.model_root
    with (model_root / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    with (model_root / "scaler.json").open() as handle:
        payload = json.load(handle)
    scaler = payload["feature_scaler"] if payload.get("scale_features") else None
    label = args.model_label or Path(args.model_root).parent.name

    names = clips[0]["feature_names"]
    if args.column_set == "rates":
        idx = [i for i, n in enumerate(names) if n.endswith("_rate")]
        if not idx:
            raise SystemExit("--column-set rates requires a state with _rate columns")
    else:
        idx = [names.index(n) for n in CORE_POSITIONS if n in names]

    true_inc, pred_inc, savgol_inc = [], [], []
    for c in clips:
        X, X_dot, dt = c["X"], c["X_dot"], c["dt"]
        if X.shape[0] < 2:
            continue
        Xs = transform_trajectories([X], scaler)[0] if scaler is not None else X
        pred = np.asarray(model.predict(Xs), dtype=np.float64)
        if scaler is not None:
            pred = inverse_transform_derivative(pred, scaler)
        # Align: forward increment at t uses x(t+1) - x(t), so drop the last frame.
        true_inc.append(X[1:] - X[:-1])
        pred_inc.append(dt * pred[:-1])
        savgol_inc.append(dt * X_dot[:-1])

    T = np.vstack(true_inc)[:, idx]
    P = np.vstack(pred_inc)[:, idx]
    S = np.vstack(savgol_inc)[:, idx]

    rows = [
        {
            "model": "persistence (no model)",
            "source": "-",
            "skill_pooled": 0.0,
            "skill_uniform": 0.0,
            "note": f"zero increment on {args.column_set}; the control this task is scored against",
        },
        {
            "model": f"SINDy fitted ({label})",
            "source": args.model_root,
            "skill_pooled": skill_pooled(T, P),
            "skill_uniform": skill_uniform(T, P),
            "note": "dt * f(x(t)) from the fitted sparse equations",
        },
        {
            "model": "savgol_derivative_reference",
            "source": "X_dot from feature payloads",
            "skill_pooled": skill_pooled(T, S),
            "skill_uniform": skill_uniform(T, S),
            "note": "CEILING for any model fit to these derivative targets",
        },
    ]

    print(f"clips: {len(clips)}   column set: {args.column_set} ({len(idx)} cols)   rows scored: {T.shape[0]}")
    print(f"model: {args.model_root}\n")
    print(f"{'model':34s} {'skill(pool)':>12s} {'skill(unif)':>12s}")
    for r in rows:
        print(f"{r['model']:34s} {r['skill_pooled']:>12.4f} {r['skill_uniform']:>12.4f}")

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\noutput: {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
