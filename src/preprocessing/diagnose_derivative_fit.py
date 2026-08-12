#!/usr/bin/env python3
"""Diagnose WHY one-step derivative R2 is ~0, at the regression level.

The fitted SINDy models score derivative R2 ~= -0.018 on held-out clips, i.e. no better
than predicting the mean derivative. That failure happens in the regression itself, before
any integration, so it cannot be explained by rollout error. This script isolates the cause
by fitting plain ridge regressions (no sparsity, so sparse-selection artifacts cannot
confound the result) over several state representations and comparing held-out derivative R2:

  global_linear          [1, x]                  - structural match to the current degree-1 SINDy
  global_linear_phase    [1, x, phi]             - additive phase drift (non-autonomous)
  global_phase_interact  [1, x, phi, phi*x]      - phase-modulated linear system A(phi) x
  phase_binned_k{K}      [1, x] fit per phase bin - piecewise-in-phase linear system
  delay_embed_d{D}       [1, x(t), x(t-tau), ...] - Takens embedding; restores Markovianity
  per_clip_within        [1, x] fit per clip, early frames -> late frames (within-clip split)
  per_clip_oracle        [1, x] fit per clip, in-sample     - upper bound, not a real score

Interpretation guide:
  - If phase / delay / phase-binned lift R2 well above global_linear, the autonomous
    dx/dt = f(x) assumption was the problem: the map is multivalued in x alone.
  - If per_clip_oracle is high but everything cross-clip is ~0, the clips are too
    heterogeneous for a single global model and no global fit will work.
  - If even per_clip_oracle is ~0, the derivative targets are dominated by pose noise
    and no model class helps until the pose or derivative estimation improves.

Cross-clip scores use GroupKFold over clips, so no clip contributes to both train and test.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.baseball.unaligned.handedness.yaml"),
        help="YAML config path (supplies processed_root and feature_subdir).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "test"),
        help="Splits to pool. Cross-clip CV is grouped by clip, so pooling is safe.",
    )
    parser.add_argument("--task-type", default="pitching")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--phase-bins", nargs="*", type=int, default=(4, 8))
    parser.add_argument("--delay-taus", nargs="*", type=int, default=(3,))
    parser.add_argument("--delay-depths", nargs="*", type=int, default=(3, 5))
    parser.add_argument(
        "--output-csv",
        default="results/analysis/derivative_fit_diagnostics.csv",
        help="Output CSV path relative to the project root.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_clips(config: dict[str, Any], splits: list[str], task_type: str) -> list[dict[str, Any]]:
    """Load per-clip (X, X_dot, phase) arrays for the requested splits."""
    dataset_cfg = config["dataset"]
    feature_subdir = config["evaluation"]["feature_subdir"]
    clips: list[dict[str, Any]] = []
    for split in splits:
        feature_root = PROJECT_ROOT / dataset_cfg["processed_root"] / split / feature_subdir
        for row in read_csv_rows(feature_root / "manifest.csv"):
            if row["task_type"] != task_type:
                continue
            obj = np.load(PROJECT_ROOT / row["filepath"])
            X = np.asarray(obj["X"], dtype=np.float64)
            X_dot = np.asarray(obj["X_dot"], dtype=np.float64)
            n = X.shape[0]
            phase = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
            clips.append(
                {
                    "clip_id": row["clip_id"],
                    "split": split,
                    "X": X,
                    "X_dot": X_dot,
                    "phase": phase,
                    "dt": float(row["dt"]),
                }
            )
    return clips


def fit_predict_r2(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, Y_te: np.ndarray
) -> np.ndarray | None:
    """Standardize on train, fit multi-output ridge, return predictions for X_te."""
    if X_tr.shape[0] <= X_tr.shape[1] + 1 or X_te.shape[0] == 0:
        return None
    scaler = StandardScaler().fit(X_tr)
    model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(X_tr), Y_tr)
    return np.asarray(model.predict(scaler.transform(X_te)), dtype=np.float64)


def score(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(r2_score(Y_true, Y_pred, multioutput="uniform_average"))


# --- design-matrix builders: each returns (design, target, valid_row_mask) per clip ---


def design_plain(clip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return clip["X"], clip["X_dot"]


def design_phase(clip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.column_stack([clip["X"], clip["phase"]]), clip["X_dot"]


def design_phase_interact(clip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    X, phase = clip["X"], clip["phase"][:, None]
    return np.column_stack([X, phase, X * phase]), clip["X_dot"]


def design_delay(clip: dict[str, Any], tau: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
    X, X_dot = clip["X"], clip["X_dot"]
    span = tau * (depth - 1)
    if X.shape[0] <= span:
        return np.zeros((0, X.shape[1] * depth)), np.zeros((0, X_dot.shape[1]))
    blocks = [X[span - tau * d : X.shape[0] - tau * d] for d in range(depth)]
    return np.column_stack(blocks), X_dot[span:]


def cross_clip_r2(
    clips: list[dict[str, Any]], builder, n_folds: int
) -> tuple[float, float, int]:
    """GroupKFold over clips. Returns (pooled_r2, mean_per_clip_r2, n_design_features)."""
    built = [builder(c) for c in clips]
    keep = [i for i, (d, _) in enumerate(built) if d.shape[0] > 0]
    if not keep:
        return float("nan"), float("nan"), 0

    designs = [built[i][0] for i in keep]
    targets = [built[i][1] for i in keep]
    groups = np.concatenate([np.full(designs[k].shape[0], k) for k in range(len(keep))])
    D = np.vstack(designs)
    Y = np.vstack(targets)

    folds = GroupKFold(n_splits=min(n_folds, len(keep)))
    preds = np.full_like(Y, np.nan)
    for tr_idx, te_idx in folds.split(D, Y, groups=groups):
        out = fit_predict_r2(D[tr_idx], Y[tr_idx], D[te_idx], Y[te_idx])
        if out is not None:
            preds[te_idx] = out

    valid = ~np.isnan(preds).any(axis=1)
    if valid.sum() == 0:
        return float("nan"), float("nan"), D.shape[1]

    pooled = score(Y[valid], preds[valid])
    per_clip = []
    for k in range(len(keep)):
        mask = (groups == k) & valid
        if mask.sum() > 2:
            per_clip.append(score(Y[mask], preds[mask]))
    return pooled, float(np.mean(per_clip)) if per_clip else float("nan"), D.shape[1]


def phase_binned_r2(clips: list[dict[str, Any]], n_bins: int, n_folds: int) -> tuple[float, float, int]:
    """Separate [1, x] fit per phase bin, evaluated cross-clip."""
    designs = [c["X"] for c in clips]
    targets = [c["X_dot"] for c in clips]
    bins = [np.clip((c["phase"] * n_bins).astype(int), 0, n_bins - 1) for c in clips]
    groups = np.concatenate([np.full(designs[k].shape[0], k) for k in range(len(clips))])
    D, Y, B = np.vstack(designs), np.vstack(targets), np.concatenate(bins)

    folds = GroupKFold(n_splits=min(n_folds, len(clips)))
    preds = np.full_like(Y, np.nan)
    for tr_idx, te_idx in folds.split(D, Y, groups=groups):
        for b in range(n_bins):
            tr_b = tr_idx[B[tr_idx] == b]
            te_b = te_idx[B[te_idx] == b]
            if te_b.size == 0:
                continue
            out = fit_predict_r2(D[tr_b], Y[tr_b], D[te_b], Y[te_b])
            if out is not None:
                preds[te_b] = out

    valid = ~np.isnan(preds).any(axis=1)
    if valid.sum() == 0:
        return float("nan"), float("nan"), D.shape[1]
    pooled = score(Y[valid], preds[valid])
    per_clip = []
    for k in range(len(clips)):
        mask = (groups == k) & valid
        if mask.sum() > 2:
            per_clip.append(score(Y[mask], preds[mask]))
    return pooled, float(np.mean(per_clip)) if per_clip else float("nan"), D.shape[1]


def finite_difference_control(clips: list[dict[str, Any]], tau: int) -> tuple[float, float]:
    """Control for the delay-embedding result: NO fitting, just a backward difference.

    A delay-embedded linear model can express (x(t) - x(t-tau)) / (tau*dt), i.e. plain
    numerical differentiation. If that unfitted estimator already scores as well as the
    fitted delay model, then the delay model's apparent gain is rediscovered
    differentiation, not learned dynamics. Returns (pooled_r2, mean_per_clip_r2).
    """
    preds, trues, groups = [], [], []
    for k, c in enumerate(clips):
        X, Y, dt = c["X"], c["X_dot"], c["dt"]
        if X.shape[0] <= tau:
            continue
        est = (X[tau:] - X[:-tau]) / (tau * dt)
        preds.append(est)
        trues.append(Y[tau:])
        groups.append(np.full(est.shape[0], k))
    if not preds:
        return float("nan"), float("nan")
    P, T, G = np.vstack(preds), np.vstack(trues), np.concatenate(groups)
    per_clip = [score(T[G == k], P[G == k]) for k in np.unique(G) if (G == k).sum() > 2]
    return score(T, P), float(np.mean(per_clip)) if per_clip else float("nan")


def per_clip_scores(clips: list[dict[str, Any]], train_frac: float = 0.7) -> tuple[float, float]:
    """Per-clip fits: (mean within-clip extrapolation R2, mean in-sample oracle R2)."""
    within, oracle = [], []
    for c in clips:
        X, Y = c["X"], c["X_dot"]
        n = X.shape[0]
        cut = int(n * train_frac)
        if cut > X.shape[1] + 2 and n - cut > 2:
            out = fit_predict_r2(X[:cut], Y[:cut], X[cut:], Y[cut:])
            if out is not None:
                within.append(score(Y[cut:], out))
        if n > X.shape[1] + 2:
            out = fit_predict_r2(X, Y, X, Y)
            if out is not None:
                oracle.append(score(Y, out))
    return (
        float(np.mean(within)) if within else float("nan"),
        float(np.mean(oracle)) if oracle else float("nan"),
    )


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    clips = load_clips(config, list(args.splits), args.task_type)
    if not clips:
        raise SystemExit("no clips loaded; check --config / --splits / --task-type")

    lengths = [c["X"].shape[0] for c in clips]
    print(f"clips: {len(clips)}  frames: total={sum(lengths)} min={min(lengths)} max={max(lengths)}")
    print(f"cross-clip protocol: GroupKFold(n_splits={min(args.n_folds, len(clips))}) grouped by clip\n")

    rows: list[dict[str, Any]] = []

    def record(name: str, pooled: float, per_clip: float, n_feat: int, kind: str, note: str) -> None:
        rows.append(
            {
                "representation": name,
                "protocol": kind,
                "n_design_features": n_feat,
                "pooled_derivative_r2": pooled,
                "mean_per_clip_derivative_r2": per_clip,
                "note": note,
            }
        )
        print(f"{name:24s} {kind:16s} feat={n_feat:>4d}  pooled R2={pooled:>9.4f}  per-clip R2={per_clip:>9.4f}")

    p, c, f = cross_clip_r2(clips, design_plain, args.n_folds)
    record("global_linear", p, c, f, "cross-clip", "structural match to current degree-1 SINDy")

    p, c, f = cross_clip_r2(clips, design_phase, args.n_folds)
    record("global_linear_phase", p, c, f, "cross-clip", "additive normalized-phase term")

    p, c, f = cross_clip_r2(clips, design_phase_interact, args.n_folds)
    record("global_phase_interact", p, c, f, "cross-clip", "phase-modulated linear system A(phi)x")

    for k in args.phase_bins:
        p, c, f = phase_binned_r2(clips, k, args.n_folds)
        record(f"phase_binned_k{k}", p, c, f, "cross-clip", f"separate linear fit in {k} phase bins")

    for tau in args.delay_taus:
        for depth in args.delay_depths:
            p, c, f = cross_clip_r2(
                clips, lambda cl, t=tau, d=depth: design_delay(cl, t, d), args.n_folds
            )
            record(
                f"delay_embed_tau{tau}_d{depth}", p, c, f, "cross-clip",
                f"Takens embedding, {depth} delays spaced {tau} frames",
            )

    for tau in args.delay_taus:
        p, c = finite_difference_control(clips, tau)
        record(
            f"finite_diff_control_tau{tau}", p, c, 0, "no-fit control",
            "unfitted backward difference - the control for delay_embed",
        )

    within, oracle = per_clip_scores(clips)
    record("per_clip_within", float("nan"), within, clips[0]["X"].shape[1], "within-clip",
           "fit early 70% of a clip, predict its late 30%")
    record("per_clip_oracle", float("nan"), oracle, clips[0]["X"].shape[1], "in-sample",
           "UPPER BOUND ONLY - fit and scored on the same clip")

    output_path = PROJECT_ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "representation",
                "protocol",
                "n_design_features",
                "pooled_derivative_r2",
                "mean_per_clip_derivative_r2",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\noutput: {output_path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
