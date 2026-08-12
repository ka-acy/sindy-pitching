#!/usr/bin/env python3
"""Increment-parameterized forecasting: skill over persistence, horizon crossover, noise floor.

Three diagnostics, run together because they share data loading.

PART A -- parameterization fix and skill over persistence.
    Fitting `x(t+1) = W z(t)` with a ridge penalty is badly posed here: the true map is
    near-identity, but ridge shrinks W toward 0, i.e. toward the standardized mean, which
    is *away* from persistence. Predicting the increment instead,
        delta(t) = x(t+1) - x(t) = W z(t),
    makes ridge shrink toward delta = 0, which IS persistence. Persistence becomes the
    model's floor instead of something it must rediscover and pay a penalty for.
    Both parameterizations are fit on identical features so the effect is isolated.

    Headline metric is skill over persistence:
        skill = 1 - SSE(model) / SSE(zero-increment)
    Positive skill means genuine predictive structure beyond "nothing changes". R2 of the
    state is also reported, since it is what earlier runs reported and it is saturated by
    the near-identity component (persistence alone scores ~0.85 there).

PART B -- horizon crossover.
    Persistence wins at 40 frames because 40 frames is ~0.67 s and the joints have barely
    moved. Over a full pitch they move a great deal, so persistence must degrade eventually.
    This sweeps horizons and reports the horizon (if any) at which a rolled-out learned model
    overtakes persistence. Clip counts per horizon are reported, since only longer clips
    qualify at long horizons.

PART C -- noise floor.
    Bounds what any model could achieve. Two assumption-light estimates per feature column:
      - high-frequency residual scale from second differences, sigma^2 ~ mean(d2x^2)/6,
        compared against the per-frame signal scale std(delta x);
      - lag-1 autocorrelation of the increment sequence. If increments are white, they are
        unpredictable by construction and positive skill is impossible regardless of model.
    Caveat: the features are already Savitzky-Golay smoothed, so the residual estimate
    measures leftover high-frequency content in the features actually modeled, not raw
    MediaPipe jitter. It is a floor on the floor.
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

from src.preprocessing.evaluate_baseball_sindy import CORE_POSITIONS

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/preprocessing.baseball.unaligned.handedness.yaml"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--task-type", default="pitching")
    parser.add_argument("--depth", type=int, default=3, help="Delay depth (frames of history).")
    parser.add_argument(
        "--horizons", nargs="*", type=int,
        default=(5, 10, 20, 40, 60, 80, 120, 160, 240),
        help="Rollout horizons for Part B.",
    )
    parser.add_argument("--output-dir", default="results/analysis/increment_skill")
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
                "feature_names": obj["feature_names"].tolist(),
                "dt": float(row["dt"]),
            }
        )
    return clips


def core_indices(feature_names: list[str]) -> list[int]:
    return [feature_names.index(n) for n in CORE_POSITIONS if n in feature_names]


def build_design(X: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z, next_state, increment) aligned rows.

    Z(t) = [x(t), x(t)-x(t-1), ..., x(t-depth+2)-x(t-depth+1)] -- level plus history of
    increments, i.e. position and successive velocity estimates.
    """
    n = X.shape[0]
    if n < depth + 1:
        empty_z = np.zeros((0, X.shape[1] * depth))
        return empty_z, np.zeros((0, X.shape[1])), np.zeros((0, X.shape[1]))
    rows, nxt, inc = [], [], []
    for t in range(depth - 1, n - 1):
        feats = [X[t]] + [X[t - k] - X[t - k - 1] for k in range(depth - 1)]
        rows.append(np.concatenate(feats))
        nxt.append(X[t + 1])
        inc.append(X[t + 1] - X[t])
    return np.asarray(rows), np.asarray(nxt), np.asarray(inc)


def skill_score(true_delta: np.ndarray, pred_delta: np.ndarray, idx: list[int]) -> float:
    """1 - SSE(model)/SSE(zero-increment) on the core columns. Positive = beats persistence."""
    t, p = true_delta[:, idx], pred_delta[:, idx]
    sse_model = float(np.sum((t - p) ** 2))
    sse_zero = float(np.sum(t**2))
    return 1.0 - sse_model / sse_zero if sse_zero > 0 else float("nan")


def skill_uniform(true_delta: np.ndarray, pred_delta: np.ndarray, idx: list[int]) -> float:
    """Uniform per-column skill: mean_j (1 - SSE_j(model)/SSE_j(zero)).

    The pooled skill score is dominated by the largest-amplitude columns. Core positions
    span a ~13x range in per-frame signal std (lead_knee_lift 0.012 vs dominant_elbow 0.158),
    so pooled and uniform weighting can disagree in sign. Uniform matches how
    `core_position_metrics` aggregates, so it is the comparable number.
    """
    t, p = true_delta[:, idx], pred_delta[:, idx]
    num = np.sum((t - p) ** 2, axis=0)
    den = np.sum(t**2, axis=0)
    keep = den > 0
    return float(np.mean(1.0 - num[keep] / den[keep])) if keep.any() else float("nan")


def r2_uniform(true: np.ndarray, pred: np.ndarray, idx: list[int]) -> float:
    """Uniform-average R2 over core columns, measured against each column's own mean."""
    t, p = true[:, idx], pred[:, idx]
    num = np.sum((t - p) ** 2, axis=0)
    den = np.sum((t - t.mean(axis=0)) ** 2, axis=0)
    keep = den > 0
    return float(np.mean(1.0 - num[keep] / den[keep])) if keep.any() else float("nan")


def part_a(train, test, depth, idx) -> list[dict[str, Any]]:
    Z_tr, next_tr, inc_tr = map(np.vstack, zip(*[build_design(c["X"], depth) for c in train]))
    Z_te, next_te, inc_te = map(np.vstack, zip(*[build_design(c["X"], depth) for c in test]))
    scaler = StandardScaler().fit(Z_tr)
    Ztr_s, Zte_s = scaler.transform(Z_tr), scaler.transform(Z_te)
    x_te = next_te - inc_te  # x(t) recovered from the aligned pair

    rows = []

    # Persistence: zero increment.
    rows.append({
        "model": "persistence (no model)",
        "target_parameterization": "-",
        "skill_over_persistence": 0.0,
        "skill_uniform": 0.0,
        "r2_increment_vs_mean": r2_uniform(inc_te, np.zeros_like(inc_te), idx),
        "r2_state": r2_uniform(next_te, x_te, idx),
    })

    # Buggy parameterization: predict the next state directly.
    m_level = RidgeCV(alphas=ALPHAS).fit(Ztr_s, next_tr)
    pred_next = np.asarray(m_level.predict(Zte_s), dtype=np.float64)
    rows.append({
        "model": "delay ridge",
        "target_parameterization": "next state x(t+1)",
        "skill_over_persistence": skill_score(inc_te, pred_next - x_te, idx),
        "skill_uniform": skill_uniform(inc_te, pred_next - x_te, idx),
        "r2_increment_vs_mean": r2_uniform(inc_te, pred_next - x_te, idx),
        "r2_state": r2_uniform(next_te, pred_next, idx),
    })

    # Fixed parameterization: predict the increment.
    m_inc = RidgeCV(alphas=ALPHAS).fit(Ztr_s, inc_tr)
    pred_inc = np.asarray(m_inc.predict(Zte_s), dtype=np.float64)
    rows.append({
        "model": "delay ridge",
        "target_parameterization": "increment x(t+1)-x(t)",
        "skill_over_persistence": skill_score(inc_te, pred_inc, idx),
        "skill_uniform": skill_uniform(inc_te, pred_inc, idx),
        "r2_increment_vs_mean": r2_uniform(inc_te, pred_inc, idx),
        "r2_state": r2_uniform(next_te, x_te + pred_inc, idx),
    })
    return rows, (scaler, m_inc)


def rollout_increment(scaler, model, X: np.ndarray, depth: int, upto: int) -> np.ndarray:
    """Autoregressive rollout of the increment model, seeded with `depth` observed frames."""
    out = np.zeros((upto, X.shape[1]))
    out[:depth] = X[:depth]
    for t in range(depth - 1, upto - 1):
        feats = [out[t]] + [out[t - k] - out[t - k - 1] for k in range(depth - 1)]
        z = scaler.transform(np.concatenate(feats)[None, :])
        out[t + 1] = out[t] + np.asarray(model.predict(z), dtype=np.float64)[0]
    return out


def part_b(test, scaler, model, depth, horizons) -> list[dict[str, Any]]:
    """Rolled-out increment model vs persistence at each horizon.

    core_r2_* are per-clip uniform-average R2 over core columns, averaged across clips --
    the SAME aggregation `core_position_metrics` uses, so these numbers are comparable to
    every other core R2 reported in this project. skill is pooled SSE-based, which is the
    self-consistent quantity for a persistence comparison.
    """
    rows = []
    for H in horizons:
        model_sse, pers_sse, n_used = 0.0, 0.0, 0
        r2_model, r2_pers, skill_u = [], [], []
        for c in test:
            X, idx = c["X"], core_indices(c["feature_names"])
            upto = depth + H
            if X.shape[0] < upto:
                continue
            n_used += 1
            sim = rollout_increment(scaler, model, X, depth, upto)
            true_w = X[depth:upto][:, idx]
            sim_w = sim[depth:upto][:, idx]
            anchor = np.repeat(X[depth - 1][None, :], upto - depth, axis=0)[:, idx]
            model_sse += float(np.sum((true_w - sim_w) ** 2))
            pers_sse += float(np.sum((true_w - anchor) ** 2))
            den_p = np.sum((true_w - anchor) ** 2, axis=0)
            keep_p = den_p > 0
            if keep_p.any():
                skill_u.append(float(np.mean(1.0 - np.sum((true_w - sim_w) ** 2, axis=0)[keep_p] / den_p[keep_p])))
            den = np.sum((true_w - true_w.mean(axis=0)) ** 2, axis=0)
            keep = den > 0
            if keep.any():
                r2_model.append(float(np.mean(1.0 - np.sum((true_w - sim_w) ** 2, axis=0)[keep] / den[keep])))
                r2_pers.append(float(np.mean(1.0 - np.sum((true_w - anchor) ** 2, axis=0)[keep] / den[keep])))
        if n_used == 0:
            continue
        rows.append({
            "horizon_frames": H,
            "horizon_seconds": round(H * test[0]["dt"], 3),
            "num_clips": n_used,
            "skill_over_persistence": 1.0 - model_sse / pers_sse if pers_sse > 0 else float("nan"),
            "skill_uniform": float(np.mean(skill_u)) if skill_u else float("nan"),
            "core_r2_model": float(np.mean(r2_model)) if r2_model else float("nan"),
            "core_r2_persistence": float(np.mean(r2_pers)) if r2_pers else float("nan"),
        })
    return rows


def part_c(clips) -> list[dict[str, Any]]:
    names = clips[0]["feature_names"]
    rows = []
    for j, name in enumerate(names):
        sig, noise, acf = [], [], []
        for c in clips:
            x = c["X"][:, j]
            if x.size < 4:
                continue
            d1 = np.diff(x)
            d2 = np.diff(x, n=2)
            sig.append(np.std(d1))
            noise.append(np.sqrt(max(np.mean(d2**2) / 6.0, 0.0)))
            if d1.size > 2 and np.std(d1) > 0:
                a, b = d1[:-1], d1[1:]
                acf.append(float(np.corrcoef(a, b)[0, 1]))
        s, nz = float(np.mean(sig)), float(np.mean(noise))
        rows.append({
            "feature": name,
            "is_core_position": name in CORE_POSITIONS,
            "signal_std_per_frame": s,
            "hf_residual_scale": nz,
            "signal_to_noise": s / nz if nz > 0 else float("inf"),
            "increment_lag1_autocorr": float(np.mean(acf)) if acf else float("nan"),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    train = load_clips(config, args.train_split, args.task_type)
    test = load_clips(config, args.eval_split, args.task_type)
    idx = core_indices(train[0]["feature_names"])
    out_dir = PROJECT_ROOT / args.output_dir
    print(f"train {len(train)} clips / test {len(test)} clips | depth={args.depth} | core cols={len(idx)}\n")

    print("=== PART A: parameterization and skill over persistence (one-step, held-out) ===")
    rows_a, (scaler, inc_model) = part_a(train, test, args.depth, idx)
    print(f"{'model':14s} {'target':26s} {'skill(pool)':>12s} {'skill(unif)':>12s} {'R2(state)':>10s}")
    for r in rows_a:
        print(f"{r['model']:14s} {r['target_parameterization']:26s} "
              f"{r['skill_over_persistence']:>12.4f} {r['skill_uniform']:>12.4f} {r['r2_state']:>10.4f}")
    write_csv(out_dir / "part_a_parameterization.csv", rows_a)

    print("\n=== PART B: horizon crossover (rolled out, increment model vs persistence) ===")
    rows_b = part_b(test, scaler, inc_model, args.depth, list(args.horizons))
    print(f"{'H(frames)':>10s} {'H(sec)':>8s} {'clips':>6s} {'skill(pool)':>12s} {'skill(unif)':>12s} {'R2 model':>11s} {'R2 persist':>11s}")
    for r in rows_b:
        print(f"{r['horizon_frames']:>10d} {r['horizon_seconds']:>8.3f} {r['num_clips']:>6d} "
              f"{r['skill_over_persistence']:>12.4f} {r['skill_uniform']:>12.4f} "
              f"{r['core_r2_model']:>11.4f} {r['core_r2_persistence']:>11.4f}")
    write_csv(out_dir / "part_b_horizon_crossover.csv", rows_b)
    wins = [r for r in rows_b if r["skill_uniform"] > 0]
    print(f"\n  crossover: {'first beats persistence at H=' + str(wins[0]['horizon_frames']) if wins else 'NONE - persistence wins at every horizon tested'}")

    print("\n=== PART C: noise floor and increment predictability ===")
    rows_c = part_c(train + test)
    print(f"{'feature':34s} {'core':>5s} {'sig/frame':>10s} {'hf resid':>9s} {'SNR':>7s} {'ACF1(delta)':>12s}")
    for r in rows_c:
        print(f"{r['feature']:34s} {str(r['is_core_position']):>5s} {r['signal_std_per_frame']:>10.4f} "
              f"{r['hf_residual_scale']:>9.4f} {r['signal_to_noise']:>7.2f} {r['increment_lag1_autocorr']:>12.4f}")
    write_csv(out_dir / "part_c_noise_floor.csv", rows_c)

    print(f"\noutput dir: {out_dir.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
