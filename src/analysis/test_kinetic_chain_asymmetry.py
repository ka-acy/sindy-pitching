#!/usr/bin/env python3
"""Does proximal-to-distal coupling beat distal-to-proximal, beyond chance?

Section 6.9 establishes that cross-joint terms carry information, but that on its own is not a
biomechanical finding: the hips and shoulders are mechanically linked through the trunk, so any
coordinated movement produces coupling. The proximal-to-distal kinetic chain is a claim about
DIRECTION and precedence -- energy is transferred from the ground up, so proximal segments should
lead distal ones. The suggestive observation is that the fitted hip-to-shoulder coefficient
(+0.185) is roughly twice shoulder-to-hip (+0.092). This tests whether that direction preference
is real.

Method. Order the state variables proximally to distally. For each target equation, split the
available cross-joint columns into those from MORE proximal variables and those from MORE distal
ones, and fit three models:

    self only                  each equation sees only its own variable's history
    self + proximal-to-distal  plus columns from variables closer to the ground
    self + distal-to-proximal  plus columns from variables further from the ground

The statistic is the asymmetry:

    asym = [skill(self + P2D) - skill(self)] - [skill(self + D2P) - skill(self)]

Positive means proximal history predicts distal motion better than the reverse.

The two directions offer different numbers of columns to different targets, which would bias a
raw comparison. That is exactly why the null is needed: it circularly rolls the cross columns
within each clip, per target, preserving column counts and each column's autocorrelation while
destroying temporal alignment. Under it the asymmetry should centre on zero regardless of the
count imbalance.

Uses the stabilized 27-column design (derived hip_shoulder_separation_angle removed), since on
the original design rank-deficient columns produce cancelling coefficients of order 1e3 that
swamp effects of this size.

Usage:
    python src/analysis/test_kinetic_chain_asymmetry.py [--n-null 60] [--null-splits 5]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.evaluate_baseball_sindy import CORE_POSITIONS

# Proximal-to-distal ordering for a pitching delivery: energy runs from the ground up through
# the stride and legs, into pelvis rotation, then trunk/shoulder rotation, then the arm. Equal
# ranks are segments at the same level of the chain; cross terms between them belong to neither
# direction and are excluded from both, so the comparison is strictly directional.
CHAIN_RANK = {
    "stride_ankle_separation": 1,
    "lead_knee_lift": 2,
    "trail_knee_flexion_angle": 3,
    "dominant_knee_flexion_angle": 3,
    "hip_rotation_angle": 4,
    "shoulder_rotation_angle": 5,
    "trail_elbow_flexion_angle": 6,
    "dominant_elbow_flexion_angle": 6,
    "dominant_wrist_speed": 7,
    "hip_shoulder_separation_angle": 5,   # dropped from the design, listed for completeness
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-subdir", default="features")
    p.add_argument("--task-type", default="pitching")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--n-null", type=int, default=60)
    p.add_argument("--null-splits", type=int, default=5)
    p.add_argument("--n-splits", type=int, default=30)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--drop-derived", default="hip_shoulder_separation_angle")
    p.add_argument("--output-dir", default="results/analysis/kinetic_chain_asymmetry")
    return p.parse_args()


def load_all(feature_subdir: str, task_type: str) -> list[dict]:
    clips = []
    for split in ("train", "test"):
        root = PROJECT_ROOT / "data/baseball/processed" / split / feature_subdir
        with (root / "manifest.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["task_type"] != task_type:
                    continue
                obj = np.load(PROJECT_ROOT / r["filepath"])
                clips.append({"X": np.asarray(obj["X"], dtype=np.float64),
                              "names": obj["feature_names"].tolist()})
    return clips


def block(X, depth, target_idx, drop_idx):
    start = depth - 1
    n_feat = X.shape[1] - (1 if drop_idx is not None else 0)
    if X.shape[0] < start + 2:
        return np.zeros((0, n_feat * depth)), np.zeros((0, len(target_idx)))
    D, Y = [], []
    for t in range(start, X.shape[0] - 1):
        feats = [X[t]] + [X[t - k] - X[t - k - 1] for k in range(depth - 1)]
        if drop_idx is not None:
            feats = [np.delete(f, drop_idx) for f in feats]
        D.append(np.concatenate(feats))
        Y.append(X[t + 1, target_idx] - X[t, target_idx])
    return np.asarray(D), np.asarray(Y)


def base_of(label: str) -> str:
    return label[label.index("(") + 1:label.rindex(")")] if label.startswith("d") and "(" in label else label


def build_masks(labels, target_names):
    """Per target: (self columns, proximal-to-distal columns, distal-to-proximal columns)."""
    n_t, n_c = len(target_names), len(labels)
    self_m = np.zeros((n_t, n_c), bool)
    p2d = np.zeros((n_t, n_c), bool)
    d2p = np.zeros((n_t, n_c), bool)
    for i, tname in enumerate(target_names):
        tr = CHAIN_RANK[tname]
        for j, lab in enumerate(labels):
            b = base_of(lab)
            if b == tname:
                self_m[i, j] = True
                continue
            br = CHAIN_RANK.get(b)
            if br is None or br == tr:
                continue          # same level of the chain: neither direction
            if br < tr:
                p2d[i, j] = True  # source is closer to the ground than the target
            else:
                d2p[i, j] = True
    return self_m, p2d, d2p


def fit_one(Dtr, ytr, Dte, cols, threshold):
    import pysindy as ps
    Xtr, Xte = Dtr[:, cols], Dte[:, cols]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    widest = sd.max() if sd.size else 1.0
    keep = sd > max(widest * 1e-6, 1e-12)
    if not keep.any():
        return np.zeros(Xte.shape[0])
    Xtr, Xte = (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    if ysd < 1e-12:
        return np.full(Xte.shape[0], ymu)
    opt = ps.STLSQ(threshold=threshold, alpha=0.001, max_iter=20)
    opt.fit(Xtr, (ytr - ymu) / ysd)
    return (Xte @ np.asarray(opt.coef_).ravel()) * ysd + ymu


def score(Dtr_per_t, Ytr, Dte_per_t, Yte, col_sets, threshold):
    out = []
    for j in range(Ytr.shape[1]):
        cols = col_sets[j]
        if cols.size == 0:
            continue
        pred = fit_one(Dtr_per_t[j], Ytr[:, j], Dte_per_t[j], cols, threshold)
        den = float(np.sum(Yte[:, j] ** 2))
        if den > 0:
            out.append(1.0 - float(np.sum((Yte[:, j] - pred) ** 2)) / den)
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    clips = load_all(args.feature_subdir, args.task_type)
    names = clips[0]["names"]
    target_idx = [names.index(n) for n in CORE_POSITIONS if n in names]
    target_names = [names[i] for i in target_idx]
    drop_name = args.drop_derived.strip() or None
    drop_idx = names.index(drop_name) if drop_name in names else None
    kept = [n for n in names if n != drop_name] if drop_name else list(names)
    labels = list(kept) + [f"d{k+1}({m})" for k in range(args.depth - 1) for m in kept]

    self_m, p2d, d2p = build_masks(labels, target_names)
    blocks = [block(c["X"], args.depth, target_idx, drop_idx) for c in clips]
    blocks = [b for b in blocks if b[0].shape[0] > 0]
    n, n_t = len(blocks), len(target_names)
    n_te = max(int(round(n * args.test_fraction)), 2)

    print(f"clips {n}   design {len(labels)} cols   targets {n_t}")
    print("proximal-to-distal ordering: " + " < ".join(
        k for k, _ in sorted(((k, v) for k, v in CHAIN_RANK.items() if k in kept), key=lambda x: x[1])))
    print(f"columns per target  self {int(self_m[0].sum())}  "
          f"P2D {[int(x) for x in p2d.sum(axis=1)]}  D2P {[int(x) for x in d2p.sum(axis=1)]}\n")

    cols_self = [np.where(self_m[j])[0] for j in range(n_t)]
    cols_p2d = [np.where(self_m[j] | p2d[j])[0] for j in range(n_t)]
    cols_d2p = [np.where(self_m[j] | d2p[j])[0] for j in range(n_t)]

    def stack(idxs):
        return np.vstack([blocks[i][0] for i in idxs]), np.vstack([blocks[i][1] for i in idxs])

    def stack_shifted(idxs, cols_to_roll):
        out = []
        for i in idxs:
            D = blocks[i][0].copy()
            if D.shape[0] > 1 and cols_to_roll.size:
                k = int(rng.integers(1, D.shape[0]))
                D[:, cols_to_roll] = np.roll(D[:, cols_to_roll], k, axis=0)
            out.append(D)
        return np.vstack(out)

    # --- observed asymmetry across resplits ---
    rows_split = []
    for _ in range(args.n_splits):
        perm = rng.permutation(n)
        te, tr = perm[:n_te], perm[n_te:]
        Dtr, Ytr = stack(tr)
        Dte, Yte = stack(te)
        A, B = [Dtr] * n_t, [Dte] * n_t
        s0 = score(A, Ytr, B, Yte, cols_self, args.threshold)
        sp = score(A, Ytr, B, Yte, cols_p2d, args.threshold)
        sd_ = score(A, Ytr, B, Yte, cols_d2p, args.threshold)
        if max(abs(s0), abs(sp), abs(sd_)) > 10:
            continue
        rows_split.append({"self": s0, "p2d": sp, "d2p": sd_,
                           "gain_p2d": sp - s0, "gain_d2p": sd_ - s0,
                           "asymmetry": (sp - s0) - (sd_ - s0)})
    a = np.asarray([r["asymmetry"] for r in rows_split])
    se = a.std(ddof=1) / np.sqrt(a.size)
    ci = (float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se))
    print(f"--- {len(rows_split)} resplits ---")
    print(f"  self only            {np.mean([r['self'] for r in rows_split]):+.4f}")
    print(f"  self + P2D           {np.mean([r['p2d'] for r in rows_split]):+.4f}   "
          f"gain {np.mean([r['gain_p2d'] for r in rows_split]):+.4f}")
    print(f"  self + D2P           {np.mean([r['d2p'] for r in rows_split]):+.4f}   "
          f"gain {np.mean([r['gain_d2p'] for r in rows_split]):+.4f}")
    print(f"  asymmetry (P2D-D2P)  {a.mean():+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"positive in {int((a > 0).sum())}/{a.size}")

    # --- shift null on the asymmetry ---
    print(f"\n--- per-target circular-shift null ({args.n_null} draws x {args.null_splits} splits) ---")
    per_split = []
    for si in range(args.null_splits):
        perm = rng.permutation(n)
        te, tr = perm[:n_te], perm[n_te:]
        Dtr, Ytr = stack(tr)
        Dte, Yte = stack(te)
        A, B = [Dtr] * n_t, [Dte] * n_t
        s0 = score(A, Ytr, B, Yte, cols_self, args.threshold)
        obs = (score(A, Ytr, B, Yte, cols_p2d, args.threshold) - s0) \
            - (score(A, Ytr, B, Yte, cols_d2p, args.threshold) - s0)
        draws = []
        for _ in range(args.n_null):
            Dtr_p = [stack_shifted(tr, np.where(p2d[j])[0]) for j in range(n_t)]
            Dte_p = [stack_shifted(te, np.where(p2d[j])[0]) for j in range(n_t)]
            Dtr_d = [stack_shifted(tr, np.where(d2p[j])[0]) for j in range(n_t)]
            Dte_d = [stack_shifted(te, np.where(d2p[j])[0]) for j in range(n_t)]
            gp = score(Dtr_p, Ytr, Dte_p, Yte, cols_p2d, args.threshold) - s0
            gd = score(Dtr_d, Ytr, Dte_d, Yte, cols_d2p, args.threshold) - s0
            if abs(gp) < 10 and abs(gd) < 10:
                draws.append(gp - gd)
        d = np.asarray(draws)
        pv = (int((d >= obs).sum()) + 1) / (d.size + 1) if d.size else float("nan")
        per_split.append({"split": si + 1, "observed_asymmetry": obs,
                          "null_mean": float(d.mean()) if d.size else None,
                          "null_sd": float(d.std(ddof=1)) if d.size > 1 else None,
                          "n_draws": int(d.size), "p": pv})
        print(f"  split {si+1}: observed {obs:+.4f}  null mean {d.mean():+.4f} "
              f"sd {d.std(ddof=1):.4f}  p = {pv:.4f}")

    ps = [r["p"] for r in per_split if r["p"] == r["p"]]
    n_sig = sum(1 for v in ps if v < 0.05)
    print(f"\n  splits with p < 0.05: {n_sig}/{len(ps)}   median p = {np.median(ps):.4f}")

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "chain_rank": {k: v for k, v in CHAIN_RANK.items() if k in kept},
        "n_resplits": len(rows_split),
        "self_mean": float(np.mean([r["self"] for r in rows_split])),
        "p2d_mean": float(np.mean([r["p2d"] for r in rows_split])),
        "d2p_mean": float(np.mean([r["d2p"] for r in rows_split])),
        "gain_p2d_mean": float(np.mean([r["gain_p2d"] for r in rows_split])),
        "gain_d2p_mean": float(np.mean([r["gain_d2p"] for r in rows_split])),
        "asymmetry_mean": float(a.mean()), "asymmetry_ci95": ci,
        "asymmetry_positive_fraction": float((a > 0).mean()),
        "per_split_null": per_split, "n_splits_significant": n_sig,
        "n_splits_tested": len(ps),
    }, indent=2))

    print("\n--- verdict ---")
    if n_sig == len(ps) and len(ps) > 0 and ci[0] > 0:
        print("  Proximal-to-distal coupling beats the reverse on every tested split and the")
        print("  resplit CI excludes zero. This is directional evidence consistent with the")
        print("  kinetic chain. Report the effect size alongside it.")
    elif ci[0] > 0 and n_sig > 0:
        print(f"  Asymmetry is positive across resplits and clears the null on {n_sig}/{len(ps)}")
        print("  splits. Suggestive but not uniform; state it as such, not as established.")
    else:
        print("  No reliable directional preference. The coupling is symmetric as far as this")
        print("  test can tell, and the kinetic chain is NOT evidenced by these coefficients.")
    print(f"\noutput: {(out_dir / 'summary.json').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
