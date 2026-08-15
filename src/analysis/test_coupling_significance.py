#!/usr/bin/env python3
"""Does the cross-joint coupling in the delay-SINDy model survive a null and a resplit?

Section 6.9 finds that allowing each equation to use terms from other joints adds ~0.017
uniform skill over restricting it to that joint's own history, and that the discovered
hip-to-shoulder coefficient is about twice the shoulder-to-hip coefficient -- the direction the
proximal-to-distal kinetic chain predicts. That is the only potentially biomechanical result in
this project, it rests on a single train/test split, and it has no null. Four earlier results
here were false positives that survived only because the matching control was missing, so this
one is tested before it is believed.

Statistic:  gain = skill_uniform(all terms) - skill_uniform(self terms only)

Test 1, circular-shift null. Per target equation, every column that is NOT part of that
target's own variable is circularly rolled within each clip by a random offset, then the
unrestricted model is refitted and the gain recomputed. The shift must be per target: the
coupling of interest is one joint's history appearing in another joint's equation, and any such
column is some other target's self column, so a single shared shift mask leaves nearly all of
them untouched. A circular roll preserves each column's marginal distribution and
autocorrelation while destroying its temporal alignment with the target -- a plain row shuffle
would also destroy the smoothness that makes these columns predictive at all, giving an
unrealistically easy null.

Test 2, repeated clip-level resplits, for a confidence interval rather than a point estimate.

Usage:
    python src/analysis/test_coupling_significance.py [--n-null 200] [--n-splits 20]
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

# Columns whose training spread is this small relative to the widest column are dropped:
# standardizing them amplifies noise and produced predictions of order 1e6 on some splits.
MIN_REL_STD = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-subdir", default="features")
    p.add_argument("--task-type", default="pitching")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--n-null", type=int, default=200,
                   help="Null draws PER split.")
    p.add_argument("--null-splits", type=int, default=5,
                   help="Number of independent held-out splits to run the null on. "
                        "The gain varies across splits, so testing one split cannot "
                        "separate a real effect from split luck.")
    p.add_argument("--n-splits", type=int, default=20)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--drop-derived", default="hip_shoulder_separation_angle",
                   help="Feature to remove from the DESIGN (not the targets). "
                        "hip_shoulder_separation_angle is shoulder_rotation minus hip_rotation, "
                        "and the identity holds near-exactly on some clip subsets, making those "
                        "training draws rank-deficient. Removing it took divergence from 2/40 "
                        "splits to 0/40 at a cost of 0.0009 skill. Pass an empty string to "
                        "reproduce the unstabilized design.")
    p.add_argument("--output-dir", default="results/analysis/coupling_significance")
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


def block(X: np.ndarray, depth: int, target_idx: list[int],
          drop_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip design and target. `drop_idx` removes one feature from the DESIGN only;
    targets are read from X directly and are unaffected."""
    n_feat = X.shape[1] - (1 if drop_idx is not None else 0)
    start = depth - 1
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


def labels_for(names: list[str], depth: int, drop: str | None = None) -> list[str]:
    kept = [n for n in names if n != drop] if drop else list(names)
    out = list(kept)
    for k in range(depth - 1):
        out += [f"d{k+1}({nm})" for nm in kept]
    return out


def self_mask(labels: list[str], target_names: list[str]) -> np.ndarray:
    mask = np.zeros((len(target_names), len(labels)), dtype=bool)
    for i, tname in enumerate(target_names):
        for j, lab in enumerate(labels):
            base = lab[lab.index("(") + 1:lab.rindex(")")] if lab.startswith("d") and "(" in lab else lab
            if base == tname:
                mask[i, j] = True
    return mask


def col_skill(true_col: np.ndarray, pred_col: np.ndarray) -> float | None:
    den = float(np.sum(true_col**2))
    if den <= 0:
        return None
    return 1.0 - float(np.sum((true_col - pred_col) ** 2)) / den


def fit_one_target(Dtr, ytr, Dte, cols, threshold):
    """STLSQ for a single target over `cols`, standardized on train. Returns test prediction."""
    import pysindy as ps

    Xtr, Xte = Dtr[:, cols], Dte[:, cols]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    widest = sd.max() if sd.size else 1.0
    keep = sd > max(widest * MIN_REL_STD, 1e-12)
    if not keep.any():
        return np.zeros(Xte.shape[0])
    Xtr, Xte = (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]

    ymu, ysd = float(ytr.mean()), float(ytr.std())
    if ysd < 1e-12:
        return np.full(Xte.shape[0], ymu)
    opt = ps.STLSQ(threshold=threshold, alpha=0.001, max_iter=20)
    opt.fit(Xtr, (ytr - ymu) / ysd)
    return (Xte @ np.asarray(opt.coef_).ravel()) * ysd + ymu


def score_model(Dtr_per_target, Ytr, Dte_per_target, Yte, mask, threshold, restrict: bool):
    """Uniform skill over targets. `Dtr_per_target[j]` may differ per target (null draws)."""
    scores = []
    for j in range(Ytr.shape[1]):
        cols = np.where(mask[j])[0] if restrict else np.arange(Dtr_per_target[j].shape[1])
        pred = fit_one_target(Dtr_per_target[j], Ytr[:, j], Dte_per_target[j], cols, threshold)
        s = col_skill(Yte[:, j], pred)
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    clips = load_all(args.feature_subdir, args.task_type)
    names = clips[0]["names"]
    target_idx = [names.index(n) for n in CORE_POSITIONS if n in names]
    target_names = [names[i] for i in target_idx]
    drop_name = args.drop_derived.strip() or None
    drop_idx = names.index(drop_name) if drop_name in names else None
    labels = labels_for(names, args.depth, drop_name)
    mask = self_mask(labels, target_names)
    print(f"design: {'stabilized, dropped ' + drop_name if drop_idx is not None else 'baseline (unstabilized)'}")

    blocks = [block(c["X"], args.depth, target_idx, drop_idx) for c in clips]
    blocks = [b for b in blocks if b[0].shape[0] > 0]
    n = len(blocks)
    n_te = max(int(round(n * args.test_fraction)), 2)
    print(f"clips {n}   design dims {len(labels)}   targets {len(target_names)}")
    print(f"per-target non-self columns shifted under the null: {int((~mask[0]).sum())} of {len(labels)}")

    def stack(idxs):
        return np.vstack([blocks[i][0] for i in idxs]), np.vstack([blocks[i][1] for i in idxs])

    def stack_shifted(idxs, j):
        """Design for target j with its non-self columns rolled within each clip."""
        cols = np.where(~mask[j])[0]
        out = []
        for i in idxs:
            D = blocks[i][0].copy()
            if D.shape[0] > 1:
                k = int(rng.integers(1, D.shape[0]))
                D[:, cols] = np.roll(D[:, cols], k, axis=0)
            out.append(D)
        return np.vstack(out)

    n_t = len(target_names)

    # --- Test 2: repeated clip-level resplits ---
    gains, self_s, all_s = [], [], []
    for _ in range(args.n_splits):
        perm = rng.permutation(n)
        te, tr = perm[:n_te], perm[n_te:]
        Dtr, Ytr = stack(tr)
        Dte, Yte = stack(te)
        same_tr = [Dtr] * n_t
        same_te = [Dte] * n_t
        s_self = score_model(same_tr, Ytr, same_te, Yte, mask, args.threshold, True)
        s_all = score_model(same_tr, Ytr, same_te, Yte, mask, args.threshold, False)
        self_s.append(s_self)
        all_s.append(s_all)
        gains.append(s_all - s_self)
    # The unrestricted 30-column fit is ill-conditioned on some random splits: levels and
    # increments are collinear, and STLSQ's ridge term is small, so a split occasionally
    # yields predictions orders of magnitude off. Those splits are reported separately rather
    # than averaged in, where a single one would dominate the mean.
    g_all = np.asarray(gains)
    stable = np.abs(np.asarray(all_s)) < 10.0
    n_unstable = int((~stable).sum())
    g = g_all[stable]
    se = g.std(ddof=1) / np.sqrt(g.size) if g.size > 1 else float("nan")
    ci = (float(g.mean() - 1.96 * se), float(g.mean() + 1.96 * se)) if g.size > 1 else (float("nan"),) * 2
    print(f"\n--- repeated resplits ({args.n_splits} random 80/20 splits over clips) ---")
    print(f"  numerically unstable splits excluded: {n_unstable}/{args.n_splits} "
          f"(unrestricted fit diverged)")
    print(f"  self terms only   {np.mean(np.asarray(self_s)[stable]):+.4f}")
    print(f"  all terms         {np.mean(np.asarray(all_s)[stable]):+.4f}")
    print(f"  coupling gain     mean {g.mean():+.4f}  median {np.median(g):+.4f}  "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"                    positive in {int((g > 0).sum())}/{g.size} stable splits")

    # --- Test 1: per-target circular-shift null, repeated over several splits ---
    # The gain varies across splits (it is negative on some), so a null on a single split
    # confounds the effect with split luck. Each split gets its own observed gain and its own
    # null distribution, and the per-split p-values are reported together.
    print(f"\n--- per-target circular-shift null ({args.n_null} draws x "
          f"{args.null_splits} splits) ---")
    per_split = []
    for si in range(args.null_splits):
        perm = rng.permutation(n)
        te, tr = perm[:n_te], perm[n_te:]
        Dtr, Ytr = stack(tr)
        Dte, Yte = stack(te)
        o_self = score_model([Dtr] * n_t, Ytr, [Dte] * n_t, Yte, mask, args.threshold, True)
        o_all = score_model([Dtr] * n_t, Ytr, [Dte] * n_t, Yte, mask, args.threshold, False)
        if abs(o_all) > 10.0:
            print(f"  split {si + 1}: unrestricted fit diverged, skipped")
            continue
        o_gain = o_all - o_self
        draws = []
        for _ in range(args.n_null):
            Dtr_j = [stack_shifted(tr, j) for j in range(n_t)]
            Dte_j = [stack_shifted(te, j) for j in range(n_t)]
            s_all = score_model(Dtr_j, Ytr, Dte_j, Yte, mask, args.threshold, False)
            if abs(s_all) < 10.0:
                draws.append(s_all - o_self)
        d = np.asarray(draws)
        pv = (int((d >= o_gain).sum()) + 1) / (d.size + 1) if d.size else float("nan")
        per_split.append({"split": si + 1, "observed_gain": o_gain,
                          "null_mean": float(d.mean()) if d.size else None,
                          "null_sd": float(d.std(ddof=1)) if d.size > 1 else None,
                          "n_draws": int(d.size), "p": pv})
        print(f"  split {si + 1}: observed {o_gain:+.4f}  null mean {d.mean():+.4f} "
              f"sd {d.std(ddof=1):.4f}  p = {pv:.4f}")

    ps = [r["p"] for r in per_split if r["p"] == r["p"]]
    n_sig = sum(1 for v in ps if v < 0.05)
    obs_gain = float(np.mean([r["observed_gain"] for r in per_split])) if per_split else float("nan")
    nl = np.asarray([])
    p = float(np.median(ps)) if ps else float("nan")
    print(f"\n  splits with p < 0.05: {n_sig}/{len(ps)}   median p = {p:.4f}")

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "depth": args.depth, "threshold": args.threshold,
        "n_splits": args.n_splits, "n_null": args.n_null,
        "resplit_self_mean": float(np.mean(np.asarray(self_s)[stable])),
        "resplit_all_mean": float(np.mean(np.asarray(all_s)[stable])),
        "resplit_gain_mean": float(g.mean()) if g.size else None, "resplit_gain_ci95": ci,
        "resplit_gain_positive_fraction": float((g > 0).mean()) if g.size else None,
        "mean_observed_gain_over_null_splits": obs_gain,
        "per_split_null": per_split,
        "n_splits_significant": n_sig,
        "n_splits_tested": len(ps),
        "median_p": p,
        "resplit_unstable_excluded": n_unstable,
        "resplit_gain_median": float(np.median(g)) if g.size else None,
    }, indent=2))


    print("\n--- verdict ---")
    if n_sig == len(ps) and len(ps) > 0 and ci[0] > 0:
        print("  Coupling clears BOTH the shift null and the resplit CI. The cross-joint terms")
        print("  carry information. Effect size is small; report it as such.")
    elif n_sig > 0:
        print(f"  Clears the null on only {n_sig}/{len(ps)} splits: too split-dependent to claim.")
    elif ci[0] > 0:
        print("  Positive across resplits but does NOT clear the shift null. The gain is")
        print("  explained by the extra columns' flexibility, not by their information.")
    else:
        print("  Clears neither test. The coupling gain is not distinguishable from chance, and")
        print("  the hip-to-shoulder asymmetry must NOT be presented as a finding.")
    print(f"\noutput: {(out_dir / 'summary.json').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
