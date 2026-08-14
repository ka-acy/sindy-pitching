#!/usr/bin/env python3
"""Why does the delay-SINDy fit diverge on some splits, and which change fixes it?

The 30-term delay model of Section 6.9 is the one real positive result in this project
(+0.3881 uniform skill over persistence). It is also fragile: on roughly one random
clip-level split in ten the unrestricted fit produces predictions of order 1e6, which made the
coupling analysis unusable until those splits were excluded by hand.

The design is `[x(t), delta(t-1), ..., delta(t-d+1)]`. Two properties of it are candidate
causes, and they imply different fixes:

  collinearity   levels and their own increments are strongly related, so the normal equations
                 are ill-conditioned and STLSQ's small ridge term (alpha=0.001) does not
                 regularize enough
  leverage       level columns carry large between-clip variance -- different pitchers stand in
                 different places -- so standardizing across the pooled training set leaves
                 clips whose levels sit far from the pooled mean with extreme values. A test
                 clip like that is then extrapolated far outside the training range

These are distinguishable. Per-clip centring of the level columns removes the leverage source
while keeping the levels; dropping the levels entirely removes both; a stronger ridge attacks
only the conditioning. Each variant is run over the same random splits so divergence rate and
skill are directly comparable.

Divergence is defined as held-out uniform skill outside [-10, 1], which cannot occur for any
sane predictor here given persistence scores 0 and the working model scores about +0.39.

Usage:
    python src/analysis/stabilize_delay_fit.py [--n-splits 40]
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

DIVERGENCE_LIMIT = 10.0
DROP_IDX = 2   # index of hip_shoulder_separation_angle in the 10-feature state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-subdir", default="features")
    p.add_argument("--task-type", default="pitching")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--n-splits", type=int, default=40)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-csv", default="results/analysis/delay_sindy/stability.csv")
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


def block(X: np.ndarray, depth: int, target_idx: list[int], variant: str):
    """Per-clip design and target. `variant` controls how the level columns are treated."""
    start = depth - 1
    if X.shape[0] < start + 2:
        return None
    levels_mu = X.mean(axis=0)
    drop = variant == "drop_redundant_rotation"
    D, Y = [], []
    for t in range(start, X.shape[0] - 1):
        incs = [X[t - k] - X[t - k - 1] for k in range(depth - 1)]
        if variant == "increments_only":
            feats = incs
        elif variant == "levels_clip_centered":
            feats = [X[t] - levels_mu] + incs
        else:
            feats = [X[t]] + incs
        if drop:
            # hip_shoulder_separation_angle is defined as shoulder_rotation - hip_rotation.
            # The identity holds near-exactly on some clip subsets, so a training draw can be
            # rank-deficient in those three columns and STLSQ answers with enormous cancelling
            # coefficients. Dropping the derived feature removes the dependency without losing
            # information: the two rotations that define it are still present, and it is not a
            # target column.
            feats = [np.delete(f, DROP_IDX) for f in feats]
        D.append(np.concatenate(feats))
        Y.append(X[t + 1, target_idx] - X[t, target_idx])
    return np.asarray(D), np.asarray(Y)


def skill_uniform(true: np.ndarray, pred: np.ndarray) -> float:
    num = np.sum((true - pred) ** 2, axis=0)
    den = np.sum(true**2, axis=0)
    keep = den > 0
    return float(np.mean(1.0 - num[keep] / den[keep])) if keep.any() else float("nan")


def fit_score(Dtr, Ytr, Dte, Yte, threshold, alpha, whiten):
    import pysindy as ps
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Dtr)
    Ztr, Zte = scaler.transform(Dtr), scaler.transform(Dte)
    if whiten:
        # Decorrelate the design so STLSQ never sees near-duplicate columns. Costs the
        # interpretability of individual terms, which is the trade-off being measured.
        pca = PCA(n_components=0.999, svd_solver="full", whiten=True).fit(Ztr)
        Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)

    ymu, ysd = Ytr.mean(axis=0), Ytr.std(axis=0)
    ysd[ysd < 1e-12] = 1.0
    Ytr_s = (Ytr - ymu) / ysd
    coefs = np.zeros((Ytr.shape[1], Ztr.shape[1]))
    for j in range(Ytr.shape[1]):
        opt = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=20)
        opt.fit(Ztr, Ytr_s[:, j])
        coefs[j] = np.asarray(opt.coef_).ravel()
    pred = (Zte @ coefs.T) * ysd + ymu
    return skill_uniform(Yte, pred), int(np.sum(np.abs(coefs) > 1e-10))


VARIANTS = [
    ("baseline (levels + increments, alpha=1e-3)", "levels", 0.001, False),
    ("stronger ridge alpha=1e-1", "levels", 0.1, False),
    ("stronger ridge alpha=1.0", "levels", 1.0, False),
    ("levels centred per clip", "levels_clip_centered", 0.001, False),
    ("increments only (levels dropped)", "increments_only", 0.001, False),
    ("PCA-whitened design", "levels", 0.001, True),
    ("ridge alpha=1e3", "levels", 1000.0, False),
    ("drop hip_shoulder_separation", "drop_redundant_rotation", 0.001, False),
]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    clips = load_all(args.feature_subdir, args.task_type)
    names = clips[0]["names"]
    target_idx = [names.index(n) for n in CORE_POSITIONS if n in names]

    # Same splits for every variant, so differences are attributable to the variant.
    n = len(clips)
    n_te = max(int(round(n * args.test_fraction)), 2)
    splits = [rng.permutation(n) for _ in range(args.n_splits)]

    designs = {}
    for _, variant, _, _ in VARIANTS:
        if variant in designs:
            continue
        blocks = [block(c["X"], args.depth, target_idx, variant) for c in clips]
        designs[variant] = [b for b in blocks if b is not None]

    print(f"clips {n}, splits {args.n_splits}, depth {args.depth}, threshold {args.threshold}")
    print(f"divergence = held-out uniform skill outside [-{DIVERGENCE_LIMIT:.0f}, 1]\n")
    print(f"{'variant':44s} {'dims':>5s} {'diverged':>9s} {'skill':>9s} {'95% CI':>19s} {'terms':>6s}")

    rows = []
    for label, variant, alpha, whiten in VARIANTS:
        blocks = designs[variant]
        scores, terms, diverged = [], [], 0
        for perm in splits:
            te, tr = perm[:n_te], perm[n_te:]
            Dtr = np.vstack([blocks[i][0] for i in tr])
            Ytr = np.vstack([blocks[i][1] for i in tr])
            Dte = np.vstack([blocks[i][0] for i in te])
            Yte = np.vstack([blocks[i][1] for i in te])
            s, nz = fit_score(Dtr, Ytr, Dte, Yte, args.threshold, alpha, whiten)
            if not np.isfinite(s) or s < -DIVERGENCE_LIMIT or s > 1.0:
                diverged += 1
                continue
            scores.append(s)
            terms.append(nz)
        a = np.asarray(scores)
        se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
        ci = (a.mean() - 1.96 * se, a.mean() + 1.96 * se) if a.size > 1 else (float("nan"),) * 2
        dims = blocks[0][0].shape[1]
        rows.append({"variant": label, "design_dims": dims,
                     "diverged": diverged, "n_splits": args.n_splits,
                     "skill_mean": float(a.mean()) if a.size else float("nan"),
                     "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                     "mean_terms": float(np.mean(terms)) if terms else float("nan")})
        print(f"{label:44s} {dims:>5d} {diverged:>4d}/{args.n_splits:<4d} "
              f"{a.mean():>+9.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {np.mean(terms):>6.1f}")

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    base = rows[0]
    clean = [r for r in rows if r["diverged"] == 0]
    print("\n--- verdict ---")
    print(f"  baseline diverges on {base['diverged']}/{args.n_splits} splits "
          f"at skill {base['skill_mean']:+.4f}")
    if clean:
        best = max(clean, key=lambda r: r["skill_mean"])
        print(f"  best variant with zero divergence: {best['variant']}")
        print(f"    skill {best['skill_mean']:+.4f} [{best['ci_low']:+.4f},{best['ci_high']:+.4f}], "
              f"{best['mean_terms']:.1f} terms, {best['design_dims']} dims")
        delta = best["skill_mean"] - base["skill_mean"]
        print(f"    skill change vs baseline: {delta:+.4f}")
    else:
        print("  no variant eliminated divergence; the instability is not explained by "
              "collinearity or leverage alone")
    print(f"\noutput: {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
