#!/usr/bin/env python3
"""Is the short-horizon skill lost by the sparse fit, or was it never learnable?

Section 6.8 of the technical paper found that the derivative targets carry 0.69-0.70 skill
over persistence while the fitted STLSQ equations retain 0% (first-order) to 12%
(second-order acceleration), and attributed the loss to the sparse fit and its selection
rule. That attribution is a hypothesis, and this script tests it properly.

The controlled comparison holds the design matrix and the target fixed and varies ONLY the
estimator:

    target      increment of the columns of interest, delta(t) = x(t+1) - x(t)
    design      the same state the SINDy model sees, [1, x]
    estimators  RidgeCV        -- dense, minimal regularization: the most any linear model
                                  of this state can extract
                STLSQ(lambda)  -- the pipeline's sparse estimator, swept over thresholds

If ridge extracts substantially more than STLSQ, the loss really is sparsification and the
fix is the estimator or its selection rule. If ridge extracts no more than STLSQ, then the
skill was never available from this state and the Section 6.8 attribution is wrong -- the
limit is the state's information content, not the fit.

Both state definitions are tested, because they differ in a way that matters:

    first-order   [8 core positions + 2 others]; predicting a position increment means
                  inferring velocity from position alone
    second-order  [8 positions + 8 rates]; predicting a rate increment means inferring
                  acceleration from position and velocity

Usage:
    python src/analysis/diagnose_sparse_fit_loss.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.evaluate_baseball_sindy import CORE_POSITIONS

RIDGE_ALPHAS = np.logspace(-6, 3, 40)
THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5)

STATES = {
    "first_order": {
        "subdir": "features",
        "target": "core_positions",
        "note": "predict position increment from position alone",
    },
    "second_order": {
        "subdir": "features_second_order",
        "target": "rates",
        "note": "predict rate increment (acceleration) from position and velocity",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--task-type", default="pitching")
    p.add_argument("--n-splits", type=int, default=20,
                   help="Random clip-level resplits. The operative train/test split is one "
                        "draw; the claim that sparsification costs nothing should not rest on "
                        "which 29 clips happened to be held out.")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-csv", default="results/analysis/sparse_fit_loss.csv")
    return p.parse_args()


def load(subdir: str, split: str, task_type: str) -> list[dict]:
    root = PROJECT_ROOT / "data/baseball/processed" / split / subdir
    with (root / "manifest.csv").open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["task_type"] == task_type]
    out = []
    for r in rows:
        obj = np.load(PROJECT_ROOT / r["filepath"])
        out.append({"X": np.asarray(obj["X"], dtype=np.float64),
                    "names": obj["feature_names"].tolist()})
    return out


def target_indices(names: list[str], which: str) -> list[int]:
    if which == "rates":
        return [i for i, n in enumerate(names) if n.endswith("_rate")]
    return [names.index(n) for n in CORE_POSITIONS if n in names]


def stack(clips: list[dict], idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Design = state at t; target = increment of the columns of interest over t -> t+1."""
    D, Y = [], []
    for c in clips:
        X = c["X"]
        if X.shape[0] < 2:
            continue
        D.append(X[:-1])
        Y.append(X[1:, idx] - X[:-1, idx])
    return np.vstack(D), np.vstack(Y)


def skill(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(pooled, uniform) skill over persistence, i.e. over predicting zero increment."""
    sse_zero_pooled = float(np.sum(true**2))
    pooled = 1.0 - float(np.sum((true - pred) ** 2)) / sse_zero_pooled if sse_zero_pooled else float("nan")
    num = np.sum((true - pred) ** 2, axis=0)
    den = np.sum(true**2, axis=0)
    keep = den > 0
    uniform = float(np.mean(1.0 - num[keep] / den[keep])) if keep.any() else float("nan")
    return pooled, uniform


def main() -> None:
    args = parse_args()
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    import pysindy as ps
    rng = np.random.default_rng(args.seed)

    rows = []
    for state, spec in STATES.items():
        # Pool both splits and resample, so every estimator sees the same draws and the
        # comparison carries an interval rather than resting on one held-out set.
        clips = load(spec["subdir"], args.train_split, args.task_type) \
            + load(spec["subdir"], args.eval_split, args.task_type)
        idx = target_indices(clips[0]["names"], spec["target"])
        blocks = [stack([c], idx) for c in clips]
        blocks = [b for b in blocks if b[0].shape[0] > 0]
        n = len(blocks)
        n_te = max(int(round(n * args.test_fraction)), 2)
        splits = [rng.permutation(n) for _ in range(args.n_splits)]

        print(f"\n=== {state}: {spec['note']} ===")
        print(f"  {n} clips, {args.n_splits} resplits, {len(idx)} target columns")
        print(f"  {'estimator':24s} {'nonzero':>8s} {'skill(unif)':>12s} {'95% CI':>20s}")

        def run(fit_fn):
            vals, nz = [], []
            for perm in splits:
                te, tr = perm[:n_te], perm[n_te:]
                D_tr = np.vstack([blocks[i][0] for i in tr])
                Y_tr = np.vstack([blocks[i][1] for i in tr])
                D_te = np.vstack([blocks[i][0] for i in te])
                Y_te = np.vstack([blocks[i][1] for i in te])
                scaler = StandardScaler().fit(D_tr)
                Ztr, Zte = scaler.transform(D_tr), scaler.transform(D_te)
                ymu, ysd = Y_tr.mean(axis=0), Y_tr.std(axis=0)
                ysd[ysd < 1e-12] = 1.0
                pred, k = fit_fn(Ztr, (Y_tr - ymu) / ysd, Zte)
                pred = pred * ysd + ymu
                _, u = skill(Y_te, pred)
                if np.isfinite(u) and abs(u) < 10:
                    vals.append(u)
                    nz.append(k)
            a = np.asarray(vals)
            se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
            return a, (a.mean() - 1.96 * se, a.mean() + 1.96 * se), float(np.mean(nz))

        def ridge_fit(Ztr, Ytr_s, Zte):
            m = RidgeCV(alphas=RIDGE_ALPHAS).fit(Ztr, Ytr_s)
            return np.asarray(m.predict(Zte)), int(np.sum(np.abs(np.asarray(m.coef_)) > 1e-10))

        def stlsq_fit(thr):
            def f(Ztr, Ytr_s, Zte):
                coefs = np.zeros((Ytr_s.shape[1], Ztr.shape[1]))
                for j in range(Ytr_s.shape[1]):
                    o = ps.STLSQ(threshold=thr, alpha=0.001, max_iter=20)
                    o.fit(Ztr, Ytr_s[:, j])
                    coefs[j] = np.asarray(o.coef_).ravel()
                return Zte @ coefs.T, int(np.sum(np.abs(coefs) > 1e-10))
            return f

        rows.append({"state": state, "estimator": "persistence (no model)", "n_splits": 0,
                     "nonzero_coefs": 0, "skill_uniform": 0.0, "ci_low": 0.0, "ci_high": 0.0})
        print(f"  {'persistence':24s} {0:>8d} {0.0:>12.4f} {'':>20s}")

        a, ci, k = run(ridge_fit)
        rows.append({"state": state, "estimator": "RidgeCV (dense upper bound)",
                     "n_splits": a.size, "nonzero_coefs": k,
                     "skill_uniform": float(a.mean()), "ci_low": ci[0], "ci_high": ci[1]})
        print(f"  {'RidgeCV (dense)':24s} {k:>8.0f} {a.mean():>12.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}]")

        for thr in THRESHOLDS:
            a, ci, k = run(stlsq_fit(thr))
            rows.append({"state": state, "estimator": f"STLSQ threshold={thr}",
                         "n_splits": a.size, "nonzero_coefs": k,
                         "skill_uniform": float(a.mean()), "ci_low": ci[0], "ci_high": ci[1]})
            print(f"  {'STLSQ thr=' + str(thr):24s} {k:>8.0f} {a.mean():>12.4f} "
                  f"[{ci[0]:+.4f},{ci[1]:+.4f}]")

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n--- verdict ---")
    for state in STATES:
        sub = [r for r in rows if r["state"] == state and r["n_splits"]]
        ridge = next(r for r in sub if r["estimator"].startswith("RidgeCV"))
        best = max((r for r in sub if r["estimator"].startswith("STLSQ")),
                   key=lambda r: r["skill_uniform"])
        gap = ridge["skill_uniform"] - best["skill_uniform"]
        overlap = not (ridge["ci_low"] > best["ci_high"] or best["ci_low"] > ridge["ci_high"])
        print(f"  {state}: ridge {ridge['skill_uniform']:+.4f} "
              f"[{ridge['ci_low']:+.4f},{ridge['ci_high']:+.4f}] vs best STLSQ "
              f"{best['skill_uniform']:+.4f} [{best['ci_low']:+.4f},{best['ci_high']:+.4f}]")
        print(f"    gap {gap:+.4f}; intervals {'OVERLAP -> sparsification costs nothing detectable' if overlap else 'are disjoint -> sparsification has a real cost'}")
    print(f"\noutput: {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
