#!/usr/bin/env python3
"""Sparse identification on a delay-augmented state: does STLSQ keep the skill ridge finds?

Section 6.8 established that the binding constraint on this pipeline is the state, not the
estimator. A dense ridge model of joint positions alone reaches +0.0131 skill over
persistence on next-increment prediction; the same ridge given two extra frames of history
reaches +0.3874. An autonomous `dx/dt = f(x)` has no mechanism to use history, which is why
the fitted SINDy equations scored zero.

The open question is whether the project's actual method -- sparse identification -- keeps
that skill when the state is augmented with history, or whether sparsification throws it
away. If STLSQ retains it, then SINDy works on this data once the state carries history, and
the positive result belongs to the pipeline rather than to a ridge baseline. The discovered
terms are also interpretable, which a ridge fit's dense coefficients are not.

Design:  [x(t), delta(t-1), ..., delta(t-depth+1)]   levels plus recent increments
Target:  delta(t) = x(t+1) - x(t)  on the 8 shared core position columns
Control: persistence, i.e. predicting zero increment, which scores 0 by construction

History depth is swept so the skill-versus-depth curve is visible rather than assumed, and
both aggregations are reported because they disagreed in sign elsewhere in this project
(Section 6.6); a result is only claimed when both agree.

Usage:
    python src/analysis/fit_delay_sindy.py [--depths 1 2 3 4 5] [--thresholds ...]
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-subdir", default="features")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--task-type", default="pitching")
    p.add_argument("--depths", nargs="*", type=int, default=(1, 2, 3, 4, 5),
                   help="History depth. 1 means levels only, i.e. no history.")
    p.add_argument("--thresholds", nargs="*", type=float, default=(0.0, 0.01, 0.05, 0.1, 0.2),
                   help="STLSQ thresholds to sweep at the best depth.")
    p.add_argument("--equations-depth", type=int, default=3,
                   help="Depth at which to write out the discovered sparse equations.")
    p.add_argument("--equations-threshold", type=float, default=0.05)
    p.add_argument("--output-dir", default="results/analysis/delay_sindy")
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


def design(clips: list[dict], depth: int, target_idx: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rows of [x(t), delta(t-1), ..., delta(t-depth+1)] against target delta(t)."""
    names = clips[0]["names"]
    D, Y = [], []
    for c in clips:
        X = c["X"]
        n = X.shape[0]
        # need depth-1 past increments and one future frame
        start = depth - 1
        if n < start + 2:
            continue
        for t in range(start, n - 1):
            feats = [X[t]]
            for k in range(depth - 1):
                feats.append(X[t - k] - X[t - k - 1])
            D.append(np.concatenate(feats))
            Y.append(X[t + 1, target_idx] - X[t, target_idx])
    labels = list(names)
    for k in range(depth - 1):
        labels += [f"d{k+1}({nm})" for nm in names]
    return np.asarray(D), np.asarray(Y), labels


def skill(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(pooled, uniform) skill over persistence."""
    denom = float(np.sum(true**2))
    pooled = 1.0 - float(np.sum((true - pred) ** 2)) / denom if denom else float("nan")
    num = np.sum((true - pred) ** 2, axis=0)
    den = np.sum(true**2, axis=0)
    keep = den > 0
    uniform = float(np.mean(1.0 - num[keep] / den[keep])) if keep.any() else float("nan")
    return pooled, uniform


def momentum_controls(clips: list[dict], depth: int, target_idx: list[int]) -> dict[str, tuple[float, float]]:
    """Unfitted controls for the delay model. No parameters at all.

    The fitted equations are dominated by each variable's own previous increment, which is
    momentum rather than biomechanics: a joint tends to keep moving as it was moving. So the
    control that matters is not persistence (zero increment) but momentum:

        first_order   delta_hat(t) = delta(t-1)
        second_order  delta_hat(t) = 2*delta(t-1) - delta(t-2)   (constant-acceleration)

    Any claim that the fit has learned coupling between joints has to survive comparison
    against these, because they reproduce the self-terms for free.
    """
    out: dict[str, list] = {"momentum_first_order": [[], []], "momentum_second_order": [[], []]}
    for c in clips:
        X = c["X"]
        n = X.shape[0]
        start = max(depth - 1, 2)
        if n < start + 2:
            continue
        for t in range(start, n - 1):
            true = X[t + 1, target_idx] - X[t, target_idx]
            d1 = X[t, target_idx] - X[t - 1, target_idx]
            d2 = X[t - 1, target_idx] - X[t - 2, target_idx]
            out["momentum_first_order"][0].append(true)
            out["momentum_first_order"][1].append(d1)
            out["momentum_second_order"][0].append(true)
            out["momentum_second_order"][1].append(2.0 * d1 - d2)
    return {k: skill(np.asarray(v[0]), np.asarray(v[1])) for k, v in out.items()}


def self_only_mask(labels: list[str], target_names: list[str], names: list[str]) -> np.ndarray:
    """Mask selecting, for each target, only terms built from that same variable.

    Zeroing everything else turns the model into a set of independent per-joint momentum
    models. Comparing it against the unrestricted fit isolates how much the cross-joint
    coupling terms actually contribute.
    """
    mask = np.zeros((len(target_names), len(labels)), dtype=bool)
    for i, tname in enumerate(target_names):
        for j, lab in enumerate(labels):
            base = lab
            if lab.startswith("d") and "(" in lab:
                base = lab[lab.index("(") + 1:lab.rindex(")")]
            if base == tname:
                mask[i, j] = True
    return mask


def fit_stlsq(Ztr: np.ndarray, Ytr: np.ndarray, threshold: float,
              mask: np.ndarray | None = None) -> np.ndarray:
    import pysindy as ps
    coefs = np.zeros((Ytr.shape[1], Ztr.shape[1]))
    for j in range(Ytr.shape[1]):
        cols = np.where(mask[j])[0] if mask is not None else np.arange(Ztr.shape[1])
        if cols.size == 0:
            continue
        opt = ps.STLSQ(threshold=threshold, alpha=0.001, max_iter=20)
        opt.fit(Ztr[:, cols], Ytr[:, j])
        coefs[j, cols] = np.asarray(opt.coef_).ravel()
    return coefs


def main() -> None:
    args = parse_args()
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    tr = load(args.feature_subdir, args.train_split, args.task_type)
    te = load(args.feature_subdir, args.eval_split, args.task_type)
    names = tr[0]["names"]
    target_idx = [names.index(n) for n in CORE_POSITIONS if n in names]
    target_names = [names[i] for i in target_idx]
    print(f"train clips {len(tr)}, test clips {len(te)}, target cols {len(target_idx)}")

    rows: list[dict] = []
    print(f"\n{'depth':>6s} {'dims':>5s} {'estimator':>18s} {'nonzero':>8s} "
          f"{'skill(pool)':>12s} {'skill(unif)':>12s}")
    rows.append({"depth": 0, "design_dims": 0, "estimator": "persistence (no model)",
                 "nonzero_coefs": 0, "skill_pooled": 0.0, "skill_uniform": 0.0})
    print(f"{'-':>6s} {'-':>5s} {'persistence':>18s} {0:>8d} {0.0:>12.4f} {0.0:>12.4f}")

    for depth in args.depths:
        D_tr, Y_tr, labels = design(tr, depth, target_idx)
        D_te, Y_te, _ = design(te, depth, target_idx)
        scaler = StandardScaler().fit(D_tr)
        Ztr, Zte = scaler.transform(D_tr), scaler.transform(D_te)
        ymu, ysd = Y_tr.mean(axis=0), Y_tr.std(axis=0)
        ysd[ysd < 1e-12] = 1.0
        Ytr_s = (Y_tr - ymu) / ysd

        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(Ztr, Ytr_s)
        pred = np.asarray(ridge.predict(Zte)) * ysd + ymu
        sp, su = skill(Y_te, pred)
        nz = int(np.sum(np.abs(np.asarray(ridge.coef_)) > 1e-10))
        rows.append({"depth": depth, "design_dims": D_tr.shape[1], "estimator": "RidgeCV (dense)",
                     "nonzero_coefs": nz, "skill_pooled": sp, "skill_uniform": su})
        print(f"{depth:>6d} {D_tr.shape[1]:>5d} {'RidgeCV (dense)':>18s} {nz:>8d} "
              f"{sp:>12.4f} {su:>12.4f}")

        for thr in args.thresholds:
            coefs = fit_stlsq(Ztr, Ytr_s, thr)
            pred = (Zte @ coefs.T) * ysd + ymu
            sp, su = skill(Y_te, pred)
            nz = int(np.sum(np.abs(coefs) > 1e-10))
            rows.append({"depth": depth, "design_dims": D_tr.shape[1],
                         "estimator": f"STLSQ thr={thr}", "nonzero_coefs": nz,
                         "skill_pooled": sp, "skill_uniform": su})
            print(f"{depth:>6d} {D_tr.shape[1]:>5d} {'STLSQ thr=' + str(thr):>18s} {nz:>8d} "
                  f"{sp:>12.4f} {su:>12.4f}")

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "skill_by_depth.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Unfitted momentum controls: the bar the fitted model must clear to mean anything.
    print(f"\n--- unfitted controls (no parameters) ---")
    ctrl = momentum_controls(te, max(args.depths), target_idx)
    for cname, (sp, su) in ctrl.items():
        rows.append({"depth": 0, "design_dims": 0, "estimator": cname,
                     "nonzero_coefs": 0, "skill_pooled": sp, "skill_uniform": su})
        print(f"  {cname:24s} skill pooled {sp:+.4f}  uniform {su:+.4f}")

    # Coupling ablation: does anything beyond each joint's own history contribute?
    print(f"\n--- coupling ablation at depth {args.equations_depth}, "
          f"threshold {args.equations_threshold} ---")
    depth_a, thr_a = args.equations_depth, args.equations_threshold
    D_tr, Y_tr, labels = design(tr, depth_a, target_idx)
    D_te, Y_te, _ = design(te, depth_a, target_idx)
    scaler = StandardScaler().fit(D_tr)
    Ztr, Zte = scaler.transform(D_tr), scaler.transform(D_te)
    ymu, ysd = Y_tr.mean(axis=0), Y_tr.std(axis=0)
    ysd[ysd < 1e-12] = 1.0
    Ytr_s = (Y_tr - ymu) / ysd
    mask = self_only_mask(labels, target_names, names)
    for tag, m in (("self terms only", mask), ("all terms (with coupling)", None)):
        coefs = fit_stlsq(Ztr, Ytr_s, thr_a, m)
        pred = (Zte @ coefs.T) * ysd + ymu
        sp, su = skill(Y_te, pred)
        nz = int(np.sum(np.abs(coefs) > 1e-10))
        rows.append({"depth": depth_a, "design_dims": D_tr.shape[1],
                     "estimator": f"STLSQ thr={thr_a} [{tag}]", "nonzero_coefs": nz,
                     "skill_pooled": sp, "skill_uniform": su})
        print(f"  {tag:26s} {nz:>3d} nonzero  skill pooled {sp:+.4f}  uniform {su:+.4f}")

    # Discovered equations at the requested depth/threshold, for interpretation.
    depth, thr = args.equations_depth, args.equations_threshold
    D_tr, Y_tr, labels = design(tr, depth, target_idx)
    scaler = StandardScaler().fit(D_tr)
    Ztr = scaler.transform(D_tr)
    ymu, ysd = Y_tr.mean(axis=0), Y_tr.std(axis=0)
    ysd[ysd < 1e-12] = 1.0
    coefs = fit_stlsq(Ztr, (Y_tr - ymu) / ysd, thr)

    lines = [f"Sparse increment equations, history depth {depth}, STLSQ threshold {thr}.",
             "Standardized units: coefficients are comparable within an equation, and each",
             "predicts the next-frame increment of the named core position.", ""]
    for i, tname in enumerate(target_names):
        terms = [(labels[j], coefs[i, j]) for j in range(coefs.shape[1])
                 if abs(coefs[i, j]) > 1e-10]
        terms.sort(key=lambda t: -abs(t[1]))
        rhs = "  ".join(f"{c:+.3f}*{n}" for n, c in terms) if terms else "0"
        lines.append(f"delta({tname}) = {rhs}")
    (out_dir / "equations.txt").write_text("\n".join(lines) + "\n")

    print("\n--- discovered sparse equations (depth "
          f"{depth}, threshold {thr}) ---")
    for line in lines[4:]:
        print("  " + (line[:150] + ("..." if len(line) > 150 else "")))

    best = max((r for r in rows if r["estimator"].startswith("STLSQ")),
               key=lambda r: r["skill_uniform"])
    ridge_best = max((r for r in rows if r["estimator"].startswith("RidgeCV")),
                     key=lambda r: r["skill_uniform"])
    print(f"\n--- verdict ---")
    print(f"  best sparse: depth {best['depth']}, {best['estimator']}, "
          f"{best['nonzero_coefs']} nonzero -> skill pooled {best['skill_pooled']:+.4f}, "
          f"uniform {best['skill_uniform']:+.4f}")
    print(f"  best dense : depth {ridge_best['depth']} -> "
          f"uniform {ridge_best['skill_uniform']:+.4f}")
    gap = ridge_best["skill_uniform"] - best["skill_uniform"]
    if best["skill_uniform"] > 0 and best["skill_pooled"] > 0:
        print(f"  sparse model beats persistence under BOTH aggregations; "
              f"cost of sparsity vs dense = {gap:.4f}")
    else:
        print("  sparse model does not beat persistence under both aggregations")
    print(f"\noutput: {(out_dir / 'skill_by_depth.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
