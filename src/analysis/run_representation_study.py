#!/usr/bin/env python3
"""Label-validated representation study: do these representations predict pitch labels?

The dynamics pipeline elsewhere in this project is judged by self-reconstruction of the
motion, and fails at it. This study asks a different question that the data can answer:
does a data-driven representation of a pitching delivery carry structure that predicts
labels external to the pose -- release velocity and pitch type?

Pre-registration (fixed before any result was inspected):

  Primary endpoint    T1, pitch_speed_mph regression, RidgeCV, CV R2.
  Reference           R0_kinematic_summary. A dynamical representation is only interesting
                      if it beats this; R0 contains no dynamical model at all.
  Secondary           T2, binary fast vs offspeed classification, balanced accuracy.
  Significance        permutation null on the labels; a result counts only if it beats both
                      the trivial baseline and the null.

Nothing here is called a win on the strength of a point estimate alone. Every row carries a
95% CI, a baseline, and a p-value.

Usage:
    python src/analysis/run_representation_study.py [--permutations N] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.downstream_eval import (
    evaluate_classification,
    evaluate_regression,
)
from src.analysis.representations import BASELINE_REPRESENTATION, REPRESENTATIONS

FAST_TYPES = {"fastball", "sinker"}
POOL_CSV = "data/baseball/splits/pitching_local_pool.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-subdir", default="features")
    p.add_argument("--splits", nargs="*", default=("train", "test"),
                   help="Feature splits to pool. CV below is over all clips; the "
                        "train/test split belonged to the global dynamics fit.")
    p.add_argument("--task-type", default="pitching")
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pca-variance", type=float, default=0.95,
                   help="Applied identically to every representation for fairness. "
                        "Set to 0 to disable.")
    p.add_argument("--quick", action="store_true",
                   help="Fewer repeats/permutations for a smoke run.")
    p.add_argument("--output-dir", default="results/analysis/representation_study")
    return p.parse_args()


def load_clips(splits: list[str], feature_subdir: str, task_type: str) -> list[dict[str, Any]]:
    clips = []
    for split in splits:
        manifest = (PROJECT_ROOT / "data" / "baseball" / "processed" / split
                    / feature_subdir / "manifest.csv")
        with manifest.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["task_type"] != task_type:
                    continue
                obj = np.load(PROJECT_ROOT / row["filepath"])
                clips.append({
                    "clip_id": row["clip_id"],
                    "X": np.asarray(obj["X"], dtype=np.float64),
                    "feature_names": obj["feature_names"].tolist(),
                    "dt": float(row["dt"]),
                })
    return clips


def load_labels() -> dict[str, dict[str, Any]]:
    with (PROJECT_ROOT / POOL_CSV).open(newline="") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def build_matrix(clips: list[dict[str, Any]], name: str) -> tuple[np.ndarray, list[str]]:
    extractor = REPRESENTATIONS[name]
    vectors, names = [], None
    for c in clips:
        vec, fnames = extractor(c["X"], c["dt"], c["feature_names"])
        if names is None:
            names = fnames
        elif len(fnames) != len(names):
            raise SystemExit(f"{name}: inconsistent dimension on clip {c['clip_id']}")
        vectors.append(vec)
    R = np.vstack(vectors)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R, names or []


def importance_report(R: np.ndarray, y: np.ndarray, names: list[str], top: int = 15) -> list[dict[str, Any]]:
    """Standardized ridge coefficients, largest magnitude first.

    Correlated high-dimensional representations make individual coefficients unstable, so
    this is a description of what the fit leans on, not a causal claim about mechanics.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(R)
    model = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(Z, y)
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    order = np.argsort(-np.abs(coef))[:top]
    return [{"rank": i + 1, "component": names[j] if j < len(names) else f"c{j}",
             "coefficient": float(coef[j])} for i, j in enumerate(order)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.repeats, args.permutations = 3, 20
    pca = args.pca_variance if args.pca_variance and args.pca_variance > 0 else None

    clips = load_clips(list(args.splits), args.feature_subdir, args.task_type)
    labels = load_labels()
    keep = [c for c in clips if c["clip_id"] in labels]
    dropped = len(clips) - len(keep)
    if dropped:
        print(f"warning: {dropped} clip(s) had no label row and were dropped")
    if not keep:
        raise SystemExit("no labelled clips found")

    speed = np.array([float(labels[c["clip_id"]]["pitch_speed_mph"]) for c in keep])
    ptype = np.array([labels[c["clip_id"]]["pitch_type"] for c in keep])
    is_fast = np.array([1 if t in FAST_TYPES else 0 for t in ptype])

    print(f"clips: {len(keep)}   speed: {speed.min():.1f}-{speed.max():.1f} mph "
          f"(mean {speed.mean():.1f})")
    print(f"fast (fastball/sinker): {int(is_fast.sum())}   offspeed: {int((1 - is_fast).sum())}")
    print(f"protocol: {args.folds}-fold x {args.repeats} repeats, "
          f"{args.permutations} permutations, PCA={pca}\n")

    rows: list[dict[str, Any]] = []
    importances: dict[str, Any] = {}

    for name in REPRESENTATIONS:
        R, fnames = build_matrix(keep, name)
        tag = " (reference)" if name == BASELINE_REPRESENTATION else ""
        print(f"{name}{tag}: {R.shape[1]} dims")

        reg = evaluate_regression(
            R, speed, n_splits=args.folds, n_repeats=args.repeats, seed=args.seed,
            n_permutations=args.permutations, pca_variance=pca,
        )
        rows.append(reg.as_row(representation=name, task="T1_speed_regression",
                              is_reference=name == BASELINE_REPRESENTATION))
        print(f"  T1 speed   R2={reg.score_mean:+.4f} "
              f"[{reg.score_ci[0]:+.4f},{reg.score_ci[1]:+.4f}]  "
              f"baseline={reg.baseline_mean:+.4f}  p={reg.p_value:.4f}  "
              f"MAE={reg.secondary.get('mae', float('nan')):.2f} mph "
              f"(baseline {reg.secondary.get('baseline_mae', float('nan')):.2f})")

        clf = evaluate_classification(
            R, is_fast, n_splits=args.folds, n_repeats=args.repeats, seed=args.seed,
            n_permutations=args.permutations, pca_variance=pca,
        )
        rows.append(clf.as_row(representation=name, task="T2_fast_vs_offspeed",
                               is_reference=name == BASELINE_REPRESENTATION))
        print(f"  T2 class   bal.acc={clf.score_mean:.4f} "
              f"[{clf.score_ci[0]:.4f},{clf.score_ci[1]:.4f}]  "
              f"baseline={clf.baseline_mean:.4f}  p={clf.p_value:.4f}  "
              f"macroF1={clf.secondary.get('macro_f1', float('nan')):.4f}")

        importances[name] = {
            "T1_speed_regression": importance_report(R, speed, fnames),
            "T2_fast_vs_offspeed": importance_report(R, is_fast.astype(float), fnames),
        }

    out_dir = PROJECT_ROOT / args.output_dir
    write_csv(out_dir / "comparison.csv", rows)
    (out_dir / "importances.json").write_text(json.dumps(importances, indent=2))

    # Verdict on the pre-registered primary endpoint.
    print("\n--- pre-registered primary endpoint: T1 speed regression ---")
    t1 = {r["representation"]: r for r in rows if r["task"] == "T1_speed_regression"}
    ref = t1.get(BASELINE_REPRESENTATION)
    for name, r in t1.items():
        if name == BASELINE_REPRESENTATION:
            continue
        beats_ref = ref is not None and r["score_mean"] > ref["score_mean"]
        verdict = ("BEATS reference and null" if beats_ref and r["p_value"] < 0.05
                   else "does NOT clear both bars")
        print(f"  {name}: R2={r['score_mean']:+.4f} vs reference "
              f"{ref['score_mean']:+.4f}, p={r['p_value']:.4f} -> {verdict}")
    if ref is not None:
        ref_ok = ref["score_mean"] > ref["baseline_mean"] and ref["p_value"] < 0.05
        print(f"  reference R0 itself: R2={ref['score_mean']:+.4f}, p={ref['p_value']:.4f} -> "
              f"{'carries signal' if ref_ok else 'no signal above null'}")

    print(f"\noutput: {(out_dir / 'comparison.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
