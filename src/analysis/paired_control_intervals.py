#!/usr/bin/env python3
"""Confidence intervals for "no fitted model beats the frozen control", paired over clips.

Section 6.2's comparison rests on a single held-out split, while the newer results all carry
intervals. Rebuilding it by resplitting would mean refitting five configurations twenty times
over, but resplitting is not the right test anyway: every configuration was scored on the SAME
29 test clips, so the comparison is naturally paired, and a paired statistic is far tighter
than comparing two independently-resampled means.

For each fitted model this computes the per-clip difference against the frozen control and
bootstraps over clips. What matters is whether the interval for the difference excludes zero,
and in which direction.

Bootstrapping over clips rather than over frames is deliberate: frames within a clip are
strongly dependent, so a frame-level resample would understate the uncertainty badly.

Usage:
    python src/analysis/paired_control_intervals.py [--n-boot 10000]
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

CONTROL = "results/evaluation/baseball/static_baseline/per_clip.csv"
MODELS = [
    ("first-order, plain (operative)", "results/evaluation/baseball/train_on_test/reconstruction_metrics.csv"),
    ("first-order, weak-form, multi-seed", "results/evaluation/baseball/train_on_test_weakform_2026-08-12/reconstruction_metrics.csv"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-csv", default="results/analysis/paired_control_intervals.csv")
    return p.parse_args()


def load_per_clip(rel: str) -> dict[str, float]:
    path = PROJECT_ROOT / rel
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            v = r.get("core_position_recon_r2", "")
            if v not in ("", None):
                try:
                    out[r["clip_id"]] = float(v)
                except ValueError:
                    pass
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    control = load_per_clip(CONTROL)
    if not control:
        raise SystemExit(f"missing {CONTROL}; run evaluate_static_baseline.py first")

    rows = []
    print(f"frozen control: {len(control)} clips, mean core R2 "
          f"{np.mean(list(control.values())):+.4f}")
    print(f"\n{'model':36s} {'clips':>6s} {'model':>10s} {'control':>10s} "
          f"{'diff':>10s} {'95% CI on diff':>22s} {'beats':>7s}")

    for label, rel in MODELS:
        model = load_per_clip(rel)
        shared = sorted(set(model) & set(control))
        if not shared:
            print(f"{label:36s}  no overlapping clips; skipped ({rel})")
            continue
        m = np.array([model[c] for c in shared])
        f = np.array([control[c] for c in shared])
        d = m - f

        idx = rng.integers(0, len(shared), size=(args.n_boot, len(shared)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        beats = int((d > 0).sum())
        rows.append({"model": label, "n_clips": len(shared),
                     "model_mean": float(m.mean()), "control_mean": float(f.mean()),
                     "mean_difference": float(d.mean()), "ci_low": float(lo),
                     "ci_high": float(hi), "clips_beating_control": beats,
                     "n_boot": args.n_boot})
        print(f"{label:36s} {len(shared):>6d} {m.mean():>10.4f} {f.mean():>10.4f} "
              f"{d.mean():>10.4f} [{lo:+.4f},{hi:+.4f}] {beats:>3d}/{len(shared):<3d}")

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n--- verdict ---")
    for r in rows:
        if r["ci_high"] < 0:
            v = "WORSE than the control, interval excludes zero"
        elif r["ci_low"] > 0:
            v = "BETTER than the control, interval excludes zero"
        else:
            v = "indistinguishable from the control"
        print(f"  {r['model']}: {v}")
    print(f"\noutput: {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
