#!/usr/bin/env python3
"""Is the Section 7 one-step skill a property of the motion or of centred smoothing?

Every one-step model and control in Section 7 sees coordinates smoothed by a centred
Savitzky-Golay filter (window 11, order 3), so x(t) already carries information from frames up
to t+5, and `dominant_wrist_speed` is a central difference that reads frame t+1. The increment
structure the models predict is therefore partly imposed by the filter. This script rebuilds
every clip from its raw pose with the pipeline's own functions, changing only the smoother:

    centered     the operative filter; must reproduce the stored features exactly (sanity check)
    causal_sg    one-sided Savitzky-Golay, window 11, order 3, evaluated at the newest frame
    causal_sg1   one-sided, order 1: more smoothing, more lag
    none         no smoothing at all

The causal variants also compute wrist speed by backward difference, so no design input reads a
future frame. Each clip keeps the trim window and handedness the operative run recorded, so the
variants differ in nothing but the filter and the comparison is paired clip for clip.

For each variant the Section 7 comparison is re-run: persistence, damped momentum, per-joint
AR(3), and the depth-3 sparse delay fit, on the shared rows of Section 7.2, with the paired
margin over 40 clip-level resplits.

Usage:
    python src/analysis/test_causal_smoothing.py [--resplits 40]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.preprocessing.extract_utd_mhad_features as feats
from src.analysis.fit_delay_sindy import (
    CORE_POSITIONS, local_motion_controls, resplit_margins, sparse_on_rows,
)
from src.preprocessing.extract_baseball_features import canonicalize_feature_series
from src.preprocessing.process_baseball_standardized_pose import (
    center_trajectory, fill_edge_gaps, interpolate_internal_gaps, load_pose_clip,
)

STATE = ["shoulder_rotation_angle", "hip_rotation_angle", "hip_shoulder_separation_angle",
         "dominant_elbow_flexion_angle", "trail_elbow_flexion_angle",
         "dominant_knee_flexion_angle", "trail_knee_flexion_angle", "lead_knee_lift",
         "stride_ankle_separation", "dominant_wrist_speed"]
WINDOW, ORDER = 11, 3
CONTROLS = ("per-joint AR(1) [damped momentum]", "per-joint AR(3)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resplits", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--start", type=int, default=8, help="First predicted frame; Section 7.2 uses 8.")
    p.add_argument("--output-csv", default="results/analysis/delay_sindy/causal_smoothing.csv")
    return p.parse_args()


def causal_savgol(traj: np.ndarray, window: int, order: int) -> np.ndarray:
    """Least-squares polynomial over frames t-window+1..t, evaluated at t. Uses no future frame;
    the first frames use the history available, with the order capped by it."""
    weights = {}
    for n in range(1, window + 1):
        q = min(order, n - 1)
        V = np.vander(np.arange(-n + 1, 1, dtype=float), q + 1, increasing=True)
        weights[n] = np.linalg.pinv(V)[0]          # value of the fit at time 0, the newest frame
    out = np.empty_like(traj)
    for t in range(traj.shape[0]):
        n = min(window, t + 1)
        out[t] = np.tensordot(weights[n], traj[t - n + 1:t + 1], axes=1)
    return out


def backward_speed(points: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.vstack([np.zeros((1, points.shape[1])), np.diff(points, axis=0)]) / dt
    velocity[0] = velocity[1] if len(velocity) > 1 else 0.0
    return np.linalg.norm(velocity, axis=1)


def smooth(traj: np.ndarray, variant: str) -> np.ndarray:
    if variant == "centered":
        n = traj.shape[0]
        w = min(WINDOW, n if n % 2 == 1 else n - 1)
        return savgol_filter(traj, w, ORDER, axis=0, mode="interp") if w > ORDER and w >= 3 else traj.copy()
    if variant == "causal_sg":
        return causal_savgol(traj, WINDOW, ORDER)
    if variant == "causal_sg1":
        return causal_savgol(traj, WINDOW, 1)
    if variant == "none":
        return traj.copy()
    raise ValueError(variant)


def rebuild(split: str, variant: str) -> list[dict]:
    root = PROJECT_ROOT / "data/baseball/processed" / split
    with (root / "features" / "manifest.csv").open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["task_type"] == "pitching"]
    # Wrist speed is the only design input computed by differencing; the pipeline uses a central
    # difference, which reads frame t+1. Causal variants swap in a backward difference.
    feats.velocity_magnitude = (backward_speed if variant != "centered"
                                else _central_speed)
    clips = []
    for r in rows:
        fobj = np.load(PROJECT_ROOT / r["filepath"])
        fmeta = json.loads(str(fobj["metadata"]))
        raw, joints, _ = load_pose_clip(PROJECT_ROOT / fmeta["source_filepath"])
        traj, _ = fill_edge_gaps(raw)
        traj, _ = interpolate_internal_gaps(traj)
        traj = center_trajectory(traj, joints, "hip_center")
        traj = smooth(traj, variant)[int(fmeta["trim_start"]):int(fmeta["trim_end"])]
        # The pipeline stores processed coordinates and features as float32; match both casts so
        # the centred rebuild reproduces the stored features exactly.
        traj = traj.astype(np.float32).astype(np.float64)
        series = feats.build_feature_series(traj, joints, float(r["dt"]))
        series = canonicalize_feature_series(series, fmeta["resolved_handedness"])
        X = feats.build_state_matrix(series, STATE, True).astype(np.float32).astype(np.float64)
        clips.append({"X": X, "names": STATE, "stored": np.asarray(fobj["X"], dtype=np.float64)})
    return clips


_central_speed = feats.velocity_magnitude


def main() -> None:
    args = parse_args()
    target_idx = [STATE.index(n) for n in CORE_POSITIONS if n in STATE]
    out_rows = []
    for variant in ("centered", "causal_sg", "causal_sg1", "none"):
        tr, te = rebuild("train", variant), rebuild("test", variant)
        if variant == "centered":
            err = max(float(np.max(np.abs(c["X"] - c["stored"]))) for c in tr + te)
            print(f"sanity: centred rebuild vs stored features, max abs diff {err:.2e}")
            if err > 1e-8:
                raise SystemExit("rebuild does not reproduce the stored features; results would be unpaired")
        ctrl = {r["estimator"]: r for r in local_motion_controls(tr, te, args.start, 3, target_idx)}
        sp, su, nz = sparse_on_rows(tr, te, args.depth, args.threshold, args.start, target_idx)
        margins = resplit_margins(tr + te, args.depth, args.threshold, args.start, target_idx,
                                  args.resplits, args.seed) if args.resplits else []
        row = {"variant": variant, "sparse_pooled": sp, "sparse_uniform": su, "sparse_terms": nz}
        for name, key in zip(CONTROLS, ("ar1", "ar3")):
            row[f"{key}_pooled"] = ctrl[name]["skill_pooled"]
            row[f"{key}_uniform"] = ctrl[name]["skill_uniform"]
        for p, key in ((1, "ar1"), (3, "ar3")):
            d = np.array([m["sparse_uniform"] - m["ar_uniform"] for m in margins if m["lags"] == p])
            dp = np.array([m["sparse_pooled"] - m["ar_pooled"] for m in margins if m["lags"] == p])
            half = 1.96 * d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else float("nan")
            row[f"margin_{key}_uniform"] = float(d.mean()) if d.size else float("nan")
            row[f"margin_{key}_ci_half"] = float(half)
            row[f"margin_{key}_positive_uniform"] = int(np.sum(d > 0))
            row[f"margin_{key}_positive_pooled"] = int(np.sum(dp > 0))
        row["resplits"] = args.resplits
        out_rows.append(row)
        print(f"{variant:11s} sparse {sp:+.4f}/{su:+.4f} ({nz} terms)  "
              f"AR1 {row['ar1_pooled']:+.4f}/{row['ar1_uniform']:+.4f}  "
              f"AR3 {row['ar3_pooled']:+.4f}/{row['ar3_uniform']:+.4f}  "
              f"margin vs AR3 {row['margin_ar3_uniform']:+.4f} +/- {row['margin_ar3_ci_half']:.4f} "
              f"(uniform +{row['margin_ar3_positive_uniform']}/{args.resplits}, "
              f"pooled +{row['margin_ar3_positive_pooled']}/{args.resplits})")

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\noutput: {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
