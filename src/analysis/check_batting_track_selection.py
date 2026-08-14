#!/usr/bin/env python3
"""Batter selection by motion: identify the swinger, not the tallest or lowest pose.

Cropping the plate region solved detection -- the batter goes from 71 px to ~179 px of torso and
MediaPipe finds him with good landmark coverage. Selection is what remained broken. Two
geometric heuristics failed in this workstream:

  "lowest pose in frame"   picks the PITCHER, who is nearest the centerfield camera
  "tallest pose in crop"   picks the umpire and catcher FUSED into one box, or the pitcher when
                           he intrudes into the crop in a given stadium's framing

Both produced aggregate numbers that looked like batter coverage and were not. The fix here uses
the one signal that actually distinguishes the batter: he is the person swinging. The umpire and
catcher are near-static, and the pitcher has completed his delivery by the time the swing occurs.

So: detect poses per frame, associate them across frames into tracks, and select the track with
the greatest mean wrist speed. That also aligns subject selection with the quantity a swing study
would measure, rather than with an incidental property of the broadcast framing.

Selection requires the whole clip, so this is a two-pass process: collect all poses, build
tracks, choose, then score and render.

Gate for proceeding to feature design: selected-track coverage above 0.85, torso comparable to
the pitching pipeline's 80-120 px, and visual confirmation on frames spanning the swing rather
than a single frame.

Usage:
    python src/analysis/check_batting_track_selection.py [--n-clips 20] [--render-frames 6]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.check_batting_pose_feasibility import load_batting, outcome_of

MODEL = PROJECT_ROOT / "data/models/pose_landmarker_full.task"
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 11, 12, 15, 16, 23, 24


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-clips", type=int, default=20)
    p.add_argument("--crop", default="0.28,0.08,0.75,0.68")
    p.add_argument("--upscale", type=float, default=2.0)
    p.add_argument("--num-poses", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--match-radius", type=float, default=0.12,
                   help="Max normalized centroid distance to continue a track between frames.")
    p.add_argument("--exclude-below-y", type=float, default=0.75,
                   help="Drop tracks whose mean torso centroid sits below this fraction of the "
                        "crop height. The pitcher intrudes into the bottom of the crop in some "
                        "stadiums' framing and is nearest the camera, so he lands there.")
    p.add_argument("--min-coverage", type=float, default=0.25,
                   help="A track must appear in this fraction of frames to be selectable, so a "
                        "short spurious track with large jumps cannot win on speed alone.")
    p.add_argument("--render-frames", type=int, default=6,
                   help="Annotated frames per clip, spread across the clip rather than one frame.")
    p.add_argument("--render-clips", type=int, default=4)
    p.add_argument("--two-pass", action="store_true",
                   help="After locating the batter in the plate crop, re-crop tightly around him "
                        "and run pose again. On most frames MediaPipe returns only 2 detections "
                        "even with num_poses=4, so the batter is often missed outright in a crop "
                        "containing four people; a tight single-person crop removes that "
                        "competition entirely.")
    p.add_argument("--tight-margin", type=float, default=0.9,
                   help="Tight crop half-extent as a multiple of the batter bbox size.")
    p.add_argument("--tight-upscale", type=float, default=2.0)
    p.add_argument("--truncate-at-pan", action="store_true",
                   help="Stop at camera-motion onset. These broadcasts hold a static plate view "
                        "during the pitch and swing, then pan to follow the ball, so the batter "
                        "leaves a fixed crop partway through. Everything after the pan is ball "
                        "flight and irrelevant to swing kinematics.")
    p.add_argument("--pan-ratio", type=float, default=3.0,
                   help="Motion onset is where a 10-frame rolling mean frame-difference exceeds "
                        "this multiple of the clip's quiet baseline.")
    p.add_argument("--media-root", default="")
    p.add_argument("--clip-ids-file", default="")
    p.add_argument("--frames-out", default="results/analysis/batting_feasibility/track_frames")
    p.add_argument("--output-csv", default="results/analysis/batting_feasibility/track_summary.csv")
    return p.parse_args()


def load_named(clip_ids: list[str], media_root: Path) -> list[dict]:
    manifest = PROJECT_ROOT / "data/baseball/metadata/clip_manifest.csv"
    wanted, by_id = set(clip_ids), {}
    with manifest.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["clip_id"] not in wanted or r["task_type"] != "batting":
                continue
            try:
                notes = json.loads(r["notes"])
            except Exception:
                notes = {}
            path = media_root / f"{r['clip_id']}.mp4"
            if path.exists():
                by_id[r["clip_id"]] = {"clip_id": r["clip_id"], "path": path,
                                       "outcome": outcome_of(notes.get("labels", []))}
    return [by_id[i] for i in clip_ids if i in by_id]


def pan_onset(path: Path, ratio: float) -> int:
    """First frame where the camera starts moving, or a large number if it never does.

    Uses mean absolute frame difference on a downscaled grey image. A pan is gradual, so it
    shows as a sustained rise rather than a spike -- hence a rolling mean against the clip's
    own quiet baseline rather than a fixed cut threshold.
    """
    cap = cv2.VideoCapture(str(path))
    prev, diffs = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    if len(diffs) < 20:
        return 10**9
    d = np.asarray(diffs)
    baseline = float(np.mean(d[: max(len(d) // 10, 5)]))
    roll = np.convolve(d, np.ones(10) / 10, mode="valid")
    hits = np.where(roll > max(ratio * baseline, 0.02))[0]
    return int(hits[0]) if hits.size else 10**9


def centroid(lms) -> np.ndarray:
    """Torso centroid: shoulders and hips only. More stable than all 33 landmarks, which
    swing wildly during the swing itself and would break frame-to-frame association."""
    pts = [(lms[i].x, lms[i].y) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    return np.mean(pts, axis=0)


def wrist_point(lms) -> np.ndarray:
    return np.mean([(lms[L_WRIST].x, lms[L_WRIST].y), (lms[R_WRIST].x, lms[R_WRIST].y)], axis=0)


def build_tracks(frames: list[tuple[int, list]], match_radius: float) -> list[dict]:
    """Greedy nearest-centroid association across frames.

    Returns tracks as dicts with the frame indices they appear in, their centroid and wrist
    trajectories, and the raw landmark lists for rendering.
    """
    tracks: list[dict] = []
    for fi, poses in frames:
        used = set()
        # Continue existing tracks first, nearest pair first, so a clean match is not stolen.
        cands = []
        for ti, tr in enumerate(tracks):
            for pi, lms in enumerate(poses):
                d = float(np.linalg.norm(centroid(lms) - tr["centroids"][-1]))
                cands.append((d, ti, pi))
        cands.sort()
        claimed_tracks = set()
        for d, ti, pi in cands:
            if d > match_radius or ti in claimed_tracks or pi in used:
                continue
            tracks[ti]["frames"].append(fi)
            tracks[ti]["centroids"].append(centroid(poses[pi]))
            tracks[ti]["wrists"].append(wrist_point(poses[pi]))
            tracks[ti]["landmarks"][fi] = poses[pi]
            claimed_tracks.add(ti)
            used.add(pi)
        for pi, lms in enumerate(poses):
            if pi in used:
                continue
            tracks.append({"frames": [fi], "centroids": [centroid(lms)],
                           "wrists": [wrist_point(lms)], "landmarks": {fi: lms}})
    return tracks


def mean_wrist_speed(track: dict) -> float:
    w = np.asarray(track["wrists"])
    if w.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(w, axis=0), axis=1)))


def main() -> None:
    args = parse_args()
    if not MODEL.exists():
        raise SystemExit(f"missing pose model: {MODEL}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    x0f, y0f, x1f, y1f = (float(v) for v in args.crop.split(","))
    if args.clip_ids_file:
        ids = [l.strip() for l in Path(args.clip_ids_file).read_text().split("\n") if l.strip()]
        root = Path(args.media_root) if args.media_root else Path("/mnt/e/MML_RESEARCH/SEGMENTED/segmented")
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        clips = load_named(ids, root)
    else:
        clips = load_batting(args.n_clips, args.seed)
    if not clips:
        raise SystemExit("no batting clips found")
    print(f"checking {len(clips)} clips, crop=({x0f},{y0f})-({x1f},{y1f}), upscale={args.upscale}x")

    frames_dir = PROJECT_ROOT / args.frames_out
    frames_dir.mkdir(parents=True, exist_ok=True)

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    for ci, clip in enumerate(clips, 1):
        cap = cv2.VideoCapture(str(clip["path"]))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 59.94

        stop_at = pan_onset(clip["path"], args.pan_ratio) if args.truncate_at_pan else 10**9

        collected: list[tuple[int, list]] = []
        raw_frames: dict[int, np.ndarray] = {}
        crop_hw = (0, 0)
        with mp_vision.PoseLandmarker.create_from_options(opts) as lm:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or idx >= stop_at:
                    break
                if idx % args.frame_stride:
                    idx += 1
                    continue
                H, W = frame.shape[:2]
                crop = frame[int(y0f * H):int(y1f * H), int(x0f * W):int(x1f * W)]
                if args.upscale != 1.0:
                    crop = cv2.resize(crop, None, fx=args.upscale, fy=args.upscale,
                                      interpolation=cv2.INTER_CUBIC)
                crop_hw = crop.shape[:2]
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                    int(idx / fps * 1000))
                collected.append((idx, list(res.pose_landmarks or [])))
                if ci <= args.render_clips:
                    raw_frames[idx] = crop
                idx += 1
        cap.release()

        n_frames = len(collected)
        tracks = build_tracks(collected, args.match_radius)
        # Selection, fourth attempt, this time from the diagnosed track geometry rather than a
        # guess. Wrist speed turned out to be a poor discriminator: detection jitter inflates it,
        # so a briefly-tracked umpire outscored the batter and the pitcher beat him too. What does
        # separate them is (a) the pitcher sits in the bottom band of the crop, nearest the
        # camera, and (b) the batter is the most consistently detected subject in a plate-centred
        # crop, being large and unoccluded for most of the clip.
        eligible = [t for t in tracks
                    if len(t["frames"]) >= args.min_coverage * n_frames
                    and float(np.mean(t["centroids"], axis=0)[1]) <= args.exclude_below_y]
        eligible.sort(key=lambda t: -len(t["frames"]))
        chosen = eligible[0] if eligible else None

        ch, cw = crop_hw
        if chosen is None:
            row = {"clip_id": clip["clip_id"], "outcome": clip["outcome"],
                   "frames_processed": n_frames, "n_tracks": len(tracks), "n_eligible": 0,
                   "selected_coverage": 0.0, "selected_wrist_speed": 0.0,
                   "runner_up_coverage": 0.0, "batter_torso_px": 0.0, "mean_visibility": 0.0}
        else:
            covs = [len(t["frames"]) / n_frames for t in eligible]
            torso, vis = [], []
            for fi in chosen["frames"]:
                lms = chosen["landmarks"][fi]
                sh = np.mean([lms[L_SHOULDER].y, lms[R_SHOULDER].y])
                hp = np.mean([lms[L_HIP].y, lms[R_HIP].y])
                torso.append(abs(sh - hp) * ch)
                vis.append(float(np.mean([l.visibility for l in lms])))
            row = {"clip_id": clip["clip_id"], "outcome": clip["outcome"],
                   "frames_processed": n_frames, "n_tracks": len(tracks),
                   "n_eligible": len(eligible),
                   "selected_coverage": len(chosen["frames"]) / n_frames,
                   "selected_wrist_speed": mean_wrist_speed(chosen),
                   "runner_up_coverage": covs[1] if len(covs) > 1 else 0.0,
                   "batter_torso_px": float(np.mean(torso)),
                   "mean_visibility": float(np.mean(vis))}

            if ci <= args.render_clips and raw_frames:
                picks = np.linspace(0, len(chosen["frames"]) - 1,
                                    min(args.render_frames, len(chosen["frames"]))).astype(int)
                for k in picks:
                    fi = chosen["frames"][k]
                    if fi not in raw_frames:
                        continue
                    ann = raw_frames[fi].copy()
                    for _, poses in [(fi, collected[[c[0] for c in collected].index(fi)][1])]:
                        for lms in poses:
                            sel = lms is chosen["landmarks"].get(fi)
                            col = (0, 255, 0) if sel else (128, 128, 128)
                            for l in lms:
                                cv2.circle(ann, (int(l.x * cw), int(l.y * ch)), 3, col, -1)
                            bx0 = int(min(l.x for l in lms) * cw); by0 = int(min(l.y for l in lms) * ch)
                            bx1 = int(max(l.x for l in lms) * cw); by1 = int(max(l.y for l in lms) * ch)
                            cv2.rectangle(ann, (bx0, by0), (bx1, by1), col, 3 if sel else 1)
                            if sel:
                                cv2.putText(ann, "SELECTED", (bx0, max(by0 - 8, 14)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                    cv2.imwrite(str(frames_dir / f"{clip['clip_id']}_f{fi:04d}.png"), ann)

        # --- optional second pass: tight single-person crop around the located batter ---
        if args.two_pass and chosen is not None:
            cs = np.asarray([[min(l.x for l in chosen["landmarks"][f]),
                              min(l.y for l in chosen["landmarks"][f]),
                              max(l.x for l in chosen["landmarks"][f]),
                              max(l.y for l in chosen["landmarks"][f])]
                             for f in chosen["frames"]])
            bx0, by0, bx1, by1 = cs[:, 0].min(), cs[:, 1].min(), cs[:, 2].max(), cs[:, 3].max()
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            half_w = (bx1 - bx0) / 2 * (1 + args.tight_margin)
            half_h = (by1 - by0) / 2 * (1 + args.tight_margin)
            # crop-fraction coords -> full-frame fractions
            fw, fh = (x1f - x0f), (y1f - y0f)
            tx0 = max(x0f + (cx - half_w) * fw, 0.0)
            tx1 = min(x0f + (cx + half_w) * fw, 1.0)
            ty0 = max(y0f + (cy - half_h) * fh, 0.0)
            ty1 = min(y0f + (cy + half_h) * fh, 1.0)

            cap2 = cv2.VideoCapture(str(clip["path"]))
            opts2 = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
                running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4, min_tracking_confidence=0.4)
            n2 = found2 = 0
            torso2, vis2 = [], []
            tight_hits: list = []
            with mp_vision.PoseLandmarker.create_from_options(opts2) as lm2:
                j = 0
                while True:
                    ok, frame = cap2.read()
                    if not ok or j >= stop_at:
                        break
                    if j % args.frame_stride:
                        j += 1
                        continue
                    H, W = frame.shape[:2]
                    tc = frame[int(ty0 * H):int(ty1 * H), int(tx0 * W):int(tx1 * W)]
                    if tc.size == 0:
                        j += 1
                        continue
                    tc = cv2.resize(tc, None, fx=args.tight_upscale, fy=args.tight_upscale,
                                    interpolation=cv2.INTER_CUBIC)
                    th, tw = tc.shape[:2]
                    n2 += 1
                    r2 = lm2.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(tc, cv2.COLOR_BGR2RGB)),
                        int(j / fps * 1000))
                    if r2.pose_landmarks:
                        lms = r2.pose_landmarks[0]
                        found2 += 1
                        if ci <= args.render_clips:
                            tight_hits.append((j, tc.copy(), lms))
                        sh = np.mean([lms[L_SHOULDER].y, lms[R_SHOULDER].y])
                        hp = np.mean([lms[L_HIP].y, lms[R_HIP].y])
                        torso2.append(abs(sh - hp) * th)
                        vis2.append(float(np.mean([l.visibility for l in lms])))
                    j += 1
            cap2.release()
            # Render pass-2 frames spanning the clip: the pass-1 renders show the plate crop,
            # which cannot confirm what the tight single-person crop actually locked onto.
            if ci <= args.render_clips and tight_hits:
                picks = np.linspace(0, len(tight_hits) - 1,
                                    min(args.render_frames, len(tight_hits))).astype(int)
                for k in picks:
                    fj, img, lms = tight_hits[int(k)]
                    ann = img.copy()
                    th2, tw2 = ann.shape[:2]
                    for l in lms:
                        cv2.circle(ann, (int(l.x * tw2), int(l.y * th2)), 4, (0, 255, 0), -1)
                    cv2.imwrite(str(frames_dir / f"tight_{clip['clip_id']}_f{fj:04d}.png"), ann)

            row["tight_coverage"] = found2 / n2 if n2 else 0.0
            row["tight_torso_px"] = float(np.mean(torso2)) if torso2 else 0.0
            row["tight_visibility"] = float(np.mean(vis2)) if vis2 else 0.0

        rows.append(row)
        print(f"  [{ci}/{len(clips)}] {clip['clip_id']} {clip['outcome']:12s} "
              f"tracks={row['n_tracks']:>2d} elig={row['n_eligible']} "
              f"cov={row['selected_coverage']:.2f} torso={row['batter_torso_px']:.0f}px "
              f"runner_up_cov={row['runner_up_coverage']:.2f}"
              + (f" | tight cov={row.get('tight_coverage', 0):.2f} "
                 f"torso={row.get('tight_torso_px', 0):.0f}px" if args.two_pass else ""))

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)

    cov = float(np.mean([r["selected_coverage"] for r in rows]))
    torso = float(np.mean([r["batter_torso_px"] for r in rows]))
    vis = float(np.mean([r["mean_visibility"] for r in rows]))
    sep = float(np.mean([r["selected_coverage"] - r["runner_up_coverage"] for r in rows]))
    print(f"\n--- aggregate over {len(rows)} clips ---")
    print(f"selected-track coverage  {cov:.3f}   (tallest-pose rule was 0.858 but on the wrong subject)")
    print(f"torso height             {torso:.0f} px   (pitching 80-120px; full frame was 71px)")
    print(f"mean visibility          {vis:.3f}")
    print(f"coverage margin over runner-up    {sep:.3f}   (larger = less ambiguous selection)")
    if args.two_pass:
        tcov = float(np.mean([r.get("tight_coverage", 0.0) for r in rows]))
        ttor = float(np.mean([r.get("tight_torso_px", 0.0) for r in rows]))
        tvis = float(np.mean([r.get("tight_visibility", 0.0) for r in rows]))
        print(f"\n--- second pass, tight single-person crop ---")
        print(f"coverage                 {tcov:.3f}   (plate crop was {cov:.3f})")
        print(f"torso height             {ttor:.0f} px")
        print(f"mean visibility          {tvis:.3f}")
        cov, torso = tcov, ttor
    print(f"\ngate (coverage > 0.85 and torso > 80px): "
          f"{'PASS' if cov > 0.85 and torso > 80 else 'FAIL'}")
    print(f"annotated frames: {frames_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print("The numbers cannot confirm the SELECTED skeleton is the batter. Inspect frames")
    print("spanning the swing before treating this as a pass.")


if __name__ == "__main__":
    main()
