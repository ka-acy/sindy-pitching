#!/usr/bin/env python3
"""Apply handedness overrides to Stage B baseball metadata and pose payloads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root.",
    )
    parser.add_argument(
        "--overrides-csv",
        type=Path,
        default=Path("data/baseball/metadata/handedness_overrides.csv"),
        help="CSV with clip_id,handedness rows relative to project root.",
    )
    parser.add_argument(
        "--clip-manifest",
        type=Path,
        default=Path("data/baseball/metadata/clip_manifest.csv"),
        help="Baseball clip manifest relative to project root.",
    )
    parser.add_argument(
        "--pose-manifest",
        type=Path,
        default=Path("data/baseball/standardized_pose/manifest.csv"),
        help="Registered Stage B pose manifest relative to project root.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def canonical_handedness(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"R", "RIGHT"}:
        return "R"
    if normalized in {"L", "LEFT"}:
        return "L"
    raise ValueError(f"Unsupported handedness value: {value!r}")


def load_overrides(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    overrides: dict[str, str] = {}
    for row in rows:
        clip_id = row["clip_id"].strip()
        handedness = row.get("handedness", "").strip()
        if not clip_id or not handedness:
            continue
        overrides[clip_id] = canonical_handedness(handedness)
    return overrides


def update_pose_payload(path: Path, handedness: str) -> None:
    payload = np.load(path)
    metadata = json.loads(str(payload["metadata"]))
    metadata["handedness"] = handedness
    np.savez_compressed(
        path,
        coordinates=np.asarray(payload["coordinates"]),
        joint_names=np.asarray(payload["joint_names"]),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    overrides_path = project_root / args.overrides_csv
    if not overrides_path.exists():
        raise FileNotFoundError(f"Missing overrides CSV: {overrides_path}")

    overrides = load_overrides(overrides_path)
    if not overrides:
        print("applied clip manifest rows: 0")
        print("updated standardized-pose payloads: 0")
        return

    clip_manifest_path = project_root / args.clip_manifest
    clip_rows = read_csv(clip_manifest_path)
    clip_fieldnames = list(clip_rows[0].keys()) if clip_rows else []
    applied_manifest = 0
    for row in clip_rows:
        clip_id = row["clip_id"]
        if clip_id in overrides:
            row["handedness"] = overrides[clip_id]
            applied_manifest += 1
    write_csv(clip_manifest_path, clip_fieldnames, clip_rows)

    pose_manifest_path = project_root / args.pose_manifest
    updated_payloads = 0
    if pose_manifest_path.exists():
        pose_rows = read_csv(pose_manifest_path)
        for row in pose_rows:
            clip_id = row["clip_id"]
            if clip_id not in overrides:
                continue
            payload_path = project_root / row["filepath"]
            if not payload_path.exists():
                continue
            update_pose_payload(payload_path, overrides[clip_id])
            updated_payloads += 1

    print(f"applied clip manifest rows: {applied_manifest}")
    print(f"updated standardized-pose payloads: {updated_payloads}")


if __name__ == "__main__":
    main()
