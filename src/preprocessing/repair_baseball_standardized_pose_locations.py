#!/usr/bin/env python3
"""Move misplaced Stage B standardized-pose payloads into their registered source directories."""

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
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    clip_manifest = {row["clip_id"]: row for row in read_csv(project_root / "data" / "baseball" / "metadata" / "clip_manifest.csv")}

    source_root = project_root / "data" / "baseball" / "standardized_pose" / "example_source"
    moved = 0
    for path in sorted(source_root.glob("*.npz")):
        if path.stem == "example_clip":
            continue
        if path.stem not in clip_manifest:
            continue

        payload = np.load(path)
        metadata = json.loads(str(payload["metadata"]))
        clip_row = clip_manifest[path.stem]
        target_dir = project_root / "data" / "baseball" / "standardized_pose" / clip_row["source_name"]
        target_path = target_dir / path.name
        target_dir.mkdir(parents=True, exist_ok=True)

        metadata["source_name"] = clip_row["source_name"]
        np.savez_compressed(
            target_path,
            coordinates=np.asarray(payload["coordinates"]),
            joint_names=np.asarray(payload["joint_names"]),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        path.unlink()
        moved += 1

    print(f"moved payloads: {moved}")


if __name__ == "__main__":
    main()
