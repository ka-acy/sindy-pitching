#!/usr/bin/env python3
"""Resolve Stage B baseball clip IDs to locally available media files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root.",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=Path("/mnt/e/MML_RESEARCH"),
        help="Local baseball media root.",
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


def build_media_index(media_root: Path) -> dict[str, Path]:
    roots = [
        media_root / "SEGMENTED" / "segmented",
        media_root / "CONTINUOUS" / "continuous",
        media_root / "CONTINUOUS",
        media_root / "SEGMENTED",
    ]
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.mp4"):
            index.setdefault(path.stem, path)
    return index


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    media_root = args.media_root.resolve()

    clip_manifest = read_csv(project_root / "data" / "baseball" / "metadata" / "clip_manifest.csv")
    smoke_rows = read_csv(project_root / "data" / "baseball" / "splits" / "smoke_test.csv")
    media_index = build_media_index(media_root)

    resolved_rows: list[dict[str, Any]] = []
    for row in clip_manifest:
        local_path = media_index.get(row["clip_id"])
        resolved_rows.append(
            {
                "clip_id": row["clip_id"],
                "task_type": row["task_type"],
                "source_name": row["source_name"],
                "local_media_path": local_path.as_posix() if local_path else "",
                "local_media_exists": bool(local_path),
            }
        )

    write_csv(
        project_root / "data" / "baseball" / "metadata" / "local_media_manifest.csv",
        ["clip_id", "task_type", "source_name", "local_media_path", "local_media_exists"],
        resolved_rows,
    )

    smoke_lookup = {row["clip_id"] for row in smoke_rows}
    smoke_resolved = [row for row in resolved_rows if row["clip_id"] in smoke_lookup]
    write_csv(
        project_root / "data" / "baseball" / "splits" / "smoke_media_targets.csv",
        ["clip_id", "task_type", "source_name", "local_media_path", "local_media_exists"],
        smoke_resolved,
    )

    smoke_with_media = [row for row in smoke_resolved if row["local_media_exists"]]
    write_csv(
        project_root / "data" / "baseball" / "splits" / "smoke_test_local.csv",
        ["clip_id", "task_type", "source_name", "local_media_path", "local_media_exists"],
        smoke_with_media,
    )

    print(f"resolved clips: {sum(1 for row in resolved_rows if row['local_media_exists'])}")
    print(f"smoke clips with media: {sum(1 for row in smoke_resolved if row['local_media_exists'])}")


if __name__ == "__main__":
    main()
