#!/usr/bin/env python3
"""Build dataset manifests for the UTD-MHAD raw data layout.

This script scans the extracted UTD-MHAD raw dataset and writes:
- data/utd_mhad/metadata/index.csv
- data/utd_mhad/splits/proxy_subset.csv
- data/utd_mhad/splits/smoke_test.csv

The generated index intentionally excludes:
- Apple resource-fork artifacts under __MACOSX/
- sample files shipped inside Sample_Code/

The manifest builder reads lightweight MAT/AVI metadata directly so the index
includes `fps` and `num_frames` without external dependencies.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import zlib
from pathlib import Path


ACTION_NAMES = {
    1: "swipt_left",
    2: "swipt_right",
    3: "wave",
    4: "clap",
    5: "throw",
    6: "arm_cross",
    7: "basketball_shoot",
    8: "draw_x",
    9: "draw_circle_CW",
    10: "draw_circle_CCW",
    11: "draw_triangle",
    12: "bowling",
    13: "boxing",
    14: "baseball_swing",
    15: "tennis_swing",
    16: "arm_curl",
    17: "tennis_serve",
    18: "push",
    19: "knock",
    20: "catch",
    21: "pickup_throw",
    22: "jog",
    23: "walk",
    24: "sit2stand",
    25: "stand2sit",
    26: "lunge",
    27: "squat",
}

PROXY_ACTIONS = {
    "batting_like": {14, 15},
    "pitching_like": {5, 12, 17, 21},
}

FILE_PATTERNS = {
    "skeleton": ("Skeleton", re.compile(r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_skeleton\.mat$")),
    "depth": ("Depth", re.compile(r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_depth\.mat$")),
    "inertial": ("Inertial", re.compile(r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_inertial\.mat$")),
    "rgb": ("", re.compile(r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_color\.avi$")),
}

MATLAB_MI_INT32 = 5
MATLAB_MI_MATRIX = 14
MATLAB_MI_COMPRESSED = 15

# UTD-MHAD stores frame counts in MAT array dimensions but not sampling rates.
# These defaults are used for the non-AVI modalities so every manifest row has
# an explicit time base.
MODALITY_DEFAULT_FPS = {
    "skeleton": 20,
    "depth": 20,
    "inertial": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/utd_mhad/raw",
    )
    parser.add_argument(
        "--smoke-per-label",
        type=int,
        default=6,
        help="Number of proxy trials per label to include in smoke_test.csv",
    )
    return parser.parse_args()


def read_matlab_tag(buffer: bytes, offset: int) -> tuple[int, int, int, int]:
    data_type, num_bytes = struct.unpack_from("<II", buffer, offset)
    if data_type >> 16:
        num_bytes = data_type >> 16
        data_type &= 0xFFFF
        data_offset = offset + 4
        next_offset = offset + 8
    else:
        data_offset = offset + 8
        next_offset = data_offset + ((num_bytes + 7) // 8) * 8
    return data_type, num_bytes, data_offset, next_offset


def extract_matlab_dimensions(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        raw = handle.read()

    if not raw.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError(f"Unsupported MAT file header: {path}")

    payload = raw[128:]
    data_type, num_bytes, data_offset, _ = read_matlab_tag(payload, 0)
    if data_type == MATLAB_MI_COMPRESSED:
        payload = zlib.decompress(payload[data_offset : data_offset + num_bytes])
        data_type, num_bytes, data_offset, _ = read_matlab_tag(payload, 0)

    if data_type != MATLAB_MI_MATRIX:
        raise ValueError(f"Expected miMATRIX payload in {path}")

    inner_offset = read_matlab_tag(payload, data_offset)[3]
    dim_type, dim_bytes, dim_offset, _ = read_matlab_tag(payload, inner_offset)
    if dim_type != MATLAB_MI_INT32:
        raise ValueError(f"Expected dimension element in {path}")

    dim_count = dim_bytes // 4
    return struct.unpack_from(f"<{dim_count}i", payload, dim_offset)


def infer_num_frames(path: Path, modality: str) -> int:
    dimensions = extract_matlab_dimensions(path)
    if modality in {"skeleton", "depth"}:
        return int(dimensions[-1])
    if modality == "inertial":
        return int(dimensions[0])
    raise ValueError(f"Unsupported MAT modality: {modality}")


def extract_avi_metadata(path: Path) -> tuple[float, int]:
    with path.open("rb") as handle:
        header = handle.read(256)

    avih_offset = header.find(b"avih")
    if avih_offset == -1:
        raise ValueError(f"AVI header chunk not found: {path}")

    fields = struct.unpack_from("<IIIIIIIIII", header, avih_offset + 8)
    microseconds_per_frame = fields[0]
    total_frames = fields[4]
    fps = round(1_000_000 / microseconds_per_frame, 6)
    return fps, total_frames


def infer_metadata(path: Path, modality: str) -> tuple[float | int, int]:
    if modality == "rgb":
        fps, num_frames = extract_avi_metadata(path)
        return fps, num_frames

    fps = MODALITY_DEFAULT_FPS[modality]
    num_frames = infer_num_frames(path, modality)
    return fps, num_frames


def collect_rows(project_root: Path) -> list[dict[str, object]]:
    raw_root = project_root / "data" / "utd_mhad" / "raw"
    rows: list[dict[str, object]] = []

    for modality, (subdir, pattern) in FILE_PATTERNS.items():
        search_root = raw_root / subdir if subdir else raw_root
        for path in sorted(search_root.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("._"):
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            action_id = int(match.group("action"))
            subject_id = int(match.group("subject"))
            trial_id = int(match.group("trial"))
            fps, num_frames = infer_metadata(path, modality)
            rows.append(
                {
                    "subject_id": subject_id,
                    "action_id": action_id,
                    "action_name": ACTION_NAMES[action_id],
                    "trial_id": trial_id,
                    "modality": modality,
                    "filepath": path.relative_to(project_root).as_posix(),
                    "fps": fps,
                    "num_frames": num_frames,
                }
            )

    rows.sort(key=lambda row: (row["action_id"], row["subject_id"], row["trial_id"], row["modality"]))
    return rows


def collect_proxy_rows(rows: list[dict[str, object]]) -> list[dict[str, int | str]]:
    skeleton_trials = [
        row for row in rows if row["modality"] == "skeleton"
    ]
    proxy_rows: list[dict[str, int | str]] = []

    for label, actions in PROXY_ACTIONS.items():
        for row in skeleton_trials:
            if row["action_id"] not in actions:
                continue
            proxy_rows.append(
                {
                    "subset_label": label,
                    "subject_id": int(row["subject_id"]),
                    "action_id": int(row["action_id"]),
                    "trial_id": int(row["trial_id"]),
                }
            )

    proxy_rows.sort(key=lambda row: (row["subset_label"], row["action_id"], row["subject_id"], row["trial_id"]))
    return proxy_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()

    rows = collect_rows(project_root)
    proxy_rows = collect_proxy_rows(rows)

    smoke_rows: list[dict[str, int | str]] = []
    for label in ("batting_like", "pitching_like"):
        label_rows = [row for row in proxy_rows if row["subset_label"] == label]
        smoke_rows.extend(label_rows[: args.smoke_per_label])

    metadata_root = project_root / "data" / "utd_mhad" / "metadata"
    splits_root = project_root / "data" / "utd_mhad" / "splits"

    write_csv(
        metadata_root / "index.csv",
        ["subject_id", "action_id", "action_name", "trial_id", "modality", "filepath", "fps", "num_frames"],
        rows,
    )
    write_csv(
        splits_root / "proxy_subset.csv",
        ["subset_label", "subject_id", "action_id", "trial_id"],
        proxy_rows,
    )
    write_csv(
        splits_root / "smoke_test.csv",
        ["subset_label", "subject_id", "action_id", "trial_id"],
        smoke_rows,
    )

    print(f"index rows: {len(rows)}")
    print(f"proxy rows: {len(proxy_rows)}")
    print(f"smoke rows: {len(smoke_rows)}")


if __name__ == "__main__":
    main()
