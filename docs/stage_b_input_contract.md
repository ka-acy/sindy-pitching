# Stage B Input Contract

## Purpose

This document defines the input boundary for baseball-specific data.

The goal is to ensure that any baseball source:
- MLB broadcast video
- biomechanics lab video
- motion-capture exports
- third-party pose-estimation outputs

can be adapted into one consistent format before entering the existing Stage A
pipeline components.

The downstream pipeline should not need to know where the baseball data came
from. It should only consume standardized pose clips and manifests.

## Design Rule

Stage B should replace only the pose-input layer.

Everything after pose standardization should remain reusable:
- preprocessing
- feature extraction
- derivative estimation
- SINDy modeling
- evaluation

## Recommended Folder Layout

Use a baseball-specific root parallel to `data/utd_mhad/`:

```text
data/
  baseball/
    raw/
      video/
      mocap/
      pose_exports/
    metadata/
      clip_manifest.csv
    standardized_pose/
      <source_name>/
        *.npz
      manifest.csv
    processed/
      <split_name>/
    splits/
      train.csv
      test.csv
      smoke_test.csv
```

## Required Manifest: `data/baseball/metadata/clip_manifest.csv`

Each source clip should have one row with at least:

- `clip_id`
- `source_name`
- `task_type`
- `filepath`
- `modality`
- `fps`
- `num_frames`
- `subject_id`
- `handedness`
- `view_type`
- `event_start_frame`
- `event_end_frame`
- `notes`

### Field Meanings

- `clip_id`: stable identifier used across all downstream files
- `source_name`: dataset or collection name, e.g. `mlb_broadcast`
- `task_type`: `pitching`, `batting`, or another explicitly documented label
- `filepath`: path to the original raw asset
- `modality`: `video`, `mocap`, `pose_2d`, or `pose_3d`
- `fps`: source frame rate
- `num_frames`: total source frame count
- `subject_id`: player identifier if available, otherwise a clip-level placeholder
- `handedness`: `R`, `L`, or blank if unknown
- `view_type`: e.g. `broadcast_centerfield`, `side`, `lab`, `unknown`
- `event_start_frame` / `event_end_frame`: optional coarse action window

## Standardized Pose Payload

The adapter layer should output one `.npz` per clip under:

`data/baseball/standardized_pose/<source_name>/<clip_id>.npz`

Each payload should contain:

- `coordinates`
- `joint_names`
- `metadata`

### `coordinates`

Array shape:

- preferred: `(time, joints, 3)`
- fallback for 2D-only inputs: `(time, joints, 2)`

Coordinates should already be in a consistent joint order for the clip.

### `joint_names`

A string array listing the joints in `coordinates`.

Preferred Stage B joint target:
- `head`
- `neck`
- `spine`
- `left_shoulder`
- `right_shoulder`
- `left_elbow`
- `right_elbow`
- `left_wrist`
- `right_wrist`
- `left_hip`
- `right_hip`
- `left_knee`
- `right_knee`
- `left_ankle`
- `right_ankle`

This matches the Stage A standardized schema and avoids downstream redesign.

### `metadata`

JSON-serializable metadata string with at least:

- `clip_id`
- `source_name`
- `task_type`
- `fps`
- `dt`
- `coordinate_dims`
- `joint_schema`
- `pose_source`
- `subject_id`
- `handedness`
- `view_type`
- `confidence_available`

Recommended optional fields:

- `event_start_frame`
- `event_end_frame`
- `pose_model_name`
- `pose_model_version`
- `tracking_notes`
- `quality_flags`

## Standardized Pose Manifest

Also create:

`data/baseball/standardized_pose/manifest.csv`

Required columns:
- `clip_id`
- `source_name`
- `task_type`
- `filepath`
- `fps`
- `dt`
- `num_frames`
- `num_joints`
- `coordinate_dims`

This manifest should be the Stage B equivalent of the Stage A processed input list.

## Adapter Requirements

Any Stage B adapter script should:
- read raw baseball assets or external pose outputs
- map source joints to the standardized 15-joint schema
- preserve `fps` and `dt`
- document whether the input is 2D or 3D
- write `.npz` payloads and `manifest.csv`
- never mix schema variants within one manifest

In this repository, the current helper boundary is:
- `src/preprocessing/build_baseball_pose_jobs.py`
- `src/preprocessing/create_baseball_pose_example.py`
- `src/preprocessing/export_mediapipe_pose.py`
- `src/preprocessing/populate_baseball_pose_payload.py`
- `src/preprocessing/register_baseball_standardized_pose.py`
- `src/preprocessing/resolve_baseball_local_media.py`

The registration step validates that standardized pose payloads match the
contract and, by default, requires `clip_id` to already exist in
`data/baseball/metadata/clip_manifest.csv`.

If raw video already exists outside the repo, resolve those files into a
local-media manifest before pose extraction. This keeps the Stage B contract
stable while allowing large video assets to remain in place.

The pose-extraction queue can then be materialized with
`src/preprocessing/build_baseball_pose_jobs.py`, which joins clip metadata to
local video paths and assigns the target standardized-pose output paths.

To scaffold one pending payload before pose extraction is automated, use
`src/preprocessing/create_baseball_pose_example.py --clip-id <clip_id>`. When
the clip exists in `pose_jobs.csv`, the script writes a clip-specific template
with the correct target path and metadata shell.

Once external coordinates exist, import them with
`src/preprocessing/populate_baseball_pose_payload.py`. That utility updates the
clip payload in place and reorders joints into the preferred 15-joint schema
when the source joint names are available.

For MediaPipe Pose specifically, `src/preprocessing/export_mediapipe_pose.py`
exports the standard 33 landmarks into an `.npz` payload that can be consumed
directly by `populate_baseball_pose_payload.py`.

After one or more Stage B payloads are registered, preprocess them with
`src/preprocessing/process_baseball_standardized_pose.py` to generate the
baseball equivalent of the Stage A processed-trajectory manifest.

## 2D vs 3D Rule

If only 2D pose is available:
- still write the same payload structure
- set `coordinate_dims = 2`
- document the camera/view assumptions in metadata
- avoid pretending the clip is 3D

If true 3D pose or mocap is available:
- set `coordinate_dims = 3`
- preserve the native 3D interpretation in metadata

The downstream preprocessing step may need different normalization parameters,
but the file contract should remain the same.

## Minimal Stage B Success Condition

Stage B ingestion is ready when you can take one baseball clip and produce:
- one row in `clip_manifest.csv`
- one standardized pose `.npz`
- one row in `standardized_pose/manifest.csv`

At that point, the existing preprocessing and feature pipeline can be adapted to
consume the baseball standardized-pose manifest instead of the UTD-MHAD skeleton
manifest.
