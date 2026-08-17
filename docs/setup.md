# Setup Notes

## Current State

The repository raw data layout is initialized under `data/utd_mhad/`.

Available generated artifacts:
- `data/utd_mhad/metadata/index.csv`
- `data/utd_mhad/splits/proxy_subset.csv`
- `data/utd_mhad/splits/smoke_test.csv`
- `data/utd_mhad/processed/smoke_test/*.npz`
- `data/utd_mhad/processed/smoke_test/manifest.csv`
- `data/utd_mhad/processed/smoke_test/features/*.npz`
- `data/utd_mhad/processed/smoke_test/features/manifest.csv`
- `results/preprocessing_logs/smoke_test_run_log.jsonl`
- `results/preprocessing_logs/smoke_test_feature_run_log.jsonl`
- `results/equations/proxy_subset/*`
- `results/evaluation/proxy_subset/*`
- `results/plots/proxy_subset/*`

The manifest generation logic lives in:
- `src/preprocessing/build_utd_mhad_manifests.py`
- `src/preprocessing/process_utd_mhad_skeletons.py`
- `src/preprocessing/extract_utd_mhad_features.py`
- `src/preprocessing/fit_utd_mhad_sindy.py`
- `src/preprocessing/evaluate_utd_mhad_sindy.py`

Current manifest behavior:
- reads MATLAB 5 headers directly for skeleton, depth, and inertial files
- reads AVI headers directly for RGB files
- populates `num_frames` for every modality
- populates `fps` from AVI metadata for RGB and from UTD-MHAD modality defaults for skeleton/depth/inertial

## Python Environment

The project now has a working `.venv` with the required packages installed:
- `numpy`
- `pandas`
- `scipy`
- `pysindy`
- `matplotlib`
- `pyyaml`

Stage A interpretation guidance lives in:
- `docs/stage_a_validation.md`

Stage B ingestion guidance lives in:
- `docs/stage_b_input_contract.md`

The MLB-YouTube metadata can be normalized into the Stage B clip manifest with:

```bash
./.venv/bin/python src/preprocessing/build_mlb_youtube_manifest.py
```

## Rebuilding Manifests

```bash
python3 src/preprocessing/build_utd_mhad_manifests.py
```

This script uses only the Python standard library and does not require SciPy.

Stage B metadata manifest output:
- `data/baseball/metadata/clip_manifest.csv`
- `data/baseball/splits/smoke_test.csv`

Current MLB-YouTube Stage B status:
- the archive is unpacked under `data/baseball/raw/mlb-youtube-master/`
- the JSON metadata has been converted into a Stage B clip manifest
- actual standardized pose clips still need to be produced by a baseball adapter / pose-extraction step

Stage B helper scripts:
- `src/preprocessing/build_mlb_youtube_manifest.py`
- `src/preprocessing/build_baseball_pose_jobs.py`
- `src/preprocessing/create_baseball_pose_example.py`
- `src/preprocessing/export_mediapipe_pose.py`
- `src/preprocessing/populate_baseball_pose_payload.py`
- `src/preprocessing/register_baseball_standardized_pose.py`
- `src/preprocessing/resolve_baseball_local_media.py`

If local MLB media already exists outside the repo, resolve it into Stage B
manifests with:

```bash
./.venv/bin/python src/preprocessing/resolve_baseball_local_media.py --media-root /mnt/e/MML_RESEARCH
```

This writes:
- `data/baseball/metadata/local_media_manifest.csv`
- `data/baseball/splits/smoke_media_targets.csv`
- `data/baseball/splits/smoke_test_local.csv`

Build a pose-extraction job manifest from the locally backed clips with:

```bash
./.venv/bin/python src/preprocessing/build_baseball_pose_jobs.py
```

This writes:
- `data/baseball/metadata/pose_jobs.csv`

Current default behavior:
- uses `data/baseball/splits/smoke_test_local.csv` as input
- assumes video-based pose extraction will produce 2D outputs first
- records the intended standardized-pose `.npz` path for each local clip

Scaffold a clip-specific standardized-pose template from the pose-job manifest:

```bash
./.venv/bin/python src/preprocessing/create_baseball_pose_example.py --clip-id 001K4FFG09M0
```

Current template behavior:
- reads `data/baseball/metadata/pose_jobs.csv` when the `clip_id` is present
- writes the template to the clip's intended `pose_output_path`
- infers a provisional frame count from the annotated segment duration at `30.0` fps
- leaves zero-filled coordinates that must be replaced before registration

After a pose tool produces coordinates, populate the clip payload with:

```bash
./.venv/bin/python src/preprocessing/populate_baseball_pose_payload.py \
  --clip-id 001K4FFG09M0 \
  --input-path /path/to/pose_output.npz
```

Current import behavior:
- accepts `.npy` or `.npz` coordinate inputs
- uses `joint_names` from the input `.npz` or a `--joint-names` override
- reorders source joints into the preferred 15-joint Stage B schema when possible
- updates the target `.npz` in place so it can be registered immediately

If MediaPipe Pose is installed in the project venv, export landmarks directly
from the local clip with:

```bash
./.venv/bin/python src/preprocessing/export_mediapipe_pose.py \
  --input-video /mnt/e/MML_RESEARCH/SEGMENTED/segmented/001K4FFG09M0.mp4 \
  --output-path data/baseball/external_pose/001K4FFG09M0_mediapipe_pose.npz
```

The Stage B importer recognizes standard MediaPipe's 33-landmark ordering and
derives the preferred 15-joint schema automatically.

Preprocess registered baseball standardized-pose clips with:

```bash
./.venv/bin/python src/preprocessing/process_baseball_standardized_pose.py --split smoke_test
```

This writes:
- `data/baseball/processed/smoke_test/*.npz`
- `data/baseball/processed/smoke_test/manifest.csv`
- `results/preprocessing_logs/baseball_smoke_test_run_log.jsonl`

Extract features from the processed baseball clips with:

```bash
./.venv/bin/python src/preprocessing/extract_baseball_features.py --split smoke_test
```

This writes:
- `data/baseball/processed/smoke_test/features/*.npz`
- `data/baseball/processed/smoke_test/features/manifest.csv`
- `results/preprocessing_logs/baseball_smoke_test_feature_run_log.jsonl`

Fit baseline SINDy models on the baseball feature clips with:

```bash
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py --split smoke_test --task-types pitching
```

This writes task-specific outputs under:
- `results/equations/baseball/smoke_test/<task_type>/`

Build a larger local pitching train/test split with:

```bash
./.venv/bin/python src/preprocessing/build_baseball_pitching_splits.py
```

Default output:
- `data/baseball/splits/pitching_local_pool.csv` with 30 local segmented pitching clips
- `data/baseball/splits/pitching_local_train.csv` with 24 clips
- `data/baseball/splits/pitching_local_test.csv` with 6 clips

To create a handedness-annotation template for the current local pitching pool:

```bash
./.venv/bin/python src/preprocessing/export_baseball_handedness_template.py
```

This writes:
- `data/baseball/metadata/handedness_overrides.csv`

To generate a local review packet with preview images for those clips:

```bash
./.venv/bin/python src/preprocessing/export_baseball_handedness_review.py
```

This writes:
- `results/review/baseball_handedness/index.html`
- `results/review/baseball_handedness/review_manifest.csv`
- `results/review/baseball_handedness/previews/*.jpg`

Current review behavior:
- reads `data/baseball/splits/pitching_local_pool.csv`
- extracts four preview frames per local clip with OpenCV
- builds an HTML index you can review while filling `handedness_overrides.csv`

Fill the `handedness` column with `R` or `L`, then apply the overrides to the
Stage B clip manifest and any registered pose payload metadata with:

```bash
./.venv/bin/python src/preprocessing/apply_baseball_handedness_overrides.py
```

Current behavior:
- updates `data/baseball/metadata/clip_manifest.csv`
- updates `handedness` inside matching `data/baseball/standardized_pose/**/*.npz` payload metadata
- leaves clips unchanged if no override row is provided

To rerun the Stage B baseball pipeline after updating handedness labels:

```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py
```

Current refresh behavior:
- applies handedness overrides unless `--skip-overrides` is used
- rebuilds the current `pitching_local_*` split CSVs unless `--skip-splits` is used
- reruns baseball preprocessing, feature extraction, train fit, and held-out evaluation
- accepts `--config` for config-specific feature extraction, fitting, and evaluation
- supports `--dry-run` to print the command sequence without executing it

Current two-track baseball refresh entry points:

```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline_unaligned.py
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline_aligned.py
```

Current track intent:
- `refresh_baseball_pipeline_unaligned.py` uses `config/preprocessing.baseball.unaligned.yaml`
- `refresh_baseball_pipeline_aligned.py` uses `config/preprocessing.baseball.aligned.yaml`
- `unaligned` is the derivative-focused discovery track
- `aligned` is the reconstruction-focused phase-normalized track
- both wrappers accept the same flags as `refresh_baseball_pipeline.py`, including `--dry-run`

To run MediaPipe over an entire Stage B job CSV from your shell, use:

```bash
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python src/preprocessing/run_baseball_mediapipe_batch.py \
  --jobs-csv data/baseball/metadata/pitching_local_train_pose_jobs.csv \
  --skip-existing
```

After the batch export/import step, refresh the registered manifest with:

```bash
./.venv/bin/python src/preprocessing/register_baseball_standardized_pose.py
```

If earlier batch runs wrote payloads into `example_source/`, repair them with:

```bash
./.venv/bin/python src/preprocessing/repair_baseball_standardized_pose_locations.py
```

Held-out baseball evaluation after train/test preprocessing and feature extraction:

```bash
./.venv/bin/python src/preprocessing/evaluate_baseball_sindy.py --train-split train --eval-split test --task-types pitching
```

Current local-media note:
- the `E:\MML_RESEARCH` tree resolves many segmented MLB clips
- the current Stage B smoke set has local media for the three segmented pitching targets
- the batting, negative, and broadcast-context smoke rows are not yet backed by local files in that tree

## Known Gaps

- The modality FPS values for `skeleton`, `depth`, and `inertial` are dataset defaults rather than per-file embedded metadata.

## Skeleton Preprocessing

Run the smoke-test preprocessing pass:

```bash
./.venv/bin/python src/preprocessing/process_utd_mhad_skeletons.py --split smoke_test
```

Run the full proxy subset:

```bash
./.venv/bin/python src/preprocessing/process_utd_mhad_skeletons.py --split proxy_subset
```

Current preprocessing behavior:
- maps raw UTD-MHAD skeletons into a 15-joint standardized schema
- centers each frame at the pelvis proxy (`hip_center`, derived from left/right hip)
- optionally supports orientation and scale normalization through config
- interpolates short missing-value gaps
- smooths trajectories with Savitzky-Golay filtering
- trims low-motion leading and trailing frames
- writes processed `.npz` files plus a manifest and JSONL run log

Stage A interpretation:
- these outputs validate joint capture, temporal conditioning, and reproducible trajectory handling
- they are not evidence that UTD-MHAD motions behave like real baseball pitching or batting

## Feature Extraction And Derivatives

Run the smoke-test feature pass:

```bash
./.venv/bin/python src/preprocessing/extract_utd_mhad_features.py --split smoke_test
```

Run the full proxy subset:

```bash
./.venv/bin/python src/preprocessing/extract_utd_mhad_features.py --split proxy_subset
```

Current feature behavior:
- builds a 10-variable state vector from processed skeleton clips
- stores both `X(t)` and `dX/dt` in compressed `.npz` files
- uses Savitzky-Golay differentiation with explicit `dt`
- writes feature manifests and JSONL run logs for reproducibility

Stage A interpretation:
- the state vectors are used to test whether captured joints can be transformed into stable matrices for sparse modeling
- feature quality here should be judged by numerical stability and consistency, not baseball specificity

## SINDy Modeling

Train baseline models:

```bash
./.venv/bin/python src/preprocessing/fit_utd_mhad_sindy.py --split proxy_subset
```

Current modeling behavior:
- trains separate PySINDy models for `batting_like` and `pitching_like`
- uses sparse regression to produce equations and coefficient matrices from captured joint features
- stores equations, coefficients, config, metrics, scaler artifacts, and pickled models under `results/equations/`

Stage A interpretation:
- the primary outputs are sparse coefficient matrices and equation structures
- these should be treated as capture-validation artifacts, not baseball-mechanics conclusions

## Evaluation

Run baseline evaluation:

```bash
./.venv/bin/python src/preprocessing/evaluate_utd_mhad_sindy.py --split proxy_subset
```

Current evaluation behavior:
- scores derivative prediction on same-class and cross-class trajectories
- runs short-horizon reconstruction rollouts
- records per-trial reconstruction stability and errors
- writes summary CSVs under `results/evaluation/` and sample plots under `results/plots/`

Stage A interpretation:
- evaluation is used to compare preprocessing/model configurations and judge pipeline robustness
- weak reconstruction on UTD-MHAD does not by itself invalidate the baseball-domain approach
