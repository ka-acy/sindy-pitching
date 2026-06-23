# Baseball Second-Order State Vector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the Stage B baseball state vector from 10-D (positions/angles + one velocity magnitude) to a 16-D strict second-order state (8 positions + 8 paired rates), to fix the negative test reconstruction R² and the zero-collapsed equations.

**Architecture:** Add a `second_order` code path to `extract_baseball_features.py` that, per position column, runs a single Savitzky–Golay fit queried at derivative orders 0/1/2 to assemble `X = [positions, velocities]` and `X_dot = [velocities, accelerations]`. The downstream fit/evaluate scripts already consume `feature_names` + matrix shapes generically, so they are not modified. New YAML configs select the path and the 8 position names. Everything is validated against a regenerated first-order baseline on the restored 111/29 pool.

**Tech Stack:** Python 3.12, NumPy, SciPy (`savgol_filter`), PySINDy, scikit-learn, PyYAML. Run everything with the repo venv at `./.venv/bin/python`.

## Global Constraints

- Python interpreter: always `./.venv/bin/python` (never bare `python`).
- Source video at `/mnt/e/MML_RESEARCH/SEGMENTED/segmented/` is UNAVAILABLE (drive unmounted). All work must use cached pose data under `data/baseball/standardized_pose/` and cached manifests under `data/baseball/metadata/`. Do not invoke any script that re-extracts pose from video (e.g. `run_baseball_mediapipe_batch.py`).
- State-vector / SINDy-config scope ONLY. Do not change the evaluation harness, pose layer, or pipeline orchestration logic beyond pointing it at new configs. (Per spec "Out of scope (v1)".)
- No unit-test suite exists and none is added. Validation is by executable checks (shape asserts, by-construction identity check, identity gate) and metric movement, per the spec.
- `extract_utd_mhad_features.py` (shared Stage A code) must NOT be modified. Stage A keeps its current logic.
- Commit after every task. End commit messages with the project's Co-Authored-By trailer for the current model.
- Second-order target state (8 positions): `shoulder_rotation_angle`, `hip_rotation_angle`, `dominant_elbow_flexion_angle`, `trail_elbow_flexion_angle`, `dominant_knee_flexion_angle`, `trail_knee_flexion_angle`, `lead_knee_lift`, `stride_ankle_separation`. Rates are these 8 names suffixed `_rate`.
- A/B baseline to beat (first-order `unaligned.handedness` on 111/29): regenerate it freshly in Task 1 rather than trusting the stale memory numbers (≈26 terms, test_recon_rmse≈0.72, test_recon_r²≈−2.77).

---

## File Structure

- `src/preprocessing/extract_baseball_features.py` — **modified.** Add `compute_second_order_state(...)` and a `second_order` branch in `main()`. This is the only source file that changes.
- `config/preprocessing.baseball.unaligned.second_order.yaml` — **created.** Single-shot second-order config.
- `config/preprocessing.baseball.unaligned.second_order.sweep.yaml` — **created.** Sweep variant.
- `docs/feature_definitions.md` — **rewritten.** Documents the 16-D second-order spec + dropped-feature rationale.
- `scripts/` verification — **none persisted.** Verification checks are one-off `./.venv/bin/python -c "..."` invocations shown inline in steps.
- Baseline snapshot dirs (created by Task 1):
  - `results/equations/baseball/train_first_order_baseline_2026-06-23/`
  - `results/evaluation/baseball/train_on_test_first_order_baseline_2026-06-23/`

---

## Task 0: Restore the 111/29 evaluation pool from cached pose

**Files:**
- Modify (data, not code): `data/baseball/splits/pitching_local_{pool,train,test}.csv` (regenerated)
- Read: `data/baseball/standardized_pose/manifest.csv`, `data/baseball/metadata/local_media_manifest.csv`, `data/baseball/metadata/clip_manifest.csv`
- Write (data): `data/baseball/processed/{train,test}/...`

**Interfaces:**
- Produces: a restored `data/baseball/processed/train` and `data/baseball/processed/test` containing ~111 and ~29 pitching clips' standardized-pose `.npz` + `manifest.csv`, all 6 pitch types present in both. Later tasks fit/evaluate on these.

- [ ] **Step 1: Confirm cached pose coverage**

Run: `./.venv/bin/python -c "import csv; rows=list(csv.DictReader(open('data/baseball/standardized_pose/manifest.csv'))); reg=[r for r in rows if r.get('registered_clip_id')=='True']; print('registered poses:', len(reg))"`
Expected: `registered poses: 140` (or close — must be ≥ ~140 to support the 111/29 split).

- [ ] **Step 2: Rebuild the stratified split from cache**

Run:
```bash
./.venv/bin/python src/preprocessing/build_baseball_pitching_splits.py --pool-size 140 --test-size 29
```
This intersects pitching clips ∈ `local_media_manifest` (cached `local_media_exists=True`) ∩ usable poses (`standardized_pose/manifest.csv`), then stratifies the test cut by `pitch_type`. No video access occurs.

- [ ] **Step 3: Verify split counts and pitch-type coverage**

Run:
```bash
./.venv/bin/python -c "
import csv
def load(p): return list(csv.DictReader(open(p)))
tr=load('data/baseball/splits/pitching_local_train.csv'); te=load('data/baseball/splits/pitching_local_test.csv')
from collections import Counter
print('train', len(tr), 'test', len(te))
print('train types', Counter(r['pitch_type'] for r in tr))
print('test types', Counter(r['pitch_type'] for r in te))
"
```
Expected: train ≈ 111, test ≈ 29 (exact numbers depend on stratification rounding — anything in train∈[105,120], test∈[25,33] is fine). All 6 pitch types (changeup, curveball, fastball, knucklecurve, sinker, slider) must appear in BOTH train and test. If a pitch type is missing from test, that is a stratification bug — stop and report.

- [ ] **Step 4: Process standardized pose into train/test from cache**

Run:
```bash
./.venv/bin/python src/preprocessing/process_baseball_standardized_pose.py --split train --task-type pitching
./.venv/bin/python src/preprocessing/process_baseball_standardized_pose.py --split test --task-type pitching
```

- [ ] **Step 5: Verify processed clip counts**

Run:
```bash
./.venv/bin/python -c "
import csv
for s in ('train','test'):
    n=len(list(csv.DictReader(open(f'data/baseball/processed/{s}/manifest.csv'))))
    print(s, n)
"
```
Expected: train and test counts match the split counts from Step 3 (minus any clip dropped for too few frames after trimming — log any dropped clips).

- [ ] **Step 6: Commit the restored pool**

```bash
git add data/baseball/splits/pitching_local_pool.csv data/baseball/splits/pitching_local_train.csv data/baseball/splits/pitching_local_test.csv
git commit -m "data: restore 111/29 baseball pitching pool from cached pose"
```

---

## Task 1: Regenerate and snapshot the first-order baseline (A/B reference)

**Files:**
- Read: `config/preprocessing.baseball.unaligned.handedness.yaml`
- Write: `results/equations/baseball/train/...`, `results/evaluation/baseball/train_on_test/...`
- Create (snapshot copies): `results/equations/baseball/train_first_order_baseline_2026-06-23/`, `results/evaluation/baseball/train_on_test_first_order_baseline_2026-06-23/`

**Interfaces:**
- Produces: a clean first-order baseline metric set (`num_equation_terms`, `test mean_reconstruction_r2`, `test mean_reconstruction_rmse`) recorded in the snapshot dirs. Task 6 compares the second-order result against these exact numbers.

- [ ] **Step 1: Run the first-order pipeline on the restored pool**

The handedness config is the strongest existing first-order model. Re-extract features, fit, and evaluate on the restored 111/29 pool, freezing the split (`--skip-splits`) and skipping handedness-override re-application (`--skip-overrides`, since overrides target only the old annotated clips):
```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml \
  --skip-splits --skip-overrides
```
Expected: completes without error; prints trained-models and results-root lines.

- [ ] **Step 2: Record the baseline metrics**

Run:
```bash
./.venv/bin/python -c "
import csv
tr=list(csv.DictReader(open('results/equations/baseball/train/summary.csv')))
te=list(csv.DictReader(open('results/evaluation/baseball/train_on_test/summary.csv')))
print('TRAIN', [(r['num_equation_terms'], r['mean_trial_rmse']) for r in tr])
print('TEST ', [(r['mean_reconstruction_r2'], r['mean_reconstruction_rmse']) for r in te])
"
```
Expected: a printed first-order baseline. Note `num_equation_terms`, test `mean_reconstruction_r2`, and test `mean_reconstruction_rmse` — these are the A/B reference values.

- [ ] **Step 3: Snapshot the baseline outputs aside**

```bash
cp -r results/equations/baseball/train results/equations/baseball/train_first_order_baseline_2026-06-23
cp -r results/evaluation/baseball/train_on_test results/evaluation/baseball/train_on_test_first_order_baseline_2026-06-23
```

- [ ] **Step 4: Commit the baseline snapshot**

```bash
git add results/equations/baseball/train_first_order_baseline_2026-06-23 results/evaluation/baseball/train_on_test_first_order_baseline_2026-06-23
git commit -m "results: snapshot first-order 111/29 baseline before second-order redesign"
```

---

## Task 2: Add `compute_second_order_state` to extract_baseball_features.py

**Files:**
- Modify: `src/preprocessing/extract_baseball_features.py` (imports + new function)

**Interfaces:**
- Produces: `compute_second_order_state(feature_series, position_names, dt, window_length, polyorder, unwrap_angles) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(X_pos, V, A)`, each shape `(T, len(position_names))`. `V` is the first time-derivative of each position; `A` the second. Consumed by Task 3's `main()` branch.

- [ ] **Step 1: Extend the imports**

In `src/preprocessing/extract_baseball_features.py`, the existing import block from `extract_utd_mhad_features` (lines 20-28) imports `build_feature_series, build_state_matrix, estimate_derivatives, load_processed_clip, save_feature_payload, write_csv, write_jsonl`. Add `ANGLE_FEATURES` and `adjusted_window` to that import, and add a scipy import. Replace the import block with:

```python
from src.preprocessing.extract_utd_mhad_features import (
    ANGLE_FEATURES,
    adjusted_window,
    build_feature_series,
    build_state_matrix,
    estimate_derivatives,
    load_processed_clip,
    save_feature_payload,
    write_csv,
    write_jsonl,
)
from scipy.signal import savgol_filter
```

- [ ] **Step 2: Add the function**

Insert this function after `canonicalize_feature_series` (after line 113, before `zscore_series`):

```python
def compute_second_order_state(
    feature_series: dict[str, Any],
    position_names: list[str],
    dt: float,
    window_length: int,
    polyorder: int,
    unwrap_angles: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_pos, V, A) for a strict second-order state.

    Each output has shape (T, len(position_names)). A single Savitzky-Golay
    fit per position column is queried at deriv 0/1/2 so that X (positions +
    velocities) and X_dot (velocities + accelerations) stay internally
    consistent and share the same smoothing. Angle columns are unwrapped
    before filtering; rate columns carry no 2*pi discontinuity so they are
    not unwrapped.
    """
    pos_cols: list[np.ndarray] = []
    vel_cols: list[np.ndarray] = []
    acc_cols: list[np.ndarray] = []
    for name in position_names:
        series = np.asarray(feature_series[name], dtype=np.float64)
        if unwrap_angles and name in ANGLE_FEATURES:
            series = np.unwrap(series)
        window = adjusted_window(series.shape[0], window_length, polyorder)
        if window == 0:
            pos = series
            vel = np.gradient(series, dt)
            acc = np.gradient(vel, dt)
        else:
            pos = savgol_filter(series, window, polyorder, deriv=0, mode="interp")
            vel = savgol_filter(series, window, polyorder, deriv=1, delta=dt, mode="interp")
            acc = savgol_filter(series, window, polyorder, deriv=2, delta=dt, mode="interp")
        pos_cols.append(pos)
        vel_cols.append(vel)
        acc_cols.append(acc)
    return np.column_stack(pos_cols), np.column_stack(vel_cols), np.column_stack(acc_cols)
```

- [ ] **Step 3: Verify it compiles**

Run: `./.venv/bin/python -m compileall src/preprocessing/extract_baseball_features.py`
Expected: `Compiling ...` with no error.

- [ ] **Step 4: Verify the function on real data (unit-style check)**

Run:
```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
import csv, numpy as np
from src.preprocessing.extract_utd_mhad_features import load_processed_clip, build_feature_series
from src.preprocessing.extract_baseball_features import compute_second_order_state, canonicalize_feature_series
rows=list(csv.DictReader(open('data/baseball/processed/train/manifest.csv')))[:1]
r=rows[0]; coords,joints,meta=load_processed_clip(r['filepath']); dt=float(r['dt'])
fs=build_feature_series(coords,joints,dt); fs=canonicalize_feature_series(fs,'R')
pos_names=['shoulder_rotation_angle','hip_rotation_angle','dominant_elbow_flexion_angle','trail_elbow_flexion_angle','dominant_knee_flexion_angle','trail_knee_flexion_angle','lead_knee_lift','stride_ankle_separation']
X_pos,V,A=compute_second_order_state(fs,pos_names,dt,9,3,True)
print('shapes',X_pos.shape,V.shape,A.shape)
print('finite', np.isfinite(X_pos).all(), np.isfinite(V).all(), np.isfinite(A).all())
"
```
Expected: all three shapes are `(T, 8)` with the same T; all three `finite` values `True`. If any acceleration column has NaN/Inf at boundaries, note it — Task 5 may bump the SG window.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/extract_baseball_features.py
git commit -m "feat: add compute_second_order_state for 16-D baseball state"
```

---

## Task 3: Wire the `second_order` flag into the extraction main loop

**Files:**
- Modify: `src/preprocessing/extract_baseball_features.py` (`main()` body, lines ~213-239)

**Interfaces:**
- Consumes: `compute_second_order_state` (Task 2), `feature_cfg["second_order"]` flag, `feature_cfg["state_vector"]` (8 position names when the flag is set).
- Produces: feature payloads whose `X` and `X_dot` are 16 columns and whose `feature_names` are the 8 positions followed by 8 `<name>_rate` columns. The fit/evaluate scripts consume these generically.

- [ ] **Step 1: Read the flag**

In `main()`, just after `infer_handedness_for_pitching = ...` (line 189), add:
```python
    second_order = bool(feature_cfg.get("second_order", False))
```

- [ ] **Step 2: Branch the state/derivative assembly**

Replace the block that builds `state_matrix` and `derivative_matrix` (lines 213-225, from `state_matrix = build_state_matrix(` through the `estimate_derivatives(...)` call) with:

```python
        if second_order:
            position_names = feature_names
            window_length = int(feature_cfg["derivative"]["window_length"])
            polyorder = int(feature_cfg["derivative"]["polyorder"])
            X_pos, V, A = compute_second_order_state(
                feature_series,
                position_names,
                dt,
                window_length,
                polyorder,
                bool(feature_cfg["angle_unwrap"]),
            )
            state_matrix = np.hstack([X_pos, V])
            derivative_matrix = np.hstack([V, A])
            output_feature_names = list(position_names) + [f"{name}_rate" for name in position_names]
            derivative_window = adjusted_window(state_matrix.shape[0], window_length, polyorder)
        else:
            state_matrix = build_state_matrix(
                feature_series,
                feature_names,
                bool(feature_cfg["angle_unwrap"]),
            )
            derivative_matrix, derivative_window = estimate_derivatives(
                state_matrix,
                dt,
                feature_names,
                bool(feature_cfg["angle_unwrap"]),
                int(feature_cfg["derivative"]["window_length"]),
                int(feature_cfg["derivative"]["polyorder"]),
            )
            output_feature_names = feature_names
```

- [ ] **Step 3: Use `output_feature_names` in the payload and metadata**

In the same loop, the metadata dict (lines 228-238) and `save_feature_payload` call (line 239) currently use `feature_names`. Update both to use `output_feature_names`:
- In the `metadata` dict, change `"feature_names": feature_names,` to `"feature_names": output_feature_names,` and `"num_features": len(feature_names),` to `"num_features": len(output_feature_names),`.
- Change the call to `save_feature_payload(output_path, state_matrix, derivative_matrix, output_feature_names, metadata)`.

(The `manifest_rows`/`log_rows` use `state_matrix.shape[1]`, which is already 16 — no change needed there.)

- [ ] **Step 4: Verify it compiles**

Run: `./.venv/bin/python -m compileall src/preprocessing/extract_baseball_features.py`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/extract_baseball_features.py
git commit -m "feat: wire second_order flag into baseball feature extraction"
```

---

## Task 4: Create the second-order configs

**Files:**
- Create: `config/preprocessing.baseball.unaligned.second_order.yaml`
- Create: `config/preprocessing.baseball.unaligned.second_order.sweep.yaml`

**Interfaces:**
- Produces: two configs whose `features.state_vector` lists the 8 positions and `features.second_order: true`. The single-shot config sets `modeling.stlsq_threshold: 0.1`; the sweep config sets `modeling.sweep.stlsq_threshold: [0.05, 0.10, 0.15, 0.20, 0.30]`.

- [ ] **Step 1: Write the single-shot config**

Create `config/preprocessing.baseball.unaligned.second_order.yaml` (copy of `unaligned.handedness.yaml` with the 8-position state vector, `second_order: true`, and second-order plot names):

```yaml
dataset:
  name: baseball_stage_b
  standardized_pose_manifest: data/baseball/standardized_pose/manifest.csv
  processed_root: data/baseball/processed

pose:
  center_joint: hip_center
  orientation_normalization: false
  target_joints:
    - head
    - neck
    - spine
    - left_shoulder
    - right_shoulder
    - left_elbow
    - right_elbow
    - left_wrist
    - right_wrist
    - left_hip
    - right_hip
    - left_knee
    - right_knee
    - left_ankle
    - right_ankle

preprocessing:
  trim_motion_windows: true
  interpolate_short_gaps: true
  max_gap_frames: 30
  smoothing:
    method: savgol
    window_length: 11
    polyorder: 3
  trim:
    energy_threshold_ratio: 0.2
    padding_frames: 4
    min_frames: 25
  scale_normalization: false

features:
  output_subdir: features
  angle_unwrap: true
  default_handedness: R
  infer_handedness_for_pitching: true
  second_order: true
  state_vector:
    - shoulder_rotation_angle
    - hip_rotation_angle
    - dominant_elbow_flexion_angle
    - trail_elbow_flexion_angle
    - dominant_knee_flexion_angle
    - trail_knee_flexion_angle
    - lead_knee_lift
    - stride_ankle_separation
  derivative:
    method: savgol_gradient
    window_length: 9
    polyorder: 3
  alignment:
    enabled: false

modeling:
  input_feature_subdir: features
  results_root: results/equations
  polynomial_degree: 1
  include_bias: true
  include_interaction: true
  scale_features: true
  stlsq_threshold: 0.1
  stlsq_alpha: 0.001
  stlsq_max_iter: 20
  selection_metric: sparsity_weighted_r2
  complexity_penalty: 0.00005
  selection_validation_fraction: 0.22
  selection_validation_seed: 7
  sweep:
    polynomial_degree: [1]
    stlsq_threshold: [0.1]
    stlsq_alpha: [0.001]

evaluation:
  feature_subdir: features
  model_results_root: results/equations
  evaluation_root: results/evaluation
  plot_root: results/plots
  reconstruction_frames: 40
  plot_feature_names:
    - shoulder_rotation_angle
    - hip_rotation_angle
    - dominant_elbow_flexion_angle
    - lead_knee_lift
    - stride_ankle_separation
  max_plot_trials_per_label: 1

logging:
  write_run_log: true
  log_dir: results/preprocessing_logs
```

- [ ] **Step 2: Write the sweep config**

Create `config/preprocessing.baseball.unaligned.second_order.sweep.yaml` identical to Step 1 except the `modeling.sweep.stlsq_threshold` line. Copy the file and change the `sweep` block to:

```yaml
  sweep:
    polynomial_degree: [1]
    stlsq_threshold: [0.05, 0.10, 0.15, 0.20, 0.30]
    stlsq_alpha: [0.001]
```

- [ ] **Step 3: Validate YAML parses and lists 8 positions**

Run:
```bash
./.venv/bin/python -c "
import yaml
for f in ('second_order','second_order.sweep'):
    c=yaml.safe_load(open(f'config/preprocessing.baseball.unaligned.{f}.yaml'))
    sv=c['features']['state_vector']
    print(f, 'second_order=', c['features']['second_order'], 'positions=', len(sv), 'sweep_thr=', c['modeling']['sweep']['stlsq_threshold'])
    assert len(sv)==8 and c['features']['second_order'] is True
"
```
Expected: both print `second_order= True positions= 8`; single-shot sweep_thr `[0.1]`, sweep config sweep_thr `[0.05, 0.1, 0.15, 0.2, 0.3]`.

- [ ] **Step 4: Commit**

```bash
git add config/preprocessing.baseball.unaligned.second_order.yaml config/preprocessing.baseball.unaligned.second_order.sweep.yaml
git commit -m "config: add second-order baseball state-vector configs"
```

---

## Task 5: Extract + run the single-shot identity gate

**Files:**
- Write (data): `data/baseball/processed/{train,test}/features/...` (16-col payloads)
- Write (results): `results/equations/baseball/train/pitching/...`

**Interfaces:**
- Consumes: Task 3 code path, Task 4 single-shot config, Task 0 processed pool.
- Produces: a fitted second-order model whose first 8 equations (`d(position_i)/dt`) are each single-term on the paired `_rate` column. This gate must pass before the sweep.

- [ ] **Step 1: Extract second-order features for both splits**

```bash
./.venv/bin/python src/preprocessing/extract_baseball_features.py --config config/preprocessing.baseball.unaligned.second_order.yaml --split train
./.venv/bin/python src/preprocessing/extract_baseball_features.py --config config/preprocessing.baseball.unaligned.second_order.yaml --split test
```

- [ ] **Step 2: Verify payload shapes and the by-construction identity**

Run:
```bash
./.venv/bin/python -c "
import csv, numpy as np
from src.preprocessing.fit_utd_mhad_sindy import load_feature_payload
rows=list(csv.DictReader(open('data/baseball/processed/train/features/manifest.csv')))[:5]
for r in rows:
    X,Xd,names,meta=load_feature_payload(r['filepath'])
    assert X.shape[1]==16 and Xd.shape[1]==16, (r['filepath'], X.shape, Xd.shape)
    assert np.allclose(Xd[:, :8], X[:, 8:], atol=1e-5), 'position-derivative copy broke'
print('OK: 16-col payloads, X_dot[:, :8] == X[:, 8:] on', len(rows), 'clips')
print('feature_names:', names)
"
```
Expected: prints `OK: ...` and a 16-name list ending in 8 `_rate` names. If the assert fails, the Task 3 `hstack` ordering is wrong — fix before continuing.

- [ ] **Step 3: Fit once at the lowest sweep threshold**

Temporarily fit with the single-shot config (threshold 0.1 is fine for the gate; structure, not threshold, is what matters):
```bash
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py --config config/preprocessing.baseball.unaligned.second_order.yaml --split train --task-types pitching
```

- [ ] **Step 4: Inspect the identity gate**

Run:
```bash
./.venv/bin/python -c "
import csv
rows=list(csv.DictReader(open('results/equations/baseball/train/pitching/coefficients.csv')))
pos=['shoulder_rotation_angle','hip_rotation_angle','dominant_elbow_flexion_angle','trail_elbow_flexion_angle','dominant_knee_flexion_angle','trail_knee_flexion_angle','lead_knee_lift','stride_ankle_separation']
for p in pos:
    terms=[(r['library_term'], r['coefficient']) for r in rows if r['state_variable']==p]
    rate_terms=[t for t in terms if t[0].endswith('_rate')]
    other=[t for t in terms if not t[0].endswith('_rate') and t[0]!='1']
    flag='OK' if (len(rate_terms)==1 and rate_terms[0][0]==p+'_rate' and not other) else 'FAIL'
    print(flag, p, '->', terms)
"
```
Expected: every position-derivative equation prints `OK` — exactly one non-bias term, the paired `<name>_rate`. A bias (`1`) term is tolerable. If any prints `FAIL` (cross-couplings or wrong rate), the gate fails.

- [ ] **Step 5: If the gate fails, widen the SG window and re-extract**

Only if Step 4 shows FAIL: edit both second-order configs `features.derivative.window_length` 9 → 11 (then 13 if still failing), re-run Steps 1, 3, 4. Do NOT proceed to Task 6 until the gate passes. Record which window value passed.

- [ ] **Step 6: Commit the gate result**

```bash
git add config/preprocessing.baseball.unaligned.second_order.yaml config/preprocessing.baseball.unaligned.second_order.sweep.yaml
git commit -m "test: second-order identity gate passes (window=<N>)"
```
(Commit even if no config edit was needed — use `--allow-empty` to mark the gate milestone, or skip if nothing changed.)

---

## Task 6: Run the threshold sweep, evaluate on test, make the A/B call

**Files:**
- Write: `results/equations/baseball/train/pitching/...`, `results/evaluation/baseball/train_on_test/...`

**Interfaces:**
- Consumes: Task 4 sweep config, Task 5 verified extraction.
- Produces: a ship / partial-win / regression decision comparing second-order test metrics to the Task 1 first-order baseline.

- [ ] **Step 1: Run the full sweep + evaluation on the frozen pool**

```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py \
  --config config/preprocessing.baseball.unaligned.second_order.sweep.yaml \
  --skip-splits --skip-overrides
```
Expected: completes; the fit step picks one threshold via `sparsity_weighted_r2`; evaluation writes `train_on_test/summary.csv`.

- [ ] **Step 2: Print the second-order result vs. the baseline**

Run:
```bash
./.venv/bin/python -c "
import csv
def one(p, keys):
    r=list(csv.DictReader(open(p)))[0]; return {k:r[k] for k in keys}
b_tr=one('results/equations/baseball/train_first_order_baseline_2026-06-23/summary.csv',['num_equation_terms'])
b_te=one('results/evaluation/baseball/train_on_test_first_order_baseline_2026-06-23/summary.csv',['mean_reconstruction_r2','mean_reconstruction_rmse'])
n_tr=one('results/equations/baseball/train/summary.csv',['num_equation_terms','stlsq_threshold'])
n_te=one('results/evaluation/baseball/train_on_test/summary.csv',['mean_reconstruction_r2','mean_reconstruction_rmse'])
print('FIRST-ORDER baseline:', b_tr, b_te)
print('SECOND-ORDER       :', n_tr, n_te)
"
```
Expected: a side-by-side print of `num_equation_terms`, test `mean_reconstruction_r2`, test `mean_reconstruction_rmse` for both.

- [ ] **Step 3: Apply the A/B decision matrix**

Compare second-order test `mean_reconstruction_r2` (call it `r2`) and `mean_reconstruction_rmse` (call it `rmse`) to baseline (`r2_base`, `rmse_base`):
- `r2 > 0` → **Ship.** New baseline.
- `r2 ∈ [r2_base, 0)` AND `rmse < rmse_base` → **Partial win.** Ship as new baseline; queue follow-ups (`wrist_relative_height` 9th paired position, custom pairwise library).
- otherwise → **Regression.** Do not ship. Run Step 4 diagnostics.

State the call explicitly in the task report.

- [ ] **Step 4: Per-clip + per-equation diagnostics**

Run:
```bash
./.venv/bin/python -c "
import csv
# worst test clips
m=list(csv.DictReader(open('results/evaluation/baseball/train_on_test/reconstruction_metrics.csv')))
m.sort(key=lambda r: float(r.get('reconstruction_rmse', r.get('rmse', 0))), reverse=True)
print('worst 5 test clips:', [(r.get('clip_id'), r.get('pitch_type'), r.get('reconstruction_rmse', r.get('rmse'))) for r in m[:5]])
# which rate-derivative equations collapsed to zero
rows=list(csv.DictReader(open('results/equations/baseball/train/pitching/coefficients.csv')))
rate_eqs=set(r['state_variable'] for r in rows if r['state_variable'].endswith('_rate'))
all_rates=[p+'_rate' for p in ['shoulder_rotation_angle','hip_rotation_angle','dominant_elbow_flexion_angle','trail_elbow_flexion_angle','dominant_knee_flexion_angle','trail_knee_flexion_angle','lead_knee_lift','stride_ankle_separation']]
print('rate eqs with terms:', sorted(rate_eqs))
print('COLLAPSED rate eqs:', [r for r in all_rates if r not in rate_eqs])
"
```
Expected: lists worst clips (check for handedness/pitch-type/source clustering) and any rate-derivative equation that collapsed to zero (the next feature-design target). Adapt column names if the CSV headers differ.

- [ ] **Step 5: Commit the result**

```bash
git add results/equations/baseball/train results/evaluation/baseball/train_on_test
git commit -m "results: second-order state-vector sweep + test evaluation"
```

---

## Task 7: Update feature documentation and memory

**Files:**
- Rewrite: `docs/feature_definitions.md`
- Update: project memory file `project_stage_b_status.md` + `MEMORY.md` pointer

**Interfaces:**
- Produces: `docs/feature_definitions.md` documenting the 16-D second-order state, the derivative scheme, and the dropped-feature rationale; updated memory snapshot with the A/B outcome.

- [ ] **Step 1: Rewrite `docs/feature_definitions.md`**

Replace its contents with documentation of: the 8 positions and their geometric definitions (cross-reference `build_feature_series`), the 8 paired `_rate` velocities, the single-SG-fit deriv-0/1/2 scheme, `X = [pos, vel]` / `X_dot = [vel, acc]` assembly, and why `hip_shoulder_separation_angle` (algebraic identity) and `dominant_wrist_speed` (unpaired magnitude) were dropped. Note the handedness canonicalization and sign-flip still apply to the rotation positions.

- [ ] **Step 2: Verify no stale 10-D references remain**

Run: `grep -n "dominant_wrist_speed\|hip_shoulder_separation" docs/feature_definitions.md`
Expected: matches appear ONLY in the "dropped features" rationale section, nowhere as part of the active state vector.

- [ ] **Step 3: Update memory with the outcome**

Update `/home/dtmschangmin/.claude/projects/-mnt-c-Users-dtms--Documents-Research/memory/project_stage_b_status.md` with: pool restored to 111/29 from cache on 2026-06-23, second-order 16-D state implemented, the A/B call from Task 6, and any collapsed rate equations as the next target. Refresh the `MEMORY.md` one-line pointer.

- [ ] **Step 4: Commit**

```bash
git add docs/feature_definitions.md
git commit -m "docs: document 16-D second-order baseball state vector"
```

---

## Self-Review Notes

- **Spec coverage:** state vector (Task 4), derivative scheme (Task 2), SINDy library/threshold/sweep (Tasks 4–6), pipeline integration without touching fit/eval/UTD code (Tasks 2–3), baseline preservation (Task 1), identity gate + by-construction check (Task 5), A/B decision matrix (Task 6), feature-doc rewrite (Task 7). The spec assumed the 111/29 pool already existed; Task 0 restores it from cache since it had regressed to 24 clips — this is the only addition beyond the spec.
- **Out-of-scope items** (`wrist_relative_height`, custom pairwise library, pool expansion >111/29, per-pitch-type models) are explicitly deferred and surface only as Task 6 follow-up queue entries.
- **Type consistency:** `compute_second_order_state` returns `(X_pos, V, A)`; `main()` assembles `X = hstack([X_pos, V])`, `X_dot = hstack([V, A])`; `output_feature_names = positions + [name+"_rate"]`. Payload `feature_names` length 16 matches `X.shape[1]`. The fit script reads `feature_names` from the payload and scales generically — no signature changes.
