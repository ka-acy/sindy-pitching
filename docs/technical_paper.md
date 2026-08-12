# Sparse Dynamical Modeling of Baseball Pitching Kinematics from Vision-Derived Pose

## Abstract

This paper develops and validates a reproducible computational pipeline for converting human motion trajectories into sparse nonlinear dynamical models. The workflow uses standardized skeletal pose, biomechanical feature extraction, derivative estimation, and Sparse Identification of Nonlinear Dynamical Systems (SINDy) to generate interpretable coefficient matrices and governing-equation candidates. The project is structured in two stages. Stage A validates the pipeline on the UTD-MHAD human action dataset and is used only as a computational validation dataset. Stage B translates the validated workflow to baseball pitching clips using MediaPipe-derived pose trajectories from local MLB video segments. The current Stage B implementation regenerates processed trajectories, 10-variable pitching state vectors, SINDy models, coefficient tables, and held-out evaluation summaries from scripted commands on a 140-clip pitching pool (111 train / 29 held-out test). Held-out simulations remain numerically bounded on all test clips, but no fitted model achieves positive predictive accuracy: one-step derivative R2 sits near zero and free-running reconstruction R2 is negative for every configuration tried. A frozen-model control introduced in this work shows that a zero-dynamics predictor outscores every fitted model on the state-comparable reconstruction metric, which establishes that the negative scores reflect a genuine failure to recover transferable dynamics rather than a metric artifact favoring one model family. These results support the pipeline as a reproducible feasibility framework and a documented negative result, while identifying data quality, sample size, camera-view consistency, and feature refinement as the main requirements before making biomechanical claims.

## 1. Introduction

Baseball pitching is a high-speed, multi-segment movement involving coordinated lower-body drive, trunk rotation, shoulder and elbow motion, stride mechanics, and rapid distal segment acceleration. Traditional biomechanical analysis often relies on laboratory motion capture, which produces high-quality trajectories but is expensive and difficult to scale. Broadcast or locally segmented video is more accessible, but vision-derived pose introduces noise, camera-view bias, missing landmarks, and two-dimensional projection limits.

This project investigates whether a machine-vision and sparse-dynamics workflow can produce reproducible mathematical artifacts from baseball pitching motion. The goal is not to claim that the current dataset explains elite pitching mechanics. The goal is to establish a defensible technical pipeline that can ingest pose trajectories, condition them into stable feature matrices, estimate derivatives, and fit sparse dynamical models that can be compared across preprocessing and modeling configurations under a common evaluation protocol.

## 2. Research Questions

1. Can raw or vision-derived skeletal trajectories be converted into standardized baseball pitching feature matrices without manual per-clip intervention?
2. Can derivative matrices and sparse coefficient matrices be regenerated reproducibly from those feature matrices?
3. Are the fitted SINDy systems numerically stable on held-out baseball clips?
4. Do any of the fitted systems outperform a no-dynamics control baseline on held-out reconstruction?

## 3. System Overview

The repository implements a two-stage workflow.

Stage A validates the general motion-processing and sparse-modeling infrastructure on UTD-MHAD. This stage tests data loading, skeleton standardization, smoothing, derivative estimation, model fitting, and coefficient export. Stage A artifacts are infrastructure-validation outputs, not baseball findings.

Stage B replaces only the input-pose layer with baseball-specific standardized pose. Downstream processing remains the same in principle: preprocessing, feature extraction, derivative estimation, SINDy fitting, and evaluation. This design keeps source-specific video and pose-estimation issues separated from the mathematical modeling workflow.

## 4. Data

### 4.1 Stage A Validation Data

Stage A uses UTD-MHAD as a proxy human-motion dataset. The selected proxy actions are labeled as `batting_like` and `pitching_like` to test rotational and ballistic motion processing. These labels are methodological categories and do not imply that UTD-MHAD motions reproduce baseball mechanics.

### 4.2 Stage B Baseball Data

Stage B uses local baseball clips represented in `data/baseball/metadata/clip_manifest.csv`, local-media split files under `data/baseball/splits/`, and standardized pose payloads registered under `data/baseball/standardized_pose/`. MediaPipe pose exports are imported into a 15-joint schema compatible with the Stage A skeleton format.

The current pitching pool is **140 clips** (`data/baseball/splits/pitching_local_pool.csv`), split into **111 training clips** (`pitching_local_train.csv`) and **29 held-out test clips** (`pitching_local_test.csv`). All clips come from a single consistent camera view (`broadcast_centerfield`). The effective sampling interval is `dt = 0.016683 s` (approximately 59.9 fps).

Each clip additionally carries `pitch_type` and `pitch_speed_mph` labels in the pool CSV. These labels are not used by the dynamics pipeline described here; they are recorded for the label-validated representation study scoped in `docs/superpowers/specs/2026-06-23-representation-study-design.md`.

Generated arrays, processed features, models, and result summaries are intentionally excluded from version control by `.gitignore`. They are reproducible from the commands in Section 8.

## 5. Methods

### 5.1 Pose Standardization

Each baseball pose clip is stored as a compressed `.npz` payload containing:

- `coordinates`: pose array with shape `(time, joints, coordinate_dims)`
- `joint_names`: ordered joint labels
- `metadata`: JSON-serializable clip and pose-source metadata

The preferred joint schema contains 15 joints: head, neck, spine, shoulders, elbows, wrists, hips, knees, and ankles. The current MediaPipe adapter maps the larger MediaPipe landmark set into this schema.

### 5.2 Preprocessing

The baseball preprocessing script repairs leading and trailing missing values by edge fill, interpolates internal gaps, centers trajectories at the hip-center proxy, optionally applies orientation and scale normalization, smooths coordinates with a Savitzky-Golay filter, and trims low-motion frames around the action window.

The operative baseball configuration is `config/preprocessing.baseball.unaligned.handedness.yaml`, which uses:

- center joint: `hip_center`
- orientation normalization: disabled
- scale normalization: disabled
- internal gap interpolation: enabled, maximum gap 30 frames
- coordinate smoothing: Savitzky-Golay, window length 11, polynomial order 3
- motion trimming: enabled, energy threshold ratio 0.2, 4 padding frames, minimum 25 frames

### 5.3 Feature Extraction

The current Stage B pitching state vector has 10 variables:

1. `shoulder_rotation_angle`
2. `hip_rotation_angle`
3. `hip_shoulder_separation_angle`
4. `dominant_elbow_flexion_angle`
5. `trail_elbow_flexion_angle`
6. `dominant_knee_flexion_angle`
7. `trail_knee_flexion_angle`
8. `lead_knee_lift`
9. `stride_ankle_separation`
10. `dominant_wrist_speed`

Features are canonicalized by handedness: the configuration sets `default_handedness: R` and `infer_handedness_for_pitching: true`, so dominant/trail designations are resolved per clip rather than fixed to a body side. Angles are unwrapped (`angle_unwrap: true`).

Event alignment is **disabled** in the operative configuration (`features.alignment.enabled: false`). An earlier configuration track aligned pitching clips around stride-ankle separation with lead-knee lift as a secondary signal; that track is retained as `config/preprocessing.baseball.aligned.yaml` but is not the operative default. Section 6.4 records why.

Derivatives are estimated with Savitzky-Golay differentiation using window length 9 and polynomial order 3.

### 5.4 Sparse Model Identification

SINDy models are fit with PySINDy using a polynomial library with bias and interaction terms, feature scaling enabled, and STLSQ sparse regression. The operative configuration uses:

- polynomial degree: 1
- STLSQ alpha: 0.001
- maximum STLSQ iterations: 20
- model-selection metric: sparsity-weighted R2 (`selection_metric: sparsity_weighted_r2`)
- complexity penalty: 5e-05 per term
- validation fraction for selection: 0.22 (24 of 111 training clips), seed 7

The STLSQ threshold is **not fixed**; it is swept over `{0.1, 0.15, 0.2, 0.3, 0.5}` and selected by the sparsity-weighted score on the held-out selection split. For the operative first-order configuration the sweep selects **threshold 0.2**, yielding **26 equation terms**. Section 6.3 documents a degeneracy in this selection rule.

A weak-form variant is available at `config/preprocessing.baseball.unaligned.handedness.weakform.yaml`, gated by `modeling.weak_form.enabled`. It fits a `WeakPDELibrary` SINDy over stacked per-trajectory weak systems across all 111 training clips, then transplants the resulting coefficients into a plain polynomial-library SINDy so the standard predict and simulate paths apply unchanged. Because `WeakPDELibrary` places its test-function subdomains via the global NumPy RNG with no seed parameter, a single draw is neither reproducible nor low-variance; the implementation seeds locally (restoring prior RNG state) and stacks `n_seeds` independent subdomain draws. The operative weak-form settings are `K: 100`, `seed: 7`, `n_seeds: 5`.

The model outputs include equations, coefficient matrices, model-selection logs, scaler metadata, per-trial metrics, and summary CSVs.

### 5.5 Evaluation

Held-out evaluation computes one-step derivative-prediction metrics and free-running reconstruction metrics. Reconstruction integrates the fitted system forward from the first observed state for `reconstruction_frames: 40` frames using the **RK4** integrator (`evaluation.integrator: rk4`; an `euler` option is retained and was marginally worse, core R2 −29.99 vs −29.56 on the first-order baseline).

Two reconstruction metrics are reported:

- **Whole-state reconstruction R2** — variance-weighted across all 10 state columns. This metric is *not* comparable across different state-vector definitions, because high-variance columns dominate it. For the 10-D first-order state, `dominant_wrist_speed` inflates the score.
- **Core-position reconstruction R2 / NRMSE** — a state-comparable metric computed only over the 8 shared `CORE_POSITIONS` columns (all state variables except `hip_shoulder_separation_angle` and `dominant_wrist_speed`), using `r2_score(..., multioutput="uniform_average")` so no single column dominates. NRMSE normalizes each column's RMSE by that column's standard deviation, making it unit-free.

Core-position R2 is additionally reported at rollout horizons `{5, 10, 20, 40}` frames from the same forward simulation.

**Control baseline.** Reconstruction R2 is measured against the per-window mean, which is a harsh reference when the core positions move little over a short rollout. To make the reconstruction numbers interpretable, `src/preprocessing/evaluate_static_baseline.py` scores a **frozen model** — zero derivative, so the state is held at its first observed value for the whole rollout — under the identical metric, horizons, and test split. This encodes no dynamics whatsoever and therefore establishes the floor that any fitted model must clear for its reconstruction score to carry information.

Simulation stability (finite and bounded trajectories) is reported separately. Stability is a necessary but very weak condition and is not evidence of predictive accuracy: a model whose coefficients are near zero is bounded by construction.

## 6. Current Results

All results below were produced on 2026-08-12 against the same 140-clip pool (111 train / 29 test), RK4 integrator, and 40-frame reconstruction. Every number is backed by a CSV under `results/`; the consolidated table is `results/evaluation/baseball/model_comparison_2026-08-12.csv`.

### 6.1 Operative first-order model

Training summary (`results/equations/baseball/train/summary.csv`, snapshot `train_plain_first_order_2026-08-12/`):

| quantity | value |
|---|---|
| training clips | 111 |
| feature count | 10 |
| selected STLSQ threshold | 0.2 |
| selected equation terms | 26 |
| global derivative R2 (train) | 0.0063 |
| mean trial derivative R2 (train) | −0.0136 |
| mean trial derivative RMSE (train) | 10.9599 |

Held-out summary (`results/evaluation/baseball/train_on_test/summary.csv`):

| quantity | value |
|---|---|
| held-out clips | 29 |
| mean derivative R2 | −0.0184 |
| mean derivative RMSE | 9.8448 |
| stable simulation fraction | 1.0 |
| whole-state reconstruction R2 | −2.7981 |
| whole-state reconstruction RMSE | 0.7232 |
| core-position reconstruction R2 | −29.5553 |
| core-position reconstruction NRMSE | 3.2544 |

### 6.2 Model comparison against the frozen control

`core_position_recon_r2` is the state-comparable metric; higher (less negative) is better.

| model | terms | core R2 | core NRMSE | whole-state R2 |
|---|---:|---:|---:|---:|
| **frozen control (no dynamics)** | **0** | **−5.5707** | **2.2158** | n/a |
| first-order, weak-form, multi-seed | 1 | −7.8922 | 2.3781 | −6.7185 |
| first-order, plain (operative) | 26 | −29.5553 | 3.2544 | −2.7981 |
| first-order, ensemble | 40 | −36.9650 | 3.5159 | −3.0544 |
| second-order, plain | 101 | −2269.04 | 19.1044 | −6.0350 |
| second-order, ensemble | 131 | −2428.87 | 19.2161 | −6.2521 |

Short-horizon core R2, plain vs weak-form vs control:

| horizon (frames) | plain first-order | weak-form | frozen control |
|---:|---:|---:|---:|
| 5 | −75.47 | −2.6818 | **−2.1619** |
| 10 | −47.81 | −5.4027 | **−2.8330** |
| 20 | −37.99 | −4.9043 | **−4.2264** |
| 40 | −29.56 | −7.8922 | **−5.5707** |

Three observations follow directly:

1. **No fitted model beats the frozen control at any horizon.** The best fitted model (weak-form, core R2 −7.89) is outscored by a predictor that holds the initial pose constant (−5.57). Under this metric, every dynamical model tried is worse than assuming the pitcher does not move.
2. **The ordering is monotonic in model complexity, in the wrong direction.** Core R2 degrades as term count rises (0 → 1 → 26 → 40 → 101 → 131 terms). Added expressivity strictly hurts held-out reconstruction, which is the signature of fitting noise rather than dynamics.
3. **Stability is uninformative here.** Every configuration reports a stable simulation fraction of 1.0, including models that are numerically bounded only because their coefficients are near zero.

### 6.3 Why the weak-form model appeared to win

An earlier internal comparison recorded the weak-form model as the best available on the fair metric (core R2 −7.89 versus −29.56) and recommended it as operative. Regenerating that run shows the result holds numerically but does not support that conclusion. The selected weak-form model has a **single non-bias term** across the entire 10-dimensional state:

```
d(dominant_knee_flexion_angle)/dt = 0.354 * trail_elbow_flexion_angle
```

All nine other state derivatives are identically zero. The model is, in substance, the frozen control with one residual term — and it scores *worse* than the true frozen control.

The cause is a degeneracy in the model-selection rule. From the weak-form selection log (`results/equations/baseball/train_weakform_2026-08-12/pitching/model_selection.csv`):

| threshold | terms | validation mean trial R2 | selection score |
|---:|---:|---:|---:|
| 0.10 | 37 | −0.008702 | −0.010552 |
| 0.15 | 23 | −0.008969 | −0.010119 |
| 0.20 | 13 | −0.009013 | −0.009663 |
| **0.30** | **2** | **−0.009548** | **−0.009648** |
| 0.50 | 0 | −0.010045 | −0.010045 |

Every candidate's validation R2 lies in a band of width 0.0013 around zero, so the accuracy term carries no signal. Selection is therefore decided entirely by the sparsity penalty, and threshold 0.30 wins over 0.20 by **1.5e-05**. When all candidates are equally uninformative, `sparsity_weighted_r2` reduces to "prefer the emptiest model." The apparent weak-form victory is an artifact of that rule interacting with a metric that rewards not moving, not evidence that the weak-form fit recovered better dynamics.

This is why the frozen control is necessary. Without it, a near-null model reads as a 3.7x improvement.

### 6.4 Preprocessing-track comparison (superseded)

An earlier aligned-versus-unaligned comparison (`results/evaluation/baseball/config_comparison.csv`) reported the unaligned track at 78 terms / whole-state R2 −4.8905 and the aligned track at 98 terms / whole-state R2 −2.4099. That comparison predates both the restored 140-clip pool and the core-position metric, and it ranked tracks on the whole-state score that Section 5.5 shows is not comparable across configurations. It is retained for history and is **not** the basis of the current operative choice. The alignment track has not been re-scored under the current pool and metric; doing so is listed in Section 10.

## 7. Interpretation

The implementation satisfies the technical feasibility requirement: baseball pose clips can be converted into reproducible processed trajectories, feature matrices, derivatives, sparse equations, coefficient tables, and held-out evaluation outputs through scripted commands, and the results in Section 6 regenerate from the commands in Section 8.

The modeling results are a negative result, and should be stated as one rather than described as "weak." Negative R2 does not mean low accuracy; it means the model is **worse than predicting the mean of the target window**. On held-out clips:

- one-step derivative R2 is approximately zero (−0.0184), so the model captures essentially none of the frame-to-frame variance;
- free-running core-position reconstruction R2 is −29.56 for the operative model, and no configuration reaches the −5.57 scored by a model with no dynamics at all.

The defensible claim is:

> A reproducible sparse-dynamics pipeline for baseball pitching pose data was implemented and validated end-to-end, together with a state-comparable reconstruction metric and a frozen-model control. Under that protocol, no fitted configuration — first-order, second-order, ensemble, or weak-form — recovers dynamics that transfer to held-out clips: all are outperformed by a zero-dynamics baseline, and held-out reconstruction accuracy degrades monotonically with model complexity. The contribution is the reproducible framework, the evaluation protocol including the control that makes the metric interpretable, and a documented negative result establishing that a single global sparse model over heterogeneous broadcast-derived pose is not sufficient.

The results must not be presented as evidence of recovered governing laws of pitching mechanics, and the weak-form configuration must not be presented as a successful model.

A plausible mechanism for the failure is that a single global linear model is being asked to describe 140 clips from different pitchers with different mechanics, timings, and body scales, using pose that carries projection and estimation noise of the same order as the biomechanical signal. Error then compounds across a 40-frame free run. This is consistent with the near-zero one-step derivative R2: the model is not even locally accurate, so the failure is not solely one of error accumulation.

## 8. Reproducibility

All commands are run from the project root with the repository virtual environment. **The operative configuration must be passed explicitly** — the refresh script's default (`config/preprocessing.baseball.example.yaml`) does not reproduce Section 6.

Install dependencies:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Validate Python syntax:

```bash
./.venv/bin/python -m compileall src
```

Preview the operative baseball pipeline:

```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml --dry-run
```

Reproduce Section 6.1 (operative first-order train + held-out evaluation):

```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce Section 6.3 (weak-form model). Note that fit and evaluation write to the same
`results/equations/baseball/train/` and `results/evaluation/baseball/train_on_test/` paths for
every configuration, so snapshot or re-run the baseline afterwards if both are needed:

```bash
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py \
  --config config/preprocessing.baseball.unaligned.handedness.weakform.yaml \
  --split train --task-types pitching
./.venv/bin/python src/preprocessing/evaluate_baseball_sindy.py \
  --config config/preprocessing.baseball.unaligned.handedness.weakform.yaml \
  --train-split train --eval-split test --task-types pitching
```

Reproduce the frozen control in Section 6.2:

```bash
./.venv/bin/python src/preprocessing/evaluate_static_baseline.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml --eval-split test
```

Compare multiple configurations in one run:

```bash
./.venv/bin/python src/preprocessing/compare_baseball_configs.py \
  --configs config/preprocessing.baseball.unaligned.handedness.yaml \
            config/preprocessing.baseball.unaligned.handedness.weakform.yaml \
  --output-csv results/evaluation/baseball/config_comparison_weakform.csv
```

Primary generated outputs:

- `data/baseball/processed/train/features/manifest.csv`
- `data/baseball/processed/test/features/manifest.csv`
- `results/equations/baseball/train/summary.csv`
- `results/equations/baseball/train/pitching/coefficients.csv`
- `results/equations/baseball/train/pitching/equations.txt`
- `results/equations/baseball/train/pitching/model_selection.csv`
- `results/evaluation/baseball/train_on_test/summary.csv`
- `results/evaluation/baseball/train_on_test/reconstruction_metrics.csv`
- `results/evaluation/baseball/static_baseline/summary.csv`
- `results/evaluation/baseball/model_comparison_2026-08-12.csv`

## 9. Limitations

- The dataset is 140 clips from a single camera view, and may not cover enough athlete, view, and motion variability to support a global model.
- MediaPipe-derived pose from broadcast or segmented video is noisier than marker-based motion capture.
- Many clips are effectively 2D pose observations, which limits interpretation of out-of-plane motion.
- Camera angle, zoom, and occlusion may change feature values independently of true biomechanics.
- The polynomial libraries tried are low-degree; the second-order variants that add expressivity performed far worse on held-out reconstruction, so underfitting is not the binding constraint on the current data.
- Negative held-out R2 across all configurations means no configuration has demonstrated predictive validity.
- The `sparsity_weighted_r2` selection rule degenerates to a term-count minimizer when candidate accuracies are indistinguishable (Section 6.3); selected sparsity levels should not be read as evidence of a parsimonious true model.
- Results come from one train/test split with no confidence intervals, so the reported differences between configurations are not accompanied by uncertainty estimates.
- The system identifies sparse equations from measured features; it does not prove causal biomechanical mechanisms.

## 10. Next Work

1. Report the frozen control alongside every reconstruction number in all future comparisons, and treat "beats the frozen control" as the minimum bar for a model to be described as working.
2. Replace or repair the model-selection rule so that it does not reduce to term-count minimization when candidate accuracies are statistically indistinguishable; require a minimum accuracy margin before the sparsity penalty is allowed to decide.
3. Report confidence intervals across repeated train/test splits rather than relying on one split.
4. Re-score the alignment track under the current pool and the core-position metric, so the alignment decision rests on the comparable metric (Section 6.4).
5. Add automated pose-quality metrics and exclude low-confidence or heavily occluded clips.
6. Test per-pitcher or per-phase models instead of a single global fit, since clip heterogeneity is the leading candidate explanation for the failure.
7. Add figures showing representative observed versus reconstructed feature trajectories, with the frozen control drawn for reference.
8. Pursue the label-validated representation study (`docs/superpowers/specs/2026-06-23-representation-study-design.md`), which asks whether the representations carry structure predictive of release speed and pitch type even though reconstruction fails.
9. Add a References section crediting the methods used (SINDy, weak-form SINDy, MediaPipe, MLB-YouTube, UTD-MHAD).

## 11. Proposed Contribution

This work contributes a reproducible software framework for sparse dynamical modeling of baseball pitching motion from standardized pose trajectories, together with an evaluation protocol for it. The engineering contribution is an end-to-end pipeline that transforms vision-derived skeletal motion into interpretable sparse-equation artifacts with documented preprocessing, feature extraction, model fitting, and evaluation, reproducible from scripted commands.

The methodological contribution is the evaluation protocol: a state-comparable core-position reconstruction metric that permits honest comparison across different state-vector definitions, and a frozen-model control that makes the metric interpretable. The control materially changed a conclusion in this project, converting an apparent 3.7x improvement from the weak-form configuration into a correctly identified artifact of near-null dynamics (Section 6.3).

The empirical contribution is a documented negative result: across first-order, second-order, ensemble, and weak-form configurations on 140 single-view broadcast-derived pitching clips, no fitted sparse model transfers to held-out clips, and all are outperformed by a zero-dynamics baseline. This establishes the data-quality and modeling requirements that must be met before discovered equations can be interpreted as biomechanics.
