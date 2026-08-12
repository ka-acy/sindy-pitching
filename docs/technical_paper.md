# Sparse Dynamical Modeling of Baseball Pitching Kinematics from Vision-Derived Pose

## Abstract

This paper develops and validates a reproducible computational pipeline for converting human motion trajectories into sparse nonlinear dynamical models, together with an evaluation protocol in which every reported score is paired with the no-model control for its own task. The workflow uses standardized skeletal pose, biomechanical feature extraction, derivative estimation, and Sparse Identification of Nonlinear Dynamical Systems (SINDy). Stage A validates the pipeline on the UTD-MHAD human action dataset as computational validation only. Stage B applies the workflow to 140 MediaPipe-derived MLB pitching clips from a single broadcast view (111 train / 29 held-out test).

The central result is a separation between two tasks that prior evaluations of this pipeline conflated. **Free-running multi-step reconstruction fails**: no fitted configuration — first-order, second-order, ensemble, or weak-form — outperforms a frozen zero-dynamics predictor, and held-out reconstruction accuracy degrades monotonically with model complexity. **Short-horizon prediction succeeds, but not for the sparse models**: a delay-embedded linear model achieves 0.39 skill over persistence on held-out one-step increments, robust to the choice of error aggregation, and feature increments are strongly autocorrelated (lag-1 0.35 to 0.88) with signal-to-noise 2.2 to 4.4. The fitted STLSQ equations, however, retain between 0 percent (first-order) and 12 percent (second-order, acceleration) of the skill present in their own training targets. The short-horizon structure is therefore a property of the data, and the loss is localized to the sparse fit rather than to the pose, the features, the derivative estimation, or the metric.

Four intermediate results in this project were false positives that survived only because the corresponding control or aggregation was missing, and each is documented here alongside the check that overturned it. The contributions are therefore the reproducible framework, the control-paired evaluation protocol, and a characterized boundary between what this data supports (short-horizon predictive structure), what the pipeline's sparse artifacts currently extract from it (almost none), and what is not achievable with these methods (free-running reconstruction, and biomechanical interpretation of discovered equations).

## 1. Introduction

Baseball pitching is a high-speed, multi-segment movement involving coordinated lower-body drive, trunk rotation, shoulder and elbow motion, stride mechanics, and rapid distal segment acceleration. Traditional biomechanical analysis often relies on laboratory motion capture, which produces high-quality trajectories but is expensive and difficult to scale. Broadcast or locally segmented video is more accessible, but vision-derived pose introduces noise, camera-view bias, missing landmarks, and two-dimensional projection limits.

This project investigates whether a machine-vision and sparse-dynamics workflow can produce reproducible mathematical artifacts from baseball pitching motion. The goal is not to claim that the current dataset explains elite pitching mechanics. The goal is to establish a defensible technical pipeline that can ingest pose trajectories, condition them into stable feature matrices, estimate derivatives, and fit sparse dynamical models that can be compared across preprocessing and modeling configurations under a common evaluation protocol.

## 2. Research Questions

1. Can raw or vision-derived skeletal trajectories be converted into standardized baseball pitching feature matrices without manual per-clip intervention?
2. Can derivative matrices and sparse coefficient matrices be regenerated reproducibly from those feature matrices?
3. Are the fitted SINDy systems numerically stable on held-out baseball clips?
4. Do any of the fitted systems outperform a no-dynamics control baseline on held-out free-running reconstruction?
5. Do they outperform the corresponding no-model control on short-horizon prediction, and does the answer differ from RQ4?
6. Is the limiting factor the model class, the state representation, the clip heterogeneity, or the pose noise floor?

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

Event alignment is **disabled** in the operative configuration (`features.alignment.enabled: false`). An earlier configuration track aligned pitching clips around stride-ankle separation with lead-knee lift as a secondary signal; that track is retained as `config/preprocessing.baseball.aligned.yaml` but is not the operative default. Section 6.9 records why.

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

**Short-horizon prediction and skill over persistence.** Free-running reconstruction and one-step prediction are different tasks with non-comparable scores, and each needs its own control. For one-step prediction the correct control is not the frozen model but **persistence** — predict `x(t) = x(t-1)`. At approximately 60 fps consecutive frames are nearly identical, so persistence is a very strong baseline: it alone scores R2 = 0.968 on the state. State R2 is therefore saturated by the near-identity component and cannot separate models. The reported metric for this task is instead **skill over persistence** on the increment:

```
skill = 1 - SSE(model) / SSE(zero-increment)
```

where the target is `delta(t) = x(t+1) - x(t)`. Skill is 0 for persistence by construction, and positive skill means predictive structure beyond "nothing changes."

**Aggregation is reported two ways, and disagreement is treated as a null result.** Core positions span roughly a 13x range in per-frame signal scale (`lead_knee_lift` 0.012 versus `dominant_elbow_flexion_angle` 0.158, about 160x in variance), so pooled sum-of-squares is decided almost entirely by the largest-amplitude columns while uniform per-column averaging weights all eight equally. Every skill number in Section 6 is reported under both. Where the two disagree in sign, this paper reports no win rather than selecting the favorable aggregation. `core_position_metrics` uses uniform averaging, so uniform is the figure comparable to the rest of the project.

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

### 6.4 State-representation sweep: what does not help

Four candidate explanations for the near-zero derivative fit were tested directly, using plain ridge regression rather than STLSQ so that sparse-selection artifacts cannot confound the comparison, with GroupKFold over clips so no clip appears in both train and test (`src/preprocessing/diagnose_derivative_fit.py`, `results/analysis/derivative_fit_diagnostics.csv`):

| representation | held-out derivative R2 | reading |
|---|---:|---|
| `global_linear` — [1, x] | −0.0004 | structural match to the operative degree-1 SINDy |
| `global_linear_phase` — [1, x, φ] | −0.0003 | no effect |
| `global_phase_interact` — A(φ)x | +0.0071 | no meaningful gain |
| `phase_binned_k4` / `k8` | +0.0019 / −0.0008 | no gain |
| `delay_embed` τ=1, d=3 | **+0.6901** | large apparent gain |
| `finite_diff_control` (unfitted backward difference) | +0.3507 | **half the gain is rediscovered differentiation** |
| `per_clip_within` (fit early 70%, predict late 30%) | −38.89 | within-clip temporal extrapolation fails |
| `per_clip_oracle` (in-sample) | +0.2500 | upper bound for a per-clip model |

Two conclusions follow.

**The non-autonomous hypothesis is refuted.** Adding normalized phase φ — as an additive term, as a modulating interaction A(φ)x, and as piecewise-in-phase bins — produces no gain in any form. The failure of `dx/dt = f(x)` is not explained by the map being multivalued in phase.

**The state representation, not clip heterogeneity, is the binding constraint.** The per-clip oracle scores only +0.25 *in sample*, lower than a delay-embedded model scores on *held-out* clips (+0.69). A single clip cannot be fit well in the plain 10-D state even when the model is allowed to see it. This rules out the per-pitcher, per-pitch-type, and per-phase conditioning directions, which all address heterogeneity.

Delay embedding's apparent gain must be discounted: an unfitted backward difference already reaches +0.3507, so roughly half of the +0.69 is numerical differentiation recovered from position history rather than learned dynamics.

### 6.5 Short-horizon prediction: the one task that succeeds

One-step held-out prediction on the 29 test clips (`src/preprocessing/diagnose_increment_skill.py`, `results/analysis/increment_skill/part_a_parameterization.csv`):

| model | target | skill (pooled) | skill (uniform) | R2 (state) |
|---|---|---:|---:|---:|
| persistence (no model) | — | 0.0000 | 0.0000 | 0.9680 |
| delay ridge | next state x(t+1) | 0.2815 | **0.3871** | 0.9777 |
| delay ridge | increment x(t+1) − x(t) | 0.2816 | **0.3874** | 0.9777 |

**This is the project's only positive result that survives its control, and it is positive under both aggregations.** The model captures approximately 39 percent of held-out increment variance beyond persistence.

The `R2 (state)` column shows why every earlier evaluation missed it: 0.9777 against persistence's 0.9680. A 39 percent improvement in the predicted dynamics appears as a third-decimal difference once it is measured on the state, because the state metric is dominated by the near-identity component. This is a metric-selection failure, not a modeling one.

The increment parameterization itself contributed nothing (0.2815 versus 0.2816). `RidgeCV` selects alpha of order 1e-4, so shrinkage is negligible, and the two targets span the same linear space. The argument that predicting the next state penalizes the near-identity map is correct in principle but empirically void at this regularization strength. Recording this because the reparameterization was predicted in advance to be the source of the improvement, and it was not — the improvement came entirely from changing the metric.

### 6.6 Multi-step rollout: no win, and the aggregations disagree

The same increment model rolled out autoregressively (`results/analysis/increment_skill/part_b_horizon_crossover.csv`):

| H (frames) | H (sec) | clips | skill (pooled) | skill (uniform) | core R2 model | core R2 persistence |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.083 | 29 | +0.1751 | −68.15 | −354.00 | −8.38 |
| 10 | 0.167 | 29 | +0.1906 | −23.47 | −110.86 | −6.76 |
| 20 | 0.334 | 29 | +0.2145 | −24.82 | −68.56 | −5.92 |
| 40 | 0.667 | 29 | +0.1269 | −17.19 | −43.36 | −5.63 |
| 80 | 1.335 | 28 | +0.1385 | −7.72 | −33.60 | −5.14 |
| 160 | 2.669 | 21 | +0.1517 | −4.13 | −9.30 | −3.16 |
| 240 | 4.004 | 13 | +0.1438 | −0.4955 | −1.87 | −2.73 |

Pooled skill is positive at every horizon; uniform skill is negative at every horizon, and uniform core R2 favours persistence at every horizon. Per Section 5.5 this is reported as **no multi-step win**. The disagreement is fully explained by the amplitude spread across core columns: pooled sum-of-squares is decided by the large-amplitude angles, on which the model does help, while the small-amplitude columns on which it does badly count equally under uniform averaging.

The H=240 row is the only sign of the crossover one would expect as persistence degrades — core R2 −1.87 against persistence −2.73 — but uniform skill is still negative there and only 13 clips are long enough to qualify. It is recorded as a direction to test with longer clips, not as a result.

A separate verification confirms the rollout failure is genuine rather than an implementation artifact. Teacher-forcing the same model, so true history is supplied at each step and no error compounds, scores +0.7569 core R2 against persistence's +0.8453 (`results/evaluation/baseball/forecast_baselines/summary.csv`). The map is sound; free-running divergence is real error compounding. Note that this teacher-forced score is itself below persistence, so it is not an additional positive result — it is only evidence that `rollout_delay` is correct.

For completeness, the free-running comparison against all controls under a common seed and scoring window:

| method | core R2 | reading |
|---|---:|---|
| frozen (no dynamics) | **−5.56** | best free-running method |
| `delay_linear` rolled out | −42.08 | the +0.69 derivative representation |
| `mean_pitch` template | −45.01 | phase-warped mean of the training clips |
| `constant_velocity` | −1096.28 | extrapolation from the seed velocity |

The `mean_pitch` result is worth noting on its own: a template asserting that every pitch follows the average training trajectory scores −45.01, so at this pose quality pitching is not stereotyped enough for template matching either.

### 6.7 Noise floor: not the binding constraint

Per-feature signal and predictability across all 140 clips (`results/analysis/increment_skill/part_c_noise_floor.csv`):

| feature | core | signal/frame | HF residual | SNR | ACF₁(Δ) |
|---|:--:|---:|---:|---:|---:|
| `lead_knee_lift` | yes | 0.0124 | 0.0028 | **4.44** | **0.8816** |
| `stride_ankle_separation` | yes | 0.0167 | 0.0050 | 3.35 | 0.7316 |
| `hip_rotation_angle` | yes | 0.1131 | 0.0355 | 3.19 | 0.7716 |
| `shoulder_rotation_angle` | yes | 0.1145 | 0.0386 | 2.97 | 0.7429 |
| `trail_knee_flexion_angle` | yes | 0.1184 | 0.0429 | 2.76 | 0.6247 |
| `dominant_knee_flexion_angle` | yes | 0.1270 | 0.0488 | 2.60 | 0.5795 |
| `trail_elbow_flexion_angle` | yes | 0.1529 | 0.0665 | 2.30 | 0.4523 |
| `hip_shoulder_separation_angle` | no | 0.0758 | 0.0330 | 2.30 | 0.5395 |
| `dominant_elbow_flexion_angle` | yes | 0.1576 | 0.0690 | 2.28 | 0.4303 |
| `dominant_wrist_speed` | no | 0.7066 | 0.3184 | 2.22 | 0.3509 |

Increment lag-1 autocorrelation ranges from 0.35 to 0.88, so increments are far from white and carry genuinely predictable structure; signal-to-noise is 2.2 to 4.4 throughout. **The noise floor does not explain the failures in Sections 6.2 and 6.6**, and short-horizon prediction is not capped by pose quality at this level. Higher-quality pose remains desirable for biomechanical interpretation, but it is not a precondition for the Section 6.5 result.

The residual estimate is an upper bound on achievable precision, not a measurement of raw MediaPipe jitter: the features are already Savitzky-Golay smoothed, so this measures leftover high-frequency content in the modeled features. It is a floor on the floor.

### 6.8 Do the pipeline's own sparse equations capture it?

Section 6.5 established short-horizon structure using a delay-embedded ridge model, which is not what this pipeline produces. The question that determines how broadly the result can be claimed is whether the fitted STLSQ equations capture that structure. `src/preprocessing/evaluate_sindy_increment_skill.py` scores the fitted models on the identical task, with a `savgol_derivative_reference` row giving the ceiling any model fit to those derivative targets could reach.

Uniform skill over persistence, 29 held-out clips:

| model | task | skill | ceiling | fraction of ceiling |
|---|---|---:|---:|---:|
| first-order SINDy (operative) | position increments | **−0.0010** | 0.7032 | **0%** |
| second-order SINDy | position increments | 0.8905 | 0.8908 | ~100% (see below) |
| second-order SINDy | rate increments (acceleration) | **0.0826** | 0.6911 | **12%** |
| delay ridge (not SINDy) | position increments | 0.3874 | — | — |

**The answer is no.** The operative first-order model has no short-horizon skill whatever, despite the derivative targets it is fit against containing 0.70 of extractable skill. It discards all of it.

**The second-order position-increment score of 0.8905 is not a result.** Velocity is an input column of the 16-D state, and the identity gate confirms each position derivative loads on its paired rate column, so `d(position)/dt` is read off the input rather than learned. The model ties its own reference to within 0.0003 — the signature of an input-copying identity, not a fitted dynamic. This was anticipated before the run and the control was built in, which is why it was caught immediately rather than reported.

On the non-trivial task for that state — predicting rate increments, i.e. acceleration — the second-order model captures 12 percent of its ceiling. Positive, but small, and it is the same configuration that fails catastrophically on free-running reconstruction (core R2 −2269, Section 6.2).

Two consequences for what this paper may claim. First, **the Section 6.5 positive result does not belong to the SINDy pipeline**; it belongs to the data and to a delay-embedded ridge baseline. The pipeline's sparse artifacts capture between 0 and 12 percent of the structure available in their own training targets. Second, this localizes the loss precisely: it is not the pose, not the features, not the derivative estimation, and not the metric — all of those carry 0.69 to 0.70 skill. It is the sparse fit and its selection rule, which is consistent with the degeneracy documented in Section 6.3, where all sweep candidates were accuracy-indistinguishable and selection collapsed to term-count minimization.

Skill figures are comparable *within* a row's own reference, not across the first-order and second-order rows: the two configurations compute positions and derivatives with different Savitzky-Golay paths, so their targets and therefore their ceilings differ (0.7032 versus 0.8908 on the same eight core columns).

### 6.9 Preprocessing-track comparison (superseded)

An earlier aligned-versus-unaligned comparison (`results/evaluation/baseball/config_comparison.csv`) reported the unaligned track at 78 terms / whole-state R2 −4.8905 and the aligned track at 98 terms / whole-state R2 −2.4099. That comparison predates both the restored 140-clip pool and the core-position metric, and it ranked tracks on the whole-state score that Section 5.5 shows is not comparable across configurations. It is retained for history and is **not** the basis of the current operative choice. The alignment track has not been re-scored under the current pool and metric; doing so is listed in Section 10.

## 7. Interpretation

The implementation satisfies the technical feasibility requirement: baseball pose clips can be converted into reproducible processed trajectories, feature matrices, derivatives, sparse equations, coefficient tables, and held-out evaluation outputs through scripted commands, and the results in Section 6 regenerate from the commands in Section 8.

The modeling results separate cleanly into two tasks that earlier evaluations of this pipeline conflated.

**Free-running reconstruction fails, and this should be stated as a negative result rather than described as "weak."** Negative R2 does not mean low accuracy; it means the model is worse than predicting the mean of the target window. No configuration reaches the −5.57 scored by a model with no dynamics at all, held-out accuracy degrades monotonically with model complexity, and the failure is not attributable to phase, to clip heterogeneity, or to the pose noise floor — each was tested and excluded in Sections 6.4 and 6.7.

**Short-horizon prediction succeeds, and the pipeline does not exploit it.** A delay-embedded linear model achieves 0.39 skill over persistence on held-out one-step increments, positive under both aggregations, on a state whose increments are strongly autocorrelated (0.35 to 0.88). The dynamics are locally learnable. But Section 6.8 shows the fitted sparse equations retain 0 to 12 percent of the skill available in their own derivative targets, so the local structure is real and the pipeline discards it.

These are consistent, not contradictory. Section 6.6 shows the mechanism directly: teacher-forcing the same model — supplying true history so no error compounds — recovers a sound map, while its autoregressive rollout diverges. The pipeline recovers local structure and loses it to error compounding over 40 frames.

The defensible claim is:

> A reproducible sparse-dynamics pipeline for baseball pitching pose data was implemented end-to-end, together with a state-comparable reconstruction metric and a control-paired evaluation protocol in which every reported score is compared against the no-model baseline for its own task. Under that protocol three findings separate. First, no fitted configuration — first-order, second-order, ensemble, or weak-form — recovers dynamics that survive free-running reconstruction; all are outperformed by a zero-dynamics baseline, and accuracy degrades monotonically with model complexity. Second, the data does carry short-horizon structure: a delay-embedded linear model achieves 0.39 skill over persistence on held-out increment prediction, robust to error aggregation and backed by increment autocorrelation of 0.35 to 0.88 rather than by pose noise. Third, the pipeline's own sparse equations capture almost none of that structure, retaining 0 percent (first-order) to 12 percent (second-order, acceleration) of the skill present in the derivative targets they are fit against. The substantive finding is this boundary together with the localization of the loss to the sparse fit and its selection rule, not to the pose, the features, the derivative estimation, or the metric.

Two things this work must not claim. The results are not evidence of recovered governing laws of pitching mechanics: 26 coefficients that predict increments 39 percent better than persistence are not a biomechanical model, and the discovered equations have not been validated against any external quantity. And the weak-form configuration must not be presented as a successful model.

**Methodological finding on evaluation.** Three intermediate results in this project were false positives, each caught only by adding the no-model control for the specific task being scored:

| apparent result | control that overturned it | outcome |
|---|---|---|
| weak-form core R2 −7.89, "3.7x better than baseline" | frozen zero-dynamics model, −5.57 | dissolved; the model had one non-bias term |
| delay-embedding derivative R2 +0.69 | unfitted backward difference, +0.35 | roughly half was numerical differentiation |
| teacher-forced one-step core R2 +0.76 | persistence, +0.85 | dissolved |
| second-order short-horizon skill +0.89 | its own derivative reference, +0.89 | dissolved; velocity is an input column |

A further near-miss was an aggregation artifact rather than a missing control: pooled skill is positive at every rollout horizon while uniform skill is negative at every horizon (Section 6.6), and reporting only the pooled figure would have produced another spurious positive. The fourth row above was anticipated before the run — position-increment prediction is trivial when velocity is a state variable — so the control was built in and the artifact was caught on first inspection rather than after it had been written up.

The common structure is that all four involved metrics whose absolute values are uninterpretable without a reference point — R2 against a window mean when the target barely moves, and sum-of-squares pooled across columns of very different amplitude. The practice adopted here, and the one this project would recommend to similar work, is that a metric ships with its no-model control and both aggregations before any model is scored against it, not after a result looks surprising.

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

Reproduce the state-representation sweep in Section 6.4:

```bash
./.venv/bin/python src/preprocessing/diagnose_derivative_fit.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce Sections 6.5, 6.6 and 6.7 (one-step skill, horizon sweep, noise floor):

```bash
./.venv/bin/python src/preprocessing/diagnose_increment_skill.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce the free-running control comparison in Section 6.6:

```bash
./.venv/bin/python src/preprocessing/evaluate_forecast_baselines.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce Section 6.8 (fitted SINDy equations on the short-horizon task):

```bash
./.venv/bin/python src/preprocessing/evaluate_sindy_increment_skill.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml \
  --model-root results/equations/baseball/train/pitching
```

For the second-order state, extract to an isolated feature subdir first so the 10-D features
every other result depends on are not overwritten, then score the non-trivial task:

```bash
for split in train test; do
  ./.venv/bin/python src/preprocessing/extract_baseball_features.py \
    --config config/preprocessing.baseball.unaligned.second_order.isolated.yaml --split $split
done
./.venv/bin/python src/preprocessing/evaluate_sindy_increment_skill.py \
  --config config/preprocessing.baseball.unaligned.second_order.isolated.yaml \
  --model-root results/equations/baseball/train_second_order_2026-06-23/pitching \
  --column-set rates
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
- `results/evaluation/baseball/forecast_baselines/summary.csv`
- `results/analysis/derivative_fit_diagnostics.csv`
- `results/analysis/increment_skill/part_a_parameterization.csv`
- `results/analysis/increment_skill/part_b_horizon_crossover.csv`
- `results/analysis/increment_skill/part_c_noise_floor.csv`
- `results/analysis/sindy_increment_skill.csv`
- `results/analysis/sindy_increment_skill_second_order_rates.csv`

## 9. Limitations

- The dataset is 140 clips from a single camera view, and may not cover enough athlete, view, and motion variability to support a global model.
- MediaPipe-derived pose from broadcast or segmented video is noisier than marker-based motion capture.
- Many clips are effectively 2D pose observations, which limits interpretation of out-of-plane motion.
- Camera angle, zoom, and occlusion may change feature values independently of true biomechanics.
- The polynomial libraries tried are low-degree; the second-order variants that add expressivity performed far worse on held-out reconstruction, so underfitting is not the binding constraint on the current data.
- No configuration has demonstrated predictive validity on free-running reconstruction. The Section 6.5 result establishes short-horizon predictive structure only, and does not license claims about longer horizons.
- The Section 6.5 skill figure comes from a delay-embedded ridge model, not from a sparse SINDy fit. Section 6.8 tests the fitted equations directly and finds they capture 0 percent (first-order) to 12 percent (second-order, acceleration) of the structure available in their own training targets. The short-horizon result therefore characterizes the data, not the pipeline's sparse artifacts, and must not be attributed to SINDy.
- Skill over persistence measures prediction of increments. It is not a biomechanical validity measure, and 0.39 skill does not imply the fitted coefficients correspond to real biomechanical relationships.
- The H=240 horizon result rests on the 13 clips long enough to qualify, and is reported as a direction rather than a finding.
- The noise-floor estimate is computed on already-smoothed features, so it bounds achievable precision in the modeled state rather than measuring raw MediaPipe jitter.
- The `sparsity_weighted_r2` selection rule degenerates to a term-count minimizer when candidate accuracies are indistinguishable (Section 6.3); selected sparsity levels should not be read as evidence of a parsimonious true model.
- Results come from one train/test split with no confidence intervals, so the reported differences between configurations are not accompanied by uncertainty estimates.
- The system identifies sparse equations from measured features; it does not prove causal biomechanical mechanisms.

## 10. Next Work

1. Make short-horizon increment prediction the pipeline's primary reported task, since it is the one the data supports, and carry skill over persistence into `evaluate_baseball_sindy.py` so it is emitted on every run rather than by a separate diagnostic.
2. Repair the sparse fit, which Section 6.8 localizes as where the signal is lost: the targets carry 0.69 to 0.70 skill and the fitted equations retain 0 to 12 percent of it. Diagnose whether the loss is STLSQ thresholding, the degenerate selection rule of Section 6.3, or feature scaling, by sweeping each while holding the others fixed and scoring skill over persistence rather than derivative R2.
3. Report the frozen and persistence controls alongside every score in all future comparisons, with both aggregations, and treat sign disagreement between aggregations as a null result.
4. Replace or repair the model-selection rule so that it does not reduce to term-count minimization when candidate accuracies are statistically indistinguishable; require a minimum accuracy margin before the sparsity penalty is allowed to decide.
5. Report confidence intervals across repeated train/test splits rather than relying on one split.
6. Extend the horizon sweep with longer clips to test whether the crossover hinted at H=240 is real.
7. Re-score the alignment track under the current pool and the core-position metric (Section 6.8).
8. Add automated pose-quality metrics and exclude low-confidence or heavily occluded clips.
9. Add figures: observed versus reconstructed trajectories with the frozen control drawn for reference, and a skill-versus-horizon curve with persistence at zero.
10. Pursue the label-validated representation study (`docs/superpowers/specs/2026-06-23-representation-study-design.md`), which asks whether these representations predict release speed and pitch type. Given Section 6.5, the relevant question is now whether the short-horizon structure is the same structure that carries label information.
11. Add a References section crediting the methods used (SINDy, weak-form SINDy, MediaPipe, MLB-YouTube, UTD-MHAD).

Explicitly deprioritized on the evidence in Section 6.4: per-pitcher, per-pitch-type, and per-phase conditioning (the per-clip oracle result shows heterogeneity is not the constraint); non-autonomous or phase-augmented state vectors (no gain in three forms); and further model-class substitution on the current feature set.

## 11. Proposed Contribution

This work contributes a reproducible software framework for sparse dynamical modeling of baseball pitching motion from standardized pose trajectories, together with an evaluation protocol for it. The engineering contribution is an end-to-end pipeline that transforms vision-derived skeletal motion into interpretable sparse-equation artifacts with documented preprocessing, feature extraction, model fitting, and evaluation, reproducible from scripted commands.

The methodological contribution is the control-paired evaluation protocol: a state-comparable core-position reconstruction metric, a frozen-model control and a persistence control that make it interpretable, dual aggregation with sign disagreement treated as a null, and the discipline of matching each control to the specific task being scored. This protocol changed four conclusions in this project — three apparent positive results proved to be artifacts, and one real result had been invisible under the metric previously used (Sections 6.3, 6.4, 6.5, 6.6). That track record is the strongest argument for the protocol.

The empirical contribution is a characterized boundary rather than a flat negative, with the failure localized. Across first-order, second-order, ensemble, and weak-form configurations on 140 single-view broadcast-derived pitching clips, no fitted sparse model survives free-running reconstruction, and all are outperformed by a zero-dynamics baseline. Yet the same data supports short-horizon prediction: 0.39 skill over persistence on held-out increments, backed by increment autocorrelation of 0.35 to 0.88 and signal-to-noise of 2.2 to 4.4. Candidate explanations were tested and excluded one at a time — phase (no gain in three forms), clip heterogeneity (the per-clip oracle scores below a cross-clip delay model), the pose noise floor (SNR 2.2 to 4.4), and the metric (the derivative targets carry 0.69 to 0.70 skill). What remains is the sparse fit itself, which retains 0 to 12 percent of the skill in its own targets. That is a specific, testable diagnosis rather than a general negative.

The practical value of the boundary is that it says what to build next, and what not to. Short-horizon prediction is a task this data supports and is where a defensible biomechanical claim would have to start, but the sparse-fitting stage must be repaired before the pipeline can make that claim on its own artifacts. Free-running reconstruction of broadcast-derived pitching pose is not achievable with these methods and should not be the target. Better pose, larger pools, richer libraries, and per-pitcher conditioning are all secondary to the sparse fit, since none of them is where the measured loss occurs.
