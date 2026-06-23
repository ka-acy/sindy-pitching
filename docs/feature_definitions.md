# Feature Definitions

This document covers both the **operative first-order 10-D state vector** (production
baseline as of 2026-06-23) and the **experimental 16-D second-order state vector**
(implemented but NOT adopted — see outcome below).

---

## 1. Operative State Vector: First-Order 10-D (Production Baseline)

The current production baseline for Stage B uses a 10-feature biomechanical
representation computed from processed 15-joint skeleton clips.  This is the
state vector in configs that do NOT set `features.second_order: true`.

### 1.1 Feature Order

The operative config (`config/preprocessing.baseball.unaligned.handedness.yaml`)
defines this 10-feature `state_vector`:

| # | Name |
|---|------|
| 1 | `shoulder_rotation_angle` |
| 2 | `hip_rotation_angle` |
| 3 | `hip_shoulder_separation_angle` |
| 4 | `dominant_elbow_flexion_angle` |
| 5 | `trail_elbow_flexion_angle` |
| 6 | `dominant_knee_flexion_angle` |
| 7 | `trail_knee_flexion_angle` |
| 8 | `lead_knee_lift` |
| 9 | `stride_ankle_separation` |
| 10 | `dominant_wrist_speed` |

> Note: the SINDy fit producing the 26-term baseline (test recon R² = −2.768,
> test recon RMSE = 0.719) uses this 10-D state. `hip_shoulder_separation_angle`
> and `dominant_wrist_speed` remain part of the operative first-order baseline —
> they are dropped only from the *experimental* second-order redesign (Section 2).

### 1.2 Geometric Definitions

All features are computed by `build_feature_series` in
`src/preprocessing/extract_utd_mhad_features.py`.

- `shoulder_rotation_angle`: azimuth of the left-shoulder-minus-right-shoulder axis
  projected into the x-z plane (`arctan2(z, x)`).
- `hip_rotation_angle`: azimuth of the left-hip-minus-right-hip axis projected into
  the x-z plane.
- `hip_shoulder_separation_angle`: `wrap_angle(shoulder_rotation_angle − hip_rotation_angle)`
  — the torso coil between the shoulder and hip rotation planes (see Section 2.1 for
  why it is dropped from the second-order redesign).
- `dominant_wrist_speed`: linear speed magnitude of the dominant-side wrist joint
  (`velocity_magnitude`), mapped from `right_wrist_speed`/`left_wrist_speed` by resolved
  handedness.
- `dominant_elbow_flexion_angle`: angle between the dominant-side upper-arm and forearm
  segments (`angle_between`), mapped from `right_elbow_flexion_angle` or
  `left_elbow_flexion_angle` depending on resolved handedness.
- `trail_elbow_flexion_angle`: same for the trail (non-throwing) side.
- `dominant_knee_flexion_angle`: angle between the dominant-side thigh and shank
  segments, mapped from resolved handedness.
- `trail_knee_flexion_angle`: same for the trail (glove/stance) side.
- `lead_knee_lift`: hip y-coordinate minus knee y-coordinate on the lead (stride) leg —
  positive when the knee is lifted.
- `stride_ankle_separation`: Euclidean distance between left and right ankle positions
  in 3-D space.

### 1.3 Derivatives

`dX/dt` is estimated by `estimate_derivatives` in `extract_utd_mhad_features.py`:

- Angle columns (those in `ANGLE_FEATURES`) are unwrapped before differentiation.
- Savitzky-Golay first-derivative filter: `window_length=9`, `polyorder=3`.
- `delta=dt` from per-clip preprocessing output.
- Falls back to `np.gradient` for clips shorter than the window.

### 1.4 Artifacts

Per-clip feature payloads are `.npz` files containing:

- `X`: state matrix, shape `(T, 10)`.
- `X_dot`: derivative matrix, shape `(T, 10)`.
- `feature_names`: array of the 10 name strings.
- `metadata`: JSON blob including `resolved_handedness`, `alignment`, derivative params.

### 1.5 Handedness Canonicalization

`canonicalize_feature_series` in `extract_baseball_features.py` maps raw
left/right columns to dominant/trail/lead roles based on `resolved_handedness`.

**Lefty sign-flip:** for left-handed pitchers, `trunk_rotation_angle`,
`shoulder_rotation_angle`, `hip_rotation_angle`, `hip_shoulder_separation_angle`,
and `trunk_lateral_tilt_angle` are negated.  The raw axes are computed as
`left_joint − right_joint`, so a lefty's rotation sweeps opposite a righty's;
without the flip, fitting a single dynamical system across mixed handedness
collapses the rotation equations toward zero.

### 1.6 Baseline A/B Result (2026-06-23)

First-order 10-D state on 111 train / 29 test pool:

- Nonzero equation terms: **26**
- Test derivative R²: ≈ 0 (near zero)
- Test reconstruction R²: **−2.768**
- Test reconstruction RMSE: **0.719**

This is the reference point for all subsequent experiments.

---

## 2. Dropped Features — Rationale

Two features that ARE present in the operative first-order baseline (Section 1.1)
were explicitly excluded from the experimental second-order redesign:

### 2.1 `hip_shoulder_separation_angle` — algebraic identity

Defined as `wrap_angle(shoulder_rotation_angle − hip_rotation_angle)`.  SINDy
discovered coefficients ≈ +0.42 on `shoulder_rotation_angle` and −0.42 on
`hip_rotation_angle` for this equation — confirming it is a near-perfect
algebraic consequence of the other two.  Including it adds no independent
dynamical information and creates a near-linear-dependence column in the library
matrix.

### 2.2 `dominant_wrist_speed` — unpaired magnitude

Wrist speed is a scalar magnitude, not a position coordinate.  A magnitude
cannot be paired with a velocity via physical kinematics — there is no natural
`dominant_wrist_speed_rate` that closes the second-order structure.  Including it
breaks the strict `(position, velocity, acceleration)` triples required by the
second-order scheme.  It remains part of the operative
first-order baseline (Section 1.1) but is excluded from the second-order
experimental state.

---

## 3. Experimental State Vector: Second-Order 16-D [NOT ADOPTED]

> **Status: EXPERIMENTAL — REGRESSION ON RECONSTRUCTION — NOT ADOPTED**
>
> The operative production model remains the first-order 10-D state above.
> The second-order state was implemented and an A/B comparison was run on
> 2026-06-23.  The derivative fit improved but reconstruction regressed
> substantially (see Section 3.5).  The first-order baseline is retained.

### 3.1 Motivation

In a first-order SINDy system, each velocity (first derivative of a position)
is treated as an independent state variable and must be re-discovered from the
library.  By construction the position derivative must equal its paired velocity,
but with a first-order state the solver does not know they are the same object —
the "identity" equations survive only if the data is clean enough to recover
single-coefficient terms.  A strict second-order state makes this explicit:

- State `X = [pos, vel]` — positions concatenated with paired velocities.
- Derivative `X_dot = [vel, acc]` — velocities (= d/dt of positions) concatenated
  with accelerations (= d/dt of velocities).
- The first 8 rows of the equation system are constrained by construction to be
  identity equations (`d(pos_i)/dt = vel_i`); the solver cannot zero them out.

### 3.2 Positions (8-D)

The 8 positions are the operative first-order angle/position features (Section 1.1)
minus the two dropped features (`hip_shoulder_separation_angle`, `dominant_wrist_speed`;
Section 2):

`shoulder_rotation_angle`, `hip_rotation_angle`, `dominant_elbow_flexion_angle`,
`trail_elbow_flexion_angle`, `dominant_knee_flexion_angle`, `trail_knee_flexion_angle`,
`lead_knee_lift`, `stride_ankle_separation`.

### 3.3 Paired Velocities (8-D)

Each position has a corresponding `_rate` column appended after all 8 positions:

`shoulder_rotation_angle_rate`, `hip_rotation_angle_rate`,
`dominant_elbow_flexion_angle_rate`, `trail_elbow_flexion_angle_rate`,
`dominant_knee_flexion_angle_rate`, `trail_knee_flexion_angle_rate`,
`lead_knee_lift_rate`, `stride_ankle_separation_rate`.

Full state dimension: **16-D**.

### 3.4 Derivative Scheme — Single SG Fit

Implemented in `compute_second_order_state` in `extract_baseball_features.py`.

For each position column, a **single Savitzky-Golay filter** is evaluated at
derivative orders 0, 1, and 2 using the same window and polynomial:

```
pos  = savgol_filter(series, window, polyorder, deriv=0, mode="interp")
vel  = savgol_filter(series, window, polyorder, deriv=1, delta=dt, mode="interp")
acc  = savgol_filter(series, window, polyorder, deriv=2, delta=dt, mode="interp")
```

Parameters: `window_length=9`, `polyorder=3` (same SG parameters as the
first-order scheme).

Assembly:

```
X     = hstack([X_pos, V])    # shape (T, 16)
X_dot = hstack([V, A])        # shape (T, 16)
```

The single-fit approach ensures that `X_dot[:, :8]` (velocities in the
derivative matrix) is exactly consistent with `X[:, 8:]` (velocities in the
state matrix) — both come from the same `deriv=1` evaluation.  This prevents
the zero-collapse failure mode of the first-order scheme, where the solver could
drive position-derivative equations to zero because the first-order derivative
estimate was computed separately and could be noisy enough to not produce a clean
single-coefficient solution.

Angle columns (those in `ANGLE_FEATURES`) are unwrapped before filtering.
`_rate` columns carry no 2π discontinuity so they are not unwrapped.

Configs: `config/preprocessing.baseball.unaligned.second_order.yaml` and
`config/preprocessing.baseball.unaligned.second_order.sweep.yaml`
(`features.second_order: true`).

### 3.5 A/B Outcome: REGRESSION — Reconstruction Degraded

Second-order 16-D state on the same 111/29 pool (threshold sweep selected thr=0.30,
~101 nonzero terms):

| Metric | First-Order Baseline | Second-Order Experiment | Direction |
|--------|----------------------|-------------------------|-----------|
| Nonzero terms | 26 | ~101 | worse (denser) |
| Test derivative R² | ≈ −0.018 | **+0.081** | IMPROVED |
| Test reconstruction R² | −2.768 | **−9.160** | REGRESSED |
| Test reconstruction RMSE | 0.719 | **7.674** | REGRESSED |
| Rate-eq collapse | N/A | 0 of 8 | pass (identity gate) |

**Identity gate:** all 8 position-derivative equations had a dominant single-term
coefficient on their paired `_rate` column (window_length=9), confirming the
by-construction guarantee worked.

**Decision: DO NOT SHIP.** The reconstruction regression outweighs the derivative
improvement.  The first-order 10-D baseline is retained as operative.

**Worst subgroups:** left-handed pitchers (mean recon RMSE ≈ 12.8 vs ≈ 7.1 for
right-handed); fastballs worst pitch type.

### 3.6 Scale-Comparability Nuance

The 10.7× increase in reconstruction RMSE (0.719 → 7.674) is **not directly
scale-comparable** between the two experiments:

- The 16-D second-order state includes 8 `_rate` columns whose raw magnitudes are
  derivative-scale (derivative RMSE ≈ 70).  These dominate the reconstruction
  error.
- Second-order systems are stiffer to forward-simulate: integration errors
  accumulate twice (once through velocity, once through position).

The regression is partly a **forward-simulation and metric-comparability
problem**, not proof that second-order features are useless.  The derivative R²
improvement (+0.099) is real and shows the model fits `dX/dt` better.

### 3.7 Deferred Follow-Ups (Next Levers)

1. **Metric comparability**: report position-only reconstruction RMSE (ignoring
   the `_rate` components) so that the comparison across state definitions is
   on equal footing.
2. **Forward-sim stability**: address the stiffness of the second-order ODE
   before using reconstruction error as a decision criterion.
3. **Ninth paired position**: `wrist_relative_height` as a 9th biomechanically
   motivated position (deferred from original spec).
4. **Custom pairwise library**: handedness-relative joint-pair distances as
   library candidates (also deferred from original spec).

---

## 4. Fair-Metric Robustness Sweep (2026-06-23)

### 4.1 The fair metric

Whole-state `reconstruction_r2` (variance-weighted over all columns) is misleading
across different state vectors: for the 16-D second-order state the large-variance
velocity columns dominate, and for the 10-D first-order state the large-variance
`dominant_wrist_speed` column inflates the score. To compare models honestly we
added a **state-comparable** metric in `evaluate_baseball_sindy.py`, computed only
over the 8 shared `CORE_POSITIONS`:

- `core_position_recon_r2` — `r2_score(true, sim, multioutput="uniform_average")` over the 8 core columns (no column dominates).
- `core_position_recon_nrmse` — mean of per-column `RMSE / std` over the 8 core columns (unit-free, comparable across state definitions).

Whole-state metrics are still reported alongside (honesty guardrail — the fair
metric was committed before any model was re-scored against it).

Forward simulation also gained an `evaluation.integrator` option (`euler` | `rk4`);
RK4 was marginally better on the first-order baseline (core R² −29.56 vs −29.99
under euler) and was adopted.

### 4.2 Results — weak-form SINDy wins the fair metric

All models evaluated on the SAME restored 111/29 pool, RK4 integrator, 40-frame
free-running reconstruction. The fair metric is `core_position_recon_r2` (higher =
less negative = better):

| Config | core_position_recon_r2 | core_nrmse | whole-state recon R² |
|--------|-----------------------:|-----------:|---------------------:|
| **first-order, weak-form, multi-seed (BEST on fair metric)** | **−7.89** | **2.38** | −6.72 |
| first-order, plain | −29.56 | 3.25 | **−2.80** |
| first-order, ensemble | −36.96 | 3.52 | −3.05 |
| second-order, plain | −2269 | 19.10 | −6.03 |
| second-order, ensemble | −2429 | 19.22 | −6.25 |

- **Weak-form SINDy** (`WeakPDELibrary`, stacked per-trajectory weak systems across all
  111 clips — no truncation) is the **best** model on the fair metric: core R² −7.89
  (3.7× better than the plain baseline −29.56) and lower NRMSE (2.38 vs 3.25).
- **Ensemble** (bagged STLSQ) made both state vectors worse.
- **Second-order** forward-simulation diverges catastrophically on the fair metric.

**Weak-form reproducibility & variance.** `WeakPDELibrary` places its K test-function
subdomains via the global RNG with no seed parameter, so a single draw is non-reproducible
and high variance — single-seed core R² ranged −5.97 to −31.76 across seeds {0,1,2,3,7}.
`build_weakform_model` now seeds locally (and restores RNG state) and stacks `n_seeds`
independent subdomain draws (`n_seeds × K` test functions) to drive the variance down.
At `n_seeds=5` (`seed=7`) the result is a stable, reproducible core R² = **−7.89**.
Config: `config/preprocessing.baseball.unaligned.handedness.weakform.yaml`.

### 4.3 Decision and interpretation

**On the fair core-position metric, first-order weak-form (multi-seed) is the best model
and is recommended as operative for biomechanical-position reconstruction.**

Honesty guardrail: weak-form's *whole-state* recon R² (−6.72) is worse than the plain
baseline's (−2.80). Weak-form trades reconstruction of the high-variance `dominant_wrist_speed`
column (which inflated the plain baseline's whole-state score) for substantially better
reconstruction of the 8 core biomechanical positions. Because the core positions are the
quantities of interest, the fair core metric is the one to optimize — and weak-form wins it
decisively. If the whole-state number matters for a specific downstream use, the plain
first-order config remains the better choice there.

The deeper finding still holds: even the best model's `core_position_recon_r2` is negative
(−7.89), while one-step derivative R² hovers near 0. A single global linear SINDy model
predicts the *next* step but cannot be free-run for 40 frames on held-out clips — error
compounds across heterogeneous pitchers/views and noisy broadcast pose. Weak-form narrows
this gap 3.7× but does not close it.

### 4.4 Short-horizon analysis (`evaluation.recon_horizons`)

`evaluate_baseball_sindy.py` computes `core_position_recon_r2` at multiple rollout horizons
(default 5/10/20/40 frames) from the same forward simulation, written to `summary.csv` as
`mean_core_recon_r2_h{H}`. The two models have **opposite** horizon curves:

| horizon (frames) | plain first-order | weak-form (multi-seed) |
|-----------------:|------------------:|-----------------------:|
| 5  | −75.5 | **−2.68** |
| 10 | −47.8 | −5.40 |
| 20 | −38.0 | −4.90 |
| 40 | −29.6 | −7.89 |

**Weak-form dominates the plain baseline at every horizon — ~28× at h=5.** The plain model's
noisy coefficients drive the trajectory off course within the first few frames (catastrophic
on short windows where the positions barely move and the variance denominator is tiny);
weak-form's robust coefficients track the short-horizon dynamics gracefully and degrade
slowly. This is exactly the regime SINDy is designed for, and it is the strongest evidence
that weak-form is the right model.

Absolute `core_position_recon_r2` stays negative even at h=5 (−2.68): R² is measured against
the per-window mean, a harsh baseline when the positions are near-constant over a 5-frame
window. The headline metric is kept at 40-frame reconstruction for continuity; the horizon
curve is the diagnostic that exposes the qualitative plain-vs-weak-form difference.

---

## 5. Stage A Context

For Stage A (UTD-MHAD pipeline validation), the feature set uses a different
10-D state defined in `config/preprocessing.utd_mhad.example.yaml` and extracted
by `extract_utd_mhad_features.py`.  That state includes `trunk_rotation_angle`,
`shoulder_rotation_angle`, `hip_rotation_angle`, `hip_shoulder_separation_angle`,
`trunk_forward_lean_angle`, `trunk_lateral_tilt_angle`, `right_elbow_flexion_angle`,
`left_elbow_flexion_angle`, and wrist speed columns — it is intended for pipeline
validation, not biomechanical claims, and does not apply the handedness
canonicalization or second-order scheme described here.
