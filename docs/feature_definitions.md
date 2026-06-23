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

## 4. Stage A Context

For Stage A (UTD-MHAD pipeline validation), the feature set uses a different
10-D state defined in `config/preprocessing.utd_mhad.example.yaml` and extracted
by `extract_utd_mhad_features.py`.  That state includes `trunk_rotation_angle`,
`shoulder_rotation_angle`, `hip_rotation_angle`, `hip_shoulder_separation_angle`,
`trunk_forward_lean_angle`, `trunk_lateral_tilt_angle`, `right_elbow_flexion_angle`,
`left_elbow_flexion_angle`, and wrist speed columns — it is intended for pipeline
validation, not biomechanical claims, and does not apply the handedness
canonicalization or second-order scheme described here.
