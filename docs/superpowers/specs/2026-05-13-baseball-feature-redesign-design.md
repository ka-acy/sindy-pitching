# Baseball Feature Redesign — Second-Order State Vector

**Date:** 2026-05-13
**Stage:** Stage B (MLB pitching clips)
**Scope:** State vector composition + SINDy library/threshold config. Data pool, evaluation harness, and pipeline orchestration unchanged.
**Goal:** Generalization first (target `test_mean_reconstruction_r2 > 0`), interpretability second.

## Problem

Stage B's current 10-D state (post 2026-05-08 handedness fix) reconstructs train clips reasonably but generalizes badly: `test_mean_reconstruction_r2 = −2.77`, worse than predicting the held-out mean. Two equations collapse to zero (`shoulder_rotation_angle`, `hip_rotation_angle`). One equation is a tautology (`hip_shoulder_separation_angle ≈ shoulder_rot − hip_rot`).

Root cause: the state is 9 positions/angles + 1 velocity magnitude (`dominant_wrist_speed`). For an angle, `d(angle)/dt` is angular velocity — but angular velocities aren't in the state, so SINDy can only fit position derivatives via cross-couplings to other positions. That works when there's a phase-locked coupling (the ±0.42 identity for `hip_shoulder_separation_angle`) and collapses to zero otherwise.

## Design

### 1. State vector (16-D, strict second-order)

**Positions (8):**
- `shoulder_rotation_angle`
- `hip_rotation_angle`
- `dominant_elbow_flexion_angle`
- `trail_elbow_flexion_angle`
- `dominant_knee_flexion_angle`
- `trail_knee_flexion_angle`
- `lead_knee_lift`
- `stride_ankle_separation`

**Velocities (8 — first time-derivatives of each position above):**
- `shoulder_rotation_rate`
- `hip_rotation_rate`
- `dominant_elbow_flexion_rate`
- `trail_elbow_flexion_rate`
- `dominant_knee_flexion_rate`
- `trail_knee_flexion_rate`
- `lead_knee_lift_rate`
- `stride_ankle_separation_rate`

**Dropped from current state:**
- `hip_shoulder_separation_angle` — algebraically equal to `shoulder_rot − hip_rot`. The ±0.42 symmetric identity in the current `equations.txt:3` confirmed it carries no information beyond its inputs.
- `dominant_wrist_speed` — magnitude (sign-stripped) and unpaired with a position state variable. Arm-tip motion is carried implicitly by `shoulder_rotation_rate` + `dominant_elbow_flexion_rate`. A `wrist_relative_height` paired-position addition is queued as a v2 follow-up if v1 results show arm dynamics are under-represented.

**Why second-order works:** 8 of the 16 equations become near-identities (`d(position_i)/dt = rate_i`, single-term). SINDy's sparsity budget is spent entirely on the other 8 rate-derivative equations, where real dynamics live. The trivial zero collapse seen in the current `shoulder_rotation_angle` and `hip_rotation_angle` equations cannot occur because their derivative is, by construction, a column of the state.

### 2. Derivative estimation

Single Savitzky-Golay fit per position column, queried at three orders to keep X and X_dot internally consistent.

Per position column:
- `savgol_filter(pos, window, polyorder, deriv=0)` → smoothed position. Becomes `X[:, 1:8]`.
- `savgol_filter(pos, window, polyorder, deriv=1) / dt` → velocity. Becomes `X[:, 9:16]` and `X_dot[:, 1:8]` (copied — position derivatives are, by definition, the velocity columns).
- `savgol_filter(pos, window, polyorder, deriv=2) / dt²` → acceleration. Becomes `X_dot[:, 9:16]`.

Querying all three orders from one SG fit avoids the noise amplification a chained-derivative scheme would introduce.

**Settings:**
- Start with current `window_length=9, polyorder=3` (polyorder ≥ 2 required for deriv=2).
- Angle unwrap is applied before SG, only on the 6 angle columns and only for the deriv=0/1 passes. Rate columns live in `dθ/dt`-space and have no 2π discontinuity, so no unwrap on the X_dot side.
- SG boundary mode: `mode='interp'` (scipy default, matches current behavior).

**Identity gate (Section 5).**

### 3. SINDy library & sparsity config

**Library:** `PolynomialFeatures(degree=1, include_bias=True, interaction_only=False)`. 17 candidates × 16 equations = 272 total candidate terms.

Degree=1 only. At 16-D state, degree=2 inflates to ~2,200 candidates per equation against ~11k timepoints (111 clips × ~100 frames each) — past the data-starved threshold that the prior degree-2 sweep already hit on the 10-D state.

**Solver:** `STLSQ(threshold=θ, alpha=0.001, max_iter=20)` — same as current `unaligned.handedness` run.

**Threshold sweep:** `{0.05, 0.10, 0.15, 0.20, 0.30}`. Drop the prior 0.5/0.7/1.0 options — they risk killing real terms in rate-derivative equations, where post-scaling coefficient magnitudes are smaller and more variable than the position-derivative identities.

**Selection metric:** `sparsity_weighted_r2` with `complexity_penalty=5e-05` (current setting). Re-evaluate the penalty if the Pareto front clusters differently than on the 10-D state; target landing in the ~30–50 nonzero terms regime.

**Selection split:** `selection_validation_fraction=0.22, selection_validation_seed=7` carries over.

**Held in reserve, not in v1:** if the rate-derivative equations leave clear residual at the best degree=1 threshold, a targeted custom library (specific pairwise products, e.g., `hip_rotation_rate × shoulder_rotation_rate` for kinematic-sequence coupling) is the natural follow-up.

### 4. Pipeline integration

**Files that change:**

1. `src/preprocessing/extract_baseball_features.py`
   - Add `compute_second_order_state(feature_series, position_names, dt, window, polyorder)` returning `(X_pos, V, A)`.
   - Assemble `X = hstack([X_pos, V])` (16 cols), `X_dot = hstack([V, A])` (16 cols).
   - Append `_rate` suffixed names to `feature_names` automatically — config lists positions only.
   - The shared `extract_utd_mhad_features.py` is not touched; Stage A continues with its current logic.
2. `config/preprocessing.baseball.unaligned.second_order.yaml` (new)
   - Copy of `unaligned.handedness.yaml`.
   - `features.state_vector`: 8 position names from Section 1.
   - New flag `features.second_order: true` — triggers the velocity-column path in extract.
   - `modeling.stlsq_threshold: 0.1` for single-shot runs.
3. `config/preprocessing.baseball.unaligned.second_order.sweep.yaml` (new)
   - Same as above but with `modeling.sweep.stlsq_threshold: [0.05, 0.10, 0.15, 0.20, 0.30]`.
4. `docs/feature_definitions.md` — currently stale (documents an older Stage A 10-feature vector). Rewrite with the 16-D second-order spec, plus the dropped-features rationale.

**Files that do not change:**
- `fit_baseball_sindy.py`, `evaluate_baseball_sindy.py`, `sindy_utils.py` — consume `feature_names` and matrix shapes, both of which scale cleanly.
- `refresh_baseball_pipeline_unaligned.py` — already accepts `--config`; point at the new YAML and pass `--skip-splits` to freeze the 111/29 stratified split for A/B comparison.

**Baseline preservation:** before any re-extraction, copy the existing run outputs aside so the 2026-05-08 baseline is not overwritten:
- `results/equations/baseball/train/pitching/` → `results/equations/baseball/train/pitching_baseline_2026-05-08/`
- `results/evaluation/baseball/train_on_test/` → `results/evaluation/baseball/train_on_test_baseline_2026-05-08/`

Git commit those snapshots before running the new config.

### 5. Testing & evaluation

**Pre-fit checks:**
1. Snapshot baseline (above), git commit.
2. Run extraction on 3–5 train clips with `--limit 5`. Confirm `X.shape == (T, 16)` and `X_dot.shape == (T, 16)`. Confirm `np.allclose(X_dot[:, :8], X[:, 8:])` per clip — this is the by-construction position-derivative copy.
3. Spot-check `X_dot[:, 8:]` (accelerations) on one clip — finite, bounded, no NaN/Inf at boundaries. If SG window=9 produces boundary artifacts, bump to 11.

**Single-shot identity gate (θ=0.05, full train pool):**
4. Run `fit_baseball_sindy.py` once at the lowest sweep threshold. Read `equations.txt`. The first 8 equations must each be **single-term**, with the nonzero coefficient on the paired rate column and nothing else. Coefficient value will be approximately `std(rate_i)` due to feature scaling — value itself doesn't matter, single-term structure does. Per-trial RMSE on the position-derivative rows of `trial_metrics.csv` must be near zero (~10⁻⁶ relative to typical X_dot scale).
5. If the gate fails: widen SG window to 11 (then 13 if still failing), re-extract, re-fit. Do not proceed to the sweep until the gate passes.

**Full sweep:**
6. Run threshold sweep via `unaligned.second_order.sweep.yaml`. Selection metric `sparsity_weighted_r2` picks the winning threshold automatically.
7. Evaluate winning model on the test split via existing `evaluate_baseball_sindy.py` → `train_on_test/summary.csv`.

**A/B decision matrix vs. baseline (test_recon_rmse=0.72, test_recon_r²=−2.77, 26 nonzero terms):**

| Outcome | Action |
|---|---|
| `test_recon_r² > 0` | Ship. New baseline. Update memory. v1 met headline goal. |
| `test_recon_r² ∈ [−2.77, 0)` | Partial win. Ship as new baseline if `test_recon_rmse < 0.72` also. Queue follow-ups: `wrist_relative_height`, custom pairwise library. |
| `test_recon_r² < −2.77` | Regression. Do not ship. Diagnose: rate-derivative equations collapsed to zero (state still doesn't separate clips)? Train fits but test doesn't (overfit — raise penalty/threshold)? Acceleration noise (widen SG window)? |

**Per-clip diagnostics (regardless of outcome):**
- `trial_metrics.csv` sorted by test RMSE: are worst clips correlated with handedness, pitch type, or specific source videos? Distinguishes data-quality failures from feature-design failures.
- Coefficient table for the 8 rate-derivative equations: any collapsed to zero? That identifies which biomechanical variable still has no discovered dynamics — the next feature-design target.

**No unit tests.** Research-pipeline redesign is validated by metric movement, not isolated unit tests. The codebase has no test suite to plug into. The identity gate plus the by-construction allclose check serve as the closest equivalents.

## Out of scope (v1)

- Adding `wrist_relative_height` as a 9th paired position. Deferred per Section 1; will be revisited if rate-derivative equations under-represent arm dynamics.
- Custom pairwise-product library. Deferred per Section 3; will be revisited if degree=1 residuals show structure.
- Pool expansion beyond 111/29. Out of scope per agreed brainstorming scope (state vector + SINDy config, not data).
- Pipeline-level changes: per-pitch-type models, per-phase segmentation, switching modeling target, alternative model frameworks (neural ODE, weak-form SINDy).

## Success criteria

- Identity gate passes (8 single-term position-derivative equations).
- Threshold sweep completes and selects a single best threshold.
- Best-threshold model evaluated on test split.
- A/B decision matrix above produces a ship / partial-win / regression call.
- `docs/feature_definitions.md` no longer stale.
