# Fair-Metric Evaluation + Robustness Sweep — Design

**Date:** 2026-06-23
**Stage:** Stage B (MLB pitching clips)
**Branch:** continues on `feature/baseball-second-order-state` (depends on the second-order code path + fitted artifacts already on that branch)
**Goal:** Get the best *honestly-measured* generalization we can, by (1) replacing the misleading whole-state reconstruction metric with a fair, state-comparable one, then (2) pushing that metric up with robustness techniques. Pick the empirical winner.

## Problem

Two coupled problems block "best results":

1. **The decision metric is misleading.** `evaluate_baseball_sindy.py::reconstruct_trial` integrates the SINDy model with hand-rolled **Euler** stepping and scores reconstruction R² via `r2_score(..., multioutput="variance_weighted")` over **all** state columns. For the 16-D second-order state, the 8 high-variance velocity (`_rate`) columns dominate the weighted average, so second-order scored R²=−9.16 even though its *derivative* fit improved (−0.018→+0.081). We cannot optimize against a metric that conflates state-vector scale with model quality.

2. **Generalization is genuinely weak.** Across the whole project, test reconstruction R² has been negative (worse than predicting the held-out mean). A single global STLSQ fit over heterogeneous, noisy broadcast-pose clips produces coefficients that don't transfer well.

## Design

### Honesty guardrail (binding)

The fair metric is defined and committed **before** any model is re-scored against it, and every report shows the **whole-state metrics alongside** the fair ones. We do not get to "win" by choosing a metric after seeing results. A configuration is only declared better if it improves the fair metric without the whole-state metric collapsing in a way the fair metric is hiding.

### Phase 1 — Fair evaluation foundation

**1a. State-comparable reconstruction metrics.**

Define the canonical comparison set as the **8 core biomechanical positions** present in both state vectors:

```
CORE_POSITIONS = [
    "shoulder_rotation_angle", "hip_rotation_angle",
    "dominant_elbow_flexion_angle", "trail_elbow_flexion_angle",
    "dominant_knee_flexion_angle", "trail_knee_flexion_angle",
    "lead_knee_lift", "stride_ankle_separation",
]
```

At evaluation time, intersect `CORE_POSITIONS` with the model's `feature_names` to get column indices `core_idx` (preserving `feature_names` order). Both the first-order 10-D state (which has all 8 plus `hip_shoulder_separation_angle` and `dominant_wrist_speed`) and the second-order 16-D state (8 positions + 8 rates) contain all 8, so the comparison is exactly apples-to-apples.

Compute over `core_idx` columns of the true vs. simulated trajectory:
- `core_position_recon_r2` = `r2_score(X_true[:, core_idx], sim[:, core_idx], multioutput="uniform_average")` — uniform, so no single column dominates.
- `core_position_recon_nrmse` = mean over `core_idx` of `sqrt(mean((sim_c - true_c)^2)) / (std(true_c) + 1e-8)` — unit-free, comparable across state definitions. Lower is better; 1.0 ≈ as bad as predicting that column's mean.

Add these to per-clip `reconstruction_metrics.csv` and to the aggregate `summary.csv`. **Keep** existing whole-state `mean_reconstruction_r2` / `mean_reconstruction_rmse` columns unchanged (continuity + honesty guardrail).

**1b. Integrator upgrade (neutral eval improvement).**

The Euler stepper accumulates error, penalizing all models. Add an integrator option to `reconstruct_trial`: `integrator ∈ {"euler", "rk4"}` (RK4 hand-rolled in scaled space, same `model.predict` calls, no new dependency). Evaluate first-order baseline under both. **Adopt RK4 as default only if it improves the first-order baseline's fair metric**; otherwise keep Euler. This is a pure evaluation improvement applied equally to all models, decided on the operative baseline, not on the candidate.

**1c. Re-score existing fits (no refit).**

Run the upgraded `evaluate_baseball_sindy.py` against the already-fitted first-order baseline (`train_first_order_baseline_2026-06-23` model) and the second-order run. Produce the fair-metric A/B table.

**Phase 1 checkpoint:** Does second-order beat first-order on `core_position_recon_r2` / `core_position_recon_nrmse`? Record the reframed verdict. Either outcome is informative and sets the Phase-2 starting state vector(s).

### Phase 2 — Ensemble robustness

Add config-driven ensembling to `fit_baseball_sindy.py`. New `modeling.ensemble` block:

```yaml
modeling:
  ensemble:
    enabled: true
    n_models: 20            # bagged STLSQ fits
    n_subset_frac: 0.6      # fraction of training timepoints per bag
    replace: true           # bootstrap with replacement
```

When `enabled`, wrap the per-candidate optimizer: instead of bare `STLSQ(...)`, build `EnsembleOptimizer(opt=STLSQ(...), bagging=True, n_models=n_models, n_subset=int(n_subset_frac * total_timepoints), replace=replace)`. The ensemble's median/mean coefficients yield a more stable model. The existing threshold sweep, selection metric, scaling, and validation split are unchanged — ensembling wraps the optimizer only. When `enabled: false` (or absent), behavior is identical to today (bare STLSQ) — backward compatible.

`n_subset` is computed from the concatenated training-timepoint count at fit time, clamped to `[polyorder+2, total]`.

Create configs:
- `config/preprocessing.baseball.unaligned.handedness.ensemble.yaml` (first-order + ensemble)
- `config/preprocessing.baseball.unaligned.second_order.ensemble.yaml` (second-order + ensemble)

**Sweep matrix:** `{first-order, second-order} × {ensemble off, ensemble on}` (the "off" cells reuse Phase-1 results), each with its existing threshold sweep. Evaluate all on the Phase-1 fair metric via `compare_baseball_configs.py`.

**Phase 2 checkpoint:** best fair-metric model becomes the candidate winner. If ensembling helps, tune `n_models`/`n_subset_frac` once (a small secondary sweep) on the winning state vector.

### Phase 3 — Weak-form SINDy (stretch; gated behind Phase 2)

Only if Phase 2's best `core_position_recon_r2` is still clearly negative: prototype `WeakPDELibrary` (weak/integral formulation, robust to pose noise) on the winning state vector. This changes how derivatives enter the fit (the weak library forms its own integrated features and target), so it is a larger, riskier change — scoped as a prototype on one config, not a full sweep. Decision to proceed is made at the Phase-2 checkpoint with the user.

### Picking the winner & what changes

The single best configuration on `core_position_recon_r2` (tie-broken by `core_position_recon_nrmse`, then term count) becomes the new operative baseline:
- Its config is marked operative; `docs/feature_definitions.md` and project memory updated with the fair-metric result and the whole-state numbers alongside.
- If **nothing** beats the first-order baseline on the fair metric, first-order stays operative and we document why (a legitimate, honest negative result with the robustness levers ruled out).

## Files changed

- `src/preprocessing/evaluate_baseball_sindy.py` — add `CORE_POSITIONS`, core-position metric computation, `integrator` option (euler/rk4), new CSV columns. The shared `evaluate_utd_mhad_sindy.py` is NOT touched.
- `src/preprocessing/fit_baseball_sindy.py` — add `EnsembleOptimizer` wrapping in `build_model` path (driven by `modeling.ensemble`). The shared `fit_utd_mhad_sindy.py` `build_model` may need a small, backward-compatible extension; if so, guard it so Stage A behavior is unchanged when `ensemble` is absent.
- `config/preprocessing.baseball.unaligned.handedness.ensemble.yaml` — new.
- `config/preprocessing.baseball.unaligned.second_order.ensemble.yaml` — new.
- `docs/feature_definitions.md`, project memory — updated at the end with the winner + honesty-guardrail comparison table.

Configs that do NOT change: the existing first-order and second-order (non-ensemble) configs remain as-is for the "off" sweep cells.

## Out of scope (this effort)

- Per-pitch-type / per-handedness separate models (data-starved at 111 clips; was the "comprehensive" option, not chosen).
- Phase normalization / alignment track.
- Pool expansion beyond 111/29.
- Custom pairwise-product / higher-degree libraries (deferred; revisit only if a checkpoint points there).
- `wrist_relative_height` 9th paired position (deferred second-order follow-up).

## Success criteria

- A fair, state-comparable reconstruction metric (`core_position_recon_r2`, `core_position_recon_nrmse`) is implemented, committed before re-scoring, and reported alongside whole-state metrics.
- The first-order vs second-order A/B is re-judged on the fair metric (Phase 1 checkpoint produces a reframed verdict).
- Ensemble robustness is swept across both state vectors and judged on the fair metric (Phase 2 checkpoint).
- A single operative winner is chosen by the fair metric (or first-order retained with documented reason), with the honesty-guardrail comparison table in docs + memory.
- The second-order identity gate still passes wherever the second-order state is used.
