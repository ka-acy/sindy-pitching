# Fair-Metric Evaluation + Robustness Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Judge Stage B models on a fair, state-comparable reconstruction metric, then raise that metric with ensemble robustness — picking the empirical winner honestly.

**Architecture:** Phase 1 adds core-position reconstruction metrics + an RK4 integrator option to `evaluate_baseball_sindy.py`, then re-scores existing fits (no refit) to reframe the first-order vs second-order A/B. Phase 2 adds config-driven `EnsembleOptimizer` to `fit_baseball_sindy.py` (wrapping the optimizer locally — no change to shared Stage A code) and sweeps {first-order, second-order} × {ensemble off, on}. Phase 3 (weak-form) is gated behind the Phase-2 checkpoint and not pre-committed.

**Tech Stack:** Python 3.12, NumPy, SciPy, PySINDy 2.1.0 (`EnsembleOptimizer`, `WeakPDELibrary` available), scikit-learn. Run everything with `./.venv/bin/python`.

## Global Constraints

- Python interpreter: always `./.venv/bin/python` (never bare `python`).
- Work continues on branch `feature/baseball-second-order-state` (the second-order code path and fitted artifacts live there).
- Source video `/mnt/e/...` is UNAVAILABLE. Everything reads cached processed data. Do not invoke any video/pose-extraction script.
- **Frozen shared files — do NOT modify:** `src/preprocessing/evaluate_utd_mhad_sindy.py`, `src/preprocessing/fit_utd_mhad_sindy.py`, `src/preprocessing/extract_utd_mhad_features.py`. Stage A behavior must be byte-for-byte unchanged. In particular, add ensemble support by wrapping the optimizer inside `fit_baseball_sindy.py` — do NOT edit the shared `build_model`.
- **Honesty guardrail (binding):** the fair metric is committed BEFORE any model is re-scored against it; every report shows whole-state metrics ALONGSIDE the fair ones. Do not declare a winner on the fair metric while the whole-state metric collapses in a way the fair metric hides — surface both.
- **CORE_POSITIONS (exact, ordered):** `shoulder_rotation_angle`, `hip_rotation_angle`, `dominant_elbow_flexion_angle`, `trail_elbow_flexion_angle`, `dominant_knee_flexion_angle`, `trail_knee_flexion_angle`, `lead_knee_lift`, `stride_ankle_separation`. Both state vectors contain all 8.
- Fair metric definitions: `core_position_recon_r2` = `r2_score(true[:, core_idx], sim[:, core_idx], multioutput="uniform_average")`; `core_position_recon_nrmse` = mean over core columns of `sqrt(mean((sim_c-true_c)^2)) / (std(true_c)+1e-8)`.
- `results/` and `data/.../features` are gitignored — snapshot outputs on disk; do not `git add` them.
- No unit-test suite. Validation = code compiles, eval/fit run, fair-metric movement, and (for second-order) the identity gate still holds. Commit after each task.
- A/B reference (first-order baseline, whole-state): 26 terms, recon R²=−2.768, recon RMSE=0.719. Second-order whole-state: ~101 terms, recon R²=−9.160, recon RMSE=7.674, derivative R²=+0.081.

---

## File Structure

- `src/preprocessing/evaluate_baseball_sindy.py` — **modified** (Phase 1): `CORE_POSITIONS`, `core_position_metrics()`, RK4 in `reconstruct_trial`, new CSV columns, `evaluation.integrator` config read.
- `src/preprocessing/fit_baseball_sindy.py` — **modified** (Phase 2): local `build_candidate_model()` wrapping the optimizer in `EnsembleOptimizer` when `modeling.ensemble.enabled`.
- `src/preprocessing/compare_baseball_configs.py` — **modified** (Phase 2): emit the two core-position columns in the comparison CSV.
- `config/preprocessing.baseball.unaligned.handedness.ensemble.yaml` — **created** (Phase 2).
- `config/preprocessing.baseball.unaligned.second_order.ensemble.yaml` — **created** (Phase 2).
- `docs/feature_definitions.md`, project memory — **updated** (final).

---

## Task 1: Add fair-metric infrastructure + RK4 integrator to evaluate_baseball_sindy.py

**Files:**
- Modify: `src/preprocessing/evaluate_baseball_sindy.py`

**Interfaces:**
- Produces: `core_position_metrics(X_true, sim, feature_names) -> tuple[float, float]` returning `(core_position_recon_r2, core_position_recon_nrmse)`; `reconstruct_trial(..., integrator="euler")` with an added `integrator` kwarg supporting `"euler"` and `"rk4"`; new columns `core_position_recon_r2`, `core_position_recon_nrmse` in `reconstruction_metrics.csv`, and `mean_core_position_recon_r2`, `mean_core_position_recon_nrmse` in `summary.csv`.

- [ ] **Step 1: Add the CORE_POSITIONS constant and metric helper**

Near the top of the file (after imports), add:

```python
CORE_POSITIONS = [
    "shoulder_rotation_angle",
    "hip_rotation_angle",
    "dominant_elbow_flexion_angle",
    "trail_elbow_flexion_angle",
    "dominant_knee_flexion_angle",
    "trail_knee_flexion_angle",
    "lead_knee_lift",
    "stride_ankle_separation",
]


def core_position_metrics(
    X_true: np.ndarray, sim: np.ndarray, feature_names: list[str]
) -> tuple[float, float]:
    """Fair, state-comparable reconstruction metrics over the shared core positions.

    Returns (uniform-average R2, mean normalized RMSE) computed only on the
    CORE_POSITIONS columns present in feature_names. NRMSE normalizes each
    column's RMSE by that column's std so the value is unit-free and
    comparable across different state-vector definitions.
    """
    core_idx = [feature_names.index(name) for name in CORE_POSITIONS if name in feature_names]
    if not core_idx:
        return float("nan"), float("nan")
    true_c = X_true[:, core_idx]
    sim_c = sim[:, core_idx]
    r2 = float(r2_score(true_c, sim_c, multioutput="uniform_average"))
    per_col_rmse = np.sqrt(np.mean((sim_c - true_c) ** 2, axis=0))
    per_col_std = np.std(true_c, axis=0) + 1e-8
    nrmse = float(np.mean(per_col_rmse / per_col_std))
    return r2, nrmse
```

- [ ] **Step 2: Add the RK4 integrator option to `reconstruct_trial`**

Change the signature to add `integrator: str = "euler"` and replace the stepping loop. Replace the existing body between `sim_scaled[0] = scaled[0]` and the `sim = inverse_transform_trajectory(...)` line with:

```python
        sim_scaled[0] = scaled[0]
        if integrator == "rk4":
            def f(state: np.ndarray) -> np.ndarray:
                return np.asarray(model.predict(state[None, :]), dtype=np.float64)[0]
            for idx in range(1, frame_count):
                x = sim_scaled[idx - 1]
                k1 = f(x)
                k2 = f(x + 0.5 * dt * k1)
                k3 = f(x + 0.5 * dt * k2)
                k4 = f(x + dt * k3)
                sim_scaled[idx] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            for idx in range(1, frame_count):
                x_dot = np.asarray(model.predict(sim_scaled[idx - 1 : idx]), dtype=np.float64)
                sim_scaled[idx] = sim_scaled[idx - 1] + dt * x_dot[0]
```

Keep the existing signature parameters; the new `integrator` kwarg goes last. Keep the existing finite/magnitude guards unchanged.

- [ ] **Step 3: Read the integrator from config and compute core metrics in the loop**

In `main()`, after `eval_cfg = config["evaluation"]`, add:
```python
    integrator = str(eval_cfg.get("integrator", "euler"))
```

In the per-clip loop, change the `reconstruct_trial(...)` call to pass the integrator:
```python
                sim, error, recon_frames = reconstruct_trial(
                    model, X, dt, int(eval_cfg["reconstruction_frames"]), scaler, integrator=integrator
                )
```

Add core-metric accumulators next to the existing ones (near `recon_r2s: list[float] = []`):
```python
            core_r2s: list[float] = []
            core_nrmses: list[float] = []
```

Inside `if sim is not None:` (after `recon_rmses.append(...)`), add:
```python
                    core_r2, core_nrmse = core_position_metrics(X_recon, sim, feature_names)
                    core_r2s.append(core_r2)
                    core_nrmses.append(core_nrmse)
```

Set defaults before the `if sim is not None:` block (next to `recon_r2 = ""`):
```python
                core_position_recon_r2 = ""
                core_position_recon_nrmse = ""
```
and inside the `if sim is not None:` block assign them from the computed values:
```python
                    core_position_recon_r2 = core_r2
                    core_position_recon_nrmse = core_nrmse
```

Add the two fields to the `reconstruction_rows.append({...})` dict:
```python
                        "core_position_recon_r2": core_position_recon_r2,
                        "core_position_recon_nrmse": core_position_recon_nrmse,
```

Add the aggregates to the `summary_rows.append({...})` dict:
```python
                    "mean_core_position_recon_r2": float(np.mean(core_r2s)) if core_r2s else "",
                    "mean_core_position_recon_nrmse": float(np.mean(core_nrmses)) if core_nrmses else "",
```

- [ ] **Step 4: Add the new columns to both CSV header lists**

In the `write_csv(evaluation_root / "reconstruction_metrics.csv", [...header...], ...)` call, add `"core_position_recon_r2"` and `"core_position_recon_nrmse"` to the header list (after `"reconstruction_rmse"`). In the `summary.csv` `write_csv` header list, add `"mean_core_position_recon_r2"` and `"mean_core_position_recon_nrmse"`. (Read the file to find both header lists — the summary header is the second `write_csv` near the end of `main()`.)

- [ ] **Step 5: Compile and smoke-run on existing first-order baseline (euler)**

Run:
```bash
./.venv/bin/python -m compileall src/preprocessing/evaluate_baseball_sindy.py
```
Expected: no error.

The first-order baseline model still sits at `results/equations/baseball/train/pitching/model.pkl` only if the second-order run hasn't overwritten it. It HAS (Task 6 of the prior plan refit second-order there). So re-fit the first-order baseline model first, then eval:
```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py --config config/preprocessing.baseball.unaligned.handedness.yaml --skip-splits --skip-overrides
./.venv/bin/python src/preprocessing/evaluate_baseball_sindy.py --config config/preprocessing.baseball.unaligned.handedness.yaml --train-split train --eval-split test --task-types pitching
```
Then confirm the new columns exist and print them:
```bash
./.venv/bin/python -c "
import csv
r=list(csv.DictReader(open('results/evaluation/baseball/train_on_test/summary.csv')))[0]
print('whole-state recon r2/rmse:', r['mean_reconstruction_r2'], r['mean_reconstruction_rmse'])
print('CORE recon r2/nrmse      :', r['mean_core_position_recon_r2'], r['mean_core_position_recon_nrmse'])
"
```
Expected: all four values present and numeric. (The whole-state numbers should match the prior baseline ≈ −2.768 / 0.719, confirming no regression in existing behavior.)

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing/evaluate_baseball_sindy.py
git commit -m "feat: add fair core-position reconstruction metric + rk4 integrator"
```

---

## Task 2: Phase-1 analysis — fair A/B re-judgment + integrator decision

**Files:**
- Modify (config): `config/preprocessing.baseball.unaligned.handedness.yaml`, `config/preprocessing.baseball.unaligned.second_order.yaml`, `config/preprocessing.baseball.unaligned.second_order.sweep.yaml` (set `evaluation.integrator`)

**Interfaces:**
- Consumes: Task 1 metrics.
- Produces: the fair-metric A/B verdict (first-order vs second-order) and the chosen integrator written into the configs.

- [ ] **Step 1: Capture first-order fair metrics under euler and rk4**

The first-order model is currently fitted (Task 1 Step 5). Evaluate under both integrators by temporarily setting `evaluation.integrator`. Run:
```bash
for ig in euler rk4; do
  ./.venv/bin/python -c "
import yaml,sys
p='config/preprocessing.baseball.unaligned.handedness.yaml'
c=yaml.safe_load(open(p)); c['evaluation']['integrator']='$ig'; yaml.safe_dump(c,open(p,'w'),sort_keys=False)
"
  ./.venv/bin/python src/preprocessing/evaluate_baseball_sindy.py --config config/preprocessing.baseball.unaligned.handedness.yaml --train-split train --eval-split test --task-types pitching
  ./.venv/bin/python -c "
import csv
r=list(csv.DictReader(open('results/evaluation/baseball/train_on_test/summary.csv')))[0]
print('$ig  first-order  core_r2=%s core_nrmse=%s whole_r2=%s'%(r['mean_core_position_recon_r2'],r['mean_core_position_recon_nrmse'],r['mean_reconstruction_r2']))
"
done
```
Record both lines.

- [ ] **Step 2: Decide the integrator**

If `rk4` gives a better (higher) `core_position_recon_r2` on the first-order baseline than `euler`, adopt `rk4`; otherwise keep `euler`. Set the chosen value in `evaluation.integrator` of ALL THREE configs (`handedness`, `second_order`, `second_order.sweep`). State the decision and the numbers.

- [ ] **Step 3: Re-fit + evaluate the second-order model under the chosen integrator**

```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py --config config/preprocessing.baseball.unaligned.second_order.sweep.yaml --skip-splits --skip-overrides
./.venv/bin/python -c "
import csv
r=list(csv.DictReader(open('results/evaluation/baseball/train_on_test/summary.csv')))[0]
print('second-order core_r2=%s core_nrmse=%s whole_r2=%s'%(r['mean_core_position_recon_r2'],r['mean_core_position_recon_nrmse'],r['mean_reconstruction_r2']))
"
```

- [ ] **Step 4: Produce the fair A/B table and the reframed verdict**

Tabulate first-order vs second-order on `core_position_recon_r2`, `core_position_recon_nrmse`, with whole-state `mean_reconstruction_r2` shown alongside (honesty guardrail). State explicitly: on the FAIR metric, does second-order beat, tie, or lose to first-order? This sets which state vector(s) Phase 2 prioritizes. Snapshot the two summaries on disk:
```bash
cp results/evaluation/baseball/train_on_test/summary.csv /tmp/claude-1000/-mnt-c-Users-dtms--Documents-Research/fd452970-9ca1-4ac7-80f9-75fcc679ee02/scratchpad/phase1_secondorder_summary.csv
```
(and similarly capture the first-order summary into the report).

- [ ] **Step 5: Commit the integrator config change**

```bash
git add config/preprocessing.baseball.unaligned.handedness.yaml config/preprocessing.baseball.unaligned.second_order.yaml config/preprocessing.baseball.unaligned.second_order.sweep.yaml
git commit -m "config: set reconstruction integrator after phase-1 decision"
```

---

## Task 3: Add EnsembleOptimizer support to fit_baseball_sindy.py + ensemble configs

**Files:**
- Modify: `src/preprocessing/fit_baseball_sindy.py`
- Create: `config/preprocessing.baseball.unaligned.handedness.ensemble.yaml`, `config/preprocessing.baseball.unaligned.second_order.ensemble.yaml`

**Interfaces:**
- Consumes: shared `build_model` (unchanged), `modeling.ensemble` config block.
- Produces: a local `build_candidate_model(candidate, total_timepoints, ensemble_cfg)` that returns a plain SINDy when ensemble is disabled and an `EnsembleOptimizer`-wrapped SINDy when enabled. Backward compatible: absent/`enabled: false` → identical to today.

- [ ] **Step 1: Add the ensemble-aware model builder**

In `fit_baseball_sindy.py`, after the imports (note `import pysindy as ps` is already present and `build_model` is imported from `fit_utd_mhad_sindy`), add:

```python
def build_candidate_model(candidate: dict[str, Any], total_timepoints: int, ensemble_cfg: dict[str, Any]) -> ps.SINDy:
    """Build a SINDy model, wrapping the optimizer in EnsembleOptimizer when enabled.

    Leaves the shared build_model untouched: it builds the plain model, then
    swaps in an EnsembleOptimizer around the same STLSQ optimizer and reuses the
    same feature library. Disabled/absent ensemble config returns the plain model.
    """
    model = build_model(candidate)
    if not ensemble_cfg or not ensemble_cfg.get("enabled", False):
        return model
    n_models = int(ensemble_cfg.get("n_models", 20))
    frac = float(ensemble_cfg.get("n_subset_frac", 0.6))
    replace = bool(ensemble_cfg.get("replace", True))
    n_subset = max(2, min(int(frac * total_timepoints), total_timepoints - 1))
    ensemble = ps.EnsembleOptimizer(
        opt=model.optimizer,
        bagging=True,
        n_models=n_models,
        n_subset=n_subset,
        replace=replace,
    )
    return ps.SINDy(optimizer=ensemble, feature_library=model.feature_library)
```

- [ ] **Step 2: Use it for sweep candidates and the final model**

In `main()`, read the ensemble config once after `modeling_cfg = config["modeling"]`:
```python
    ensemble_cfg = dict(modeling_cfg.get("ensemble", {}))
```

In the per-group fit, compute timepoint counts and swap the two `build_model(...)` calls:
- Before the `for candidate in candidate_configs(modeling_cfg):` loop, add:
```python
            selection_timepoints = int(sum(x.shape[0] for x in selection_Xs))
            full_timepoints = int(sum(x.shape[0] for x in scaled_Xs))
```
- Change `model = build_model(candidate)` to:
```python
                model = build_candidate_model(candidate, selection_timepoints, ensemble_cfg)
```
- Change `final_model = build_model(best_result["candidate_config"])` to:
```python
            final_model = build_candidate_model(best_result["candidate_config"], full_timepoints, ensemble_cfg)
```

- [ ] **Step 3: Compile**

Run: `./.venv/bin/python -m compileall src/preprocessing/fit_baseball_sindy.py`
Expected: no error.

- [ ] **Step 4: Create the first-order ensemble config**

Create `config/preprocessing.baseball.unaligned.handedness.ensemble.yaml` as a copy of `config/preprocessing.baseball.unaligned.handedness.yaml` (read it first to copy current contents including the integrator set in Task 2), adding under `modeling:` (same indentation as `stlsq_threshold`):
```yaml
  ensemble:
    enabled: true
    n_models: 20
    n_subset_frac: 0.6
    replace: true
```

- [ ] **Step 5: Create the second-order ensemble config**

Create `config/preprocessing.baseball.unaligned.second_order.ensemble.yaml` as a copy of `config/preprocessing.baseball.unaligned.second_order.sweep.yaml` (so it carries `second_order: true`, the 8 positions, the threshold sweep, and the chosen integrator), adding the same `modeling.ensemble` block as Step 4.

- [ ] **Step 6: Smoke-fit the second-order ensemble + re-verify the identity gate**

```bash
./.venv/bin/python src/preprocessing/extract_baseball_features.py --config config/preprocessing.baseball.unaligned.second_order.ensemble.yaml --split train
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py --config config/preprocessing.baseball.unaligned.second_order.ensemble.yaml --split train --task-types pitching
./.venv/bin/python -c "
import csv
rows=list(csv.DictReader(open('results/equations/baseball/train/pitching/coefficients.csv')))
pos=['shoulder_rotation_angle','hip_rotation_angle','dominant_elbow_flexion_angle','trail_elbow_flexion_angle','dominant_knee_flexion_angle','trail_knee_flexion_angle','lead_knee_lift','stride_ankle_separation']
allpass=True
for p in pos:
    terms=[r['library_term'] for r in rows if r['state_variable']==p and r['library_term']!='1']
    ok = terms==[p+'_rate']
    allpass &= ok
    print('OK' if ok else 'FAIL', p, terms)
print('IDENTITY GATE', 'PASS' if allpass else 'FAIL', 'under ensemble')
"
```
Expected: gate PASS — the by-construction identity survives bagging. If FAIL, report it (do not proceed to Task 4); ensemble may be perturbing the identity rows and needs investigation.

- [ ] **Step 7: Commit**

```bash
git add src/preprocessing/fit_baseball_sindy.py config/preprocessing.baseball.unaligned.handedness.ensemble.yaml config/preprocessing.baseball.unaligned.second_order.ensemble.yaml
git commit -m "feat: add EnsembleOptimizer support + ensemble configs"
```

---

## Task 4: Run the robustness sweep, compare on the fair metric, pick the winner

**Files:**
- Modify: `src/preprocessing/compare_baseball_configs.py`

**Interfaces:**
- Produces: `results/evaluation/baseball/config_comparison.csv` extended with `test_core_position_recon_r2`, `test_core_position_recon_nrmse`; the chosen winning config.

- [ ] **Step 1: Add the core columns to the comparison script**

In `compare_baseball_configs.py`, in the `rows.append({...})` dict (after `test_mean_reconstruction_rmse`), add:
```python
                "test_core_position_recon_r2": eval_summary.get("mean_core_position_recon_r2", ""),
                "test_core_position_recon_nrmse": eval_summary.get("mean_core_position_recon_nrmse", ""),
```
And add the two names to the `write_csv(...)` header list (after `"test_mean_reconstruction_rmse"`). (`eval_summary` is a `dict` from `DictReader`; `.get` is safe.)

- [ ] **Step 2: Compile**

Run: `./.venv/bin/python -m compileall src/preprocessing/compare_baseball_configs.py`
Expected: no error.

- [ ] **Step 3: Run the 4-config sweep**

The comparison script runs extract→fit→eval for each config in turn (it does NOT rebuild splits, so the 111/29 pool is preserved). Run:
```bash
./.venv/bin/python src/preprocessing/compare_baseball_configs.py --configs \
  config/preprocessing.baseball.unaligned.handedness.yaml \
  config/preprocessing.baseball.unaligned.handedness.ensemble.yaml \
  config/preprocessing.baseball.unaligned.second_order.sweep.yaml \
  config/preprocessing.baseball.unaligned.second_order.ensemble.yaml \
  --output-csv results/evaluation/baseball/config_comparison_robustness.csv
```

- [ ] **Step 4: Tabulate and pick the winner**

```bash
./.venv/bin/python -c "
import csv
rows=list(csv.DictReader(open('results/evaluation/baseball/config_comparison_robustness.csv')))
for r in rows:
    print('%-55s terms=%-5s CORE_r2=%-9s CORE_nrmse=%-9s whole_r2=%s'%(
        r['config_name'], r['train_num_terms'], r['test_core_position_recon_r2'][:8], r['test_core_position_recon_nrmse'][:8], r['test_mean_reconstruction_r2'][:8]))
"
```
The winner is the config with the highest `test_core_position_recon_r2`, tie-broken by lowest `test_core_position_recon_nrmse`, then fewest `train_num_terms`. State the winner explicitly, with the honesty-guardrail note (show its whole-state number too).

- [ ] **Step 5: (Optional) Tune ensemble hyperparameters on the winning state vector**

ONLY if ensemble improved the winner: sweep `n_models ∈ {10, 20, 40}` × `n_subset_frac ∈ {0.4, 0.6, 0.8}` by editing the winning ensemble config and re-running its fit+eval, recording `core_position_recon_r2` for each. Keep the best. If ensemble did not help, skip this step and say so.

- [ ] **Step 6: Snapshot + commit**

```bash
cp results/evaluation/baseball/config_comparison_robustness.csv results/evaluation/baseball/config_comparison_robustness_2026-06-23.csv
git add src/preprocessing/compare_baseball_configs.py
git commit -m "feat: include core-position metric in config comparison; run robustness sweep"
```

---

## Task 5: Phase-2 checkpoint — decide on Phase 3 (weak-form), then finalize

**Files:**
- Modify: `docs/feature_definitions.md`; project memory files.

**Interfaces:**
- Produces: the operative-model decision, the honesty-guardrail comparison table in docs + memory.

- [ ] **Step 1: State the Phase-2 checkpoint outcome**

Report the best `core_position_recon_r2` across all 4 configs. Decision:
- If a config beats the first-order baseline on the fair metric → that config is the new operative winner.
- If the best fair `core_position_recon_r2` is still clearly negative (worse than ~0) → Phase 3 (weak-form SINDy) is warranted; STOP and surface this to the user with the numbers for a go/no-go on Phase 3 (it is gated, not auto-run).
- If first-order remains best → first-order stays operative; document why ensemble/second-order did not help.

- [ ] **Step 2: Update `docs/feature_definitions.md`**

Add a dated "2026-06-23 Fair-Metric Robustness Sweep" section with the 4-config comparison table (columns: config, terms, core_position_recon_r2, core_position_recon_nrmse, whole-state recon R²), the chosen integrator, the operative-model decision, and a one-line definition of the fair metric. Do not remove existing sections.

- [ ] **Step 3: Update project memory**

Append a 2026-06-23 entry to `/home/dtmschangmin/.claude/projects/-mnt-c-Users-dtms--Documents-Research/memory/project_stage_b_status.md` summarizing: fair metric introduced (core-position R²/NRMSE), integrator decision, the 4-config fair-metric results, the operative winner (or first-order retained), and whether Phase 3 is pending. Refresh the one-line `MEMORY.md` pointer. Do not `git add` the memory files (outside repo).

- [ ] **Step 4: Commit docs**

```bash
git add docs/feature_definitions.md
git commit -m "docs: record fair-metric robustness sweep results and operative decision"
```

---

## Task 6: Weak-form SINDy prototype (first-order winner config)

**Approved at Phase-2 checkpoint (2026-06-23): user chose weak-form.** Prototype on the first-order winner only (`unaligned.handedness`, the best fair-metric config).

**Files:**
- Modify: `src/preprocessing/fit_baseball_sindy.py` (add a config-gated weak-form fit path).
- Create: `config/preprocessing.baseball.unaligned.handedness.weakform.yaml`.

**Interfaces:**
- Produces: `build_weakform_model(candidate, scaled_Xs, dt, weak_cfg) -> ps.SINDy` — fits a `WeakPDELibrary` SINDy across (truncated, equal-length) trajectories, then transplants the robust coefficients into a plain `PolynomialLibrary` SINDy (so the existing `evaluate_baseball_sindy.py` predict/simulate flow works unchanged). Gated by `modeling.weak_form.enabled`.

**De-risked facts (verified 2026-06-23):**
- `SINDy.fit` in pysindy 2.1.0 does NOT accept `multiple_trajectories`. Passing a **list of equal-length trajectories** with a single `spatiotemporal_grid` works.
- Our clips vary in length → truncate every fit trajectory to the common **minimum length** so one weak grid applies (prototype simplification; log how many frames are dropped).
- Coefficient transplant works: fit a plain `PolynomialLibrary` SINDy to initialize, then assign `plain.optimizer.coef_ = weak_coef`; `plain.predict` then works and pickles fine.
- Weak and plain libraries must have **identical feature ordering** — assert `plain.get_feature_names() == weak_model.get_feature_names()` before transplant.

- [ ] **Step 1: Add the weak-form builder to `fit_baseball_sindy.py`**

Add (near `build_candidate_model`):

```python
def build_weakform_model(candidate: dict[str, Any], scaled_Xs: list[np.ndarray], dt: float, weak_cfg: dict[str, Any]) -> ps.SINDy:
    """Fit weak-form SINDy across trajectories; return a plain-library SINDy with transplanted coefficients.

    Trajectories are truncated to the common minimum length so a single weak
    spatiotemporal grid applies. The robust weak coefficients are copied into a
    plain PolynomialLibrary model so downstream predict/simulate is unchanged.
    """
    degree = int(candidate["polynomial_degree"])
    include_bias = bool(candidate["include_bias"])
    K = int(weak_cfg.get("K", 100))
    common_len = int(min(x.shape[0] for x in scaled_Xs))
    trimmed = [np.asarray(x[:common_len], dtype=np.float64) for x in scaled_Xs]
    grid = np.arange(common_len, dtype=np.float64) * dt
    weak_lib = ps.WeakPDELibrary(
        function_library=ps.PolynomialLibrary(degree=degree, include_bias=include_bias),
        spatiotemporal_grid=grid,
        K=K,
    )
    weak_opt = ps.STLSQ(
        threshold=float(candidate["stlsq_threshold"]),
        alpha=float(candidate["stlsq_alpha"]),
        max_iter=int(candidate["stlsq_max_iter"]),
        normalize_columns=False,
    )
    weak_model = ps.SINDy(feature_library=weak_lib, optimizer=weak_opt)
    weak_model.fit(trimmed, t=dt)
    weak_coef = np.asarray(weak_model.coefficients())

    plain = build_model(candidate)
    plain.fit(trimmed, t=dt)  # initialize library/optimizer structures
    if list(plain.get_feature_names()) != list(weak_model.get_feature_names()):
        raise ValueError(
            f"weak/plain feature mismatch: {weak_model.get_feature_names()} vs {plain.get_feature_names()}"
        )
    plain.optimizer.coef_ = weak_coef
    return plain
```

- [ ] **Step 2: Gate the fit path on `modeling.weak_form`**

After `ensemble_cfg = dict(modeling_cfg.get("ensemble", {}))`, add:
```python
    weak_cfg = dict(modeling_cfg.get("weak_form", {}))
    use_weak = bool(weak_cfg.get("enabled", False))
```
In the candidate loop, replace the `model = build_candidate_model(...)` line with:
```python
                if use_weak:
                    model = build_weakform_model(candidate, selection_Xs, dt, weak_cfg)
                else:
                    model = build_candidate_model(candidate, selection_timepoints, ensemble_cfg)
```
And the final-model line:
```python
            if use_weak:
                final_model = build_weakform_model(best_result["candidate_config"], scaled_Xs, dt, weak_cfg)
            else:
                final_model = build_candidate_model(best_result["candidate_config"], full_timepoints, ensemble_cfg)
```
Note: weak-form models are already fitted inside the builder, but the existing code calls `model.fit(...)` / `final_model.fit(...)` afterward. That re-fit would overwrite the transplanted coefficients with a plain STLSQ fit. To prevent this, guard those two `.fit(...)` calls: wrap each existing fit call as `if not use_weak: <existing model.fit(...)>`. (The weak builder returns an already-fitted plain model.)

- [ ] **Step 3: Compile**

Run: `./.venv/bin/python -m compileall src/preprocessing/fit_baseball_sindy.py`
Expected: no error.

- [ ] **Step 4: Create the weak-form config**

Create `config/preprocessing.baseball.unaligned.handedness.weakform.yaml` as a copy of `config/preprocessing.baseball.unaligned.handedness.yaml` (first-order, integrator: rk4), adding under `modeling:`:
```yaml
  weak_form:
    enabled: true
    K: 100
```

- [ ] **Step 5: Fit + evaluate the weak-form model**

```bash
./.venv/bin/python src/preprocessing/extract_baseball_features.py --config config/preprocessing.baseball.unaligned.handedness.weakform.yaml --split train
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py --config config/preprocessing.baseball.unaligned.handedness.weakform.yaml --split train --task-types pitching
./.venv/bin/python src/preprocessing/evaluate_baseball_sindy.py --config config/preprocessing.baseball.unaligned.handedness.weakform.yaml --train-split train --eval-split test --task-types pitching
./.venv/bin/python -c "
import csv
r=list(csv.DictReader(open('results/evaluation/baseball/train_on_test/summary.csv')))[0]
print('weak-form  CORE r2=%s  CORE nrmse=%s  whole r2=%s  deriv r2=%s'%(r['mean_core_position_recon_r2'],r['mean_core_position_recon_nrmse'],r['mean_reconstruction_r2'],r['mean_derivative_r2']))
"
```
Report the weak-form fair metric next to the first-order baseline (CORE r2=−29.56). State whether weak-form helped, hurt, or was neutral. Log the truncation (common_len vs original clip-length distribution).

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing/fit_baseball_sindy.py config/preprocessing.baseball.unaligned.handedness.weakform.yaml
git commit -m "feat: weak-form SINDy prototype on first-order config"
```

**Escalation:** if the multi-trajectory weak fit or the feature-name assertion fails and cannot be resolved (e.g., bias-term ordering mismatch that can't be aligned), report BLOCKED with the exact error rather than forcing a workaround — the controller will decide whether to fall back to a single-concatenated-trajectory approximation or stop the weak-form lever.

---

## Self-Review Notes

- **Spec coverage:** honesty guardrail (Global Constraints + Task 2/4/5 reporting), fair metric definition (Task 1), integrator upgrade decided on baseline (Task 2), re-judge A/B (Task 2), ensemble without touching shared code (Task 3), sweep + winner on fair metric (Task 4), weak-form gated (Task 6), winner→operative + docs/memory (Task 5). All spec sections map to tasks.
- **Frozen-file safety:** ensemble is added via a local `build_candidate_model` in `fit_baseball_sindy.py`; the shared `build_model`/`fit_utd_mhad_sindy.py`/`evaluate_utd_mhad_sindy.py` are never edited.
- **Type consistency:** `core_position_metrics(X_true, sim, feature_names) -> (float, float)`; `reconstruct_trial(..., integrator="euler")`; `build_candidate_model(candidate, total_timepoints, ensemble_cfg) -> ps.SINDy`. CSV column names are reused verbatim between the writer header lists and the row dicts.
- **Ordering note:** Task 1 Step 5 re-fits the first-order baseline because the prior plan's second-order run overwrote `results/equations/baseball/train/pitching/`. Task 4's sweep re-runs every config fresh, so transient overwrites between tasks are fine.
