# Dynamical Representations of Pitching, Validated Against Pitch Labels — Design

**Date:** 2026-06-23
**Stage:** Stage B (MLB pitching clips)
**Branch:** new feature branch off `main`
**Goal:** Elevate the contribution from "a reproducible pipeline that doesn't reconstruct well" to a **label-validated comparative study**: which dynamical/kinematic representation of pitching motion best encodes biomechanically meaningful structure, measured by predicting **release velocity** (primary) and **pitch type** (secondary), with the reconstruction failure reported honestly alongside.

## Problem / motivation

Every model tried so far is judged by self-reconstruction of the motion trajectory, and all fail (best core-position recon R² ≈ −7.9; one-step derivative R² ≈ 0). A bigger table of reconstruction failures is more thorough but not a discovery. The data, however, carries **external labels we have never used**: each of the 140 clips has `pitch_type` (6 classes) and `pitch_speed_mph` (82–103), all from one consistent view (`broadcast_centerfield`). This lets us ask a question the data can actually answer: *do data-driven representations of pitching dynamics carry structure that predicts these labels, even when full-trajectory reconstruction fails?* A positive, permutation-validated answer — with the discriminative structure traced back to known biomechanics — is the profound result. A clean negative (representations carry no label signal beyond a kinematic baseline) is still a stronger, more interesting finding than more reconstruction failures.

## Data & labels

- 140 pitching clips, per-clip 10-D first-order feature trajectories already produced under `data/baseball/processed/{train,test}/features/*.npz` (state matrix `X`, `feature_names`, `metadata.clip_id`). The study pools all 140 (the train/test split was for the global dynamics fit; the downstream study uses cross-validation over all clips).
- Labels from `data/baseball/splits/pitching_local_pool.csv`, joined by `clip_id`: `pitch_speed_mph` (continuous), `pitch_type` (changeup 33, fastball 39, curveball 38, knucklecurve 20, sinker 5, slider 5).
- Binary pitch-class target: **fast** = {fastball, sinker} (44) vs **offspeed** = {changeup, curveball, knucklecurve, slider} (96).

## Representations (per-clip fixed-length vectors)

Each extractor maps one clip's 10-D feature trajectory `X (T×10)` + `dt` to `(vector, feature_names)`. All implementable with installed NumPy/SciPy/scikit-learn/PySINDy — no new dependency for the core set.

- **R0 — Kinematic summary (baseline / control).** Per feature column: mean, std, min, max, range, normalized argmax (peak timing), and mean |first difference| (mean speed). ~7×10 = 70-dim. The "do you even need dynamics?" control that everything must beat to justify a dynamical representation.
- **R1 — SINDy per-clip coefficients.** Fit a per-clip SINDy (degree-1 polynomial library, scaled, STLSQ at the operative threshold) on that clip's `X`; flatten the coefficient matrix (10 states × 11 terms = 110-dim).
- **R2 — Weak-form SINDy per-clip coefficients.** Same shape, fit via `WeakPDELibrary` on the single clip (seeded, K capped to clip length); the robust variant. 110-dim.
- **R3 — DMD / Koopman spectrum.** Exact DMD (NumPy SVD + eig) on `X`: best-fit linear operator `A` s.t. `X[1:] ≈ A X[:-1]`. Representation = sorted eigenvalues (real, imag) + per-mode energy. Fixed length (pad/truncate to the 10 modes of the 10-D state → 30-dim).
- **R4 (stretch) — Hankel / time-delay DMD (HAVOK-style).** Delay-embed `X` (stack `d` delays), DMD on the augmented state, take the top-`r` eigenvalues + energies. Captures dynamics a single-step linear operator misses. Fixed length.
- **R5 (gated optional) — Neural ODE latent.** Only if (a) the core R0–R4 results show the tasks are learnable and (b) we want to test whether nonlinear expressivity adds signal. Requires installing `torch`+`torchdiffeq` — defer until justified at the Phase-1 checkpoint; do not install upfront.

All representations are computed **independently per clip**, so there is no train/test leakage in the downstream CV.

## Downstream tasks & protocol

**Primary endpoint (pre-registered): T1 — release-speed regression.** Predict `pitch_speed_mph` from each representation.
- Model: `RidgeCV` (alpha grid) as the fixed primary; `RandomForestRegressor` as a nonlinear secondary. Same model across representations so we compare representations, not regressors.
- Metric: cross-validated R² and MAE. Baseline: predict the training-fold mean (R²=0 by construction).
- Significance: permutation test — shuffle `pitch_speed_mph` ~1000× and recompute CV R²; empirical p = fraction of null R² ≥ observed.

**Secondary: T2 — pitch-class.** Binary fast-vs-offspeed primary; 6-class reported with explicit small-class caveats.
- Model: L2-logistic (`class_weight="balanced"`) primary; random forest secondary.
- Metric: balanced accuracy + macro-F1. Baseline: majority / stratified-random. Permutation null on balanced accuracy.

**Cross-validation:** repeated stratified k-fold (5-fold × 20 repeats); stratify regression folds by binned speed. Report mean ± 95% CI across repeats.

**Feature handling:** per fold, fit a `StandardScaler` on train only; high-dim representations (R1/R2, 110-dim) rely on the regularized model (Ridge/L2-logistic) — optionally a PCA(0.95) step recorded as a fixed preprocessing choice, applied identically to all representations for fairness.

## Interpretability (where profundity lives)

For the best representation × task, report component importances (Ridge coefficients / RF importances) and map them back to biomechanics. The target narrative: *speed prediction loads on hip–shoulder separation rate and lead-knee/stride timing — the proximal-to-distal kinetic chain.* This is what elevates "a regressor works" to "the discovered structure is biomechanically meaningful." Produce a short interpretation per winning representation.

## Architecture

- `src/analysis/__init__.py` — new package.
- `src/analysis/representations.py` — one function per representation with the common signature `extract(X: np.ndarray, dt: float, cfg: dict) -> tuple[np.ndarray, list[str]]`, plus a registry `REPRESENTATIONS: dict[str, callable]`. Pure functions, independently testable.
- `src/analysis/downstream_eval.py` — `evaluate_regression(R, y, ...)` and `evaluate_classification(R, y, ...)`, each returning CV metrics, baseline, and permutation p-value; plus `permutation_pvalue(...)`. No I/O.
- `src/analysis/run_representation_study.py` — orchestrator: load all 140 clips' `X` + labels, build each representation matrix, run T1/T2 for each, write a comparison table (`results/analysis/representation_study/comparison.csv`: representation × task × metric × CI × p-value × baseline) and per-winner importance reports. Also emits the reconstruction-failure summary alongside for the honest dual narrative.
- Reuses existing feature payloads; a small loader joins `clip_id` → labels from the pool CSV. To guarantee features exist for all 140, the orchestrator (or the plan) first ensures train+test features are extracted with the first-order config.

Small, focused files with clear interfaces; each representation and each evaluator is unit-testable on synthetic data.

## Honesty guardrails / pre-registration

- **Pre-register** T1 (speed regression, RidgeCV) as the primary endpoint and R0 (kinematic baseline) as the reference all dynamical representations must beat. Everything else is secondary/exploratory and labeled as such.
- Every reported score carries a **baseline** and a **permutation p-value**; no result is called a "win" without beating both baseline and null.
- 95% CIs on all CV metrics; **no strong claims** from the 5-sample pitch classes.
- The reconstruction failure is reported in the same study, not hidden — the contribution is explicitly "structure without reconstruction."

## Success criteria

- A reproducible study script produces the comparison table across R0–R3 (R4 if reached) for T1 and T2, each with CV mean ± CI, baseline, and permutation p-value.
- A clear verdict on the pre-registered endpoint: does any dynamical representation predict release speed **above the kinematic baseline and above the permutation null**, and by how much (with CI)?
- For any winning representation, an importance→biomechanics interpretation.
- The honest dual narrative (reconstruction fails; representation structure [does/does not] carry label signal) is documented in `docs/` and the paper draft.

## Out of scope

- New pose data, new pose extraction, additional camera views.
- Neural ODE unless gated ON at the Phase-1 checkpoint (then torch is installed).
- Changing the pose/feature layer or the existing dynamics-fit pipeline (this study consumes their outputs).
- Per-pitcher modeling (no reliable pitcher identity labels).
