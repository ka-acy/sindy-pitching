# Reproducing the technical paper

All commands are run from the project root with the repository virtual environment. **The operative configuration must be passed explicitly** — the refresh script's default (`config/preprocessing.baseball.example.yaml`) does not reproduce Section 5.

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

Reproduce Section 5 (operative first-order train + held-out evaluation):

```bash
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce the weak-form model of Section 5.2 and the Section 8 selection-rule repair. Note that fit and evaluation write to the same
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

Reproduce the frozen control in Section 5:

```bash
./.venv/bin/python src/preprocessing/evaluate_static_baseline.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml --eval-split test
```

Reproduce the state-representation sweep in Section 6:

```bash
./.venv/bin/python src/preprocessing/diagnose_derivative_fit.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce Sections 5.3, 6.2 and 7 (one-step skill, noise floor, horizon sweep):

```bash
./.venv/bin/python src/preprocessing/diagnose_increment_skill.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce the free-running control comparison in Section 5:

```bash
./.venv/bin/python src/preprocessing/evaluate_forecast_baselines.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Reproduce Section 6 (fitted SINDy equations on the short-horizon task):

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

Reproduce Sections 7.1 and 7.2 (sparse fit on a delay-augmented state, the local motion controls and their resplit margins,
and the coupling ablation):

```bash
./.venv/bin/python src/analysis/fit_delay_sindy.py
```

Reproduce Section 7.3 (the same comparison rebuilt from raw pose under causal and no smoothing;
reads `data/baseball/standardized_pose/`):

```bash
./.venv/bin/python src/analysis/test_causal_smoothing.py
```

Reproduce the Section 7.4 label study (release velocity and pitch type). This runs 200 permutations per
representation per task and takes several minutes; `--quick` reduces it for a smoke run:

```bash
./.venv/bin/python src/analysis/run_representation_study.py
```

Reproduce the Section 6 estimator comparison:

```bash
./.venv/bin/python src/analysis/diagnose_sparse_fit_loss.py
```

Reproduce the Section 7.4 coupling test (five held-out splits, 60 null draws each):

```bash
./.venv/bin/python src/analysis/test_coupling_significance.py --n-null 60 --null-splits 5
```

Regenerate the figures. They are build artifacts under `results/`, which is excluded from
version control, so this must run before the paper is built:

```bash
./.venv/bin/python src/analysis/make_paper_figures.py
```

Run the delay-augmented configuration end to end (Section 7):

```bash
for split in train test; do
  ./.venv/bin/python src/preprocessing/extract_baseball_features.py \
    --config config/preprocessing.baseball.delay.yaml --split $split
done
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py \
  --config config/preprocessing.baseball.delay.yaml --split train --task-types pitching
./.venv/bin/python src/preprocessing/evaluate_baseball_sindy.py \
  --config config/preprocessing.baseball.delay.yaml \
  --train-split train --eval-split test --task-types pitching
```

Compare the old and repaired selection rules (Section 8):

```bash
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py \
  --config config/preprocessing.baseball.unaligned.handedness.weakform.yaml \
  --split train --task-types pitching
./.venv/bin/python src/preprocessing/fit_baseball_sindy.py \
  --config config/preprocessing.baseball.unaligned.handedness.weakform.gated.yaml \
  --split train --task-types pitching
```

Reproduce the Section 5 paired control intervals (requires the per-clip control scores, which
`evaluate_static_baseline.py` writes alongside its summary):

```bash
./.venv/bin/python src/preprocessing/evaluate_static_baseline.py
./.venv/bin/python src/analysis/paired_control_intervals.py
```

Reproduce the Section 7.2 stability comparison:

```bash
./.venv/bin/python src/analysis/stabilize_delay_fit.py --n-splits 40
```

Reproduce the Section 7.4 directional asymmetry test:

```bash
./.venv/bin/python src/analysis/test_kinetic_chain_asymmetry.py --n-null 60 --null-splits 5
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
- `results/analysis/sparse_fit_loss.csv`
- `results/analysis/delay_sindy/skill_by_depth.csv`
- `results/analysis/delay_sindy/equations.txt`
- `results/analysis/coupling_significance/summary.json`
- `results/analysis/delay_sindy/stability.csv`
- `results/analysis/kinetic_chain_asymmetry/summary.json`
- `results/analysis/paired_control_intervals.csv`
- `results/evaluation/baseball/static_baseline/per_clip.csv`
- `results/analysis/representation_study/comparison.csv`

Analysis code is Python 3.12. Numerical work uses NumPy [31] and SciPy [32]; Savitzky-Golay
smoothing and differentiation use `scipy.signal.savgol_filter`. Ridge regression,
cross-validation, standardization, and R2 computation use scikit-learn [33]. Sparse model
identification uses PySINDy 2.1.0 [9], [10]. Pose extraction uses MediaPipe [19]. Exact pinned
versions are in `requirements.txt`.
