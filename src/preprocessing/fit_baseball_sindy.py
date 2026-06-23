#!/usr/bin/env python3
"""Fit baseline PySINDy models for processed Stage B baseball feature trajectories."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pysindy as ps
import yaml
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.fit_utd_mhad_sindy import (
    build_model,
    candidate_configs,
    coefficient_rows,
    equation_text,
    load_feature_payload,
    pick_best_result,
    write_csv,
    write_json,
    write_text,
)
from src.preprocessing.sindy_utils import (
    fit_feature_scaler,
    scaler_summary,
    transform_derivatives,
    transform_trajectories,
)


def build_weakform_model(candidate: dict[str, Any], scaled_Xs: list[np.ndarray], dt: float, weak_cfg: dict[str, Any]) -> ps.SINDy:
    """Fit weak-form SINDy across variable-length trajectories by stacking each
    trajectory's weak system (features + integrated target) computed on its own
    grid, then a single STLSQ. Uses ALL data (no truncation). The robust weak
    coefficients are transplanted into a plain PolynomialLibrary model so the
    downstream predict/simulate path is unchanged.
    """
    degree = int(candidate["polynomial_degree"])
    include_bias = bool(candidate["include_bias"])
    # Match the plain build_model library exactly so weak/plain feature ordering
    # stays aligned for the coefficient transplant (matters once degree >= 2).
    include_interaction = bool(candidate["include_interaction"])
    K = int(weak_cfg.get("K", 100))
    # WeakPDELibrary places its K test-function subdomains using the global RNG
    # and exposes no seed, so a single draw is both non-reproducible and high
    # variance. Seed locally for reproducibility, and stack `n_seeds` independent
    # subdomain draws (n_seeds * K test functions total) to drive the variance
    # down. Restore the prior RNG state so the rest of the pipeline is unaffected.
    seed = int(weak_cfg.get("seed", 7))
    n_seeds = max(1, int(weak_cfg.get("n_seeds", 1)))
    rng_state = np.random.get_state()
    thetas: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for draw in range(n_seeds):
        np.random.seed(seed + draw)
        for x in scaled_Xs:
            x = np.asarray(x, dtype=np.float64)
            n = x.shape[0]
            k_eff = max(8, min(K, n // 2))
            grid = np.arange(n, dtype=np.float64) * dt
            lib = ps.WeakPDELibrary(
                function_library=ps.PolynomialLibrary(degree=degree, include_bias=include_bias, include_interaction=include_interaction),
                spatiotemporal_grid=grid,
                K=k_eff,
            )
            lib.fit(x)
            thetas.append(np.asarray(lib.transform(x), dtype=np.float64))
            targets.append(np.asarray(lib.convert_u_dot_integral(x), dtype=np.float64))
    np.random.set_state(rng_state)
    Theta = np.vstack(thetas)
    Y = np.vstack(targets)
    weak_opt = ps.STLSQ(
        threshold=float(candidate["stlsq_threshold"]),
        alpha=float(candidate["stlsq_alpha"]),
        max_iter=int(candidate["stlsq_max_iter"]),
        normalize_columns=False,
    )
    weak_opt.fit(Theta, Y)
    weak_coef = np.asarray(weak_opt.coef_)

    plain = build_model(candidate)
    plain.fit(np.asarray(scaled_Xs[0], dtype=np.float64), t=dt)  # initialize structures
    plain_coef_shape = np.asarray(plain.optimizer.coef_).shape
    if weak_coef.shape != plain_coef_shape:
        raise ValueError(f"weak/plain coef shape mismatch: {weak_coef.shape} vs {plain_coef_shape}")
    plain.optimizer.coef_ = weak_coef
    return plain


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.baseball.example.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--split",
        default="smoke_test",
        help="Baseball feature split to model.",
    )
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=("pitching", "batting"),
        help="Task types to train.",
    )
    parser.add_argument(
        "--group-by",
        default="",
        help="Optional metadata field used to train separate subgroup models, e.g. resolved_handedness.",
    )
    parser.add_argument(
        "--min-group-clips",
        type=int,
        default=3,
        help="Minimum number of clips required to fit a subgroup model.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_model(
    model: ps.SINDy,
    Xs: list[np.ndarray],
    X_dots: list[np.ndarray],
    dt: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_trial: list[dict[str, float]] = []
    for trial_idx, (X, X_dot) in enumerate(zip(Xs, X_dots, strict=True)):
        pred = model.predict(X)
        per_trial.append(
            {
                "trial_index": trial_idx,
                "r2": float(r2_score(X_dot, pred, multioutput="variance_weighted")),
                "rmse": float(np.sqrt(mean_squared_error(X_dot, pred))),
                "num_frames": int(X.shape[0]),
            }
        )

    aggregate = {
        "global_r2": float(model.score(Xs, t=dt, x_dot=X_dots)),
        "mean_trial_r2": float(np.mean([row["r2"] for row in per_trial])),
        "mean_trial_rmse": float(np.mean([row["rmse"] for row in per_trial])),
    }
    return aggregate, per_trial


def split_validation_indices(
    num_rows: int,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if num_rows < 3 or holdout_fraction <= 0.0:
        return list(range(num_rows)), []

    holdout_count = max(1, int(round(num_rows * holdout_fraction)))
    holdout_count = min(holdout_count, num_rows - 1)
    rng = np.random.default_rng(seed)
    indices = np.arange(num_rows)
    rng.shuffle(indices)
    val_idx = sorted(indices[:holdout_count].tolist())
    train_idx = sorted(indices[holdout_count:].tolist())
    return train_idx, val_idx


def subset_by_indices[T](items: list[T], indices: list[int]) -> list[T]:
    return [items[idx] for idx in indices]


def main() -> None:
    args = parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())

    modeling_cfg = config["modeling"]
    ensemble_cfg = dict(modeling_cfg.get("ensemble", {}))
    weak_cfg = dict(modeling_cfg.get("weak_form", {}))
    use_weak = bool(weak_cfg.get("enabled", False))
    validation_fraction = float(modeling_cfg.get("selection_validation_fraction", 0.0))
    validation_seed = int(modeling_cfg.get("selection_validation_seed", 7))
    feature_root = PROJECT_ROOT / "data" / "baseball" / "processed" / args.split / modeling_cfg["input_feature_subdir"]
    manifest_rows = read_csv_rows(feature_root / "manifest.csv")
    results_root = PROJECT_ROOT / "results" / "equations" / "baseball" / args.split
    summary_path = results_root / "summary.csv"
    if args.group_by:
        summary_path = results_root / f"summary_by_{args.group_by}.csv"

    summary_rows: list[dict[str, Any]] = []

    for task_type in args.task_types:
        task_rows = [row for row in manifest_rows if row["task_type"] == task_type]
        if not task_rows:
            continue

        loaded_rows: list[dict[str, Any]] = []
        for row in task_rows:
            path = PROJECT_ROOT / row["filepath"]
            X, X_dot, current_feature_names, metadata = load_feature_payload(path)
            loaded_rows.append(
                {
                    "row": row,
                    "X": X,
                    "X_dot": X_dot,
                    "feature_names": current_feature_names,
                    "metadata": metadata,
                }
            )

        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        if args.group_by:
            for item in loaded_rows:
                group_value = str(item["metadata"].get(args.group_by, "")).strip() or "unknown"
                grouped_rows.setdefault(group_value, []).append(item)
        else:
            grouped_rows["all"] = loaded_rows

        for group_value, group_items in grouped_rows.items():
            if len(group_items) < args.min_group_clips:
                continue

            Xs: list[np.ndarray] = []
            X_dots: list[np.ndarray] = []
            trial_log_rows: list[dict[str, Any]] = []
            feature_names: list[str] | None = None
            dt_values: set[float] = set()

            for item in group_items:
                row = item["row"]
                metadata = item["metadata"]
                current_feature_names = item["feature_names"]
                if feature_names is None:
                    feature_names = current_feature_names
                elif feature_names != current_feature_names:
                    raise ValueError(f"Inconsistent feature ordering in {PROJECT_ROOT / row['filepath']}")
                Xs.append(item["X"])
                X_dots.append(item["X_dot"])
                dt_values.add(float(row["dt"]))
                trial_log_rows.append(
                    {
                        "clip_id": metadata.get("clip_id", row["clip_id"]),
                        "source_name": row["source_name"],
                        "task_type": row["task_type"],
                        "group_value": group_value,
                        "filepath": row["filepath"],
                        "num_frames": int(row["num_frames"]),
                    }
                )

            dt = float(np.median(list(dt_values)))
            scaler = fit_feature_scaler(Xs)
            scaled_Xs = transform_trajectories(Xs, scaler) if modeling_cfg.get("scale_features", False) else Xs
            scaled_X_dots = transform_derivatives(X_dots, scaler) if modeling_cfg.get("scale_features", False) else X_dots
            train_idx, val_idx = split_validation_indices(len(group_items), validation_fraction, validation_seed)
            selection_Xs = subset_by_indices(scaled_Xs, train_idx)
            selection_X_dots = subset_by_indices(scaled_X_dots, train_idx)
            validation_Xs = subset_by_indices(scaled_Xs, val_idx)
            validation_X_dots = subset_by_indices(scaled_X_dots, val_idx)

            selection_timepoints = int(sum(x.shape[0] for x in selection_Xs))
            full_timepoints = int(sum(x.shape[0] for x in scaled_Xs))
            sweep_results: list[dict[str, Any]] = []
            for candidate in candidate_configs(modeling_cfg):
                if use_weak:
                    model = build_weakform_model(candidate, selection_Xs, dt, weak_cfg)
                else:
                    model = build_candidate_model(candidate, selection_timepoints, ensemble_cfg)
                if not use_weak:
                    model.fit(selection_Xs, t=dt, x_dot=selection_X_dots, feature_names=feature_names)
                selection_metrics, selection_per_trial = evaluate_model(model, selection_Xs, selection_X_dots, dt)
                if validation_Xs:
                    validation_metrics, validation_per_trial = evaluate_model(model, validation_Xs, validation_X_dots, dt)
                else:
                    validation_metrics, validation_per_trial = selection_metrics, selection_per_trial
                coef_rows = coefficient_rows(model, feature_names)
                sweep_results.append(
                    {
                        "candidate_config": candidate,
                        "selection_metrics": selection_metrics,
                        "selection_per_trial_metrics": selection_per_trial,
                        "validation_metrics": validation_metrics,
                        "validation_per_trial_metrics": validation_per_trial,
                        "num_equation_terms": len(coef_rows),
                        "selection_penalty": float(modeling_cfg.get("complexity_penalty", 0.0)) * len(coef_rows),
                        "coefficient_rows": coef_rows,
                    }
                )
            normalized_results = [{**result, "aggregate_metrics": result["validation_metrics"]} for result in sweep_results]
            best_result = pick_best_result(normalized_results, modeling_cfg["selection_metric"])

            if use_weak:
                final_model = build_weakform_model(best_result["candidate_config"], scaled_Xs, dt, weak_cfg)
            else:
                final_model = build_candidate_model(best_result["candidate_config"], full_timepoints, ensemble_cfg)
            if not use_weak:
                final_model.fit(scaled_Xs, t=dt, x_dot=scaled_X_dots, feature_names=feature_names)
            aggregate_metrics, per_trial_metrics = evaluate_model(final_model, scaled_Xs, scaled_X_dots, dt)
            coef_rows = coefficient_rows(final_model, feature_names)

            task_root = results_root / task_type
            if args.group_by:
                task_root = task_root / args.group_by / group_value
            task_root.mkdir(parents=True, exist_ok=True)

            with (task_root / "model.pkl").open("wb") as handle:
                pickle.dump(final_model, handle)

            write_csv(task_root / "coefficients.csv", ["state_variable", "library_term", "coefficient"], coef_rows)
            write_text(task_root / "equations.txt", equation_text(final_model))
            write_json(
                task_root / "metrics.json",
                {
                    "task_type": task_type,
                    "group_by": args.group_by,
                    "group_value": group_value,
                    "num_clips": len(group_items),
                    "dt": dt,
                    "scaler_applied": bool(modeling_cfg.get("scale_features", False)),
                    "selection_train_clips": len(train_idx),
                    "selection_validation_clips": len(val_idx),
                    **aggregate_metrics,
                },
            )
            write_csv(task_root / "trial_metrics.csv", ["trial_index", "r2", "rmse", "num_frames"], per_trial_metrics)
            write_json(
                task_root / "config.json",
                {
                    "task_type": task_type,
                    "group_by": args.group_by,
                    "group_value": group_value,
                    "split": args.split,
                    "feature_names": feature_names,
                    "modeling": best_result["candidate_config"],
                    "num_clips": len(group_items),
                    "dt": dt,
                    "selection_metric": modeling_cfg["selection_metric"],
                    "selection_validation_fraction": validation_fraction,
                    "selection_validation_seed": validation_seed,
                    "selection_train_clips": len(train_idx),
                    "selection_validation_clips": len(val_idx),
                },
            )
            write_json(
                task_root / "scaler.json",
                {
                    "task_type": task_type,
                    "group_by": args.group_by,
                    "group_value": group_value,
                    "scale_features": bool(modeling_cfg.get("scale_features", False)),
                    "feature_scaler": scaler,
                },
            )
            write_csv(task_root / "scaler.csv", ["feature_name", "mean", "scale"], scaler_summary(scaler, feature_names))
            write_csv(
                task_root / "model_selection.csv",
                [
                    "polynomial_degree",
                    "stlsq_threshold",
                    "stlsq_alpha",
                    "selection_global_r2",
                    "selection_mean_trial_r2",
                    "selection_mean_trial_rmse",
                    "validation_global_r2",
                    "validation_mean_trial_r2",
                    "validation_mean_trial_rmse",
                    "num_equation_terms",
                    "selection_penalty",
                    "selection_score",
                ],
                [
                    {
                        "polynomial_degree": result["candidate_config"]["polynomial_degree"],
                        "stlsq_threshold": result["candidate_config"]["stlsq_threshold"],
                        "stlsq_alpha": result["candidate_config"]["stlsq_alpha"],
                        "selection_global_r2": result["selection_metrics"]["global_r2"],
                        "selection_mean_trial_r2": result["selection_metrics"]["mean_trial_r2"],
                        "selection_mean_trial_rmse": result["selection_metrics"]["mean_trial_rmse"],
                        "validation_global_r2": result["validation_metrics"]["global_r2"],
                        "validation_mean_trial_r2": result["validation_metrics"]["mean_trial_r2"],
                        "validation_mean_trial_rmse": result["validation_metrics"]["mean_trial_rmse"],
                        "num_equation_terms": result["num_equation_terms"],
                        "selection_penalty": result["selection_penalty"],
                        "selection_score": result["validation_metrics"]["mean_trial_r2"] - result["selection_penalty"],
                    }
                    for result in sweep_results
                ],
            )
            write_csv(task_root / "clips.csv", ["clip_id", "source_name", "task_type", "group_value", "filepath", "num_frames"], trial_log_rows)

            summary_rows.append(
                {
                    "task_type": task_type,
                    "group_by": args.group_by,
                    "group_value": group_value,
                    "num_clips": len(group_items),
                    "dt": dt,
                    "global_r2": aggregate_metrics["global_r2"],
                    "mean_trial_r2": aggregate_metrics["mean_trial_r2"],
                    "mean_trial_rmse": aggregate_metrics["mean_trial_rmse"],
                    "num_equation_terms": len(coef_rows),
                    "polynomial_degree": best_result["candidate_config"]["polynomial_degree"],
                    "stlsq_threshold": best_result["candidate_config"]["stlsq_threshold"],
                    "stlsq_alpha": best_result["candidate_config"]["stlsq_alpha"],
                    "selection_validation_clips": len(val_idx),
                }
            )

    write_csv(
        summary_path,
        [
            "task_type",
            "group_by",
            "group_value",
            "num_clips",
            "dt",
            "global_r2",
            "mean_trial_r2",
            "mean_trial_rmse",
            "num_equation_terms",
            "polynomial_degree",
            "stlsq_threshold",
            "stlsq_alpha",
            "selection_validation_clips",
        ],
        summary_rows,
    )

    print(f"trained models: {len(summary_rows)}")
    print(f"results root: {results_root.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
