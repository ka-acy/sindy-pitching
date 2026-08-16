#!/usr/bin/env python3
"""Fit baseline PySINDy models for the processed UTD-MHAD feature trajectories."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pysindy as ps
import yaml
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.sindy_utils import (
    fit_feature_scaler,
    scaler_summary,
    transform_derivatives,
    transform_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.utd_mhad.example.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--split",
        choices=("smoke_test", "proxy_subset"),
        default="proxy_subset",
        help="Feature split to model.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=("batting_like", "pitching_like"),
        help="Subset labels to train.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_feature_payload(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    obj = np.load(path)
    X = np.asarray(obj["X"], dtype=np.float64)
    X_dot = np.asarray(obj["X_dot"], dtype=np.float64)
    feature_names = obj["feature_names"].tolist()
    metadata = json.loads(str(obj["metadata"]))
    return X, X_dot, feature_names, metadata


def build_model(model_cfg: dict[str, Any]) -> ps.SINDy:
    library = ps.PolynomialLibrary(
        degree=int(model_cfg["polynomial_degree"]),
        include_bias=bool(model_cfg["include_bias"]),
        include_interaction=bool(model_cfg["include_interaction"]),
    )
    optimizer = ps.STLSQ(
        threshold=float(model_cfg["stlsq_threshold"]),
        alpha=float(model_cfg["stlsq_alpha"]),
        max_iter=int(model_cfg["stlsq_max_iter"]),
        normalize_columns=False,
    )
    return ps.SINDy(
        optimizer=optimizer,
        feature_library=library,
    )


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

    agg = {
        "global_r2": float(model.score(Xs, t=dt, x_dot=X_dots)),
        "mean_trial_r2": float(np.mean([row["r2"] for row in per_trial])),
        "mean_trial_rmse": float(np.mean([row["rmse"] for row in per_trial])),
    }
    return agg, per_trial


def candidate_configs(model_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = model_cfg.get("sweep", {})
    degrees = sweep.get("polynomial_degree", [model_cfg["polynomial_degree"]])
    thresholds = sweep.get("stlsq_threshold", [model_cfg["stlsq_threshold"]])
    alphas = sweep.get("stlsq_alpha", [model_cfg["stlsq_alpha"]])

    configs: list[dict[str, Any]] = []
    for degree, threshold, alpha in itertools.product(degrees, thresholds, alphas):
        cfg = dict(model_cfg)
        cfg["polynomial_degree"] = degree
        cfg["stlsq_threshold"] = threshold
        cfg["stlsq_alpha"] = alpha
        configs.append(cfg)
    return configs


def selection_is_degenerate(results: list[dict[str, Any]], min_accuracy: float) -> bool:
    """True when no candidate reaches a usable accuracy, so the choice is not meaningful.

    `sparsity_weighted_r2` maximizes accuracy minus a term-count penalty. That is sound while
    candidates differ in accuracy, and vacuous when they do not: with every candidate at the
    same accuracy the penalty decides alone and the sweep returns whichever model is emptiest.
    This is not hypothetical -- it selected a two-term weak-form model whose accuracy was
    indistinguishable from a 37-term one, and that model then read as a large improvement on a
    metric that rewards predicting no change.

    The rule cannot tell "all tied and good" from "all tied and worthless", so callers get this
    flag and can refuse to interpret the selection.
    """
    best = max(float(r["aggregate_metrics"]["mean_trial_r2"]) for r in results)
    return best < min_accuracy


def pick_best_result(
    results: list[dict[str, Any]],
    metric_name: str,
    accuracy_tolerance: float = 0.0,
    min_accuracy: float = float("-inf"),
) -> dict[str, Any]:
    if metric_name == "accuracy_gated_sparsity":
        # Sparsity breaks ties; it does not overrule accuracy. Keep every candidate within
        # `accuracy_tolerance` of the best accuracy, then take the sparsest of those. With a
        # tolerance of zero this reduces to "most accurate, ties broken by fewest terms".
        best_acc = max(float(r["aggregate_metrics"]["mean_trial_r2"]) for r in results)
        tied = [r for r in results
                if float(r["aggregate_metrics"]["mean_trial_r2"]) >= best_acc - accuracy_tolerance]
        return min(
            tied,
            key=lambda result: (
                float(result["num_equation_terms"]),
                -float(result["aggregate_metrics"]["mean_trial_r2"]),
            ),
        )
    if metric_name == "sparsity_weighted_r2":
        return max(
            results,
            key=lambda result: (
                float(result["aggregate_metrics"]["mean_trial_r2"])
                - float(result["selection_penalty"]),
                float(result["aggregate_metrics"]["global_r2"]),
                -float(result["num_equation_terms"]),
            ),
        )
    return max(
        results,
        key=lambda result: (
            float(result["aggregate_metrics"][metric_name]),
            -float(result["aggregate_metrics"]["mean_trial_rmse"]),
            -float(result["num_equation_terms"]),
        ),
    )


def coefficient_rows(
    model: ps.SINDy,
    state_names: list[str],
) -> list[dict[str, Any]]:
    library_names = model.get_feature_names()
    coefficients = model.coefficients()
    rows: list[dict[str, Any]] = []
    for state_idx, state_name in enumerate(state_names):
        for term_idx, term_name in enumerate(library_names):
            coef = float(coefficients[state_idx, term_idx])
            if abs(coef) < 1e-12:
                continue
            rows.append(
                {
                    "state_variable": state_name,
                    "library_term": term_name,
                    "coefficient": coef,
                }
            )
    return rows


def equation_text(model: ps.SINDy) -> str:
    lines: list[str] = []
    equations = model.equations()
    feature_names = model.feature_names
    for state_name, eq in zip(feature_names, equations, strict=True):
        lines.append(f"d({state_name})/dt = {eq}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config((project_root / args.config).resolve())

    modeling_cfg = config["modeling"]
    feature_root = (
        project_root
        / config["dataset"]["processed_root"]
        / args.split
        / modeling_cfg["input_feature_subdir"]
    )
    manifest_rows = read_csv_rows(feature_root / "manifest.csv")
    results_root = project_root / modeling_cfg["results_root"] / args.split

    summary_rows: list[dict[str, Any]] = []

    for label in args.labels:
        label_rows = [row for row in manifest_rows if row["subset_label"] == label]
        if not label_rows:
            continue

        Xs: list[np.ndarray] = []
        X_dots: list[np.ndarray] = []
        trial_log_rows: list[dict[str, Any]] = []
        feature_names: list[str] | None = None
        dt_values: set[float] = set()

        for row in label_rows:
            path = project_root / row["filepath"]
            X, X_dot, current_feature_names, metadata = load_feature_payload(path)
            if feature_names is None:
                feature_names = current_feature_names
            elif feature_names != current_feature_names:
                raise ValueError(f"Inconsistent feature ordering in {path}")

            Xs.append(X)
            X_dots.append(X_dot)
            dt_values.add(float(row["dt"]))
            trial_log_rows.append(
                {
                    "subject_id": int(row["subject_id"]),
                    "action_id": int(row["action_id"]),
                    "trial_id": int(row["trial_id"]),
                    "filepath": row["filepath"],
                    "num_frames": int(row["num_frames"]),
                    "metadata_subset_label": metadata["subset_label"],
                }
            )

        if len(dt_values) != 1:
            raise ValueError(f"Expected a single dt per label, found {sorted(dt_values)} for {label}")
        dt = next(iter(dt_values))

        scaler = fit_feature_scaler(Xs)
        scaled_Xs = transform_trajectories(Xs, scaler) if modeling_cfg.get("scale_features", False) else Xs
        scaled_X_dots = transform_derivatives(X_dots, scaler) if modeling_cfg.get("scale_features", False) else X_dots

        sweep_results: list[dict[str, Any]] = []
        for candidate in candidate_configs(modeling_cfg):
            model = build_model(candidate)
            model.fit(scaled_Xs, t=dt, x_dot=scaled_X_dots, feature_names=feature_names)
            aggregate_metrics, per_trial_metrics = evaluate_model(model, scaled_Xs, scaled_X_dots, dt)
            coef_rows = coefficient_rows(model, feature_names)
            sweep_results.append(
                {
                    "candidate_config": candidate,
                    "model": model,
                    "aggregate_metrics": aggregate_metrics,
                    "per_trial_metrics": per_trial_metrics,
                    "num_equation_terms": len(coef_rows),
                    "selection_penalty": float(modeling_cfg.get("complexity_penalty", 0.0)) * len(coef_rows),
                    "coefficient_rows": coef_rows,
                }
            )

        best_result = pick_best_result(sweep_results, modeling_cfg["selection_metric"])
        model = best_result["model"]
        aggregate_metrics = best_result["aggregate_metrics"]
        per_trial_metrics = best_result["per_trial_metrics"]

        label_root = results_root / label
        label_root.mkdir(parents=True, exist_ok=True)

        with (label_root / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

        coef_rows = best_result["coefficient_rows"]
        write_csv(
            label_root / "coefficients.csv",
            ["state_variable", "library_term", "coefficient"],
            coef_rows,
        )
        write_text(label_root / "equations.txt", equation_text(model))
        write_json(
            label_root / "metrics.json",
            {
                "subset_label": label,
                "num_trials": len(label_rows),
                "dt": dt,
                "scaler_applied": bool(modeling_cfg.get("scale_features", False)),
                **aggregate_metrics,
            },
        )
        write_csv(
            label_root / "trial_metrics.csv",
            ["trial_index", "r2", "rmse", "num_frames"],
            per_trial_metrics,
        )
        write_json(
            label_root / "config.json",
            {
                "subset_label": label,
                "split": args.split,
                "feature_names": feature_names,
                "modeling": best_result["candidate_config"],
                "num_trials": len(label_rows),
                "dt": dt,
                "selection_metric": modeling_cfg["selection_metric"],
            },
        )
        write_json(
            label_root / "scaler.json",
            {
                "subset_label": label,
                "scale_features": bool(modeling_cfg.get("scale_features", False)),
                "feature_scaler": scaler,
            },
        )
        write_csv(
            label_root / "scaler.csv",
            ["feature_name", "mean", "scale"],
            scaler_summary(scaler, feature_names),
        )
        write_csv(
            label_root / "model_selection.csv",
            [
                "polynomial_degree",
                "stlsq_threshold",
                "stlsq_alpha",
                "global_r2",
                "mean_trial_r2",
                "mean_trial_rmse",
                "num_equation_terms",
                "selection_penalty",
                "selection_score",
            ],
            [
                {
                    "polynomial_degree": result["candidate_config"]["polynomial_degree"],
                    "stlsq_threshold": result["candidate_config"]["stlsq_threshold"],
                    "stlsq_alpha": result["candidate_config"]["stlsq_alpha"],
                    "global_r2": result["aggregate_metrics"]["global_r2"],
                    "mean_trial_r2": result["aggregate_metrics"]["mean_trial_r2"],
                    "mean_trial_rmse": result["aggregate_metrics"]["mean_trial_rmse"],
                    "num_equation_terms": result["num_equation_terms"],
                    "selection_penalty": result["selection_penalty"],
                    "selection_score": result["aggregate_metrics"]["mean_trial_r2"] - result["selection_penalty"],
                }
                for result in sweep_results
            ],
        )
        write_csv(
            label_root / "trials.csv",
            ["subject_id", "action_id", "trial_id", "filepath", "num_frames", "metadata_subset_label"],
            trial_log_rows,
        )

        summary_rows.append(
            {
                "subset_label": label,
                "num_trials": len(label_rows),
                "dt": dt,
                "global_r2": aggregate_metrics["global_r2"],
                "mean_trial_r2": aggregate_metrics["mean_trial_r2"],
                "mean_trial_rmse": aggregate_metrics["mean_trial_rmse"],
                "num_equation_terms": len(coef_rows),
                "polynomial_degree": best_result["candidate_config"]["polynomial_degree"],
                "stlsq_threshold": best_result["candidate_config"]["stlsq_threshold"],
                "stlsq_alpha": best_result["candidate_config"]["stlsq_alpha"],
            }
        )

    write_csv(
        results_root / "summary.csv",
        [
            "subset_label",
            "num_trials",
            "dt",
            "global_r2",
            "mean_trial_r2",
            "mean_trial_rmse",
            "num_equation_terms",
            "polynomial_degree",
            "stlsq_threshold",
            "stlsq_alpha",
        ],
        summary_rows,
    )

    print(f"trained models: {len(summary_rows)}")
    print(f"results root: {results_root.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
