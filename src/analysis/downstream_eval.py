#!/usr/bin/env python3
"""Cross-validated evaluation of a representation against a label, with baseline and null.

Every score returned here is accompanied by the two things the rest of this project learned
to demand: a trivial-baseline reference, and a permutation null. Four intermediate results
elsewhere in this project were false positives that survived only because one of those was
missing, so neither is optional.

  evaluate_regression      RidgeCV, repeated stratified K-fold over binned targets,
                           reported as CV R2 and MAE against a train-fold-mean baseline.
  evaluate_classification  L2-logistic with balanced class weights, reported as balanced
                           accuracy and macro-F1 against a stratified-random baseline.
  permutation_pvalue       fraction of label-shuffled runs scoring at or above observed.

Scaling and any dimensionality reduction are fit inside each fold on training data only.
No I/O here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-3, 4, 30)


@dataclass
class Result:
    """CV outcome for one representation x task, with its baseline and null."""

    metric: str
    score_mean: float
    score_ci: tuple[float, float]
    baseline_mean: float
    secondary: dict[str, float] = field(default_factory=dict)
    p_value: float | None = None
    n_samples: int = 0
    n_features: int = 0
    n_permutations: int = 0

    def as_row(self, **extra: Any) -> dict[str, Any]:
        return {
            **extra,
            "metric": self.metric,
            "score_mean": self.score_mean,
            "ci_low": self.score_ci[0],
            "ci_high": self.score_ci[1],
            "baseline_mean": self.baseline_mean,
            "beats_baseline": self.score_mean > self.baseline_mean,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            **{f"secondary_{k}": v for k, v in self.secondary.items()},
        }


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    a = np.asarray(values, dtype=np.float64)
    se = a.std(ddof=1) / np.sqrt(len(a))
    return (float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se))


def _strata(y: np.ndarray, n_bins: int, classification: bool) -> np.ndarray:
    """Fold strata: the labels themselves for classification, quantile bins for regression."""
    if classification:
        return y
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, edges)


def _pipeline(classification: bool, pca_variance: float | None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if pca_variance is not None:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))
    if classification:
        steps.append(("model", LogisticRegression(
            penalty="l2", class_weight="balanced", max_iter=5000, C=1.0)))
    else:
        steps.append(("model", RidgeCV(alphas=RIDGE_ALPHAS)))
    return Pipeline(steps)


def _cv_scores(
    R: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    n_splits: int,
    n_repeats: int,
    seed: int,
    pca_variance: float | None,
) -> tuple[list[float], list[float], dict[str, list[float]]]:
    """Returns (model scores per fold, baseline scores per fold, secondary metrics)."""
    strata = _strata(y, n_splits, classification)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    scores, baselines = [], []
    secondary: dict[str, list[float]] = {}

    for tr, te in cv.split(R, strata):
        pipe = _pipeline(classification, pca_variance)
        pipe.fit(R[tr], y[tr])
        pred = pipe.predict(R[te])

        if classification:
            scores.append(balanced_accuracy_score(y[te], pred))
            secondary.setdefault("macro_f1", []).append(
                f1_score(y[te], pred, average="macro", zero_division=0))
            dummy = DummyClassifier(strategy="stratified", random_state=seed).fit(R[tr], y[tr])
            baselines.append(balanced_accuracy_score(y[te], dummy.predict(R[te])))
        else:
            scores.append(r2_score(y[te], pred))
            secondary.setdefault("mae", []).append(mean_absolute_error(y[te], pred))
            # Baseline: predict the training-fold mean. R2 of that is <= 0 by construction.
            const = np.full_like(y[te], y[tr].mean(), dtype=np.float64)
            baselines.append(r2_score(y[te], const))
            secondary.setdefault("baseline_mae", []).append(mean_absolute_error(y[te], const))

    return scores, baselines, secondary


def permutation_pvalue(
    R: np.ndarray,
    y: np.ndarray,
    observed: float,
    *,
    classification: bool,
    n_permutations: int,
    n_splits: int,
    seed: int,
    pca_variance: float | None,
) -> float:
    """Empirical p: fraction of label-shuffled runs scoring >= observed.

    Shuffling the labels destroys any real association while preserving the representation's
    dimensionality and covariance, so the null captures how well this pipeline scores on
    noise of exactly this shape. Uses a single CV repeat per permutation for tractability.
    """
    rng = np.random.default_rng(seed)
    at_or_above = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        scores, _, _ = _cv_scores(
            R, y_perm, classification=classification, n_splits=n_splits,
            n_repeats=1, seed=seed, pca_variance=pca_variance,
        )
        if float(np.mean(scores)) >= observed:
            at_or_above += 1
    # +1 smoothing: with n permutations the smallest reportable p is 1/(n+1), never 0.
    return (at_or_above + 1) / (n_permutations + 1)


def evaluate_regression(
    R: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 7,
    n_permutations: int = 200,
    pca_variance: float | None = None,
) -> Result:
    scores, baselines, secondary = _cv_scores(
        R, y, classification=False, n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, pca_variance=pca_variance,
    )
    observed = float(np.mean(scores))
    p = permutation_pvalue(
        R, y, observed, classification=False, n_permutations=n_permutations,
        n_splits=n_splits, seed=seed, pca_variance=pca_variance,
    ) if n_permutations else None
    return Result(
        metric="cv_r2",
        score_mean=observed,
        score_ci=_ci95(scores),
        baseline_mean=float(np.mean(baselines)),
        secondary={k: float(np.mean(v)) for k, v in secondary.items()},
        p_value=p,
        n_samples=len(y),
        n_features=R.shape[1],
        n_permutations=n_permutations,
    )


def evaluate_classification(
    R: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 7,
    n_permutations: int = 200,
    pca_variance: float | None = None,
) -> Result:
    scores, baselines, secondary = _cv_scores(
        R, y, classification=True, n_splits=n_splits, n_repeats=n_repeats,
        seed=seed, pca_variance=pca_variance,
    )
    observed = float(np.mean(scores))
    p = permutation_pvalue(
        R, y, observed, classification=True, n_permutations=n_permutations,
        n_splits=n_splits, seed=seed, pca_variance=pca_variance,
    ) if n_permutations else None
    return Result(
        metric="balanced_accuracy",
        score_mean=observed,
        score_ci=_ci95(scores),
        baseline_mean=float(np.mean(baselines)),
        secondary={k: float(np.mean(v)) for k, v in secondary.items()},
        p_value=p,
        n_samples=len(y),
        n_features=R.shape[1],
        n_permutations=n_permutations,
    )
