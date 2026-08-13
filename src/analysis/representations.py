#!/usr/bin/env python3
"""Fixed-length per-clip representations of a pitching feature trajectory.

Each extractor maps one clip's feature trajectory `X` (T x n_features) plus `dt` to a
fixed-length vector and matching feature names, so clips of different length become rows of
a design matrix for the downstream label-prediction tasks.

Every representation is computed independently per clip, so there is no leakage across the
cross-validation folds used downstream.

  R0  kinematic_summary   per-column descriptive statistics. The control: this carries no
                          dynamical model at all, and any dynamical representation has to
                          beat it to justify itself.
  R1  sindy_coefficients  per-clip SINDy fit, flattened coefficient matrix.

R2-R4 (weak-form, DMD, Hankel-DMD) are added only after the pre-registered endpoints have
been evaluated on R0 and R1, so that a winner cannot be selected post hoc.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Summary statistics for R0, in a fixed order so the feature names line up.
_R0_STATS: list[tuple[str, Callable[[np.ndarray], float]]] = [
    ("mean", lambda c: float(np.mean(c))),
    ("std", lambda c: float(np.std(c))),
    ("min", lambda c: float(np.min(c))),
    ("max", lambda c: float(np.max(c))),
    ("range", lambda c: float(np.max(c) - np.min(c))),
    ("argmax_norm", lambda c: float(np.argmax(c) / max(len(c) - 1, 1))),
    ("mean_abs_diff", lambda c: float(np.mean(np.abs(np.diff(c)))) if len(c) > 1 else 0.0),
]


def kinematic_summary(
    X: np.ndarray, dt: float, feature_names: list[str], cfg: dict | None = None
) -> tuple[np.ndarray, list[str]]:
    """R0 -- per-feature descriptive statistics. 7 x n_features dimensional.

    Deliberately contains no dynamics: mean, spread, extremes, peak timing, and mean
    per-frame speed. This is the "do you even need a dynamical model?" control.
    """
    values, names = [], []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        for stat_name, fn in _R0_STATS:
            values.append(fn(col))
            names.append(f"{name}__{stat_name}")
    return np.asarray(values, dtype=np.float64), names


def sindy_coefficients(
    X: np.ndarray, dt: float, feature_names: list[str], cfg: dict | None = None
) -> tuple[np.ndarray, list[str]]:
    """R1 -- per-clip SINDy fit, coefficient matrix flattened.

    Degree-1 polynomial library with bias, STLSQ at the operative threshold, fit to this
    clip alone. Dimension is n_features x (n_features + 1).

    Columns are standardized before fitting so the threshold means the same thing across
    clips of differing amplitude; without that, the sparsity pattern would track clip
    amplitude rather than clip dynamics.
    """
    cfg = cfg or {}
    threshold = float(cfg.get("stlsq_threshold", 0.2))
    alpha = float(cfg.get("stlsq_alpha", 0.001))
    max_iter = int(cfg.get("stlsq_max_iter", 20))

    import pysindy as ps

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=max_iter),
        feature_library=ps.PolynomialLibrary(degree=1, include_bias=True),
    )
    model.fit(Xs, t=dt)
    coef = np.asarray(model.coefficients(), dtype=np.float64)

    lib_names = ["bias"] + list(feature_names)
    names = [f"d({feature_names[i]})/dt__{lib_names[k]}"
             for i in range(coef.shape[0]) for k in range(coef.shape[1])]
    return coef.reshape(-1), names


REPRESENTATIONS: dict[str, Callable[..., tuple[np.ndarray, list[str]]]] = {
    "R0_kinematic_summary": kinematic_summary,
    "R1_sindy_coefficients": sindy_coefficients,
}

# The control every dynamical representation must beat for its result to mean anything.
BASELINE_REPRESENTATION = "R0_kinematic_summary"
