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


def weakform_coefficients(
    X: np.ndarray, dt: float, feature_names: list[str], cfg: dict | None = None
) -> tuple[np.ndarray, list[str]]:
    """R2 -- per-clip weak-form SINDy coefficients, flattened.

    The weak formulation integrates against test functions instead of differentiating
    pointwise, so it is markedly less sensitive to the pose noise these trajectories carry.
    Test-function subdomains are placed by the global RNG with no seed parameter, so the draw
    is seeded locally and the prior state restored: without that a representation would differ
    between runs and the study would not be reproducible.

    A single clip is short, so the fit can fail outright. On failure the row is zeros, which is
    a truthful "no information from this clip" rather than a silent exception.
    """
    cfg = cfg or {}
    n = len(feature_names)
    names = [f"weak_d({feature_names[i]})/dt__t{k}" for i in range(n) for k in range(n + 1)]
    try:
        import pysindy as ps

        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        Xs = (X - mu) / sigma
        state = np.random.get_state()
        try:
            np.random.seed(int(cfg.get("seed", 7)))
            lib = ps.WeakPDELibrary(
                library_functions=[lambda x: x],
                function_names=[lambda s: s],
                spatiotemporal_grid=np.arange(Xs.shape[0]) * dt,
                is_uniform=True,
                K=min(int(cfg.get("K", 50)), max(Xs.shape[0] - 2, 2)),
            )
            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=float(cfg.get("stlsq_threshold", 0.05)), alpha=0.001),
                feature_library=lib,
            )
            model.fit(Xs, t=dt)
        finally:
            np.random.set_state(state)
        coef = np.asarray(model.coefficients(), dtype=np.float64).reshape(-1)
        if coef.size != len(names):
            out = np.zeros(len(names))
            out[: min(coef.size, len(names))] = coef[: min(coef.size, len(names))]
            return out, names
        return coef, names
    except Exception:
        return np.zeros(len(names)), names


def dmd_spectrum(
    X: np.ndarray, dt: float, feature_names: list[str], cfg: dict | None = None
) -> tuple[np.ndarray, list[str]]:
    """R3 -- exact DMD: eigenvalues of the best-fit linear operator, plus per-mode energy.

    Fits A minimising ||X[1:] - A X[:-1]|| by least squares and returns its spectrum. Unlike
    R1 this describes the clip's dynamics by its modes rather than by library coefficients, so
    it is invariant to which basis the library happened to use. Eigenvalues are sorted by
    magnitude for a stable ordering across clips, and returned as (real, imaginary, energy).
    """
    n = len(feature_names)
    names = [f"dmd_{part}_{i}" for i in range(n) for part in ("real", "imag", "energy")]
    if X.shape[0] < n + 2:
        return np.zeros(3 * n), names
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma
    A, *_ = np.linalg.lstsq(Xs[:-1], Xs[1:], rcond=None)
    vals, vecs = np.linalg.eig(A.T)
    order = np.argsort(-np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    energy = np.linalg.norm(vecs, axis=0)
    out = np.empty(3 * n)
    out[0::3], out[1::3], out[2::3] = vals.real, vals.imag, energy
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), names


def hankel_dmd_spectrum(
    X: np.ndarray, dt: float, feature_names: list[str], cfg: dict | None = None
) -> tuple[np.ndarray, list[str]]:
    """R4 -- time-delay (Hankel) DMD, HAVOK-style: DMD on a delay-embedded state.

    A single-step linear operator cannot represent dynamics that depend on history, and
    Section 6.8 of the paper shows history is exactly what these trajectories need. Delay
    embedding restores it (Takens), so this is R3's construction on an augmented state, kept to
    the top r modes for a fixed length.
    """
    cfg = cfg or {}
    depth = int(cfg.get("depth", 4))
    r = int(cfg.get("modes", 10))
    names = [f"hankel_{part}_{i}" for i in range(r) for part in ("real", "imag", "energy")]
    span = depth - 1
    if X.shape[0] <= span + 3:
        return np.zeros(3 * r), names
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma
    blocks = [Xs[span - k: Xs.shape[0] - k] for k in range(depth)]
    H = np.hstack(blocks)
    if H.shape[0] < H.shape[1] + 2:
        return np.zeros(3 * r), names
    A, *_ = np.linalg.lstsq(H[:-1], H[1:], rcond=None)
    vals, vecs = np.linalg.eig(A.T)
    order = np.argsort(-np.abs(vals))
    vals, vecs = vals[order][:r], vecs[:, order][:, :r]
    energy = np.linalg.norm(vecs, axis=0)
    out = np.zeros(3 * r)
    k = min(r, vals.size)
    out[0:3 * k:3], out[1:3 * k:3], out[2:3 * k:3] = vals[:k].real, vals[:k].imag, energy[:k]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), names


REPRESENTATIONS: dict[str, Callable[..., tuple[np.ndarray, list[str]]]] = {
    "R0_kinematic_summary": kinematic_summary,
    "R1_sindy_coefficients": sindy_coefficients,
    "R2_weakform_coefficients": weakform_coefficients,
    "R3_dmd_spectrum": dmd_spectrum,
    "R4_hankel_dmd_spectrum": hankel_dmd_spectrum,
}

# The control every dynamical representation must beat for its result to mean anything.
BASELINE_REPRESENTATION = "R0_kinematic_summary"
