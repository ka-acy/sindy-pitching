"""Shared helpers for scaled SINDy training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_feature_scaler(trajectories: list[np.ndarray]) -> dict[str, list[float]]:
    stacked = np.vstack(trajectories)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def transform_trajectories(
    trajectories: list[np.ndarray],
    scaler: dict[str, list[float]],
) -> list[np.ndarray]:
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return [(traj - mean) / scale for traj in trajectories]


def transform_derivatives(
    derivatives: list[np.ndarray],
    scaler: dict[str, list[float]],
) -> list[np.ndarray]:
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return [der / scale for der in derivatives]


def inverse_transform_trajectory(
    trajectory: np.ndarray,
    scaler: dict[str, list[float]],
) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return trajectory * scale + mean


def inverse_transform_derivative(
    derivative: np.ndarray,
    scaler: dict[str, list[float]],
) -> np.ndarray:
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return derivative * scale


def scaler_summary(scaler: dict[str, list[float]], feature_names: list[str]) -> list[dict[str, Any]]:
    mean = scaler["mean"]
    scale = scaler["scale"]
    return [
        {
            "feature_name": name,
            "mean": float(mu),
            "scale": float(sig),
        }
        for name, mu, sig in zip(feature_names, mean, scale, strict=True)
    ]
