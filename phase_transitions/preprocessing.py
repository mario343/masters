"""Feature selection and train-only preprocessing."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def select_features(X: np.ndarray, columns: Iterable[int]) -> np.ndarray:
    columns_array = np.asarray(list(columns), dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if len(columns_array) == 0:
        raise ValueError("at least one feature is required")
    if np.any(columns_array < 0) or np.any(columns_array >= X.shape[1]):
        raise IndexError("feature index outside X")
    return X[:, columns_array]


def make_preprocessor(*, standardize: bool) -> StandardScaler | None:
    """Return the scaler to be fit on the model-training data only."""

    return StandardScaler() if standardize else None


def fit_transform_train(
    X_train: np.ndarray,
    X_other: np.ndarray,
    *,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    scaler = make_preprocessor(standardize=standardize)
    if scaler is None:
        return X_train, X_other, None
    return scaler.fit_transform(X_train), scaler.transform(X_other), scaler
