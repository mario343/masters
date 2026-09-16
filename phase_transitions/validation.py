"""Validation splits for endpoint training data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray


def endpoint_indices(
    parameter: np.ndarray,
    *,
    low_value: float,
    high_value: float,
) -> np.ndarray:
    mask = np.isclose(parameter, low_value) | np.isclose(parameter, high_value)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("no endpoint samples found")
    return indices


def block_validation_split(
    labels: np.ndarray,
    *,
    validation_fraction: float = 0.2,
) -> ValidationSplit:
    """Split each phase into contiguous train/validation blocks.

    The processed endpoint datasets are ordered in Monte Carlo blocks.  A
    deterministic contiguous split avoids the optimistic leakage of randomly
    interleaving correlated neighbouring samples.
    """

    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    train_parts = []
    validation_parts = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        if validation_count >= len(indices):
            raise ValueError("validation split leaves no training samples for a class")
        train_parts.append(indices[:-validation_count])
        validation_parts.append(indices[-validation_count:])

    train_indices = np.sort(np.concatenate(train_parts))
    validation_indices = np.sort(np.concatenate(validation_parts))
    if np.intersect1d(train_indices, validation_indices).size:
        raise AssertionError("train and validation indices overlap")
    return ValidationSplit(train_indices, validation_indices)
