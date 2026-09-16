"""Validated loading of processed transition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TransitionConfig


@dataclass(frozen=True)
class ProcessedDataset:
    X: np.ndarray
    y: np.ndarray | None
    parameter: np.ndarray


def _load(path: Path, parameter_name: str, *, require_labels: bool) -> ProcessedDataset:
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as archive:
        required = {"X", parameter_name}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")

        X = np.asarray(archive["X"])
        parameter = np.asarray(archive[parameter_name], dtype=float)
        # Full-range archives may contain a legacy object/pickle ``y`` entry,
        # but the full-range pipeline intentionally does not use labels. Avoid
        # deserialising it unless the caller explicitly requires labels.
        y = np.asarray(archive["y"]) if require_labels and "y" in archive.files else None

    if X.ndim != 2 or X.shape[1] != 30:
        raise ValueError(f"{path}: expected X with shape (n, 30), got {X.shape}")
    if parameter.ndim != 1 or len(parameter) != len(X):
        raise ValueError(f"{path}: parameter and X have incompatible lengths")
    if require_labels and y is None:
        raise ValueError(f"{path}: labelled data are required")
    if y is not None and len(y) != len(X):
        raise ValueError(f"{path}: y and X have incompatible lengths")

    return ProcessedDataset(X=X, y=y, parameter=parameter)


def load_train(config: TransitionConfig, *, stride: int | None = None) -> ProcessedDataset:
    data = _load(config.train_path, config.parameter_name, require_labels=True)
    return _slice(data, stride)


def load_full(config: TransitionConfig, *, stride: int | None = None) -> ProcessedDataset:
    data = _load(config.full_path, config.parameter_name, require_labels=False)
    return _slice(data, stride)


def _slice(data: ProcessedDataset, stride: int | None) -> ProcessedDataset:
    if stride is None:
        return data
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    selection = slice(None, None, stride)
    return ProcessedDataset(
        X=data.X[selection],
        y=None if data.y is None else data.y[selection],
        parameter=data.parameter[selection],
    )
