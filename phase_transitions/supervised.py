"""Canonical supervised endpoint-training experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import TransitionConfig, feature_sets
from .data import load_full, load_train
from .models import build_classifier
from .preprocessing import select_features
from .statistics import (
    block_jackknife,
    find_first_crossing,
    find_peak,
    population_variance,
)
from .validation import block_validation_split


@dataclass(frozen=True)
class Curve:
    parameter: np.ndarray
    mean: np.ndarray
    mean_error: np.ndarray
    chi: np.ndarray
    chi_error: np.ndarray


@dataclass(frozen=True)
class SupervisedResult:
    transition: str
    model: str
    feature_set: str
    standardize: bool
    method: str
    calibrated: bool
    calibration_cv: int
    validation_accuracy: float
    validation_passed: bool
    training_size: int
    validation_size: int
    full_size: int
    curve: Curve
    crossing: float | None
    crossing_error: float | None
    crossing_nearest: float | None
    chi_peak: float


def _probability_of_class_one(model: Any, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "canonical supervised pipeline requires predict_proba; "
            "use calibrated=True for SVM and MLP"
        )
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    class_one = np.flatnonzero(classes == 1)
    if len(class_one) != 1:
        raise ValueError(f"expected a binary class labelled 1, got {classes}")
    return probabilities[:, class_one[0]]


def _curve_for_parameter(
    parameter: np.ndarray,
    probabilities: np.ndarray,
    *,
    block_counts: range,
) -> Curve:
    means: list[float] = []
    mean_errors: list[float] = []
    chis: list[float] = []
    chi_errors: list[float] = []
    parameters: list[float] = []

    for value in np.unique(parameter):
        values = probabilities[np.isclose(parameter, value)]
        if len(values) < 2:
            continue

        mean_results = []
        chi_results = []
        for block_count in block_counts:
            mean_out = block_jackknife(values, np.mean, block_count)
            chi_out = block_jackknife(values, population_variance, block_count)
            if mean_out is not None:
                mean_results.append(
                    {"theta_jack": mean_out[1], "se": mean_out[2]}
                )
            if chi_out is not None:
                chi_results.append(
                    {"theta_jack": chi_out[1], "se": chi_out[2]}
                )
        if not mean_results or not chi_results:
            continue

        mean_best = max(mean_results, key=lambda row: row["se"])
        chi_best = max(chi_results, key=lambda row: row["se"])
        parameters.append(float(value))
        means.append(float(mean_best["theta_jack"]))
        mean_errors.append(float(mean_best["se"]))
        chis.append(float(chi_best["theta_jack"]))
        chi_errors.append(float(chi_best["se"]))

    if not parameters:
        raise ValueError("no parameter group had enough samples for jackknife")

    order = np.argsort(parameters)
    return Curve(
        parameter=np.asarray(parameters, dtype=float)[order],
        mean=np.asarray(means, dtype=float)[order],
        mean_error=np.asarray(mean_errors, dtype=float)[order],
        chi=np.asarray(chis, dtype=float)[order],
        chi_error=np.asarray(chi_errors, dtype=float)[order],
    )


def _crossing_error(curve: Curve, crossing: float, left_index: int) -> float:
    x0, x1 = curve.parameter[left_index : left_index + 2]
    y0, y1 = curve.mean[left_index : left_index + 2]
    slope = (y1 - y0) / (x1 - x0)
    if abs(slope) < 1e-12:
        return float("nan")
    sigma = np.interp(
        crossing,
        [x0, x1],
        curve.mean_error[left_index : left_index + 2],
    )
    return float(sigma / abs(slope))


def run_endpoint_experiment(
    config: TransitionConfig,
    *,
    model_name: str,
    feature_set_name: str = "30",
    standardize: bool = False,
    train_stride: int | None = None,
    full_stride: int | None = None,
    validation_fraction: float | None = None,
    calibrated: bool | None = None,
    block_counts: range = range(2, 101),
) -> SupervisedResult:
    """Train on endpoint training data, validate, then analyse full curves."""

    train_data = load_train(config, stride=train_stride or config.train_stride)
    full_data = load_full(config, stride=full_stride or config.full_stride)
    if train_data.y is None:
        raise ValueError("endpoint training data must contain labels")

    columns = feature_sets().get(feature_set_name)
    if columns is None:
        raise KeyError(f"unknown feature set: {feature_set_name}")

    split = block_validation_split(
        train_data.y,
        validation_fraction=(
            config.validation_fraction
            if validation_fraction is None
            else validation_fraction
        ),
    )
    X_train = select_features(train_data.X, columns)
    X_full = select_features(full_data.X, columns)
    if calibrated is None:
        calibrated = model_name.lower().replace(" ", "_") in {"svm_rbf", "mlp"}

    model = build_classifier(
        model_name,
        calibrated=calibrated,
        standardize=standardize,
        calibration_cv=config.calibration_cv,
    )
    model.fit(X_train[split.train_indices], train_data.y[split.train_indices])

    validation_probability = _probability_of_class_one(
        model,
        X_train[split.validation_indices],
    )
    validation_predictions = validation_probability >= 0.5
    validation_accuracy = float(
        np.mean(validation_predictions == train_data.y[split.validation_indices])
    )
    validation_passed = validation_accuracy >= config.validation_threshold

    probabilities = _probability_of_class_one(model, X_full)
    curve = _curve_for_parameter(
        full_data.parameter,
        probabilities,
        block_counts=block_counts,
    )
    crossing_data = find_first_crossing(curve.parameter, curve.mean)
    if crossing_data is None:
        crossing = crossing_error = crossing_nearest = None
    else:
        crossing, crossing_nearest, left_index = crossing_data
        crossing_error = _crossing_error(curve, crossing, left_index)

    return SupervisedResult(
        transition=config.name,
        model=model_name,
        feature_set=feature_set_name,
        standardize=standardize,
        method="endpoint_validation",
        calibrated=calibrated,
        calibration_cv=config.calibration_cv,
        validation_accuracy=validation_accuracy,
        validation_passed=validation_passed,
        training_size=len(split.train_indices),
        validation_size=len(split.validation_indices),
        full_size=len(full_data.X),
        curve=curve,
        crossing=crossing,
        crossing_error=crossing_error,
        crossing_nearest=crossing_nearest,
        chi_peak=find_peak(curve.parameter, curve.chi),
    )
