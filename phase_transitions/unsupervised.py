"""Endpoint-trained unsupervised phase identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import TransitionConfig, feature_sets
from .data import load_full, load_train
from .models import build_clusterer
from .preprocessing import fit_transform_train, select_features
from .statistics import find_first_crossing, find_peak
from .supervised import Curve, _crossing_error, _curve_for_parameter
from .validation import block_validation_split


@dataclass(frozen=True)
class UnsupervisedResult:
    transition: str
    model: str
    feature_set: str
    standardize: bool
    method: str
    validation_accuracy: float
    validation_passed: bool
    training_size: int
    validation_size: int
    full_size: int
    birch_threshold: float | None
    curve: Curve
    crossing: float | None
    crossing_error: float | None
    crossing_nearest: float | None
    chi_peak: float


class EndpointValidationError(RuntimeError):
    """Raised when an endpoint model is not eligible for curve evaluation."""

    def __init__(self, *, transition: str, model: str, feature_set: str,
                 standardize: bool, accuracy: float, threshold: float):
        self.transition = transition
        self.model = model
        self.feature_set = feature_set
        self.standardize = standardize
        self.accuracy = accuracy
        self.threshold = threshold
        super().__init__(
            f"endpoint validation failed for {transition} {model} "
            f"features={feature_set} standardize={standardize}: "
            f"accuracy={accuracy:.6f} < threshold={threshold:.6f}"
        )


def median_nearest_neighbor_distance(X: np.ndarray) -> float:
    """Return the median distance to the nearest distinct sample."""

    if len(X) < 2:
        raise ValueError("at least two samples are required")
    neighbours = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X)
    distances, _ = neighbours.kneighbors(X)
    return float(np.median(distances[:, 1]))


def _phase_cluster_mapping(
    labels: np.ndarray,
    parameter: np.ndarray,
) -> tuple[int, int]:
    labels = np.asarray(labels)
    parameter = np.asarray(parameter, dtype=float)
    valid = labels != -1
    unique = np.unique(labels[valid])
    if len(unique) < 2:
        raise ValueError(f"expected at least two valid clusters, got {unique}")
    means = {int(label): np.mean(parameter[valid][labels[valid] == label]) for label in unique}
    low = min(means, key=means.get)
    high = max(means, key=means.get)
    return int(low), int(high)


def _predict_labels(model, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict"):
        raise TypeError(f"{type(model).__name__} does not support prediction")
    return np.asarray(model.predict(X))


def _probability_of_high_phase(model, X: np.ndarray, high_cluster: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        labels = np.asarray(model.predict(X))
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        # Gaussian mixture component indices are the columns of predict_proba;
        # for hard clusterers this branch is not used.
        if hasattr(model, "weights_"):
            return probabilities[:, high_cluster]
        return (labels == high_cluster).astype(float)
    return (_predict_labels(model, X) == high_cluster).astype(float)


def endpoint_validation_accuracy(
    config: TransitionConfig,
    *,
    model_name: str,
    feature_set_name: str = "30",
    standardize: bool = False,
    train_stride: int | None = None,
    validation_fraction: float | None = None,
) -> tuple[float, bool, float | None, int, int]:
    """Validate endpoint clustering without evaluating the full transition curve.

    This is intentionally separate from :func:`run_endpoint_clustering` so a
    complete validation matrix does not spend time on jackknife curves for
    configurations that already fail the endpoint criterion.
    """

    train_data = load_train(
        config,
        stride=(
            config.unsupervised_train_stride
            if train_stride is None
            else train_stride
        ),
    )
    columns = feature_sets().get(feature_set_name)
    if columns is None:
        raise KeyError(f"unknown feature set: {feature_set_name}")
    if train_data.y is None:
        raise ValueError("endpoint validation requires endpoint labels")

    split = block_validation_split(
        train_data.y,
        validation_fraction=(
            config.validation_fraction
            if validation_fraction is None
            else validation_fraction
        ),
    )
    X = select_features(train_data.X, columns)
    X_fit, _, scaler = fit_transform_train(
        X[split.train_indices],
        X[split.train_indices],
        standardize=standardize,
    )
    X_validation = X[split.validation_indices]
    if scaler is not None:
        X_validation = scaler.transform(X_validation)

    normalized_name = model_name.lower().replace(" ", "_")
    birch_threshold = None
    if normalized_name == "birch":
        birch_threshold = median_nearest_neighbor_distance(X_fit)
        model = build_clusterer(model_name)
        model.set_params(threshold=birch_threshold)
    else:
        model = build_clusterer(model_name)

    if normalized_name == "meanshift":
        X_model_fit = X_fit[:: config.meanshift_fit_stride]
    else:
        X_model_fit = X_fit
    model.fit(X_model_fit)
    train_labels = _predict_labels(model, X_fit)
    _, high_cluster = _phase_cluster_mapping(
        train_labels,
        train_data.parameter[split.train_indices],
    )
    validation_labels = _predict_labels(model, X_validation)
    validation_expected = np.isclose(
        train_data.parameter[split.validation_indices],
        config.phase_high_value,
    )
    accuracy = float(
        np.mean((validation_labels == high_cluster) == validation_expected)
    )
    return (
        accuracy,
        accuracy >= config.validation_threshold,
        birch_threshold,
        len(split.train_indices),
        len(split.validation_indices),
    )


def run_endpoint_clustering(
    config: TransitionConfig,
    *,
    model_name: str,
    feature_set_name: str = "30",
    standardize: bool = False,
    train_stride: int | None = None,
    full_stride: int | None = None,
    validation_fraction: float | None = None,
    block_counts: range = range(2, 101),
    require_validation: bool = True,
) -> UnsupervisedResult:
    """Fit a clusterer on endpoint training blocks and validate on endpoints."""

    train_data = load_train(
        config,
        stride=(
            config.unsupervised_train_stride
            if train_stride is None
            else train_stride
        ),
    )
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
    X = select_features(train_data.X, columns)

    X_fit, _, scaler = fit_transform_train(
        X[split.train_indices],
        X[split.train_indices],
        standardize=standardize,
    )
    X_validation = X[split.validation_indices]
    if scaler is not None:
        X_validation = scaler.transform(X_validation)

    normalized_name = model_name.lower().replace(" ", "_")
    birch_threshold = None
    if normalized_name == "birch":
        birch_threshold = median_nearest_neighbor_distance(X_fit)
        model = build_clusterer(model_name)
        model.set_params(threshold=birch_threshold)
    else:
        model = build_clusterer(model_name)

    if normalized_name == "meanshift":
        X_model_fit = X_fit[:: config.meanshift_fit_stride]
    else:
        X_model_fit = X_fit
    model.fit(X_model_fit)

    train_labels = _predict_labels(model, X_fit)
    low_cluster, high_cluster = _phase_cluster_mapping(
        train_labels,
        train_data.parameter[split.train_indices],
    )
    validation_labels = _predict_labels(model, X_validation)
    validation_expected = np.isclose(
        train_data.parameter[split.validation_indices],
        config.phase_high_value,
    )
    validation_accuracy = float(
        np.mean((validation_labels == high_cluster) == validation_expected)
    )
    validation_passed = validation_accuracy >= config.validation_threshold

    if require_validation and not validation_passed:
        raise EndpointValidationError(
            transition=config.name,
            model=model_name,
            feature_set=feature_set_name,
            standardize=standardize,
            accuracy=validation_accuracy,
            threshold=config.validation_threshold,
        )

    full_data = load_full(
        config,
        stride=(
            config.unsupervised_full_stride
            if full_stride is None
            else full_stride
        ),
    )
    X_full = select_features(full_data.X, columns)
    if scaler is None:
        X_full_transformed = X_full
    else:
        X_full_transformed = scaler.transform(X_full)

    probabilities = _probability_of_high_phase(
        model,
        X_full_transformed,
        high_cluster,
    )
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

    return UnsupervisedResult(
        transition=config.name,
        model=model_name,
        feature_set=feature_set_name,
        standardize=standardize,
        method="endpoint_validation",
        validation_accuracy=validation_accuracy,
        validation_passed=validation_passed,
        training_size=len(split.train_indices),
        validation_size=len(split.validation_indices),
        full_size=len(full_data.X),
        birch_threshold=birch_threshold,
        curve=curve,
        crossing=crossing,
        crossing_error=crossing_error,
        crossing_nearest=crossing_nearest,
        chi_peak=find_peak(curve.parameter, curve.chi),
    )


def run_full_range_clustering(
    config: TransitionConfig,
    *,
    model_name: str,
    feature_set_name: str = "30",
    standardize: bool = False,
    full_stride: int | None = None,
    block_counts: range = range(2, 101),
) -> UnsupervisedResult:
    """Fit a clusterer on the complete transition trajectory.

    This is the separate full-range method.  The control parameter is used
    only after fitting, to map clusters to the low/high phases; it is not
    supplied to the clusterer during ``fit``.
    """

    full_data = load_full(
        config,
        stride=(
            config.unsupervised_full_stride
            if full_stride is None
            else full_stride
        ),
    )
    columns = feature_sets().get(feature_set_name)
    if columns is None:
        raise KeyError(f"unknown feature set: {feature_set_name}")
    X = select_features(full_data.X, columns)
    if standardize:
        X_fit, X_transformed, _ = fit_transform_train(X, X, standardize=True)
    else:
        X_fit = X_transformed = X

    normalized_name = model_name.lower().replace(" ", "_")
    birch_threshold = None
    if normalized_name == "birch":
        birch_threshold = median_nearest_neighbor_distance(X_fit)
        model = build_clusterer(model_name)
        model.set_params(threshold=birch_threshold)
    else:
        model = build_clusterer(model_name)

    if normalized_name == "meanshift":
        X_model_fit = X_fit[:: config.meanshift_fit_stride]
    else:
        X_model_fit = X_fit
    model.fit(X_model_fit)
    labels = _predict_labels(model, X_fit)
    low_cluster, high_cluster = _phase_cluster_mapping(labels, full_data.parameter)
    probabilities = _probability_of_high_phase(model, X_transformed, high_cluster)
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

    return UnsupervisedResult(
        transition=config.name,
        model=model_name,
        feature_set=feature_set_name,
        standardize=standardize,
        method="full_range",
        validation_accuracy=float("nan"),
        validation_passed=None,
        training_size=len(full_data.X),
        validation_size=0,
        full_size=len(full_data.X),
        birch_threshold=birch_threshold,
        curve=curve,
        crossing=crossing,
        crossing_error=crossing_error,
        crossing_nearest=crossing_nearest,
        chi_peak=find_peak(curve.parameter, curve.chi),
    )
