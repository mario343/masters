"""Timing-only benchmark for the KNN implementation variants.

KNN is not a separate scientific result in this project.  The benchmark keeps
the historical timing decomposition while leaving the canonical supervised
pipeline focused on the models used for the phase-transition analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .config import TransitionConfig, feature_sets
from .data import load_full, load_train
from .preprocessing import select_features
from .statistics import block_jackknife, find_first_crossing, population_variance


DEFAULT_ALGORITHMS = ("auto", "kd_tree", "ball_tree", "brute")
DEFAULT_NEIGHBORS = tuple(list(range(1, 52, 2)) + [61, 81, 101, 151, 201])


@dataclass(frozen=True)
class KNNTimingRow:
    transition: str
    feature_set: str
    n_features: int
    algorithm: str
    n_neighbors: int
    train_accuracy: float
    crossing: float | None
    runtime_model: float
    runtime_jackknife_mean: float
    runtime_jackknife_chi: float
    runtime_summary_mean: float
    runtime_summary_chi: float
    runtime_crossing: float
    runtime_total: float


def _timed(function, *args, **kwargs):
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started


def _jackknife_over_parameter(
    parameter: np.ndarray,
    probabilities: np.ndarray,
    statistic,
    block_counts: range,
) -> dict[float, list[tuple[float, float, float]]]:
    results: dict[float, list[tuple[float, float, float]]] = {}
    for value in np.unique(parameter):
        values = probabilities[parameter == value]
        if len(values) < 2:
            continue
        rows = []
        for block_count in block_counts:
            result = block_jackknife(values, statistic, block_count)
            if result is not None:
                rows.append(result)
        if rows:
            results[float(value)] = rows
    return results


def _summarise_jackknife(
    results: dict[float, list[tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameters = []
    estimates = []
    errors = []
    for value in sorted(results):
        row = max(results[value], key=lambda item: item[2])
        parameters.append(value)
        estimates.append(row[1])
        errors.append(row[2])
    return (
        np.asarray(parameters, dtype=float),
        np.asarray(estimates, dtype=float),
        np.asarray(errors, dtype=float),
    )


def _fit_and_predict(model, X_train, y_train, X_full):
    model.fit(X_train, y_train)
    train_accuracy = float(model.score(X_train, y_train))
    probabilities = np.asarray(model.predict_proba(X_full), dtype=float)[:, 1]
    return train_accuracy, probabilities


def _crossing(parameter: np.ndarray, mean: np.ndarray) -> float | None:
    result = find_first_crossing(parameter, mean)
    return None if result is None else float(result[0])


def benchmark_knn(
    config: TransitionConfig,
    *,
    feature_set_names: tuple[str, ...] = ("30", "20", "12"),
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS,
    neighbors: tuple[int, ...] = DEFAULT_NEIGHBORS,
    train_stride: int | None = None,
    full_stride: int | None = None,
    block_counts: range = range(2, 101),
) -> list[dict[str, object]]:
    """Benchmark historical KNN variants without producing scientific curves."""

    train = load_train(config, stride=train_stride)
    full = load_full(config, stride=full_stride)
    if train.y is None:
        raise ValueError("KNN benchmark requires labelled endpoint data")

    rows: list[dict[str, object]] = []
    sets = feature_sets()
    for feature_set_name in feature_set_names:
        if feature_set_name not in sets:
            raise KeyError(f"unknown feature set: {feature_set_name}")
        columns = sets[feature_set_name]
        X_train = select_features(train.X, columns)
        X_full = select_features(full.X, columns)

        for algorithm in algorithms:
            for n_neighbors in neighbors:
                estimator = Pipeline([
                    (
                        "classifier",
                        KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            algorithm=algorithm,
                            n_jobs=-1,
                        ),
                    )
                ])
                (train_accuracy, probabilities), runtime_model = _timed(
                    _fit_and_predict,
                    estimator,
                    X_train,
                    train.y,
                    X_full,
                )

                mean_results, runtime_jackknife_mean = _timed(
                    _jackknife_over_parameter,
                    full.parameter,
                    probabilities,
                    np.mean,
                    block_counts,
                )
                chi_results, runtime_jackknife_chi = _timed(
                    _jackknife_over_parameter,
                    full.parameter,
                    probabilities,
                    population_variance,
                    block_counts,
                )
                (parameters, mean, mean_error), runtime_summary_mean = _timed(
                    _summarise_jackknife,
                    mean_results,
                )
                (_, _, _), runtime_summary_chi = _timed(
                    _summarise_jackknife,
                    chi_results,
                )
                crossing, runtime_crossing = _timed(
                    _crossing,
                    parameters,
                    mean,
                )
                runtime_total = (
                    runtime_model
                    + runtime_jackknife_mean
                    + runtime_jackknife_chi
                    + runtime_summary_mean
                    + runtime_summary_chi
                    + runtime_crossing
                )
                rows.append(asdict(KNNTimingRow(
                    transition=config.name,
                    feature_set=feature_set_name,
                    n_features=len(columns),
                    algorithm=algorithm,
                    n_neighbors=n_neighbors,
                    train_accuracy=train_accuracy,
                    crossing=crossing,
                    runtime_model=runtime_model,
                    runtime_jackknife_mean=runtime_jackknife_mean,
                    runtime_jackknife_chi=runtime_jackknife_chi,
                    runtime_summary_mean=runtime_summary_mean,
                    runtime_summary_chi=runtime_summary_chi,
                    runtime_crossing=runtime_crossing,
                    runtime_total=runtime_total,
                )))
    return rows
