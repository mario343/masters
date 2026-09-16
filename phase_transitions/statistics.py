"""Statistical primitives shared by supervised and unsupervised analyses.

The susceptibility definition follows the accompanying article:

    chi(O) = <O**2> - <O>**2

This is the population variance of the analysed Monte Carlo ensemble
(`ddof=0`).  The `ddof=1` correction is intentionally retained only for the
variance of jackknife replicates, where it is part of the standard
single-elimination jackknife standard-error estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


def _as_1d_float(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite numbers")
    return array


def population_variance(values: ArrayLike) -> float:
    """Return ``<O**2> - <O>**2`` for one analysed ensemble.

    This is equivalent to ``np.var(values, ddof=0)`` and is the susceptibility
    definition used in the reference article.
    """

    values_array = _as_1d_float(values)
    mean = np.mean(values_array)
    return float(np.mean(values_array * values_array) - mean * mean)


def weighted_mean(values: ArrayLike, weights: ArrayLike) -> float:
    """Return the normalized weighted mean of an observable."""

    values_array = _as_1d_float(values)
    weights_array = _as_1d_float(weights)
    if values_array.shape != weights_array.shape:
        raise ValueError("values and weights must have the same shape")
    if np.any(weights_array < 0):
        raise ValueError("weights must be non-negative")
    weight_sum = np.sum(weights_array)
    if weight_sum <= 0:
        raise ValueError("weights must have a positive sum")
    return float(np.sum(weights_array * values_array) / weight_sum)


def weighted_population_variance(
    values: ArrayLike,
    weights: ArrayLike,
) -> float:
    """Return the reweighted article susceptibility.

    For normalized weights this is
    ``sum(w * O**2) - sum(w * O)**2``.  No Bessel correction is applied.
    """

    values_array = _as_1d_float(values)
    weights_array = _as_1d_float(weights)
    if values_array.shape != weights_array.shape:
        raise ValueError("values and weights must have the same shape")
    if np.any(weights_array < 0):
        raise ValueError("weights must be non-negative")
    weight_sum = np.sum(weights_array)
    if weight_sum <= 0:
        raise ValueError("weights must have a positive sum")
    normalized_weights = weights_array / weight_sum
    mean = np.sum(normalized_weights * values_array)
    return float(
        np.sum(normalized_weights * values_array * values_array) - mean * mean
    )


def block_jackknife(
    data: ArrayLike,
    stat_func: Callable[[np.ndarray], float],
    block_count: int,
) -> tuple[float, float, float] | None:
    """Compute one blocked single-elimination jackknife estimate.

    The implementation preserves the existing notebook convention:

    - truncate to ``m * block_count`` samples,
    - use contiguous blocks,
    - apply the existing bias correction,
    - estimate replicate variance with ``ddof=1``.

    Returns ``(theta_hat, theta_jack, standard_error)`` or ``None`` when a
    requested block count is larger than the number of samples.
    """

    values = _as_1d_float(data)
    if not isinstance(block_count, (int, np.integer)) or block_count < 1:
        raise ValueError("block_count must be a positive integer")

    block_size = len(values) // block_count
    if block_size < 1:
        return None

    truncated = values[: block_size * block_count]
    blocks = truncated.reshape(block_count, block_size)
    theta_hat = float(stat_func(truncated))

    replicates = np.asarray(
        [
            stat_func(np.concatenate((blocks[:i], blocks[i + 1 :])).ravel())
            for i in range(block_count)
        ],
        dtype=float,
    )

    bias = (block_count - 1) * (np.mean(replicates) - theta_hat)
    theta_jack = theta_hat - bias

    if block_count == 1:
        # There is no leave-one-block-out sample for B=1.  This case is not
        # used by the project (the configured ranges start at B=2), but a
        # defined result is safer than silently returning NaN.
        return theta_hat, theta_jack, 0.0

    replicate_variance = np.var(replicates, ddof=1)
    variance = replicate_variance * (block_count - 1) ** 2 / block_count
    return theta_hat, theta_jack, float(np.sqrt(variance))


def summarise_jackknife(
    results_by_parameter: dict[float, Iterable[dict[str, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the maximum-SE replicate for every parameter value.

    This matches the current notebook summary rule.  Each value in
    ``results_by_parameter`` may be a list of dictionaries or a structured
    iterable containing ``theta_jack`` and ``se``.
    """

    parameters: list[float] = []
    estimates: list[float] = []
    errors: list[float] = []

    for parameter in sorted(results_by_parameter):
        rows = [row for row in results_by_parameter[parameter] if np.isfinite(row["se"])]
        if not rows:
            continue
        selected = max(rows, key=lambda row: row["se"])
        parameters.append(float(parameter))
        estimates.append(float(selected["theta_jack"]))
        errors.append(float(selected["se"]))

    return (
        np.asarray(parameters, dtype=float),
        np.asarray(estimates, dtype=float),
        np.asarray(errors, dtype=float),
    )


def find_first_crossing(
    parameter: ArrayLike,
    values: ArrayLike,
    level: float = 0.5,
) -> tuple[float, float, int] | None:
    """Find the first linear crossing of ``values`` with ``level``.

    Returns ``(crossing, nearest_grid_value, left_index)``.  The project
    assumes the physical curves are monotonic, so the first crossing is the
    intended one and no extra monotonicity filter is applied.
    """

    parameter_array = _as_1d_float(parameter)
    values_array = _as_1d_float(values)
    if parameter_array.shape != values_array.shape:
        raise ValueError("parameter and values must have the same shape")
    if len(parameter_array) < 2:
        raise ValueError("at least two points are required")

    nearest_index = int(np.argmin(np.abs(values_array - level)))
    nearest = float(parameter_array[nearest_index])
    crossings = np.flatnonzero(
        (values_array[:-1] - level) * (values_array[1:] - level) <= 0
    )
    if len(crossings) == 0:
        return None

    index = int(crossings[0])
    x0, x1 = parameter_array[index : index + 2]
    y0, y1 = values_array[index : index + 2]
    if y1 == y0:
        crossing = float(x0)
    else:
        crossing = float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return crossing, nearest, index


def find_peak(parameter: ArrayLike, values: ArrayLike) -> float:
    """Return the location of a peak using the existing parabolic refinement."""

    parameter_array = _as_1d_float(parameter)
    values_array = _as_1d_float(values)
    if parameter_array.shape != values_array.shape:
        raise ValueError("parameter and values must have the same shape")

    index = int(np.argmax(values_array))
    if index == 0 or index == len(values_array) - 1:
        return float(parameter_array[index])

    x = parameter_array[index - 1 : index + 2]
    y = values_array[index - 1 : index + 2]
    try:
        quadratic, linear, _ = np.polyfit(x, y, 2)
    except (np.linalg.LinAlgError, ValueError):
        return float(parameter_array[index])
    if quadratic >= 0:
        return float(parameter_array[index])

    peak = -linear / (2 * quadratic)
    if x[0] <= peak <= x[-1]:
        return float(peak)
    return float(parameter_array[index])
