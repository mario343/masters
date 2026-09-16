"""Histogram-reweighting statistics with the article susceptibility definition."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .statistics import weighted_mean, weighted_population_variance


def reweighted_observables(
    values: Iterable[float],
    weights: Iterable[float],
) -> dict[str, float]:
    """Return weighted mean, population susceptibility and ESS."""

    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    if values_array.ndim != 1 or weights_array.ndim != 1:
        raise ValueError("values and weights must be one-dimensional")
    if values_array.shape != weights_array.shape:
        raise ValueError("values and weights must have the same shape")
    mean = weighted_mean(values_array, weights_array)
    chi = weighted_population_variance(values_array, weights_array)
    weight_sum = np.sum(weights_array)
    ess = float(weight_sum**2 / np.sum(weights_array**2))
    return {"mean": mean, "chi": chi, "ess": ess}


def _from_sums(sum_w, sum_wp, sum_wp2):
    mean = sum_wp / sum_w
    chi = sum_wp2 / sum_w - mean**2
    return float(mean), float(chi)


def weighted_block_jackknife(
    values: Iterable[float],
    weights: Iterable[float],
    block_count: int,
) -> dict[str, tuple[float, float, float]] | None:
    """Return weighted mean/chi jackknife estimates for one block count.

    Each tuple is ``(theta_hat, theta_jack, standard_error)``.  The
    susceptibility is the weighted population variance from the article.  The
    variance of leave-one-block-out replicates uses ``ddof=1`` as required by
    the jackknife estimator.
    """

    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    if values_array.ndim != 1 or weights_array.ndim != 1:
        raise ValueError("values and weights must be one-dimensional")
    if values_array.shape != weights_array.shape:
        raise ValueError("values and weights must have the same shape")
    if np.any(weights_array < 0) or np.sum(weights_array) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    if not isinstance(block_count, (int, np.integer)) or block_count < 1:
        raise ValueError("block_count must be a positive integer")

    block_size = len(values_array) // block_count
    if block_size < 1:
        return None
    used = block_size * block_count
    values_array = values_array[:used]
    weights_array = weights_array[:used]
    values_blocks = values_array.reshape(block_count, block_size)
    weights_blocks = weights_array.reshape(block_count, block_size)

    total_w = np.sum(weights_array)
    total_wp = np.sum(weights_array * values_array)
    total_wp2 = np.sum(weights_array * values_array * values_array)
    theta_hat = _from_sums(total_w, total_wp, total_wp2)

    replicates = np.asarray(
        [
            _from_sums(
                total_w - np.sum(weights_blocks[i]),
                total_wp - np.sum(weights_blocks[i] * values_blocks[i]),
                total_wp2
                - np.sum(weights_blocks[i] * values_blocks[i] * values_blocks[i]),
            )
            for i in range(block_count)
        ],
        dtype=float,
    )

    result = {}
    for index, name in enumerate(("mean", "chi")):
        replicate_values = replicates[:, index]
        bias = (block_count - 1) * (
            np.mean(replicate_values) - theta_hat[index]
        )
        theta_jack = theta_hat[index] - bias
        if block_count == 1:
            standard_error = 0.0
        else:
            standard_error = float(
                np.sqrt(
                    np.var(replicate_values, ddof=1)
                    * (block_count - 1) ** 2
                    / block_count
                )
            )
        result[name] = (
            float(theta_hat[index]),
            float(theta_jack),
            standard_error,
        )
    return result
