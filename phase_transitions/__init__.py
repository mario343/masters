"""Shared statistical primitives for the phase-transition analyses."""

from .statistics import (
    block_jackknife,
    find_peak,
    find_first_crossing,
    population_variance,
    summarise_jackknife,
    weighted_mean,
    weighted_population_variance,
)

__all__ = [
    "block_jackknife",
    "find_peak",
    "find_first_crossing",
    "population_variance",
    "summarise_jackknife",
    "weighted_mean",
    "weighted_population_variance",
]
