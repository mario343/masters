"""Stable CSV serialization for canonical experiment results."""

from __future__ import annotations

import csv
from pathlib import Path

from .supervised import SupervisedResult
from .unsupervised import UnsupervisedResult


def _common_row(result):
    return {
        "transition": result.transition,
        "model": result.model,
        "feature_set": result.feature_set,
        "standardize": result.standardize,
        "method": result.method,
        "validation_accuracy": result.validation_accuracy,
        "validation_passed": result.validation_passed,
        "training_size": result.training_size,
        "validation_size": result.validation_size,
        "full_size": result.full_size,
        "critical_method_crossing": "P=0.5",
        "critical_method_peak": "chi_peak",
        "critical_crossing": result.crossing,
        "critical_crossing_err": result.crossing_error,
        "critical_crossing_close": result.crossing_nearest,
        "critical_chi_peak": result.chi_peak,
        "chi_definition": "mean(P^2)-mean(P)^2",
    }


def summary_row(result: SupervisedResult | UnsupervisedResult) -> dict:
    row = _common_row(result)
    if isinstance(result, SupervisedResult):
        row.update({
            "calibrated": result.calibrated,
            "calibration_cv": result.calibration_cv,
        })
    else:
        row.update({
            "calibrated": None,
            "calibration_cv": None,
            "birch_threshold": result.birch_threshold,
        })
    return row


def curve_rows(result: SupervisedResult | UnsupervisedResult) -> list[dict]:
    rows = []
    for parameter, mean, mean_error, chi, chi_error in zip(
        result.curve.parameter,
        result.curve.mean,
        result.curve.mean_error,
        result.curve.chi,
        result.curve.chi_error,
    ):
        rows.append({
            "transition": result.transition,
            "model": result.model,
            "feature_set": result.feature_set,
            "standardize": result.standardize,
            "parameter": parameter,
            "mean_probability": mean,
            "mean_error": mean_error,
            "chi": chi,
            "chi_error": chi_error,
            "chi_definition": "mean(P^2)-mean(P)^2",
        })
    return rows


def write_result(result, output_dir: Path) -> tuple[Path, Path]:
    """Write one result without overwriting an existing result file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.transition}_{result.model}_{result.feature_set}"
    if result.standardize:
        stem += "_STD"
    summary_path = output_dir / f"{stem}_summary.csv"
    curve_path = output_dir / f"{stem}_curve.csv"
    for path in (summary_path, curve_path):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing result: {path}"
            )

    summary = summary_row(result)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    curves = curve_rows(result)
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    return summary_path, curve_path
