#!/usr/bin/env python3
"""Run one validated endpoint experiment and write new Results/ CSV files.

This script deliberately refuses to overwrite existing result files.  Use a
new output directory when comparing a methodological variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_transitions.config import TRANSITIONS
from phase_transitions.reporting import write_result
from phase_transitions.supervised import run_endpoint_experiment
from phase_transitions.unsupervised import (
    EndpointValidationError,
    run_endpoint_clustering,
    run_full_range_clustering,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", choices=sorted(TRANSITIONS))
    parser.add_argument("--kind", choices=("supervised", "unsupervised"), default="supervised")
    parser.add_argument(
        "--method",
        choices=("endpoint_validation", "full_range"),
        default="endpoint_validation",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-set", default="30", choices=("30", "20", "12"))
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--train-stride", type=int)
    parser.add_argument("--full-stride", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--block-min", type=int, default=2)
    parser.add_argument("--block-max", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TRANSITIONS[args.transition]
    if args.kind == "supervised" and args.method != "endpoint_validation":
        raise SystemExit("full_range is currently defined only for unsupervised runs")
    if args.block_min < 2 or args.block_max < args.block_min:
        raise SystemExit("invalid block range")
    block_counts = range(args.block_min, args.block_max + 1)

    if args.kind == "supervised":
        result = run_endpoint_experiment(
            config,
            model_name=args.model,
            feature_set_name=args.feature_set,
            standardize=args.standardize,
            train_stride=args.train_stride,
            full_stride=args.full_stride,
            validation_fraction=args.validation_fraction,
            block_counts=block_counts,
        )
        default_dir = Path("Results") / "SUPERVISED" / args.transition
    else:
        if args.method == "endpoint_validation":
            try:
                result = run_endpoint_clustering(
                    config,
                    model_name=args.model,
                    feature_set_name=args.feature_set,
                    standardize=args.standardize,
                    train_stride=args.train_stride,
                    full_stride=args.full_stride,
                    validation_fraction=args.validation_fraction,
                    block_counts=block_counts,
                )
            except EndpointValidationError as exc:
                raise SystemExit(f"SKIPPED: {exc}") from exc
        else:
            result = run_full_range_clustering(
                config,
                model_name=args.model,
                feature_set_name=args.feature_set,
                standardize=args.standardize,
                full_stride=args.full_stride,
                block_counts=block_counts,
            )
        default_dir = Path("Results") / "UNSUPERVISED" / args.transition

    output_dir = args.output_dir or default_dir
    summary_path, curve_path = write_result(result, output_dir)
    print(f"validation_accuracy={result.validation_accuracy:.6f}")
    print(f"validation_passed={result.validation_passed}")
    print(f"crossing={result.crossing}")
    print(f"chi_peak={result.chi_peak}")
    print(f"summary={summary_path}")
    print(f"curve={curve_path}")


if __name__ == "__main__":
    main()
