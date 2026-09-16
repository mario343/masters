#!/usr/bin/env python3
"""Run an explicit experiment matrix into a new results directory."""

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
    parser.add_argument("--kind", choices=("supervised", "unsupervised"), required=True)
    parser.add_argument(
        "--method",
        choices=("endpoint_validation", "full_range"),
        default="endpoint_validation",
    )
    parser.add_argument("--transitions", nargs="+", choices=sorted(TRANSITIONS), default=sorted(TRANSITIONS))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--feature-sets", nargs="+", choices=("30", "20", "12"), default=["30", "20", "12"])
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--train-stride", type=int)
    parser.add_argument("--full-stride", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--block-min", type=int, default=2)
    parser.add_argument("--block-max", type=int, default=100)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_min < 2 or args.block_max < args.block_min:
        raise SystemExit("invalid block range")
    if args.kind == "supervised" and args.method != "endpoint_validation":
        raise SystemExit("full_range is currently defined only for unsupervised runs")

    block_counts = range(args.block_min, args.block_max + 1)
    total = 0
    for transition in args.transitions:
        config = TRANSITIONS[transition]
        for model in args.models:
            for feature_set in args.feature_sets:
                if args.kind == "supervised":
                    result = run_endpoint_experiment(
                        config,
                        model_name=model,
                        feature_set_name=feature_set,
                        standardize=args.standardize,
                        train_stride=args.train_stride,
                        full_stride=args.full_stride,
                        validation_fraction=args.validation_fraction,
                        block_counts=block_counts,
                    )
                    method_dir = "endpoint_validation"
                elif args.method == "endpoint_validation":
                    try:
                        result = run_endpoint_clustering(
                            config,
                            model_name=model,
                            feature_set_name=feature_set,
                            standardize=args.standardize,
                            train_stride=args.train_stride,
                            full_stride=args.full_stride,
                            validation_fraction=args.validation_fraction,
                            block_counts=block_counts,
                        )
                    except EndpointValidationError as exc:
                        print(f"SKIPPED: {exc}")
                        continue
                    method_dir = "endpoint_validation"
                else:
                    result = run_full_range_clustering(
                        config,
                        model_name=model,
                        feature_set_name=feature_set,
                        standardize=args.standardize,
                        full_stride=args.full_stride,
                        block_counts=block_counts,
                    )
                    method_dir = "full_range"

                output_dir = args.output_root / args.kind / method_dir / transition
                summary, curve = write_result(result, output_dir)
                total += 1
                print(
                    f"[{total}] {transition} {model} {feature_set} "
                    f"validation={result.validation_accuracy} "
                    f"crossing={result.crossing} "
                    f"summary={summary} curve={curve}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
