"""Run the inexpensive endpoint-validation matrix for all unsupervised models."""

from __future__ import annotations

import argparse
import sys
from time import perf_counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase_transitions.config import ROOT as DATA_ROOT, TRANSITIONS
from phase_transitions.unsupervised import endpoint_validation_accuracy


def build_matrix(
    *,
    train_stride: int | None = None,
    transitions: tuple[str, ...] = ("AB", "BCB", "CA"),
    models: tuple[str, ...] = (
        "kmeans",
        "gaussian_mixture",
        "bayesian_gmm",
        "meanshift",
        "birch",
    ),
    output_path: Path | None = None,
) -> pd.DataFrame:
    rows = []
    configurations = [
        (transition_name, model_name, feature_set_name, standardize)
        for transition_name in transitions
        for model_name in models
        for feature_set_name in ("30", "20", "12")
        for standardize in (False, True)
    ]
    total = len(configurations)
    for number, (transition_name, model_name, feature_set_name, standardize) in enumerate(configurations, 1):
        config = TRANSITIONS[transition_name]
        started = perf_counter()
        print(
            f"[{number}/{total}] START {transition_name} {model_name} "
            f"features={feature_set_name} std={standardize}",
            flush=True,
        )
        try:
            (
                accuracy,
                passed,
                birch_threshold,
                training_size,
                validation_size,
            ) = endpoint_validation_accuracy(
                config,
                model_name=model_name,
                feature_set_name=feature_set_name,
                standardize=standardize,
                train_stride=train_stride,
            )
            error = None
        except Exception as exc:  # retain failures as diagnostics
            accuracy = None
            passed = False
            birch_threshold = None
            training_size = None
            validation_size = None
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "transition": transition_name,
            "model": model_name,
            "feature_set": feature_set_name,
            "standardize": standardize,
            "train_stride": (
                config.unsupervised_train_stride
                if train_stride is None
                else train_stride
            ),
            "validation_accuracy": accuracy,
            "validation_passed": passed,
            "birch_threshold": birch_threshold,
            "training_size": training_size,
            "validation_size": validation_size,
            "error": error,
        })
        status = "ERROR" if error else ("PASS" if passed else "FAIL")
        print(
            f"[{number}/{total}] {status} {transition_name} {model_name} "
            f"features={feature_set_name} std={standardize} "
            f"accuracy={accuracy} elapsed={perf_counter() - started:.1f}s",
            flush=True,
        )

    result = pd.DataFrame(rows)
    if output_path is not None:
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument(
        "--transitions",
        nargs="+",
        choices=tuple(TRANSITIONS),
        default=tuple(TRANSITIONS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("kmeans", "gaussian_mixture", "bayesian_gmm", "meanshift", "birch"),
        default=("kmeans", "gaussian_mixture", "bayesian_gmm", "meanshift", "birch"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "Results" / "REFactored" / "endpoint_validation_matrix.csv",
    )
    args = parser.parse_args()
    matrix = build_matrix(
        train_stride=args.train_stride,
        transitions=tuple(args.transitions),
        models=tuple(args.models),
        output_path=args.output,
    )
    print(matrix.to_string(index=False), flush=True)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
