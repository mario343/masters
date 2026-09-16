"""Evaluate full curves only for endpoint configurations that passed validation."""

from __future__ import annotations

import argparse
import sys
from time import perf_counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase_transitions.config import TRANSITIONS
from phase_transitions.reporting import write_result
from phase_transitions.unsupervised import run_endpoint_clustering


MODEL_NAMES = {
    "kmeans": "kmeans",
    "gaussian_mixture": "gaussian_mixture",
    "bayesian_gmm": "bayesian_gmm",
    "meanshift": "meanshift",
    "birch": "birch",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "Results" / "REFactored" / "endpoint_validation_matrix.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "Results" / "REFactored" / "unsupervised",
    )
    parser.add_argument("--block-min", type=int, default=2)
    parser.add_argument("--block-max", type=int, default=100)
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix)
    selected = matrix[(matrix["validation_passed"] == True) & matrix["error"].isna()]
    block_counts = range(args.block_min, args.block_max + 1)
    print(f"Selected {len(selected)} validated configurations", flush=True)

    for number, row in enumerate(selected.to_dict("records"), start=1):
        transition = row["transition"]
        model = MODEL_NAMES[row["model"]]
        feature_set = str(row["feature_set"])
        standardize = bool(row["standardize"])
        started = perf_counter()
        print(
            f"[{number}/{len(selected)}] START {transition} {model} "
            f"features={feature_set} std={standardize}",
            flush=True,
        )
        result = run_endpoint_clustering(
            TRANSITIONS[transition],
            model_name=model,
            feature_set_name=feature_set,
            standardize=standardize,
            block_counts=block_counts,
        )
        output_dir = args.output_root / "endpoint_validation" / transition
        summary, curve = write_result(result, output_dir)
        print(
            f"[{number}/{len(selected)}] {transition} {model} "
            f"features={feature_set} std={standardize} "
            f"crossing={result.crossing} elapsed={perf_counter() - started:.1f}s "
            f"summary={summary.name} curve={curve.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
