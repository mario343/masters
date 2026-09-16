import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase_transitions.config import TransitionConfig
from phase_transitions.knn_timing import DEFAULT_NEIGHBORS, benchmark_knn


class KNNTimingTests(unittest.TestCase):
    def test_default_neighbor_grid_matches_legacy_benchmark(self):
        self.assertEqual(len(DEFAULT_NEIGHBORS), 31)
        self.assertEqual(DEFAULT_NEIGHBORS[:3], (1, 3, 5))
        self.assertEqual(DEFAULT_NEIGHBORS[-5:], (61, 81, 101, 151, 201))

    def test_small_benchmark_returns_timing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            rng = np.random.default_rng(0)
            X_train = rng.normal(size=(20, 30))
            y_train = np.array([0] * 10 + [1] * 10)
            X_full = rng.normal(size=(20, 30))
            parameter = np.array([0.0] * 10 + [1.0] * 10)
            np.savez(data_dir / "TRAIN_30.npz", X=X_train, y=y_train, Delta=parameter)
            np.savez(data_dir / "FULL_30.npz", X=X_full, Delta=parameter)

            config = TransitionConfig(
                name="TEST",
                parameter_name="Delta",
                phase_low_value=0.0,
                phase_high_value=1.0,
                phase_low_name="low",
                phase_high_name="high",
                data_dir=data_dir,
            )
            rows = benchmark_knn(
                config,
                feature_set_names=("12",),
                algorithms=("auto",),
                neighbors=(1,),
                block_counts=range(2, 5),
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transition"], "TEST")
        self.assertEqual(rows[0]["algorithm"], "auto")
        self.assertEqual(rows[0]["n_neighbors"], 1)
        self.assertGreaterEqual(rows[0]["runtime_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
