import unittest

import numpy as np

from phase_transitions.unsupervised import (
    _phase_cluster_mapping,
    median_nearest_neighbor_distance,
)


class UnsupervisedTests(unittest.TestCase):
    def test_phase_mapping_uses_extreme_parameter_clusters(self):
        labels = np.array([1, 1, 2, 0, 0, 0])
        parameter = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        low, high = _phase_cluster_mapping(labels, parameter)
        self.assertEqual(low, 1)
        self.assertEqual(high, 0)

    def test_median_nearest_neighbor_distance(self):
        X = np.array([[0.0], [1.0], [3.0], [6.0]])
        self.assertAlmostEqual(median_nearest_neighbor_distance(X), 1.5)


if __name__ == "__main__":
    unittest.main()
