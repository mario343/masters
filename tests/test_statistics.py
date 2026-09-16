import unittest

import numpy as np

from phase_transitions.statistics import (
    block_jackknife,
    find_first_crossing,
    find_peak,
    population_variance,
    weighted_population_variance,
)


class StatisticsTests(unittest.TestCase):
    def test_population_variance_matches_article_definition(self):
        values = np.array([0.0, 2.0])
        self.assertAlmostEqual(population_variance(values), 1.0)
        self.assertAlmostEqual(np.var(values, ddof=0), 1.0)
        self.assertAlmostEqual(np.var(values, ddof=1), 2.0)

    def test_weighted_variance_is_population_variance(self):
        values = np.array([0.0, 2.0])
        weights = np.array([1.0, 3.0])
        self.assertAlmostEqual(weighted_population_variance(values, weights), 0.75)

    def test_block_jackknife_keeps_ddof_one_for_replicates(self):
        values = np.arange(8.0)
        result = block_jackknife(values, population_variance, 2)
        self.assertIsNotNone(result)

        theta_hat, theta_jack, standard_error = result
        expected_hat = np.var(values, ddof=0)
        replicates = np.array([
            np.var(values[4:], ddof=0),
            np.var(values[:4], ddof=0),
        ])
        expected_bias = np.mean(replicates) - expected_hat
        expected_jack = expected_hat - expected_bias
        expected_se = np.sqrt(np.var(replicates, ddof=1) / 2)

        self.assertAlmostEqual(theta_hat, expected_hat)
        self.assertAlmostEqual(theta_jack, expected_jack)
        self.assertAlmostEqual(standard_error, expected_se)

    def test_first_crossing_uses_first_linear_crossing(self):
        crossing = find_first_crossing(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.1, 0.7, 0.9]),
        )
        self.assertIsNotNone(crossing)
        value, nearest, left_index = crossing
        self.assertAlmostEqual(value, 2 / 3)
        self.assertAlmostEqual(nearest, 1.0)
        self.assertEqual(left_index, 0)

    def test_peak_uses_parabolic_refinement(self):
        parameter = np.array([0.0, 1.0, 2.0])
        values = -(parameter - 1.25) ** 2 + 3.0
        self.assertAlmostEqual(find_peak(parameter, values), 1.25)


if __name__ == "__main__":
    unittest.main()
