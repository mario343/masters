import unittest

import numpy as np

from phase_transitions.reweighting import (
    reweighted_observables,
    weighted_block_jackknife,
)


class ReweightingTests(unittest.TestCase):
    def test_weighted_observables_use_population_variance(self):
        result = reweighted_observables([0.0, 2.0], [1.0, 3.0])
        self.assertAlmostEqual(result["mean"], 1.5)
        self.assertAlmostEqual(result["chi"], 0.75)
        self.assertAlmostEqual(result["ess"], 1.6)

    def test_weighted_jackknife_matches_direct_unweighted_case(self):
        values = np.arange(8.0)
        weights = np.ones_like(values)
        result = weighted_block_jackknife(values, weights, 2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["mean"][0], np.mean(values))
        self.assertAlmostEqual(result["chi"][0], np.var(values, ddof=0))


if __name__ == "__main__":
    unittest.main()
