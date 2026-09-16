import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phase_transitions.reporting import curve_rows, summary_row, write_result
from phase_transitions.supervised import Curve, SupervisedResult


class ReportingTests(unittest.TestCase):
    def make_result(self):
        curve = Curve(
            parameter=np.array([0.0, 1.0]),
            mean=np.array([0.2, 0.8]),
            mean_error=np.array([0.01, 0.02]),
            chi=np.array([0.1, 0.2]),
            chi_error=np.array([0.01, 0.02]),
        )
        return SupervisedResult(
            transition="TEST",
            model="logistic_regression",
            feature_set="30",
            standardize=False,
            method="endpoint_validation",
            calibrated=False,
            calibration_cv=5,
            validation_accuracy=1.0,
            validation_passed=True,
            training_size=10,
            validation_size=4,
            full_size=20,
            curve=curve,
            crossing=0.5,
            crossing_error=0.01,
            crossing_nearest=1.0,
            chi_peak=1.0,
        )

    def test_summary_declares_method_and_chi_definition(self):
        row = summary_row(self.make_result())
        self.assertEqual(row["method"], "endpoint_validation")
        self.assertEqual(row["chi_definition"], "mean(P^2)-mean(P)^2")

    def test_curve_has_one_row_per_parameter(self):
        self.assertEqual(len(curve_rows(self.make_result())), 2)

    def test_writer_refuses_overwrite(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = self.make_result()
            write_result(result, output_dir)
            with self.assertRaises(FileExistsError):
                write_result(result, output_dir)


if __name__ == "__main__":
    unittest.main()
