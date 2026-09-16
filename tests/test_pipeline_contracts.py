import unittest

import numpy as np

from phase_transitions.config import TRANSITIONS, feature_sets
from phase_transitions.data import load_full, load_train
from phase_transitions.preprocessing import fit_transform_train
from phase_transitions.models import build_classifier
from phase_transitions.validation import block_validation_split, endpoint_indices


class PipelineContractTests(unittest.TestCase):
    def test_feature_sets_have_expected_sizes(self):
        sets = feature_sets()
        self.assertEqual({key: len(value) for key, value in sets.items()}, {
            "30": 30,
            "20": 20,
            "12": 12,
        })

    def test_all_processed_datasets_have_thirty_features(self):
        for config in TRANSITIONS.values():
            train = load_train(config, stride=1000)
            full = load_full(config, stride=1000)
            self.assertEqual(train.X.shape[1], 30)
            self.assertEqual(full.X.shape[1], 30)
            self.assertEqual(len(train.X), len(train.parameter))
            self.assertEqual(len(full.X), len(full.parameter))

    def test_endpoint_split_is_disjoint_and_stratified(self):
        labels = np.array([0] * 10 + [1] * 20)
        split = block_validation_split(labels, validation_fraction=0.2)
        self.assertEqual(len(np.intersect1d(split.train_indices, split.validation_indices)), 0)
        self.assertEqual(set(labels[split.train_indices]), {0, 1})
        self.assertEqual(set(labels[split.validation_indices]), {0, 1})

    def test_endpoint_indices_find_both_phases(self):
        parameter = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
        indices = endpoint_indices(parameter, low_value=1.0, high_value=3.0)
        np.testing.assert_array_equal(indices, np.array([0, 1, 3, 4]))

    def test_scaler_is_fit_on_train_only(self):
        train = np.array([[0.0], [2.0]])
        other = np.array([[100.0]])
        transformed_train, transformed_other, scaler = fit_transform_train(
            train,
            other,
            standardize=True,
        )
        np.testing.assert_allclose(transformed_train.ravel(), [-1.0, 1.0])
        self.assertAlmostEqual(float(scaler.mean_[0]), 1.0)
        self.assertAlmostEqual(float(transformed_other[0, 0]), 99.0)

    def test_svm_calibration_is_explicitly_five_fold(self):
        model = build_classifier(
            "svm_rbf",
            calibrated=True,
            standardize=True,
            calibration_cv=5,
        )
        self.assertEqual(model.cv, 5)

    def test_unsupervised_sampling_is_transition_specific(self):
        self.assertEqual(
            (TRANSITIONS["AB"].unsupervised_train_stride,
             TRANSITIONS["AB"].unsupervised_full_stride),
            (1, 1),
        )
        self.assertEqual(
            (TRANSITIONS["BCB"].unsupervised_train_stride,
             TRANSITIONS["BCB"].unsupervised_full_stride),
            (1, 1),
        )
        self.assertEqual(
            (TRANSITIONS["CA"].unsupervised_train_stride,
             TRANSITIONS["CA"].unsupervised_full_stride),
            (10, 10),
        )
        self.assertEqual(
            {config.meanshift_fit_stride for config in TRANSITIONS.values()},
            {10},
        )


if __name__ == "__main__":
    unittest.main()
