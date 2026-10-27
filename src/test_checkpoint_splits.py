"""Boundary regression checks: python -m unittest discover -s src -p test_checkpoint_splits.py."""
import unittest

import numpy as np

from create_checkpoint_splits import connected_groups, choose_validation
from preview_checkpoint_splits import valid_feature_matrix, keys_for, FEATURES
import pandas as pd


class SplitBoundaryTests(unittest.TestCase):
    def test_indirect_bridge_stays_together(self):
        # Rows 0/1 share exact inputs, rows 1/2 share a near key.
        groups = connected_groups([b"a", b"a", b"b", b"c"], [1, 2, 2, 3])
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])
        blocked = set(groups[[False, False, True, False]])
        self.assertEqual([g in blocked for g in groups], [True, True, True, False])

    def test_group_split_handles_mixed_labels_reproducibly(self):
        groups = np.repeat(np.arange(50), 4)
        labels = np.tile([0, 1, 1, 0], 50)
        train, validation = choose_validation(labels, groups)
        train2, validation2 = choose_validation(labels, groups)
        self.assertFalse(set(groups[train]) & set(groups[validation]))
        self.assertEqual(set(train) | set(validation), set(range(200)))
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(validation, validation2)

    def test_quality_and_zero_equality(self):
        frame = pd.DataFrame([[0, 1, 2], [-0.0, 1, 2], [-1, 1, 2], [np.inf, 1, 2], [np.nan, 1, 2]], columns=FEATURES)
        values, valid = valid_feature_matrix(frame)
        self.assertEqual(valid.tolist(), [True, True, False, False, False])
        self.assertEqual(keys_for(values[:1]), keys_for(values[1:2]))


if __name__ == "__main__":
    unittest.main()
