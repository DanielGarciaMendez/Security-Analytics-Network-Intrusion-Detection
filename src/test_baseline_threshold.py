import unittest
import numpy as np
from train_checkpoint_baseline import select_threshold


class ThresholdTests(unittest.TestCase):
    def test_ties_cannot_split_negative_rows(self):
        y = np.array([0] * 20 + [1, 1])
        scores = np.array([0.1] * 18 + [0.8, 0.8] + [0.9, 0.7])
        threshold = select_threshold(y, scores)
        self.assertEqual(threshold, .9)
        self.assertLessEqual(np.mean(scores[y == 0] >= threshold), .05)

    def test_above_max_fallback_is_finite(self):
        threshold = select_threshold([0, 1], [.9, .2])
        self.assertTrue(np.isfinite(threshold))
        self.assertGreater(threshold, .9)


if __name__ == "__main__":
    unittest.main()
