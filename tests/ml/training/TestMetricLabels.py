import math
import unittest

from clefts.ml.training.fragment_tree_training.metric_labels import (
    ce_range_label,
    tensorboard_label,
)


class TestMetricLabels(unittest.TestCase):
    def test_tensorboard_label_sanitizes_slashes_and_spaces(self):
        self.assertEqual(tensorboard_label("[M+H]+"), "[M+H]+")
        self.assertEqual(tensorboard_label(" [M+HCOO]-/2 "), "[M+HCOO]-∕2")
        self.assertEqual(tensorboard_label("M + H"), "M_+_H")

    def test_ce_range_label_bins_by_quartile_cuts(self):
        cuts = {"q1": 10.0, "median": 20.0, "q3": 30.0}
        self.assertEqual(ce_range_label(5.0, **cuts), "min-to-q1")
        self.assertEqual(ce_range_label(15.0, **cuts), "q1-to-median")
        self.assertEqual(ce_range_label(25.0, **cuts), "median-to-q3")
        self.assertEqual(ce_range_label(35.0, **cuts), "q3-to-max")
        self.assertEqual(ce_range_label(float("nan"), **cuts), "non-finite")
        self.assertEqual(ce_range_label(math.inf, **cuts), "non-finite")


if __name__ == "__main__":
    unittest.main()
