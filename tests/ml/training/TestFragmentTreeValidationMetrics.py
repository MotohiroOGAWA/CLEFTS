import unittest

import numpy as np
import pandas as pd

from clefts.libs.msentity.msentity.core.MSDataset import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.ml.training.fragment_tree_training.training_model import (
    calculate_peak_selection_metrics,
    summarize_distribution,
)


def _dataset(spectra):
    lengths = [len(peaks) for peaks in spectra]
    data = np.concatenate(spectra, axis=0) if sum(lengths) else np.empty((0, 2))
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return MSDataset(
        spectrum_metadata=pd.DataFrame({"id": range(len(spectra))}),
        peak_series=PeakSeries(data=np.asarray(data, dtype=float), offsets=offsets),
    )


class TestFragmentTreeValidationMetrics(unittest.TestCase):
    def test_selection_coverage_rank_and_matched_intensity(self):
        target = _dataset(
            [np.array([[100.0, 10.0], [101.0, 5.0], [102.0, 1.0]])]
        )
        predicted = _dataset(
            [np.array([[100.005, 1.0], [102.0, 0.5], [110.0, 0.2]])]
        )

        metrics = calculate_peak_selection_metrics(
            predicted, target, mz_tolerance_da=0.01, top_k=(1, 2, 3)
        )

        self.assertAlmostEqual(metrics["selection_precision"][0], 2 / 3)
        self.assertAlmostEqual(metrics["selection_recall"][0], 2 / 3)
        self.assertAlmostEqual(metrics["selected_intensity_fraction"][0], 11 / 16)
        self.assertEqual(metrics["top1_recall"][0], 1.0)
        self.assertEqual(metrics["top2_recall"][0], 0.5)
        self.assertAlmostEqual(metrics["matched_intensity_mae"][0], 0.2)
        self.assertAlmostEqual(metrics["matched_intensity_weighted_mae"][0], 0.4 / 11)

    def test_closest_match_is_one_to_one_and_nan_is_ignored_in_summary(self):
        target = _dataset([np.array([[100.0, 1.0], [100.006, 0.5]])])
        predicted = _dataset([np.array([[100.004, 1.0]])])

        metrics = calculate_peak_selection_metrics(predicted, target)
        self.assertEqual(metrics["selection_precision"][0], 1.0)
        self.assertEqual(metrics["selection_recall"][0], 0.5)
        summary = summarize_distribution(np.array([float("nan"), 0.25, 0.75]))
        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary["median"], 0.5)


if __name__ == "__main__":
    unittest.main()
