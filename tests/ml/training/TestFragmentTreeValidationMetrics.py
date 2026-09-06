import unittest

import numpy as np
import pandas as pd

from clefts.libs.msentity.msentity.core.MSDataset import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.ml.training.fragment_tree_training.training_model import (
    calculate_peak_selection_metrics,
    calculate_precursor_detection_metrics,
    exclude_precursor_peaks,
    summarize_distribution,
)


def _dataset(spectra, precursor_mz=None):
    lengths = [len(peaks) for peaks in spectra]
    data = np.concatenate(spectra, axis=0) if sum(lengths) else np.empty((0, 2))
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    metadata = {"id": range(len(spectra))}
    if precursor_mz is not None:
        metadata["PrecursorMZ"] = precursor_mz
    return MSDataset(
        spectrum_metadata=pd.DataFrame(metadata),
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

    def test_precursor_detection_metrics(self):
        target = _dataset(
            [
                np.array([[100.0, 10.0], [101.0, 5.0], [300.0, 1.0]]),
                np.array([[300.0, 8.0]]),
            ],
            precursor_mz=[300.0, 300.0],
        )
        predicted = _dataset(
            [
                np.array([[100.005, 1.0], [300.001, 0.5], [110.0, 0.2]]),
                np.array([[301.0, 1.0]]),
            ],
            precursor_mz=[300.0, 300.0],
        )

        metrics = calculate_precursor_detection_metrics(predicted, target)

        # spectrum 0: precursor present in both -> true positive
        # spectrum 1: precursor present in target only -> false negative
        self.assertAlmostEqual(metrics["precursor_detection_precision"], 1.0)
        self.assertAlmostEqual(metrics["precursor_detection_recall"], 0.5)
        self.assertAlmostEqual(metrics["precursor_detection_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precursor_detection_f1"], 2 / 3)

    def test_exclude_precursor_peaks_removes_only_matching_peaks(self):
        dataset = _dataset(
            [
                np.array([[100.0, 10.0], [101.0, 5.0], [300.0, 1.0]]),
                np.array([[301.0, 1.0]]),
            ],
            precursor_mz=[300.0, 300.0],
        )

        filtered = exclude_precursor_peaks(dataset)

        self.assertEqual(list(filtered.peaks.lengths), [2, 1])
        self.assertTrue(
            np.allclose(filtered.peaks[0].data, [[100.0, 10.0], [101.0, 5.0]])
        )
        # 301.0 is outside the tolerance window around the precursor (300.0)
        # and must be kept as a distinct fragment peak.
        self.assertTrue(np.allclose(filtered.peaks[1].data, [[301.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
