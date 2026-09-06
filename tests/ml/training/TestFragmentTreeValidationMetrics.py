import unittest

import numpy as np
import pandas as pd

from clefts.libs.msentity.msentity.core.MSDataset import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.ml.training.fragment_tree_training.training_model import (
    calculate_peak_selection_metrics,
    calculate_precursor_detection_metrics,
    exclude_precursor_peaks,
    log_distribution_cards,
    summarize_distribution,
)


class _FakeWriter:
    def __init__(self):
        self.calls = {}

    def add_scalars(self, tag, values, step):
        self.calls[tag] = dict(values)


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
    def test_log_distribution_cards_groups_scoped_variants_onto_one_card(self):
        writer = _FakeWriter()
        summaries = {
            "validation": {
                "selection_precision_mean": 0.9,
                "selection_precision_min": 0.5,
                "selection_precision@excl_precursor_mean": 0.8,
                "selection_precision@by_adduct:[M+H]+_mean": 0.7,
                "cosine_mean": 0.95,
            }
        }

        log_distribution_cards(writer, "peak_selection", summaries, 10)

        # One card per base metric, not one per scope variant.
        self.assertEqual(set(writer.calls.keys()), {
            "peak_selection/selection_precision",
            "peak_selection/cosine",
        })
        precision_card = writer.calls["peak_selection/selection_precision"]
        self.assertEqual(precision_card["validation_mean"], 0.9)
        self.assertEqual(precision_card["validation_min"], 0.5)
        self.assertEqual(precision_card["validation_excl_precursor_mean"], 0.8)
        self.assertEqual(precision_card["validation_by_adduct:[M+H]+_mean"], 0.7)
        self.assertEqual(writer.calls["peak_selection/cosine"]["validation_mean"], 0.95)

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

    def test_precursor_detection_metrics_spectrum_indexes_matches_manual_subset(self):
        target = _dataset(
            [
                np.array([[100.0, 10.0], [101.0, 5.0], [300.0, 1.0]]),
                np.array([[300.0, 8.0]]),
                np.array([[300.0, 8.0]]),
            ],
            precursor_mz=[300.0, 300.0, 300.0],
        )
        predicted = _dataset(
            [
                np.array([[100.005, 1.0], [300.001, 0.5], [110.0, 0.2]]),
                np.array([[301.0, 1.0]]),
                np.array([[300.0, 1.0]]),
            ],
            precursor_mz=[300.0, 300.0, 300.0],
        )

        # Omitting spectrum_indexes reproduces the full corpus-wide result.
        full = calculate_precursor_detection_metrics(predicted, target)
        full_manual = calculate_precursor_detection_metrics(
            predicted, target, spectrum_indexes=np.array([0, 1, 2])
        )
        for name in full:
            self.assertAlmostEqual(full[name], full_manual[name])

        # A subset (spectra 0 and 2: both true positives) should reproduce
        # what manually pre-filtering the datasets to that subset would give.
        subset_metrics = calculate_precursor_detection_metrics(
            predicted, target, spectrum_indexes=np.array([0, 2])
        )
        self.assertAlmostEqual(subset_metrics["precursor_detection_precision"], 1.0)
        self.assertAlmostEqual(subset_metrics["precursor_detection_recall"], 1.0)
        self.assertAlmostEqual(subset_metrics["precursor_detection_accuracy"], 1.0)

    def test_summarize_distribution_excludes_outliers_from_min_max(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        summary = summarize_distribution(values)

        # q1=2.25, q3=4.75, iqr=2.5 -> fence = [-1.5, 8.5]; 100.0 is an
        # outlier and must not become the reported max.
        self.assertAlmostEqual(summary["q3"], 4.75)
        self.assertLess(summary["max"], 100.0)
        self.assertAlmostEqual(summary["max"], 5.0)
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["mean"], values.mean())

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
