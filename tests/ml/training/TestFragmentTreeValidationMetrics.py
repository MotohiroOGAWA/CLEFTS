import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from clefts.libs.msentity.msentity.core.MSDataset import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.ml.training.fragment_tree_training.training_model import (
    calculate_peak_selection_metrics,
    calculate_precursor_detection_metrics,
    exclude_precursor_peaks,
    evaluate_validation_cosine,
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
    def test_validation_ce_groups_parse_units_like_structure_builder(self):
        raw_ce = ["20 V", "30 eV", "0.04 keV", "50%", 10, None, "unknown"]
        parsed_ce = [20, 30, 40, 25, 10, np.nan, np.nan]
        dataset = _dataset(
            [np.array([[100.0, 1.0], [250.0, 0.5]]) for _ in raw_ce],
            precursor_mz=[250.0] * len(raw_ce),
        )
        dataset["AdductType"] = ["[M+H]+"] * len(raw_ce)
        model = SimpleNamespace(
            intensity_predictor=object(),
            parameters=lambda: iter([torch.zeros(1)]),
        )
        summaries = []
        for ce_values in (raw_ce, parsed_ce):
            dataset["CollisionEnergy"] = ce_values
            metrics = {}
            with patch(
                "clefts.ml.training.fragment_tree_training.training_model.predict_validation_msdataset",
                return_value=(dataset, dataset),
            ):
                score = evaluate_validation_cosine(
                    model=model, dataset=dataset, selection_metric_means=metrics,
                )
            self.assertAlmostEqual(score, 1.0)
            summaries.append(metrics)
        self.assertIn("cosine@by_ce_range:non-finite", summaries[0])
        self.assertEqual(summaries[0].keys(), summaries[1].keys())
        for key in summaries[0]:
            np.testing.assert_allclose(summaries[0][key], summaries[1][key], equal_nan=True)

    def test_cards_separate_stages_and_conditions_but_overlay_splits(self):
        writer = _FakeWriter()
        values = {
            "edge_retain_precision_mean": 0.9,
            "edge_retain_precision@by_adduct:[M+H]+_mean": 0.7,
            "edge_retain_precision@by_ce_range:min-to-q1_mean": 0.6,
            "pairwise_ranking_accuracy_mean": 0.8,
            "precursor/keep_loss_mean": 0.1,
            "intensity_cosine_similarity_mean": 0.95,
        }
        log_distribution_cards(writer, "metrics", {
            split: values for split in ("train", "train_window", "validation")
        }, 10)
        self.assertEqual(set(writer.calls), {
            "edge_selection/edge_retain_precision/overall",
            "edge_selection/edge_retain_precision/by_adduct",
            "edge_selection/edge_retain_precision/by_ce_range",
            "edge_absolute_score/pairwise_ranking_accuracy/overall",
            "loss/precursor/keep_loss/overall",
            "intensity/intensity_cosine_similarity/overall",
        })
        self.assertEqual(set(writer.calls["edge_selection/edge_retain_precision/overall"]),
                         {"train_mean", "train_window_mean", "validation_mean"})
        self.assertEqual(writer.calls["edge_selection/edge_retain_precision/by_adduct"]["train_[M+H]+_mean"], 0.7)

    def test_peak_diagnostics_keep_scopes_on_separate_cards(self):
        writer = _FakeWriter()
        log_distribution_cards(writer, "peak_selection", {"validation": {
            "selection_precision_mean": 0.9,
            "selection_precision@excl_precursor_mean": 0.8,
            "selection_precision@by_adduct:[M+H]+_mean": 0.7,
            "cosine_mean": 0.95,
        }}, 10)
        self.assertEqual(set(writer.calls), {
            "peak_selection/selection_precision/overall",
            "peak_selection/selection_precision/excl_precursor",
            "peak_selection/selection_precision/by_adduct",
            "intensity/cosine/overall",
        })

    def test_validation_generates_cosine_and_mirror_images(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        target = _dataset([np.array([[100., 1000.], [250., 500.]])], precursor_mz=[250.])
        predicted = _dataset([np.array([[100., 1.], [250., 0.5]])], precursor_mz=[250.])
        model = SimpleNamespace(intensity_predictor=object(), parameters=lambda: iter([torch.zeros(1)]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with SummaryWriter(str(output / 'events')) as writer:
                with patch(
                    "clefts.ml.training.fragment_tree_training.training_model.predict_validation_msdataset",
                    return_value=(predicted, target),
                ) as predict:
                    score = evaluate_validation_cosine(
                        model=model, dataset=target, writer=writer, global_step=100,
                        output_dir=output / 'validation', validation_scope='validation_below_threshold',
                    )
                predict.assert_called_once()
            self.assertAlmostEqual(score, 1.)
            events = EventAccumulator(str(output / 'events')).Reload()
            images = events.Images('validation_spectra/validation_below_threshold/level_1_high_to_low')
            self.assertEqual(images[0].step, 100)
            self.assertGreater(len(images[0].encoded_image_string), 100)
            self.assertTrue((output / 'validation/spectra/step_00000100/level_1.png').exists())
            np.testing.assert_equal(target.peaks.data[:, 1], [1000., 500.])

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
