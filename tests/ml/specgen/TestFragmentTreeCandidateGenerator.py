from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from clefts.libs.mmkit.mmkit import Formula
from clefts.ml.specgen.fragment_tree_candidate_selector import FragmentIonCandidate
from clefts.ml.specgen.fragment_tree_formula_intensity_model import FormulaIntensityPredictor
from clefts.ml.specgen.fragment_tree_spectrum_predictor import fragment_spectrum_output_to_msdataset
from clefts.ml.training.fragment_tree_training.model import (
    FormulaGroupCoverageLoss,
    FragmentTreeIntensityTrainingLoss,
    FragmentTreeSelectionTrainingLoss,
)
from clefts.ml.specgen.fragment_tree_spectrum_predictor import (
    GeneratedMassSpectrum,
    GeneratedSpectrumPeak,
)


class TestFormulaGroupCoverageLoss(unittest.TestCase):
    def test_loss_is_finite_when_each_target_formula_has_candidate(self) -> None:
        loss_fn = FormulaGroupCoverageLoss()
        candidate_logit = torch.tensor([2.0, -1.0, 0.5])
        candidate_formula = torch.tensor(
            [
                [6.0, 6.0, 0.0],
                [7.0, 8.0, 0.0],
                [6.0, 6.0, 0.0],
            ]
        )
        target_formula = torch.tensor([[6.0, 6.0, 0.0]])

        loss = loss_fn(candidate_logit, candidate_formula, target_formula)

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.item()), 1.0)

    def test_loss_penalizes_missing_formula_group(self) -> None:
        loss_fn = FormulaGroupCoverageLoss()
        candidate_logit = torch.tensor([2.0])
        candidate_formula = torch.tensor([[6.0, 6.0, 0.0]])
        target_formula = torch.tensor([[7.0, 8.0, 0.0]])

        loss = loss_fn(candidate_logit, candidate_formula, target_formula)

        self.assertEqual(float(loss.item()), 32.0)


class TestFormulaIntensityPredictor(unittest.TestCase):
    def test_forward_returns_positive_intensity_per_formula(self) -> None:
        predictor = FormulaIntensityPredictor(formula_dim=3, hidden_dim=8)

        out = predictor(
            torch.tensor(
                [
                    [6.0, 6.0, 0.0],
                    [7.0, 8.0, 1.0],
                ]
            ),
            torch.tensor([0.8, 0.2]),
            torch.tensor([2.0, 1.0]),
        )

        self.assertEqual(out.shape, (2,))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out >= 0).all())

    def test_predict_from_candidates_groups_by_sample_and_formula(self) -> None:
        predictor = FormulaIntensityPredictor(formula_dim=3, hidden_dim=8)
        formula = Formula.parse("C6H6+")
        candidates = [
            FragmentIonCandidate(
                sample_id=0,
                batch_node_index=0,
                global_node_id=0,
                ion_index=0,
                unsaturation_index=0,
                radical_index=0,
                formula=formula,
                formula_tensor=torch.tensor([6.0, 6.0, 1.0]),
                score=1.0,
                keep_logit=0.5,
                candidate_logit=0.5,
                probability=0.7,
            ),
            FragmentIonCandidate(
                sample_id=0,
                batch_node_index=1,
                global_node_id=1,
                ion_index=0,
                unsaturation_index=0,
                radical_index=0,
                formula=formula,
                formula_tensor=torch.tensor([6.0, 6.0, 1.0]),
                score=2.0,
                keep_logit=1.0,
                candidate_logit=1.0,
                probability=0.8,
            ),
        ]

        output = predictor.predict_from_candidates(candidates)

        self.assertEqual(len(output.formula_predictions), 1)
        prediction = output.formula_predictions[0]
        self.assertEqual(prediction.sample_id, 0)
        self.assertEqual(len(prediction.candidates), 2)
        self.assertGreaterEqual(prediction.intensity, 0.0)


class TestFragmentSpectrumOutputToMSDataset(unittest.TestCase):
    def test_converts_generated_spectra_to_msdataset_with_peak_metadata(self) -> None:
        output = SimpleNamespace(
            spectra=[
                GeneratedMassSpectrum(
                    sample_id=10,
                    peaks=[
                        GeneratedSpectrumPeak(
                            mz=101.0,
                            intensity=0.5,
                            sample_id=10,
                            formula="C6H7+",
                        ),
                        GeneratedSpectrumPeak(
                            mz=50.0,
                            intensity=1.0,
                            sample_id=10,
                            formula="C3H3+",
                        ),
                    ],
                ),
                GeneratedMassSpectrum(
                    sample_id=20,
                    peaks=[],
                ),
            ]
        )
        metadata = pd.DataFrame({"source_id": ["a", "b"]})

        dataset = fragment_spectrum_output_to_msdataset(output, metadata=metadata)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.columns, ["source_id"])
        self.assertEqual(dataset[0].n_peaks, 2)
        self.assertEqual(dataset[1].n_peaks, 0)
        self.assertEqual(dataset[0].spectrum.mz.tolist(), [50.0, 101.0])
        self.assertEqual(dataset[0].spectrum.intensity.tolist(), [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()

class TestRelativeFormulaIntensity(unittest.TestCase):
    def test_peak_intensity_weight_has_order_and_minimum(self) -> None:
        loss = FragmentTreeSelectionTrainingLoss(intensity_alpha=0.2, intensity_gamma=0.5)
        weights = loss.intensity_weight(torch.tensor([0.0, 0.01, 1.0]))
        self.assertAlmostEqual(float(weights[0]), 0.2, places=6)
        self.assertGreater(float(weights[2]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_relative_intensity_presence_normalization_and_competition(self) -> None:
        sample_index = torch.tensor([0, 0, 1, 1])
        presence = torch.tensor([8.0, -8.0, 0.0, 0.0])
        abundance = torch.zeros(4)
        intensity = FormulaIntensityPredictor.relative_intensity(presence, abundance, sample_index)
        self.assertLess(float(intensity[1]), float(intensity[0]))
        self.assertAlmostEqual(float(intensity[:2].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(intensity[2:].sum()), 1.0, places=6)
        raised = FormulaIntensityPredictor.relative_intensity(torch.zeros(2), torch.tensor([2.0, 0.0]), torch.zeros(2, dtype=torch.long))
        baseline = FormulaIntensityPredictor.relative_intensity(torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.long))
        self.assertGreater(float(raised[0]), float(baseline[0]))
        self.assertLess(float(raised[1]), float(baseline[1]))

    def test_cosine_and_balanced_presence_losses(self) -> None:
        loss = FragmentTreeIntensityTrainingLoss()
        target = torch.tensor([0.8, 0.2])
        self.assertAlmostEqual(float(loss.cosine_loss(target, target)), 0.0, places=6)
        self.assertGreater(float(loss.cosine_loss(torch.tensor([0.2, 0.8]), target)), 0.0)
        labels = torch.tensor([1.0, 0.0])
        good = loss.balanced_presence_loss(torch.tensor([5.0, -5.0]), labels)
        bad = loss.balanced_presence_loss(torch.tensor([-5.0, 5.0]), labels)
        self.assertLess(float(good), float(bad))

    def test_formula_encoder_is_called_and_samples_are_isolated(self) -> None:
        class RecordingEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, value, src_key_padding_mask):
                self.batch_sizes.append(int(value.size(0)))
                valid = (~src_key_padding_mask)[:, :, None]
                return value + (value * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True)

        predictor = object.__new__(FormulaIntensityPredictor)
        torch.nn.Module.__init__(predictor)
        predictor.formula_node_encoder = RecordingEncoder()
        rows = torch.tensor([[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]])
        samples = torch.tensor([0, 0, 1])
        encoded = predictor._encode_formula_nodes_by_sample(rows, samples)
        self.assertEqual(predictor.formula_node_encoder.batch_sizes, [2])
        self.assertTrue(torch.equal(encoded[2], torch.tensor([200.0, 0.0])))

    def test_forward_candidate_output_uses_formula_encoder(self) -> None:
        class RecordingEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, value, src_key_padding_mask):
                self.batch_sizes.append(int(value.size(0)))
                return value

        predictor = object.__new__(FormulaIntensityPredictor)
        torch.nn.Module.__init__(predictor)
        predictor.formula_dim = 2
        predictor.formula_node_input = torch.nn.Identity()
        predictor.formula_node_encoder = RecordingEncoder()
        predictor.formula_presence_head = torch.nn.Linear(2, 1)
        predictor.formula_node_head = torch.nn.Linear(2, 1)
        predictor._candidate_formula_node_repr = lambda output, group: torch.stack([item.repr for item in group])
        candidates = [
            SimpleNamespace(sample_id=0, formula_tensor=torch.tensor([1.0, 0.0]), repr=torch.tensor([1.0, 0.0]), probability=0.8),
            SimpleNamespace(sample_id=0, formula_tensor=torch.tensor([2.0, 0.0]), repr=torch.tensor([2.0, 0.0]), probability=0.2),
            SimpleNamespace(sample_id=1, formula_tensor=torch.tensor([1.0, 0.0]), repr=torch.tensor([100.0, 0.0]), probability=1.0),
        ]
        output = predictor.forward_candidate_output(SimpleNamespace(kept_candidates=candidates, keep_logit=torch.zeros(3)))
        self.assertEqual(predictor.formula_node_encoder.batch_sizes, [2])
        self.assertEqual(output.presence_logit.shape, (3,))
        self.assertAlmostEqual(float(output.logit[:2].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(output.logit[2:].sum()), 1.0, places=6)

    def test_candidate_intensity_is_predicted_before_formula_merge(self) -> None:
        sample_index = torch.tensor([0, 0, 0])
        selection_probability = torch.tensor([0.8, 0.1, 0.1])
        intensity = FormulaIntensityPredictor.relative_intensity(
            torch.zeros(3),
            torch.zeros(3),
            sample_index,
            selection_probability=selection_probability,
        )
        self.assertTrue(torch.allclose(intensity, selection_probability))
        # If candidates 0 and 2 resolve to the same formula, aggregation is
        # performed only here, after their individual intensities exist.
        merged = torch.stack([intensity[[0, 2]].sum(), intensity[[1]].sum()])
        self.assertTrue(torch.allclose(merged, torch.tensor([0.9, 0.1])))

if __name__ == "__main__":
    unittest.main()
