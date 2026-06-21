from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from clefts.libs.mmkit.mmkit import Formula
from clefts.ml.specgen.fragment_tree_candidate_selector import FragmentIonCandidate
from clefts.ml.specgen.fragment_tree_formula_intensity_model import FormulaIntensityPredictor
from clefts.ml.specgen.fragment_tree_spectrum_predictor import fragment_spectrum_output_to_msdataset
from clefts.ml.specgen.fragment_tree_training_model import FormulaGroupCoverageLoss
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
