from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from clefts.ml.specgen.fragment_tree_candidate_generator import (
    FormulaGroupCoverageLoss,
    FormulaIntensityPredictor,
    fragment_spectrum_output_to_msdataset,
)
from clefts.ml.specgen.fragment_spectrum_generator import (
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
        )

        self.assertEqual(out.shape, (2,))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out >= 0).all())


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
