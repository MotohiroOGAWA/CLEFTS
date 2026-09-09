from __future__ import annotations

import pandas as pd
import pytest

from clefts.libs.msentity.msentity import MSDataset
from clefts.ml.specgen.predict_spectrum import (
    direct_input_dataset,
    make_precompute_batches,
    prediction_metadata,
    validate_prediction_input,
    validate_prediction_output,
)


def test_precompute_batches_assign_batch_size_compounds_per_worker_task() -> None:
    batches = make_precompute_batches([f"SMILES-{index}" for index in range(100)], 32)

    assert len(batches) == 4
    assert [len(batch) for batch in batches] == [32, 32, 32, 4]
    assert batches[0][0] == (0, "SMILES-0")
    assert batches[-1][-1] == (99, "SMILES-99")


def _dataset():
    dataset = direct_input_dataset(
        smiles_values=["CCO", "CCN"],
        collision_energy="20 eV",
        adduct_type="[M+H]+",
    )
    dataset["SpecID"] = ["source-1", "source-2"]
    dataset["DB"] = ["MoNA-source", "MoNA-source"]
    dataset["ExactMass"] = [46.0419, 45.0578]
    return dataset


def test_prediction_metadata_preserves_source_values_and_adds_provenance(tmp_path) -> None:
    dataset = _dataset()
    validate_prediction_input(
        dataset,
        smiles_column="SMILES",
        precursor_mz_column="PrecursorMZ",
        adduct_type_column="AdductType",
        collision_energy_column="CollisionEnergy",
        spec_id_column="SpecID",
        instrument_column=None,
    )
    original_precursor = dataset.metadata["PrecursorMZ"].copy()
    original_exact_mass = dataset.metadata["ExactMass"].copy()
    result = prediction_metadata(
        dataset.metadata,
        db="MoNA",
        model_path="/models/clefts-model.pt",
        spec_id_column="SpecID",
        smiles_column="SMILES",
        adduct_type_column="AdductType",
        collision_energy_column="CollisionEnergy",
        precursor_mz_column="PrecursorMZ",
        instrument_column=None,
        first_sequence=1,
        timestamp="2026-01-02T03:04:05+00:00",
    )

    assert result["SpecID"].tolist() == ["clefts-MoNA-000000001", "clefts-MoNA-000000002"]
    assert result["SourceSpecID"].tolist() == ["source-1", "source-2"]
    assert result["DB"].tolist() == ["MoNA", "MoNA"]
    assert result["SourceDB"].tolist() == ["MoNA-source", "MoNA-source"]
    assert result["PredictionTool"].tolist() == ["CLEFTS", "CLEFTS"]
    assert result["PredictionModel"].tolist() == ["clefts-model.pt", "clefts-model.pt"]
    assert result["PredictionCollisionEnergy"].tolist() == [20.0, 20.0]
    assert result["PredictionCollisionEnergyUnit"].tolist() == ["eV", "eV"]
    assert result["PredictionCollisionEnergyEV"].tolist() == [20.0, 20.0]
    pd.testing.assert_series_equal(result["PrecursorMZ"], original_precursor, check_names=False)
    pd.testing.assert_series_equal(result["ExactMass"], original_exact_mass, check_names=False)
    assert result["PredictedPrecursorMZ"].notna().all()

    output = MSDataset(
        spectrum_metadata=result,
        peak_series=dataset.peaks.copy(),
        description="Predicted spectra",
        attributes={"prediction_tool": "CLEFTS"},
        tags=["predicted", "CLEFTS"],
    )
    validate_prediction_output(output)
    output_path = tmp_path / "predicted.msds"
    output.save(str(output_path))
    restored = MSDataset.load(str(output_path))
    assert restored.metadata["SourceSpecID"].tolist() == ["source-1", "source-2"]
    assert restored.metadata["PredictionCollisionEnergyUnit"].tolist() == ["eV", "eV"]
    assert restored.attributes["prediction_tool"] == "CLEFTS"
    assert restored.tags == ["predicted", "CLEFTS"]


def test_prediction_input_rejects_duplicate_source_ids() -> None:
    dataset = _dataset()
    dataset["SpecID"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="must be unique"):
        validate_prediction_input(
            dataset,
            smiles_column="SMILES",
            precursor_mz_column="PrecursorMZ",
            adduct_type_column="AdductType",
            collision_energy_column="CollisionEnergy",
            spec_id_column="SpecID",
            instrument_column=None,
        )
