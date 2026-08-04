from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from clefts.libs.mmkit.mmkit import Adduct, Compound
from clefts.libs.msentity.msentity import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.ml.input.fragment_tree_structure import FragmentTreeStructure
from clefts.ml.input.single_fragment_tree_structure_builder import (
    SingleFragmentTreeStructureBuilder,
)
from clefts.ml.specgen.fragment_tree_spectrum_predictor import (
    FragmentSpectrumGenerator,
    fragment_spectrum_output_to_msdataset,
)
from clefts.ml.specgen.fragment_tree_training_model import FragmentTreeTrainingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict spectra from an MSDataset with a model file and save "
            "the predictions as an MSDataset."
        )
    )
    parser.add_argument("--input", help="Input MSDataset path.")
    parser.add_argument(
        "--output",
        default="predicted.msds",
        help="Output MSDataset path. Default: predicted.msds",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model file path. Supports raw state_dict files and checkpoints "
            "with model_state_dict or state_dict."
        ),
    )
    parser.add_argument(
        "--params",
        default=None,
        help=(
            "Optional FragmentSpectrumGenerator parameter JSON. If omitted, "
            "model_config is read from the checkpoint."
        ),
    )
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--precursor-mz-column", default="PrecursorMZ")
    parser.add_argument("--adduct-type-column", default="AdductType")
    parser.add_argument("--collision-energy-column", default="CollisionEnergy")
    parser.add_argument("--instrument-column", default=None)
    parser.add_argument(
        "--smiles",
        nargs="+",
        default=None,
        help=(
            "SMILES to predict. With --input, selects matching records; without "
            "--input, creates one prediction sample per SMILES."
        ),
    )
    parser.add_argument(
        "--ce",
        default=None,
        help="Collision energy for direct prediction, normally an eV number (for example 20).",
    )
    parser.add_argument(
        "--adduct-type",
        "--adduct",
        dest="adduct_type",
        default=None,
        help='AdductType for direct prediction (for example "[M+H]+").',
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--no-formula-annotation",
        action="store_true",
        help="Do not write formula strings into generated peak metadata.",
    )
    parser.add_argument(
        "--include-fragment-ion-annotation",
        action="store_true",
        help="Keep verbose fragment-ion annotations during generation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict state_dict loading.",
    )
    args = parser.parse_args()
    if args.input is None:
        missing = [
            name
            for name, value in (
                ("--smiles", args.smiles),
                ("--ce", args.ce),
                ("--adduct-type", args.adduct_type),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "direct prediction without --input requires " + ", ".join(missing)
            )
    return args


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_params_from_checkpoint(
    checkpoint: object,
    *,
    params_path: Optional[str],
) -> Dict[str, Any]:
    if params_path is not None:
        return load_json(params_path)

    if isinstance(checkpoint, dict):
        model_config = checkpoint.get("model_config")
        if isinstance(model_config, dict):
            params = model_config.get("params", model_config)
            if isinstance(params, dict):
                return params

    raise ValueError(
        "Model parameters were not found in the checkpoint. Provide --params "
        "with a FragmentSpectrumGenerator parameter JSON."
    )


def extract_state_dict(checkpoint: object) -> Dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not find a state_dict in --model. Expected a raw state_dict, "
        "or a checkpoint containing model_state_dict/state_dict."
    )


def load_generator(
    *,
    model_path: str,
    params_path: Optional[str],
    device: torch.device,
    strict: bool,
) -> FragmentSpectrumGenerator:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    params = model_params_from_checkpoint(checkpoint, params_path=params_path)
    generator = FragmentSpectrumGenerator(**params).to(device)
    state_dict = extract_state_dict(checkpoint)

    # Fragment-tree training checkpoints contain the training wrapper, whose
    # modules are shared with the generator.  Load through that wrapper so a
    # checkpoint emitted by training is immediately usable for inference.
    is_training_checkpoint = (
        any(key.startswith("candidate_selector.") for key in state_dict)
        and not any(key.startswith("feature_model.") for key in state_dict)
        and not any(key.startswith("spectrum_predictor.") for key in state_dict)
    )
    if is_training_checkpoint:
        training_model = FragmentTreeTrainingModel(
            generator.candidate_selector,
            intensity_predictor=generator.formula_intensity_predictor,
        ).to(device)
        incompatible = training_model.load_state_dict(state_dict, strict=strict)
    else:
        incompatible = generator.load_state_dict(state_dict, strict=strict)
    missing_keys, unexpected_keys = incompatible
    if missing_keys:
        print(f"missing keys while loading model: {len(missing_keys)}")
    if unexpected_keys:
        print(f"unexpected keys while loading model: {len(unexpected_keys)}")

    generator.eval()
    return generator


def direct_input_dataset(
    *,
    smiles_values: Sequence[str],
    collision_energy: str,
    adduct_type: str,
) -> MSDataset:
    """Create metadata-only samples for direct CLI prediction."""
    adduct = Adduct.parse(adduct_type)
    rows = []
    for smiles in smiles_values:
        compound = Compound.from_smiles(str(smiles))
        rows.append(
            {
                "SMILES": str(smiles),
                "PrecursorMZ": float(adduct.apply_to_mz(compound.exact_mass)),
                "AdductType": adduct_type,
                "CollisionEnergy": collision_energy,
            }
        )
    metadata = pd.DataFrame(rows)
    peaks = PeakSeries(
        data=np.empty((0, 2), dtype=np.float64),
        offsets=np.zeros(len(metadata) + 1, dtype=np.int64),
    )
    return MSDataset(
        spectrum_metadata=metadata,
        peak_series=peaks,
        columns=metadata.columns.tolist(),
        description="Direct spectrum prediction input",
    )


def select_smiles_values(
    dataset: MSDataset,
    *,
    smiles_column: str,
    smiles_values: Optional[Sequence[str]],
) -> List[str]:
    if smiles_values:
        return [str(value) for value in smiles_values]
    return [str(value) for value in dataset[smiles_column].dropna().unique()]


def record_metadata(
    record: object,
    *,
    source_group: str,
    source_record_index: int,
) -> Dict[str, Any]:
    metadata = {column: record[column] for column in record.columns}
    metadata["source_group_smiles"] = source_group
    metadata["source_record_index_in_group"] = int(source_record_index)
    return metadata


def build_structure_from_dataset(
    *,
    dataset: MSDataset,
    generator: FragmentSpectrumGenerator,
    smiles_values: Sequence[str],
    smiles_column: str,
    precursor_mz_column: str,
    adduct_type_column: str,
    collision_energy_column: str,
    instrument_column: Optional[str],
    device: torch.device,
) -> Tuple[FragmentTreeStructure, pd.DataFrame]:
    builder = SingleFragmentTreeStructureBuilder(generator.probability_model)
    structures: List[FragmentTreeStructure] = []
    metadata_by_sample_id: Dict[int, Dict[str, Any]] = {}
    sample_offset = 0

    for smiles in smiles_values:
        sub_dataset = dataset[dataset[smiles_column] == smiles]
        if len(sub_dataset) == 0:
            print(f"skip {smiles!r}: no records")
            continue

        builder.reset()
        sample_indexes = builder.add_same_smiles_dataset_first_cleavage(
            sub_dataset,
            precursor_mz_column=precursor_mz_column,
            adduct_type_column=adduct_type_column,
            collision_energy_column=collision_energy_column,
            smiles_column=smiles_column,
            instrument_column=instrument_column,
        )

        valid_pairs = [
            (record_index, int(sample_index))
            for record_index, sample_index in enumerate(sample_indexes.tolist())
            if int(sample_index) >= 0
        ]
        if not valid_pairs:
            print(f"skip {smiles!r}: builder produced no valid samples")
            continue

        structure = builder.to_structure()
        structures.append(structure)

        for record_index, local_sample_id in valid_pairs:
            metadata_by_sample_id[sample_offset + local_sample_id] = record_metadata(
                sub_dataset[record_index],
                source_group=smiles,
                source_record_index=record_index,
            )

        print(
            f"built {smiles!r}: records={len(sub_dataset)}, "
            f"valid_samples={structure.num_samples}, "
            f"nodes={structure.num_nodes}, edges={structure.num_edges}"
        )
        sample_offset += structure.num_samples

    if not structures:
        raise ValueError("No FragmentTreeStructure objects were built.")

    structure = FragmentTreeStructure.from_structures(structures, device=device)
    metadata_rows = []
    for sample_id in range(structure.num_samples):
        row = metadata_by_sample_id.get(sample_id, {}).copy()
        row["predicted_sample_id"] = int(sample_id)
        metadata_rows.append(row)

    return structure, pd.DataFrame(metadata_rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = (
        MSDataset.load(args.input)
        if args.input is not None
        else direct_input_dataset(
            smiles_values=args.smiles,
            collision_energy=args.ce,
            adduct_type=args.adduct_type,
        )
    )
    smiles_values = select_smiles_values(
        dataset,
        smiles_column=args.smiles_column,
        smiles_values=args.smiles,
    )

    print(f"input: {args.input or 'direct arguments'}")
    print(f"model: {args.model}")
    print(f"output: {args.output}")
    print(f"device: {device}")
    print(f"selected SMILES count: {len(smiles_values)}")

    generator = load_generator(
        model_path=args.model,
        params_path=args.params,
        device=device,
        strict=args.strict,
    )

    structure, metadata = build_structure_from_dataset(
        dataset=dataset,
        generator=generator,
        smiles_values=smiles_values,
        smiles_column=args.smiles_column,
        precursor_mz_column=args.precursor_mz_column,
        adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,
        instrument_column=args.instrument_column,
        device=device,
    )

    with torch.no_grad():
        output = generator(
            structure,
            include_formula_annotation=not args.no_formula_annotation,
            include_fragment_ion_annotation=args.include_fragment_ion_annotation,
        )

    output_sample_ids = [int(spectrum.sample_id) for spectrum in output.spectra]
    metadata_for_output = metadata.iloc[output_sample_ids].reset_index(drop=True)
    predicted = fragment_spectrum_output_to_msdataset(
        output,
        metadata=metadata_for_output,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted.save(str(output_path))

    print(
        f"predicted spectra: n_spectra={len(predicted)}, "
        f"n_peaks={predicted.n_peaks_total}"
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
