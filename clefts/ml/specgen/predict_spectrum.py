from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm

from clefts.libs.mmkit.mmkit import Adduct, Compound
from clefts.libs.msentity.msentity import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.ml.input.fragment_tree_structure import FragmentTreeStructure
from clefts.ml.input.single_fragment_tree_structure_builder import (
    SingleFragmentTreeStructureBuilder,
)
from clefts.ml.specgen.fragment_tree_spectrum_predictor import (
    FragmentSpectrumGenerator,
    fragment_spectrum_output_to_msdataset,
)
from clefts.ml.training.fragment_tree_training.model import FragmentTreeTrainingModel
from clefts.utils.parallel_subprocess import run_parallel_subprocesses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict spectra from an MSDataset with a model file and save "
            "the predictions as an MSDataset."
        )
    )
    parser.add_argument("--input", help="Input MSDataset path.")
    parser.add_argument(
        "--output-dir",
        default="prediction-output",
        help="Directory receiving predicted.msds, run information, and temporary caches.",
    )
    parser.add_argument(
        "--output-name",
        default="predicted.msds",
        help="Final MSDataset filename inside --output-dir. Default: predicted.msds",
    )
    parser.add_argument("--output", default=None, help=argparse.SUPPRESS)
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
    parser.add_argument("--spec-id-column", default="SpecID")
    parser.add_argument(
        "--db", default="unspecified",
        help="Database label written to DB and generated SpecID values.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Compounds processed serially by each parallel precompute worker task. Default: 32.",
    )
    parser.add_argument(
        "--precompute-workers", type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Threads used to precompute precursor/first-cleavage structures.",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep the first-cleavage cache directory after prediction.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacement of an existing output MSDataset.",
    )
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
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.precompute_workers <= 0:
        parser.error("--precompute-workers must be a positive integer")
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a filename, not a path")
    if not args.output_name.endswith(".msds"):
        args.output_name += ".msds"
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


def validate_prediction_input(
    dataset: MSDataset,
    *,
    smiles_column: str,
    precursor_mz_column: str,
    adduct_type_column: str,
    collision_energy_column: str,
    spec_id_column: str,
    instrument_column: Optional[str],
) -> None:
    required = [
        smiles_column, precursor_mz_column, adduct_type_column,
        collision_energy_column, spec_id_column,
    ]
    if instrument_column:
        required.append(instrument_column)
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Input MSDataset is missing required columns: {missing}")
    spec_ids = dataset[spec_id_column]
    if spec_ids.isna().any() or spec_ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{spec_id_column} must not contain null or empty values.")
    duplicated = spec_ids.astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = spec_ids.astype(str)[duplicated].drop_duplicates().head(5).tolist()
        raise ValueError(f"{spec_id_column} must be unique; duplicates include: {examples}")


def _tool_version() -> str:
    try:
        return version("clefts")
    except PackageNotFoundError:
        return "0.1.0"


def _id_component(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    return token or "unspecified"


def prediction_metadata(
    metadata: pd.DataFrame,
    *,
    db: str,
    model_path: str,
    spec_id_column: str,
    smiles_column: str,
    adduct_type_column: str,
    collision_energy_column: str,
    precursor_mz_column: str,
    instrument_column: Optional[str],
    first_sequence: int,
    timestamp: str,
) -> pd.DataFrame:
    """Apply the shared predicted-MSDataset provenance contract.

    Original metadata is retained. The immediate input identifier and DB are
    moved to SourceSpecID/SourceDB, while calculated values receive Predicted*
    names so measured PrecursorMZ and ExactMass are never overwritten.
    """
    result = metadata.reset_index(drop=True).copy()
    immediate_source_ids = result[spec_id_column].astype(str).copy()
    if "SourceSpecID" in result.columns and spec_id_column != "SourceSpecID":
        result.rename(columns={"SourceSpecID": "UpstreamSourceSpecID"}, inplace=True)
    if "SpecID" in result.columns and spec_id_column != "SpecID":
        result.rename(columns={"SpecID": "InputSpecID"}, inplace=True)
    result["SourceSpecID"] = immediate_source_ids
    if "DB" in result.columns:
        if "SourceDB" in result.columns:
            result.rename(columns={"SourceDB": "UpstreamSourceDB"}, inplace=True)
        result["SourceDB"] = result["DB"]

    db_label = str(db).strip() or "unspecified"
    safe_db = _id_component(db_label)
    result["SpecID"] = [
        f"clefts-{safe_db}-{sequence:09d}"
        for sequence in range(first_sequence, first_sequence + len(result))
    ]
    result["DB"] = db_label
    result["PredictionTool"] = "CLEFTS"
    result["PredictionToolVersion"] = _tool_version()
    result["PredictionModel"] = Path(model_path).name
    result["PredictionTimestamp"] = timestamp
    result["SpectrumType"] = "MS2"

    predicted_precursors: list[float] = []
    collision_energy_ev: list[float] = []
    for _, row in result.iterrows():
        try:
            exact_mass = Compound.from_smiles(str(row[smiles_column])).exact_mass
            predicted_mz = float(Adduct.parse(str(row[adduct_type_column])).apply_to_mz(exact_mass))
        except Exception:
            predicted_mz = float("nan")
        predicted_precursors.append(predicted_mz)
        instrument = row.get(instrument_column) if instrument_column else None
        parsed_ce = parse_ce_to_ev(row.get(collision_energy_column), row.get(precursor_mz_column), instrument)
        collision_energy_ev.append(float(parsed_ce) if parsed_ce is not None else float("nan"))
    result["PredictedPrecursorMZ"] = predicted_precursors
    result["PredictionCollisionEnergy"] = collision_energy_ev
    result["PredictionCollisionEnergyUnit"] = "eV"
    result["PredictionCollisionEnergyEV"] = collision_energy_ev
    result["PredictionAdductType"] = result[adduct_type_column].astype(str)
    return result


def validate_prediction_output(dataset: MSDataset) -> None:
    required = [
        "SpecID", "SourceSpecID", "DB", "PredictionTool",
        "PredictionToolVersion", "PredictionModel", "PredictionTimestamp",
        "PredictedPrecursorMZ", "PredictionCollisionEnergy",
        "PredictionCollisionEnergyUnit", "PredictionAdductType",
    ]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Predicted MSDataset is missing required columns: {missing}")
    spec_ids = dataset["SpecID"].astype(str)
    if spec_ids.str.strip().eq("").any() or spec_ids.duplicated().any():
        raise ValueError("Predicted SpecID values must be non-empty and unique.")
    if dataset["SourceSpecID"].isna().any() or dataset["SourceSpecID"].astype(str).str.strip().eq("").any():
        raise ValueError("Predicted SourceSpecID values must be non-empty.")


def _chunks(values: Sequence[str], size: int) -> Sequence[Sequence[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def make_precompute_batches(
    smiles_values: Sequence[str], batch_size: int
) -> list[list[tuple[int, str]]]:
    """Assign one fixed-size compound batch to each parallel worker task."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indexed = list(enumerate(str(value) for value in smiles_values))
    return [
        indexed[start:start + batch_size]
        for start in range(0, len(indexed), batch_size)
    ]


def record_metadata(
    record: object,
    *,
    source_group: str,
    source_record_index: int,
) -> Dict[str, Any]:
    del source_group, source_record_index
    return {column: record[column] for column in record.columns}


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

    with tqdm(
        smiles_values, desc="Building structures", unit="SMILES",
        leave=False, dynamic_ncols=True,
    ) as structure_progress:
        for smiles in structure_progress:
            structure_progress.set_postfix_str(str(smiles)[:36], refresh=False)
            sub_dataset = dataset[dataset[smiles_column] == smiles]
            if len(sub_dataset) == 0:
                tqdm.write(f"skip {smiles!r}: no records")
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
                tqdm.write(f"skip {smiles!r}: builder produced no valid samples")
                continue

            structure = builder.to_structure()
            structures.append(structure)

            for record_index, local_sample_id in valid_pairs:
                metadata_by_sample_id[sample_offset + local_sample_id] = record_metadata(
                    sub_dataset[record_index],
                    source_group=smiles,
                    source_record_index=record_index,
                )

            tqdm.write(
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
        metadata_rows.append(metadata_by_sample_id.get(sample_id, {}).copy())

    return structure, pd.DataFrame(metadata_rows)


def precompute_first_cleavage_caches(
    *,
    dataset: MSDataset,
    generator: FragmentSpectrumGenerator,
    smiles_values: Sequence[str],
    cache_dir: Path,
    workers: int,
    batch_size: int,
    smiles_column: str,
    precursor_mz_column: str,
    adduct_type_column: str,
    collision_energy_column: str,
    instrument_column: Optional[str],
    spec_id_column: str,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Build caches in independent batch subprocesses, as Data Preparation does."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    batches = make_precompute_batches(smiles_values, batch_size)
    context_path = cache_dir / "preprocessing-context.json"
    probability_model = generator.probability_model
    context_path.write_text(json.dumps({
        "symbols": list(probability_model.mol_encoder.symbols),
        "fragmenter_params": probability_model.fragmenter.to_dict(),
        "max_node": -1,
        "max_edge": -1,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    commands: list[list[str]] = []
    result_paths: list[Path] = []
    for batch_index, batch in enumerate(batches):
        batch_smiles = {smiles for _, smiles in batch}
        part_input = cache_dir / f"part-{batch_index:06d}.msds"
        task_path = cache_dir / f"part-{batch_index:06d}.tasks.json"
        result_path = cache_dir / f"part-{batch_index:06d}.result.json"
        dataset[dataset[smiles_column].astype(str).isin(batch_smiles)].copy().save(
            str(part_input)
        )
        task_path.write_text(json.dumps([
            {"index": index, "smiles": smiles} for index, smiles in batch
        ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable, "-m",
            "clefts.ml.specgen.precompute_first_cleavage_worker",
            "--input", str(part_input),
            "--task-json", str(task_path),
            "--context-json", str(context_path),
            "--cache-dir", str(cache_dir),
            "--result-json", str(result_path),
            "--smiles-column", smiles_column,
            "--precursor-mz-column", precursor_mz_column,
            "--adduct-type-column", adduct_type_column,
            "--collision-energy-column", collision_energy_column,
            "--spec-id-column", spec_id_column,
        ]
        if instrument_column is not None:
            command.extend(["--instrument-column", instrument_column])
        commands.append(command)
        result_paths.append(result_path)

    print(
        f"first-cleavage subprocesses: {len(commands)} batches, "
        f"up to {min(workers, len(commands))} concurrent, "
        f"{batch_size} compounds per batch"
    )
    run_parallel_subprocesses(
        commands_list=commands,
        max_workers=workers,
        print_output=False,
        desc="Precomputing first-cleavage subprocess batches",
        unit="batch",
    )

    completed: list[tuple[int, Path]] = []
    failures: list[dict[str, str]] = []
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completed.extend(
            (int(row["index"]), Path(row["path"]))
            for row in result.get("completed", [])
        )
        failures.extend(result.get("failures", []))
    completed.sort(key=lambda item: item[0])
    return [path for _, path in completed], failures


def predict_msdataset(
    *,
    dataset: MSDataset,
    generator: FragmentSpectrumGenerator,
    model_path: str,
    db: str,
    batch_size: int,
    cache_paths: Sequence[Path],
    precompute_failures: Sequence[dict[str, str]],
    spec_id_column: str,
    smiles_column: str,
    adduct_type_column: str,
    collision_energy_column: str,
    precursor_mz_column: str,
    instrument_column: Optional[str],
    device: torch.device,
    include_formula_annotation: bool,
    include_fragment_ion_annotation: bool,
) -> tuple[MSDataset, pd.DataFrame]:
    """Finish each cached compound, expanding only model-selected fragments."""
    predicted_parts: list[MSDataset] = []
    failures = [dict(row) for row in precompute_failures]
    next_sequence = 1
    timestamp = datetime.now(timezone.utc).isoformat()

    with tqdm(
        total=len(cache_paths), desc="Completing spectrum predictions", unit="compound",
        mininterval=1.0, dynamic_ncols=True,
    ) as prediction_progress:
        for cache_batch in _chunks(list(cache_paths), batch_size):
            for cache_path in cache_batch:
                metadata = pd.DataFrame()
                try:
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if cached.get("schema") != "clefts.first-cleavage-cache":
                        raise ValueError(f"Invalid first-cleavage cache: {cache_path}")
                    metadata = cached["metadata"].reset_index(drop=True)
                    builder = SingleFragmentTreeStructureBuilder.from_state(
                        generator.probability_model, cached["builder_state"]
                    )
                    structure = builder.to_structure().to(device)

                    def expand_selected(
                        depth: int, candidate_output: object
                    ) -> FragmentTreeStructure:
                        candidates = getattr(
                            candidate_output, "next_cleavage_candidates", ()
                        )
                        pairs = [
                            (int(candidate.sample_id), int(candidate.global_node_id))
                            for candidate in candidates
                        ]
                        prediction_progress.set_postfix(
                            phase=f"depth {depth}: {len(pairs)} selected nodes",
                            spectra=next_sequence - 1,
                            failures=len(failures),
                            refresh=False,
                        )
                        builder.add_selected_node_cleavages(pairs)
                        return builder.to_structure().to(device)

                    with torch.no_grad():
                        prediction_progress.set_postfix(
                            phase="model inference",
                            spectra=next_sequence - 1,
                            failures=len(failures),
                            refresh=False,
                        )
                        output = generator(
                            structure,
                            include_formula_annotation=include_formula_annotation,
                            include_fragment_ion_annotation=include_fragment_ion_annotation,
                            structure_expander=expand_selected,
                        )
                    output_sample_ids = [
                        int(spectrum.sample_id) for spectrum in output.spectra
                    ]
                    raw_metadata = metadata.iloc[output_sample_ids].reset_index(drop=True)
                    enriched = prediction_metadata(
                        raw_metadata,
                        db=db,
                        model_path=model_path,
                        spec_id_column=spec_id_column,
                        smiles_column=smiles_column,
                        adduct_type_column=adduct_type_column,
                        collision_energy_column=collision_energy_column,
                        precursor_mz_column=precursor_mz_column,
                        instrument_column=instrument_column,
                        first_sequence=next_sequence,
                        timestamp=timestamp,
                    )
                    predicted_part = fragment_spectrum_output_to_msdataset(
                        output, metadata=enriched
                    )
                    peak_metadata = predicted_part.peaks._metadata_ref
                    if peak_metadata is not None and "sample_id" in peak_metadata.columns:
                        peak_metadata["sample_id"] = (
                            peak_metadata["sample_id"].astype(int) + next_sequence - 1
                        )
                    predicted_parts.append(predicted_part)
                    next_sequence += len(enriched)
                except Exception as exc:
                    source_ids = (
                        metadata[spec_id_column].astype(str).tolist()
                        if spec_id_column in metadata.columns else [cache_path.name]
                    )
                    for source_id in source_ids:
                        failures.append({
                            "SourceSpecID": source_id,
                            "FailureType": type(exc).__name__,
                            "FailureMessage": str(exc),
                        })
                    tqdm.write(
                        f"failed cache {cache_path.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    prediction_progress.update(1)
                    prediction_progress.set_postfix(
                        phase="compound complete",
                        spectra=next_sequence - 1,
                        failures=len(failures),
                        refresh=False,
                    )

    if not predicted_parts:
        raise ValueError("No spectra were predicted successfully.")
    tags = list(dict.fromkeys([*dataset.tags, "predicted", "CLEFTS"]))
    attributes = {
        **dataset.attributes,
        "prediction_tool": "CLEFTS",
        "prediction_tool_version": _tool_version(),
        "prediction_model": Path(model_path).name,
        "prediction_db": str(db).strip() or "unspecified",
        "prediction_timestamp": timestamp,
    }
    predicted = MSDataset.concat(
        predicted_parts,
        description="In silico MS/MS spectra generated by CLEFTS",
        attributes=attributes,
        tags=tags,
    )
    validate_prediction_output(predicted)
    return predicted, pd.DataFrame(
        failures, columns=["SourceSpecID", "FailureType", "FailureMessage"]
    )


def _atomic_save(dataset: MSDataset, output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        dataset.save(str(temporary))
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    overall = tqdm(
        total=7, desc="[1/7] Checking paths", unit="stage",
        mininterval=1.0, dynamic_ncols=True,
    )

    if args.output:
        legacy_output = Path(args.output).resolve()
        output_dir = legacy_output.parent
        output_path = legacy_output
        print("warning: --output is deprecated; use --output-dir and --output-name.", file=sys.stderr)
    else:
        output_dir = Path(args.output_dir).resolve()
        output_path = output_dir / args.output_name
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite: {output_path}")
    if args.input is not None and Path(args.input).resolve() == output_path:
        raise ValueError("The input and final output MSDataset must be different paths.")

    output_dir.mkdir(parents=True, exist_ok=True)
    overall.set_description("[1/7] Loading input MSDataset")
    dataset = (
        MSDataset.load(args.input)
        if args.input is not None
        else direct_input_dataset(
            smiles_values=args.smiles,
            collision_energy=args.ce,
            adduct_type=args.adduct_type,
        )
    )
    overall.update(1)
    overall.set_description("[2/7] Validating input and selecting SMILES")
    if args.input is None and args.spec_id_column not in dataset.columns:
        dataset[args.spec_id_column] = [f"direct-{index + 1:09d}" for index in range(len(dataset))]
    validate_prediction_input(
        dataset,
        smiles_column=args.smiles_column,
        precursor_mz_column=args.precursor_mz_column,
        adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,
        spec_id_column=args.spec_id_column,
        instrument_column=args.instrument_column,
    )
    smiles_values = select_smiles_values(
        dataset,
        smiles_column=args.smiles_column,
        smiles_values=args.smiles,
    )
    overall.update(1)

    print(f"input: {args.input or 'direct arguments'}")
    print(f"model: {args.model}")
    print(f"output directory: {output_dir}")
    print(f"output MSDataset: {output_path.name}")
    print(f"device: {device}")
    print(f"db: {args.db}")
    print(f"chemical precompute batch size: {args.batch_size}")
    print(f"first-cleavage precompute workers: {args.precompute_workers}")
    print(f"selected SMILES count: {len(smiles_values)}")

    started = time.time()

    overall.set_description("[3/7] Loading prediction model")
    generator = load_generator(
        model_path=args.model,
        params_path=args.params,
        device=device,
        strict=args.strict,
    )
    overall.update(1)

    cache_dir = Path(tempfile.mkdtemp(
        prefix=".first-cleavage-cache-",
        dir=output_dir,
    ))
    try:
        overall.set_description("[4/7] Precomputing first-cleavage structures")
        cache_paths, precompute_failures = precompute_first_cleavage_caches(
            dataset=dataset,
            generator=generator,
            smiles_values=smiles_values,
            cache_dir=cache_dir,
            workers=args.precompute_workers,
            batch_size=args.batch_size,
            smiles_column=args.smiles_column,
            precursor_mz_column=args.precursor_mz_column,
            adduct_type_column=args.adduct_type_column,
            collision_energy_column=args.collision_energy_column,
            instrument_column=args.instrument_column,
            spec_id_column=args.spec_id_column,
        )
        overall.update(1)

        overall.set_description("[5/7] Predicting and expanding selected fragments")
        predicted, failures = predict_msdataset(
            dataset=dataset,
            generator=generator,
            model_path=args.model,
            db=args.db,
            batch_size=args.batch_size,
            cache_paths=cache_paths,
            precompute_failures=precompute_failures,
            smiles_column=args.smiles_column,
            precursor_mz_column=args.precursor_mz_column,
            adduct_type_column=args.adduct_type_column,
            collision_energy_column=args.collision_energy_column,
            instrument_column=args.instrument_column,
            spec_id_column=args.spec_id_column,
            device=device,
            include_formula_annotation=not args.no_formula_annotation,
            include_fragment_ion_annotation=args.include_fragment_ion_annotation,
        )
        overall.update(1)

        overall.set_description("[6/7] Saving predicted MSDataset")
        _atomic_save(predicted, output_path)
        overall.update(1)

        overall.set_description("[7/7] Writing run information")
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        failures.to_csv(run_dir / "failures.tsv", sep="\t", index=False)
        run_info = {
            "tool": "CLEFTS",
            "tool_version": _tool_version(),
            "model": str(Path(args.model).resolve()),
            "input": str(Path(args.input).resolve()) if args.input else None,
            "output": str(output_path),
            "output_directory": str(output_dir),
            "db": args.db,
            "device": str(device),
            "batch_size": args.batch_size,
            "precompute_workers": args.precompute_workers,
            "first_cleavage_cache_count": len(cache_paths),
            "temporary_cache_directory": str(cache_dir),
            "temporary_cache_kept": bool(args.keep_temp),
            "success_count": len(predicted),
            "failure_count": len(failures),
            "elapsed_seconds": time.time() - started,
            "arguments": vars(args),
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        overall.update(1)
        overall.set_description("[7/7] Prediction complete")
    finally:
        overall.close()
        if args.keep_temp:
            print(f"kept first-cleavage cache: {cache_dir}")
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)

    print(
        f"predicted spectra: n_spectra={len(predicted)}, "
        f"n_peaks={predicted.n_peaks_total}"
    )
    print(f"saved: {output_path}")
    print(f"run information: {run_dir}")
    if len(failures):
        print(f"partial success: {len(failures)} record(s) failed", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
