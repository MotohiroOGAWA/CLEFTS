from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from clefts.libs.mmkit.mmkit import Adduct, Compound
from clefts.libs.msentity.msentity import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from clefts.ml.specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator


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
        help="Directory receiving predicted.msds and run information.",
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
            "Optional SourceAnchoredFragmentSpectrumGenerator parameter JSON. "
            "If omitted, model_config is read from the checkpoint."
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
        "--strict",
        action="store_true",
        help="Use strict state_dict loading.",
    )
    args = parser.parse_args()
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
        "with a SourceAnchoredFragmentSpectrumGenerator parameter JSON."
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
) -> SourceAnchoredFragmentSpectrumGenerator:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    params = model_params_from_checkpoint(checkpoint, params_path=params_path)
    generator = create_spectrum_generator(params).to(device)
    state_dict = extract_state_dict(checkpoint)

    # Training checkpoints contain the training wrapper (feature_model /
    # downstream_model), whose modules are shared with the generator. Load
    # through that wrapper so a checkpoint emitted by training is immediately
    # usable for inference.
    if any(key.startswith("downstream_model.") for key in state_dict):
        from ..training.fragment_tree_training.model import ActionFragmentTreeTrainingModel
        wrapper = ActionFragmentTreeTrainingModel(generator.feature_model, downstream_model=generator.post_model)
        wrapper.load_state_dict(state_dict, strict=strict)
    else:
        generator.load_state_dict(state_dict, strict=strict)
    return generator.eval()


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


def predict_msdataset(
    *,
    dataset: MSDataset,
    generator: SourceAnchoredFragmentSpectrumGenerator,
    model_path: str,
    db: str,
    spec_id_column: str,
    smiles_column: str,
    adduct_type_column: str,
    collision_energy_column: str,
    precursor_mz_column: str,
    instrument_column: Optional[str],
    include_formula_annotation: bool,
) -> tuple[MSDataset, pd.DataFrame]:
    """Predict every selected compound in one batched forward, then apply the
    shared provenance contract (SpecID/PredictionTool/... columns)."""
    from .source_action_msdataset import predict_source_msdataset
    predicted, failures = predict_source_msdataset(
        dataset, generator, smiles_column=smiles_column, adduct_type_column=adduct_type_column,
        collision_energy_column=collision_energy_column, precursor_mz_column=precursor_mz_column,
        instrument_column=instrument_column, include_formula_annotation=include_formula_annotation)
    timestamp = datetime.now(timezone.utc).isoformat()
    enriched_metadata = prediction_metadata(
        predicted.metadata, db=db, model_path=model_path, spec_id_column=spec_id_column,
        smiles_column=smiles_column, adduct_type_column=adduct_type_column,
        collision_energy_column=collision_energy_column, precursor_mz_column=precursor_mz_column,
        instrument_column=instrument_column, first_sequence=1, timestamp=timestamp)
    tags = list(dict.fromkeys([*predicted.tags, "predicted", "CLEFTS"]))
    attributes = {
        **predicted.attributes,
        "prediction_tool": "CLEFTS",
        "prediction_tool_version": _tool_version(),
        "prediction_model": Path(model_path).name,
        "prediction_db": str(db).strip() or "unspecified",
        "prediction_timestamp": timestamp,
    }
    enriched = MSDataset(
        spectrum_metadata=enriched_metadata, peak_series=predicted.peaks,
        description="In silico MS/MS spectra generated by CLEFTS",
        attributes=attributes, tags=tags,
    )
    validate_prediction_output(enriched)
    return enriched, failures


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
    print("[1/4] Loading input MSDataset")
    dataset = (
        MSDataset.load(args.input)
        if args.input is not None
        else direct_input_dataset(
            smiles_values=args.smiles,
            collision_energy=args.ce,
            adduct_type=args.adduct_type,
        )
    )
    print("[2/4] Validating input and selecting SMILES")
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

    print(f"input: {args.input or 'direct arguments'}")
    print(f"model: {args.model}")
    print(f"output directory: {output_dir}")
    print(f"output MSDataset: {output_path.name}")
    print(f"device: {device}")
    print(f"db: {args.db}")
    print(f"selected SMILES count: {len(smiles_values)}")

    started = time.time()

    print("[3/4] Loading prediction model")
    generator = load_generator(
        model_path=args.model,
        params_path=args.params,
        device=device,
        strict=args.strict,
    )

    print("[4/4] Predicting spectra")
    predicted, failures = predict_msdataset(
        dataset=dataset,
        generator=generator,
        model_path=args.model,
        db=args.db,
        smiles_column=args.smiles_column,
        precursor_mz_column=args.precursor_mz_column,
        adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,
        instrument_column=args.instrument_column,
        spec_id_column=args.spec_id_column,
        include_formula_annotation=not args.no_formula_annotation,
    )

    _atomic_save(predicted, output_path)

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
        "success_count": len(predicted),
        "failure_count": len(failures),
        "elapsed_seconds": time.time() - started,
        "arguments": vars(args),
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

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
