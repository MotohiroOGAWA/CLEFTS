from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from clefts.libs.msentity.msentity import MSDataset
from clefts.utils.parallel_subprocess import run_parallel_subprocesses

ORIGINAL_INDEX_COLUMN = "__fragment_tree_original_index"
STRUCTURE_DATA_DIR_NAME = "data"
DEFAULT_PROJECT_MODEL_CONFIG_NAME = "model_config.json"

try:
    from .fragment_tree_training_data import (
        build_fragment_tree_structure_files,
        group_record_indexes_by_smiles,
        load_fragment_tree_structure_file,
    )
    from ..specgen.fragment_tree_spectrum_predictor import FragmentSpectrumGenerator
except ImportError:
    from clefts.ml.input.fragment_tree_training_data import (
        build_fragment_tree_structure_files,
        group_record_indexes_by_smiles,
        load_fragment_tree_structure_file,
    )
    from clefts.ml.specgen.fragment_tree_spectrum_predictor import FragmentSpectrumGenerator


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build training FragmentTreeStructure files from an MSDataset. "
            "Records are grouped by SMILES, and each SMILES group is saved as one .pt file."
        )
    )
    parser.add_argument(
        "--train-input",
        default="data/raw/NIST/NIST23/MSMS-Pos-NIST23_v20_mini.msds",
        help="Training input MSDataset path.",
    )
    parser.add_argument(
        "--validation-input",
        default=None,
        help="Optional validation input MSDataset path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/test/fragment_tree_training_structures",
        help=(
            "Output root directory. Structure .pt files are written under "
            "train_structures/data, and validation .pt files under "
            "validation_structures/data when --validation-input is provided."
        ),
    )
    parser.add_argument(
        "--params",
        default="clefts/ml/specgen/presets/fragment_spectrum_generator_param.json",
        help="FragmentSpectrumGenerator parameter JSON used to construct the feature model.",
    )
    parser.add_argument(
        "--model-config-output",
        default=None,
        help=(
            "Path to copy the model config after the model is constructed. "
            "Defaults to OUTPUT_DIR/config/model_config.json."
        ),
    )
    parser.add_argument(
        "--overwrite-model-config",
        action="store_true",
        help="Overwrite an existing copied model config without prompting.",
    )
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--precursor-mz-column", default="PrecursorMZ")
    parser.add_argument("--adduct-type-column", default="AdductType")
    parser.add_argument("--collision-energy-column", default="CollisionEnergy")
    parser.add_argument("--instrument-column", default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model construction while building structures.",
    )
    parser.add_argument(
        "--max-node",
        type=int,
        default=-1,
        help="Maximum number of nodes when building fragment_ion_tree. Use -1 for no limit.",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=-1,
        help="Maximum number of edges when building fragment_ion_tree. Use -1 for no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt structure files.",
    )
    parser.add_argument(
        "--save-train-valid-records",
        action="store_true",
        help="Save valid training records as an MSDataset.",
    )
    parser.add_argument(
        "--no-save-validation-valid-records",
        dest="save_validation_valid_records",
        action="store_false",
        help="Do not save valid validation records as an MSDataset.",
    )
    parser.set_defaults(save_validation_valid_records=True)
    parser.add_argument(
        "--train-valid-output",
        default=None,
        help="Output .msds path for valid training records.",
    )
    parser.add_argument(
        "--validation-valid-output",
        default=None,
        help="Output .msds path for valid validation records.",
    )
    parser.add_argument(
        "--train-assignment-score-output",
        default=None,
        help="Output TSV path for training peak assignment scores.",
    )
    parser.add_argument(
        "--validation-assignment-score-output",
        default=None,
        help="Output TSV path for validation peak assignment scores.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel subprocess workers. Use 1 to disable parallel processing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of SMILES groups per parallel chunk.",
    )
    parser.add_argument(
        "--parallel-temp-dir",
        default=None,
        help="Temporary directory for parallel chunk inputs and outputs.",
    )
    parser.add_argument(
        "--keep-parallel-temp",
        action="store_true",
        help="Keep parallel temporary files after merging.",
    )
    parser.add_argument(
        "--manifest-file",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--structure-output-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--valid-records-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--assignment-score-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-valid-records",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def load_generator(params_path: str, device: torch.device) -> FragmentSpectrumGenerator:
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    generator = FragmentSpectrumGenerator(**params).to(device)
    generator.eval()
    return generator


def default_model_config_output(output_root: str | Path) -> Path:
    return Path(output_root) / "config" / DEFAULT_PROJECT_MODEL_CONFIG_NAME


def copy_model_config_after_model_creation(
    *,
    source_file: str | Path,
    output_file: str | Path,
    overwrite: bool = False,
) -> None:
    source_path = Path(source_file)
    output_path = Path(output_file)

    if output_path.exists() and not overwrite:
        answer = input(f"Model config already exists: {output_path}. Overwrite? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print(f"kept existing model config: {output_path}")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, output_path)
    print(f"copied model config: {source_path} -> {output_path}")


def structure_data_dir(structure_dir: str | Path) -> Path:
    return Path(structure_dir) / STRUCTURE_DATA_DIR_NAME


def save_valid_records(
    *,
    dataset: MSDataset,
    valid_record_indexes: list[int],
    output_file: str | Path,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_indexes = list(dict.fromkeys(int(index) for index in valid_record_indexes))
    dataset[unique_indexes].save(str(output_path))
    print(f"saved valid records: {len(unique_indexes)} -> {output_path}")


def _metadata_value(row: pd.Series, column: str) -> object:
    if column not in row.index:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    return value


def _assigned_peak_indexes_by_sample(structure) -> dict[int, set[int]]:
    sample_to_peak_indexes: dict[int, set[int]] = {}
    if not hasattr(structure, "target_sample_index") or structure.target_sample_index.numel() == 0:
        return sample_to_peak_indexes

    sample_indexes = structure.target_sample_index.detach().cpu().tolist()
    peak_indexes = structure.target_peak_index.detach().cpu().tolist()
    for sample_index, peak_index in zip(sample_indexes, peak_indexes):
        sample_to_peak_indexes.setdefault(int(sample_index), set()).add(int(peak_index))
    return sample_to_peak_indexes


def write_assignment_score_tsv(
    *,
    dataset: MSDataset,
    structure_files: list[Path],
    output_file: str | Path,
    smiles_column: str,
) -> None:
    rows: list[dict[str, object]] = []
    metadata = dataset.metadata

    for structure_file in tqdm(
        structure_files,
        desc="Writing assignment score TSV",
        mininterval=1.0,
    ):
        if not structure_file.exists():
            print(
                f"[WARN] Missing structure file while writing assignment scores: {structure_file}",
                file=sys.stderr,
            )
            continue

        item = load_fragment_tree_structure_file(structure_file, map_location="cpu")
        structure = item.structure
        record_indexes = [int(index) for index in item.metadata.get("record_indexes", [])]
        sample_indexes = [int(index) for index in item.metadata.get("sample_indexes", [])]
        assigned_by_sample = _assigned_peak_indexes_by_sample(structure)

        for local_record_index, sample_index in zip(record_indexes, sample_indexes):
            if int(sample_index) < 0:
                continue
            if local_record_index < 0 or local_record_index >= len(dataset):
                continue

            row = metadata.iloc[local_record_index]
            spectrum = dataset[local_record_index]
            intensities = np.asarray(
                [float(peak.intensity) for peak in spectrum.peaks],
                dtype=np.float64,
            )
            finite_mask = np.isfinite(intensities)
            total_intensity = float(intensities[finite_mask].sum()) if intensities.size else 0.0
            total_peak_count = int(intensities.size)

            assigned_peak_indexes = sorted(
                peak_index
                for peak_index in assigned_by_sample.get(int(sample_index), set())
                if 0 <= int(peak_index) < total_peak_count
            )
            assigned_intensity = float(intensities[assigned_peak_indexes].sum()) if assigned_peak_indexes else 0.0
            assignment_score = assigned_intensity / total_intensity if total_intensity > 0 else 0.0
            source_index = _metadata_value(row, ORIGINAL_INDEX_COLUMN)
            if source_index == "":
                source_index = int(local_record_index)

            rows.append(
                {
                    "index": int(source_index),
                    "SpecID": _metadata_value(row, "SpecID"),
                    "assignment_score": assignment_score,
                    "assigned_intensity": assigned_intensity,
                    "total_intensity": total_intensity,
                    "assigned_peak_count": int(len(assigned_peak_indexes)),
                    "total_peak_count": total_peak_count,
                    "smiles": _metadata_value(row, smiles_column),
                    "structure_file": Path(structure_file).name,
                }
            )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "index",
        "SpecID",
        "assignment_score",
        "assigned_intensity",
        "total_intensity",
        "assigned_peak_count",
        "total_peak_count",
        "smiles",
        "structure_file",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, sep="\t", index=False)
    print(f"saved assignment scores: {len(rows)} -> {output_path}")


def build_structure_files_for_input(
    *,
    input_path: str,
    output_dir: str | Path,
    args: argparse.Namespace,
    generator: FragmentSpectrumGenerator,
    manifest_file: str | Path | None = None,
    save_valid: bool = False,
    valid_records_output: str | Path | None = None,
    assignment_score_output: str | Path | None = None,
) -> list[Path]:
    dataset = MSDataset.load(input_path)
    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")

    valid_record_indexes: list[int] = []
    saved_files = build_fragment_tree_structure_files(
        dataset=dataset,
        feature_model=generator.feature_model,
        output_dir=output_dir,
        smiles_column=args.smiles_column,
        precursor_mz_column=args.precursor_mz_column,
        adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,
        instrument_column=args.instrument_column,
        max_node=args.max_node,
        max_edge=args.max_edge,
        overwrite=args.overwrite,
        manifest_file=manifest_file,
        valid_record_indexes=valid_record_indexes if save_valid else None,
    )

    if save_valid:
        if valid_records_output is None:
            raise ValueError("valid_records_output is required when save_valid=True.")
        save_valid_records(
            dataset=dataset,
            valid_record_indexes=valid_record_indexes,
            output_file=valid_records_output,
        )

    if assignment_score_output is not None:
        write_assignment_score_tsv(
            dataset=dataset,
            structure_files=saved_files,
            output_file=assignment_score_output,
            smiles_column=args.smiles_column,
        )

    print(f"saved structure files: {len(saved_files)}")
    for path in saved_files[:5]:
        print(f"  {path}")

    return saved_files


def make_smiles_chunks(
    dataset: MSDataset,
    *,
    smiles_column: str,
    chunk_size: int,
) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    groups = group_record_indexes_by_smiles(dataset, smiles_column=smiles_column)
    items = list(groups.items())
    chunks: list[list[int]] = []
    for start in range(0, len(items), chunk_size):
        chunk_items = items[start : start + chunk_size]
        chunks.append([index for _, indexes in chunk_items for index in indexes])
    return chunks


def merge_tsv_files(input_files: list[Path], output_file: Path) -> None:
    frames = [pd.read_csv(path, sep="\t") for path in input_files if path.exists()]
    if not frames:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output_file, sep="\t", index=False)


def merge_msdatasets(input_files: list[Path], output_file: Path, *, empty_like: MSDataset) -> None:
    datasets = [MSDataset.load(str(path)) for path in input_files if path.exists()]
    datasets = [dataset for dataset in datasets if len(dataset) > 0]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if datasets:
        MSDataset.concat(datasets, description=f"Valid records merged for {output_file.name}").save(str(output_file))
    else:
        empty_like[[]].save(str(output_file))
    print(f"saved merged valid records: {output_file}")


def run_parallel_for_input(
    *,
    input_path: str,
    structure_output_dir: Path,
    args: argparse.Namespace,
    manifest_file: Path,
    save_valid: bool,
    valid_records_output: Optional[Path],
    assignment_score_output: Path,
    split_name: str,
) -> None:
    if args.num_workers <= 1:
        raise ValueError("num_workers must be greater than 1 for parallel processing.")

    dataset = MSDataset.load(input_path)
    chunks = make_smiles_chunks(
        dataset,
        smiles_column=args.smiles_column,
        chunk_size=args.chunk_size,
    )
    if not chunks:
        raise RuntimeError(f"No SMILES groups were found for {split_name}.")

    temp_root = parallel_temp_dir(args, Path(args.output_dir), split_name)
    temp_root.mkdir(parents=True, exist_ok=True)

    print(
        f"{split_name}: split into {len(chunks)} chunks "
        f"({args.chunk_size} SMILES groups per chunk)."
    )

    commands: list[list[str]] = []
    part_manifests: list[Path] = []
    part_valid_outputs: list[Path] = []
    part_score_outputs: list[Path] = []
    module_name = "clefts.ml.input.create_fragment_tree_training_data"

    for chunk_index, record_indexes in enumerate(
        tqdm(chunks, desc=f"Preparing {split_name} chunks", mininterval=1.0)
    ):
        temp_input = temp_root / f"part_{chunk_index:06d}.msds"
        temp_manifest = temp_root / f"part_{chunk_index:06d}_manifest.tsv"
        temp_valid = temp_root / f"part_{chunk_index:06d}_valid.msds"
        temp_score = temp_root / f"part_{chunk_index:06d}_assignment_scores.tsv"
        chunk_dataset = dataset[record_indexes].copy()
        chunk_dataset[ORIGINAL_INDEX_COLUMN] = list(record_indexes)
        chunk_dataset.save(str(temp_input))

        command = [
            sys.executable,
            "-m",
            module_name,
            "--train-input",
            str(temp_input),
            "--output-dir",
            str(args.output_dir),
            "--structure-output-dir",
            str(structure_output_dir),
            "--params",
            str(args.params),
            "--smiles-column",
            str(args.smiles_column),
            "--precursor-mz-column",
            str(args.precursor_mz_column),
            "--adduct-type-column",
            str(args.adduct_type_column),
            "--collision-energy-column",
            str(args.collision_energy_column),
            "--device",
            str(args.device),
            "--max-node",
            str(args.max_node),
            "--max-edge",
            str(args.max_edge),
            "--manifest-file",
            str(temp_manifest),
            "--num-workers",
            "1",
            "--assignment-score-output",
            str(temp_score),
        ]
        part_score_outputs.append(temp_score)
        if args.instrument_column is not None:
            command.extend(["--instrument-column", str(args.instrument_column)])
        if save_valid:
            command.extend(
                [
                    "--save-valid-records",
                    "--valid-records-output",
                    str(temp_valid),
                ]
            )
            part_valid_outputs.append(temp_valid)
        commands.append(command)
        part_manifests.append(temp_manifest)

    run_parallel_subprocesses(
        commands_list=commands,
        max_workers=args.num_workers,
        print_output=False,
    )

    merge_tsv_files(part_manifests, manifest_file)
    merge_tsv_files(part_score_outputs, assignment_score_output)
    print(f"saved merged assignment scores: {assignment_score_output}")
    if save_valid:
        if valid_records_output is None:
            raise ValueError("valid_records_output is required when save_valid=True.")
        merge_msdatasets(part_valid_outputs, valid_records_output, empty_like=dataset)

    if not args.keep_parallel_temp:
        for path in temp_root.glob("part_*"):
            path.unlink(missing_ok=True)
        try:
            temp_root.rmdir()
        except OSError:
            pass


def default_valid_output(structure_dir: Path) -> Path:
    return structure_dir / "valid_records.msds"


def default_assignment_score_output(structure_dir: Path) -> Path:
    return structure_dir / "assignment_scores.tsv"


def parallel_temp_dir(args: argparse.Namespace, output_root: Path, split_name: str) -> Path:
    if args.parallel_temp_dir is not None:
        return Path(args.parallel_temp_dir) / split_name
    return output_root / f"_{split_name}_parallel_tmp"


def remove_existing_dirs(directories: list[Path], *, description: str) -> None:
    for directory in directories:
        if not directory.exists():
            continue
        if not directory.is_dir() or directory.is_symlink():
            raise NotADirectoryError(f"Expected a {description} directory: {directory}")
        shutil.rmtree(directory)
        print(f"removed existing {description} directory: {directory}")


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    device = torch.device(args.device)
    generator = None
    output_root = Path(args.output_dir)

    if args.structure_output_dir is not None:
        generator = load_generator(args.params, device=device)
        build_structure_files_for_input(
            input_path=args.train_input,
            output_dir=Path(args.structure_output_dir),
            args=args,
            generator=generator,
            manifest_file=args.manifest_file,
            save_valid=args.save_valid_records,
            valid_records_output=args.valid_records_output,
            assignment_score_output=args.assignment_score_output,
        )
        return

    train_structure_dir = output_root / "train_structures"
    validation_structure_dir = output_root / "validation_structures"
    train_structure_data_dir = structure_data_dir(train_structure_dir)
    validation_structure_data_dir = structure_data_dir(validation_structure_dir)
    train_manifest_file = (
        Path(args.manifest_file)
        if args.manifest_file is not None
        else train_structure_dir / "manifest.tsv"
    )
    train_valid_output = (
        Path(args.train_valid_output)
        if args.train_valid_output is not None
        else default_valid_output(train_structure_dir)
    )
    validation_valid_output = (
        Path(args.validation_valid_output)
        if args.validation_valid_output is not None
        else default_valid_output(validation_structure_dir)
    )
    train_assignment_score_output = (
        Path(args.train_assignment_score_output)
        if args.train_assignment_score_output is not None
        else default_assignment_score_output(train_structure_dir)
    )
    validation_assignment_score_output = (
        Path(args.validation_assignment_score_output)
        if args.validation_assignment_score_output is not None
        else default_assignment_score_output(validation_structure_dir)
    )

    if args.overwrite:
        structure_dirs = [train_structure_dir]
        parallel_temp_dirs = [parallel_temp_dir(args, output_root, "train")]
        if args.validation_input is not None:
            structure_dirs.append(validation_structure_dir)
            parallel_temp_dirs.append(parallel_temp_dir(args, output_root, "validation"))
        remove_existing_dirs(structure_dirs, description="structure")
        remove_existing_dirs(parallel_temp_dirs, description="parallel temp")

    generator = load_generator(args.params, device=device)
    model_config_output = (
        Path(args.model_config_output)
        if args.model_config_output is not None
        else default_model_config_output(output_root)
    )
    copy_model_config_after_model_creation(
        source_file=args.params,
        output_file=model_config_output,
        overwrite=bool(args.overwrite_model_config),
    )

    if args.num_workers > 1:
        run_parallel_for_input(
            input_path=args.train_input,
            structure_output_dir=train_structure_data_dir,
            args=args,
            manifest_file=train_manifest_file,
            save_valid=bool(args.save_train_valid_records),
            valid_records_output=train_valid_output,
            assignment_score_output=train_assignment_score_output,
            split_name="train",
        )
        if args.validation_input is not None:
            run_parallel_for_input(
                input_path=args.validation_input,
                structure_output_dir=validation_structure_data_dir,
                args=args,
                manifest_file=validation_structure_dir / "manifest.tsv",
                save_valid=bool(args.save_validation_valid_records),
                valid_records_output=validation_valid_output,
                assignment_score_output=validation_assignment_score_output,
                split_name="validation",
            )
        return

    build_structure_files_for_input(
        input_path=args.train_input,
        output_dir=train_structure_data_dir,
        args=args,
        generator=generator,
        manifest_file=train_manifest_file,
        save_valid=bool(args.save_train_valid_records),
        valid_records_output=train_valid_output,
        assignment_score_output=train_assignment_score_output,
    )

    if args.validation_input is not None:
        build_structure_files_for_input(
            input_path=args.validation_input,
            output_dir=validation_structure_data_dir,
            args=args,
            generator=generator,
            manifest_file=validation_structure_dir / "manifest.tsv",
            save_valid=bool(args.save_validation_valid_records),
            valid_records_output=validation_valid_output,
            assignment_score_output=validation_assignment_score_output,
        )


if __name__ == "__main__":
    main()
