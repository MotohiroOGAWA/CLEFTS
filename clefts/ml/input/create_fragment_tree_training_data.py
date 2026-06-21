from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from clefts.libs.msentity.msentity import MSDataset
from clefts.utils.parallel_subprocess import run_parallel_subprocesses

try:
    from .fragment_tree_training_data import (
        build_fragment_tree_structure_files,
        group_record_indexes_by_smiles,
    )
    from ..specgen.fragment_tree_spectrum_predictor import FragmentSpectrumGenerator
except ImportError:
    from clefts.ml.input.fragment_tree_training_data import (
        build_fragment_tree_structure_files,
        group_record_indexes_by_smiles,
    )
    from clefts.ml.specgen.fragment_tree_spectrum_predictor import FragmentSpectrumGenerator


def parse_args() -> argparse.Namespace:
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
            "Output root directory. Training files are written under "
            "train_structures, and validation files under validation_structures "
            "when --validation-input is provided."
        ),
    )
    parser.add_argument(
        "--params",
        default="clefts/ml/specgen/presets/fragment_spectrum_generator_param.json",
        help="FragmentSpectrumGenerator parameter JSON used to construct the feature model.",
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
        "--save-valid-records",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_generator(params_path: str, device: torch.device) -> FragmentSpectrumGenerator:
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    generator = FragmentSpectrumGenerator(**params).to(device)
    generator.eval()
    return generator


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


def build_structure_files_for_input(
    *,
    input_path: str,
    output_dir: str | Path,
    args: argparse.Namespace,
    generator: FragmentSpectrumGenerator,
    manifest_file: str | Path | None = None,
    save_valid: bool = False,
    valid_records_output: str | Path | None = None,
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

    temp_root = (
        Path(args.parallel_temp_dir) / split_name
        if args.parallel_temp_dir is not None
        else Path(args.output_dir) / f"_{split_name}_parallel_tmp"
    )
    temp_root.mkdir(parents=True, exist_ok=True)

    print(
        f"{split_name}: split into {len(chunks)} chunks "
        f"({args.chunk_size} SMILES groups per chunk)."
    )

    commands: list[list[str]] = []
    part_manifests: list[Path] = []
    part_valid_outputs: list[Path] = []
    script_path = Path(__file__).resolve()

    for chunk_index, record_indexes in enumerate(
        tqdm(chunks, desc=f"Preparing {split_name} chunks", mininterval=1.0)
    ):
        temp_input = temp_root / f"part_{chunk_index:06d}.msds"
        temp_manifest = temp_root / f"part_{chunk_index:06d}_manifest.tsv"
        temp_valid = temp_root / f"part_{chunk_index:06d}_valid.msds"
        dataset[record_indexes].save(str(temp_input))

        command = [
            sys.executable,
            str(script_path),
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
            "--manifest-file",
            str(temp_manifest),
            "--num-workers",
            "1",
        ]
        if args.instrument_column is not None:
            command.extend(["--instrument-column", str(args.instrument_column)])
        if args.overwrite:
            command.append("--overwrite")
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


def main() -> None:
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
        )
        return

    train_structure_dir = output_root / "train_structures"
    validation_structure_dir = output_root / "validation_structures"
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

    if args.num_workers > 1:
        run_parallel_for_input(
            input_path=args.train_input,
            structure_output_dir=train_structure_dir,
            args=args,
            manifest_file=train_manifest_file,
            save_valid=bool(args.save_train_valid_records),
            valid_records_output=train_valid_output,
            split_name="train",
        )
        if args.validation_input is not None:
            run_parallel_for_input(
                input_path=args.validation_input,
                structure_output_dir=validation_structure_dir,
                args=args,
                manifest_file=validation_structure_dir / "manifest.tsv",
                save_valid=bool(args.save_validation_valid_records),
                valid_records_output=validation_valid_output,
                split_name="validation",
            )
        return

    generator = load_generator(args.params, device=device)
    build_structure_files_for_input(
        input_path=args.train_input,
        output_dir=train_structure_dir,
        args=args,
        generator=generator,
        manifest_file=train_manifest_file,
        save_valid=bool(args.save_train_valid_records),
        valid_records_output=train_valid_output,
    )

    if args.validation_input is not None:
        build_structure_files_for_input(
            input_path=args.validation_input,
            output_dir=validation_structure_dir,
            args=args,
            generator=generator,
            manifest_file=None,
            save_valid=bool(args.save_validation_valid_records),
            valid_records_output=validation_valid_output,
        )


if __name__ == "__main__":
    main()
