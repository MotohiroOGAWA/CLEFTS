from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clefts.libs.msentity.msentity import MSDataset

try:
    from .fragment_tree_training_data import (
        build_fragment_tree_structure_files,
    )
    from ..specgen.fragment_tree_spectrum_predictor import FragmentSpectrumGenerator
except ImportError:
    from clefts.ml.input.fragment_tree_training_data import (
        build_fragment_tree_structure_files,
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
        "--manifest-file",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_generator(params_path: str, device: torch.device) -> FragmentSpectrumGenerator:
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    generator = FragmentSpectrumGenerator(**params).to(device)
    generator.eval()
    return generator


def build_structure_files_for_input(
    *,
    input_path: str,
    output_dir: str | Path,
    args: argparse.Namespace,
    generator: FragmentSpectrumGenerator,
    manifest_file: str | Path | None = None,
) -> list[Path]:
    dataset = MSDataset.load(input_path)
    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")

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
    )

    print(f"saved structure files: {len(saved_files)}")
    for path in saved_files[:5]:
        print(f"  {path}")

    return saved_files


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    generator = load_generator(args.params, device=device)
    output_root = Path(args.output_dir)

    train_manifest_file = (
        None if args.manifest_file is None else Path(args.manifest_file)
    )
    build_structure_files_for_input(
        input_path=args.train_input,
        output_dir=output_root / "train_structures",
        args=args,
        generator=generator,
        manifest_file=train_manifest_file,
    )

    if args.validation_input is not None:
        build_structure_files_for_input(
            input_path=args.validation_input,
            output_dir=output_root / "validation_structures",
            args=args,
            generator=generator,
            manifest_file=None,
        )


if __name__ == "__main__":
    main()
