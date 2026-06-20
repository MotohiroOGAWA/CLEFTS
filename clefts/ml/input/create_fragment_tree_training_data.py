from __future__ import annotations

import argparse
import json
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
        "--input",
        default="data/raw/NIST/NIST23/MSMS-Pos-NIST23_v20_mini.msds",
        help="Input MSDataset path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/test/fragment_tree_training_structures",
        help="Output directory for per-SMILES .pt structure files.",
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


def main() -> None:
    args = parse_args()
    dataset = MSDataset.load(args.input)

    print(f"input: {args.input}")
    print(f"output_dir: {args.output_dir}")
    device = torch.device(args.device)
    generator = load_generator(args.params, device=device)

    saved_files = build_fragment_tree_structure_files(
        dataset=dataset,
        feature_model=generator.feature_model,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        precursor_mz_column=args.precursor_mz_column,
        adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,
        instrument_column=args.instrument_column,
        overwrite=args.overwrite,
        manifest_file=args.manifest_file,
    )

    print(f"saved structure files: {len(saved_files)}")
    for path in saved_files[:5]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
