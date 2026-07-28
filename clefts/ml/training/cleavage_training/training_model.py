from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from typing import Dict

from clefts.domain.fragment.cleavage import CleavagePatternSet
from clefts.ml.mol.mol_encoder import MolEncoder
from clefts.ml.specgen.components.cleavage.cleavage_edge_feature_net import CleavageEdgeFeatureNet
from clefts.ml.training.cleavage_training.pretraining_model import CleavagePretrainingModel
from clefts.ml.training.cleavage_training.trainer import train


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def load_cleavage_pattern_set_params(path: str | Path) -> Dict:
    data = load_json(path)
    if "patterns" in data:
        return data
    if "fragment_ion_tree_builder" in data and "cleavage_pattern_set" in data["fragment_ion_tree_builder"]:
        return dict(data["fragment_ion_tree_builder"]["cleavage_pattern_set"])
    if "cleavage_pattern_set" in data:
        return dict(data["cleavage_pattern_set"])
    raise KeyError("Could not find a cleavage pattern set in the JSON file.")


def class_counts_from_pattern_set(params: Dict) -> Dict[str, int]:
    pattern_set = CleavagePatternSet.from_dict(params)
    num_patterns = 0
    num_reactions = 0
    num_products = 0
    for pattern in pattern_set.patterns:
        num_patterns = max(num_patterns, int(pattern.pattern_id) + 1)
        for reaction in pattern.cleavage_reactions:
            num_reactions = max(num_reactions, int(reaction.id) + 1)
            num_products = max(num_products, len(reaction.prod_idx_to_maps))
    return {
        "num_patterns": num_patterns,
        "num_reactions": num_reactions,
        "num_product_molecules": num_products,
    }



def load_mol_encoder_checkpoint(path: str | Path) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Mol encoder checkpoint must be a dict: {path}")
    if "mol_encoder_state_dict" not in checkpoint:
        raise KeyError(f"Mol encoder checkpoint is missing mol_encoder_state_dict: {path}")
    return checkpoint


def build_mol_encoder(args: argparse.Namespace) -> MolEncoder:
    checkpoint = load_mol_encoder_checkpoint(args.mol_encoder_checkpoint)
    params = dict(checkpoint.get("mol_encoder_params") or {})
    if not params:
        raise KeyError(
            "Mol encoder checkpoint is missing mol_encoder_params; "
            "cleavage training restores MolEncoder from the checkpoint only."
        )
    if "symbols" in params:
        params["symbols"] = tuple(params["symbols"])
    mol_encoder = MolEncoder(**params)
    missing, unexpected = mol_encoder.load_state_dict(
        checkpoint["mol_encoder_state_dict"],
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load MolEncoder checkpoint cleanly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mol_encoder

def build_model(args: argparse.Namespace) -> CleavagePretrainingModel:
    pattern_set_params = load_cleavage_pattern_set_params(args.cleavage_pattern_set_json)
    counts = class_counts_from_pattern_set(pattern_set_params)
    mol_encoder = build_mol_encoder(args)
    cleavage_edge_fnet = CleavageEdgeFeatureNet(
        cleavage_pattern_set_params=pattern_set_params,
        feature_dim=args.edge_feature_dim,
        mol_dim=mol_encoder.graph_dim,
        atom_dim=mol_encoder.node_dim,
        fc_dims=tuple(int(v) for v in args.cleavage_fc_dims.split(",") if v.strip()),
        dropout=args.dropout,
    )
    model = CleavagePretrainingModel(
        mol_encoder=mol_encoder,
        cleavage_edge_fnet=cleavage_edge_fnet,
        num_patterns=counts["num_patterns"],
        num_reactions=counts["num_reactions"],
        num_product_molecules=counts["num_product_molecules"],
        hidden_dim=args.hidden_dim,
        pattern_loss_weight=args.pattern_loss_weight,
        reaction_loss_weight=args.reaction_loss_weight,
        product_loss_weight=args.product_loss_weight,
        reactant_structure_loss_weight=args.reactant_structure_loss_weight,
        surrounding_structure_loss_weight=args.surrounding_structure_loss_weight,
        surrounding_structure_radius=args.surrounding_structure_radius,
        dropout=args.dropout,
    )
    model.freeze_mol_encoder()
    return model



def prepare_output_dir(
    path: str | Path,
    *,
    overwrite: bool = False,
    preserve: tuple[str | Path, ...] = (),
) -> Path:
    output_path = Path(path)
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            if not sys.stdin.isatty():
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_path}. "
                    "Use --overwrite to replace it."
                )
            answer = input(f"Output directory already exists: {output_path}. Overwrite? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                raise SystemExit("Aborted because output directory was not overwritten.")
        preserved_paths = {
            Path(value).resolve()
            for value in preserve
            if Path(value).resolve().is_relative_to(output_path.resolve())
        }
        for child in output_path.iterdir():
            child_resolved = child.resolve()
            if any(
                preserved == child_resolved
                or preserved.is_relative_to(child_resolved)
                for preserved in preserved_paths
            ):
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def build_parser() -> argparse.ArgumentParser:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be at least 1")
        return parsed

    parser = argparse.ArgumentParser(description="Pretrain cleavage-edge features from FragmentTreeStructure files.")
    parser.add_argument("--train-dir", required=True, help="Training structure split directory, usually <project>/train_structures.")
    parser.add_argument("--val-dir", required=True, help="Validation structure split directory, usually <project>/validation_structures.")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints and metrics.")
    parser.add_argument("--cleavage-pattern-set-json", required=True, help="Cleavage pattern set JSON, or fragmenter preset JSON containing one.")
    parser.add_argument("--mol-encoder-checkpoint", required=True, help="mol_training checkpoint containing mol_encoder_state_dict and mol_encoder_params. MolEncoder is restored from this file and frozen.")
    parser.add_argument("--edge-feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cleavage-fc-dims", default="256,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pattern-loss-weight", type=float, default=1.0)
    parser.add_argument("--reaction-loss-weight", type=float, default=1.0)
    parser.add_argument("--product-loss-weight", type=float, default=1.0)
    parser.add_argument("--reactant-structure-loss-weight", type=float, default=1.0)
    parser.add_argument("--surrounding-structure-loss-weight", type=float, default=1.0)
    parser.add_argument("--surrounding-structure-radius", type=int, default=2, help="Maximum bond distance predicted around each reactant SMARTS-matched atom.")
    parser.add_argument(
        "--preprocessing-cache",
        default=None,
        help="Reusable cleavage event/file index cache. Defaults beside output-dir.",
    )
    parser.add_argument(
        "--rebuild-preprocessing-cache",
        action="store_true",
        help="Ignore and rebuild an existing preprocessing cache.",
    )
    parser.add_argument(
        "--min-data-count",
        type=int,
        default=1000,
        help="Targets with at most this many events are covered by forced sampling.",
    )
    parser.add_argument(
        "--mask-balance-patience",
        type=int,
        default=32,
        help="Number of batches a target may be absent before cached sampling forces it.",
    )
    parser.add_argument(
        "--mask-balance-max-forced-per-batch",
        type=int,
        default=8,
        help="Maximum cached rare-target files added per batch.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-cleavage-events",
        type=positive_int,
        default=None,
        help=(
            "Maximum cleavage events encoded per SMILES structure load. "
            "Regular events are randomly resampled; forced rare-target events remain selected."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Stop after this many non-improving validation epochs; disabled by default.")
    parser.add_argument("--early-stopping-window-size", type=int, default=1)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output-dir if it already exists and is not empty.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_dir)
    default_cache = output_path / "main_preprocessing_cache.pt"
    cache_path = (
        Path(args.preprocessing_cache)
        if args.preprocessing_cache not in {None, ""}
        else default_cache
    )
    output_dir = prepare_output_dir(
        output_path,
        overwrite=args.overwrite,
        preserve=(cache_path,),
    )
    model = build_model(args)
    with (output_dir / "cleavage_pretraining_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    train(
        model=model,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_window_size=args.early_stopping_window_size,
        early_stopping_min_delta=args.early_stopping_min_delta,
        preprocessing_cache=args.preprocessing_cache,
        rebuild_preprocessing_cache=args.rebuild_preprocessing_cache,
        min_data_count=args.min_data_count,
        mask_balance_patience=args.mask_balance_patience,
        mask_balance_max_forced_per_batch=args.mask_balance_max_forced_per_batch,
        max_cleavage_events=args.max_cleavage_events,
    )


if __name__ == "__main__":
    main()
