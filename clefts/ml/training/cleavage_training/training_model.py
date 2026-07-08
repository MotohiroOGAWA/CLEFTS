from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from typing import Dict, Sequence

from clefts.domain.fragment.cleavage import CleavagePatternSet
from clefts.ml.mol.mol_encoder import MolEncoder
from clefts.ml.specgen.components.cleavage.cleavage_edge_feature_net import CleavageEdgeFeatureNet
from clefts.ml.training.cleavage_training.pretraining_model import CleavagePretrainingModel
from clefts.ml.training.cleavage_training.trainer import train


def parse_symbols(text: str) -> Sequence[str]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


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


def mol_encoder_params_from_args(args: argparse.Namespace) -> Dict:
    return {
        "symbols": parse_symbols(args.symbols),
        "node_dim": args.node_dim,
        "graph_dim": args.graph_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "max_degree": args.max_degree,
        "max_spatial_dist": args.max_spatial_dist,
        "max_edge_dist": args.max_edge_dist,
        "dropout": args.dropout,
    }


def build_mol_encoder(args: argparse.Namespace) -> MolEncoder:
    checkpoint = None
    params = None
    if args.mol_encoder_checkpoint:
        checkpoint = load_mol_encoder_checkpoint(args.mol_encoder_checkpoint)
        params = dict(checkpoint.get("mol_encoder_params") or {})
        if not params:
            params = mol_encoder_params_from_args(args)
    else:
        params = mol_encoder_params_from_args(args)
    if "symbols" in params:
        params["symbols"] = tuple(params["symbols"])
    mol_encoder = MolEncoder(**params)
    if checkpoint is not None:
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
        observed_edge_loss_weight=args.observed_edge_loss_weight,
        pattern_loss_weight=args.pattern_loss_weight,
        reaction_loss_weight=args.reaction_loss_weight,
        product_loss_weight=args.product_loss_weight,
        atom_location_loss_weight=args.atom_location_loss_weight,
        dropout=args.dropout,
    )
    if args.freeze_mol_encoder:
        model.freeze_mol_encoder()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain cleavage-edge features from FragmentTreeStructure files.")
    parser.add_argument("--train-dir", required=True, help="Directory containing training FragmentTreeStructure .pt files.")
    parser.add_argument("--val-dir", required=True, help="Directory containing validation FragmentTreeStructure .pt files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints and metrics.")
    parser.add_argument("--cleavage-pattern-set-json", required=True, help="Cleavage pattern set JSON, or fragmenter preset JSON containing one.")
    parser.add_argument("--mol-encoder-checkpoint", default=None, help="Optional mol_training mol_encoder_pretrained.pt checkpoint.")
    parser.add_argument("--freeze-mol-encoder", action="store_true", help="Freeze MolEncoder during cleavage pretraining.")
    parser.add_argument("--symbols", default="C,H,N,O,P,S,F,Cl,Br,I,Si,B,Na,K", help="Comma-separated atom symbols for MolEncoder.")
    parser.add_argument("--node-dim", type=int, default=128)
    parser.add_argument("--graph-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--max-spatial-dist", type=int, default=32)
    parser.add_argument("--max-edge-dist", type=int, default=32)
    parser.add_argument("--edge-feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cleavage-fc-dims", default="256,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--observed-edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--pattern-loss-weight", type=float, default=1.0)
    parser.add_argument("--reaction-loss-weight", type=float, default=1.0)
    parser.add_argument("--product-loss-weight", type=float, default=1.0)
    parser.add_argument("--atom-location-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = build_model(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    )


if __name__ == "__main__":
    main()
