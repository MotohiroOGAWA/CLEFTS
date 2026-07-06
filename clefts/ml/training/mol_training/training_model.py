#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    app_root = Path(__file__).resolve().parents[4]
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))

    from clefts.libs.mmkit.mmkit import Compound
    from clefts.ml.mol.mol_encoder import MolEncoder
    from clefts.ml.mol.mol_graphormer import (
        DEFAULT_MOL_GRAPHORMER_DROPOUT,
        DEFAULT_MOL_GRAPHORMER_MAX_DEGREE,
        DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST,
        DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
    )
    from clefts.ml.training.mol_training.dataset import (
        DEFAULT_DESCRIPTOR_NAMES,
        DescriptorNormalizer,
        MolPretrainingDataset,
        load_smiles_file,
    )
    from clefts.ml.training.mol_training.feature_schema import atom_feature_groups, bond_feature_groups
    from clefts.ml.training.mol_training.pretraining_model import MolPretrainingModel
    from clefts.ml.training.mol_training.trainer import make_loader, train_epochs
else:
    from ....libs.mmkit.mmkit import Compound
    from ...mol.mol_encoder import MolEncoder
    from ...mol.mol_graphormer import (
        DEFAULT_MOL_GRAPHORMER_DROPOUT,
        DEFAULT_MOL_GRAPHORMER_MAX_DEGREE,
        DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST,
        DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
    )
    from .dataset import (
        DEFAULT_DESCRIPTOR_NAMES,
        DescriptorNormalizer,
        MolPretrainingDataset,
        load_smiles_file,
    )
    from .feature_schema import atom_feature_groups, bond_feature_groups
    from .pretraining_model import MolPretrainingModel
    from .trainer import make_loader, train_epochs


@dataclass(frozen=True)
class MolEncoderConfig:
    node_dim: int
    graph_dim: int
    num_layers: int
    num_heads: int
    max_degree: int
    max_spatial_dist: int
    max_edge_dist: int
    dropout: float

    @property
    def id(self) -> str:
        return (
            f"node{self.node_dim}_gdim{self.graph_dim}_"
            f"layers{self.num_layers}_heads{self.num_heads}_"
            f"deg{self.max_degree}_spd{self.max_spatial_dist}_"
            f"edged{self.max_edge_dist}_drop{self.dropout:g}"
        ).replace(".", "p")


def parse_csv_ints(text: str) -> List[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_csv_strings(text: str | None) -> Sequence[str] | None:
    if text is None or text == "":
        return None
    return tuple(part.strip() for part in text.split(",") if part.strip())


def normalize_path_list(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or [value]
    return [str(item) for item in value if str(item).strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    args.train_smiles = normalize_path_list(args.train_smiles)
    args.val_smiles = normalize_path_list(args.val_smiles)
    if not args.train_smiles:
        parser.error("--train-smiles is required.")
    if not args.val_smiles:
        parser.error("--val-smiles is required.")
    if args.output_dir in {None, ""}:
        parser.error("--output-dir is required.")
    if args.symbols in {None, ""}:
        parser.error("--symbols is required.")
    return args

def unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def canonicalize_unique_smiles(
    smiles_values: Sequence[str],
    *,
    desc: str = "Canonicalizing SMILES",
    canonical_cache: Dict[str, str | None] | None = None,
) -> Tuple[List[str], int]:
    out = []
    seen = set()
    invalid_count = 0
    for smiles in tqdm(smiles_values, desc=desc, unit="smiles"):
        if canonical_cache is not None and smiles in canonical_cache:
            canonical = canonical_cache[smiles]
            if canonical is None:
                invalid_count += 1
                continue
        else:
            try:
                compound = Compound.from_smiles(smiles)
            except Exception:
                invalid_count += 1
                if canonical_cache is not None:
                    canonical_cache[smiles] = None
                continue
            canonical = compound.smiles
            if canonical_cache is not None:
                canonical_cache[smiles] = canonical
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out, invalid_count


def load_raw_smiles_files(paths: Sequence[str]) -> Tuple[List[str], List[Dict[str, object]]]:
    all_smiles: List[str] = []
    per_file = []
    for path in paths:
        smiles = load_smiles_file(path)
        unique_smiles = unique_preserve_order(smiles)
        all_smiles.extend(smiles)
        per_file.append(
            {
                "path": str(path),
                "raw_smiles": len(smiles),
                "unique_raw_smiles": len(unique_smiles),
            }
        )
    return all_smiles, per_file


def annotate_per_file_canonical_smiles_counts(
    per_file: List[Dict[str, object]],
    *,
    desc_prefix: str,
    canonical_cache: Dict[str, str | None],
) -> None:
    for entry in per_file:
        path = str(entry["path"])
        raw_unique = unique_preserve_order(load_smiles_file(path))
        canonical_unique, invalid_count = canonicalize_unique_smiles(
            raw_unique,
            desc=f"{desc_prefix} {Path(path).name}",
            canonical_cache=canonical_cache,
        )
        entry["invalid_after_raw_unique"] = invalid_count
        entry["canonical_unique_smiles"] = len(canonical_unique)


def prepare_smiles_split(
    *,
    train_smiles_paths: Sequence[str],
    val_smiles_paths: Sequence[str],
    output_dir: Path,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    train_raw, train_files = load_raw_smiles_files(train_smiles_paths)
    val_raw, val_files = load_raw_smiles_files(val_smiles_paths)
    canonical_cache: Dict[str, str | None] = {}
    annotate_per_file_canonical_smiles_counts(
        train_files,
        desc_prefix="Canonicalizing train file",
        canonical_cache=canonical_cache,
    )
    annotate_per_file_canonical_smiles_counts(
        val_files,
        desc_prefix="Canonicalizing validation file",
        canonical_cache=canonical_cache,
    )

    train_raw_unique = unique_preserve_order(train_raw)
    val_raw_unique = unique_preserve_order(val_raw)
    train_smiles, train_invalid = canonicalize_unique_smiles(
        train_raw_unique,
        desc="Canonicalizing train SMILES",
        canonical_cache=canonical_cache,
    )
    val_canonical, val_invalid = canonicalize_unique_smiles(
        val_raw_unique,
        desc="Canonicalizing validation SMILES",
        canonical_cache=canonical_cache,
    )

    train_set = set(train_smiles)
    val_smiles = [smiles for smiles in val_canonical if smiles not in train_set]
    val_removed_train_overlap = len(val_canonical) - len(val_smiles)

    if not train_smiles:
        raise ValueError("No valid training SMILES remain after canonicalization.")
    if not val_smiles:
        raise ValueError("No validation SMILES remain after removing molecules already present in training.")

    summary: Dict[str, object] = {
        "train_smiles": list(map(str, train_smiles_paths)),
        "val_smiles": list(map(str, val_smiles_paths)),
        "train_files": train_files,
        "val_files": val_files,
        "train_raw_total": len(train_raw),
        "train_raw_unique": len(train_raw_unique),
        "train_invalid_after_raw_unique": train_invalid,
        "train_canonical_unique": len(train_smiles),
        "val_raw_total": len(val_raw),
        "val_raw_unique": len(val_raw_unique),
        "val_invalid_after_raw_unique": val_invalid,
        "val_canonical_unique_before_train_filter": len(val_canonical),
        "val_removed_train_overlap": val_removed_train_overlap,
        "val_canonical_unique": len(val_smiles),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "smiles_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "train_smiles.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_smiles) + "\n")
    with open(output_dir / "val_smiles.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_smiles) + "\n")

    print(
        "SMILES split: "
        f"train raw={len(train_raw)} raw_unique={len(train_raw_unique)} canonical_unique={len(train_smiles)}; "
        f"val raw={len(val_raw)} raw_unique={len(val_raw_unique)} "
        f"canonical_unique={len(val_smiles)} removed_train_overlap={val_removed_train_overlap}"
    )
    return train_smiles, val_smiles, summary


def input_file_manifest(paths: Sequence[str]) -> List[Dict[str, object]]:
    out = []
    for path in paths:
        p = Path(path)
        stat = p.stat()
        out.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return out


def smiles_split_cache_manifest(
    *,
    train_smiles_paths: Sequence[str],
    val_smiles_paths: Sequence[str],
) -> Dict[str, object]:
    return {
        "train_files": input_file_manifest(train_smiles_paths),
        "val_files": input_file_manifest(val_smiles_paths),
    }


def write_smiles_split_outputs(
    *,
    output_dir: Path,
    train_smiles: Sequence[str],
    val_smiles: Sequence[str],
    summary: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "smiles_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "train_smiles.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_smiles) + "\n")
    with open(output_dir / "val_smiles.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_smiles) + "\n")


def load_or_prepare_smiles_split(
    *,
    train_smiles_paths: Sequence[str],
    val_smiles_paths: Sequence[str],
    output_dir: Path,
    rebuild_cache: bool = False,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    cache_path = output_dir / "pretraining_smiles_split_cache.json"
    manifest = smiles_split_cache_manifest(
        train_smiles_paths=train_smiles_paths,
        val_smiles_paths=val_smiles_paths,
    )
    if not rebuild_cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("input_manifest") == manifest:
            train_smiles = list(payload["train_smiles"])
            val_smiles = list(payload["val_smiles"])
            summary = dict(payload["summary"])
            summary["smiles_split_cache"] = {
                "hit": True,
                "path": str(cache_path),
            }
            write_smiles_split_outputs(
                output_dir=output_dir,
                train_smiles=train_smiles,
                val_smiles=val_smiles,
                summary=summary,
            )
            print(f"Loaded SMILES split cache: {cache_path}")
            return train_smiles, val_smiles, summary
        print(f"SMILES split cache input mismatch, rebuilding: {cache_path}")

    train_smiles, val_smiles, summary = prepare_smiles_split(
        train_smiles_paths=train_smiles_paths,
        val_smiles_paths=val_smiles_paths,
        output_dir=output_dir,
    )
    summary["smiles_split_cache"] = {
        "hit": False,
        "path": str(cache_path),
        "rebuilt": bool(rebuild_cache),
    }
    write_smiles_split_outputs(
        output_dir=output_dir,
        train_smiles=train_smiles,
        val_smiles=val_smiles,
        summary=summary,
    )
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_manifest": manifest,
                "train_smiles": list(train_smiles),
                "val_smiles": list(val_smiles),
                "summary": summary,
            },
            f,
            indent=2,
        )
    return train_smiles, val_smiles, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain MolEncoder with node, edge, and graph-level tasks.")
    parser.add_argument("--train-smiles", nargs="+", default=None, help="Newline-delimited SMILES file(s) for training.")
    parser.add_argument("--val-smiles", nargs="+", default=None, help="Newline-delimited SMILES file(s) for validation.")
    parser.add_argument("--output-dir", default=None, help="Directory where configs, logs, TensorBoard files, summaries, and checkpoints are written.")
    parser.add_argument("--device", default="cpu", help="Torch device for training, for example cpu, cuda, or cuda:0.")

    parser.add_argument("--symbols", default=None, help="Comma-separated atom symbols. Required.")
    parser.add_argument("--node-dim", "--node_dim", default="64,128,256", help="Comma-separated MolEncoder node embedding dims. One value means fixed.")
    parser.add_argument("--graph-dim", "--graph_dim", default="128", help="Comma-separated MolEncoder graph dims.")
    parser.add_argument("--num-layers", default="4", help="Comma-separated Graphormer layer counts.")
    parser.add_argument("--num-heads", default="8", help="Comma-separated attention head counts.")
    parser.add_argument("--max-degree", default=str(DEFAULT_MOL_GRAPHORMER_MAX_DEGREE), help="Comma-separated max degree values.")
    parser.add_argument("--max-spatial-dist", default=str(DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST), help="Comma-separated max shortest-path distances.")
    parser.add_argument("--max-edge-dist", default=str(DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST), help="Comma-separated max edge-path distances.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_MOL_GRAPHORMER_DROPOUT, help="MolEncoder dropout. This is a single value, not a candidate list.")

    parser.add_argument("--batch-size", type=int, default=32, help="Number of molecules per training batch.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--preprocessing-cache",
        default=None,
        help="Path to a torch cache for canonical SMILES, graph data, and descriptor statistics. Default: output-dir/pretraining_preprocessed_dataset.pt.",
    )
    parser.add_argument(
        "--rebuild-preprocessing-cache",
        action="store_true",
        help="Ignore any existing preprocessing cache and rebuild it.",
    )

    parser.add_argument("--epochs", "--full-epochs", dest="epochs", type=int, default=100, help="Maximum number of pretraining epochs.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Enable early stopping after this many non-improving epochs. Default: disabled.",
    )
    parser.add_argument("--early-stopping-window-size", type=int, default=1, help="Number of recent validation losses to smooth inside early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Minimum validation-loss improvement required to reset early stopping.")
    parser.add_argument("--early-stopping-reset-step", type=float, default=1.0, help="Counter reset amount used by the early stopping helper after improvement.")
    parser.add_argument("--early-stopping-verbose", action="store_true", help="Print early stopping state updates.")

    parser.add_argument("--node-mask-ratio", type=float, default=0.25, help="Random node masking ratio for node attribute prediction.")
    parser.add_argument("--edge-mask-ratio", type=float, default=0.25, help="Random edge masking ratio for edge attribute prediction.")
    parser.add_argument("--disable-balanced-attribute-masking", action="store_true", help="Disable class-balancing when selecting node and edge attributes to mask.")
    parser.add_argument("--disable-balanced-record-sampling", action="store_true", help="Disable forced molecule sampling for rare node and edge attribute classes.")
    parser.add_argument("--mask-balance-patience", type=int, default=20, help="Number of batches a class may be absent before balanced masking or sampling forces it.")
    parser.add_argument("--mask-balance-max-forced-per-batch", type=int, default=8, help="Maximum forced rare classes added per batch by balanced masking or sampling.")
    parser.add_argument("--disable-balanced-validation-masks", action="store_true", help="Disable validation-time forced coverage for node and edge attribute classes present in a batch.")
    parser.add_argument("--min-validation-target-count", type=int, default=1, help="Warn when validation has fewer targets than this for any node or edge attribute class.")

    parser.add_argument("--disable-node-attribute", action="store_true", help="Disable masked node attribute prediction.")
    parser.add_argument("--disable-node-context", action="store_true", help="Disable node context prediction between K-hop neighborhoods and context graphs.")
    parser.add_argument("--disable-edge-attribute", action="store_true", help="Disable masked edge attribute prediction.")
    parser.add_argument("--disable-graph-contrastive", action="store_true", help="Disable graph-level contrastive learning with augmented molecule views.")
    parser.add_argument("--disable-graph-descriptors", action="store_true", help="Disable graph-level descriptor regression.")
    parser.add_argument("--graph-contrastive-node-mask-ratio", type=float, default=0.25, help="Node masking ratio used when creating graph contrastive views.")
    parser.add_argument("--graph-contrastive-edge-drop-ratio", type=float, default=0.25, help="Undirected bond deletion ratio used when creating graph contrastive views.")
    parser.add_argument("--graph-contrastive-temperature", type=float, default=0.2, help="Temperature for graph contrastive and context pair logits.")
    parser.add_argument("--context-k", type=int, default=2, help="K-hop radius for the center-node neighborhood in context prediction.")
    parser.add_argument("--context-r1", type=int, default=1, help="Inner hop radius of the context graph ring.")
    parser.add_argument("--context-r2", type=int, default=4, help="Outer hop radius of the context graph ring.")
    parser.add_argument("--disable-graph-ecfp", action="store_true", help="Disable graph-level ECFP fingerprint prediction.")
    parser.add_argument("--ecfp-radius", type=int, default=2, help="Morgan/ECFP fingerprint radius used as the graph-level target.")
    parser.add_argument("--ecfp-n-bits", type=int, default=2048, help="Number of bits in the ECFP fingerprint target.")
    parser.add_argument("--descriptor-names", default=",".join(DEFAULT_DESCRIPTOR_NAMES), help="Comma-separated RDKit descriptor targets for graph-level regression.")

    parser.add_argument("--node-loss-weight", type=float, default=1.0, help="Loss weight for masked node attribute prediction.")
    parser.add_argument("--context-loss-weight", type=float, default=1.0, help="Loss weight for node context prediction.")
    parser.add_argument("--edge-loss-weight", type=float, default=1.0, help="Loss weight for masked edge attribute prediction.")
    parser.add_argument("--graph-contrastive-loss-weight", type=float, default=1.0, help="Loss weight for graph-level contrastive learning.")
    parser.add_argument("--descriptor-loss-weight", type=float, default=1.0, help="Loss weight for graph-level descriptor regression.")
    parser.add_argument("--ecfp-loss-weight", type=float, default=1.0, help="Loss weight for graph-level ECFP fingerprint prediction.")
    parser.add_argument(
        "--dimension-penalty",
        type=float,
        default=1e-4,
        help="Added to selection loss as penalty * node_dim so smaller embeddings win close ties.",
    )
    return parser


def mol_encoder_configs(args: argparse.Namespace) -> List[MolEncoderConfig]:
    configs = []
    for values in itertools.product(
        parse_csv_ints(args.node_dim),
        parse_csv_ints(args.graph_dim),
        parse_csv_ints(args.num_layers),
        parse_csv_ints(args.num_heads),
        parse_csv_ints(args.max_degree),
        parse_csv_ints(args.max_spatial_dist),
        parse_csv_ints(args.max_edge_dist),
        [float(args.dropout)],
    ):
        config = MolEncoderConfig(*values)
        if config.node_dim % config.num_heads != 0:
            raise ValueError(
                f"Invalid MolEncoder config {config}: node_dim must be divisible by num_heads."
            )
        if config.graph_dim < config.node_dim or config.graph_dim % config.node_dim != 0:
            raise ValueError(
                f"Invalid MolEncoder config {config}: graph_dim must be a positive "
                "multiple of node_dim. If you meant four graph tokens, set "
                "graph_dim to node_dim * 4, for example 256 when node_dim is 64."
            )
        configs.append(config)
    return configs


def make_mol_encoder(config: MolEncoderConfig, *, symbols: Sequence[str]) -> MolEncoder:
    return MolEncoder(
        symbols=tuple(symbols),
        node_dim=config.node_dim,
        graph_dim=config.graph_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_degree=config.max_degree,
        max_spatial_dist=config.max_spatial_dist,
        max_edge_dist=config.max_edge_dist,
        dropout=config.dropout,
    )


def make_pretraining_model(
    args: argparse.Namespace,
    *,
    symbols: Sequence[str],
    config: MolEncoderConfig,
    descriptor_dim: int,
    descriptor_names: Sequence[str],
) -> MolPretrainingModel:
    mol_encoder = make_mol_encoder(config, symbols=symbols)
    return MolPretrainingModel(
        mol_encoder=mol_encoder,
        descriptor_dim=descriptor_dim,
        descriptor_names=descriptor_names,
        use_node_attribute=not args.disable_node_attribute,
        use_node_context=not args.disable_node_context,
        use_edge_attribute=not args.disable_edge_attribute,
        use_graph_contrastive=not args.disable_graph_contrastive,
        use_graph_descriptors=not args.disable_graph_descriptors,
        use_graph_ecfp=not args.disable_graph_ecfp,
        node_loss_weight=args.node_loss_weight,
        context_loss_weight=args.context_loss_weight,
        edge_loss_weight=args.edge_loss_weight,
        graph_contrastive_loss_weight=args.graph_contrastive_loss_weight,
        descriptor_loss_weight=args.descriptor_loss_weight,
        ecfp_loss_weight=args.ecfp_loss_weight,
        context_k=args.context_k,
        context_r1=args.context_r1,
        context_r2=args.context_r2,
        ecfp_dim=args.ecfp_n_bits,
        graph_contrastive_node_mask_ratio=args.graph_contrastive_node_mask_ratio,
        graph_contrastive_edge_drop_ratio=args.graph_contrastive_edge_drop_ratio,
        graph_contrastive_temperature=args.graph_contrastive_temperature,
        balanced_attribute_masking=not args.disable_balanced_attribute_masking,
        mask_balance_patience=args.mask_balance_patience,
        mask_balance_max_forced_per_batch=args.mask_balance_max_forced_per_batch,
        balanced_validation_masks=not args.disable_balanced_validation_masks,
    )


def write_summary_table(rows: List[Dict[str, object]], path: Path, *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def is_loss_or_accuracy_metric(name: str) -> bool:
    return name == "loss" or name.endswith("_loss") or name.endswith("_acc") or name.endswith("_r2")


def final_metric_summary_columns(best: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "final_epoch": int(best.get("final_epoch", best.get("epoch", -1))),
        "stopped_epoch": int(best.get("stopped_epoch", best.get("final_epoch", -1))),
        "early_stopping_enabled": bool(best.get("early_stopping_enabled", False)),
        "early_stopping_counter": int(best.get("early_stopping_counter", 0)),
        "last_validation_seconds": float(best.get("last_validation_seconds", 0.0)),
    }
    for split, metrics_key in (("train", "final_train_metrics"), ("val", "final_val_metrics")):
        metrics = best.get(metrics_key, {})
        if not isinstance(metrics, dict):
            continue
        for name in sorted(metrics):
            if is_loss_or_accuracy_metric(str(name)):
                out[f"final_{split}_{name}"] = metrics[name]
    return out


PREPROCESSING_CACHE_VERSION = 2


def preprocessing_cache_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.preprocessing_cache not in {None, ""}:
        return Path(args.preprocessing_cache)
    return output_dir / "pretraining_preprocessed_dataset.pt"


def preprocessing_cache_manifest(
    *,
    symbols: Sequence[str],
    descriptor_names: Sequence[str],
    train_smiles: Sequence[str],
    val_smiles: Sequence[str],
    ecfp_radius: int,
    ecfp_n_bits: int,
) -> Dict[str, object]:
    return {
        "version": PREPROCESSING_CACHE_VERSION,
        "symbols": list(symbols),
        "descriptor_names": list(descriptor_names),
        "train_smiles": list(train_smiles),
        "val_smiles": list(val_smiles),
        "ecfp_radius": int(ecfp_radius),
        "ecfp_n_bits": int(ecfp_n_bits),
    }


def load_preprocessing_cache(
    path: Path,
    *,
    manifest: Dict[str, object],
    symbols: Sequence[str],
    descriptor_names: Sequence[str],
    ecfp_radius: int,
    ecfp_n_bits: int,
) -> Tuple[MolPretrainingDataset, MolPretrainingDataset, DescriptorNormalizer, Dict[str, object]] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("manifest") != manifest:
        print(f"Preprocessing cache manifest mismatch, rebuilding: {path}")
        return None

    normalizer = DescriptorNormalizer(
        mean=payload["descriptor_mean"],
        std=payload["descriptor_std"],
    )
    train_dataset = MolPretrainingDataset.from_items(
        payload["train_items"],
        symbols=symbols,
        descriptor_names=descriptor_names,
        descriptor_normalizer=normalizer,
        ecfp_radius=ecfp_radius,
        ecfp_n_bits=ecfp_n_bits,
    )
    val_dataset = MolPretrainingDataset.from_items(
        payload["val_items"],
        symbols=symbols,
        descriptor_names=descriptor_names,
        descriptor_normalizer=normalizer,
        ecfp_radius=ecfp_radius,
        ecfp_n_bits=ecfp_n_bits,
    )
    summary = {
        "enabled": True,
        "hit": True,
        "path": str(path),
        "version": PREPROCESSING_CACHE_VERSION,
        "num_train_molecules": len(train_dataset),
        "num_val_molecules": len(val_dataset),
    }
    return train_dataset, val_dataset, normalizer, summary


def save_preprocessing_cache(
    path: Path,
    *,
    manifest: Dict[str, object],
    train_dataset: MolPretrainingDataset,
    val_dataset: MolPretrainingDataset,
    normalizer: DescriptorNormalizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "manifest": manifest,
            "descriptor_mean": normalizer.mean.detach().cpu(),
            "descriptor_std": normalizer.std.detach().cpu(),
            "train_items": train_dataset.items,
            "val_items": val_dataset.items,
        },
        path,
    )


def build_or_load_preprocessed_datasets(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    train_smiles: Sequence[str],
    val_smiles: Sequence[str],
    symbols: Sequence[str],
    descriptor_names: Sequence[str],
) -> Tuple[MolPretrainingDataset, MolPretrainingDataset, DescriptorNormalizer, Dict[str, object]]:
    cache_path = preprocessing_cache_path(args, output_dir)
    manifest = preprocessing_cache_manifest(
        symbols=symbols,
        descriptor_names=descriptor_names,
        train_smiles=train_smiles,
        val_smiles=val_smiles,
        ecfp_radius=args.ecfp_radius,
        ecfp_n_bits=args.ecfp_n_bits,
    )
    if not args.rebuild_preprocessing_cache:
        cached = load_preprocessing_cache(
            cache_path,
            manifest=manifest,
            symbols=symbols,
            descriptor_names=descriptor_names,
            ecfp_radius=args.ecfp_radius,
            ecfp_n_bits=args.ecfp_n_bits,
        )
        if cached is not None:
            print(f"Loaded preprocessing cache: {cache_path}")
            return cached

    print(f"Building preprocessing cache: {cache_path}")
    train_dataset = MolPretrainingDataset(
        train_smiles,
        symbols=symbols,
        descriptor_names=descriptor_names,
        ecfp_radius=args.ecfp_radius,
        ecfp_n_bits=args.ecfp_n_bits,
        progress_desc="Building train dataset",
    )
    normalizer = DescriptorNormalizer.fit(train_dataset.descriptor_matrix())
    train_dataset.descriptor_normalizer = normalizer
    val_dataset = MolPretrainingDataset(
        val_smiles,
        symbols=symbols,
        descriptor_names=descriptor_names,
        descriptor_normalizer=normalizer,
        ecfp_radius=args.ecfp_radius,
        ecfp_n_bits=args.ecfp_n_bits,
        progress_desc="Building validation dataset",
    )
    save_preprocessing_cache(
        cache_path,
        manifest=manifest,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        normalizer=normalizer,
    )
    summary = {
        "enabled": True,
        "hit": False,
        "path": str(cache_path),
        "version": PREPROCESSING_CACHE_VERSION,
        "rebuilt": bool(args.rebuild_preprocessing_cache),
        "num_train_molecules": len(train_dataset),
        "num_val_molecules": len(val_dataset),
    }
    return train_dataset, val_dataset, normalizer, summary


def build_feature_record_index(
    dataset: MolPretrainingDataset,
    *,
    symbols: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    return {
        "node": dataset.feature_record_index("x", atom_feature_groups(symbols)),
        "edge": dataset.feature_record_index("edge_attr", bond_feature_groups()),
    }


def feature_record_index_counts(
    index: Dict[str, Dict[str, Dict[str, List[int]]]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        level: {
            group_name: {label: len(records) for label, records in labels.items()}
            for group_name, labels in groups.items()
        }
        for level, groups in index.items()
    }


def write_feature_target_summary(
    *,
    train_dataset: MolPretrainingDataset,
    val_dataset: MolPretrainingDataset,
    symbols: Sequence[str],
    output_dir: Path,
    min_validation_target_count: int,
) -> Dict[str, object]:
    atom_groups = atom_feature_groups(symbols)
    bond_groups = bond_feature_groups()
    summary: Dict[str, object] = {
        "train_node": train_dataset.feature_target_counts("x", atom_groups),
        "val_node": val_dataset.feature_target_counts("x", atom_groups),
        "train_edge": train_dataset.feature_target_counts("edge_attr", bond_groups),
        "val_edge": val_dataset.feature_target_counts("edge_attr", bond_groups),
        "min_validation_target_count": int(min_validation_target_count),
        "validation_warnings": [],
    }

    warnings = []
    for level in ("node", "edge"):
        counts_by_group = summary[f"val_{level}"]
        for group_name, counts in counts_by_group.items():
            for label, count in counts.items():
                if int(count) < int(min_validation_target_count):
                    warnings.append(
                        {
                            "level": level,
                            "group": group_name,
                            "label": label,
                            "count": int(count),
                        }
                    )
    summary["validation_warnings"] = warnings
    with open(output_dir / "feature_target_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if warnings:
        print(
            "Validation target warning: "
            f"{len(warnings)} node/edge classes have fewer than "
            f"{min_validation_target_count} targets. See feature_target_summary.json."
        )
    return summary


def run_pretraining_stage(
    args: argparse.Namespace,
    *,
    symbols: Sequence[str],
    config: MolEncoderConfig,
    train_dataset: MolPretrainingDataset,
    val_dataset: MolPretrainingDataset,
    train_feature_record_index: Dict[str, Dict[str, Dict[str, List[int]]]],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    balanced_feature_record_index = None
    if not args.disable_balanced_attribute_masking and not args.disable_balanced_record_sampling:
        balanced_feature_record_index = train_feature_record_index
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        balanced_feature_record_index=balanced_feature_record_index,
        balance_patience=args.mask_balance_patience,
        balance_max_forced_per_batch=args.mask_balance_max_forced_per_batch,
    )
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = make_pretraining_model(
        args,
        symbols=symbols,
        config=config,
        descriptor_dim=len(train_dataset.descriptor_names),
        descriptor_names=train_dataset.descriptor_names,
    ).to(device)

    stage_dir = output_dir / "runs" / config.id
    best = train_epochs(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        node_mask_ratio=args.node_mask_ratio,
        edge_mask_ratio=args.edge_mask_ratio,
        output_dir=stage_dir,
        stage_name="pretraining",
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_window_size=args.early_stopping_window_size,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_reset_step=args.early_stopping_reset_step,
        early_stopping_verbose=args.early_stopping_verbose,
    )
    return {
        **asdict(config),
        "config_id": config.id,
        "stage": "pretraining",
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "selection_score": float(best["val_loss"]) + float(args.dimension_penalty) * float(config.node_dim),
        "checkpoint_path": str(stage_dir / "pretraining_best.pt"),
        **final_metric_summary_columns(best),
    }


def save_selected_checkpoint(summary_row: Dict[str, object], output_dir: Path) -> None:
    source = Path(str(summary_row["checkpoint_path"]))
    if not source.exists():
        raise FileNotFoundError(f"Selected checkpoint was not found: {source}")
    checkpoint = torch.load(source, map_location="cpu")
    checkpoint.setdefault("extra", {})
    checkpoint["extra"].update({"selected": True, "selection_source": str(source)})
    torch.save(checkpoint, output_dir / "mol_encoder_pretrained.pt")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    configs = mol_encoder_configs(args)
    train_smiles, val_smiles, smiles_split_summary = load_or_prepare_smiles_split(
        train_smiles_paths=args.train_smiles,
        val_smiles_paths=args.val_smiles,
        output_dir=output_dir,
        rebuild_cache=args.rebuild_preprocessing_cache,
    )
    symbols = parse_csv_strings(args.symbols)
    descriptor_names = tuple(parse_csv_strings(args.descriptor_names) or DEFAULT_DESCRIPTOR_NAMES)

    train_dataset, val_dataset, normalizer, preprocessing_cache_summary = build_or_load_preprocessed_datasets(
        args=args,
        output_dir=output_dir,
        train_smiles=train_smiles,
        val_smiles=val_smiles,
        symbols=symbols,
        descriptor_names=descriptor_names,
    )
    feature_target_summary = write_feature_target_summary(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        symbols=symbols,
        output_dir=output_dir,
        min_validation_target_count=args.min_validation_target_count,
    )
    train_feature_record_index = build_feature_record_index(train_dataset, symbols=symbols)
    balanced_record_sampling_summary = {
        "enabled": bool(
            not args.disable_balanced_attribute_masking
            and not args.disable_balanced_record_sampling
        ),
        "patience": int(args.mask_balance_patience),
        "max_forced_per_batch": int(args.mask_balance_max_forced_per_batch),
        "train_record_counts": feature_record_index_counts(train_feature_record_index),
    }

    config_record = {
        "args": vars(args),
        "mol_encoder_configs": [asdict(config) for config in configs],
        "symbols": tuple(symbols),
        "descriptor_names": descriptor_names,
        "descriptor_mean": normalizer.mean.tolist(),
        "descriptor_std": normalizer.std.tolist(),
        "ecfp_radius": int(args.ecfp_radius),
        "ecfp_n_bits": int(args.ecfp_n_bits),
        "num_train_molecules": len(train_dataset),
        "num_val_molecules": len(val_dataset),
        "smiles_split_summary": smiles_split_summary,
        "feature_target_summary": feature_target_summary,
        "preprocessing_cache": preprocessing_cache_summary,
        "balanced_record_sampling": balanced_record_sampling_summary,
    }
    with open(output_dir / "pretraining_config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    all_rows = []
    for config in configs:
        all_rows.append(
            run_pretraining_stage(
                args,
                symbols=symbols,
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_feature_record_index=train_feature_record_index,
                device=device,
                output_dir=output_dir,
            )
        )

    all_rows = sorted(all_rows, key=lambda row: (float(row["selection_score"]), int(row["node_dim"])))
    selected_row = all_rows[0]
    selected_config = MolEncoderConfig(
        node_dim=int(selected_row["node_dim"]),
        graph_dim=int(selected_row["graph_dim"]),
        num_layers=int(selected_row["num_layers"]),
        num_heads=int(selected_row["num_heads"]),
        max_degree=int(selected_row["max_degree"]),
        max_spatial_dist=int(selected_row["max_spatial_dist"]),
        max_edge_dist=int(selected_row["max_edge_dist"]),
        dropout=float(selected_row["dropout"]),
    )

    with open(output_dir / "selected_mol_encoder_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(selected_config), f, indent=2)

    save_selected_checkpoint(selected_row, output_dir)
    write_summary_table(all_rows, output_dir / "mol_encoder_pretraining_summary.csv")
    write_summary_table(all_rows, output_dir / "mol_encoder_pretraining_summary.tsv", delimiter="	")
    with open(output_dir / "mol_encoder_pretraining_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
