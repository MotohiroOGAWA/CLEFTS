from __future__ import annotations

import argparse
import csv
import json
import shutil
import math
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from ...common.torch_utils.check_point_manager import CheckPointManager, CkptNode
from ...common.torch_utils.training_setup import get_optimizer
from ...input.fragment_tree_structure import FragmentTreeStructure
from ...input.fragment_tree_training_data import (
    FragmentTreeStructureFileDataset,
    collate_fragment_tree_structure_items,
    group_record_indexes_by_smiles,
)
from ...input.single_fragment_tree_structure_builder import (
    SingleFragmentTreeStructureBuilder,
)
from ...specgen.fragment_tree_spectrum_predictor import (
    FragmentSpectrumGenerator,
    FragmentTreeSpectrumPredictor,
    fragment_spectrum_output_to_msdataset,
)
from .model import FragmentTreeTrainingModel
from ....libs.msentity.msentity import MSDataset
from ....libs.msentity.msentity.processing.spectrum_similarity import cosine_similarity_pair

METRIC_COLUMNS = (
    "event",
    "epoch",
    "global_step",
    "train_loss",
    "train_selection_loss",
    "train_intensity_loss",
    "train_window_loss",
    "train_window_selection_loss",
    "train_window_intensity_loss",
    "val_loss",
    "val_selection_loss",
    "val_intensity_loss",
    "val_cosine",
    "val_selection_precision",
    "val_selection_recall",
    "val_selection_f1",
    "val_predicted_peak_count",
    "val_target_peak_count",
    "val_matched_peak_count",
    "val_selected_intensity_fraction",
    "val_top5_recall",
    "val_top10_recall",
    "val_top20_recall",
    "val_matched_intensity_mae",
    "val_matched_intensity_weighted_mae",
    "lr",
)

PEAK_SELECTION_TOP_K = (5, 10, 20)
PEAK_SELECTION_MZ_TOLERANCE_DA = 0.01
PEAK_SELECTION_METRIC_NAMES = (
    "selection_precision",
    "selection_recall",
    "selection_f1",
    "predicted_peak_count",
    "target_peak_count",
    "matched_peak_count",
    "selected_intensity_fraction",
    *(f"top{k}_recall" for k in PEAK_SELECTION_TOP_K),
    "matched_intensity_mae",
    "matched_intensity_weighted_mae",
)
PEAK_SELECTION_QUANTILES = (
    ("min", 0.0),
    ("q1", 0.25),
    ("median", 0.5),
    ("q3", 0.75),
    ("max", 1.0),
)


DEFAULT_TRAIN_CONFIG_NAME = "train_config.json"
DEFAULT_PROJECT_MODEL_CONFIG_NAME = "model_config.json"
DEFAULT_PREPROCESSING_CONFIG_NAME = "preprocessing_config.json"
DEFAULT_EXPERIMENT_NAME = "exp_main"
DEFAULT_OPTIMIZER_INFO = {
    "name": "AdamW",
    "lr": 1e-5,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


@dataclass(frozen=True)
class TrainState:
    model: FragmentTreeTrainingModel
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    initial_epoch: int
    global_step: int
    best_val_loss: float


@dataclass(frozen=True)
class EpochLossMetrics:
    loss: float
    selection_loss: float
    intensity_loss: float
    samples: int
    steps: int = 0


def make_loss_metrics(
    *,
    total_loss: float,
    total_selection_loss: float,
    total_intensity_loss: float,
    total_samples: int,
    total_selection_samples: int,
    total_intensity_samples: int,
    steps: int = 0,
    empty_value: float = 0.0,
) -> EpochLossMetrics:
    if total_samples <= 0:
        return EpochLossMetrics(
            loss=empty_value,
            selection_loss=empty_value,
            intensity_loss=empty_value,
            samples=0,
            steps=int(steps),
        )
    return EpochLossMetrics(
        loss=total_loss / total_samples,
        selection_loss=total_selection_loss / max(total_selection_samples, 1),
        intensity_loss=total_intensity_loss / max(total_intensity_samples, 1),
        samples=total_samples,
        steps=int(steps),
    )


def nan_loss_metrics() -> EpochLossMetrics:
    return EpochLossMetrics(
        loss=float("nan"),
        selection_loss=float("nan"),
        intensity_loss=float("nan"),
        samples=0,
        steps=0,
    )


def sample_training_edges(
    structure,
    *,
    max_edges_per_sample: Optional[int],
):
    """Teacher-force required edges and fill the window with random negatives.

    This is deliberately a one-shot training/validation sampler.  Sequential
    survivor windows are reserved for inference.
    """
    if max_edges_per_sample is None or structure.sample_edge_index.numel() == 0:
        return structure
    limit = int(max_edges_per_sample)
    if limit <= 0:
        raise ValueError("max_edges_per_sample must be positive or None.")

    existing: Dict[int, List[int]] = {}
    for sample_id, edge_id in structure.sample_edge_index.detach().cpu().t().tolist():
        if int(edge_id) >= 0:
            existing.setdefault(int(sample_id), []).append(int(edge_id))

    required: Dict[int, set[int]] = {}
    if structure.target_edge_index.numel() > 0:
        for sample_id, edge_id in structure.target_edge_index.detach().cpu().t().tolist():
            required.setdefault(int(sample_id), set()).add(int(edge_id))

    # Intermediate molecular nodes on a target path must also remain reachable.
    edge_dst = structure.edge_index[1].detach().cpu().long()
    assignment_samples = structure.target_sample_index.detach().cpu().long()
    expand_nodes = structure.target_expand_node_index.detach().cpu().long()
    expand_ptr = structure.terminal_expand_ptr.detach().cpu().long()
    for assignment_id, sample_id in enumerate(assignment_samples.tolist()):
        start = int(expand_ptr[assignment_id].item())
        end = int(expand_ptr[assignment_id + 1].item())
        for node_id in expand_nodes[start:end].tolist():
            incoming = (edge_dst == int(node_id)).nonzero(as_tuple=False).view(-1)
            required.setdefault(int(sample_id), set()).update(
                int(edge_id) for edge_id in incoming.tolist()
            )

    sampled_pairs: List[Tuple[int, int]] = []
    for sample_id, raw_edges in sorted(existing.items()):
        candidates = sorted(set(raw_edges))
        must_keep = sorted(required.get(sample_id, set()).intersection(candidates))
        if len(must_keep) > limit:
            raise ValueError(
                f"sample {sample_id} has {len(must_keep)} required edges, "
                f"which exceeds max_edges_per_sample={limit}."
            )
        negatives = [edge_id for edge_id in candidates if edge_id not in set(must_keep)]
        remaining = limit - len(must_keep)
        if len(negatives) > remaining:
            order = torch.randperm(len(negatives))[:remaining].tolist()
            negatives = [negatives[index] for index in order]
        sampled_pairs.extend((sample_id, edge_id) for edge_id in must_keep + negatives)

    device = structure.sample_edge_index.device
    sampled = torch.tensor(sampled_pairs, dtype=torch.long, device=device).t().contiguous()
    if not sampled_pairs:
        sampled = torch.empty((2, 0), dtype=torch.long, device=device)
    return replace(structure, sample_edge_index=sampled)


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("PyYAML is required to read YAML config files.")
            return dict(yaml.safe_load(f) or {})
        return dict(json.load(f))


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if output_path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        else:
            json.dump(config, f, indent=2)


def _checkpoint_dict(path: str | Path, *, label: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{label} checkpoint must contain a dictionary: {path}")
    return checkpoint


def _fragmenter_params_from_file(path: str | Path) -> Dict[str, Any]:
    data = load_config(path)
    if "probability_model_params" in data:
        data = dict(data["probability_model_params"])
    if "fragmenter_params" in data:
        data = dict(data["fragmenter_params"])
    if "fragment_ion_tree_builder" not in data:
        raise KeyError(
            "Fragmenter parameter file must contain fragment_ion_tree_builder "
            f"(directly or below fragmenter_params): {path}"
        )
    return dict(data)


def build_model_config_from_pretrained(
    *,
    mol_encoder_checkpoint: str | Path,
    cleavage_edge_fnet_checkpoint: str | Path,
    fragmenter_params: Dict[str, Any],
    condition_encoder_params: Dict[str, Any],
    tree_encoder_params: Dict[str, Any],
    dropout: float,
    generator_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build spectrum-training config from the two pretrained checkpoints."""
    mol_checkpoint = _checkpoint_dict(
        mol_encoder_checkpoint, label="MolEncoder"
    )
    cleavage_checkpoint = _checkpoint_dict(
        cleavage_edge_fnet_checkpoint, label="CleavageEdgeFNet"
    )
    mol_params = dict(mol_checkpoint.get("mol_encoder_params") or {})
    if not mol_params:
        raise KeyError(
            "MolEncoder checkpoint is missing mol_encoder_params: "
            f"{mol_encoder_checkpoint}"
        )
    cleavage_params_full = dict(
        cleavage_checkpoint.get("cleavage_edge_fnet_params") or {}
    )
    if not cleavage_params_full:
        raise KeyError(
            "CleavageEdgeFNet checkpoint is missing cleavage_edge_fnet_params: "
            f"{cleavage_edge_fnet_checkpoint}"
        )
    pattern_set = cleavage_params_full.get("cleavage_pattern_set_params")
    if not isinstance(pattern_set, dict):
        raise KeyError(
            "CleavageEdgeFNet checkpoint is missing "
            f"cleavage_pattern_set_params: {cleavage_edge_fnet_checkpoint}"
        )
    cleavage_mol_params = dict(cleavage_checkpoint.get("mol_encoder_params") or {})
    if not cleavage_mol_params:
        raise KeyError(
            "CleavageEdgeFNet checkpoint is missing mol_encoder_params: "
            f"{cleavage_edge_fnet_checkpoint}"
        )
    if mol_params != cleavage_mol_params:
        raise ValueError(
            "The MolEncoder checkpoint does not match the MolEncoder configuration "
            "used to pretrain the CleavageEdgeFNet. Select the same MolEncoder "
            "checkpoint/configuration."
        )

    dimension_pairs = (
        ("graph_dim", "mol_dim"),
        ("node_dim", "atom_dim"),
    )
    incompatible_dimensions = {
        f"mol_encoder_params.{mol_key} / cleavage_edge_fnet_params.{edge_key}": (
            mol_params.get(mol_key),
            cleavage_params_full.get(edge_key),
        )
        for mol_key, edge_key in dimension_pairs
        if mol_params.get(mol_key) != cleavage_params_full.get(edge_key)
    }
    if incompatible_dimensions:
        raise ValueError(
            "The MolEncoder and CleavageEdgeFNet checkpoints were trained with "
            "incompatible encoder dimensions: "
            f"{incompatible_dimensions}. Select the MolEncoder checkpoint used "
            "to pretrain the cleavage model."
        )

    fragmenter_params = dict(fragmenter_params)
    tree_builder_params = dict(fragmenter_params["fragment_ion_tree_builder"])
    data_pattern_set = tree_builder_params.get("cleavage_pattern_set")
    if data_pattern_set != pattern_set:
        raise ValueError(
            "The fragmenter saved with the training data has a cleavage pattern "
            "set that is incompatible with the CleavageEdgeFNet checkpoint."
        )

    cleavage_params = {
        key: cleavage_params_full[key]
        for key in ("feature_dim", "fc_dims")
        if key in cleavage_params_full
    }
    missing_cleavage_params = {
        "feature_dim", "fc_dims"
    } - set(cleavage_params)
    if missing_cleavage_params:
        raise KeyError(
            "CleavageEdgeFNet checkpoint is missing construction parameters: "
            f"{sorted(missing_cleavage_params)}"
        )

    config: Dict[str, Any] = {
        "probability_model_params": {
            "mol_encoder_params": mol_params,
            "condition_encoder_params": dict(condition_encoder_params),
            "cleavage_edge_fnet_params": cleavage_params,
            "tree_encoder_params": dict(tree_encoder_params),
            "fragmenter_params": fragmenter_params,
            "dropout": float(dropout),
        },
        "mol_encoder_checkpoint": str(mol_encoder_checkpoint),
        "cleavage_edge_fnet_checkpoint": str(cleavage_edge_fnet_checkpoint),
        "freeze_mol_encoder": True,
        "freeze_cleavage_edge_fnet": True,
    }
    config.update(dict(generator_params or {}))
    return config


def load_generator(model_config: Dict[str, Any], device: torch.device) -> FragmentSpectrumGenerator:
    params = model_config.get("params", model_config)
    generator = FragmentSpectrumGenerator(**params).to(device)
    return generator


def validate_preprocessing_compatibility(
    *,
    project_dir: str | Path,
    model_config: Dict[str, Any],
    preprocessing_config_path: Optional[str | Path] = None,
) -> None:
    config_path = (
        Path(preprocessing_config_path)
        if preprocessing_config_path is not None
        else Path(project_dir) / "config" / "preprocessing_config.json"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Preprocessing config not found: {config_path}. Training requires the "
            "immutable definitions used to create the structure data."
        )
    preprocessing = load_config(config_path)
    params = model_config.get("params", model_config).get("probability_model_params", {})
    # AtomFeatureLayer canonicalizes symbols with sorted(set(symbols)).  Compare
    # the effective feature-column order, not the user-facing input order saved
    # by preprocessing.
    model_symbols = tuple(
        sorted(set(params.get("mol_encoder_params", {}).get("symbols", ())))
    )
    expected_symbols = tuple(sorted(set(preprocessing.get("symbols", ()))))
    if model_symbols != expected_symbols:
        raise ValueError(
            "Model symbols do not match preprocessing data: "
            f"expected={expected_symbols}, actual={model_symbols}."
        )

    from clefts.domain.fragment import Fragmenter

    expected_fragmenter = Fragmenter.from_dict(
        preprocessing["fragmenter_params"]
    ).to_dict()
    actual_fragmenter = Fragmenter.from_dict(params["fragmenter_params"]).to_dict()
    if actual_fragmenter != expected_fragmenter:
        raise ValueError(
            "Model fragmenter_params do not match the immutable preprocessing "
            f"configuration in {config_path}."
        )


def build_training_model(
    model_config: Dict[str, Any],
    device: torch.device,
) -> FragmentTreeTrainingModel:
    generator = load_generator(model_config, device=device)
    model = FragmentTreeTrainingModel(
        generator.candidate_selector,
        intensity_predictor=generator.formula_intensity_predictor,
    ).to(device)
    model.set_checkpoint_model_config(model_config)
    return model


OPTIMIZER_RUNTIME_KEYS = {"grad_clip_norm"}


def optimizer_config_for_torch(optimizer_info: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(optimizer_info)
    for key in OPTIMIZER_RUNTIME_KEYS:
        config.pop(key, None)
    return config


def build_optimizer_and_scheduler(
    model: FragmentTreeTrainingModel,
    optimizer_info: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    optimizer, scheduler = get_optimizer(
        model,
        optimizer_config_for_torch(optimizer_info),
        is_return_scheduler=True,
    )
    return optimizer, scheduler


def create_training_model_from_checkpoint_config(
    model_config: Dict[str, Any],
) -> FragmentTreeTrainingModel:
    return build_training_model(model_config, device=torch.device("cpu"))


def step_scheduler(
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    val_loss: float,
) -> None:
    if scheduler is None:
        return
    scheduler_name = scheduler.__class__.__name__.lower()
    if "plateau" in scheduler_name:
        scheduler.step(val_loss)
    else:
        scheduler.step()


def normalize_train_config(
    project_dir: str | Path,
    train_config: Dict[str, Any],
) -> Dict[str, Any]:
    project_path = Path(project_dir)
    config = dict(train_config)

    config["experiment_name"] = config.get("experiment_name") or DEFAULT_EXPERIMENT_NAME
    config["ckpt_id"] = None if config.get("ckpt_id") in {"", None} else config.get("ckpt_id")
    config["batch_size"] = int(config.get("batch_size", 1))
    config["device"] = str(config.get("device", "cpu"))
    config["epoch"] = int(config.get("epoch", config.get("epochs", 10)))

    validation_interval_steps = config.get("validation_interval_steps", 100)
    config["validation_interval_steps"] = (
        None
        if validation_interval_steps in {None, ""}
        else int(validation_interval_steps)
    )
    if (
        config["validation_interval_steps"] is not None
        and config["validation_interval_steps"] <= 0
    ):
        raise ValueError("validation_interval_steps must be positive when specified.")
    train_log_interval_steps = config.get("train_log_interval_steps", 100)
    config["train_log_interval_steps"] = (
        None if train_log_interval_steps in {None, ""} else int(train_log_interval_steps)
    )
    if (
        config["train_log_interval_steps"] is not None
        and config["train_log_interval_steps"] <= 0
    ):
        raise ValueError("train_log_interval_steps must be positive when specified.")

    config["save_interval"] = int(config.get("save_interval", 1))
    save_interval_steps = config.get(
        "save_interval_steps",
        config.get("save_interval_iters", 100),
    )
    config["save_interval_steps"] = (
        None
        if save_interval_steps in {None, ""}
        else int(save_interval_steps)
    )
    if config["save_interval"] < 0:
        raise ValueError("save_interval must be zero or positive.")
    if config["save_interval_steps"] is not None and config["save_interval_steps"] < 0:
        raise ValueError("save_interval_steps must be zero or positive when specified.")

    optimizer_info = dict(DEFAULT_OPTIMIZER_INFO)
    optimizer_info.update(dict(config.get("optimizer") or {}))
    config["optimizer"] = optimizer_info
    config["early_stopping"] = dict(config.get("early_stopping", {}))

    training_structure_dir = config.get(
        "training_structure_dir",
        config.get("traing_structure_dir"),
    )
    validation_structure_dir = config.get(
        "validation_structure_dir",
        config.get("val_structure_dir"),
    )
    config["training_structure_dir"] = str(
        training_structure_dir or project_path / "train_structures" / "data"
    )
    config["validation_structure_dir"] = str(
        validation_structure_dir or project_path / "validation_structures" / "data"
    )

    validation_valid_records_file = config.get(
        "validation_valid_records_file",
        config.get("validation_msdataset_file"),
    )
    if not validation_valid_records_file:
        validation_valid_records_file = (
            Path(config["validation_structure_dir"]) / "valid_records.msds"
        )
    config["validation_valid_records_file"] = str(validation_valid_records_file)
    config["shuffle"] = bool(config.get("shuffle", True))
    config["validate_at_start"] = bool(config.get("validate_at_start", False))

    return config


def build_train_config(
    *,
    project_dir: str | Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    ckpt_id: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
    epoch: int = 10,
    validation_interval_steps: Optional[int] = 100,
    train_log_interval_steps: Optional[int] = 100,
    save_interval: int = 1,
    save_interval_steps: Optional[int] = 100,
    optimizer_name: str = "AdamW",
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    grad_clip_norm: Optional[float] = 1.0,
    training_structure_dir: Optional[str | Path] = None,
    validation_structure_dir: Optional[str | Path] = None,
    shuffle: bool = True,
    validate_at_start: bool = False,
) -> Dict[str, Any]:
    optimizer_info: Dict[str, Any] = {
        "name": optimizer_name,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }
    if grad_clip_norm is not None:
        optimizer_info["grad_clip_norm"] = float(grad_clip_norm)

    return normalize_train_config(
        project_dir,
        {
            "experiment_name": experiment_name,
            "ckpt_id": ckpt_id,
            "batch_size": int(batch_size),
            "device": device,
            "epoch": int(epoch),
            "validation_interval_steps": validation_interval_steps,
            "train_log_interval_steps": train_log_interval_steps,
            "save_interval": int(save_interval),
            "save_interval_steps": save_interval_steps,
            "optimizer": optimizer_info,
            "training_structure_dir": (
                None if training_structure_dir is None else str(training_structure_dir)
            ),
            "validation_structure_dir": (
                None if validation_structure_dir is None else str(validation_structure_dir)
            ),
            "shuffle": bool(shuffle),
            "validate_at_start": bool(validate_at_start),
        },
    )


def prepare_train_from_config(
    project_dir: str | Path,
    train_config: Dict[str, Any],
    *,
    train_config_source: Optional[str | Path] = None,
) -> Tuple[Path, Optional[str], int, torch.device, int, int, Optional[int], Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[int], Path]:
    train_config = normalize_train_config(project_dir, train_config)

    load_name = train_config.get("experiment_name") or DEFAULT_EXPERIMENT_NAME
    experiment_dir = Path(project_dir) / "experiments" / str(load_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    ckpt_id = train_config.get("ckpt_id")
    batch_size = int(train_config["batch_size"])
    device = torch.device(train_config["device"])
    epochs = int(train_config["epoch"])
    save_interval = int(train_config["save_interval"])
    save_interval_steps = train_config.get("save_interval_steps")
    save_interval_steps = (
        None
        if save_interval_steps in {None, ""}
        else int(save_interval_steps)
    )
    validation_interval_steps = train_config.get("validation_interval_steps")
    validation_interval_steps = (
        None
        if validation_interval_steps in {None, ""}
        else int(validation_interval_steps)
    )
    optimizer_info = dict(train_config["optimizer"])
    early_stopping_info = dict(train_config.get("early_stopping", {}))

    dataset_info = {
        "training_structure_dir": str(train_config["training_structure_dir"]),
        "validation_structure_dir": str(train_config["validation_structure_dir"]),
        "validation_valid_records_file": str(train_config["validation_valid_records_file"]),
        "shuffle": bool(train_config.get("shuffle", True)),
        "validation_interval_steps": validation_interval_steps,
        "train_log_interval_steps": train_config.get("train_log_interval_steps"),
        "validate_at_start": bool(train_config.get("validate_at_start", False)),
        "pattern": "*.pt",
    }

    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = experiment_dir / "runs" / now_str
    run_dir.mkdir(parents=True, exist_ok=True)
    if train_config_source is not None:
        shutil.copy(train_config_source, run_dir / Path(train_config_source).name)
    else:
        save_config(train_config, run_dir / DEFAULT_TRAIN_CONFIG_NAME)

    return (
        experiment_dir,
        None if ckpt_id in ("", None) else str(ckpt_id),
        batch_size,
        device,
        epochs,
        save_interval,
        save_interval_steps,
        optimizer_info,
        early_stopping_info,
        dataset_info,
        validation_interval_steps,
        run_dir,
    )


def prepare_train(
    project_dir: str | Path,
    train_config_path: str | Path,
    *,
    root_run_dir: Optional[str | Path] = None,
) -> Tuple[Path, Optional[str], int, torch.device, int, int, Optional[int], Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[int], Path]:
    train_config = load_config(train_config_path)
    prepared = prepare_train_from_config(
        project_dir,
        train_config,
        train_config_source=train_config_path,
    )
    if root_run_dir is None:
        return prepared

    (
        experiment_dir,
        ckpt_id,
        batch_size,
        device,
        epochs,
        save_interval,
        save_interval_steps,
        optimizer_info,
        early_stopping_info,
        dataset_info,
        validation_interval_steps,
        old_run_dir,
    ) = prepared
    now_str = old_run_dir.name
    run_dir = Path(root_run_dir) / now_str
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(train_config_path, run_dir / Path(train_config_path).name)
    return (
        experiment_dir,
        ckpt_id,
        batch_size,
        device,
        epochs,
        save_interval,
        save_interval_steps,
        optimizer_info,
        early_stopping_info,
        dataset_info,
        validation_interval_steps,
        run_dir,
    )


def setup_dataset(
    dataset_info: Dict[str, Any],
    batch_size: int,
    *,
    num_workers: int = 0,
) -> Tuple[FragmentTreeStructureFileDataset, FragmentTreeStructureFileDataset, DataLoader, DataLoader, Dict[str, Any]]:
    pattern = str(dataset_info.get("pattern", "*.pt"))
    train_dataset = FragmentTreeStructureFileDataset(
        Path(dataset_info["training_structure_dir"]),
        pattern=pattern,
    )
    val_dataset = FragmentTreeStructureFileDataset(
        Path(dataset_info["validation_structure_dir"]),
        pattern=pattern,
    )
    if len(train_dataset) == 0:
        raise ValueError("training_structure_dir contains no training structure files.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fragment_tree_structure_items,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(dataset_info.get("shuffle", True)),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    extra_data = {
        "training_structure_dir": str(dataset_info["training_structure_dir"]),
        "validation_structure_dir": str(dataset_info["validation_structure_dir"]),
        "validation_valid_records_file": str(
            dataset_info["validation_valid_records_file"]
        ),
        "pattern": pattern,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "validation_interval_steps": dataset_info.get("validation_interval_steps"),
        "train_log_interval_steps": dataset_info.get("train_log_interval_steps"),
    }
    return train_dataset, val_dataset, train_loader, val_loader, extra_data


def load_or_initialize_state(
    *,
    model_config: Dict[str, Any],
    device: torch.device,
    optimizer_info: Dict[str, Any],
    ckpt_manager: CheckPointManager,
    ckpt_id: Optional[str],
) -> TrainState:
    if ckpt_id is None:
        ckpt_manager.checkout_base()
        model = build_training_model(model_config, device=device)
        optimizer, scheduler = build_optimizer_and_scheduler(model, optimizer_info)
        return TrainState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            initial_epoch=1,
            global_step=0,
            best_val_loss=float("inf"),
        )

    ckpt_id_int = int(ckpt_id)
    if CkptNode.is_branch_node(ckpt_id_int):
        ckpt_node = ckpt_manager.checkout_latest(ckpt_id_int)
    else:
        ckpt_node = ckpt_manager.load_ckpt(ckpt_id_int)

    (
        model,
        saved_epoch,
        global_step,
        optimizer,
        scheduler,
        saved_optimizer_info,
        checkpoint_extra_data,
    ) = ckpt_node.load_model(
        create_training_model_from_checkpoint_config,
        device=device,
    )

    checkpoint_extra_data = dict(checkpoint_extra_data or {})
    if optimizer is None:
        optimizer, scheduler = build_optimizer_and_scheduler(model, optimizer_info)
    elif scheduler is None and saved_optimizer_info is not None:
        optimizer_info = dict(saved_optimizer_info)

    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        initial_epoch=int(saved_epoch) + 1,
        global_step=int(global_step),
        best_val_loss=float(checkpoint_extra_data.get("best_val_loss", float("inf"))),
    )


def run_epoch(
    *,
    model: FragmentTreeTrainingModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None,
    desc: str,
    start_global_step: int = 0,
    validation_interval_steps: Optional[int] = None,
    on_validation_step: Optional[
        Callable[[int, EpochLossMetrics, EpochLossMetrics], None]
    ] = None,
    train_log_interval_steps: Optional[int] = 100,
    on_train_log_step: Optional[
        Callable[[int, EpochLossMetrics, EpochLossMetrics], None]
    ] = None,
    on_step_end: Optional[
        Callable[[int, EpochLossMetrics, EpochLossMetrics], None]
    ] = None,
    max_training_edges: Optional[int] = None,
) -> EpochLossMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    if max_training_edges is None:
        max_training_edges = model.candidate_selector.max_edges_per_step

    total_loss = 0.0
    total_selection_loss = 0.0
    total_intensity_loss = 0.0
    total_samples = 0
    total_selection_samples = 0
    total_intensity_samples = 0
    window_loss = 0.0
    window_selection_loss = 0.0
    window_intensity_loss = 0.0
    window_samples = 0
    window_selection_samples = 0
    window_intensity_samples = 0
    step_count = 0
    iterator = tqdm(loader, desc=desc)

    for batch in iterator:
        try:
            structure = batch["structure"].to(device)
            structure = sample_training_edges(
                structure,
                max_edges_per_sample=max_training_edges,
            )
            num_samples = int(structure.num_samples)
            sample_weight = max(num_samples, 1)

            with torch.set_grad_enabled(is_train):
                output = model(structure)
                loss = output["loss"]

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            step_count += 1
            selection_loss = output.get("selection_loss")
            intensity_loss = output.get("intensity_loss")
            loss_value = float(loss.detach().cpu().item())
            selection_loss_value = (
                float(selection_loss.detach().cpu().item())
                if selection_loss is not None
                else float("nan")
            )
            intensity_loss_value = (
                float(intensity_loss.detach().cpu().item())
                if intensity_loss is not None
                else float("nan")
            )

            total_loss += loss_value * sample_weight
            window_loss += loss_value * sample_weight
            if not math.isnan(selection_loss_value):
                total_selection_loss += selection_loss_value * sample_weight
                total_selection_samples += sample_weight
                window_selection_loss += selection_loss_value * sample_weight
                window_selection_samples += sample_weight
            if not math.isnan(intensity_loss_value):
                total_intensity_loss += intensity_loss_value * sample_weight
                total_intensity_samples += sample_weight
                window_intensity_loss += intensity_loss_value * sample_weight
                window_intensity_samples += sample_weight
            total_samples += sample_weight
            window_samples += sample_weight

            cumulative_metrics = make_loss_metrics(
                total_loss=total_loss,
                total_selection_loss=total_selection_loss,
                total_intensity_loss=total_intensity_loss,
                total_samples=total_samples,
                total_selection_samples=total_selection_samples,
                total_intensity_samples=total_intensity_samples,
                steps=step_count,
            )
            window_metrics = make_loss_metrics(
                total_loss=window_loss,
                total_selection_loss=window_selection_loss,
                total_intensity_loss=window_intensity_loss,
                total_samples=window_samples,
                total_selection_samples=window_selection_samples,
                total_intensity_samples=window_intensity_samples,
                steps=step_count,
            )
            iterator.set_postfix(
                loss=cumulative_metrics.loss,
                selection_loss=cumulative_metrics.selection_loss,
                intensity_loss=cumulative_metrics.intensity_loss,
                window_loss=window_metrics.loss,
            )

            global_step = int(start_global_step) + step_count
            validation_due = (
                is_train
                and validation_interval_steps is not None
                and validation_interval_steps > 0
                and on_validation_step is not None
                and global_step % validation_interval_steps == 0
            )
            if validation_due:
                on_validation_step(global_step, cumulative_metrics, window_metrics)
                model.train(is_train)

            train_log_due = (
                is_train
                and train_log_interval_steps is not None
                and train_log_interval_steps > 0
                and on_train_log_step is not None
                and global_step % train_log_interval_steps == 0
            )
            if train_log_due:
                on_train_log_step(global_step, cumulative_metrics, window_metrics)

            if validation_due or train_log_due:
                window_loss = 0.0
                window_selection_loss = 0.0
                window_intensity_loss = 0.0
                window_samples = 0
                window_selection_samples = 0
                window_intensity_samples = 0

            if on_step_end is not None:
                on_step_end(global_step, cumulative_metrics, window_metrics)
        except Exception as exc:
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            # print(
            #     "[WARN] Skipping failed batch "
            #     f"in {desc} at attempted_step={int(start_global_step) + step_count + 1}: "
            #     f"{type(exc).__name__}: {exc}"
            # )
            # traceback.print_exc()
            continue

    return make_loss_metrics(
        total_loss=total_loss,
        total_selection_loss=total_selection_loss,
        total_intensity_loss=total_intensity_loss,
        total_samples=total_samples,
        total_selection_samples=total_selection_samples,
        total_intensity_samples=total_intensity_samples,
        steps=step_count,
    )


def _match_peaks_one_to_one(
    target_peaks: np.ndarray,
    predicted_peaks: np.ndarray,
    *,
    mz_tolerance_da: float,
) -> List[Tuple[int, int]]:
    """Greedily make the closest one-to-one target/prediction peak matches."""
    if mz_tolerance_da <= 0:
        raise ValueError("mz_tolerance_da must be positive.")
    if len(target_peaks) == 0 or len(predicted_peaks) == 0:
        return []

    candidates: List[Tuple[float, int, int]] = []
    for target_index, target_mz in enumerate(target_peaks[:, 0]):
        errors = np.abs(predicted_peaks[:, 0] - target_mz)
        for predicted_index in np.flatnonzero(errors <= mz_tolerance_da).tolist():
            candidates.append(
                (float(errors[predicted_index]), int(target_index), int(predicted_index))
            )
    candidates.sort()

    matched_targets: set[int] = set()
    matched_predictions: set[int] = set()
    matches: List[Tuple[int, int]] = []
    for _, target_index, predicted_index in candidates:
        if target_index in matched_targets or predicted_index in matched_predictions:
            continue
        matched_targets.add(target_index)
        matched_predictions.add(predicted_index)
        matches.append((target_index, predicted_index))
    return matches


def calculate_peak_selection_metrics(
    predicted_dataset: MSDataset,
    target_dataset: MSDataset,
    *,
    mz_tolerance_da: float = PEAK_SELECTION_MZ_TOLERANCE_DA,
    top_k: Tuple[int, ...] = PEAK_SELECTION_TOP_K,
) -> Dict[str, np.ndarray]:
    """Calculate spectrum-wise selection and matched-intensity diagnostics."""
    count = min(len(target_dataset), len(predicted_dataset))
    values: Dict[str, List[float]] = {
        "selection_precision": [],
        "selection_recall": [],
        "selection_f1": [],
        "predicted_peak_count": [],
        "target_peak_count": [],
        "matched_peak_count": [],
        "selected_intensity_fraction": [],
        **{f"top{k}_recall": [] for k in top_k},
        "matched_intensity_mae": [],
        "matched_intensity_weighted_mae": [],
    }
    for spectrum_index in range(count):
        target_start = int(target_dataset.peaks.offsets[spectrum_index])
        target_end = int(target_dataset.peaks.offsets[spectrum_index + 1])
        predicted_start = int(predicted_dataset.peaks.offsets[spectrum_index])
        predicted_end = int(predicted_dataset.peaks.offsets[spectrum_index + 1])
        target_peaks = np.asarray(
            target_dataset.peaks.data[target_start:target_end], dtype=np.float64
        )
        predicted_peaks = np.asarray(
            predicted_dataset.peaks.data[predicted_start:predicted_end], dtype=np.float64
        )
        matches = _match_peaks_one_to_one(
            target_peaks, predicted_peaks, mz_tolerance_da=mz_tolerance_da
        )
        matched_target = np.asarray([pair[0] for pair in matches], dtype=np.int64)
        matched_predicted = np.asarray([pair[1] for pair in matches], dtype=np.int64)
        target_count = len(target_peaks)
        predicted_count = len(predicted_peaks)

        precision = len(matches) / predicted_count if predicted_count else 0.0
        recall = len(matches) / target_count if target_count else float("nan")
        values["selection_precision"].append(precision)
        values["selection_recall"].append(recall)
        values["selection_f1"].append(
            2.0 * precision * recall / (precision + recall)
            if target_count and precision + recall > 0
            else (0.0 if target_count else float("nan"))
        )
        values["predicted_peak_count"].append(float(predicted_count))
        values["target_peak_count"].append(float(target_count))
        values["matched_peak_count"].append(float(len(matches)))
        target_intensity = (
            np.clip(target_peaks[:, 1], 0.0, None)
            if target_count
            else np.empty(0, dtype=np.float64)
        )
        total_intensity = float(target_intensity.sum())
        selected_intensity = (
            float(target_intensity[matched_target].sum()) if len(matches) else 0.0
        )
        values["selected_intensity_fraction"].append(
            selected_intensity / total_intensity if total_intensity > 0 else float("nan")
        )

        intensity_rank = np.argsort(target_intensity, kind="stable")[::-1]
        matched_target_set = set(matched_target.tolist())
        for k in top_k:
            denominator = min(int(k), target_count)
            selected_top_k = sum(
                int(index) in matched_target_set for index in intensity_rank[:denominator]
            )
            values[f"top{k}_recall"].append(
                selected_top_k / denominator if denominator else float("nan")
            )

        if matches:
            predicted_intensity = np.clip(predicted_peaks[:, 1], 0.0, None)
            target_scale = max(float(target_intensity.max()), 1e-12)
            predicted_scale = max(float(predicted_intensity.max()), 1e-12)
            target_normalized = target_intensity[matched_target] / target_scale
            predicted_normalized = predicted_intensity[matched_predicted] / predicted_scale
            absolute_error = np.abs(target_normalized - predicted_normalized)
            values["matched_intensity_mae"].append(float(absolute_error.mean()))
            weight_sum = float(target_normalized.sum())
            values["matched_intensity_weighted_mae"].append(
                float(np.dot(absolute_error, target_normalized) / weight_sum)
                if weight_sum > 0
                else float("nan")
            )
        else:
            values["matched_intensity_mae"].append(float("nan"))
            values["matched_intensity_weighted_mae"].append(float("nan"))

    return {
        name: np.asarray(metric_values, dtype=np.float64)
        for name, metric_values in values.items()
    }


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            **{name: float("nan") for name, _ in PEAK_SELECTION_QUANTILES},
        }
    quantiles = np.quantile(
        finite, [quantile for _, quantile in PEAK_SELECTION_QUANTILES]
    )
    return {
        "mean": float(finite.mean()),
        **{
            name: float(quantiles[index])
            for index, (name, _) in enumerate(PEAK_SELECTION_QUANTILES)
        },
    }


def evaluate_validation_cosine(
    *,
    model: FragmentTreeTrainingModel,
    dataset: MSDataset,
    batch_size: int = 128,
    writer=None,
    global_step: Optional[int] = None,
    output_dir: Optional[Path] = None,
    selection_metric_means: Optional[Dict[str, float]] = None,
) -> float:
    if model.intensity_predictor is None:
        return float("nan")

    device = next(model.parameters()).device
    predicted_dataset, target_dataset = predict_validation_msdataset(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
    )
    if predicted_dataset is None or target_dataset is None:
        return float("nan")

    count = min(len(target_dataset), len(predicted_dataset))
    if count <= 0:
        return float("nan")

    scores = cosine_similarity_pair(
        target_dataset,
        np.arange(count, dtype=np.int64),
        predicted_dataset,
        np.arange(count, dtype=np.int64),
        show_progress=True,
    )
    if scores.size == 0:
        return float("nan")

    val_cosine = float(scores.mean())
    selection_metrics = calculate_peak_selection_metrics(
        predicted_dataset, target_dataset
    )
    selection_summaries = {
        name: summarize_distribution(values)
        for name, values in selection_metrics.items()
    }
    if selection_metric_means is not None:
        selection_metric_means.update(
            {name: summary["mean"] for name, summary in selection_summaries.items()}
        )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        score_file = output_dir / "validation_cosine.tsv"
        write_header = not score_file.exists()
        with open(score_file, "a", encoding="utf-8", newline="") as f:
            tsv = csv.writer(f, delimiter="\t")
            if write_header:
                tsv.writerow(["global_step", "spectrum_index", "cosine_similarity"])
            for spectrum_index, score in enumerate(scores.tolist()):
                tsv.writerow([global_step, spectrum_index, float(score)])
        detail_file = output_dir / "validation_peak_selection.tsv"
        write_header = not detail_file.exists()
        with open(detail_file, "a", encoding="utf-8", newline="") as f:
            tsv = csv.writer(f, delimiter="\t")
            if write_header:
                tsv.writerow(["global_step", "spectrum_index", *selection_metrics])
            for spectrum_index in range(count):
                tsv.writerow(
                    [global_step, spectrum_index]
                    + [selection_metrics[name][spectrum_index] for name in selection_metrics]
                )
        summary_file = output_dir / "validation_peak_selection_summary.tsv"
        write_header = not summary_file.exists()
        with open(summary_file, "a", encoding="utf-8", newline="") as f:
            tsv = csv.writer(f, delimiter="\t")
            if write_header:
                tsv.writerow(
                    [
                        "global_step",
                        "metric",
                        "mean",
                        *[name for name, _ in PEAK_SELECTION_QUANTILES],
                    ]
                )
            for metric_name, summary in selection_summaries.items():
                tsv.writerow(
                    [global_step, metric_name, summary["mean"]]
                    + [summary[name] for name, _ in PEAK_SELECTION_QUANTILES]
                )
    if writer is not None and global_step is not None:
        cosine_summary = np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 1.0])
        writer.add_scalars(
            "similarity/cosine_distribution",
            {
                "min": float(cosine_summary[0]),
                "q1": float(cosine_summary[1]),
                "median": float(cosine_summary[2]),
                "q3": float(cosine_summary[3]),
                "max": float(cosine_summary[4]),
            },
            int(global_step),
        )
        for metric_name, summary in selection_summaries.items():
            finite_quantiles = {
                name: summary[name]
                for name, _ in PEAK_SELECTION_QUANTILES
                if math.isfinite(summary[name])
            }
            if finite_quantiles:
                writer.add_scalars(
                    f"peak_selection/{metric_name}_distribution",
                    finite_quantiles,
                    int(global_step),
                )
            if math.isfinite(summary["mean"]):
                writer.add_scalar(
                    f"peak_selection/{metric_name}_mean",
                    summary["mean"],
                    int(global_step),
                )
        log_validation_spectrum_quantiles(
            writer=writer,
            predicted_dataset=predicted_dataset,
            target_dataset=target_dataset,
            scores=scores,
            global_step=int(global_step),
        )
        writer.flush()
    with tqdm(total=1, desc="ValCosine", leave=True) as iterator:
        iterator.set_postfix(val_cosine=val_cosine)
        iterator.update(1)
    return val_cosine


def find_split_config(split_dir: str | Path, filename: str) -> Path:
    """Find a config saved in a structure split, accepting split/data paths."""
    split_path = Path(split_dir).resolve()
    candidates = []
    for directory in (split_path, split_path.parent, *split_path.parents):
        candidates.extend((directory / filename, directory / "config" / filename))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} in the structure split: {split_dir}"
    )


def load_and_validate_split_preprocessing(
    train_dir: str | Path,
    val_dir: str | Path,
) -> Tuple[Dict[str, Any], Path]:
    """Load immutable preprocessing data and require identical train/val settings."""
    train_preprocessing_path = find_split_config(
        train_dir, DEFAULT_PREPROCESSING_CONFIG_NAME
    )
    val_preprocessing_path = find_split_config(
        val_dir, DEFAULT_PREPROCESSING_CONFIG_NAME
    )
    train_preprocessing = load_config(train_preprocessing_path)
    val_preprocessing = load_config(val_preprocessing_path)
    if train_preprocessing != val_preprocessing:
        raise ValueError(
            "Training and validation structures were prepared with different "
            "symbols or fragmenter settings."
        )

    train_fragmenter_path = find_split_config(train_dir, "fragmenter.json")
    val_fragmenter_path = find_split_config(val_dir, "fragmenter.json")
    train_fragmenter = _fragmenter_params_from_file(train_fragmenter_path)
    val_fragmenter = _fragmenter_params_from_file(val_fragmenter_path)
    if train_fragmenter != val_fragmenter:
        raise ValueError(
            "Training and validation structure directories contain different "
            "fragmenter.json configurations."
        )
    if train_fragmenter != dict(train_preprocessing.get("fragmenter_params") or {}):
        raise ValueError(
            "fragmenter.json does not match preprocessing_config.json in the "
            "training structure directory."
        )
    return train_preprocessing, train_preprocessing_path


def log_validation_spectrum_quantiles(
    *, writer, predicted_dataset: MSDataset, target_dataset: MSDataset,
    scores: np.ndarray, global_step: int,
) -> None:
    """Show five representative mirror plots from high to low cosine."""
    if scores.size == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is unavailable; spectrum figures were skipped.")
        return

    ranked = np.argsort(scores)[::-1]
    positions = np.linspace(0, len(ranked) - 1, num=min(5, len(ranked))).round().astype(int)
    for level, position in enumerate(positions, start=1):
        spectrum_index = int(ranked[int(position)])
        target_start = int(target_dataset.peaks.offsets[spectrum_index])
        target_end = int(target_dataset.peaks.offsets[spectrum_index + 1])
        pred_start = int(predicted_dataset.peaks.offsets[spectrum_index])
        pred_end = int(predicted_dataset.peaks.offsets[spectrum_index + 1])
        target_peaks = target_dataset.peaks.data[target_start:target_end]
        predicted_peaks = predicted_dataset.peaks.data[pred_start:pred_end]
        figure, axis = plt.subplots(figsize=(10, 4))
        if len(target_peaks):
            axis.vlines(target_peaks[:, 0], 0, target_peaks[:, 1], color="black", label="measured")
        if len(predicted_peaks):
            axis.vlines(predicted_peaks[:, 0], 0, -predicted_peaks[:, 1], color="tab:red", label="generated")
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(xlabel="m/z", ylabel="intensity", title=f"cosine={float(scores[spectrum_index]):.4f}")
        axis.legend(loc="upper right")
        figure.tight_layout()
        writer.add_figure(
            f"validation_spectra/level_{level}_high_to_low",
            figure,
            global_step=global_step,
            close=True,
        )


def predict_validation_msdataset(
    *,
    model: FragmentTreeTrainingModel,
    dataset: MSDataset,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[Optional[MSDataset], Optional[MSDataset]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    feature_model = model.candidate_selector.feature_model
    builder = SingleFragmentTreeStructureBuilder(feature_model)
    groups = group_record_indexes_by_smiles(dataset, smiles_column="SMILES")

    target_record_indexes: List[int] = []
    predicted_data_parts: List[np.ndarray] = []
    predicted_lengths: List[int] = []

    predictor = FragmentTreeSpectrumPredictor(
        model.candidate_selector,
        model.intensity_predictor,
        expand_cleavages=True,
        normalize_intensity=True,
        max_edges_per_step=model.candidate_selector.max_edges_per_step,
        max_retained_edges=model.candidate_selector.max_retained_edges,
        max_next_cleavage_candidates=(
            model.candidate_selector.max_next_cleavage_candidates
        ),
    ).to(device)
    predictor.eval()

    for smiles, record_indexes in tqdm(
        groups.items(),
        desc="Building validation structures from MSDataset",
        mininterval=1.0,
    ):
        record_indexes = [int(index) for index in record_indexes]
        for batch_start in range(0, len(record_indexes), batch_size):
            batch_record_indexes = record_indexes[batch_start:batch_start + batch_size]
            sub_dataset = dataset[batch_record_indexes]
            builder.reset()
            try:
                sample_indexes = builder.add_same_smiles_dataset_first_cleavage(
                    sub_dataset,
                    precursor_mz_column="PrecursorMZ",
                    adduct_type_column="AdductType",
                    collision_energy_column="CollisionEnergy",
                    smiles_column="SMILES",
                    instrument_column=None,
                )
            except Exception as exc:
                print(f"[WARN] validation prediction skipped smiles={smiles!r}: {exc}")
                continue

            valid_pairs = [
                (int(record_index), int(sample_index))
                for record_index, sample_index in enumerate(sample_indexes.tolist())
                if int(sample_index) >= 0 and int(record_index) < len(batch_record_indexes)
            ]
            if not valid_pairs:
                continue

            # A record can pass metadata parsing and receive a sample index
            # while fragmentation produces no usable molecular nodes.  Such a
            # sample cannot be represented as FragmentTreeStructure and must
            # not abort validation for all remaining records.
            if len(builder.node_graph) == 0:
                print(
                    "[WARN] validation prediction skipped "
                    f"smiles={smiles!r}: fragmentation produced no nodes"
                )
                continue

            valid_pairs.sort(key=lambda pair: pair[1])
            try:
                structure = FragmentTreeStructure.from_structures(
                    [builder.to_structure()],
                    device=device,
                )
                with torch.no_grad():
                    output = predictor(structure)
            except Exception as exc:
                print(
                    "[WARN] validation prediction skipped "
                    f"smiles={smiles!r}: {type(exc).__name__}: {exc}"
                )
                continue

            batch_predicted_dataset = fragment_spectrum_output_to_msdataset(output)
            batch_peak_data = batch_predicted_dataset.peaks.data
            batch_offsets = batch_predicted_dataset.peaks.offsets
            record_index_by_sample_index = {
                sample_index: record_index
                for record_index, sample_index in valid_pairs
            }
            for sample_index in sorted(record_index_by_sample_index):
                if sample_index >= len(batch_offsets) - 1:
                    continue
                start = int(batch_offsets[sample_index])
                end = int(batch_offsets[sample_index + 1])
                predicted_data_parts.append(batch_peak_data[start:end])
                predicted_lengths.append(end - start)
                target_record_indexes.append(
                    batch_record_indexes[record_index_by_sample_index[sample_index]]
                )

    if not predicted_lengths or not target_record_indexes:
        return None, None

    predicted_data = (
        np.concatenate(predicted_data_parts, axis=0)
        if predicted_data_parts
        else np.empty((0, 2), dtype=np.float64)
    )
    predicted_offsets = np.empty(len(predicted_lengths) + 1, dtype=np.int64)
    predicted_offsets[0] = 0
    predicted_offsets[1:] = np.cumsum(np.asarray(predicted_lengths, dtype=np.int64))

    target_dataset = dataset[target_record_indexes].copy()
    predicted_dataset = target_dataset.copy()
    predicted_dataset.peaks.replace_data(predicted_data, predicted_offsets)
    return predicted_dataset, target_dataset


def save_managed_checkpoint(
    *,
    ckpt_manager: CheckPointManager,
    model: FragmentTreeTrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    optimizer_info: Dict[str, Any],
    extra_data: Dict[str, Any],
    comment: str,
) -> None:
    ckpt_node = ckpt_manager.checkout_new_ckpt()
    if comment == "iter_interval":
        ckpt_node.name = f"epoch_{epoch:04d}_step_{global_step:08d}"
    else:
        ckpt_node.name = f"epoch_{epoch:04d}"
    ckpt_node.epoch = int(epoch)
    ckpt_node.iter = int(global_step)
    ckpt_node.comment = comment

    checkpoint_extra_data = dict(extra_data)
    checkpoint_extra_data.update(
        {
            "best_val_loss": float(best_val_loss),
            "optimizer_info": dict(optimizer_info),
        }
    )
    ckpt_node.save_model(
        model=model,
        epoch=epoch,
        iter=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_info=optimizer_config_for_torch(optimizer_info),
        extra_data=checkpoint_extra_data,
    )
    ckpt_manager.flush_metrics(flush_dir=ckpt_node.dir)
    ckpt_manager.flush_run_dir_path(flush_dir=ckpt_node.dir)
    ckpt_manager.update()


def main(
    *,
    model_config: Dict[str, Any],
    experiment_dir: Path,
    ckpt_id: Optional[str],
    device: torch.device,
    epoch: int,
    save_interval: int,
    save_interval_steps: Optional[int],
    batch_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_info: Dict[str, Any],
    early_stopping_info: Dict[str, Any],
    run_dir: Path,
    extra_data: Dict[str, Any],
    validation_interval_steps: Optional[int] = None,
    validate_at_start: bool = False,
) -> None:
    ckpt_manager = CheckPointManager(str(experiment_dir))
    ckpt_manager.set_run_dir(str(run_dir))
    ckpt_manager.initialize_metrics(list(METRIC_COLUMNS))

    state = load_or_initialize_state(
        model_config=model_config,
        device=device,
        optimizer_info=optimizer_info,
        ckpt_manager=ckpt_manager,
        ckpt_id=ckpt_id,
    )
    model = state.model
    optimizer = state.optimizer
    scheduler = state.scheduler
    global_step = state.global_step
    best_val_loss = state.best_val_loss

    patience = early_stopping_info.get("patience")
    patience = None if patience in {None, ""} else int(patience)
    min_delta = float(early_stopping_info.get("min_delta", 0.0))
    topk = int(early_stopping_info.get("topk", 1))
    bad_epochs = 0
    grad_clip_norm = optimizer_info.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm in {None, ""} else float(grad_clip_norm)

    max_epoch = state.initial_epoch + int(epoch) - 1
    writer = ckpt_manager.summary_writer
    # Create a useful dashboard immediately. Without this, a new run contains
    # only the event-file header until its first scheduled validation.
    writer.add_scalar("optimizer/lr", float(optimizer.param_groups[0]["lr"]), global_step)
    writer.flush()
    validation_valid_records_file = Path(extra_data["validation_valid_records_file"])
    train_log_interval_steps = extra_data.get("train_log_interval_steps", 100)
    validation_dataset = (
        MSDataset.load(str(validation_valid_records_file))
        if validation_valid_records_file.exists()
        else None
    )
    if validation_dataset is None:
        print(
            f"[WARN] validation valid-record MSDataset was not found: "
            f"{validation_valid_records_file}"
        )

    def add_scalar_if_finite(tag: str, value: float, step: int) -> None:
        if not math.isnan(float(value)):
            writer.add_scalar(tag, float(value), step)

    def add_scalars_if_finite(
        main_tag: str,
        values: Dict[str, float],
        step: int,
    ) -> None:
        finite_values = {
            key: float(value)
            for key, value in values.items()
            if not math.isnan(float(value))
        }
        if finite_values:
            writer.add_scalars(main_tag, finite_values, step)

    def evaluate_current_validation(
        desc: str,
        *,
        step_value: int,
    ) -> Tuple[EpochLossMetrics, float, Dict[str, float]]:
        if val_loader is None or len(val_loader) <= 0:
            return nan_loss_metrics(), float("nan"), {}
        selection_metric_means: Dict[str, float] = {}
        with torch.no_grad():
            metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                desc=desc,
            )
            cosine = (
                evaluate_validation_cosine(
                    model=model,
                    dataset=validation_dataset,
                    batch_size=batch_size,
                    writer=writer,
                    global_step=step_value,
                    output_dir=run_dir / "validation",
                    selection_metric_means=selection_metric_means,
                )
                if validation_dataset is not None
                else float("nan")
            )
        return metrics, cosine, selection_metric_means

    def log_training_metrics(
        *,
        event: str,
        epoch_value: int,
        step_value: int,
        train_metrics: EpochLossMetrics,
        train_window_metrics: Optional[EpochLossMetrics],
        val_metrics: EpochLossMetrics,
        val_cosine: float,
        val_peak_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        lr = float(optimizer.param_groups[0]["lr"])
        window_metrics = train_window_metrics or nan_loss_metrics()
        metric_row = {
            "event": event,
            "epoch": int(epoch_value),
            "global_step": int(step_value),
            "train_loss": train_metrics.loss,
            "train_selection_loss": train_metrics.selection_loss,
            "train_intensity_loss": train_metrics.intensity_loss,
            "train_window_loss": window_metrics.loss,
            "train_window_selection_loss": window_metrics.selection_loss,
            "train_window_intensity_loss": window_metrics.intensity_loss,
            "val_loss": val_metrics.loss,
            "val_selection_loss": val_metrics.selection_loss,
            "val_intensity_loss": val_metrics.intensity_loss,
            "val_cosine": val_cosine,
            **{
                f"val_{name}": (val_peak_metrics or {}).get(name, float("nan"))
                for name in PEAK_SELECTION_METRIC_NAMES
            },
            "lr": lr,
        }
        ckpt_manager.log_metrics(**metric_row)
        ckpt_manager.flush_metrics(flush_dir=str(run_dir))
        add_scalars_if_finite(
            "loss/total",
            {
                "train": train_metrics.loss,
                "train_window": window_metrics.loss,
                "val": val_metrics.loss,
            },
            step_value,
        )
        add_scalars_if_finite(
            "loss/selection",
            {
                "train": train_metrics.selection_loss,
                "train_window": window_metrics.selection_loss,
                "val": val_metrics.selection_loss,
            },
            step_value,
        )
        add_scalars_if_finite(
            "loss/intensity",
            {
                "train": train_metrics.intensity_loss,
                "train_window": window_metrics.intensity_loss,
                "val": val_metrics.intensity_loss,
            },
            step_value,
        )
        add_scalars_if_finite("similarity/cosine", {"val": val_cosine}, step_value)
        add_scalar_if_finite("similarity/val_cosine", val_cosine, step_value)
        writer.add_scalar("optimizer/lr", lr, step_value)
        writer.flush()
        print(
            f"event={event} epoch={epoch_value} step={step_value} "
            f"train_loss={train_metrics.loss:.6f} "
            f"train_selection_loss={train_metrics.selection_loss:.6f} "
            f"train_intensity_loss={train_metrics.intensity_loss:.6f} "
            f"train_window_loss={window_metrics.loss:.6f} "
            f"train_window_selection_loss={window_metrics.selection_loss:.6f} "
            f"train_window_intensity_loss={window_metrics.intensity_loss:.6f} "
            f"val_loss={val_metrics.loss:.6f} "
            f"val_selection_loss={val_metrics.selection_loss:.6f} "
            f"val_intensity_loss={val_metrics.intensity_loss:.6f} "
            f"val_cosine={val_cosine:.6f} "
            f"lr={lr:.6g} samples={train_metrics.samples} "
            f"branch_id={ckpt_manager.current_branch_id}"
        )

    last_epoch_index = state.initial_epoch - 1
    last_train_metrics = nan_loss_metrics()

    for epoch_index in range(state.initial_epoch, max_epoch + 1):
        validation_epoch = epoch_index - 1
        should_validate_at_epoch_start = (
            epoch_index > state.initial_epoch or validate_at_start
        )
        if should_validate_at_epoch_start:
            val_metrics, val_cosine, val_peak_metrics = evaluate_current_validation(
                desc=f"ValStart({validation_epoch})",
                step_value=global_step,
            )
            log_training_metrics(
                event="epoch_start",
                epoch_value=validation_epoch,
                step_value=global_step,
                train_metrics=last_train_metrics,
                train_window_metrics=None,
                val_metrics=val_metrics,
                val_cosine=val_cosine,
                val_peak_metrics=val_peak_metrics,
            )

        if validation_epoch >= state.initial_epoch and should_validate_at_epoch_start:
            val_loss = val_metrics.loss
            step_scheduler(scheduler, val_loss)
            improved = val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_loss
                bad_epochs = 0
                ckpt_manager.update_topk(
                    score=val_loss,
                    epoch=validation_epoch,
                    iter=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_info=optimizer_config_for_torch(optimizer_info),
                    extra_data={
                        **extra_data,
                        "best_val_loss": float(best_val_loss),
                        "optimizer_info": dict(optimizer_info),
                    },
                    topk=topk,
                    comment="best_val_loss",
                )
            else:
                bad_epochs += 1

            if patience is not None and bad_epochs >= patience:
                print(f"Early stopping before epoch {epoch_index}.")
                break

        def on_validation_step(
            step_value: int,
            train_epoch_metrics: EpochLossMetrics,
            train_window_metrics: EpochLossMetrics,
        ) -> None:
            step_val_metrics, step_val_cosine, step_peak_metrics = evaluate_current_validation(
                desc=f"ValStep({step_value})",
                step_value=step_value,
            )
            log_training_metrics(
                event="step",
                epoch_value=epoch_index,
                step_value=step_value,
                train_metrics=train_epoch_metrics,
                train_window_metrics=train_window_metrics,
                val_metrics=step_val_metrics,
                val_cosine=step_val_cosine,
                val_peak_metrics=step_peak_metrics,
            )

        def on_train_log_step(
            step_value: int,
            train_epoch_metrics: EpochLossMetrics,
            train_window_metrics: EpochLossMetrics,
        ) -> None:
            # Validation at the same step already includes all train metrics.
            if (
                validation_interval_steps is not None
                and validation_interval_steps > 0
                and step_value % validation_interval_steps == 0
            ):
                return
            log_training_metrics(
                event="train_step",
                epoch_value=epoch_index,
                step_value=step_value,
                train_metrics=train_epoch_metrics,
                train_window_metrics=train_window_metrics,
                val_metrics=nan_loss_metrics(),
                val_cosine=float("nan"),
            )

        def on_step_end(
            step_value: int,
            train_epoch_metrics: EpochLossMetrics,
            train_window_metrics: EpochLossMetrics,
        ) -> None:
            if (
                save_interval_steps is None
                or save_interval_steps <= 0
                or step_value % save_interval_steps != 0
            ):
                return
            save_managed_checkpoint(
                ckpt_manager=ckpt_manager,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index,
                global_step=step_value,
                best_val_loss=best_val_loss,
                optimizer_info=optimizer_info,
                extra_data={
                    **extra_data,
                    "train_loss": train_epoch_metrics.loss,
                    "train_window_loss": train_window_metrics.loss,
                },
                comment="iter_interval",
            )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            desc=f"Train({epoch_index}/{max_epoch})",
            start_global_step=global_step,
            validation_interval_steps=validation_interval_steps,
            on_validation_step=on_validation_step,
            train_log_interval_steps=train_log_interval_steps,
            on_train_log_step=on_train_log_step,
            on_step_end=on_step_end,
        )
        global_step += train_metrics.steps
        last_epoch_index = epoch_index
        last_train_metrics = train_metrics

        should_save = save_interval > 0 and epoch_index % save_interval == 0
        if should_save:
            save_managed_checkpoint(
                ckpt_manager=ckpt_manager,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index,
                global_step=global_step,
                best_val_loss=best_val_loss,
                optimizer_info=optimizer_info,
                extra_data=extra_data,
                comment="interval",
            )

    if last_epoch_index >= state.initial_epoch:
        val_metrics, val_cosine, val_peak_metrics = evaluate_current_validation(
            desc=f"ValFinal({last_epoch_index})",
            step_value=global_step,
        )
        log_training_metrics(
            event="epoch_end",
            epoch_value=last_epoch_index,
            step_value=global_step,
            train_metrics=last_train_metrics,
            train_window_metrics=None,
            val_metrics=val_metrics,
            val_cosine=val_cosine,
            val_peak_metrics=val_peak_metrics,
        )
        val_loss = val_metrics.loss
        step_scheduler(scheduler, val_loss)
        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            ckpt_manager.update_topk(
                score=val_loss,
                epoch=last_epoch_index,
                iter=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_info=optimizer_config_for_torch(optimizer_info),
                extra_data={
                    **extra_data,
                    "best_val_loss": float(best_val_loss),
                    "optimizer_info": dict(optimizer_info),
                },
                topk=topk,
                comment="best_val_loss",
            )

    save_managed_checkpoint(
        ckpt_manager=ckpt_manager,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=last_epoch_index,
        global_step=global_step,
        best_val_loss=best_val_loss,
        optimizer_info=optimizer_info,
        extra_data=extra_data,
        comment="last",
    )
    writer.close()

def _candidate_config_paths(
    project_dir: str | Path,
    value: Optional[str | Path],
    *,
    default_names: Tuple[str, ...],
) -> List[Path]:
    project_path = Path(project_dir)
    config_dir = project_path / "config"

    if value is None or str(value) == "":
        return [config_dir / name for name in default_names]

    value_path = Path(value)
    candidates: List[Path] = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(value_path)
    if value_path.suffix == "":
        add_candidate(value_path.with_suffix(".json"))

    if not value_path.is_absolute():
        add_candidate(config_dir / value_path)
        if value_path.suffix == "":
            add_candidate(config_dir / f"{value_path}.json")

    return candidates


def resolve_config_path(
    project_dir: str | Path,
    value: Optional[str | Path],
    *,
    default_names: Tuple[str, ...],
    label: str,
) -> Path:
    candidates = _candidate_config_paths(
        project_dir,
        value,
        default_names=default_names,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {label} config. Tried:\n  - {candidate_text}"
    )


def resolve_train_config_path(
    project_dir: str | Path,
    train_config_path: Optional[str | Path] = None,
) -> Path:
    return resolve_config_path(
        project_dir,
        train_config_path,
        default_names=(DEFAULT_TRAIN_CONFIG_NAME,),
        label="train",
    )


def resolve_model_config_path(
    project_dir: str | Path,
    model_config_path: Optional[str | Path] = None,
) -> Path:
    if model_config_path is None or str(model_config_path) == "":
        return Path(project_dir) / "config" / DEFAULT_PROJECT_MODEL_CONFIG_NAME

    return resolve_config_path(
        project_dir,
        model_config_path,
        default_names=(),
        label="model",
    )


def run_training_from_config(
    *,
    project_dir: str | Path,
    model_config_path: Optional[str | Path] = None,
    train_config: Dict[str, Any],
) -> None:
    model_config_resolved = resolve_model_config_path(
        project_dir,
        model_config_path,
    )

    (
        experiment_dir,
        ckpt_id,
        batch_size,
        device,
        epochs,
        save_interval,
        save_interval_steps,
        optimizer_info,
        early_stopping_info,
        dataset_info,
        validation_interval_steps,
        run_dir,
    ) = prepare_train_from_config(
        project_dir,
        train_config,
        train_config_source=None,
    )

    _, _, train_loader, val_loader, extra_data = setup_dataset(
        dataset_info,
        batch_size,
        num_workers=0,
    )

    model_config = load_config(model_config_resolved)
    validate_preprocessing_compatibility(
        project_dir=project_dir, model_config=model_config
    )
    # Persist the effective configuration used by this run.
    save_config(model_config, run_dir / model_config_resolved.name)

    main(
        model_config=model_config,
        experiment_dir=experiment_dir,
        ckpt_id=ckpt_id,
        device=device,
        epoch=epochs,
        save_interval=save_interval,
        save_interval_steps=save_interval_steps,
        batch_size=batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_info=optimizer_info,
        early_stopping_info=early_stopping_info,
        run_dir=run_dir,
        extra_data=extra_data,
        validation_interval_steps=validation_interval_steps,
        validate_at_start=bool(dataset_info.get("validate_at_start", False)),
    )

def run_training(
    *,
    project_dir: str | Path,
    model_config_path: Optional[str | Path] = None,
    train_config_path: Optional[str | Path] = None,
    root_run_dir: Optional[str | Path] = None,
    num_workers: int = 0,
    model_overrides: Optional[Dict[str, Any]] = None,
    model_config_inline: Optional[Dict[str, Any]] = None,
    train_config_inline: Optional[Dict[str, Any]] = None,
    preprocessing_config_path: Optional[str | Path] = None,
) -> None:
    model_config_resolved = (
        None
        if model_config_inline is not None
        else resolve_model_config_path(project_dir, model_config_path)
    )
    if train_config_inline is None:
        train_config_resolved = resolve_train_config_path(
            project_dir,
            train_config_path,
        )
        prepared = prepare_train(
            project_dir,
            train_config_resolved,
            root_run_dir=root_run_dir,
        )
    else:
        prepared = prepare_train_from_config(
            project_dir,
            train_config_inline,
            train_config_source=None,
        )

    (
        experiment_dir,
        ckpt_id,
        batch_size,
        device,
        epochs,
        save_interval,
        save_interval_steps,
        optimizer_info,
        early_stopping_info,
        dataset_info,
        validation_interval_steps,
        run_dir,
    ) = prepared
    if root_run_dir is not None and train_config_inline is not None:
        inline_run_dir = Path(root_run_dir) / run_dir.name
        inline_run_dir.mkdir(parents=True, exist_ok=True)
        run_dir = inline_run_dir
        save_config(
            normalize_train_config(project_dir, train_config_inline),
            run_dir / DEFAULT_TRAIN_CONFIG_NAME,
        )

    _, _, train_loader, val_loader, extra_data = setup_dataset(
        dataset_info,
        batch_size,
        num_workers=num_workers,
    )

    model_config = (
        dict(model_config_inline)
        if model_config_inline is not None
        else load_config(model_config_resolved)
    )
    model_config.update(dict(model_overrides or {}))
    validate_preprocessing_compatibility(
        project_dir=project_dir,
        model_config=model_config,
        preprocessing_config_path=preprocessing_config_path,
    )
    # Persist the effective configuration, including command-line overrides.
    save_config(
        model_config,
        run_dir
        / (
            DEFAULT_PROJECT_MODEL_CONFIG_NAME
            if model_config_resolved is None
            else model_config_resolved.name
        ),
    )

    main(
        model_config=model_config,
        experiment_dir=experiment_dir,
        ckpt_id=ckpt_id,
        device=device,
        epoch=epochs,
        save_interval=save_interval,
        save_interval_steps=save_interval_steps,
        batch_size=batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_info=optimizer_info,
        early_stopping_info=early_stopping_info,
        run_dir=run_dir,
        extra_data=extra_data,
        validation_interval_steps=validation_interval_steps,
        validate_at_start=bool(dataset_info.get("validate_at_start", False)),
    )

def _csv_int_tuple(value: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FragmentTreeTrainingModel.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mol-encoder-checkpoint", required=True)
    parser.add_argument("--cleavage-edge-fnet-checkpoint", required=True)
    parser.add_argument("--condition-adduct-embedding-dim", type=int, default=16)
    parser.add_argument("--condition-ce-feature-dim", type=int, default=16)
    parser.add_argument("--condition-ce-fc-dims", type=_csv_int_tuple, default=(32,))
    parser.add_argument("--condition-feature-dim", type=int, default=64)
    parser.add_argument("--condition-fc-dims", type=_csv_int_tuple, default=(128, 64))
    parser.add_argument("--tree-hidden-dim", type=int, default=128)
    parser.add_argument("--tree-num-layers", type=int, default=2)
    parser.add_argument("--tree-num-heads", type=int, default=8)
    parser.add_argument("--tree-max-degree", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-edges-per-step", type=int, default=128)
    parser.add_argument("--max-retained-edges", type=int, default=30)
    parser.add_argument("--max-next-cleavage-candidates", type=int, default=3)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--ckpt-id", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=10)
    parser.add_argument("--validation-interval-steps", type=int, default=100)
    parser.add_argument(
        "--train-log-interval-steps",
        type=int,
        default=100,
        help="Log averaged training metrics every N successful steps. Default: 100.",
    )
    parser.add_argument(
        "--validate-at-start",
        action="store_true",
        help="Run ValStart before the first training epoch (disabled by default).",
    )
    parser.add_argument("--save-interval-epochs", type=int, default=1)
    parser.add_argument("--save-interval-steps", type=int, default=100)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    for option, value in (("--train-dir", args.train_dir), ("--val-dir", args.val_dir)):
        if not Path(value).is_dir():
            parser.error(f"{option} does not exist or is not a directory: {value}")
    return args


if __name__ == "__main__":
    args = parse_args()
    project_dir = args.output_dir
    train_split_dir = Path(args.train_dir)
    val_split_dir = Path(args.val_dir)
    train_data_dir = train_split_dir / "data" if (train_split_dir / "data").is_dir() else train_split_dir
    val_data_dir = val_split_dir / "data" if (val_split_dir / "data").is_dir() else val_split_dir
    preprocessing, preprocessing_config_path = load_and_validate_split_preprocessing(
        train_split_dir, val_split_dir
    )
    model_config_inline = build_model_config_from_pretrained(
        mol_encoder_checkpoint=args.mol_encoder_checkpoint,
        cleavage_edge_fnet_checkpoint=args.cleavage_edge_fnet_checkpoint,
        fragmenter_params=dict(preprocessing["fragmenter_params"]),
        condition_encoder_params={
            "adduct_embedding_dim": args.condition_adduct_embedding_dim,
            "ce_feature_dim": args.condition_ce_feature_dim,
            "ce_fc_dims": args.condition_ce_fc_dims,
            "feature_dim": args.condition_feature_dim,
            "fc_dims": args.condition_fc_dims,
        },
        tree_encoder_params={
            "hidden_dim": args.tree_hidden_dim,
            "num_layers": args.tree_num_layers,
            "num_heads": args.tree_num_heads,
            "max_degree": args.tree_max_degree,
        },
        dropout=args.dropout,
        generator_params={
            "max_edges_per_step": args.max_edges_per_step,
            "max_retained_edges": args.max_retained_edges,
            "max_next_cleavage_candidates": args.max_next_cleavage_candidates,
        },
    )
    train_config_inline = build_train_config(
        project_dir=project_dir,
        experiment_name=args.experiment_name,
        ckpt_id=args.ckpt_id,
        batch_size=args.batch_size,
        device=args.device,
        epoch=args.epochs,
        validation_interval_steps=(
            args.validation_interval_steps if args.validation_interval_steps > 0 else None
        ),
        train_log_interval_steps=(
            args.train_log_interval_steps if args.train_log_interval_steps > 0 else None
        ),
        save_interval=args.save_interval_epochs,
        save_interval_steps=(
            args.save_interval_steps if args.save_interval_steps > 0 else None
        ),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        training_structure_dir=train_data_dir,
        validation_structure_dir=val_data_dir,
        shuffle=args.shuffle,
        validate_at_start=args.validate_at_start,
    )
    train_config_inline["validation_valid_records_file"] = str(
        val_split_dir / "valid_records.msds"
    )
    if args.early_stopping_patience is not None:
        train_config_inline["early_stopping"] = {
            "patience": args.early_stopping_patience
        }
    run_training(
        project_dir=project_dir,
        num_workers=args.num_workers,
        model_config_inline=model_config_inline,
        train_config_inline=train_config_inline,
        preprocessing_config_path=preprocessing_config_path,
    )
