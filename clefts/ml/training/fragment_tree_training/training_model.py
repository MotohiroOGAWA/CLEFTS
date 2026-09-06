from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import math
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from ...common.progress import fixed_tqdm, iteration_edge_progress, set_edge_progress_phase
from . import metric_labels
from .performance_profile import run_training_performance_profile
from .workflow import run_shape_preflight

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
from ....domain.mass import parse_ce_to_ev
from ....libs.msentity.msentity import MSDataset
from ....libs.msentity.msentity.processing.spectrum_similarity import cosine_similarity_pair

PEAK_SELECTION_TOP_K = (5, 10, 20)
PEAK_SELECTION_MZ_TOLERANCE_DA = 0.01
PRECURSOR_MZ_COLUMN = "PrecursorMZ"

# Base peak-selection diagnostics, computed both on the full spectrum and
# (mirrored below) on the spectrum with the precursor-ion peak removed, since
# the precursor peak tends to dominate spectral similarity and can mask how
# well the fragment peaks themselves are predicted.
_PEAK_SELECTION_BASE_METRIC_NAMES = (
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
# Spectrum-level detection of the precursor ion as an observed peak: whether a
# peak near PrecursorMZ is present in the target and/or predicted spectrum.
PRECURSOR_DETECTION_METRIC_NAMES = (
    "precursor_detection_precision",
    "precursor_detection_recall",
    "precursor_detection_accuracy",
    "precursor_detection_f1",
    "precursor_detection_target_present_fraction",
    "precursor_detection_predicted_present_fraction",
)
PEAK_SELECTION_METRIC_NAMES = (
    *_PEAK_SELECTION_BASE_METRIC_NAMES,
    *PRECURSOR_DETECTION_METRIC_NAMES,
    "cosine@excl_precursor",
    *(f"{name}@excl_precursor" for name in _PEAK_SELECTION_BASE_METRIC_NAMES),
)
PEAK_SELECTION_QUANTILES = (
    ("min", 0.0),
    ("q1", 0.25),
    ("median", 0.5),
    ("q3", 0.75),
    ("max", 1.0),
)
# Per-batch training-time diagnostics reported by the model (see
# FragmentTreeTrainingModel._edge_retain_metrics / _selected_peak_metrics /
# _intensity_similarity_metrics), summarized in run_epoch() and surfaced here
# as train_*/val_* columns and TensorBoard cards. The "precursor/" entries
# isolate the same diagnostics for precursor-root candidates.
EDGE_METRIC_NAMES = (
    "edge_ranking_loss",
    "pairwise_ranking_accuracy",
    "edge_retain_precision",
    "edge_retain_recall",
    "edge_retain_accuracy",
    "edge_total_loss",
    "selected_peak_intensity_coverage",
    "peak_selection_accuracy",
    "peak_selection_precision",
    "peak_selection_recall",
    "intensity_cosine_similarity",
    "precursor/keep_loss",
    "precursor/peak_selection_accuracy",
    "precursor/peak_selection_precision",
    "precursor/peak_selection_recall",
    "precursor/selected_peak_intensity_coverage",
)


def _edge_metric_column(name: str) -> str:
    """Flatten a namespaced metric name (e.g. "precursor/keep_loss") into a
    TSV/CSV-safe column suffix."""
    return name.replace("/", "_")


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
    *(f"val_{name}" for name in PEAK_SELECTION_METRIC_NAMES),
    *(f"train_{_edge_metric_column(name)}" for name in EDGE_METRIC_NAMES),
    *(f"val_{_edge_metric_column(name)}" for name in EDGE_METRIC_NAMES),
    "lr",
)
DEFAULT_ASSIGNMENT_SCORE_THRESHOLD = 0.8


DEFAULT_TRAIN_CONFIG_NAME = "train_config.json"
DEFAULT_PROJECT_MODEL_CONFIG_NAME = "model_config.json"
DEFAULT_PREPROCESSING_CONFIG_NAME = "preprocessing_config.pftprep.json"
# Recognized as a fallback when locating a preprocessing config, so structure
# directories created before the dedicated ``.pftprep.json`` extension was
# introduced keep loading correctly.
LEGACY_PREPROCESSING_CONFIG_NAME = "preprocessing_config.json"
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
    absolute_ranker_summary: Dict[str, float] = field(default_factory=dict)


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


def combine_epoch_metrics(first: EpochLossMetrics, second: EpochLossMetrics) -> EpochLossMetrics:
    """Combine disjoint validation partitions without evaluating either twice."""
    total = first.samples + second.samples
    if total <= 0:
        return nan_loss_metrics()
    def weighted(name: str) -> float:
        values = [(getattr(item, name), item.samples) for item in (first, second) if item.samples]
        finite = [(value, count) for value, count in values if math.isfinite(value)]
        return sum(value * count for value, count in finite) / max(sum(count for _, count in finite), 1)
    summary: Dict[str, float] = {}
    for key in set(first.absolute_ranker_summary) | set(second.absolute_ranker_summary):
        # Means are exactly composable. Quantiles are reported per scope below,
        # rather than pretending that quantiles-of-quantiles are exact.
        if key.endswith("_mean"):
            pairs = [(item.absolute_ranker_summary.get(key), item.samples) for item in (first, second)]
            pairs = [(float(v), n) for v, n in pairs if v is not None and math.isfinite(float(v)) and n]
            if pairs:
                summary[key] = sum(v * n for v, n in pairs) / sum(n for _, n in pairs)
    return EpochLossMetrics(
        loss=weighted("loss"), selection_loss=weighted("selection_loss"),
        intensity_loss=weighted("intensity_loss"), samples=total,
        steps=first.steps + second.steps, absolute_ranker_summary=summary,
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
    fragmenter_params: Dict[str, Any],
    condition_encoder_params: Dict[str, Any],
    fragment_edge_encoder_params: Dict[str, Any],
    tree_encoder_params: Dict[str, Any],
    dropout: float,
    generator_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build spectrum-training config from a pretrained MolEncoder only."""
    mol_checkpoint = _checkpoint_dict(
        mol_encoder_checkpoint, label="MolEncoder"
    )
    mol_params = dict(mol_checkpoint.get("mol_encoder_params") or {})
    if not mol_params:
        raise KeyError(
            "MolEncoder checkpoint is missing mol_encoder_params: "
            f"{mol_encoder_checkpoint}"
        )
    fragmenter_params = dict(fragmenter_params)

    config: Dict[str, Any] = {
        "probability_model_params": {
            "mol_encoder_params": mol_params,
            "condition_encoder_params": dict(condition_encoder_params),
            "fragment_edge_encoder_params": dict(fragment_edge_encoder_params),
            "tree_encoder_params": dict(tree_encoder_params),
            "fragmenter_params": fragmenter_params,
            "dropout": float(dropout),
        },
        "mol_encoder_checkpoint": str(mol_encoder_checkpoint),
        "freeze_mol_encoder": True,
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
    if preprocessing_config_path is not None:
        config_path = Path(preprocessing_config_path)
    else:
        config_dir = Path(project_dir) / "config"
        config_path = config_dir / DEFAULT_PREPROCESSING_CONFIG_NAME
        if not config_path.exists():
            legacy_path = config_dir / LEGACY_PREPROCESSING_CONFIG_NAME
            if legacy_path.exists():
                config_path = legacy_path
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
    config["max_samples"] = int(config.get("max_samples", 100))
    if config["max_samples"] < 1:
        raise ValueError("max_samples must be positive.")

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
    train_log_interval_steps = config.get("train_log_interval_steps", 50)
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
    config["detect_anomaly"] = bool(config.get("detect_anomaly", False))
    config["profile_performance"] = bool(config.get("profile_performance", False))
    config["assignment_score_threshold"] = float(
        config.get("assignment_score_threshold", DEFAULT_ASSIGNMENT_SCORE_THRESHOLD)
    )
    if not 0.0 <= config["assignment_score_threshold"] <= 1.0:
        raise ValueError("assignment_score_threshold must be between 0 and 1.")

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
    train_log_interval_steps: Optional[int] = 50,
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
    detect_anomaly: bool = False,
    profile_performance: bool = False,
    max_samples: int = 100,
    assignment_score_threshold: float = DEFAULT_ASSIGNMENT_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
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
            "detect_anomaly": bool(detect_anomaly),
            "profile_performance": bool(profile_performance),
            "max_samples": int(max_samples),
            "assignment_score_threshold": float(assignment_score_threshold),
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
        "detect_anomaly": bool(train_config.get("detect_anomaly", False)),
        "profile_performance": bool(train_config.get("profile_performance", False)),
        "max_samples": int(train_config.get("max_samples", 100)),
        "assignment_score_threshold": float(train_config["assignment_score_threshold"]),
        "pattern": "*.preft.pt",
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
    pattern = str(dataset_info.get("pattern", "*.preft.pt"))
    threshold = float(dataset_info.get("assignment_score_threshold", DEFAULT_ASSIGNMENT_SCORE_THRESHOLD))
    train_scores = find_assignment_score_file(dataset_info["training_structure_dir"])
    val_scores = find_assignment_score_file(dataset_info["validation_structure_dir"])
    train_included, _, train_rows = load_assignment_score_selection(train_scores, threshold)
    val_included, _, val_rows = load_assignment_score_selection(val_scores, threshold)
    train_dataset = FragmentTreeStructureFileDataset(
        Path(dataset_info["training_structure_dir"]),
        pattern=pattern,
        included_samples_by_file=train_included,
    )
    val_dataset = FragmentTreeStructureFileDataset(
        Path(dataset_info["validation_structure_dir"]),
        pattern=pattern,
        included_samples_by_file=val_included,
    )
    if len(train_dataset) == 0:
        raise ValueError("training_structure_dir contains no training structure files.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fragment_tree_structure_items,
    }
    if num_workers > 0:
        # Keep deserialisation/collation workers alive across epochs and let
        # them prepare subsequent batches while the GPU handles this one.
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
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
        "max_samples": int(dataset_info.get("max_samples", 100)),
        "assignment_score_threshold": threshold,
        "training_assignment_score_file": str(train_scores),
        "validation_assignment_score_file": str(val_scores),
        "assignment_score_report": {
            "threshold": threshold,
            "training": assignment_selection_summary(train_rows, threshold),
            "validation": assignment_selection_summary(val_rows, threshold),
        },
    }
    return train_dataset, val_dataset, train_loader, val_loader, extra_data


def find_assignment_score_file(structure_dir: str | Path) -> Path:
    directory = Path(structure_dir)
    candidates = (directory / "assignment_scores.tsv", directory.parent / "assignment_scores.tsv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"assignment_scores.tsv was not found in {directory} or {directory.parent}"
    )


def load_assignment_score_selection(
    path: str | Path, threshold: float
) -> Tuple[Dict[str, set[int]], Dict[str, set[int]], List[float]]:
    included: Dict[str, set[int]] = {}
    excluded: Dict[str, set[int]] = {}
    scores: List[float] = []
    next_sample: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            filename = str(row.get("structure_file", "")).strip()
            if not filename:
                raise ValueError(f"Missing structure_file in {path}")
            try:
                score = float(row["assignment_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid assignment_score in {path}: {row}") from exc
            # Rows are emitted in structure/sample order; rejected samples are
            # absent, so their ordinal is exactly the saved sample index.
            sample_id = next_sample.get(filename, 0)
            next_sample[filename] = sample_id + 1
            destination = included if score >= threshold else excluded
            destination.setdefault(filename, set()).add(sample_id)
            scores.append(score)
    if not included:
        raise ValueError(f"No samples meet assignment_score_threshold={threshold:g} in {path}")
    return included, excluded, scores


def assignment_selection_summary(scores: Sequence[float], threshold: float) -> Dict[str, Any]:
    array = np.asarray(scores, dtype=np.float64)
    finite = array[np.isfinite(array)]
    distribution = summarize_distribution(finite)
    return {
        "total_samples": int(finite.size),
        "selected_samples": int((finite >= threshold).sum()),
        "excluded_samples": int((finite < threshold).sum()),
        "assignment_score": distribution,
    }


def split_validation_records_by_assignment_score(
    dataset: MSDataset, score_file: str | Path, threshold: float
) -> Tuple[MSDataset, Optional[MSDataset]]:
    """Split valid records using their stable source index (SpecID fallback)."""
    high_indexes: set[str] = set()
    low_indexes: set[str] = set()
    high_spec_ids: set[str] = set()
    low_spec_ids: set[str] = set()
    with open(score_file, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            high = float(row["assignment_score"]) >= threshold
            (high_indexes if high else low_indexes).add(str(row.get("index", "")))
            spec_id = str(row.get("SpecID", ""))
            if spec_id:
                (high_spec_ids if high else low_spec_ids).add(spec_id)
    metadata = dataset.metadata
    index_column = "__fragment_tree_original_index"
    high_rows: List[int] = []
    low_rows: List[int] = []
    for row_index, row in metadata.iterrows():
        source_index = str(row.get(index_column, row_index))
        spec_id = str(row.get("SpecID", ""))
        if source_index in high_indexes or (spec_id and spec_id in high_spec_ids):
            high_rows.append(int(row_index))
        elif source_index in low_indexes or (spec_id and spec_id in low_spec_ids):
            low_rows.append(int(row_index))
    if not high_rows:
        raise ValueError("No validation MSDataset records meet the assignment-score threshold.")
    return dataset[high_rows].copy(), (dataset[low_rows].copy() if low_rows else None)


def write_combined_validation_cosine_summary(output_dir: Path, global_step: int) -> Dict[str, float]:
    """Combine disjoint cached score rows; no spectrum is inferred twice."""
    values: List[float] = []
    for scope in ("filtered", "below_threshold"):
        path = output_dir / scope / "validation_cosine.tsv"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if str(row["global_step"]) == str(global_step):
                    values.append(float(row["cosine_similarity"]))
    summary = summarize_distribution(np.asarray(values, dtype=np.float64))
    path = output_dir / "unfiltered_cosine_summary.tsv"
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        tsv = csv.writer(handle, delimiter="\t")
        if write_header:
            tsv.writerow(["global_step", "mean", "q1", "median", "q3"])
        tsv.writerow([global_step, *[summary[name] for name in ("mean", "q1", "median", "q3")]])
    return summary


def write_combined_peak_selection_summary(output_dir: Path, global_step: int) -> None:
    """Summarize cached filtered + below-threshold per-spectrum metrics."""
    values: Dict[str, List[float]] = {}
    for scope in ("filtered", "below_threshold"):
        path = output_dir / scope / "validation_peak_selection.tsv"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if str(row["global_step"]) != str(global_step):
                    continue
                for name, raw in row.items():
                    if name not in {"global_step", "spectrum_index"}:
                        values.setdefault(name, []).append(float(raw))
    path = output_dir / "unfiltered_peak_selection_summary.tsv"
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        tsv = csv.writer(handle, delimiter="\t")
        if write_header:
            tsv.writerow(["global_step", "metric", "mean", "q1", "median", "q3"])
        for name, raw_values in values.items():
            summary = summarize_distribution(np.asarray(raw_values, dtype=np.float64))
            tsv.writerow([global_step, name, *[summary[key] for key in ("mean", "q1", "median", "q3")]])


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
    train_log_interval_steps: Optional[int] = 50,
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
    failed_batch_count = 0
    absolute_ranker_values: Dict[str, List[float]] = {}
    window_absolute_ranker_values: Dict[str, List[float]] = {}
    iterator = fixed_tqdm(loader, desc=desc, position=1, leave=False)

    def summarize_ranker(values_by_name: Dict[str, List[float]]) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, values in values_by_name.items():
            for statistic, value in summarize_distribution(np.asarray(values)).items():
                summary[f"{name}_{statistic}"] = value
        return summary

    for batch in iterator:
        try:
            # Shape queries against the pre-transfer CPU structure so the
            # progress bar opens (and shows the transfer itself) before any
            # device work happens.
            global_edges = int(batch["structure"].edge_index.size(1))
            with iteration_edge_progress(
                min(global_edges, int(max_training_edges)),
                desc=f"edges iter {step_count + 1}",
            ):
                set_edge_progress_phase("host→device transfer")
                structure = batch["structure"].to(device)
                # Target/formula pairs are selected by the edge encoder before
                # its expensive attention pass. ``max_edges_per_step`` is a
                # hard cap.
                num_samples = int(structure.num_samples)
                sample_weight = max(num_samples, 1)

                with torch.set_grad_enabled(is_train):
                    output = model(structure)
                    loss = output["loss"]
                    if not is_train:
                        output["absolute_ranker_metrics"].update(model.evaluate_depth_rollout(structure))

                    if is_train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()

            step_count += 1
            selection_loss = output.get("selection_loss")
            intensity_loss = output.get("intensity_loss")
            for name, value in output.get("absolute_ranker_metrics", {}).items():
                absolute_ranker_values.setdefault(name, []).append(float(value))
                window_absolute_ranker_values.setdefault(name, []).append(float(value))
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
            cumulative_metrics = replace(
                cumulative_metrics,
                absolute_ranker_summary=summarize_ranker(absolute_ranker_values),
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
            window_metrics = replace(
                window_metrics,
                absolute_ranker_summary=summarize_ranker(window_absolute_ranker_values),
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
            train_log_due = (
                is_train
                and train_log_interval_steps is not None
                and train_log_interval_steps > 0
                and on_train_log_step is not None
                and global_step % train_log_interval_steps == 0
            )
            if train_log_due:
                on_train_log_step(global_step, cumulative_metrics, window_metrics)

            # Persist and flush the cheap training metrics before starting a
            # potentially long validation pass at the same step. This keeps
            # TensorBoard live at exactly train_log_interval_steps.
            if validation_due:
                on_validation_step(global_step, cumulative_metrics, window_metrics)
                model.train(is_train)

            if validation_due or train_log_due:
                window_loss = 0.0
                window_selection_loss = 0.0
                window_intensity_loss = 0.0
                window_samples = 0
                window_selection_samples = 0
                window_intensity_samples = 0
                window_absolute_ranker_values.clear()

            if on_step_end is not None:
                on_step_end(global_step, cumulative_metrics, window_metrics)
        except Exception as exc:
            failed_batch_count += 1
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            print(
                "[WARN] Skipping failed batch "
                f"in {desc} at attempted_batch={step_count + failed_batch_count}: "
                f"{type(exc).__name__}: {exc}"
            )
            if failed_batch_count == 1:
                traceback.print_exc()
            continue

    metrics = make_loss_metrics(
        total_loss=total_loss,
        total_selection_loss=total_selection_loss,
        total_intensity_loss=total_intensity_loss,
        total_samples=total_samples,
        total_selection_samples=total_selection_samples,
        total_intensity_samples=total_intensity_samples,
        steps=step_count,
    )
    return replace(
        metrics,
        absolute_ranker_summary=summarize_ranker(absolute_ranker_values),
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


def calculate_precursor_detection_metrics(
    predicted_dataset: MSDataset,
    target_dataset: MSDataset,
    *,
    precursor_mz_column: str = PRECURSOR_MZ_COLUMN,
    mz_tolerance_da: float = PEAK_SELECTION_MZ_TOLERANCE_DA,
    spectrum_indexes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Spectrum-level precision/recall/accuracy for detecting the precursor
    ion as an observed peak in the predicted spectrum.

    The precursor peak tends to dominate spectral similarity, so this is
    reported separately from (and in addition to) the fragment-peak
    selection metrics. ``spectrum_indexes`` restricts the confusion matrix to
    a subset of spectra (e.g. spectra sharing an adduct type or a
    collision-energy bucket); by default every spectrum is used, reproducing
    the corpus-wide metric.
    """
    count = min(len(target_dataset), len(predicted_dataset))
    if count <= 0:
        return {name: float("nan") for name in PRECURSOR_DETECTION_METRIC_NAMES}

    precursor_mz = np.asarray(target_dataset[precursor_mz_column], dtype=np.float64)[:count]
    target_offsets = target_dataset.peaks.offsets
    predicted_offsets = predicted_dataset.peaks.offsets
    target_data = target_dataset.peaks.data
    predicted_data = predicted_dataset.peaks.data

    target_present = np.zeros(count, dtype=bool)
    predicted_present = np.zeros(count, dtype=bool)
    for spectrum_index in range(count):
        pmz = precursor_mz[spectrum_index]
        if not np.isfinite(pmz):
            continue
        t_start, t_end = int(target_offsets[spectrum_index]), int(target_offsets[spectrum_index + 1])
        if t_end > t_start:
            target_present[spectrum_index] = bool(
                np.any(np.abs(target_data[t_start:t_end, 0] - pmz) <= mz_tolerance_da)
            )
        p_start, p_end = int(predicted_offsets[spectrum_index]), int(predicted_offsets[spectrum_index + 1])
        if p_end > p_start:
            predicted_present[spectrum_index] = bool(
                np.any(np.abs(predicted_data[p_start:p_end, 0] - pmz) <= mz_tolerance_da)
            )

    if spectrum_indexes is not None:
        subset = np.asarray(spectrum_indexes, dtype=np.int64)
        target_present = target_present[subset]
        predicted_present = predicted_present[subset]
        count = int(subset.size)
    if count <= 0:
        return {name: float("nan") for name in PRECURSOR_DETECTION_METRIC_NAMES}

    true_positive = int(np.sum(target_present & predicted_present))
    false_positive = int(np.sum(~target_present & predicted_present))
    false_negative = int(np.sum(target_present & ~predicted_present))
    true_negative = int(np.sum(~target_present & ~predicted_present))

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else float("nan")
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else float("nan")
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    accuracy = (true_positive + true_negative) / count

    return {
        "precursor_detection_precision": precision,
        "precursor_detection_recall": recall,
        "precursor_detection_accuracy": accuracy,
        "precursor_detection_f1": f1,
        "precursor_detection_target_present_fraction": float(target_present.mean()),
        "precursor_detection_predicted_present_fraction": float(predicted_present.mean()),
    }


def exclude_precursor_peaks(
    dataset: MSDataset,
    *,
    precursor_mz_column: str = PRECURSOR_MZ_COLUMN,
    mz_tolerance_da: float = PEAK_SELECTION_MZ_TOLERANCE_DA,
) -> MSDataset:
    """Return a copy of ``dataset`` with each spectrum's precursor-ion peak(s)
    (peaks within ``mz_tolerance_da`` of that spectrum's PrecursorMZ) removed.

    Used to evaluate similarity/selection metrics on fragment peaks alone,
    since a correctly placed precursor peak can inflate whole-spectrum
    similarity independently of fragment-peak prediction quality.
    """
    filtered = dataset.copy()
    precursor_mz = np.asarray(dataset[precursor_mz_column], dtype=np.float64)
    offsets = dataset.peaks.offsets
    data = dataset.peaks.data
    metadata = dataset.peaks.metadata

    keep_mask = np.ones(len(data), dtype=bool)
    for spectrum_index in range(len(dataset)):
        start, end = int(offsets[spectrum_index]), int(offsets[spectrum_index + 1])
        if start == end:
            continue
        pmz = precursor_mz[spectrum_index]
        if not np.isfinite(pmz):
            continue
        keep_mask[start:end] = np.abs(data[start:end, 0] - pmz) > mz_tolerance_da

    lengths = offsets[1:] - offsets[:-1]
    new_lengths = np.array(
        [
            int(keep_mask[int(offsets[i]):int(offsets[i]) + int(lengths[i])].sum())
            for i in range(len(dataset))
        ],
        dtype=np.int64,
    )
    new_offsets = np.zeros(len(dataset) + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(new_lengths)
    new_data = data[keep_mask]
    new_metadata = metadata.iloc[keep_mask].reset_index(drop=True) if metadata is not None else None
    filtered.peaks.replace_data(new_data, new_offsets, metadata=new_metadata)
    return filtered


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    """Summarize a distribution as mean/min/q1/median/q3/max.

    ``min``/``max`` are boxplot whiskers (the most extreme values still inside
    ``[q1 - 1.5*iqr, q3 + 1.5*iqr]``), not the true extremes, so a handful of
    outlier samples cannot dominate the reported range. ``q1``/``median``/``q3``
    remain the plain quantiles.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            **{name: float("nan") for name, _ in PEAK_SELECTION_QUANTILES},
        }
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    inliers = finite[(finite >= q1 - 1.5 * iqr) & (finite <= q3 + 1.5 * iqr)]
    min_value = float(inliers.min()) if inliers.size else float(q1)
    max_value = float(inliers.max()) if inliers.size else float(q3)
    return {
        "mean": float(finite.mean()),
        "min": min_value,
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": max_value,
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
    validation_scope: str = "validation",
) -> float:
    if model.intensity_predictor is None:
        raise ValueError("Spectrum validation requires an intensity predictor.")

    device = next(model.parameters()).device
    predicted_dataset, target_dataset = predict_validation_msdataset(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
    )
    if predicted_dataset is None or target_dataset is None:
        raise RuntimeError("Validation could not generate any spectra; inspect the prediction warnings above.")

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
    cosine_summary = summarize_distribution(np.asarray(scores, dtype=np.float64))
    selection_metrics = calculate_peak_selection_metrics(
        predicted_dataset, target_dataset
    )

    # Fragment-only diagnostics: recompute cosine similarity and peak
    # selection metrics after stripping the precursor-ion peak from both the
    # target and predicted spectra, so a correctly placed precursor peak
    # cannot mask poor fragment-peak prediction.
    excl_target_dataset = exclude_precursor_peaks(target_dataset)
    excl_predicted_dataset = exclude_precursor_peaks(predicted_dataset)
    excl_scores = cosine_similarity_pair(
        excl_target_dataset,
        np.arange(count, dtype=np.int64),
        excl_predicted_dataset,
        np.arange(count, dtype=np.int64),
        show_progress=False,
    ).astype(np.float64)
    # A spectrum with no non-precursor peaks in either the target or the
    # prediction carries no fragment information to compare; leave it out of
    # the aggregate rather than scoring it as similarity 0.
    excl_both_empty = (excl_target_dataset.peaks.lengths == 0) & (
        excl_predicted_dataset.peaks.lengths == 0
    )
    excl_scores[excl_both_empty] = float("nan")
    excl_selection_metrics = calculate_peak_selection_metrics(
        excl_predicted_dataset, excl_target_dataset
    )
    selection_metrics["cosine@excl_precursor"] = excl_scores
    selection_metrics.update(
        {f"{name}@excl_precursor": values for name, values in excl_selection_metrics.items()}
    )
    selection_summaries = {
        name: summarize_distribution(values)
        for name, values in selection_metrics.items()
    }

    # Precursor-ion peak detection (precision/recall/accuracy) is reported
    # separately: the precursor peak dominates whole-spectrum similarity, so
    # this isolates how reliably it is placed from how well fragments match.
    # These are corpus-level aggregates rather than per-spectrum values, so
    # they are added to the summary only (not the per-spectrum detail file).
    def _scalar_summary(value: float) -> Dict[str, float]:
        value = float(value)
        return {"mean": value, **{name: value for name, _ in PEAK_SELECTION_QUANTILES}}

    precursor_detection = calculate_precursor_detection_metrics(
        predicted_dataset, target_dataset
    )
    for name, value in precursor_detection.items():
        selection_summaries[name] = _scalar_summary(value)

    # Adduct-type and collision-energy breakdown: every per-spectrum metric
    # (cosine, peak-selection) gets a genuine within-group distribution;
    # precursor detection has no per-spectrum value, so instead each group
    # gets its own corpus-style precision/recall/f1/accuracy, and the spread
    # *across* groups is reported as a distribution.
    try:
        adduct_values = np.asarray(target_dataset["AdductType"])[:count]
        # Match the parser used by validation structure construction (which
        # supplies no instrument). Metadata can contain units such as "20 V"
        # or normalized energies such as "30%" that require precursor m/z.
        ce_values = np.asarray([
            parse_ce_to_ev(
                row["CollisionEnergy"],
                precursor_mz=row.get(PRECURSOR_MZ_COLUMN, float("nan")),
            )
            for _, row in target_dataset.metadata.iloc[:count].iterrows()
        ], dtype=np.float64)
    except KeyError:
        adduct_values = None
        ce_values = None
    if adduct_values is not None and ce_values is not None:
        adduct_labels = np.asarray(
            [metric_labels.tensorboard_label(str(value)) for value in adduct_values]
        )
        finite_ce = ce_values[np.isfinite(ce_values)]
        if finite_ce.size:
            ce_q1, ce_median, ce_q3 = np.quantile(finite_ce, [0.25, 0.5, 0.75])
        else:
            ce_q1 = ce_median = ce_q3 = 0.0
        ce_labels = np.asarray([
            metric_labels.ce_range_label(value, q1=ce_q1, median=ce_median, q3=ce_q3)
            for value in ce_values
        ])
        groupings = {
            "by_adduct": {
                label: np.flatnonzero(adduct_labels == label)
                for label in np.unique(adduct_labels)
            },
            "by_ce_range": {
                label: np.flatnonzero(ce_labels == label)
                for label in np.unique(ce_labels)
            },
        }
        for group_kind, groups in groupings.items():
            group_precursor_values: Dict[str, List[float]] = {
                name: [] for name in PRECURSOR_DETECTION_METRIC_NAMES
            }
            for label, indexes in groups.items():
                if indexes.size == 0:
                    continue
                selection_summaries[f"cosine@{group_kind}:{label}"] = summarize_distribution(
                    scores[indexes]
                )
                for metric_name, values in selection_metrics.items():
                    selection_summaries[
                        f"{metric_name}@{group_kind}:{label}"
                    ] = summarize_distribution(values[indexes])
                group_precursor = calculate_precursor_detection_metrics(
                    predicted_dataset, target_dataset, spectrum_indexes=indexes,
                )
                for name, value in group_precursor.items():
                    selection_summaries[
                        f"{name}@{group_kind}:{label}"
                    ] = _scalar_summary(value)
                    group_precursor_values[name].append(value)
            for name, collected in group_precursor_values.items():
                if not collected:
                    continue
                selection_summaries[
                    f"{name}@{group_kind}_distribution"
                ] = summarize_distribution(np.asarray(collected))

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
        cosine_summary_file = output_dir / "validation_cosine_summary.tsv"
        write_header = not cosine_summary_file.exists()
        with open(cosine_summary_file, "a", encoding="utf-8", newline="") as f:
            tsv = csv.writer(f, delimiter="\t")
            if write_header:
                tsv.writerow(["global_step", "mean", "q1", "median", "q3"])
            tsv.writerow([global_step, cosine_summary["mean"], cosine_summary["q1"],
                          cosine_summary["median"], cosine_summary["q3"]])
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
        distribution_stats = ("min", "q1", "mean", "median", "q3", "max")
        log_distribution_cards(
            writer,
            "peak_selection",
            {
                validation_scope: {
                    f"{metric_name}_{stat}": summary[stat]
                    for metric_name, summary in selection_summaries.items()
                    for stat in distribution_stats
                }
            },
            int(global_step),
        )
        writer.flush()
    if writer is not None or output_dir is not None:
        log_validation_spectrum_quantiles(
            writer=writer, predicted_dataset=predicted_dataset,
            target_dataset=target_dataset, scores=scores,
            global_step=int(global_step or 0), scope=validation_scope,
            output_dir=output_dir,
        )
        if writer is not None:
            writer.flush()
    with fixed_tqdm(
        total=1, desc="ValCosine", position=1, leave=False
    ) as iterator:
        iterator.set_postfix(val_cosine=val_cosine)
        iterator.update(1)
    return val_cosine


def find_split_config(
    split_dir: str | Path, filename: str | Sequence[str]
) -> Path:
    """Find a config saved in a structure split, accepting split/data paths.

    ``filename`` may be a single name or a sequence of names to try in
    priority order (e.g. a current dedicated extension followed by a legacy
    plain name), returning the first match found. Directory proximity to
    ``split_dir`` still takes priority over name order.
    """
    split_path = Path(split_dir).resolve()
    filenames = (filename,) if isinstance(filename, str) else tuple(filename)
    candidates = []
    for directory in (split_path, split_path.parent, *split_path.parents):
        for name in filenames:
            candidates.extend((directory / name, directory / "config" / name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find any of {list(filenames)} in the structure split: {split_dir}"
    )


def load_and_validate_split_preprocessing(
    train_dir: str | Path,
    val_dir: str | Path,
) -> Tuple[Dict[str, Any], Path]:
    """Load immutable preprocessing data and require identical train/val settings."""
    preprocessing_config_names = (
        DEFAULT_PREPROCESSING_CONFIG_NAME, LEGACY_PREPROCESSING_CONFIG_NAME
    )
    train_preprocessing_path = find_split_config(
        train_dir, preprocessing_config_names
    )
    val_preprocessing_path = find_split_config(
        val_dir, preprocessing_config_names
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
            f"fragmenter.json does not match {DEFAULT_PREPROCESSING_CONFIG_NAME} "
            "in the training structure directory."
        )
    return train_preprocessing, train_preprocessing_path


def log_validation_spectrum_quantiles(
    *, writer, predicted_dataset: MSDataset, target_dataset: MSDataset,
    scores: np.ndarray, global_step: int, scope: str = "validation",
    output_dir: Optional[Path] = None,
) -> None:
    """Show five representative mirror plots from high to low cosine."""
    if scores.size == 0:
        return
    import matplotlib.pyplot as plt

    ranked = np.argsort(scores)[::-1]
    positions = np.linspace(0, len(ranked) - 1, num=min(5, len(ranked))).round().astype(int)
    for level, position in enumerate(positions, start=1):
        spectrum_index = int(ranked[int(position)])
        target_start = int(target_dataset.peaks.offsets[spectrum_index])
        target_end = int(target_dataset.peaks.offsets[spectrum_index + 1])
        pred_start = int(predicted_dataset.peaks.offsets[spectrum_index])
        pred_end = int(predicted_dataset.peaks.offsets[spectrum_index + 1])
        target_peaks = target_dataset.peaks.data[target_start:target_end].copy()
        predicted_peaks = predicted_dataset.peaks.data[pred_start:pred_end].copy()
        for peaks in (target_peaks, predicted_peaks):
            if len(peaks):
                peaks[:, 1] = np.nan_to_num(peaks[:, 1], nan=0.0, posinf=0.0, neginf=0.0).clip(min=0)
                maximum = peaks[:, 1].max()
                if maximum > 0:
                    peaks[:, 1] /= maximum
        figure, axis = plt.subplots(figsize=(10, 4))
        if len(target_peaks):
            axis.vlines(target_peaks[:, 0], 0, target_peaks[:, 1], color="black", label="measured")
        if len(predicted_peaks):
            axis.vlines(predicted_peaks[:, 0], 0, -predicted_peaks[:, 1], color="tab:red", label="generated")
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(xlabel="m/z", ylabel="relative intensity", ylim=(-1.1, 1.1),
                 title=f"spectrum {spectrum_index} · cosine={float(scores[spectrum_index]):.4f}")
        axis.legend(loc="upper right")
        figure.tight_layout()
        try:
            if output_dir is not None:
                figure_dir = output_dir / "spectra" / f"step_{global_step:08d}"
                figure_dir.mkdir(parents=True, exist_ok=True)
                figure.savefig(figure_dir / f"level_{level}.png")
            if writer is not None:
                writer.add_figure(
                    f"validation_spectra/{scope}/level_{level}_high_to_low",
                    figure, global_step=global_step, close=False,
                )
        finally:
            plt.close(figure)



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

    for smiles, record_indexes in fixed_tqdm(
        groups.items(),
        desc="Building validation structures from MSDataset",
        position=1,
        leave=False,
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


def add_scalar_if_finite(writer, tag: str, value: float, step: int) -> None:
    if math.isfinite(float(value)):
        writer.add_scalar(tag, float(value), step)


def add_scalars_if_finite(
    writer,
    main_tag: str,
    values: Dict[str, float],
    step: int,
) -> None:
    finite_values = {
        key: float(value)
        for key, value in values.items()
        if math.isfinite(float(value))
    }
    if finite_values:
        writer.add_scalars(main_tag, finite_values, step)


def log_distribution_cards(
    writer,
    namespace: str,
    summaries: Dict[str, Dict[str, float]],
    step: int,
) -> None:
    """Compare means by stage; give each statistic its own detail card."""
    statistics = ("min", "q1", "mean", "median", "q3", "max")
    grouped: Dict[str, Dict[str, float]] = {}
    for split, summary in summaries.items():
        for name, value in summary.items():
            for statistic in statistics:
                suffix = f"_{statistic}"
                if name.endswith(suffix):
                    metric_path = name[: -len(suffix)]
                    metric, _, scope = metric_path.partition("@")
                    card, condition = metric_labels.tensorboard_metric_card(namespace, metric, scope)
                    series = f"{split}_{condition}" if condition else split
                    if statistic == "mean":
                        grouped.setdefault(card, {})[f"{series}_mean"] = value
                    group, _, metric_card = card.partition("/")
                    detail_card = f"{group}_statistics/{metric_card}/{statistic}"
                    grouped.setdefault(detail_card, {})[series] = value
                    break
    for metric, values in grouped.items():
        add_scalars_if_finite(writer, metric, values, step)


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
    detect_anomaly: bool = False,
    profile_performance: bool = False,
) -> None:
    torch.autograd.set_detect_anomaly(detect_anomaly)
    if detect_anomaly:
        print("[INFO] torch autograd anomaly detection is enabled.")

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

    if profile_performance:
        print("[INFO] Profiling the largest estimated training batch.")
        performance_report = run_training_performance_profile(
            model=model,
            loader=train_loader,
            device=device,
            output_dir=run_dir / "performance_profile",
        )
        extra_data["performance_profile"] = performance_report
        print(
            "[INFO] Performance profile written to "
            f"{run_dir / 'performance_profile'}"
        )

    patience = early_stopping_info.get("patience")
    patience = None if patience in {None, ""} else int(patience)
    min_delta = float(early_stopping_info.get("min_delta", 0.0))
    topk = int(early_stopping_info.get("topk", 1))
    bad_epochs = 0
    grad_clip_norm = optimizer_info.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm in {None, ""} else float(grad_clip_norm)

    max_epoch = state.initial_epoch + int(epoch) - 1
    writer = ckpt_manager.summary_writer
    for name, value in dict(extra_data.get("preflight") or {}).items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"preflight/{name}", float(value), 0)
    for name, value in dict(extra_data.get("performance_profile") or {}).items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"performance_profile/{name}", float(value), 0)
    writer.flush()
    # Create a useful dashboard immediately. Without this, a new run contains
    # only the event-file header until its first scheduled validation.
    writer.add_scalar("optimizer/lr", float(optimizer.param_groups[0]["lr"]), global_step)
    writer.flush()
    validation_valid_records_file = Path(extra_data["validation_valid_records_file"])
    train_log_interval_steps = extra_data.get("train_log_interval_steps", 50)
    validation_dataset = (
        MSDataset.load(str(validation_valid_records_file))
        if validation_valid_records_file.exists()
        else None
    )
    if validation_dataset is None:
        raise FileNotFoundError(
            f"Spectrum validation requires the valid-record MSDataset: "
            f"{validation_valid_records_file}"
        )
    validation_excluded_dataset = None
    if validation_dataset is not None:
        validation_dataset, validation_excluded_dataset = split_validation_records_by_assignment_score(
            validation_dataset,
            extra_data["validation_assignment_score_file"],
            float(extra_data["assignment_score_threshold"]),
        )
    _, validation_excluded, _ = load_assignment_score_selection(
        extra_data["validation_assignment_score_file"],
        float(extra_data["assignment_score_threshold"]),
    )
    val_excluded_loader = None
    if validation_excluded:
        excluded_dataset = FragmentTreeStructureFileDataset(
            Path(extra_data["validation_structure_dir"]),
            pattern=str(extra_data.get("pattern", "*.preft.pt")),
            included_samples_by_file=validation_excluded,
        )
        val_excluded_loader = DataLoader(
            excluded_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fragment_tree_structure_items,
        )

    def evaluate_current_validation(
        desc: str,
        *,
        step_value: int,
    ) -> Tuple[EpochLossMetrics, float, Dict[str, float]]:
        selection_metric_means: Dict[str, float] = {}
        with torch.no_grad():
            metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                desc=desc,
            ) if val_loader is not None and len(val_loader) > 0 else nan_loss_metrics()
            excluded_metrics = (
                run_epoch(
                    model=model, loader=val_excluded_loader, device=device,
                    optimizer=None, desc=f"{desc}-below-threshold",
                )
                if val_excluded_loader is not None else nan_loss_metrics()
            )
            unfiltered_metrics = (
                combine_epoch_metrics(metrics, excluded_metrics)
                if excluded_metrics.samples else metrics
            )
            add_scalars_if_finite(
                writer,
                "loss/validation_scope",
                {"filtered": metrics.loss, "unfiltered": unfiltered_metrics.loss},
                step_value,
            )
            log_distribution_cards(
                writer,
                "validation_scope/absolute_ranker",
                {"filtered": metrics.absolute_ranker_summary,
                 "below_threshold": excluded_metrics.absolute_ranker_summary},
                step_value,
            )
            scope_file = run_dir / "validation" / "validation_scope_summary.tsv"
            scope_file.parent.mkdir(parents=True, exist_ok=True)
            write_header = not scope_file.exists()
            with open(scope_file, "a", encoding="utf-8", newline="") as handle:
                tsv = csv.writer(handle, delimiter="\t")
                if write_header:
                    tsv.writerow(["global_step", "scope", "samples", "loss", "selection_loss", "intensity_loss"])
                for scope, item in (("filtered", metrics), ("below_threshold", excluded_metrics), ("unfiltered", unfiltered_metrics)):
                    tsv.writerow([step_value, scope, item.samples, item.loss, item.selection_loss, item.intensity_loss])
            cosine = (
                evaluate_validation_cosine(
                    model=model,
                    dataset=validation_dataset,
                    batch_size=batch_size,
                    writer=writer,
                    global_step=step_value,
                    output_dir=run_dir / "validation" / "filtered",
                    selection_metric_means=selection_metric_means,
                )
                if validation_dataset is not None and len(validation_dataset) > 0
                else float("nan")
            )
            excluded_peak_means: Dict[str, float] = {}
            excluded_cosine = (
                evaluate_validation_cosine(
                    model=model, dataset=validation_excluded_dataset,
                    batch_size=batch_size, writer=writer, global_step=step_value,
                    validation_scope="validation_below_threshold",
                    output_dir=run_dir / "validation" / "below_threshold",
                    selection_metric_means=excluded_peak_means,
                )
                if validation_excluded_dataset is not None and len(validation_excluded_dataset) > 0
                else float("nan")
            )
            if math.isfinite(cosine) and math.isfinite(excluded_cosine):
                high_count, low_count = len(validation_dataset), len(validation_excluded_dataset)
                unfiltered_cosine = (cosine * high_count + excluded_cosine * low_count) / (high_count + low_count)
            else:
                unfiltered_cosine = cosine if math.isfinite(cosine) else excluded_cosine
            add_scalars_if_finite(
                writer,
                "intensity/validation_scope_cosine",
                {"filtered": cosine, "below_threshold": excluded_cosine,
                 "unfiltered": unfiltered_cosine}, step_value,
            )
            if math.isfinite(unfiltered_cosine):
                combined_cosine = write_combined_validation_cosine_summary(
                    run_dir / "validation", step_value
                )
                log_distribution_cards(
                    writer,
                    "peak_selection",
                    {"validation": {f"cosine@unfiltered_{name}": value
                                    for name, value in combined_cosine.items()}},
                    step_value,
                )
                write_combined_peak_selection_summary(
                    run_dir / "validation", step_value
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
        edge_metric_names = EDGE_METRIC_NAMES
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
                f"train_{_edge_metric_column(name)}": train_metrics.absolute_ranker_summary.get(
                    f"{name}_mean", float("nan")
                )
                for name in edge_metric_names
            },
            **{
                f"val_{_edge_metric_column(name)}": val_metrics.absolute_ranker_summary.get(
                    f"{name}_mean", float("nan")
                )
                for name in edge_metric_names
            },
            **{
                f"val_{name}": (val_peak_metrics or {}).get(name, float("nan"))
                for name in PEAK_SELECTION_METRIC_NAMES
            },
            "lr": lr,
        }
        ckpt_manager.log_metrics(**metric_row)
        ckpt_manager.flush_metrics(flush_dir=str(run_dir))
        add_scalars_if_finite(
            writer,
            "loss/total",
            {"train": train_metrics.loss, "train_window": window_metrics.loss,
             "validation": val_metrics.loss}, step_value,
        )
        add_scalars_if_finite(
            writer,
            "loss/selection",
            {"train": train_metrics.selection_loss,
             "train_window": window_metrics.selection_loss,
             "validation": val_metrics.selection_loss}, step_value,
        )
        add_scalars_if_finite(
            writer,
            "loss/intensity",
            {"train": train_metrics.intensity_loss,
             "train_window": window_metrics.intensity_loss,
             "validation": val_metrics.intensity_loss}, step_value,
        )
        add_scalar_if_finite(writer, "intensity/generated_spectrum_cosine", val_cosine, step_value)
        log_distribution_cards(
            writer,
            "metrics",
            {"train": train_metrics.absolute_ranker_summary,
             "train_window": window_metrics.absolute_ranker_summary,
             "validation": val_metrics.absolute_ranker_summary},
            step_value,
        )
        distribution_file = run_dir / "metric_distributions.tsv"
        write_header = not distribution_file.exists()
        with distribution_file.open("a", encoding="utf-8", newline="") as handle:
            tsv = csv.writer(handle, delimiter="\t")
            if write_header:
                tsv.writerow(["event", "epoch", "global_step", "split", "metric", "value"])
            for split, summary in (("train", train_metrics.absolute_ranker_summary),
                                   ("train_window", window_metrics.absolute_ranker_summary),
                                   ("validation", val_metrics.absolute_ranker_summary)):
                for name, value in sorted(summary.items()):
                    tsv.writerow([event, epoch_value, step_value, split, name, value])
        writer.add_scalar("training/phase", int(model.training_phase.item()), step_value)
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

    epoch_iterator = fixed_tqdm(
        range(state.initial_epoch, max_epoch + 1),
        total=max_epoch - state.initial_epoch + 1,
        desc="epochs",
        position=0,
        leave=True,
    )
    for epoch_index in epoch_iterator:
        epoch_iterator.set_description_str(f"epoch {epoch_index}/{max_epoch}")
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
            phase_summary = {
                f"{metric}_{quantile}": min(
                    float(last_train_metrics.absolute_ranker_summary.get(
                        f"{metric}_{quantile}", float("-inf")
                    )),
                    float(val_metrics.absolute_ranker_summary.get(
                        f"{metric}_{quantile}", float("-inf")
                    )),
                )
                for metric in (
                    "target_edge_recall", "target_group_recall", "target_node_recall"
                )
                for quantile in ("min", "q1", "median")
            }
            if model.update_training_phase(phase_summary):
                print(
                    "AbsoluteRanker train/validation thresholds were met; enabling "
                    "conditioned edge and spectrum training."
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
            phase_summary = {
                key: min(
                    float(train_epoch_metrics.absolute_ranker_summary.get(key, float("-inf"))),
                    float(step_val_metrics.absolute_ranker_summary.get(key, float("-inf"))),
                )
                for key in (
                    f"{metric}_{quantile}"
                    for metric in (
                        "target_edge_recall",
                        "target_group_recall",
                        "target_node_recall",
                    )
                    for quantile in ("min", "q1", "median")
                )
            }
            if model.update_training_phase(phase_summary):
                print(
                    "AbsoluteRanker validation thresholds were met; enabling "
                    "conditioned edge and spectrum training."
                )

        def on_train_log_step(
            step_value: int,
            train_epoch_metrics: EpochLossMetrics,
            train_window_metrics: EpochLossMetrics,
        ) -> None:
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

        # Always emit the complete epoch aggregate, even when the epoch ends
        # before the next periodic 50-step training log.
        log_training_metrics(
            event="train_epoch_end",
            epoch_value=epoch_index,
            step_value=global_step,
            train_metrics=train_metrics,
            train_window_metrics=None,
            val_metrics=nan_loss_metrics(),
            val_cosine=float("nan"),
        )

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
        phase_summary = {
                key: min(
                    float(last_train_metrics.absolute_ranker_summary.get(key, float("-inf"))),
                    float(val_metrics.absolute_ranker_summary.get(key, float("-inf"))),
                )
                for key in (
                    f"{metric}_{quantile}"
                    for metric in (
                        "target_edge_recall", "target_group_recall", "target_node_recall"
                    )
                    for quantile in ("min", "q1", "median")
                )
        }
        if model.update_training_phase(phase_summary):
            print(
                "AbsoluteRanker train/validation thresholds were met; enabling "
                "conditioned edge and spectrum training."
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
    save_config(extra_data["assignment_score_report"], run_dir / "assignment_score_report.json")

    model_config = load_config(model_config_resolved)
    validate_preprocessing_compatibility(
        project_dir=project_dir, model_config=model_config
    )
    # Persist the effective configuration used by this run.
    save_config(model_config, run_dir / model_config_resolved.name)

    edge_params = model_config["probability_model_params"]["fragment_edge_encoder_params"]
    extra_data["preflight"] = run_shape_preflight(
        max_samples=int(dataset_info["max_samples"]),
        max_edges_per_step=int(edge_params.get("max_edges_per_step", 128)),
        feature_dim=int(edge_params.get("feature_dim", 256)),
        device=device,
        output_file=run_dir / "preflight.json",
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
        detect_anomaly=bool(dataset_info.get("detect_anomaly", False)),
        profile_performance=bool(dataset_info.get("profile_performance", False)),
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
    workbench_config: Optional[Dict[str, Any]] = None,
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
    if workbench_config is not None:
        save_config(workbench_config, run_dir / "fragment_tree.pfttrain.json")

    _, _, train_loader, val_loader, extra_data = setup_dataset(
        dataset_info,
        batch_size,
        num_workers=num_workers,
    )
    save_config(extra_data["assignment_score_report"], run_dir / "assignment_score_report.json")

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

    edge_params = model_config["probability_model_params"]["fragment_edge_encoder_params"]
    extra_data["preflight"] = run_shape_preflight(
        max_samples=int(dataset_info["max_samples"]),
        max_edges_per_step=int(edge_params.get("max_edges_per_step", 128)),
        feature_dim=int(edge_params.get("feature_dim", 256)),
        device=device,
        output_file=run_dir / "preflight.json",
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
        detect_anomaly=bool(dataset_info.get("detect_anomaly", False)),
        profile_performance=bool(dataset_info.get("profile_performance", False)),
    )

def _csv_int_tuple(value: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


class _AllDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults even for arguments that do not define help text."""

    def _format_action_invocation(self, action: argparse.Action) -> str:
        invocation = super()._format_action_invocation(action)
        if (
            action.option_strings
            and action.help is None
            and not action.required
            and action.default is not argparse.SUPPRESS
        ):
            invocation += f" (default: {action.default})"
        return invocation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FragmentTreeTrainingModel.",
        formatter_class=_AllDefaultsHelpFormatter,
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for one fragment-tree training project.",
    )
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--mol-encoder-checkpoint", required=True)
    parser.add_argument("--condition-adduct-embedding-dim", type=int, default=16)
    parser.add_argument("--condition-ce-feature-dim", type=int, choices=(16,), default=16)
    parser.add_argument("--condition-ce-fc-dims", type=_csv_int_tuple, default=(32,))
    parser.add_argument("--condition-feature-dim", type=int, default=64)
    parser.add_argument("--condition-fc-dims", type=_csv_int_tuple, default=(128, 64))
    parser.add_argument("--tree-hidden-dim", type=int, default=128)
    parser.add_argument("--tree-num-layers", type=int, default=2)
    parser.add_argument("--tree-num-heads", type=int, default=8)
    parser.add_argument("--tree-max-degree", type=int, default=16)
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="Global dropout used throughout the constructed fragment-tree model.",
    )
    parser.add_argument("--edge-feature-dim", type=int, default=256)
    parser.add_argument("--edge-category-dim", type=int, default=32)
    parser.add_argument("--edge-attention-heads", type=int, default=8)
    parser.add_argument("--attention-max-graph-distance", type=int, default=4)
    parser.add_argument(
        "--max-edges-per-depth", type=_csv_int_tuple, default=(128, 64, 32)
    )
    parser.add_argument("--training-edges-per-sample", type=int, default=32)
    parser.add_argument("--training-zero-edge-fraction", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--assignment-score-threshold", type=float,
        default=DEFAULT_ASSIGNMENT_SCORE_THRESHOLD,
        help="Use samples whose assignment score is at least this value.",
    )
    parser.add_argument("--max-edges-per-step", type=int, default=128)
    parser.add_argument("--max-retained-edges", type=int, default=30)
    parser.add_argument("--max-edges-per-tree", type=int, default=256)
    parser.add_argument("--max-next-cleavage-candidates", type=int, default=3)
    parser.add_argument("--edge-condition-interaction-dim", type=int, default=64)
    parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--nearest-lower-partners", type=int, default=1)
    parser.add_argument("--extended-lower-partners", type=int, default=3)
    parser.add_argument("--background-partners", type=int, default=10)
    parser.add_argument("--ranking-intensity-threshold", type=float, default=0.05)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--ckpt-id", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=10)
    parser.add_argument("--validation-interval-steps", type=int, default=100)
    parser.add_argument(
        "--train-log-interval-steps",
        type=int,
        default=50,
        help="Log averaged training metrics every N successful steps. Default: 50.",
    )
    parser.add_argument(
        "--validate-at-start",
        action="store_true",
        help="Run ValStart before the first training epoch (disabled by default).",
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable PyTorch autograd anomaly detection (disabled by default).",
    )
    parser.add_argument(
        "--profile-performance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profile the largest estimated batch before training (default: disabled).",
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


def _workbench_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a training configuration that the VS Code Workbench can reload."""
    config: Dict[str, Any] = {"application": "fragment-tree-training"}
    for key, value in vars(args).items():
        parts = key.split("_")
        workbench_key = parts[0] + "".join(part.title() for part in parts[1:])
        config[workbench_key] = value
    return config


if __name__ == "__main__":
    args = parse_args()
    project_dir = Path(args.output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    train_split_dir = Path(args.train_dir)
    val_split_dir = Path(args.val_dir)
    train_data_dir = train_split_dir / "data" if (train_split_dir / "data").is_dir() else train_split_dir
    val_data_dir = val_split_dir / "data" if (val_split_dir / "data").is_dir() else val_split_dir
    preprocessing, preprocessing_config_path = load_and_validate_split_preprocessing(
        train_split_dir, val_split_dir
    )
    model_config_inline = build_model_config_from_pretrained(
        mol_encoder_checkpoint=args.mol_encoder_checkpoint,
        fragmenter_params=dict(preprocessing["fragmenter_params"]),
        condition_encoder_params={
            "adduct_embedding_dim": args.condition_adduct_embedding_dim,
            "ce_feature_dim": args.condition_ce_feature_dim,
            "ce_fc_dims": args.condition_ce_fc_dims,
            "feature_dim": args.condition_feature_dim,
            "fc_dims": args.condition_fc_dims,
        },
        fragment_edge_encoder_params={
            "feature_dim": args.edge_feature_dim,
            "category_dim": args.edge_category_dim,
            "num_heads": args.edge_attention_heads,
            "attention_max_graph_distance": args.attention_max_graph_distance,
            "max_edges_per_step": args.max_edges_per_step,
            "max_edges_per_depth": args.max_edges_per_depth,
            "training_edges_per_sample": args.training_edges_per_sample,
            "training_zero_edge_fraction": args.training_zero_edge_fraction,
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
            "max_edges_per_tree": args.max_edges_per_tree,
            "max_next_cleavage_candidates": args.max_next_cleavage_candidates,
            "edge_condition_interaction_dim": args.edge_condition_interaction_dim,
            "ranking_loss_weight": args.ranking_loss_weight,
            "top_n": args.top_n,
            "nearest_lower_partners": args.nearest_lower_partners,
            "extended_lower_partners": args.extended_lower_partners,
            "background_partners": args.background_partners,
            "ranking_intensity_threshold": args.ranking_intensity_threshold,
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
        detect_anomaly=args.detect_anomaly,
        profile_performance=args.profile_performance,
        max_samples=args.max_samples,
        assignment_score_threshold=args.assignment_score_threshold,
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
        workbench_config=_workbench_training_config(args),
    )
