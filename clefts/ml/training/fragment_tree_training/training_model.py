from __future__ import annotations

import argparse
import json
import shutil
import math
import traceback
from dataclasses import dataclass
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
from ...specgen.fragment_tree_training_model import FragmentTreeTrainingModel
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
    "lr",
)


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


def load_generator(model_config: Dict[str, Any], device: torch.device) -> FragmentSpectrumGenerator:
    params = model_config.get("params", model_config)
    generator = FragmentSpectrumGenerator(**params).to(device)
    return generator


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


def prepare_train(
    project_dir: str | Path,
    train_config_path: str | Path,
    *,
    root_run_dir: Optional[str | Path] = None,
) -> Tuple[Path, Optional[str], int, torch.device, int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[int], Path]:
    train_config = load_config(train_config_path)

    load_name = train_config.get("experiment_name")
    if not load_name:
        load_name = "exp_" + datetime.now().strftime("%Y%m%d%H%M%S")

    experiment_dir = Path(project_dir) / "experiments" / str(load_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    ckpt_id = train_config.get("ckpt_id")
    batch_size = int(train_config.get("batch_size", 1))
    device = torch.device(train_config.get("device", "cpu"))
    epochs = int(train_config.get("epoch", train_config.get("epochs", 10)))
    save_interval = int(train_config.get("save_interval", 1))
    validation_interval_steps_value = train_config.get("validation_interval_steps")
    validation_interval_steps = (
        None
        if validation_interval_steps_value in {None, ""}
        else int(validation_interval_steps_value)
    )
    if validation_interval_steps is not None and validation_interval_steps <= 0:
        raise ValueError("validation_interval_steps must be positive when specified.")
    optimizer_info = dict(train_config.get("optimizer", {"name": "AdamW", "lr": 1e-4}))
    early_stopping_info = dict(train_config.get("early_stopping", {}))
    training_structure_dir = train_config.get(
        "training_structure_dir",
        train_config.get("traing_structure_dir"),
    )
    validation_structure_dir = train_config.get(
        "validation_structure_dir",
        train_config.get("val_structure_dir"),
    )
    if not training_structure_dir:
        raise ValueError("train_config requires training_structure_dir.")
    if not validation_structure_dir:
        raise ValueError("train_config requires validation_structure_dir.")
    validation_valid_records_file = train_config.get(
        "validation_valid_records_file",
        train_config.get("validation_msdataset_file"),
    )
    if not validation_valid_records_file:
        validation_valid_records_file = (
            Path(validation_structure_dir) / "valid_records.msds"
        )
    dataset_info = {
        "training_structure_dir": str(training_structure_dir),
        "validation_structure_dir": str(validation_structure_dir),
        "validation_valid_records_file": str(validation_valid_records_file),
        "shuffle": bool(train_config.get("shuffle", True)),
        "validation_interval_steps": validation_interval_steps,
        "pattern": "*.pt",
    }

    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = Path(root_run_dir) / now_str if root_run_dir is not None else experiment_dir / "runs" / now_str
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(train_config_path, run_dir / Path(train_config_path).name)

    return (
        experiment_dir,
        None if ckpt_id in ("", None) else str(ckpt_id),
        batch_size,
        device,
        epochs,
        save_interval,
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
) -> EpochLossMetrics:
    is_train = optimizer is not None
    model.train(is_train)

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
            if (
                is_train
                and validation_interval_steps is not None
                and validation_interval_steps > 0
                and on_validation_step is not None
                and global_step % validation_interval_steps == 0
            ):
                on_validation_step(global_step, cumulative_metrics, window_metrics)
                model.train(is_train)
                window_loss = 0.0
                window_selection_loss = 0.0
                window_intensity_loss = 0.0
                window_samples = 0
                window_selection_samples = 0
                window_intensity_samples = 0
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


def evaluate_validation_cosine(
    *,
    model: FragmentTreeTrainingModel,
    dataset: MSDataset,
    batch_size: int = 128,
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
    with tqdm(total=1, desc="ValCosine", leave=True) as iterator:
        iterator.set_postfix(val_cosine=val_cosine)
        iterator.update(1)
    return val_cosine


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
        max_generation_steps=0,
        normalize_intensity=True,
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

            valid_pairs.sort(key=lambda pair: pair[1])
            structure = FragmentTreeStructure.from_structures(
                [builder.to_structure()],
                device=device,
            )

            with torch.no_grad():
                output = predictor(structure)

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
    batch_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_info: Dict[str, Any],
    early_stopping_info: Dict[str, Any],
    run_dir: Path,
    extra_data: Dict[str, Any],
    validation_interval_steps: Optional[int] = None,
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
    validation_valid_records_file = Path(extra_data["validation_valid_records_file"])
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

    def evaluate_current_validation(desc: str) -> Tuple[EpochLossMetrics, float]:
        if val_loader is None or len(val_loader) <= 0:
            return nan_loss_metrics(), float("nan")
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
                )
                if validation_dataset is not None
                else float("nan")
            )
        return metrics, cosine

    def log_training_metrics(
        *,
        event: str,
        epoch_value: int,
        step_value: int,
        train_metrics: EpochLossMetrics,
        train_window_metrics: Optional[EpochLossMetrics],
        val_metrics: EpochLossMetrics,
        val_cosine: float,
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
        val_metrics, val_cosine = evaluate_current_validation(
            desc=f"ValStart({validation_epoch})"
        )
        log_training_metrics(
            event="epoch_start",
            epoch_value=validation_epoch,
            step_value=global_step,
            train_metrics=last_train_metrics,
            train_window_metrics=None,
            val_metrics=val_metrics,
            val_cosine=val_cosine,
        )

        if validation_epoch >= state.initial_epoch:
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
            step_val_metrics, step_val_cosine = evaluate_current_validation(
                desc=f"ValStep({step_value})"
            )
            log_training_metrics(
                event="step",
                epoch_value=epoch_index,
                step_value=step_value,
                train_metrics=train_epoch_metrics,
                train_window_metrics=train_window_metrics,
                val_metrics=step_val_metrics,
                val_cosine=step_val_cosine,
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
        val_metrics, val_cosine = evaluate_current_validation(
            desc=f"ValFinal({last_epoch_index})"
        )
        log_training_metrics(
            event="epoch_end",
            epoch_value=last_epoch_index,
            step_value=global_step,
            train_metrics=last_train_metrics,
            train_window_metrics=None,
            val_metrics=val_metrics,
            val_cosine=val_cosine,
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FragmentTreeTrainingModel.")
    parser.add_argument("-project", "--project-dir", default="data/training/fragment_tree_model")
    parser.add_argument("-model", "--model-config-path", default="clefts/ml/specgen/presets/fragment_spectrum_generator_param.json")
    parser.add_argument("-train", "--train-config-path", required=True)
    parser.add_argument("--root-run-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    (
        experiment_dir,
        ckpt_id,
        batch_size,
        device,
        epochs,
        save_interval,
        optimizer_info,
        early_stopping_info,
        dataset_info,
        validation_interval_steps,
        run_dir,
    ) = prepare_train(
        args.project_dir,
        args.train_config_path,
        root_run_dir=args.root_run_dir,
    )

    _, _, train_loader, val_loader, extra_data = setup_dataset(
        dataset_info,
        batch_size,
        num_workers=args.num_workers,
    )

    model_config = load_config(args.model_config_path)
    shutil.copy(args.model_config_path, run_dir / Path(args.model_config_path).name)

    main(
        model_config=model_config,
        experiment_dir=experiment_dir,
        ckpt_id=ckpt_id,
        device=device,
        epoch=epochs,
        save_interval=save_interval,
        batch_size=batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_info=optimizer_info,
        early_stopping_info=early_stopping_info,
        run_dir=run_dir,
        extra_data=extra_data,
        validation_interval_steps=validation_interval_steps,
    )
