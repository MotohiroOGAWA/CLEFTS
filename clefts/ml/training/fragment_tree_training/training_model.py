from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from ...common.torch_utils.check_point_manager import CheckPointManager, CkptNode
from ...common.torch_utils.training_setup import get_optimizer
from ...input.fragment_tree_training_data import (
    FragmentTreeStructureFileDataset,
    collate_fragment_tree_structure_items,
)
from ...specgen.fragment_tree_spectrum_predictor import FragmentSpectrumGenerator
from ...specgen.fragment_tree_training_model import FragmentTreeTrainingModel

METRIC_COLUMNS = (
    "epoch",
    "global_step",
    "train_loss",
    "val_loss",
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
    model = FragmentTreeTrainingModel(generator.candidate_selector).to(device)
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
) -> Tuple[Path, Optional[str], int, torch.device, int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any], Path]:
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
    dataset_info = {
        "training_structure_dir": str(training_structure_dir),
        "validation_structure_dir": str(validation_structure_dir),
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
    # val_dataset = FragmentTreeStructureFileDataset(
    #     Path(dataset_info["validation_structure_dir"]),
    #     pattern=pattern,
    # )
    val_dataset = None
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
    # val_loader = DataLoader(
    #     val_dataset,
    #     shuffle=False,
    #     **loader_kwargs,
    # )
    val_loader = None

    extra_data = {
        "training_structure_dir": str(dataset_info["training_structure_dir"]),
        "validation_structure_dir": str(dataset_info["validation_structure_dir"]),
        "pattern": pattern,
        "train_size": len(train_dataset),
        # "val_size": len(val_dataset),
        "val_size": 0,
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
) -> Tuple[float, int]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    iterator = tqdm(loader, desc=desc)

    for batch in iterator:
        structure = batch["structure"].to(device)
        num_samples = int(structure.num_samples)

        with torch.set_grad_enabled(is_train):
            output = model(structure)
            loss = output["loss"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * max(num_samples, 1)
        total_samples += max(num_samples, 1)
        iterator.set_postfix(loss=total_loss / max(total_samples, 1))

    if total_samples == 0:
        return 0.0, 0
    return total_loss / total_samples, total_samples


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
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_info: Dict[str, Any],
    early_stopping_info: Dict[str, Any],
    run_dir: Path,
    extra_data: Dict[str, Any],
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

    for epoch_index in range(state.initial_epoch, max_epoch + 1):
        train_loss, train_samples = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            desc=f"Train({epoch_index}/{max_epoch})",
        )
        global_step += len(train_loader)

        # if len(val_loader) > 0:
        #     with torch.no_grad():
        #         val_loss, _ = run_epoch(
        #             model=model,
        #             loader=val_loader,
        #             device=device,
        #             optimizer=None,
        #             desc=f"Val({epoch_index}/{max_epoch})",
        #         )
        # else:
        #     val_loss = train_loss
        val_loss = train_loss

        step_scheduler(scheduler, val_loss)

        lr = float(optimizer.param_groups[0]["lr"])
        metric_row = {
            "epoch": epoch_index,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        }
        ckpt_manager.log_metrics(**metric_row)
        ckpt_manager.flush_metrics(flush_dir=str(run_dir))
        writer.add_scalar("loss/train", train_loss, global_step)
        writer.add_scalar("loss/val", val_loss, global_step)
        writer.add_scalar("optimizer/lr", lr, global_step)

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
            ckpt_manager.update_topk(
                score=val_loss,
                epoch=epoch_index,
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

        should_save = improved or (save_interval > 0 and epoch_index % save_interval == 0)
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
                comment="best" if improved else "interval",
            )

        print(
            f"epoch={epoch_index} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} lr={lr:.6g} samples={train_samples} "
            f"branch_id={ckpt_manager.current_branch_id}"
        )

        if patience is not None and bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch_index}.")
            break

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
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_info=optimizer_info,
        early_stopping_info=early_stopping_info,
        run_dir=run_dir,
        extra_data=extra_data,
    )
