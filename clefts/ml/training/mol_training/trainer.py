from __future__ import annotations

import csv
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ...common.torch_utils.early_stopping import EarlyStopping
from .dataset import MolPretrainingDataset, collate_mol_graphs
from .pretraining_model import MolPretrainingModel


DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_MIN_DELTA = 1e-4


class BalancedFeatureBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        feature_record_index: Dict[str, Dict[str, Dict[str, List[int]]]],
        patience: int,
        max_forced_per_batch: int,
        shuffle: bool = True,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(max(1, batch_size))
        self.patience = int(max(1, patience))
        self.max_forced_per_batch = int(max(0, max_forced_per_batch))
        self.shuffle = bool(shuffle)
        self._rng = random.Random()
        self._records_by_key: Dict[Tuple[str, str, str], List[int]] = {}
        self._record_sets_by_key: Dict[Tuple[str, str, str], set[int]] = {}
        for level, groups in feature_record_index.items():
            for group_name, labels in groups.items():
                for label, records in labels.items():
                    clean_records = sorted({int(index) for index in records})
                    if not clean_records:
                        continue
                    key = (str(level), str(group_name), str(label))
                    self._records_by_key[key] = clean_records
                    self._record_sets_by_key[key] = set(clean_records)
        self._steps_since_sampled = {key: self.patience for key in self._records_by_key}

    def __len__(self) -> int:
        if self.dataset_size <= 0:
            return 0
        return int(math.ceil(self.dataset_size / self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        if self.dataset_size <= 0:
            return
        if self.shuffle:
            order = torch.randperm(self.dataset_size).tolist()
        else:
            order = list(range(self.dataset_size))
        for start in range(0, self.dataset_size, self.batch_size):
            batch = [int(index) for index in order[start : start + self.batch_size]]
            if batch and self.max_forced_per_batch > 0 and self._records_by_key:
                self._force_due_records(batch)
            self._update_steps(batch)
            yield batch

    def _force_due_records(self, batch: List[int]) -> None:
        batch_set = set(batch)
        due_keys = sorted(
            self._records_by_key,
            key=lambda key: (
                self._steps_since_sampled[key],
                -len(self._records_by_key[key]),
            ),
            reverse=True,
        )
        replacement_slot = 0
        forced = 0
        for key in due_keys:
            if forced >= self.max_forced_per_batch or replacement_slot >= len(batch):
                break
            if self._steps_since_sampled[key] < self.patience:
                continue
            if batch_set & self._record_sets_by_key[key]:
                continue
            record_index = self._rng.choice(self._records_by_key[key])
            old_index = batch[replacement_slot]
            batch[replacement_slot] = record_index
            batch_set.discard(old_index)
            batch_set.add(record_index)
            replacement_slot += 1
            forced += 1

    def _update_steps(self, batch: Sequence[int]) -> None:
        batch_set = set(int(index) for index in batch)
        for key, record_set in self._record_sets_by_key.items():
            if batch_set & record_set:
                self._steps_since_sampled[key] = 0
            else:
                self._steps_since_sampled[key] += 1


def move_batch(batch, device: torch.device):
    return batch.to(device)


def mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if key.endswith("_count"):
            out[key] = float(sum(vals))
            continue
        count_key = key[:-4] + "_count" if key.endswith("_acc") else None
        if count_key and any(count_key in row for row in rows):
            weighted_sum = 0.0
            total_count = 0.0
            for row in rows:
                if key not in row or count_key not in row:
                    continue
                count = float(row[count_key])
                weighted_sum += float(row[key]) * count
                total_count += count
            if total_count > 0.0:
                out[key] = weighted_sum / total_count
                continue
        out[key] = float(sum(vals) / max(len(vals), 1))

    descriptor_names = sorted(
        match.group(1)
        for key in list(out.keys())
        if (match := re.match(r"^descriptor_(.+)_sse$", key)) is not None
    )
    for name in descriptor_names:
        sse = float(out.get(f"descriptor_{name}_sse", 0.0))
        target_sum = float(out.get(f"descriptor_{name}_target_sum", 0.0))
        target_sq_sum = float(out.get(f"descriptor_{name}_target_sq_sum", 0.0))
        count = float(out.get(f"descriptor_{name}_count", 0.0))
        denom = target_sq_sum - (target_sum * target_sum / count) if count > 0.0 else 0.0
        if denom > 1e-12:
            out[f"descriptor_{name}_r2"] = 1.0 - sse / denom
        else:
            out[f"descriptor_{name}_r2"] = 1.0 if sse <= 1e-12 else 0.0
    for key in list(out.keys()):
        if DESCRIPTOR_R2_ACCUM_RE.match(key):
            del out[key]
    return out


FEATURE_GROUP_PATTERN = "symbol|charge|ring_type|hybridization|num_hydrogens|valence_electrons|bond_type"
CLASS_METRIC_RE = re.compile(rf"^(node|edge)_({FEATURE_GROUP_PATTERN})_(.+)_(acc|count)$")
GROUP_METRIC_RE = re.compile(rf"^(node|edge)_({FEATURE_GROUP_PATTERN})_(.+)$")
DESCRIPTOR_ITEM_LOSS_RE = re.compile(r"^descriptor_(.+)_loss$")
DESCRIPTOR_ITEM_R2_RE = re.compile(r"^descriptor_(.+)_r2$")
DESCRIPTOR_R2_ACCUM_RE = re.compile(r"^descriptor_(.+)_(sse|target_sum|target_sq_sum|count)$")
FEATURE_DISPLAY_NAMES = {
    "symbol": "element",
    "charge": "charge",
    "ring_type": "ring_type",
    "hybridization": "hybridization",
    "num_hydrogens": "hydrogen_count",
    "valence_electrons": "valence_electrons",
    "bond_type": "bond_type",
}


def tensorboard_metric_tag(name: str) -> str:
    match = GROUP_METRIC_RE.match(name)
    if not match:
        return name
    prefix, group_name, metric_name = match.groups()
    display_name = FEATURE_DISPLAY_NAMES.get(group_name, group_name)
    if metric_name == "acc":
        return f"{prefix}_acc/{display_name}"
    return f"{prefix}/{display_name}_{metric_name}"


def should_skip_metric_pair(name: str) -> bool:
    descriptor_loss_match = DESCRIPTOR_ITEM_LOSS_RE.match(name)
    if descriptor_loss_match and descriptor_loss_match.group(1) != "":
        return name != "descriptor_loss"
    if DESCRIPTOR_ITEM_R2_RE.match(name):
        return True
    if DESCRIPTOR_R2_ACCUM_RE.match(name):
        return True
    return CLASS_METRIC_RE.match(name) is not None


def write_tensorboard_grouped_class_metrics(
    writer: SummaryWriter,
    *,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch: int,
) -> None:
    grouped: Dict[str, Dict[str, float]] = {}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for name, value in metrics.items():
            match = CLASS_METRIC_RE.match(name)
            if not match:
                continue
            prefix, group_name, label, metric_name = match.groups()
            display_name = FEATURE_DISPLAY_NAMES.get(group_name, group_name)
            tag_prefix = f"{prefix}_acc" if metric_name == "acc" else prefix
            tag = f"{tag_prefix}/{display_name}_{metric_name}_by_class"
            grouped.setdefault(tag, {})[f"{split}/{label}"] = float(value)
    for tag, values in grouped.items():
        if values:
            writer.add_scalars(tag, values, epoch)


def write_tensorboard_grouped_descriptor_losses(
    writer: SummaryWriter,
    *,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch: int,
) -> None:
    loss_values = {}
    r2_values = {}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for name, value in metrics.items():
            loss_match = DESCRIPTOR_ITEM_LOSS_RE.match(name)
            if loss_match and name != "descriptor_loss":
                loss_values[f"{split}/{loss_match.group(1)}"] = float(value)
                continue
            r2_match = DESCRIPTOR_ITEM_R2_RE.match(name)
            if r2_match:
                r2_values[f"{split}/{r2_match.group(1)}"] = float(value)
    if loss_values:
        writer.add_scalars("descriptor/loss_by_target", loss_values, epoch)
    if r2_values:
        writer.add_scalars("descriptor/r2_by_target", r2_values, epoch)


def write_tensorboard_metric_pairs(
    writer: SummaryWriter,
    *,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch: int,
) -> None:
    for name in sorted(set(train_metrics) | set(val_metrics)):
        if should_skip_metric_pair(name):
            continue
        values = {}
        if name in train_metrics:
            values["train"] = float(train_metrics[name])
        if name in val_metrics:
            values["val"] = float(val_metrics[name])
        writer.add_scalars(tensorboard_metric_tag(name), values, epoch)


def write_tensorboard_command(log_dir: Path) -> None:
    command = f"tensorboard --logdir {log_dir} --port 6006 --host 0.0.0.0"
    with open(log_dir.parent / "tensorboard_command.txt", "w", encoding="utf-8") as f:
        f.write(command + "\n")



@torch.no_grad()
def evaluate(
    model: MolPretrainingModel,
    loader: DataLoader,
    *,
    device: torch.device,
    node_mask_ratio: float,
    edge_mask_ratio: float,
) -> Dict[str, float]:
    model.eval()
    metrics = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch,
            node_mask_ratio=node_mask_ratio,
            edge_mask_ratio=edge_mask_ratio,
        )
        metrics.append(output.metrics)
    return mean_metrics(metrics)


def evaluate_timed(
    model: MolPretrainingModel,
    loader: DataLoader,
    *,
    device: torch.device,
    node_mask_ratio: float,
    edge_mask_ratio: float,
) -> Tuple[Dict[str, float], float]:
    start = time.perf_counter()
    metrics = evaluate(
        model,
        loader,
        device=device,
        node_mask_ratio=node_mask_ratio,
        edge_mask_ratio=edge_mask_ratio,
    )
    return metrics, time.perf_counter() - start


def train_epochs(
    model: MolPretrainingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    node_mask_ratio: float,
    edge_mask_ratio: float,
    output_dir: Path,
    stage_name: str,
    early_stopping_patience: Optional[int] = None,
    early_stopping_window_size: int = 1,
    early_stopping_min_delta: float = DEFAULT_MIN_DELTA,
    early_stopping_reset_step: float = 1.0,
    early_stopping_verbose: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    best = {
        "val_loss": float("inf"),
        "epoch": -1,
        "stopped_epoch": int(epochs),
        "early_stopping_enabled": bool(early_stopping_patience is not None and early_stopping_patience > 0),
    }
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        window_size=early_stopping_window_size,
        min_delta=early_stopping_min_delta,
        reset_step=early_stopping_reset_step,
        mode="min",
        verbose=early_stopping_verbose,
    )
    metrics_path = output_dir / f"{stage_name}_metrics.csv"
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
    write_tensorboard_command(tensorboard_dir)

    metric_records: List[Dict[str, float]] = []
    metric_fieldnames = ["epoch"]

    def write_epoch_record(
        csv_file,
        *,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        metric_records.append(record)
        for key in record.keys():
            if key not in metric_fieldnames:
                metric_fieldnames.append(key)

        csv_file.seek(0)
        csv_file.truncate()
        csv_writer = csv.DictWriter(csv_file, fieldnames=metric_fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(metric_records)
        csv_file.flush()
        write_tensorboard_metric_pairs(
            tb_writer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch=epoch,
        )
        write_tensorboard_grouped_class_metrics(
            tb_writer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch=epoch,
        )
        write_tensorboard_grouped_descriptor_losses(
            tb_writer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch=epoch,
        )
        tb_writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch)
        tb_writer.flush()

    with open(metrics_path, "w+", newline="", encoding="utf-8") as f:

        epoch0_train_metrics = evaluate(
            model,
            train_loader,
            device=device,
            node_mask_ratio=node_mask_ratio,
            edge_mask_ratio=edge_mask_ratio,
        )
        epoch0_val_metrics, epoch0_val_seconds = evaluate_timed(
            model,
            val_loader,
            device=device,
            node_mask_ratio=node_mask_ratio,
            edge_mask_ratio=edge_mask_ratio,
        )
        write_epoch_record(
            f,
            epoch=0,
            train_metrics=epoch0_train_metrics,
            val_metrics=epoch0_val_metrics,
        )
        val_loss = float(epoch0_val_metrics.get("loss", float("inf")))
        early_stopping(val_loss)
        final_epoch = 0
        final_train_metrics = epoch0_train_metrics
        final_val_metrics = epoch0_val_metrics
        final_val_seconds = epoch0_val_seconds
        if val_loss < best["val_loss"] - DEFAULT_MIN_DELTA:
            best.update({"val_loss": val_loss, "epoch": 0})
            save_checkpoint(model, output_dir / f"{stage_name}_best.pt", extra={"stage": stage_name, **best})

        for epoch in range(1, epochs + 1):
            model.train()
            rows = []
            progress = tqdm(train_loader, desc=f"{stage_name} epoch {epoch}", leave=False)
            for batch in progress:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    batch,
                    node_mask_ratio=node_mask_ratio,
                    edge_mask_ratio=edge_mask_ratio,
                )
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP_NORM)
                optimizer.step()
                rows.append(output.metrics)
                progress.set_postfix(loss=f"{output.metrics.get('loss', 0.0):.4f}")

            train_metrics = mean_metrics(rows)
            val_metrics, val_seconds = evaluate_timed(
                model,
                val_loader,
                device=device,
                node_mask_ratio=node_mask_ratio,
                edge_mask_ratio=edge_mask_ratio,
            )
            write_epoch_record(
                f,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            final_epoch = epoch
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics
            final_val_seconds = val_seconds
            val_loss = float(val_metrics.get("loss", float("inf")))
            if val_loss < best["val_loss"] - DEFAULT_MIN_DELTA:
                best.update({"val_loss": val_loss, "epoch": epoch})
                save_checkpoint(model, output_dir / f"{stage_name}_best.pt", extra={"stage": stage_name, **best})
            early_stopping(val_loss)
            if early_stopping.early_stop:
                best["stopped_epoch"] = epoch
                break

    best["final_epoch"] = final_epoch
    best["final_train_metrics"] = final_train_metrics
    best["final_val_metrics"] = final_val_metrics
    best["last_validation_seconds"] = final_val_seconds
    best["early_stopping_counter"] = early_stopping.counter
    tb_writer.close()
    save_checkpoint(model, output_dir / f"{stage_name}_last.pt", extra={"stage": stage_name, **best})
    with open(output_dir / f"{stage_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best


def save_checkpoint(model: MolPretrainingModel, path: Path, *, extra: Optional[Dict] = None) -> None:
    mol_encoder = model.mol_encoder
    payload = {
        "model_state_dict": model.state_dict(),
        "mol_encoder_state_dict": mol_encoder.state_dict(),
        "descriptor_names": tuple(getattr(model, "descriptor_names", ())),
        "mol_encoder_params": {
            "symbols": mol_encoder.symbols,
            "node_dim": mol_encoder.node_dim,
            "graph_dim": mol_encoder.graph_dim,
            "num_layers": mol_encoder.encoder.num_layers,
            "num_heads": mol_encoder.encoder.num_heads,
            "max_degree": mol_encoder.encoder.max_degree,
            "max_spatial_dist": mol_encoder.encoder.max_spatial_dist,
            "max_edge_dist": mol_encoder.encoder.max_edge_dist,
            "dropout": mol_encoder.encoder.blocks[0].mha.self_attn.mha.dropout.p
            if mol_encoder.encoder.num_layers > 0
            else 0.0,
        },
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)


def make_loader(
    dataset: MolPretrainingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    balanced_feature_record_index: Optional[Dict[str, Dict[str, Dict[str, List[int]]]]] = None,
    balance_patience: int = 20,
    balance_max_forced_per_batch: int = 8,
) -> DataLoader:
    if shuffle and balanced_feature_record_index is not None and balance_max_forced_per_batch > 0:
        batch_sampler = BalancedFeatureBatchSampler(
            dataset_size=len(dataset),
            batch_size=batch_size,
            feature_record_index=balanced_feature_record_index,
            patience=balance_patience,
            max_forced_per_batch=balance_max_forced_per_batch,
            shuffle=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_mol_graphs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_mol_graphs,
    )
