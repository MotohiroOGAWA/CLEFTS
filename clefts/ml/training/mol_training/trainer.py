from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import MolPretrainingDataset, collate_mol_graphs
from .pretraining_model import MolPretrainingModel


DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_PATIENCE = 8
DEFAULT_MIN_DELTA = 1e-4


def move_batch(batch, device: torch.device):
    return batch.to(device)


def mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        out[key] = float(sum(vals) / max(len(vals), 1))
    return out


def write_tensorboard_metric_pairs(
    writer: SummaryWriter,
    *,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch: int,
) -> None:
    for name in sorted(set(train_metrics) | set(val_metrics)):
        values = {}
        if name in train_metrics:
            values["train"] = float(train_metrics[name])
        if name in val_metrics:
            values["val"] = float(val_metrics[name])
        writer.add_scalars(name, values, epoch)


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
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    best = {"val_loss": float("inf"), "epoch": -1}
    stale_epochs = 0
    metrics_path = output_dir / f"{stage_name}_metrics.csv"
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
    write_tensorboard_command(tensorboard_dir)

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = None
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
            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                node_mask_ratio=node_mask_ratio,
                edge_mask_ratio=edge_mask_ratio,
            )
            record = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                writer.writeheader()
            writer.writerow(record)
            f.flush()
            write_tensorboard_metric_pairs(
                tb_writer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
            )
            tb_writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

            val_loss = float(val_metrics.get("loss", float("inf")))
            if val_loss < best["val_loss"] - DEFAULT_MIN_DELTA:
                best = {"val_loss": val_loss, "epoch": epoch}
                stale_epochs = 0
                save_checkpoint(model, output_dir / f"{stage_name}_best.pt", extra={"stage": stage_name, **best})
            else:
                stale_epochs += 1
                if DEFAULT_PATIENCE > 0 and stale_epochs >= DEFAULT_PATIENCE:
                    break

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


def make_loader(dataset: MolPretrainingDataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_mol_graphs,
    )
