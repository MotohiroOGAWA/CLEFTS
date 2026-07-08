from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import make_cleavage_structure_dataloader
from .pretraining_model import CleavagePretrainingModel



def mol_encoder_params(model: CleavagePretrainingModel) -> Dict[str, object]:
    mol_encoder = model.mol_encoder
    dropout = 0.0
    if mol_encoder.encoder.num_layers > 0:
        dropout = float(mol_encoder.encoder.blocks[0].mha.self_attn.mha.dropout.p)
    return {
        "symbols": tuple(mol_encoder.symbols),
        "node_dim": int(mol_encoder.node_dim),
        "graph_dim": int(mol_encoder.graph_dim),
        "num_layers": int(mol_encoder.encoder.num_layers),
        "num_heads": int(mol_encoder.encoder.num_heads),
        "max_degree": int(mol_encoder.encoder.max_degree),
        "max_spatial_dist": int(mol_encoder.encoder.max_spatial_dist),
        "max_edge_dist": int(mol_encoder.encoder.max_edge_dist),
        "dropout": dropout,
    }


def checkpoint_payload(
    model: CleavagePretrainingModel,
    *,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "mol_encoder_state_dict": model.mol_encoder.state_dict(),
        "mol_encoder_params": mol_encoder_params(model),
        "cleavage_edge_fnet_state_dict": model.cleavage_edge_fnet.state_dict(),
        "cleavage_edge_fnet_params": model.cleavage_edge_fnet.config_dict(),
        "cleavage_pretraining_params": model.training_config_dict(),
        "metrics": {"train": train_metrics, "val": val_metrics},
        "extra": dict(extra or {}),
    }
    return payload

def move_structure_batch(batch: Dict[str, object], device: torch.device):
    return batch["structure"].to(device)


def mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if key.endswith("_count"):
            out[key] = float(sum(vals))
            continue
        count_key = key[:-4] + "_count" if key.endswith("_acc") else None
        if count_key and any(count_key in row for row in rows):
            total = 0.0
            count = 0.0
            for row in rows:
                if key in row and count_key in row:
                    w = float(row[count_key])
                    total += float(row[key]) * w
                    count += w
            if count > 0.0:
                out[key] = total / count
                continue
        out[key] = float(sum(vals) / max(len(vals), 1))
    return out


def tensorboard_tag(name: str) -> str:
    if name == "loss":
        return "loss"
    for prefix in ("observed_edge", "pattern", "reaction", "product_molecule", "atom_location"):
        if name.startswith(prefix + "_"):
            metric = name.rsplit("_", 1)[-1]
            if metric in {"loss", "acc", "count"}:
                chart = prefix if prefix != "observed_edge" else "edge"
                return f"{prefix}_{metric}/{chart}"
    return name


def tensorboard_series(name: str) -> str:
    for suffix in ("_loss", "_acc", "_count"):
        if name.endswith(suffix):
            rest = name[: -len(suffix)]
            for prefix in ("observed_edge", "pattern", "reaction", "product_molecule", "atom_location"):
                if rest == prefix:
                    return "total"
                if rest.startswith(prefix + "_"):
                    return rest[len(prefix) + 1 :]
    return "total"


def write_tensorboard_pairs(writer: SummaryWriter, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int) -> None:
    grouped: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(train_metrics) | set(val_metrics)):
        tag = tensorboard_tag(name)
        series = tensorboard_series(name)
        values = grouped.setdefault(tag, {})
        if name in train_metrics:
            values[f"train/{series}"] = float(train_metrics[name])
        if name in val_metrics:
            values[f"val/{series}"] = float(val_metrics[name])
    for tag, values in grouped.items():
        writer.add_scalars(tag, values, epoch)


def run_epoch(model: CleavagePretrainingModel, loader, *, device: torch.device, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    rows = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if is_train else "val", mininterval=1.0):
            structure = move_structure_batch(batch, device)
            output = model(structure)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            rows.append(output.metrics)
    return mean_metrics(rows)


def train(
    *,
    model: CleavagePretrainingModel,
    train_dir: str | Path,
    val_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> None:
    device = torch.device(device)
    model.to(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tb_dir = output_path / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    (output_path / "tensorboard_command.txt").write_text(f"tensorboard --logdir {tb_dir} --port 6006 --host 0.0.0.0\n")

    train_loader = make_cleavage_structure_dataloader(train_dir, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_cleavage_structure_dataloader(val_dir, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    metrics_file = output_path / "metrics.csv"
    best_val = float("inf")
    with metrics_file.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer_csv.writeheader()
        for epoch in range(1, int(epochs) + 1):
            train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
            val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
            write_tensorboard_pairs(writer, train_metrics, val_metrics, epoch)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", float("nan")),
                "val_loss": val_metrics.get("loss", float("nan")),
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer_csv.writerow(row)
            f.flush()
            ckpt = checkpoint_payload(
                model,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                extra={"best_val_loss": best_val},
            )
            torch.save(ckpt, output_path / "last.pt")
            val_loss = float(val_metrics.get("loss", float("inf")))
            if val_loss < best_val:
                best_val = val_loss
                ckpt["extra"]["best_val_loss"] = best_val
                torch.save(ckpt, output_path / "best.pt")
    writer.close()
