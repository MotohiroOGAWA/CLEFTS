from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ...common.torch_utils.early_stopping import EarlyStopping
from .dataset import first_stage_edge_mask, make_cleavage_structure_dataloader
from .pretraining_model import (
    CleavagePretrainingModel,
    atom_neighborhood_indices,
    canonical_fragment_smiles,
)
from .preprocessing import (
    BalancedCleavageBatchSampler,
    load_or_build_preprocessing_cache,
    write_preprocessing_reports,
)


DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_MIN_DELTA = 1e-4



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
    return (
        batch["structure"].to(device),
        batch["selected_cleavage_event_rows"].to(device),
    )


def structure_target_counts(loader, *, surrounding_radius: int) -> Dict[str, object]:
    pattern = Counter()
    reaction = Counter()
    product = Counter()
    joint = Counter()
    reactant_structure = Counter()
    surrounding_structure = Counter()
    structure_count = event_count = edge_count = 0
    all_stage_event_count = all_stage_edge_count = 0
    for item in tqdm(loader.dataset, desc="Scanning cleavage targets", leave=False):
        structure = item.structure
        structure_count += 1
        all_stage_edge_count += int(structure.num_edges)
        all_stage_event_count += int(structure.num_cleavage_events)
        stage_edge_mask = first_stage_edge_mask(structure)
        edge_count += int(stage_edge_mask.sum().item())
        for event_idx, event in enumerate(structure.cleavage_event.tolist()):
            edge_idx = int(structure.cleavage_event_edge_index[event_idx].item())
            if edge_idx < 0 or edge_idx >= int(structure.num_edges):
                continue
            if not bool(stage_edge_mask[edge_idx]):
                continue
            event_count += 1
            pattern_id, reaction_id, product_id = map(int, event[:3])
            pattern[pattern_id] += 1
            reaction[reaction_id] += 1
            product[product_id] += 1
            joint[(pattern_id, reaction_id, product_id)] += 1
            src_node = int(structure.edge_index[0, edge_idx].item())

            reactant_table = structure.reactant_tuple_length_table.long()
            reactant_mask = (
                (reactant_table[:, 0] == pattern_id)
                & (reactant_table[:, 1] == reaction_id)
            )
            if not bool(reactant_mask.any()):
                continue
            reactant_length = int(reactant_table[reactant_mask][0, 2].item())
            reactant_atoms = structure.cleavage_atom_idxs[reactant_length][
                int(event[3])
            ].long()

            structure_key = canonical_fragment_smiles(
                str(structure.node_smiles[src_node]),
                tuple(int(value) for value in reactant_atoms.tolist()),
            )
            if structure_key is not None:
                reactant_structure[structure_key] += 1

            src_smiles = str(structure.node_smiles[src_node])
            neighborhoods = [
                atom_neighborhood_indices(
                    src_smiles, center_atom=int(center), radius=surrounding_radius
                )
                for center in reactant_atoms.tolist()
            ]
            if any(value is not None for value in neighborhoods):
                surrounding_structure["local_graph"] += 1
    return {
        "structure_count": structure_count,
        "edge_count": edge_count,
        "event_count": event_count,
        "all_stage_edge_count": all_stage_edge_count,
        "all_stage_event_count": all_stage_event_count,
        "stage": 1,
        "pattern": {str(key): value for key, value in sorted(pattern.items())},
        "reaction": {str(key): value for key, value in sorted(reaction.items())},
        "product_molecule": {str(key): value for key, value in sorted(product.items())},
        "pattern_reaction_product": {
            "/".join(map(str, key)): value for key, value in sorted(joint.items())
        },
        "reactant_structure": dict(sorted(reactant_structure.items())),
        "surrounding_structure": dict(sorted(surrounding_structure.items())),
        "surrounding_structure_radius": int(surrounding_radius),
    }


def write_target_report(
    output_dir: Path,
    *,
    train_loader,
    val_loader,
    writer: SummaryWriter,
    surrounding_radius: int,
) -> Dict[str, object]:
    report = {
        "train": structure_target_counts(
            train_loader, surrounding_radius=surrounding_radius
        ),
        "val": structure_target_counts(
            val_loader, surrounding_radius=surrounding_radius
        ),
    }
    with (output_dir / "target_class_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (output_dir / "target_class_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=["split", "target", "class_id", "count"]
        )
        csv_writer.writeheader()
        for split, split_report in report.items():
            for target in (
                "pattern",
                "reaction",
                "product_molecule",
                "pattern_reaction_product",
                "reactant_structure",
                "surrounding_structure",
            ):
                for class_id, count in split_report[target].items():
                    csv_writer.writerow(
                        {
                            "split": split,
                            "target": target,
                            "class_id": class_id,
                            "count": count,
                        }
                    )
                    writer.add_scalar(
                        f"target_count/{target}_by_class/{split}/{class_id}",
                        float(count),
                        0,
                    )
    return report


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
    for prefix in ("pattern", "reaction", "product_molecule", "reactant_structure", "surrounding_structure"):
        if name.startswith(prefix + "_"):
            metric = name.rsplit("_", 1)[-1]
            if metric in {"loss", "acc", "count"}:
                middle = name[len(prefix) + 1 : -(len(metric) + 1)]
                if middle.startswith("class_"):
                    chart = f"{prefix}_by_class"
                elif middle in {"pos", "neg"}:
                    chart = f"{prefix}_pos_neg"
                else:
                    chart = prefix if prefix != "observed_edge" else "edge"
                return f"{prefix}_{metric}/{chart}"
    return name


def tensorboard_series(name: str) -> str:
    for suffix in ("_loss", "_acc", "_count"):
        if name.endswith(suffix):
            rest = name[: -len(suffix)]
            for prefix in ("pattern", "reaction", "product_molecule", "reactant_structure", "surrounding_structure"):
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
            structure, selected_event_rows = move_structure_batch(batch, device)
            output = model(
                structure, selected_event_rows=selected_event_rows
            )
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
    early_stopping_patience: Optional[int] = None,
    early_stopping_window_size: int = 1,
    early_stopping_min_delta: float = DEFAULT_MIN_DELTA,
    preprocessing_cache: Optional[str | Path] = None,
    rebuild_preprocessing_cache: bool = False,
    min_data_count: int = 1000,
    mask_balance_patience: int = 20,
    mask_balance_max_forced_per_batch: int = 8,
) -> Dict[str, object]:
    device = torch.device(device)
    model.to(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tb_dir = output_path / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    (output_path / "tensorboard_command.txt").write_text(f"tensorboard --logdir {tb_dir} --port 6006 --host 0.0.0.0\n")

    cache_path = (
        Path(preprocessing_cache)
        if preprocessing_cache not in {None, ""}
        else output_path / "main_preprocessing_cache.pt"
    )
    preprocessing_payload, preprocessing_summary = load_or_build_preprocessing_cache(
        train_dir=train_dir,
        val_dir=val_dir,
        cache_path=cache_path,
        rebuild=rebuild_preprocessing_cache,
        surrounding_radius=model.surrounding_structure_radius,
    )
    write_preprocessing_reports(preprocessing_payload, output_path)

    train_probe = make_cleavage_structure_dataloader(
        train_dir, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_probe = make_cleavage_structure_dataloader(
        val_dir, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    train_sampler = BalancedCleavageBatchSampler(
        dataset_size=len(train_probe.dataset),
        batch_size=batch_size,
        index=preprocessing_payload["train"],
        min_data_count=min_data_count,
        patience=mask_balance_patience,
        max_forced_per_batch=mask_balance_max_forced_per_batch,
        shuffle=True,
    )
    val_sampler = BalancedCleavageBatchSampler(
        dataset_size=len(val_probe.dataset),
        batch_size=batch_size,
        index=preprocessing_payload["val"],
        min_data_count=min_data_count,
        patience=mask_balance_patience,
        max_forced_per_batch=mask_balance_max_forced_per_batch,
        shuffle=False,
    )
    train_loader = make_cleavage_structure_dataloader(
        train_dir, num_workers=num_workers, batch_sampler=train_sampler
    )
    val_loader = make_cleavage_structure_dataloader(
        val_dir, num_workers=num_workers, batch_sampler=val_sampler
    )
    target_report = write_target_report(
        output_path,
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        surrounding_radius=model.surrounding_structure_radius,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        window_size=early_stopping_window_size,
        min_delta=early_stopping_min_delta,
        mode="min",
    )

    metrics_file = output_path / "metrics.csv"
    best_val = float("inf")
    best_epoch = -1
    stopped_epoch = int(epochs)
    metric_records: List[Dict[str, float]] = []
    metric_fieldnames = ["epoch", "lr", "validation_seconds"]

    def write_epoch_record(
        csv_file,
        *,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        validation_seconds: float,
    ) -> None:
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "validation_seconds": validation_seconds,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        metric_records.append(record)
        for key in record:
            if key not in metric_fieldnames:
                metric_fieldnames.append(key)
        csv_file.seek(0)
        csv_file.truncate()
        csv_writer = csv.DictWriter(csv_file, fieldnames=metric_fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(metric_records)
        csv_file.flush()
        write_tensorboard_pairs(writer, train_metrics, val_metrics, epoch)
        writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("time/validation_seconds", validation_seconds, epoch)
        writer.flush()

    final_epoch = 0
    final_train_metrics: Dict[str, float] = {}
    final_val_metrics: Dict[str, float] = {}
    with metrics_file.open("w+", newline="", encoding="utf-8") as f:
        epoch0_train = run_epoch(model, train_loader, device=device, optimizer=None)
        start = time.perf_counter()
        epoch0_val = run_epoch(model, val_loader, device=device, optimizer=None)
        epoch0_seconds = time.perf_counter() - start
        write_epoch_record(
            f,
            epoch=0,
            train_metrics=epoch0_train,
            val_metrics=epoch0_val,
            validation_seconds=epoch0_seconds,
        )
        best_val = float(epoch0_val.get("loss", float("inf")))
        best_epoch = 0
        final_train_metrics = epoch0_train
        final_val_metrics = epoch0_val
        initial_ckpt = checkpoint_payload(
            model,
            epoch=0,
            train_metrics=epoch0_train,
            val_metrics=epoch0_val,
            extra={"best_val_loss": best_val, "best_epoch": best_epoch},
        )
        torch.save(initial_ckpt, output_path / "best.pt")
        early_stopping(best_val)

        for epoch in range(1, int(epochs) + 1):
            train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
            start = time.perf_counter()
            val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
            validation_seconds = time.perf_counter() - start
            write_epoch_record(
                f,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                validation_seconds=validation_seconds,
            )
            final_epoch = epoch
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics
            ckpt = checkpoint_payload(
                model,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                extra={"best_val_loss": best_val, "best_epoch": best_epoch},
            )
            torch.save(ckpt, output_path / "last.pt")
            val_loss = float(val_metrics.get("loss", float("inf")))
            if val_loss < best_val - float(early_stopping_min_delta):
                best_val = val_loss
                best_epoch = epoch
                ckpt["extra"]["best_val_loss"] = best_val
                ckpt["extra"]["best_epoch"] = best_epoch
                torch.save(ckpt, output_path / "best.pt")
            early_stopping(val_loss)
            if early_stopping.early_stop:
                stopped_epoch = epoch
                break

    summary: Dict[str, object] = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "stopped_epoch": stopped_epoch,
        "early_stopping_enabled": bool(
            early_stopping_patience is not None and early_stopping_patience > 0
        ),
        "early_stopping_counter": early_stopping.counter,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "target_report": target_report,
        "preprocessing_cache": preprocessing_summary,
        "balanced_sampling": {
            "min_data_count": int(min_data_count),
            "patience": int(mask_balance_patience),
            "max_forced_per_batch": int(mask_balance_max_forced_per_batch),
            "reactant_pos_neg_forced": True,
        },
    }
    final_ckpt = checkpoint_payload(
        model,
        epoch=final_epoch,
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
        extra=summary,
    )
    torch.save(final_ckpt, output_path / "last.pt")
    with (output_path / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    writer.close()
    return summary
