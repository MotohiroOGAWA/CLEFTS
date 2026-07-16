from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from ...input.fragment_tree_training_data import FragmentTreeStructureFileDataset
from .dataset import first_stage_edge_mask, resolve_structure_data_dir
from .pretraining_model import canonical_fragment_smiles


PREPROCESSING_CACHE_VERSION = 2


def _file_manifest(root_dir: str | Path) -> List[Dict[str, object]]:
    data_dir = resolve_structure_data_dir(root_dir)
    return [
        {
            "path": str(path.resolve()),
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in sorted(data_dir.glob("*.pt"))
    ]


def preprocessing_manifest(
    train_dir: str | Path, val_dir: str | Path, *, surrounding_radius: int
) -> Dict[str, object]:
    return {
        "version": PREPROCESSING_CACHE_VERSION,
        "surrounding_radius": int(surrounding_radius),
        "train_files": _file_manifest(train_dir),
        "val_files": _file_manifest(val_dir),
    }


def _tuple_length(
    table: torch.Tensor, values: Tuple[int, ...], length_column: int
) -> int | None:
    table = table.long()
    mask = torch.ones((table.size(0),), dtype=torch.bool)
    for column, value in enumerate(values):
        mask &= table[:, column].cpu() == int(value)
    if not bool(mask.any()):
        return None
    return int(table[mask][0, length_column].item())


def scan_dataset(dataset: FragmentTreeStructureFileDataset, *, split: str) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    files_by_target: Dict[str, set[int]] = defaultdict(set)
    events_by_target: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    identity_files: Dict[str, set[int]] = defaultdict(set)
    identity_events: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    identity_occurrences: Counter[str] = Counter()

    for file_index, item in enumerate(
        tqdm(dataset, desc=f"Preprocessing {split} cleavage events", leave=False)
    ):
        structure = item.structure
        stage_mask = first_stage_edge_mask(structure)
        for event_index, event_tensor in enumerate(structure.cleavage_event):
            edge_index = int(structure.cleavage_event_edge_index[event_index].item())
            if edge_index < 0 or edge_index >= int(structure.num_edges):
                continue
            if not bool(stage_mask[edge_index]):
                continue
            event = [int(value) for value in event_tensor.tolist()]
            pattern_id, reaction_id, product_id = event[:3]
            src_node = int(structure.edge_index[0, edge_index].item())
            reactant_length = _tuple_length(
                structure.reactant_tuple_length_table,
                (pattern_id, reaction_id),
                2,
            )
            identity = None
            reactant_atom_values: List[int] = []
            if reactant_length is not None:
                reactant_atoms = structure.cleavage_atom_idxs[reactant_length][
                    int(event[3])
                ].long()
                reactant_atom_values = [
                    int(value) for value in reactant_atoms.tolist()
                ]
                identity = canonical_fragment_smiles(
                    str(structure.node_smiles[src_node]),
                    tuple(reactant_atom_values),
                )

            target_keys = (
                f"pattern:{pattern_id}",
                f"reaction:{reaction_id}",
                f"product_molecule:{product_id}",
                f"pattern_reaction_product:{pattern_id}/{reaction_id}/{product_id}",
            )
            for key in target_keys:
                counts[key] += 1
                files_by_target[key].add(file_index)
                events_by_target[key].append((file_index, event_index))
            if identity is not None:
                identity_key = f"reactant_structure:{identity}"
                counts[identity_key] += 1
                files_by_target[identity_key].add(file_index)
                identity_files[identity].add(file_index)
                identity_events[identity].append((file_index, event_index))
                identity_occurrences[identity] += 1

            records.append(
                {
                    "split": split,
                    "file_index": file_index,
                    "file": str(item.path),
                    "edge_index": edge_index,
                    "event_index": event_index,
                    "source_node": src_node,
                    "pattern": pattern_id,
                    "reaction": reaction_id,
                    "product_molecule": product_id,
                    "reactant_structure": identity,
                    "reactant_atoms": reactant_atom_values,
                }
            )

    return {
        "records": records,
        "counts": dict(sorted(counts.items())),
        "files_by_target": {
            key: sorted(value) for key, value in sorted(files_by_target.items())
        },
        "events_by_target": {
            key: sorted(value) for key, value in sorted(events_by_target.items())
        },
        "identity_files": {
            key: sorted(value) for key, value in sorted(identity_files.items())
        },
        "identity_occurrences": dict(sorted(identity_occurrences.items())),
        "identity_events": {
            key: sorted(value) for key, value in sorted(identity_events.items())
        },
        "num_files": len(dataset),
        "num_events": len(records),
    }


def load_or_build_preprocessing_cache(
    *,
    train_dir: str | Path,
    val_dir: str | Path,
    cache_path: str | Path,
    rebuild: bool,
    surrounding_radius: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    path = Path(cache_path)
    manifest = preprocessing_manifest(
        train_dir, val_dir, surrounding_radius=surrounding_radius
    )
    if path.exists() and not rebuild:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("manifest") == manifest:
            return payload, {
                "path": str(path),
                "hit": True,
                "version": PREPROCESSING_CACHE_VERSION,
            }

    train_dataset = FragmentTreeStructureFileDataset(
        resolve_structure_data_dir(train_dir)
    )
    val_dataset = FragmentTreeStructureFileDataset(resolve_structure_data_dir(val_dir))
    payload = {
        "manifest": manifest,
        "train": scan_dataset(train_dataset, split="train"),
        "val": scan_dataset(val_dataset, split="val"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload, {
        "path": str(path),
        "hit": False,
        "rebuilt": bool(rebuild),
        "version": PREPROCESSING_CACHE_VERSION,
    }


def write_preprocessing_reports(payload: Mapping[str, object], output_dir: Path) -> None:
    summary = {
        split: {
            "num_files": payload[split]["num_files"],
            "num_events": payload[split]["num_events"],
            "counts": payload[split]["counts"],
            "identity_occurrences": payload[split]["identity_occurrences"],
        }
        for split in ("train", "val")
    }
    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    fieldnames = [
        "split", "file_index", "file", "edge_index", "event_index", "source_node",
        "pattern", "reaction", "product_molecule", "reactant_structure",
        "reactant_atoms",
    ]
    with (output_dir / "preprocessing_event_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val"):
            writer.writerows(payload[split]["records"])


class BalancedCleavageBatchSampler(Sampler[List[int]]):
    """File sampler backed by the preprocessing event-to-file index.

    Rare targets are inserted after ``patience`` absent batches.  In addition,
    every batch is expanded, when the split permits it, so reactant-structure
    pairs contain both a positive and a negative target.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        index: Mapping[str, object],
        min_data_count: int,
        patience: int,
        max_forced_per_batch: int,
        shuffle: bool,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.patience = max(1, int(patience))
        self.max_forced = max(0, int(max_forced_per_batch))
        counts = index["counts"]
        self.events_by_target = {
            key: [tuple(value) for value in events]
            for key, events in index["events_by_target"].items()
            if int(counts[key]) <= int(min_data_count) and events
        }
        self.steps = {key: self.patience for key in self.events_by_target}
        self.identity_events = {
            key: [tuple(value) for value in events]
            for key, events in index["identity_events"].items()
            if events
        }
        self.rng = random.Random()

    def __len__(self) -> int:
        return int(math.ceil(self.dataset_size / self.batch_size))

    def __iter__(self) -> Iterator[List[int | Tuple[int, int]]]:
        order = (
            torch.randperm(self.dataset_size).tolist()
            if self.shuffle
            else list(range(self.dataset_size))
        )
        for start in range(0, self.dataset_size, self.batch_size):
            batch = [int(i) for i in order[start : start + self.batch_size]]
            self._force_reactant_pos_neg(batch)
            self._force_rare_targets(batch)
            self._update_steps(batch)
            yield batch

    def _force_reactant_pos_neg(self, batch: List[int | Tuple[int, int]]) -> None:
        identities = sorted(self.identity_events)
        if len(identities) < 2:
            return
        positive_identity = self.rng.choice(identities)
        negative_identity = self.rng.choice(
            [identity for identity in identities if identity != positive_identity]
        )
        positive_event = self.rng.choice(self.identity_events[positive_identity])
        negative_event = self.rng.choice(self.identity_events[negative_identity])
        # A forced tuple means: load this file, but include only this local
        # cleavage-event row in the next forward/loss calculation.
        batch.extend([positive_event, positive_event, negative_event])

    def _force_rare_targets(self, batch: List[int | Tuple[int, int]]) -> None:
        forced = 0
        selected_events = {
            tuple(value) for value in batch if isinstance(value, tuple)
        }
        base_files = {int(value) for value in batch if not isinstance(value, tuple)}
        for key in sorted(self.events_by_target, key=lambda item: self.steps[item], reverse=True):
            if forced >= self.max_forced:
                break
            events = self.events_by_target[key]
            if self.steps[key] < self.patience:
                continue
            if any(
                file_index in base_files or (file_index, event_index) in selected_events
                for file_index, event_index in events
            ):
                continue
            selected = self.rng.choice(events)
            batch.append(selected)
            selected_events.add(selected)
            forced += 1

    def _update_steps(self, batch: Sequence[int | Tuple[int, int]]) -> None:
        selected_events = {
            tuple(value) for value in batch if isinstance(value, tuple)
        }
        base_files = {int(value) for value in batch if not isinstance(value, tuple)}
        for key, events in self.events_by_target.items():
            present = any(
                file_index in base_files or (file_index, event_index) in selected_events
                for file_index, event_index in events
            )
            self.steps[key] = 0 if present else self.steps[key] + 1
