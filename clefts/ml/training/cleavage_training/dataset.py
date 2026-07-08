from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ...input.fragment_tree_structure import FragmentTreeStructure
from ...input.fragment_tree_training_data import (
    FragmentTreeStructureFileDataset,
    collate_fragment_tree_structure_items,
)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class CleavageEdgeTargets:
    """Edge-level and atom-location targets derived from FragmentTreeStructure."""

    observed_edge: Tensor
    # [E] 1 when the edge is used by an assigned training pathway.

    pattern: Tensor
    # [E] cleavage pattern id, or IGNORE_INDEX when no event exists.

    reaction: Tensor
    # [E] reaction id, or IGNORE_INDEX when no event exists.

    product_molecule: Tensor
    # [E] product molecule id, or IGNORE_INDEX when no event exists.

    event_index: Tensor
    # [E] first cleavage event row for location targets, or IGNORE_INDEX.


def build_cleavage_edge_targets(structure: FragmentTreeStructure) -> CleavageEdgeTargets:
    """Create edge pretraining targets from a batched FragmentTreeStructure.

    The positive edge labels use ``target_edge_index`` when the input is a
    TrainingFragmentTreeStructure. These are the spectrum-assigned traversal
    edges produced by ``clefts.ml.input``. For non-training structures, the
    fallback is ``sample_edge_index`` so the same model can still run on saved
    candidate structures.
    """

    device = structure.edge_index.device
    num_edges = int(structure.num_edges)
    observed = torch.zeros((num_edges,), dtype=torch.float32, device=device)

    target_edge_index = getattr(structure, "target_edge_index", None)
    if target_edge_index is not None and target_edge_index.numel() > 0:
        observed[target_edge_index[1].long().unique()] = 1.0
    elif structure.sample_edge_index.numel() > 0:
        observed[structure.sample_edge_index[1].long().unique()] = 1.0

    pattern = torch.full((num_edges,), IGNORE_INDEX, dtype=torch.long, device=device)
    reaction = torch.full_like(pattern, IGNORE_INDEX)
    product_molecule = torch.full_like(pattern, IGNORE_INDEX)
    event_index = torch.full_like(pattern, IGNORE_INDEX)

    if structure.num_cleavage_events > 0:
        for event_row in range(int(structure.num_cleavage_events)):
            edge_idx = int(structure.cleavage_event_edge_index[event_row].item())
            if edge_idx < 0 or edge_idx >= num_edges:
                continue
            if int(event_index[edge_idx].item()) != IGNORE_INDEX:
                continue
            event = structure.cleavage_event[event_row]
            pattern[edge_idx] = int(event[0].item())
            reaction[edge_idx] = int(event[1].item())
            product_molecule[edge_idx] = int(event[2].item())
            event_index[edge_idx] = int(event_row)

    return CleavageEdgeTargets(
        observed_edge=observed,
        pattern=pattern,
        reaction=reaction,
        product_molecule=product_molecule,
        event_index=event_index,
    )


def make_cleavage_structure_dataloader(
    root_dir: str | Path,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pattern: str = "*.pt",
    map_location: str | torch.device = "cpu",
    device: Optional[str | torch.device] = None,
) -> DataLoader:
    dataset = FragmentTreeStructureFileDataset(
        root_dir,
        pattern=pattern,
        map_location=map_location,
        device=device,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fragment_tree_structure_items,
    )


def count_event_classes(structures: Sequence[FragmentTreeStructure]) -> Dict[str, int]:
    """Infer class counts from structures, useful for smoke tests and configs."""

    max_pattern = max_reaction = max_product = -1
    for structure in structures:
        if structure.num_cleavage_events == 0:
            continue
        event = structure.cleavage_event
        max_pattern = max(max_pattern, int(event[:, 0].max().item()))
        max_reaction = max(max_reaction, int(event[:, 1].max().item()))
        max_product = max(max_product, int(event[:, 2].max().item()))
    return {
        "num_patterns": max_pattern + 1,
        "num_reactions": max_reaction + 1,
        "num_product_molecules": max_product + 1,
    }
