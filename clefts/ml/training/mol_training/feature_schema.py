from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from ...mol.atom_feature import AtomFeatureLayer
from ...mol.bond_feature import BondFeatureLayer


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    start: int
    stop: int

    @property
    def dim(self) -> int:
        return self.stop - self.start


def _groups_from_feature_sets(feature_sets: Dict[str, Tuple[object, ...]]) -> Tuple[FeatureGroup, ...]:
    groups: List[FeatureGroup] = []
    offset = 0
    for name, values in feature_sets.items():
        stop = offset + len(values)
        groups.append(FeatureGroup(name=name, start=offset, stop=stop))
        offset = stop
    return tuple(groups)


def atom_feature_groups(symbols: Iterable[str]) -> Tuple[FeatureGroup, ...]:
    layer = AtomFeatureLayer(symbols=tuple(symbols))
    return _groups_from_feature_sets(layer.feature_sets)


def bond_feature_groups() -> Tuple[FeatureGroup, ...]:
    layer = BondFeatureLayer()
    return _groups_from_feature_sets(layer.feature_sets)


def grouped_cross_entropy(
    logits_by_group: Dict[str, torch.Tensor],
    target_features: torch.Tensor,
    groups: Tuple[FeatureGroup, ...],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if target_features.numel() == 0:
        device = next(iter(logits_by_group.values())).device
        return torch.zeros((), device=device), {}

    losses = []
    metrics: Dict[str, float] = {}
    for group in groups:
        target_slice = target_features[:, group.start : group.stop]
        valid = target_slice.sum(dim=-1) > 0
        if not bool(valid.any()):
            continue
        target = target_slice[valid].argmax(dim=-1)
        logits = logits_by_group[group.name][valid]
        loss = F.cross_entropy(logits, target)
        losses.append(loss)
        metrics[f"{group.name}_loss"] = float(loss.detach().cpu())
        pred = logits.argmax(dim=-1)
        metrics[f"{group.name}_acc"] = float((pred == target).float().mean().detach().cpu())

    if not losses:
        device = target_features.device
        return torch.zeros((), device=device), metrics
    return torch.stack(losses).mean(), metrics


def grouped_soft_cross_entropy(
    logits_by_group: Dict[str, torch.Tensor],
    target_features: torch.Tensor,
    groups: Tuple[FeatureGroup, ...],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if target_features.numel() == 0:
        device = next(iter(logits_by_group.values())).device
        return torch.zeros((), device=device), {}

    losses = []
    metrics: Dict[str, float] = {}
    for group in groups:
        target_slice = target_features[:, group.start : group.stop]
        denom = target_slice.sum(dim=-1, keepdim=True)
        valid = denom.squeeze(-1) > 0
        if not bool(valid.any()):
            continue

        target = target_slice[valid] / denom[valid].clamp_min(1e-12)
        logits = logits_by_group[group.name][valid]
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(target * log_prob).sum(dim=-1).mean()
        losses.append(loss)
        metrics[f"{group.name}_loss"] = float(loss.detach().cpu())
        pred = logits.argmax(dim=-1)
        target_mode = target.argmax(dim=-1)
        metrics[f"{group.name}_acc"] = float((pred == target_mode).float().mean().detach().cpu())

    if not losses:
        device = target_features.device
        return torch.zeros((), device=device), metrics
    return torch.stack(losses).mean(), metrics
