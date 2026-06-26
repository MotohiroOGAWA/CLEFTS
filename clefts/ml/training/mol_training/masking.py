from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch_geometric.data import Batch

from .feature_schema import FeatureGroup


@dataclass(frozen=True)
class PredictionMaskInfo:
    node_mask: torch.Tensor
    edge_mask: torch.Tensor


class FeatureMaskBalancer:
    def __init__(
        self,
        groups: Sequence[FeatureGroup],
        *,
        patience: int = 20,
        max_forced_per_batch: int = 8,
    ) -> None:
        self.groups = tuple(groups)
        self.patience = int(max(1, patience))
        self.max_forced_per_batch = int(max(0, max_forced_per_batch))
        self.steps_since_masked = {
            group.name: [self.patience for _ in range(group.dim)] for group in self.groups
        }

    def apply(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.max_forced_per_batch <= 0 or features.numel() == 0:
            self._update(features, mask)
            return mask

        out = mask.clone()
        forced = 0
        for group in self.groups:
            target_slice = features[:, group.start : group.stop]
            valid = target_slice.sum(dim=-1) > 0
            if not bool(valid.any()):
                continue
            target = target_slice.argmax(dim=-1)
            masked_target = target[out & valid]
            covered = set(masked_target.detach().cpu().tolist())
            candidates = sorted(
                range(group.dim),
                key=lambda class_idx: self.steps_since_masked[group.name][class_idx],
                reverse=True,
            )
            for class_idx in candidates:
                if forced >= self.max_forced_per_batch:
                    self._update(features, out)
                    return out
                if class_idx in covered:
                    continue
                if self.steps_since_masked[group.name][class_idx] < self.patience:
                    continue
                idx = ((target == class_idx) & valid).nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                selected = idx[torch.randint(idx.numel(), (1,), device=features.device)]
                out[selected] = True
                covered.add(class_idx)
                forced += 1

        self._update(features, out)
        return out

    def _update(self, features: torch.Tensor, mask: torch.Tensor) -> None:
        if features.numel() == 0:
            return
        for group in self.groups:
            target_slice = features[:, group.start : group.stop]
            valid = target_slice.sum(dim=-1) > 0
            if not bool(valid.any()):
                continue
            target = target_slice.argmax(dim=-1)
            masked_target = target[mask & valid]
            covered = set(masked_target.detach().cpu().tolist())
            for class_idx in range(group.dim):
                if class_idx in covered:
                    self.steps_since_masked[group.name][class_idx] = 0
                else:
                    self.steps_since_masked[group.name][class_idx] += 1


def force_feature_class_coverage(
    features: torch.Tensor,
    mask: torch.Tensor,
    groups: Sequence[FeatureGroup],
    *,
    max_forced_per_group: int = 32,
) -> torch.Tensor:
    if features.numel() == 0 or max_forced_per_group <= 0:
        return mask
    out = mask.clone()
    for group in groups:
        forced = 0
        target_slice = features[:, group.start : group.stop]
        valid = target_slice.sum(dim=-1) > 0
        if not bool(valid.any()):
            continue
        target = target_slice.argmax(dim=-1)
        masked_target = target[out & valid]
        covered = set(masked_target.detach().cpu().tolist())
        present = sorted(set(target[valid].detach().cpu().tolist()))
        for class_idx in present:
            if forced >= max_forced_per_group:
                break
            if class_idx in covered:
                continue
            idx = ((target == class_idx) & valid).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            selected = idx[torch.randint(idx.numel(), (1,), device=features.device)]
            out[selected] = True
            forced += 1
    return out


def _sample_at_least_one_per_graph(
    graph_ids: torch.Tensor,
    num_graphs: int,
    ratio: float,
) -> torch.Tensor:
    device = graph_ids.device
    selected = torch.zeros(graph_ids.size(0), dtype=torch.bool, device=device)
    ratio = float(max(0.0, min(1.0, ratio)))
    if graph_ids.numel() == 0 or ratio <= 0.0:
        return selected

    rand = torch.rand(graph_ids.size(0), device=device)
    selected = rand < ratio
    for graph_id in range(num_graphs):
        idx = (graph_ids == graph_id).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        if not bool(selected[idx].any()):
            selected[idx[torch.randint(idx.numel(), (1,), device=device)]] = True
    return selected


def make_prediction_masks(
    batch: Batch,
    *,
    node_mask_ratio: float,
    edge_mask_ratio: float,
    node_features: torch.Tensor | None = None,
    edge_features: torch.Tensor | None = None,
    node_groups: Sequence[FeatureGroup] = (),
    edge_groups: Sequence[FeatureGroup] = (),
    node_balancer: FeatureMaskBalancer | None = None,
    edge_balancer: FeatureMaskBalancer | None = None,
    force_eval_coverage: bool = False,
) -> PredictionMaskInfo:
    num_graphs = int(batch.num_graphs)
    node_mask = _sample_at_least_one_per_graph(batch.batch, num_graphs, node_mask_ratio)

    if batch.edge_index.numel() == 0:
        edge_graph = torch.empty((0,), dtype=torch.long, device=batch.x.device)
    else:
        edge_graph = batch.batch[batch.edge_index[0]]

    edge_mask = _sample_at_least_one_per_graph(edge_graph, num_graphs, edge_mask_ratio)

    if node_features is not None:
        if node_balancer is not None:
            node_mask = node_balancer.apply(node_features, node_mask)
        elif force_eval_coverage:
            node_mask = force_feature_class_coverage(node_features, node_mask, node_groups)
    if edge_features is not None:
        if edge_balancer is not None:
            edge_mask = edge_balancer.apply(edge_features, edge_mask)
        elif force_eval_coverage:
            edge_mask = force_feature_class_coverage(edge_features, edge_mask, edge_groups)

    return PredictionMaskInfo(
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

def edge_pair_repr(node_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    h_src = node_h[src]
    h_dst = node_h[dst]
    return torch.cat([h_src + h_dst, h_src * h_dst, torch.abs(h_src - h_dst)], dim=-1)


def node_context_targets(
    original_x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    symbol_dim: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = original_x.new_zeros((original_x.size(0), symbol_dim))
    if edge_index.numel() == 0:
        return targets
    src, dst = edge_index
    symbols = original_x[:, :symbol_dim]
    targets.index_add_(0, src, symbols[dst])
    targets = (targets > 0).to(original_x.dtype)
    if mask is not None:
        targets = targets[mask]
    return targets


def _drop_undirected_edge_pairs(batch: Batch, drop_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    if edge_index.numel() == 0 or drop_ratio <= 0.0:
        return edge_index, edge_attr

    device = edge_index.device
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    pair_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx, (u, v) in enumerate(zip(src, dst)):
        key = (u, v) if u <= v else (v, u)
        pair_to_indices.setdefault(key, []).append(idx)

    keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
    for indices in pair_to_indices.values():
        if torch.rand((), device=device).item() < drop_ratio:
            keep[torch.tensor(indices, dtype=torch.long, device=device)] = False

    return edge_index[:, keep], edge_attr[keep]


def make_contrastive_view(
    batch: Batch,
    *,
    node_mask_ratio: float,
    edge_drop_ratio: float,
    node_mask_token: torch.Tensor,
) -> Batch:
    graphs = batch.to_data_list()
    out = []
    for data in graphs:
        view = data.clone()
        node_graph_ids = torch.zeros(view.x.size(0), dtype=torch.long, device=view.x.device)
        node_mask = _sample_at_least_one_per_graph(
            node_graph_ids,
            1,
            node_mask_ratio,
        )
        if bool(node_mask.any()):
            view.x[node_mask] = node_mask_token.to(view.x.device, view.x.dtype)
        view.edge_index, view.edge_attr = _drop_undirected_edge_pairs(
            view,
            float(max(0.0, min(1.0, edge_drop_ratio))),
        )
        out.append(view)
    return Batch.from_data_list(out)

