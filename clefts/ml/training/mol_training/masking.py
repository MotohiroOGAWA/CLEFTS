from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.data import Batch


@dataclass(frozen=True)
class PredictionMaskInfo:
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    graph_node_mask: torch.Tensor
    graph_edge_mask: torch.Tensor


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
    graph_mask_ratio: float,
) -> PredictionMaskInfo:
    num_graphs = int(batch.num_graphs)
    node_mask = _sample_at_least_one_per_graph(batch.batch, num_graphs, node_mask_ratio)

    if batch.edge_index.numel() == 0:
        edge_graph = torch.empty((0,), dtype=torch.long, device=batch.x.device)
    else:
        edge_graph = batch.batch[batch.edge_index[0]]

    edge_mask = _sample_at_least_one_per_graph(edge_graph, num_graphs, edge_mask_ratio)
    graph_node_mask = _sample_at_least_one_per_graph(batch.batch, num_graphs, graph_mask_ratio)
    graph_edge_mask = _sample_at_least_one_per_graph(edge_graph, num_graphs, graph_mask_ratio)

    return PredictionMaskInfo(
        node_mask=node_mask,
        edge_mask=edge_mask,
        graph_node_mask=graph_node_mask,
        graph_edge_mask=graph_edge_mask,
    )

def edge_pair_repr(node_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    h_src = node_h[src]
    h_dst = node_h[dst]
    return torch.cat([h_src + h_dst, h_src * h_dst, torch.abs(h_src - h_dst)], dim=-1)


def graph_mean_by_mask(values: torch.Tensor, graph_ids: torch.Tensor, mask: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = values.new_zeros((num_graphs, values.size(-1)))
    counts = values.new_zeros((num_graphs, 1))
    if values.numel() == 0 or not bool(mask.any()):
        return out
    selected_values = values[mask]
    selected_graphs = graph_ids[mask]
    out.index_add_(0, selected_graphs, selected_values)
    counts.index_add_(0, selected_graphs, torch.ones((selected_values.size(0), 1), device=values.device, dtype=values.dtype))
    return out / counts.clamp_min(1.0)


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
