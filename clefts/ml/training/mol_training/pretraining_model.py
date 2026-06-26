from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from ...mol.mol_encoder import MolEncoder
from .feature_schema import (
    FeatureGroup,
    atom_feature_groups,
    bond_feature_groups,
    grouped_cross_entropy,
    grouped_soft_cross_entropy,
)
from .masking import edge_pair_repr, graph_mean_by_mask, make_prediction_masks, node_context_targets


def _group_decoders(in_dim: int, groups: Tuple[FeatureGroup, ...]) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            group.name: nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, group.dim),
            )
            for group in groups
        }
    )


@dataclass(frozen=True)
class MolPretrainingOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


class MolPretrainingModel(nn.Module):
    def __init__(
        self,
        *,
        mol_encoder: MolEncoder,
        descriptor_dim: int,
        use_node_attribute: bool = True,
        use_node_context: bool = True,
        use_edge_attribute: bool = True,
        use_graph_masked_attributes: bool = True,
        use_graph_descriptors: bool = True,
        node_loss_weight: float = 1.0,
        context_loss_weight: float = 0.5,
        edge_loss_weight: float = 1.0,
        graph_mask_loss_weight: float = 0.5,
        descriptor_loss_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.atom_groups = atom_feature_groups(mol_encoder.symbols)
        self.bond_groups = bond_feature_groups()
        self.descriptor_dim = int(descriptor_dim)

        self.use_node_attribute = use_node_attribute
        self.use_node_context = use_node_context
        self.use_edge_attribute = use_edge_attribute
        self.use_graph_masked_attributes = use_graph_masked_attributes
        self.use_graph_descriptors = use_graph_descriptors

        self.node_loss_weight = float(node_loss_weight)
        self.context_loss_weight = float(context_loss_weight)
        self.edge_loss_weight = float(edge_loss_weight)
        self.graph_mask_loss_weight = float(graph_mask_loss_weight)
        self.descriptor_loss_weight = float(descriptor_loss_weight)

        node_dim = mol_encoder.node_dim
        graph_dim = mol_encoder.graph_dim
        self.atom_decoders = _group_decoders(node_dim, self.atom_groups)
        self.edge_decoders = _group_decoders(node_dim * 3, self.bond_groups)
        self.context_decoder = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, self.atom_groups[0].dim),
        )
        self.graph_atom_decoders = _group_decoders(graph_dim, self.atom_groups)
        self.graph_edge_decoders = _group_decoders(graph_dim, self.bond_groups)
        self.descriptor_decoder = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, self.descriptor_dim),
        )

    def forward(
        self,
        batch: Batch,
        *,
        node_mask_ratio: float,
        edge_mask_ratio: float,
        graph_mask_ratio: float,
    ) -> MolPretrainingOutput:
        mask_info = make_prediction_masks(
            batch,
            node_mask_ratio=node_mask_ratio,
            edge_mask_ratio=edge_mask_ratio,
            graph_mask_ratio=graph_mask_ratio,
        )
        original_x = batch.x.clone()
        original_edge_attr = batch.edge_attr.clone()
        encoded = self.mol_encoder(batch)
        node_h = encoded.x
        graph_h = encoded.embeddings

        total = node_h.new_zeros(())
        metrics: Dict[str, float] = {}

        if self.use_node_attribute and bool(mask_info.node_mask.any()):
            logits = {
                name: head(node_h[mask_info.node_mask])
                for name, head in self.atom_decoders.items()
            }
            loss, group_metrics = grouped_cross_entropy(
                logits,
                original_x[mask_info.node_mask],
                self.atom_groups,
            )
            total = total + self.node_loss_weight * loss
            metrics["node_attr_loss"] = float(loss.detach().cpu())
            metrics.update({f"node_{k}": v for k, v in group_metrics.items()})

        if self.use_node_context and bool(mask_info.node_mask.any()):
            symbol_dim = self.atom_groups[0].dim
            target = node_context_targets(
                original_x,
                batch.edge_index,
                symbol_dim=symbol_dim,
                mask=mask_info.node_mask,
            )
            logits = self.context_decoder(node_h[mask_info.node_mask])
            loss = F.binary_cross_entropy_with_logits(logits, target)
            total = total + self.context_loss_weight * loss
            metrics["node_context_loss"] = float(loss.detach().cpu())

        if self.use_edge_attribute and bool(mask_info.edge_mask.any()) and batch.edge_index.numel() > 0:
            edge_repr = edge_pair_repr(node_h, batch.edge_index)[mask_info.edge_mask]
            logits = {name: head(edge_repr) for name, head in self.edge_decoders.items()}
            loss, group_metrics = grouped_cross_entropy(
                logits,
                original_edge_attr[mask_info.edge_mask],
                self.bond_groups,
            )
            total = total + self.edge_loss_weight * loss
            metrics["edge_attr_loss"] = float(loss.detach().cpu())
            metrics.update({f"edge_{k}": v for k, v in group_metrics.items()})

        if self.use_graph_masked_attributes:
            graph_node_target = graph_mean_by_mask(
                original_x,
                batch.batch,
                mask_info.graph_node_mask,
                int(batch.num_graphs),
            )
            graph_node_logits = {
                name: head(graph_h) for name, head in self.graph_atom_decoders.items()
            }
            node_loss, node_metrics = grouped_soft_cross_entropy(
                graph_node_logits,
                graph_node_target,
                self.atom_groups,
            )

            if batch.edge_index.numel() > 0:
                edge_graph = batch.batch[batch.edge_index[0]]
                graph_edge_target = graph_mean_by_mask(
                    original_edge_attr,
                    edge_graph,
                    mask_info.graph_edge_mask,
                    int(batch.num_graphs),
                )
                graph_edge_logits = {
                    name: head(graph_h) for name, head in self.graph_edge_decoders.items()
                }
                edge_loss, edge_metrics = grouped_soft_cross_entropy(
                    graph_edge_logits,
                    graph_edge_target,
                    self.bond_groups,
                )
            else:
                edge_loss = node_h.new_zeros(())
                edge_metrics = {}

            loss = node_loss + edge_loss
            total = total + self.graph_mask_loss_weight * loss
            metrics["graph_mask_node_loss"] = float(node_loss.detach().cpu())
            metrics["graph_mask_edge_loss"] = float(edge_loss.detach().cpu())
            metrics.update({f"graph_node_{k}": v for k, v in node_metrics.items()})
            metrics.update({f"graph_edge_{k}": v for k, v in edge_metrics.items()})

        if self.use_graph_descriptors:
            target = batch.descriptors.view(int(batch.num_graphs), self.descriptor_dim).to(graph_h.device)
            pred = self.descriptor_decoder(graph_h)
            loss = F.mse_loss(pred, target)
            total = total + self.descriptor_loss_weight * loss
            metrics["descriptor_loss"] = float(loss.detach().cpu())

        metrics["loss"] = float(total.detach().cpu())
        return MolPretrainingOutput(loss=total, metrics=metrics)
