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
)
from .masking import (
    edge_pair_repr,
    make_contrastive_view,
    make_prediction_masks,
    node_context_targets,
)


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
        use_graph_contrastive: bool = True,
        use_graph_descriptors: bool = True,
        node_loss_weight: float = 1.0,
        context_loss_weight: float = 0.5,
        edge_loss_weight: float = 1.0,
        graph_contrastive_loss_weight: float = 0.5,
        descriptor_loss_weight: float = 0.2,
        graph_contrastive_node_mask_ratio: float = 0.15,
        graph_contrastive_edge_drop_ratio: float = 0.15,
        graph_contrastive_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.atom_groups = atom_feature_groups(mol_encoder.symbols)
        self.bond_groups = bond_feature_groups()
        self.descriptor_dim = int(descriptor_dim)

        self.use_node_attribute = use_node_attribute
        self.use_node_context = use_node_context
        self.use_edge_attribute = use_edge_attribute
        self.use_graph_contrastive = use_graph_contrastive
        self.use_graph_descriptors = use_graph_descriptors

        self.node_loss_weight = float(node_loss_weight)
        self.context_loss_weight = float(context_loss_weight)
        self.edge_loss_weight = float(edge_loss_weight)
        self.graph_contrastive_loss_weight = float(graph_contrastive_loss_weight)
        self.descriptor_loss_weight = float(descriptor_loss_weight)
        self.graph_contrastive_node_mask_ratio = float(graph_contrastive_node_mask_ratio)
        self.graph_contrastive_edge_drop_ratio = float(graph_contrastive_edge_drop_ratio)
        self.graph_contrastive_temperature = float(graph_contrastive_temperature)

        node_dim = mol_encoder.node_dim
        graph_dim = mol_encoder.graph_dim
        self.atom_decoders = _group_decoders(node_dim, self.atom_groups)
        self.edge_decoders = _group_decoders(node_dim * 3, self.bond_groups)
        self.context_decoder = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, self.atom_groups[0].dim),
        )
        self.graph_contrastive_node_mask_token = nn.Parameter(torch.zeros(mol_encoder.atom_dim))
        self.graph_projection = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.ReLU(),
            nn.Linear(graph_dim, graph_dim),
        )
        self.descriptor_decoder = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, self.descriptor_dim),
        )


    def _graph_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.size(0) <= 1:
            return z1.sum() * 0.0
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.t()) / max(self.graph_contrastive_temperature, 1e-6)
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def _encode_graph_projection(self, batch: Batch) -> torch.Tensor:
        encoded = self.mol_encoder(batch)
        return self.graph_projection(encoded.embeddings)

    def forward(
        self,
        batch: Batch,
        *,
        node_mask_ratio: float,
        edge_mask_ratio: float,
    ) -> MolPretrainingOutput:
        mask_info = make_prediction_masks(
            batch,
            node_mask_ratio=node_mask_ratio,
            edge_mask_ratio=edge_mask_ratio,
        )
        original_x = batch.x.clone()
        original_edge_attr = batch.edge_attr.clone()
        original_batch = batch.clone()
        original_batch.x = original_x.clone()
        original_batch.edge_attr = original_edge_attr.clone()
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

        if self.use_graph_contrastive:
            view1 = make_contrastive_view(
                original_batch,
                node_mask_ratio=self.graph_contrastive_node_mask_ratio,
                edge_drop_ratio=self.graph_contrastive_edge_drop_ratio,
                node_mask_token=self.graph_contrastive_node_mask_token,
            )
            view2 = make_contrastive_view(
                original_batch,
                node_mask_ratio=self.graph_contrastive_node_mask_ratio,
                edge_drop_ratio=self.graph_contrastive_edge_drop_ratio,
                node_mask_token=self.graph_contrastive_node_mask_token,
            )
            z1 = self._encode_graph_projection(view1)
            z2 = self._encode_graph_projection(view2)
            loss = self._graph_contrastive_loss(z1, z2)
            total = total + self.graph_contrastive_loss_weight * loss
            metrics["graph_contrastive_loss"] = float(loss.detach().cpu())

        if self.use_graph_descriptors:
            target = batch.descriptors.view(int(batch.num_graphs), self.descriptor_dim).to(graph_h.device)
            pred = self.descriptor_decoder(graph_h)
            loss = F.mse_loss(pred, target)
            total = total + self.descriptor_loss_weight * loss
            metrics["descriptor_loss"] = float(loss.detach().cpu())

        metrics["loss"] = float(total.detach().cpu())
        return MolPretrainingOutput(loss=total, metrics=metrics)
