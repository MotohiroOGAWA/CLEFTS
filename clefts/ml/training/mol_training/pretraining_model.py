from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from ...mol.mol_encoder import MolEncoder
from .feature_schema import (
    FeatureGroup,
    atom_feature_groups,
    bond_feature_groups,
    grouped_cross_entropy,
)
from .masking import (
    FeatureMaskBalancer,
    edge_pair_repr,
    make_contrastive_view,
    make_prediction_masks,
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


class SimpleSubgraphGNN(nn.Module):
    def __init__(self, *, node_in_dim: int, edge_in_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim + edge_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(max(1, int(num_layers)))
        )
        self.update_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(max(1, int(num_layers)))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(max(1, int(num_layers))))

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.input_proj(batch.x)
        if batch.edge_index.numel() == 0:
            for update, norm in zip(self.update_layers, self.norms):
                h = norm(h + update(torch.cat([h, torch.zeros_like(h)], dim=-1)))
            return h

        src, dst = batch.edge_index
        for message, update, norm in zip(self.message_layers, self.update_layers, self.norms):
            msg = message(torch.cat([h[src], batch.edge_attr], dim=-1))
            agg = h.new_zeros(h.shape)
            agg.index_add_(0, dst, msg)
            degree = h.new_zeros((h.size(0), 1))
            degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
            agg = agg / degree.clamp_min(1.0)
            h = norm(h + update(torch.cat([h, agg], dim=-1)))
        return h


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
        descriptor_names: Sequence[str] | None = None,
        use_node_attribute: bool = True,
        use_node_context: bool = True,
        use_edge_attribute: bool = True,
        use_graph_contrastive: bool = True,
        use_graph_descriptors: bool = True,
        use_graph_ecfp: bool = True,
        node_loss_weight: float = 1.0,
        context_loss_weight: float = 0.5,
        edge_loss_weight: float = 1.0,
        graph_contrastive_loss_weight: float = 0.5,
        descriptor_loss_weight: float = 0.2,
        ecfp_loss_weight: float = 0.2,
        context_k: int = 2,
        context_r1: int = 1,
        context_r2: int = 4,
        ecfp_dim: int = 2048,
        graph_contrastive_node_mask_ratio: float = 0.15,
        graph_contrastive_edge_drop_ratio: float = 0.15,
        graph_contrastive_temperature: float = 0.2,
        balanced_attribute_masking: bool = True,
        mask_balance_patience: int = 20,
        mask_balance_max_forced_per_batch: int = 8,
        balanced_validation_masks: bool = True,
    ) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.atom_groups = atom_feature_groups(mol_encoder.symbols)
        self.bond_groups = bond_feature_groups()
        self.descriptor_dim = int(descriptor_dim)
        if descriptor_names is None:
            self.descriptor_names = tuple(f"descriptor_{idx}" for idx in range(self.descriptor_dim))
        else:
            self.descriptor_names = tuple(descriptor_names)
        if len(self.descriptor_names) != self.descriptor_dim:
            raise ValueError(
                f"descriptor_names length ({len(self.descriptor_names)}) must match "
                f"descriptor_dim ({self.descriptor_dim})."
            )

        self.use_node_attribute = use_node_attribute
        self.use_node_context = use_node_context
        self.use_edge_attribute = use_edge_attribute
        self.use_graph_contrastive = use_graph_contrastive
        self.use_graph_descriptors = use_graph_descriptors
        self.use_graph_ecfp = use_graph_ecfp

        self.node_loss_weight = float(node_loss_weight)
        self.context_loss_weight = float(context_loss_weight)
        self.edge_loss_weight = float(edge_loss_weight)
        self.graph_contrastive_loss_weight = float(graph_contrastive_loss_weight)
        self.descriptor_loss_weight = float(descriptor_loss_weight)
        self.ecfp_loss_weight = float(ecfp_loss_weight)
        self.context_k = int(max(1, context_k))
        self.context_r1 = int(max(0, context_r1))
        self.context_r2 = int(max(self.context_r1 + 1, context_r2))
        self.ecfp_dim = int(ecfp_dim)
        self.graph_contrastive_node_mask_ratio = float(graph_contrastive_node_mask_ratio)
        self.graph_contrastive_edge_drop_ratio = float(graph_contrastive_edge_drop_ratio)
        self.graph_contrastive_temperature = float(graph_contrastive_temperature)
        self.balanced_attribute_masking = bool(balanced_attribute_masking)
        self.balanced_validation_masks = bool(balanced_validation_masks)
        self.node_mask_balancer = (
            FeatureMaskBalancer(
                self.atom_groups,
                patience=mask_balance_patience,
                max_forced_per_batch=mask_balance_max_forced_per_batch,
            )
            if self.balanced_attribute_masking
            else None
        )
        self.edge_mask_balancer = (
            FeatureMaskBalancer(
                self.bond_groups,
                patience=mask_balance_patience,
                max_forced_per_batch=mask_balance_max_forced_per_batch,
            )
            if self.balanced_attribute_masking
            else None
        )

        node_dim = mol_encoder.node_dim
        graph_dim = mol_encoder.graph_dim
        self.atom_decoders = _group_decoders(node_dim, self.atom_groups)
        self.edge_decoders = _group_decoders(node_dim * 3, self.bond_groups)
        self.context_neighborhood_gnn = SimpleSubgraphGNN(
            node_in_dim=node_dim,
            edge_in_dim=mol_encoder.bond_dim,
            hidden_dim=node_dim,
            num_layers=self.context_k,
        )
        self.context_graph_gnn = SimpleSubgraphGNN(
            node_in_dim=node_dim,
            edge_in_dim=mol_encoder.bond_dim,
            hidden_dim=node_dim,
            num_layers=max(1, self.context_r2 - self.context_r1 + 1),
        )
        self.context_neighborhood_projection = nn.Linear(node_dim, graph_dim)
        self.context_graph_projection = nn.Linear(node_dim, graph_dim)
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
        self.ecfp_decoder = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, self.ecfp_dim),
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

    @staticmethod
    def _node_distances(edge_index: torch.Tensor, center: int, max_hops: int, num_nodes: int) -> Dict[int, int]:
        adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
        if edge_index.numel() > 0:
            src_values = edge_index[0].detach().cpu().tolist()
            dst_values = edge_index[1].detach().cpu().tolist()
            for src, dst in zip(src_values, dst_values):
                adjacency[int(src)].append(int(dst))
                adjacency[int(dst)].append(int(src))
        distances = {int(center): 0}
        frontier = [int(center)]
        for depth in range(1, int(max_hops) + 1):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor in distances:
                        continue
                    distances[neighbor] = depth
                    next_frontier.append(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
        return distances

    @staticmethod
    def _induced_subgraph(
        data: Data,
        node_indices: Sequence[int],
        *,
        center: int,
        anchor_nodes: Sequence[int],
    ) -> Data:
        device = data.x.device
        nodes = torch.tensor(sorted(set(int(index) for index in node_indices)), dtype=torch.long, device=device)
        if nodes.numel() == 0:
            nodes = torch.tensor([int(center)], dtype=torch.long, device=device)
        mapping = torch.full((data.x.size(0),), -1, dtype=torch.long, device=device)
        mapping[nodes] = torch.arange(nodes.numel(), device=device)
        if data.edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = data.edge_attr.new_empty((0, data.edge_attr.size(-1)))
        else:
            src, dst = data.edge_index
            keep = (mapping[src] >= 0) & (mapping[dst] >= 0)
            edge_index = mapping[data.edge_index[:, keep]]
            edge_attr = data.edge_attr[keep].clone()
        anchor_mask = torch.zeros(nodes.numel(), dtype=torch.bool, device=device)
        anchor_set = set(int(index) for index in anchor_nodes)
        for local_idx, global_idx in enumerate(nodes.detach().cpu().tolist()):
            if int(global_idx) in anchor_set:
                anchor_mask[local_idx] = True
        if not bool(anchor_mask.any()):
            center_local = int(mapping[int(center)].item()) if int(center) < mapping.numel() and mapping[int(center)] >= 0 else 0
            anchor_mask[center_local] = True
        center_pos = mapping[int(center)].view(1).clone() if int(center) < mapping.numel() and mapping[int(center)] >= 0 else torch.zeros((1,), dtype=torch.long, device=device)
        return Data(
            x=data.x[nodes].clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            center_pos=center_pos,
            anchor_mask=anchor_mask,
        )

    def _context_prediction_loss(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        neighborhoods = []
        contexts = []
        for data in batch.to_data_list():
            if data.x.numel() == 0:
                continue
            center = int(torch.randint(data.x.size(0), (1,), device=data.x.device).item())
            distances = self._node_distances(data.edge_index, center, self.context_r2, data.x.size(0))
            neighborhood_nodes = [node for node, distance in distances.items() if distance <= self.context_k]
            context_nodes = [node for node, distance in distances.items() if self.context_r1 <= distance <= self.context_r2]
            anchor_nodes = [
                node
                for node, distance in distances.items()
                if self.context_r1 <= distance <= self.context_k
            ]
            if not neighborhood_nodes or not context_nodes or not anchor_nodes:
                continue
            neighborhoods.append(
                self._induced_subgraph(
                    data,
                    neighborhood_nodes,
                    center=center,
                    anchor_nodes=[center],
                )
            )
            contexts.append(
                self._induced_subgraph(
                    data,
                    context_nodes,
                    center=anchor_nodes[0],
                    anchor_nodes=anchor_nodes,
                )
            )

        if len(neighborhoods) <= 1:
            return batch.x.sum() * 0.0, {"node_context_count": float(len(neighborhoods))}

        neighborhood_batch = Batch.from_data_list(neighborhoods).to(batch.x.device)
        context_batch = Batch.from_data_list(contexts).to(batch.x.device)
        neighborhood_h = self.context_neighborhood_gnn(neighborhood_batch)
        context_h = self.context_graph_gnn(context_batch)
        neighborhood_centers = neighborhood_batch.ptr[:-1] + neighborhood_batch.center_pos.view(-1)
        neighborhood_z = self.context_neighborhood_projection(neighborhood_h[neighborhood_centers])
        context_values = []
        for graph_idx in range(int(context_batch.num_graphs)):
            start = int(context_batch.ptr[graph_idx].item())
            stop = int(context_batch.ptr[graph_idx + 1].item())
            anchor_mask = context_batch.anchor_mask[start:stop]
            if not bool(anchor_mask.any()):
                anchor_mask = torch.ones((stop - start,), dtype=torch.bool, device=context_h.device)
            context_values.append(context_h[start:stop][anchor_mask].mean(dim=0))
        context_z = self.context_graph_projection(torch.stack(context_values, dim=0))
        neighborhood_z = F.normalize(neighborhood_z, dim=-1)
        context_z = F.normalize(context_z, dim=-1)
        logits = torch.matmul(neighborhood_z, context_z.t()) / max(self.graph_contrastive_temperature, 1e-6)
        positive_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        negative_mask = ~positive_mask
        targets = positive_mask.to(logits.dtype)
        per_pair_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        positive_loss = per_pair_loss[positive_mask].mean()
        negative_loss = per_pair_loss[negative_mask].mean() if bool(negative_mask.any()) else positive_loss * 0.0
        loss = 0.5 * (positive_loss + negative_loss) if bool(negative_mask.any()) else positive_loss
        pred_positive = torch.sigmoid(logits) >= 0.5
        positive_acc = pred_positive[positive_mask].float().mean()
        negative_acc = (~pred_positive[negative_mask]).float().mean() if bool(negative_mask.any()) else positive_acc.new_zeros(())
        acc = 0.5 * (positive_acc + negative_acc) if bool(negative_mask.any()) else positive_acc
        return loss, {
            "node_context_acc": float(acc.detach().cpu()),
            "node_context_count": float(logits.numel()),
            "node_context_pos_loss": float(positive_loss.detach().cpu()),
            "node_context_neg_loss": float(negative_loss.detach().cpu()),
            "node_context_pos_acc": float(positive_acc.detach().cpu()),
            "node_context_neg_acc": float(negative_acc.detach().cpu()),
            "node_context_pos_count": float(positive_mask.sum().detach().cpu()),
            "node_context_neg_count": float(negative_mask.sum().detach().cpu()),
        }

    @staticmethod
    def _balanced_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        per_bit = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pos_mask = target > 0.5
        neg_mask = ~pos_mask
        zero = per_bit.sum() * 0.0
        pos_loss = per_bit[pos_mask].mean() if bool(pos_mask.any()) else zero
        neg_loss = per_bit[neg_mask].mean() if bool(neg_mask.any()) else zero
        if bool(pos_mask.any()) and bool(neg_mask.any()):
            loss = 0.5 * (pos_loss + neg_loss)
        elif bool(pos_mask.any()):
            loss = pos_loss
        else:
            loss = neg_loss
        pred = torch.sigmoid(logits) >= 0.5
        metrics = {
            "ecfp_pos_loss": float(pos_loss.detach().cpu()),
            "ecfp_neg_loss": float(neg_loss.detach().cpu()),
            "ecfp_pos_count": float(pos_mask.sum().detach().cpu()),
            "ecfp_neg_count": float(neg_mask.sum().detach().cpu()),
            "ecfp_acc": float((pred == target.bool()).float().mean().detach().cpu()),
        }
        if bool(pos_mask.any()):
            metrics["ecfp_pos_acc"] = float((pred[pos_mask] == target[pos_mask].bool()).float().mean().detach().cpu())
        if bool(neg_mask.any()):
            metrics["ecfp_neg_acc"] = float((pred[neg_mask] == target[neg_mask].bool()).float().mean().detach().cpu())
        return loss, metrics

    def forward(
        self,
        batch: Batch,
        *,
        node_mask_ratio: float,
        edge_mask_ratio: float,
    ) -> MolPretrainingOutput:
        original_x = batch.x.clone()
        original_edge_attr = batch.edge_attr.clone()
        train_balancing = self.training and self.balanced_attribute_masking
        eval_coverage = (not self.training) and self.balanced_validation_masks
        mask_info = make_prediction_masks(
            batch,
            node_mask_ratio=node_mask_ratio,
            edge_mask_ratio=edge_mask_ratio,
            node_features=original_x,
            edge_features=original_edge_attr,
            node_groups=self.atom_groups,
            edge_groups=self.bond_groups,
            node_balancer=self.node_mask_balancer if train_balancing else None,
            edge_balancer=self.edge_mask_balancer if train_balancing else None,
            force_eval_coverage=eval_coverage,
        )
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

        if self.use_node_context:
            context_source_batch = original_batch.clone()
            context_source_batch.x = node_h
            context_source_batch.edge_attr = original_edge_attr.clone()
            loss, context_metrics = self._context_prediction_loss(context_source_batch)
            total = total + self.context_loss_weight * loss
            metrics["node_context_loss"] = float(loss.detach().cpu())
            metrics.update(context_metrics)

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
            per_descriptor_loss = F.mse_loss(pred, target, reduction="none").mean(dim=0)
            loss = per_descriptor_loss.mean()
            total = total + self.descriptor_loss_weight * loss
            metrics["descriptor_loss"] = float(loss.detach().cpu())
            descriptor_errors = (pred - target).detach()
            target_detached = target.detach()
            for idx, (name, item_loss) in enumerate(zip(self.descriptor_names, per_descriptor_loss)):
                item_target = target_detached[:, idx]
                item_sse = descriptor_errors[:, idx].pow(2).sum()
                metrics[f"descriptor_{name}_loss"] = float(item_loss.detach().cpu())
                metrics[f"descriptor_{name}_sse"] = float(item_sse.cpu())
                metrics[f"descriptor_{name}_target_sum"] = float(item_target.sum().cpu())
                metrics[f"descriptor_{name}_target_sq_sum"] = float(item_target.pow(2).sum().cpu())
                metrics[f"descriptor_{name}_count"] = float(item_target.numel())

        if self.use_graph_ecfp and hasattr(batch, "ecfp"):
            target = batch.ecfp.view(int(batch.num_graphs), self.ecfp_dim).to(graph_h.device)
            logits = self.ecfp_decoder(graph_h)
            loss, ecfp_metrics = self._balanced_bce_loss(logits, target)
            total = total + self.ecfp_loss_weight * loss
            metrics["ecfp_loss"] = float(loss.detach().cpu())
            metrics.update(ecfp_metrics)

        metrics["loss"] = float(total.detach().cpu())
        return MolPretrainingOutput(loss=total, metrics=metrics)
