from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from clefts.domain.fragment.cleavage import CleavagePatternSet
from clefts.ml.common.progress import (
    advance_edge_progress,
    set_edge_progress_phase,
    set_edge_progress_total,
)
from ....input.fragment_tree_features import FragmentTreeFeatures


@dataclass(frozen=True)
class FragmentEdgeEncoderOutput:
    edge_attr: Tensor
    absolute_score_logit: Tensor
    selected_edge_index: Tensor
    original_edge_count: int


class StructuralEdgeEncoder(nn.Module):
    """Condition-independent fragment-edge encoder.

    All edges receive attention-aware features and an independent score.
    ``max_edges_per_step`` only bounds a compute chunk and never prunes edges.
    """

    def __init__(
        self,
        *,
        cleavage_pattern_set_params: Dict,
        mol_dim: int,
        atom_dim: int,
        condition_dim: int,
        feature_dim: int = 256,
        category_dim: int = 32,
        num_heads: int = 8,
        attention_max_graph_distance: int = 4,
        max_edges_per_step: int = 128,
        max_edges_per_depth: Tuple[int, ...] = (128, 64, 32),
        training_edges_per_sample: int = 32,
        training_zero_edge_fraction: float = 0.25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads.")
        if attention_max_graph_distance < 1:
            raise ValueError("attention_max_graph_distance must be positive.")
        if max_edges_per_step < 1:
            raise ValueError("max_edges_per_step must be positive.")

        pattern_set = CleavagePatternSet.from_dict(cleavage_pattern_set_params)
        reaction_rows = []
        product_rows = []
        reactant_lengths: Dict[Tuple[int, int], int] = {}
        product_lengths: Dict[Tuple[int, int, int], int] = {}
        for pattern in pattern_set.patterns:
            pattern_id = int(pattern.pattern_id)
            for reaction in pattern.cleavage_reactions:
                reaction_id = int(reaction.id)
                global_reaction_id = len(reaction_rows)
                reaction_rows.append((pattern_id, reaction_id, global_reaction_id))
                reactant_lengths[(pattern_id, reaction_id)] = len(reaction.react_idx_to_map)
                for product_id, _ in enumerate(reaction.prod_idx_to_maps):
                    product_lengths[(pattern_id, reaction_id, int(product_id))] = len(
                        reaction.prod_idx_to_maps[product_id]
                    )
                    product_rows.append(
                        (pattern_id, reaction_id, int(product_id), len(product_rows))
                    )

        num_patterns = max((int(p.pattern_id) for p in pattern_set.patterns), default=-1) + 1
        self.pattern_embedding = nn.Embedding(max(num_patterns, 1), category_dim)
        self.reaction_embedding = nn.Embedding(max(len(reaction_rows), 1), category_dim)
        self.product_embedding = nn.Embedding(max(len(product_rows), 1), category_dim)
        self.register_buffer(
            "reaction_lookup_table",
            torch.tensor(reaction_rows, dtype=torch.long).reshape(-1, 3),
        )
        self.register_buffer(
            "product_lookup_table",
            torch.tensor(product_rows, dtype=torch.long).reshape(-1, 4),
        )
        self.reactant_lengths = reactant_lengths
        self.reactant_tuple_length_by_event_type = reactant_lengths
        self.product_tuple_length_by_event_type = product_lengths
        self.feature_dim = int(feature_dim)
        self.mol_dim = int(mol_dim)
        self.atom_dim = int(atom_dim)
        self.condition_dim = int(condition_dim)
        self.max_edges_per_step = int(max_edges_per_step)
        self.max_edges_per_depth = tuple(int(value) for value in max_edges_per_depth)
        self.training_edges_per_sample = int(training_edges_per_sample)
        self.training_zero_edge_fraction = float(training_zero_edge_fraction)
        if self.training_edges_per_sample < 1:
            raise ValueError("training_edges_per_sample must be positive.")
        if not 0.0 <= self.training_zero_edge_fraction <= 1.0:
            raise ValueError("training_zero_edge_fraction must be between 0 and 1.")
        if not self.max_edges_per_depth or min(self.max_edges_per_depth) < 1:
            raise ValueError("max_edges_per_depth values must be positive.")
        self.attention_max_graph_distance = int(attention_max_graph_distance)
        self.num_heads = int(num_heads)

        category_input_dim = category_dim * 3
        pair_dim = mol_dim * 5
        self.absolute_score_body = nn.Sequential(
            nn.LayerNorm(category_input_dim + pair_dim),
            nn.Linear(category_input_dim + pair_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.absolute_score_head = nn.Linear(feature_dim, 1)
        self.atom_projection = nn.Linear(atom_dim, feature_dim)
        self.center_projection = nn.Linear(atom_dim, feature_dim)
        # Kept (unused) so older state dictionaries remain loadable.  Edge
        # encoding no longer consumes this projection.
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim), nn.Linear(condition_dim, feature_dim)
        )
        self.slot_projection = nn.Sequential(
            nn.Linear(16, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim)
        )
        self.slot_relation_projection = nn.Sequential(
            nn.Linear(3, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim)
        )
        self.distance_bias_embedding = nn.Embedding(
            self.attention_max_graph_distance + 1, self.num_heads
        )
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(feature_dim)
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.attention_gate = nn.Parameter(torch.tensor(0.1))
        self.ffn_gate = nn.Parameter(torch.tensor(0.1))
        self.edge_output = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim))

    def config_dict(self) -> Dict[str, int | float | Dict]:
        return {
            "feature_dim": self.feature_dim,
            "mol_dim": self.mol_dim,
            "atom_dim": self.atom_dim,
            "condition_dim": self.condition_dim,
            "num_heads": self.num_heads,
            "attention_max_graph_distance": self.attention_max_graph_distance,
            "max_edges_per_step": self.max_edges_per_step,
            "max_edges_per_depth": self.max_edges_per_depth,
        }

    @staticmethod
    def _grow_embedding(embedding: nn.Embedding, size: int) -> nn.Embedding:
        if size <= embedding.num_embeddings:
            return embedding
        grown = nn.Embedding(size, embedding.embedding_dim).to(
            device=embedding.weight.device, dtype=embedding.weight.dtype
        )
        with torch.no_grad():
            grown.weight[: embedding.num_embeddings].copy_(embedding.weight)
            nn.init.normal_(grown.weight[embedding.num_embeddings :], std=0.02)
        return grown

    def extend_category_vocabulary(
        self,
        *,
        num_patterns: int,
        reaction_rows: Tensor,
        product_rows: Tensor,
    ) -> None:
        """Append fine-tuning IDs before constructing the optimizer.

        Rows are `[pattern, local_reaction, global_reaction]` and
        `[pattern, local_reaction, local_product, global_product]`.
        Existing lookup rows and global IDs must remain unchanged.
        """
        reaction_rows = reaction_rows.to(self.reaction_lookup_table.device).long()
        product_rows = product_rows.to(self.product_lookup_table.device).long()
        if reaction_rows.dim() != 2 or reaction_rows.size(1) != 3:
            raise ValueError("reaction_rows must have shape [N,3].")
        if product_rows.dim() != 2 or product_rows.size(1) != 4:
            raise ValueError("product_rows must have shape [N,4].")
        old_reactions = self.reaction_lookup_table
        old_products = self.product_lookup_table
        if old_reactions.numel() and not torch.equal(
            reaction_rows[: old_reactions.size(0)], old_reactions
        ):
            raise ValueError("Existing reaction lookup rows cannot be reordered.")
        if old_products.numel() and not torch.equal(
            product_rows[: old_products.size(0)], old_products
        ):
            raise ValueError("Existing product lookup rows cannot be reordered.")
        self.pattern_embedding = self._grow_embedding(self.pattern_embedding, num_patterns)
        reaction_size = int(reaction_rows[:, 2].max().item()) + 1 if reaction_rows.numel() else 1
        product_size = int(product_rows[:, 3].max().item()) + 1 if product_rows.numel() else 1
        self.reaction_embedding = self._grow_embedding(self.reaction_embedding, reaction_size)
        self.product_embedding = self._grow_embedding(self.product_embedding, product_size)
        self.reaction_lookup_table = reaction_rows
        self.product_lookup_table = product_rows

    def _global_ids(self, event: Tensor) -> Tuple[Tensor, Tensor]:
        pattern = event[:, 0].long()
        reaction = event[:, 1].long()
        product = event[:, 2].long()
        reaction_ids = torch.zeros_like(pattern)
        product_ids = torch.zeros_like(pattern)
        reaction_found = torch.zeros_like(pattern, dtype=torch.bool)
        product_found = torch.zeros_like(pattern, dtype=torch.bool)
        for row in self.reaction_lookup_table.tolist():
            mask = (pattern == row[0]) & (reaction == row[1])
            reaction_ids[mask] = int(row[2])
            reaction_found |= mask
        for row in self.product_lookup_table.tolist():
            mask = (pattern == row[0]) & (reaction == row[1]) & (product == row[2])
            product_ids[mask] = int(row[3])
            product_found |= mask
        if not reaction_found.all() or not product_found.all():
            missing = event[~(reaction_found & product_found), :3].detach().cpu().tolist()
            raise KeyError(f"Cleavage category lookup is missing event IDs: {missing[:8]}")
        return reaction_ids, product_ids

    def _event_category(self, event: Tensor) -> Tensor:
        reaction_ids, product_ids = self._global_ids(event)
        return torch.cat(
            (
                self.pattern_embedding(event[:, 0].long()),
                self.reaction_embedding(reaction_ids),
                self.product_embedding(product_ids),
            ),
            dim=-1,
        )

    @staticmethod
    def _pair_features(source: Tensor, target: Tensor) -> Tensor:
        return torch.cat(
            (source, target, target - source, torch.abs(target - source), source * target),
            dim=-1,
        )

    def encode_base(self, features: FragmentTreeFeatures) -> Tuple[Tensor, Tensor]:
        structure = features.structure
        event = structure.cleavage_event
        event_edge = structure.cleavage_event_edge_index.long()
        src = structure.edge_index[0, event_edge].long()
        dst = structure.edge_index[1, event_edge].long()
        event_h = self.absolute_score_body(
            torch.cat(
                (self._event_category(event), self._pair_features(features.mol_x[src], features.mol_x[dst])),
                dim=-1,
            )
        )
        event_logit = self.absolute_score_head(event_h).squeeze(-1)
        num_edges = int(structure.edge_index.size(1))
        edge_h = event_h.new_zeros((num_edges, self.feature_dim))
        edge_logit = event_logit.new_full((num_edges,), -torch.inf)
        for edge_id in event_edge.unique(sorted=True).tolist():
            mask = event_edge == int(edge_id)
            edge_h[int(edge_id)] = event_h[mask].mean(dim=0)
            edge_logit[int(edge_id)] = torch.logsumexp(event_logit[mask], dim=0)
        return edge_h, edge_logit

    @staticmethod
    def _slot_encoding(slot_count: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        position = torch.arange(slot_count, device=device, dtype=dtype).view(-1, 1)
        frequency = torch.pow(2.0, torch.arange(8, device=device, dtype=dtype)).view(1, -1)
        angle = position * frequency
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)

    def _distances(self, edge_index: Tensor, centers: Tensor, num_atoms: int) -> Tensor:
        distances = torch.full(
            (centers.numel(), num_atoms),
            self.attention_max_graph_distance + 1,
            dtype=torch.long,
            device=edge_index.device,
        )
        adjacency = [[] for _ in range(num_atoms)]
        for left, right in edge_index.detach().cpu().t().tolist():
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
        for slot, center in enumerate(centers.detach().cpu().tolist()):
            distances[slot, int(center)] = 0
            frontier = {int(center)}
            visited = set(frontier)
            for distance in range(1, self.attention_max_graph_distance + 1):
                nxt = {neighbor for atom in frontier for neighbor in adjacency[atom]} - visited
                if not nxt:
                    break
                index = torch.tensor(sorted(nxt), dtype=torch.long, device=edge_index.device)
                distances[slot, index] = distance
                visited.update(nxt)
                frontier = nxt
        return distances

    def _attend_event(self, features: FragmentTreeFeatures, event_index: int, base_h: Tensor) -> Tensor:
        structure = features.structure
        event = structure.cleavage_event[event_index]
        edge_id = int(structure.cleavage_event_edge_index[event_index].item())
        src_node = int(structure.edge_index[0, edge_id].item())
        pattern_id, reaction_id = int(event[0]), int(event[1])
        tuple_length = self.reactant_lengths[(pattern_id, reaction_id)]
        centers = structure.cleavage_atom_idxs[tuple_length][int(event[3])].long()
        start = int(structure.node_graph_offset[src_node].item())
        stop = int(structure.node_graph_offset[src_node + 1].item())
        atom_h = features.node_graphs.x[start:stop]
        global_edge = features.node_graphs.edge_index.long()
        mask = (
            (global_edge[0] >= start) & (global_edge[0] < stop)
            & (global_edge[1] >= start) & (global_edge[1] < stop)
        )
        local_edge = global_edge[:, mask] - start
        distance = self._distances(local_edge, centers, stop - start)
        allowed_atom = distance.min(dim=0).values <= self.attention_max_graph_distance
        center_distance = distance[:, centers]
        if centers.numel() > 1:
            other_mask = ~torch.eye(
                centers.numel(), dtype=torch.bool, device=centers.device
            )
            other = center_distance.masked_fill(~other_mask, self.attention_max_graph_distance + 1)
            nearest_other = other.min(dim=1).values.float()
            tied_nearest = (other == nearest_other.long().unsqueeze(1)).sum(dim=1).float()
        else:
            nearest_other = distance.new_zeros((1,), dtype=torch.float32)
            tied_nearest = distance.new_zeros((1,), dtype=torch.float32)
        relation = torch.stack(
            (
                nearest_other / max(self.attention_max_graph_distance, 1),
                tied_nearest / max(int(centers.numel()) - 1, 1),
                torch.arange(centers.numel(), device=centers.device).float()
                / max(int(centers.numel()) - 1, 1),
            ),
            dim=-1,
        ).to(atom_h.dtype)
        key_value = self.atom_projection(atom_h).unsqueeze(0)
        query = (
            base_h.view(1, 1, -1)
            + self.center_projection(atom_h[centers]).unsqueeze(0)
            + self.slot_projection(
                self._slot_encoding(
                    centers.numel(), device=atom_h.device, dtype=atom_h.dtype
                )
            ).unsqueeze(0)
            + self.slot_relation_projection(relation).unsqueeze(0)
        )
        head_bias = self.distance_bias_embedding(
            distance.clamp_max(self.attention_max_graph_distance)
        ).permute(2, 0, 1)
        attention_mask = head_bias
        attention_mask[:, :, ~allowed_atom] = -torch.inf
        attended, _ = self.cross_attention(
            self.attention_norm(query), key_value, key_value,
            attn_mask=attention_mask, need_weights=False,
        )
        query = query + self.attention_gate * attended
        query = query + self.ffn_gate * self.ffn(self.ffn_norm(query))
        return self.edge_output(query.mean(dim=1).squeeze(0))

    def _bounded_attend_event(
        self, features: FragmentTreeFeatures, event_index: int, base_h: Tensor
    ) -> Tensor:
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda value: self._attend_event(features, event_index, value),
                base_h,
                use_reentrant=False,
            )
        return self._attend_event(features, event_index, base_h)

    def condition_edges(
        self,
        features: FragmentTreeFeatures,
        edge_h: Tensor,
        edge_ids: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Compatibility shim returning shared structural representations.

        Conditions are intentionally ignored here.  They are applied only by
        :class:`ConditionEdgeScorer`, after every structural edge has been
        encoded once.
        """
        return edge_h[edge_ids]

    def _sample_training_edges(self, structure, edge_logit: Tensor) -> Tensor:
        """Choose influential positives, hard negatives, and zero-intensity exploration."""
        if not hasattr(structure, "target_edge_index"):
            return torch.arange(edge_logit.numel(), device=edge_logit.device)
        target_pairs = structure.target_edge_index.to(edge_logit.device).long()
        target_groups = structure.target_edge_group_index.to(edge_logit.device).long()
        formula_peak = structure.formula_peak_index.to(edge_logit.device).long()
        intensities = structure.sample_peak_intensity.to(edge_logit.device).float()
        sample_edges = structure.sample_edge_index.to(edge_logit.device).long()
        selected: set[int] = set()
        sample_order = torch.randperm(int(structure.num_samples)).tolist()
        for sample_id in sample_order:
            if len(selected) >= self.max_edges_per_step:
                break
            assigned = sample_edges[1, sample_edges[0] == sample_id]
            assigned = assigned[(assigned >= 0) & (assigned < edge_logit.numel())].unique()
            rows = (target_pairs[0] == sample_id).nonzero(as_tuple=False).flatten()
            groups = target_groups[rows].unique(sorted=True) if rows.numel() else target_groups[:0]
            positive_budget = max(
                1,
                self.training_edges_per_sample
                - int(round(self.training_edges_per_sample * self.training_zero_edge_fraction)),
            )
            sample_positive: set[int] = set()
            all_alternatives: set[int] = set()
            group_list = groups.tolist()
            if group_list:
                group_weight = torch.tensor([
                    float(intensities[formula_peak[int(group)]].clamp_min(0).item())
                    for group in group_list
                ], device=edge_logit.device)
                if float(group_weight.sum()) <= 0:
                    group_weight.fill_(1.0)
                chosen = torch.multinomial(
                    group_weight,
                    num_samples=min(positive_budget, len(group_list)),
                    replacement=False,
                ).tolist()
                sampled_groups = [group_list[index] for index in chosen]
            else:
                sampled_groups = []
            for group in sampled_groups:
                group_rows = rows[target_groups[rows] == int(group)]
                alternatives = target_pairs[1, group_rows]
                alternatives = alternatives[
                    (alternatives >= 0) & (alternatives < edge_logit.numel())
                ].unique()
                all_alternatives.update(int(value) for value in alternatives.tolist())
                if alternatives.numel():
                    probability = torch.softmax(edge_logit[alternatives].detach(), dim=0)
                    choice = int(alternatives[torch.multinomial(probability, 1)].item())
                    required_for_pair = {choice}
                    # Once a peak/formula pair has been sampled, compute only
                    # the ordered path edges needed by that comparison.  Do
                    # not admit another pair if doing so would exceed the
                    # configured expensive-edge budget.
                    if hasattr(structure, "terminal_path_ptr"):
                        assignment_formula = structure.assignment_formula_index.to(edge_logit.device).long()
                        path_ptr = structure.terminal_path_ptr.to(edge_logit.device).long()
                        path_edges = structure.target_path_edge_index.to(edge_logit.device).long()
                        for assignment_id in (assignment_formula == int(group)).nonzero(as_tuple=False).flatten().tolist():
                            start, end = int(path_ptr[assignment_id]), int(path_ptr[assignment_id + 1])
                            required_for_pair.update(int(value) for value in path_edges[start:end].tolist())
                    if (
                        len(sample_positive | required_for_pair) <= positive_budget
                        and len(selected | sample_positive | required_for_pair)
                        <= self.max_edges_per_step
                    ):
                        sample_positive.update(required_for_pair)
            selected.update(sample_positive)
            negative_budget = min(
                max(self.training_edges_per_sample - len(sample_positive), 0),
                max(self.max_edges_per_step - len(selected), 0),
            )
            negative = torch.tensor(
                [int(value) for value in assigned.tolist() if int(value) not in all_alternatives],
                dtype=torch.long,
                device=edge_logit.device,
            )
            if negative.numel() and negative_budget:
                hard_count = min((negative_budget + 1) // 2, int(negative.numel()))
                hard = negative[torch.topk(edge_logit[negative], hard_count).indices]
                remaining = negative[~torch.isin(negative, hard)]
                random_count = min(negative_budget - hard_count, int(remaining.numel()))
                random = (
                    remaining[torch.randperm(remaining.numel(), device=remaining.device)[:random_count]]
                    if random_count else remaining[:0]
                )
                selected.update(int(value) for value in torch.cat((hard, random)).tolist())
        if not selected:
            finite = torch.isfinite(edge_logit)
            count = min(
                self.training_edges_per_sample,
                self.max_edges_per_step,
                int(finite.sum().item()),
            )
            return torch.topk(edge_logit.masked_fill(~finite, -torch.inf), count).indices
        selected_tensor = torch.tensor(
            sorted(selected), dtype=torch.long, device=edge_logit.device
        )
        return selected_tensor

    def forward(
        self,
        features: FragmentTreeFeatures,
        *,
        selected_edge_index: Optional[Tensor] = None,
    ) -> FragmentEdgeEncoderOutput:
        set_edge_progress_phase("edge base logits")
        edge_h, edge_logit = self.encode_base(features)
        if selected_edge_index is not None:
            selected = selected_edge_index.to(edge_h.device).long().unique(sorted=True)
        elif self.training and torch.is_grad_enabled():
            set_edge_progress_phase("sampling training edges")
            selected = self._sample_training_edges(features.structure, edge_logit)
        else:
            selected = torch.arange(edge_h.size(0), device=edge_h.device)
        set_edge_progress_total(int(selected.numel()))
        set_edge_progress_phase("attention")
        selected_set = set(int(value) for value in selected.detach().cpu().tolist())
        event_edge = features.structure.cleavage_event_edge_index.long()
        updated_edge_h = []
        all_edges = torch.arange(edge_h.size(0), device=edge_h.device)
        for edge_chunk in all_edges.split(self.max_edges_per_step):
            for edge_id_tensor in edge_chunk:
                edge_id = int(edge_id_tensor.item())
                if edge_id not in selected_set:
                    updated_edge_h.append(edge_h[edge_id])
                    continue
                event_ids = (event_edge == int(edge_id)).nonzero(as_tuple=False).flatten()
                if event_ids.numel() == 0:
                    updated_edge_h.append(edge_h[edge_id])
                    advance_edge_progress()
                    continue
                base_h = edge_h[edge_id]
                attended = [
                    self._bounded_attend_event(features, int(event_id), base_h)
                    for event_id in event_ids.tolist()
                ]
                updated_edge_h.append(torch.stack(attended).mean(dim=0))
                advance_edge_progress()
        if updated_edge_h:
            edge_h = torch.stack(updated_edge_h, dim=0)
        return FragmentEdgeEncoderOutput(
            edge_attr=edge_h,
            absolute_score_logit=edge_logit,
            selected_edge_index=selected,
            original_edge_count=int(edge_h.size(0)),
        )


class ConditionEdgeScorer(nn.Module):
    """Lightweight ``base(edge) + dot(edge, condition)`` scorer."""

    def __init__(
        self,
        *,
        edge_dim: int,
        condition_dim: int,
        interaction_dim: int = 64,
    ) -> None:
        super().__init__()
        if interaction_dim < 1:
            raise ValueError("interaction_dim must be positive.")
        self.base_edge_head = nn.Sequential(nn.LayerNorm(edge_dim), nn.Linear(edge_dim, 1))
        self.edge_projection = nn.Linear(edge_dim, interaction_dim, bias=False)
        self.condition_projection = nn.Linear(condition_dim, interaction_dim, bias=False)
        self.scale = float(interaction_dim) ** -0.5

    def forward(
        self,
        shared_edge_h: Tensor,
        condition_h: Tensor,
        sample_edge_ids: Tensor,
        edge_sample_index: Tensor,
    ) -> Tensor:
        """Score only valid sample/tree-edge pairs without an ``[S,E,H]`` tensor."""
        edge_ids = sample_edge_ids.long()
        sample_ids = edge_sample_index.long()
        if edge_ids.numel() != sample_ids.numel():
            raise ValueError("sample_edge_ids and edge_sample_index must be aligned.")
        projected_edges = self.edge_projection(shared_edge_h)
        projected_conditions = self.condition_projection(condition_h)
        base = self.base_edge_head(shared_edge_h).squeeze(-1)
        interaction = (
            projected_edges[edge_ids] * projected_conditions[sample_ids]
        ).sum(dim=-1) * self.scale
        return base[edge_ids] + interaction


# Backward-compatible import/config name used by existing checkpoints.
ConditionedFragmentEdgeEncoder = StructuralEdgeEncoder
