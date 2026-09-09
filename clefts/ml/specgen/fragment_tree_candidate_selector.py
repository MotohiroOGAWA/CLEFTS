from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ...libs.mmkit.mmkit import Formula
from ..common.progress import set_edge_progress_phase
from ..input.fragment_tree_features import FragmentTreeFeatures
from ..input.fragment_tree_structure import FragmentTreeStructure
from .fragment_tree_feature_model import FragmentTreeFeatureModel
from .components.fragment_edge import ConditionEdgeScorer


@dataclass(frozen=True)
class FragmentIonCandidate:
    sample_id: int
    batch_node_index: int
    global_node_id: int
    ion_index: int
    unsaturation_index: int
    radical_index: int
    formula: Formula
    formula_tensor: Tensor
    score: float
    keep_logit: float
    candidate_logit: float
    probability: float
    score_tensor: Optional[Tensor] = None
    edge_probability: float = 1.0
    adduct_probability: float = 1.0
    probability_tensor: Optional[Tensor] = None


@dataclass(frozen=True)
class NextCleavageCandidate:
    sample_id: int
    batch_node_index: int
    global_node_id: int
    score: float
    probability: float


@dataclass(frozen=True)
class FragmentTreeCandidateSelectionOutput:
    features: FragmentTreeFeatures
    sample_tree_batch: object
    keep_logit: Tensor
    cleave_logit: Tensor
    edge_absolute_logit: Tensor
    edge_cleave_logit: Tensor
    ion_logit: Tensor
    unsaturation_logit: Tensor
    radical_logit: Tensor
    ion_valid_mask_by_role_adduct: Tensor
    unsaturation_valid_mask_by_role_adduct: Tensor
    radical_valid_mask_by_role_adduct: Tensor
    kept_candidates: List[FragmentIonCandidate]
    next_cleavage_candidates: List[NextCleavageCandidate]
    absolute_score_logit: Optional[Tensor] = None
    selected_edge_index: Optional[Tensor] = None



class FragmentTreeCandidateSelector(nn.Module):
    """Select fragments, ion states, and next cleavage candidates directly."""

    # Large enough to dominate ordinary condition-scored logits, so a
    # target/positive edge is never dropped by the per-tree budget in favor
    # of a merely high-scoring non-target edge.
    _TARGET_EDGE_IMPORTANCE_BONUS = 1e6
    _ION_CANDIDATE_SCORE_BLOCK_ELEMENTS = 262144

    def __init__(
        self,
        feature_model: FragmentTreeFeatureModel,
        *,
        max_fragment_ion_candidates: int = 30,
        max_next_cleavage_candidates: int = 3,
        max_edges_per_step: Optional[int] = 128,
        max_retained_edges: Optional[int] = 30,
        max_edges_per_tree: Optional[int] = 256,
        max_nodes_for_ion_candidates: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        edge_condition_interaction_dim: int = 64,
        ranking_loss_weight: float = 1.0,
        top_n: int = 10,
        nearest_lower_partners: int = 1,
        extended_lower_partners: int = 3,
        background_partners: int = 10,
        ranking_intensity_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        if max_fragment_ion_candidates <= 0:
            raise ValueError("max_fragment_ion_candidates must be positive.")
        if max_next_cleavage_candidates <= 0:
            raise ValueError("max_next_cleavage_candidates must be positive.")
        if max_edges_per_step is not None and max_edges_per_step <= 0:
            raise ValueError("max_edges_per_step must be positive or None.")
        if max_retained_edges is not None and max_retained_edges <= 0:
            raise ValueError("max_retained_edges must be positive or None.")
        if max_edges_per_tree is not None and max_edges_per_tree <= 0:
            raise ValueError("max_edges_per_tree must be positive or None.")
        if max_nodes_for_ion_candidates is not None and max_nodes_for_ion_candidates <= 0:
            raise ValueError("max_nodes_for_ion_candidates must be positive or None.")
        self.feature_model = feature_model
        self.mol_encoder = feature_model.mol_encoder
        self.fragmenter = feature_model.fragmenter
        self.fragment_edge_encoder = feature_model.fragment_edge_encoder
        self.max_fragment_ion_candidates = int(max_fragment_ion_candidates)
        self.max_next_cleavage_candidates = int(max_next_cleavage_candidates)
        self.max_edges_per_step = max_edges_per_step
        self.max_retained_edges = max_retained_edges
        self.max_edges_per_tree = max_edges_per_tree
        self.max_nodes_for_ion_candidates = max_nodes_for_ion_candidates
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.top_n = int(top_n)
        self.nearest_lower_partners = int(nearest_lower_partners)
        self.extended_lower_partners = int(extended_lower_partners)
        self.background_partners = int(background_partners)
        self.ranking_intensity_threshold = float(ranking_intensity_threshold)
        if self.ranking_loss_weight < 0:
            raise ValueError("ranking_loss_weight must be non-negative.")
        if self.top_n < 1:
            raise ValueError("top_n must be positive.")
        if self.nearest_lower_partners < 0 or self.extended_lower_partners < 0 or self.background_partners < 0:
            raise ValueError("nearest_lower_partners, extended_lower_partners, and background_partners must be non-negative.")
        if self.ranking_intensity_threshold < 0:
            raise ValueError("ranking_intensity_threshold must be non-negative.")
        tree_dim = int(feature_model.tree_encoder.hidden_dim)
        hidden_dim = int(hidden_dim or tree_dim)
        self.node_keep_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.node_cleave_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.edge_cleave_head = nn.Sequential(
            nn.Linear(tree_dim * 2 + feature_model.fragment_edge_encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.condition_edge_scorer = ConditionEdgeScorer(
            edge_dim=feature_model.fragment_edge_encoder.feature_dim,
            condition_dim=feature_model._condition_encoder.feature_dim,
            interaction_dim=edge_condition_interaction_dim,
        )
        # Legacy head retained for checkpoint compatibility; conditional edge
        # scores above are now the only logits used by forward().
        self.edge_absolute_head = nn.Sequential(
            nn.LayerNorm(feature_model.fragment_edge_encoder.feature_dim),
            nn.Linear(feature_model.fragment_edge_encoder.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ion_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.ion_flat_candidates)))
        self.unsaturation_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.unsaturation_flat_candidates)))
        self.radical_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.radical_flat_candidates)))

    def forward(
        self,
        data: Union[FragmentTreeStructure, FragmentTreeFeatures],
        *,
        selected_edge_index: Optional[Tensor] = None,
    ) -> FragmentTreeCandidateSelectionOutput:
        if selected_edge_index is None and isinstance(data, FragmentTreeStructure):
            selected_edge_index = self._tree_bounded_edge_selection(data)
        feature_output = self.feature_model(data, selected_edge_index=selected_edge_index)
        features = feature_output.ft_features
        sample_tree_batch = feature_output.sample_tree_batch
        keep_logit = self.node_keep_head(sample_tree_batch.x).squeeze(-1)
        cleave_logit = self.node_cleave_head(sample_tree_batch.x).squeeze(-1)
        edge_graph = sample_tree_batch.batch[sample_tree_batch.edge_index[0]].long()
        edge_absolute_logit = self.condition_edge_scorer(
            feature_output.shared_edge_h,
            feature_output.condition_h,
            sample_tree_batch.edge_id_global,
            edge_graph,
        )
        edge_cleave_logit = self._edge_cleave_logit(sample_tree_batch)
        ion_logit = self.ion_head(sample_tree_batch.x)
        unsaturation_logit = self.unsaturation_head(sample_tree_batch.x)
        radical_logit = self.radical_head(sample_tree_batch.x)
        return FragmentTreeCandidateSelectionOutput(
            features=features,
            sample_tree_batch=sample_tree_batch,
            keep_logit=keep_logit,
            cleave_logit=cleave_logit,
            edge_absolute_logit=edge_absolute_logit,
            edge_cleave_logit=edge_cleave_logit,
            ion_logit=ion_logit,
            unsaturation_logit=unsaturation_logit,
            radical_logit=radical_logit,
            ion_valid_mask_by_role_adduct=self.feature_model.ion_candidate_valid_mask_by_role_adduct,
            unsaturation_valid_mask_by_role_adduct=self.feature_model.unsaturation_candidate_valid_mask_by_role_adduct,
            radical_valid_mask_by_role_adduct=self.feature_model.radical_candidate_valid_mask_by_role_adduct,
            kept_candidates=self._select_fragment_ion_candidates(features=features, sample_tree_batch=sample_tree_batch, keep_logit=keep_logit, edge_cleave_logit=edge_cleave_logit, ion_logit=ion_logit, unsaturation_logit=unsaturation_logit, radical_logit=radical_logit),
            next_cleavage_candidates=self._select_next_cleavage_candidates(
                features=features,
                sample_tree_batch=sample_tree_batch,
                cleave_logit=cleave_logit,
            ),
            absolute_score_logit=feature_output.absolute_score_logit,
            selected_edge_index=feature_output.selected_edge_index,
        )

    def _edge_cleave_logit(self, sample_tree_batch) -> Tensor:
        if sample_tree_batch.edge_index.numel() == 0:
            return sample_tree_batch.x.new_empty((0,))
        src = sample_tree_batch.edge_index[0].long()
        dst = sample_tree_batch.edge_index[1].long()
        edge_repr = torch.cat(
            [
                sample_tree_batch.x[src],
                sample_tree_batch.x[dst],
                sample_tree_batch.edge_attr,
            ],
            dim=-1,
        )
        return self.edge_cleave_head(edge_repr).squeeze(-1)

    def _tree_bounded_edge_selection(
        self, structure: FragmentTreeStructure
    ) -> Optional[Tensor]:
        """Rank stored edges by summed cross-sample importance per tree.

        Returns a global edge-index tensor capped at ``max_edges_per_tree``
        edges per stored tree, or ``None`` when the mechanism is disabled
        (``max_edges_per_tree is None``) or the structure predates
        ``tree_sample_ptr``. Importance is the condition-scored base logit
        (cheap, no attention) summed over every sample that references the
        edge, with a large bonus for edges that are a target/positive edge
        for at least one such sample, so the budget cannot silently drop
        edges the model must see to learn from.

        Only edges actually referenced by some sample (``sample_edge_index``)
        are ever candidates for selection: an edge no sample references has
        zero importance and would never be worth the attention budget
        anyway, so the per-tree grouping is derived from ``tree_sample_ptr``
        (which sample belongs to which tree) rather than needing a separate
        edge-level tree boundary.
        """
        if self.max_edges_per_tree is None:
            return None
        if not hasattr(structure, "tree_sample_ptr"):
            return None
        set_edge_progress_phase("tree edge importance scoring")
        device = structure.edge_index.device
        with torch.no_grad():
            base_features = self.feature_model.build_node_features(structure)
            edge_h_base, _ = self.fragment_edge_encoder.encode_base(base_features)
            condition_h = self.feature_model._condition_encoder(
                structure.sample_adduct_type_index.to(device).long(),
                structure.sample_ce_value.to(device),
            )
            sample_edge_index = structure.sample_edge_index.to(device).long()
            valid = sample_edge_index[1] >= 0
            sample_ids = sample_edge_index[0, valid]
            edge_ids = sample_edge_index[1, valid]
            if edge_ids.numel() == 0:
                return None
            scores = self.condition_edge_scorer(edge_h_base, condition_h, edge_ids, sample_ids)
            num_edges = int(structure.num_edges)
            target_edge_index = getattr(structure, "target_edge_index", None)
            if target_edge_index is not None and target_edge_index.numel():
                target_pairs = target_edge_index.to(device).long()
                key = sample_ids * (num_edges + 1) + edge_ids
                target_key = target_pairs[0] * (num_edges + 1) + target_pairs[1]
                is_target = torch.isin(key, target_key)
                scores = torch.where(
                    is_target, scores + self._TARGET_EDGE_IMPORTANCE_BONUS, scores
                )
            edge_importance = torch.zeros(num_edges, device=device, dtype=scores.dtype)
            edge_importance.scatter_add_(0, edge_ids, scores)

            # Each referenced edge belongs to exactly one tree, via whichever
            # sample references it (a sample only ever references edges of
            # its own stored tree).
            tree_sample_ptr = structure.tree_sample_ptr.to(device).long()
            num_trees = int(tree_sample_ptr.numel()) - 1
            sample_tree_id = torch.zeros(int(structure.num_samples), dtype=torch.long, device=device)
            for tree_id in range(num_trees):
                start, end = int(tree_sample_ptr[tree_id]), int(tree_sample_ptr[tree_id + 1])
                sample_tree_id[start:end] = tree_id
            edge_to_tree = torch.full((num_edges,), -1, dtype=torch.long, device=device)
            edge_to_tree[edge_ids] = sample_tree_id[sample_ids]
            referenced_edges = edge_ids.unique()

            selected_parts: List[Tensor] = []
            for tree_id in range(num_trees):
                tree_edges = referenced_edges[edge_to_tree[referenced_edges] == tree_id]
                if tree_edges.numel() == 0:
                    continue
                k = min(int(self.max_edges_per_tree), int(tree_edges.numel()))
                top_local = torch.topk(edge_importance[tree_edges], k=k).indices
                selected_parts.append(tree_edges[top_local])
        if not selected_parts:
            return None
        return torch.cat(selected_parts).unique(sorted=True)

    def generate_depth_limited_candidates(
        self,
        data: Union[FragmentTreeStructure, FragmentTreeFeatures],
        *,
        max_depth: int,
        on_step: Optional[Callable[[int, FragmentTreeCandidateSelectionOutput], None]] = None,
        structure_expander: Optional[
            Callable[[int, FragmentTreeCandidateSelectionOutput], FragmentTreeStructure]
        ] = None,
    ) -> FragmentTreeCandidateSelectionOutput:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        # Molecular and cleavage-event encoders are evaluated once.  The much
        # larger sample-tree encoder is then run on bounded edge windows.
        selected_edge_index = (
            self._tree_bounded_edge_selection(data)
            if isinstance(data, FragmentTreeStructure)
            else None
        )
        features = self.feature_model.build_features(data, selected_edge_index=selected_edge_index)
        features = self._initial_depth_features(features)
        output = self._forward_progressive(features, selected_edge_index=selected_edge_index)
        if on_step is not None:
            on_step(0, output)
        for expansion_depth in range(1, max_depth + 1):
            if len(output.next_cleavage_candidates) == 0:
                break
            if structure_expander is not None:
                expanded_structure = structure_expander(expansion_depth, output)
                selected_edge_index = self._tree_bounded_edge_selection(expanded_structure)
                expanded_features = self.feature_model.build_features(
                    expanded_structure, selected_edge_index=selected_edge_index
                )
                output = replace(output, features=expanded_features)
            next_features = self._expand_features_for_next_cleavage(
                output, expansion_depth=expansion_depth
            )
            if next_features is None:
                if on_step is not None:
                    on_step(expansion_depth, output)
                break
            output = self._forward_progressive(next_features, selected_edge_index=selected_edge_index)
            if on_step is not None:
                on_step(expansion_depth, output)
        return output

    def _forward_progressive(
        self,
        features: FragmentTreeFeatures,
        *,
        selected_edge_index: Optional[Tensor] = None,
    ) -> FragmentTreeCandidateSelectionOutput:
        """Rank arbitrary-size edge sets using ``new window + survivors``.

        Training deliberately calls :meth:`forward` directly and therefore
        remains a single teacher-forced pass.  This staged path is inference
        only and bounds the tree-transformer input for each sample.
        """
        limit = self.max_edges_per_step
        retain = self.max_retained_edges
        pairs = features.structure.sample_edge_index.detach().cpu().t().tolist()
        by_sample: dict[int, List[int]] = {}
        for sample_id, edge_id in pairs:
            if int(edge_id) >= 0:
                by_sample.setdefault(int(sample_id), []).append(int(edge_id))
        if limit is None or all(len(set(v)) <= limit for v in by_sample.values()):
            return self.forward(features, selected_edge_index=selected_edge_index)

        survivors: dict[int, List[int]] = {sample_id: [] for sample_id in by_sample}
        offsets = {sample_id: 0 for sample_id in by_sample}
        while any(offsets[sample_id] < len(set(edges)) for sample_id, edges in by_sample.items()):
            window_pairs: List[Tuple[int, int]] = []
            for sample_id, raw_edges in by_sample.items():
                edges = sorted(set(raw_edges))
                start = offsets[sample_id]
                new_edges = edges[start : start + int(limit)]
                offsets[sample_id] += len(new_edges)
                for edge_id in sorted(set(survivors[sample_id] + new_edges)):
                    window_pairs.append((sample_id, edge_id))

            window_features = self._with_sample_edges(features, window_pairs)
            window_output = self.forward(window_features, selected_edge_index=selected_edge_index)
            survivors = self._rank_edges(
                window_output,
                window_pairs,
                max_per_sample=retain,
            )

        final_pairs = [
            (sample_id, edge_id)
            for sample_id, edge_ids in sorted(survivors.items())
            for edge_id in edge_ids
        ]
        return self.forward(
            self._with_sample_edges(features, final_pairs),
            selected_edge_index=selected_edge_index,
        )

    @staticmethod
    def _with_sample_edges(
        features: FragmentTreeFeatures,
        pairs: List[Tuple[int, int]],
    ) -> FragmentTreeFeatures:
        device = features.structure.sample_edge_index.device
        sample_edge_index = torch.tensor(
            pairs, dtype=torch.long, device=device
        ).t().contiguous()
        if not pairs:
            sample_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        structure = replace(
            features.structure,
            sample_edge_index=sample_edge_index,
        )
        return replace(features, structure=structure)

    @staticmethod
    def _rank_edges(
        output: FragmentTreeCandidateSelectionOutput,
        pairs: List[Tuple[int, int]],
        *,
        max_per_sample: Optional[int],
    ) -> dict[int, List[int]]:
        """Beam-rank edges by cumulative path score.

        The additive path score is the autoregressive log-score analogue used
        for fragmentation-graph decoding: a deep edge survives only when its
        complete sequence of preceding cleavages remains plausible.
        """
        structure = output.features.structure
        batch = output.sample_tree_batch
        node_score: dict[Tuple[int, int], float] = {}
        kept_sample_ids = batch.kept_sample_ids.detach().cpu().long()
        for batch_node in range(int(batch.x.size(0))):
            graph_index = int(batch.batch[batch_node].detach().cpu().item())
            sample_id = int(kept_sample_ids[graph_index].item())
            global_node = int(batch.node_id_global[batch_node].detach().cpu().item())
            node_score[(sample_id, global_node)] = float(
                output.keep_logit[batch_node].detach().cpu().item()
            )

        edge_score: dict[Tuple[int, int], float] = {}
        if hasattr(batch, "edge_id_global"):
            for local_edge, global_edge in enumerate(
                batch.edge_id_global.detach().cpu().long().tolist()
            ):
                graph_index = int(batch.batch[batch.edge_index[0, local_edge]].detach().cpu().item())
                sample_id = int(kept_sample_ids[graph_index].item())
                # Log-probability increments are never positive.  Therefore a
                # child path cannot outrank its own prefix merely because its
                # local raw logit is large.
                local_score = (
                    torch.nn.functional.logsigmoid(output.edge_absolute_logit[local_edge])
                    + torch.nn.functional.logsigmoid(output.edge_cleave_logit[local_edge])
                )
                edge_score[(sample_id, int(global_edge))] = float(local_score.detach().cpu().item())

        ranked: dict[int, List[Tuple[float, int]]] = {}
        path_score: dict[Tuple[int, int], float] = {}
        edge_src = structure.edge_index[0].detach().cpu().long()
        edge_dst = structure.edge_index[1].detach().cpu().long()
        for sample_id, edge_id in pairs:
            dst = int(edge_dst[edge_id].item())
            score = edge_score.get((sample_id, edge_id), float("-inf"))
            score += float(torch.nn.functional.logsigmoid(
                torch.tensor(node_score.get((sample_id, dst), 0.0))
            ).item())
            parent_node = int(edge_src[edge_id].item())
            parent_scores = [
                value for (sid, previous_edge), value in path_score.items()
                if sid == sample_id and int(edge_dst[previous_edge].item()) == parent_node
            ]
            if parent_scores:
                score += max(parent_scores)
            path_score[(sample_id, edge_id)] = score
            ranked.setdefault(sample_id, []).append((score, edge_id))
        result: dict[int, List[int]] = {}
        for sample_id, rows in ranked.items():
            rows.sort(key=lambda row: (-row[0], row[1]))
            k = len(rows) if max_per_sample is None else min(max_per_sample, len(rows))
            result[sample_id] = [edge_id for _, edge_id in rows[:k]]
        return result

    def _expand_features_for_next_cleavage(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        *,
        expansion_depth: int = 1,
    ) -> Optional[FragmentTreeFeatures]:
        """Expand every stored outgoing edge of at most K selected nodes.

        No new chemistry is run here: children are read exclusively from the
        precomputed ``structure.edge_index`` DAG.
        """
        features = output.features
        structure = features.structure
        device = structure.edge_index.device
        if structure.edge_index.numel() == 0:
            return None

        edge_src = structure.edge_index[0].to(device).long()
        sample_edge_pairs = []
        if structure.sample_edge_index.numel() > 0:
            sample_edge_pairs.extend(
                (int(sample_id), int(edge_id))
                for sample_id, edge_id in structure.sample_edge_index.detach().cpu().t().tolist()
                if int(edge_id) >= 0
            )

        existing_pairs = set(sample_edge_pairs)
        frontier_budget = self.max_next_cleavage_candidates
        frontier_by_sample: dict[int, List[NextCleavageCandidate]] = {}
        for candidate in output.next_cleavage_candidates:
            frontier_by_sample.setdefault(int(candidate.sample_id), []).append(candidate)
        bounded_frontier: List[NextCleavageCandidate] = []
        for rows in frontier_by_sample.values():
            rows.sort(key=lambda item: (-item.score, item.global_node_id))
            bounded_frontier.extend(rows[:frontier_budget])

        candidate_pairs: List[Tuple[int, int]] = []
        for candidate in bounded_frontier:
            outgoing_edges = (edge_src == int(candidate.global_node_id)).nonzero(as_tuple=False).view(-1)
            for edge_id in outgoing_edges.detach().cpu().tolist():
                pair = (int(candidate.sample_id), int(edge_id))
                if pair in existing_pairs:
                    continue
                candidate_pairs.append(pair)

        candidate_pairs = self._top_conditioned_edge_pairs(
            features,
            candidate_pairs,
            max_per_sample=self._depth_budget(
                self.fragment_edge_encoder.max_edges_per_depth, expansion_depth
            ),
        )
        added = bool(candidate_pairs)
        sample_edge_pairs.extend(candidate_pairs)

        if not added:
            return None

        sample_edge_pairs.sort()
        sample_edge_index = torch.tensor(
            sample_edge_pairs,
            dtype=torch.long,
            device=device,
        ).t().contiguous()

        expanded_structure = FragmentTreeStructure(
            node_smiles=structure.node_smiles,
            node_graph=structure.node_graph,
            node_graph_offset=structure.node_graph_offset,
            node_formula=structure.node_formula,
            formula_element_order=structure.formula_element_order,
            edge_index=structure.edge_index,
            tree_sample_ptr=structure.tree_sample_ptr,
            cleavage_event_edge_index=structure.cleavage_event_edge_index,
            cleavage_event=structure.cleavage_event,
            cleavage_atom_idxs=structure.cleavage_atom_idxs,
            reactant_tuple_length_table=structure.reactant_tuple_length_table,
            product_tuple_length_table=structure.product_tuple_length_table,
            ion_formula_delta=structure.ion_formula_delta,
            unsaturation_formula_delta=structure.unsaturation_formula_delta,
            radical_formula_delta=structure.radical_formula_delta,
            sample_adduct_type_index=structure.sample_adduct_type_index,
            sample_ce_value=structure.sample_ce_value,
            sample_edge_index=sample_edge_index,
            precursor_edge_index_path=structure.precursor_edge_index_path,
            precursor_unsaturation_index=structure.precursor_unsaturation_index,
            precursor_radical_index=structure.precursor_radical_index,
            precursor_sample_index=structure.precursor_sample_index,
        )
        return FragmentTreeFeatures(
            node_graphs=features.node_graphs,
            edge_attr=features.edge_attr,
            structure=expanded_structure,
        )

    @staticmethod
    def _depth_budget(values: Tuple[int, ...], expansion_depth: int) -> int:
        if not values:
            raise ValueError("depth budget must not be empty.")
        if expansion_depth < 0 or expansion_depth >= len(values):
            raise IndexError(
                f"expansion_depth={expansion_depth} is outside {len(values)} depth budgets."
            )
        return int(values[expansion_depth])

    def _initial_depth_features(self, features: FragmentTreeFeatures) -> FragmentTreeFeatures:
        """Keep only depth-1 sample edges, capped by the first depth budget."""
        structure = features.structure
        edge_src, edge_dst = structure.edge_index.long()
        has_parent = torch.zeros(structure.num_nodes, dtype=torch.bool, device=edge_src.device)
        has_parent[edge_dst] = True
        roots = (~has_parent).nonzero(as_tuple=False).flatten()
        root_set = set(int(value) for value in roots.detach().cpu().tolist())
        precursor_sources: dict[int, set[int]] = {
            sample: set(root_set) for sample in range(structure.num_samples)
        }
        paths = structure.precursor_edge_index_path.detach().cpu().long()
        precursor_samples = structure.precursor_sample_index.detach().cpu().long()
        edge_dst_cpu = edge_dst.detach().cpu()
        for row, sample_id in zip(paths, precursor_samples.tolist()):
            valid = row[row >= 0]
            if valid.numel():
                precursor_sources.setdefault(int(sample_id), set()).add(
                    int(edge_dst_cpu[int(valid[-1])].item())
                )
        pairs = []
        for sample_id, edge_id in structure.sample_edge_index.detach().cpu().t().tolist():
            if int(edge_id) < 0:
                continue
            if int(edge_src[int(edge_id)].item()) in precursor_sources.get(int(sample_id), root_set):
                pairs.append((int(sample_id), int(edge_id)))
        pairs = self._top_conditioned_edge_pairs(
            features, pairs, max_per_sample=self.fragment_edge_encoder.max_edges_per_depth[0]
        )
        return self._with_sample_edges(features, pairs)

    def _top_conditioned_edge_pairs(
        self,
        features: FragmentTreeFeatures,
        pairs: List[Tuple[int, int]],
        *,
        max_per_sample: int,
    ) -> List[Tuple[int, int]]:
        """Apply the configured edge budget independently per sample."""
        if not pairs:
            return []
        structure = features.structure
        device = features.edge_attr.device
        condition_h = self.feature_model._condition_encoder(
            structure.sample_adduct_type_index.to(device).long(),
            structure.sample_ce_value.to(device),
        )
        edge_ids = torch.tensor([edge for _, edge in pairs], dtype=torch.long, device=device)
        sample_ids = torch.tensor([sample for sample, _ in pairs], dtype=torch.long, device=device)
        with torch.no_grad():
            scores = self.condition_edge_scorer(
                features.edge_attr, condition_h, edge_ids, sample_ids
            ).detach().cpu().tolist()
        by_sample: dict[int, List[Tuple[float, int]]] = {}
        for (sample_id, edge_id), score in zip(pairs, scores):
            by_sample.setdefault(sample_id, []).append((float(score), edge_id))
        selected: List[Tuple[int, int]] = []
        for sample_id, rows in sorted(by_sample.items()):
            rows.sort(key=lambda row: (-row[0], row[1]))
            selected.extend((sample_id, edge_id) for _, edge_id in rows[:max_per_sample])
        return selected

    @staticmethod
    def _grouped_topk(values: Tensor, groups: Tensor, k: int, num_groups: int) -> Tensor:
        """Return score-ordered top-k rows per group, without a Python group loop.

        Stable sorting gives tied scores a deterministic input-row order.
        """
        order = torch.argsort(values, descending=True, stable=True)
        order = order[torch.argsort(groups[order], stable=True)]
        counts = torch.bincount(groups, minlength=num_groups)
        starts = counts.cumsum(0) - counts
        rank = torch.arange(order.numel(), device=order.device) - starts[groups[order]]
        return order[rank < k]

    def _select_fragment_ion_candidates(self, *, features, sample_tree_batch, keep_logit: Tensor, edge_cleave_logit: Tensor, ion_logit: Tensor, unsaturation_logit: Tensor, radical_logit: Tensor) -> List[FragmentIonCandidate]:
        structure = features.structure
        device = keep_logit.device
        num_nodes = keep_logit.numel()
        if num_nodes == 0:
            return []
        kept_sample_ids = sample_tree_batch.kept_sample_ids.to(device).long()
        num_graphs = kept_sample_ids.numel()
        graph_index = sample_tree_batch.batch.to(device).long()

        # Segmented logsumexp of incoming edges, including shared destinations.
        # Detaching the stabilizing maximum preserves logsumexp's derivative.
        edge_dst = sample_tree_batch.edge_index[1].to(device).long()
        edge_max = edge_cleave_logit.new_full((num_nodes,), -torch.inf)
        edge_max.scatter_reduce_(0, edge_dst, edge_cleave_logit.detach(), reduce="amax", include_self=True)
        edge_sum = edge_cleave_logit.new_zeros(num_nodes).scatter_add(
            0, edge_dst, (edge_cleave_logit - edge_max[edge_dst]).exp()
        )
        has_incoming = torch.bincount(edge_dst, minlength=num_nodes) > 0
        # Roots have an additive edge contribution of zero, not -inf.
        incoming_score = torch.where(has_incoming, edge_max, 0) + torch.where(
            has_incoming, edge_sum, 1
        ).log()
        node_score = keep_logit + incoming_score
        graph_max = node_score.new_full((num_graphs,), -torch.inf)
        graph_max.scatter_reduce_(0, graph_index, node_score.detach(), reduce="amax", include_self=True)
        node_exp = (node_score - graph_max[graph_index]).exp()
        node_sum = node_exp.new_zeros(num_graphs).scatter_add(0, graph_index, node_exp)
        node_probability = node_exp / node_sum[graph_index]

        # Normalize over ALL nodes before applying the optional node budget.
        node_order = self._grouped_topk(
            node_score, graph_index, self.max_nodes_for_ion_candidates or num_nodes, num_graphs
        )
        roles = (~sample_tree_batch.node_is_precursor_root.to(device).bool()).long()
        adducts = sample_tree_batch.node_main_adduct_type_index.to(device).long()
        masks = [mask.to(device).bool() for mask in (
            self.feature_model.ion_candidate_valid_mask_by_role_adduct,
            self.feature_model.unsaturation_candidate_valid_mask_by_role_adduct,
            self.feature_model.radical_candidate_valid_mask_by_role_adduct,
        )]
        num_adducts = masks[0].size(1)
        if bool(((adducts[node_order] < 0) | (adducts[node_order] >= num_adducts)).any()):
            raise IndexError("main_adduct_index is out of range for candidate masks.")
        node_groups = roles[node_order] * num_adducts + adducts[node_order]
        row_nodes, row_states, row_logits, row_adduct_probs, row_probs = [], [], [], [], []

        # Nodes sharing a role/adduct have the same valid state combinations.
        # Only this small categorical loop remains; each block evaluates many
        # nodes and all their valid states together. Formula vectors are NOT
        # expanded here, keeping temporary memory proportional to nodes*states.
        for group in range(2 * num_adducts):
            nodes = node_order[node_groups == group]
            if nodes.numel() == 0:
                continue
            role, adduct = divmod(group, num_adducts)
            indices = [mask[role, adduct].nonzero(as_tuple=False).flatten() for mask in masks]
            if any(index.numel() == 0 for index in indices):
                continue
            states = torch.cartesian_prod(*indices).reshape(-1, 3)
            per_node = min(self.max_fragment_ion_candidates, states.size(0))
            # Bound each dense score block, even for large structure batches.
            nodes_per_block = max(1, self._ION_CANDIDATE_SCORE_BLOCK_ELEMENTS // states.size(0))
            for block in nodes.split(nodes_per_block):
                logits = (ion_logit[block[:, None], states[:, 0]]
                          + unsaturation_logit[block[:, None], states[:, 1]]
                          + radical_logit[block[:, None], states[:, 2]])
                adduct_probability = torch.softmax(logits, dim=1)
                probability = node_probability[block, None] * adduct_probability
                top = torch.topk(probability, k=per_node, dim=1)
                row_nodes.append(block[:, None].expand(-1, per_node).reshape(-1))
                row_states.append(states[top.indices].reshape(-1, 3))
                row_logits.append(logits.gather(1, top.indices).reshape(-1))
                row_adduct_probs.append(adduct_probability.gather(1, top.indices).reshape(-1))
                row_probs.append(top.values.reshape(-1))
        if not row_nodes:
            return []
        nodes = torch.cat(row_nodes)
        states = torch.cat(row_states)
        logits = torch.cat(row_logits)
        adduct_probability = torch.cat(row_adduct_probs)
        probability = torch.cat(row_probs)
        scores = probability.clamp_min(1e-12).log()

        # Restore node-score order before the stable per-spectrum ranking.
        node_rank = node_order.new_empty(num_nodes)
        node_rank[node_order] = torch.arange(node_order.numel(), device=device)
        order = torch.argsort(node_rank[nodes], stable=True)
        selected = order[self._grouped_topk(
            scores[order], graph_index[nodes[order]], self.max_fragment_ion_candidates, num_graphs
        )]
        nodes = nodes[selected]
        states = states[selected]
        global_nodes = sample_tree_batch.node_id_global.to(device).long()[nodes]
        final_formula = (structure.node_formula.to(device)[global_nodes].float()
                         + (structure.ion_formula_delta.to(device)[states[:, 0]].float()
                            + structure.unsaturation_formula_delta.to(device)[states[:, 1]].float()
                            + structure.radical_formula_delta.to(device)[states[:, 2]].float()))

        # Only final survivors cross to the CPU. Retain the differentiable GPU
        # values for downstream intensity training; Python objects are the
        # existing public interface and require this final materialization.
        metadata = torch.cat((kept_sample_ids[graph_index[nodes], None], nodes[:, None],
                              global_nodes[:, None], states), dim=1).detach().cpu().tolist()
        values = torch.stack((scores[selected], keep_logit[nodes], logits[selected],
                              probability[selected], node_probability[nodes],
                              adduct_probability[selected]), dim=1).detach().cpu().tolist()
        formulas = final_formula.detach().cpu()
        score_tensors = scores[selected].unbind()
        probability_tensors = probability[selected].unbind()
        tensorizer = self.feature_model.formula_tensorizer
        return [FragmentIonCandidate(
            sample_id=meta[0], batch_node_index=meta[1], global_node_id=meta[2],
            ion_index=meta[3], unsaturation_index=meta[4], radical_index=meta[5],
            formula=tensorizer.tensor_to_formula(formulas[i]), formula_tensor=formulas[i],
            score=value[0], keep_logit=value[1], candidate_logit=value[2], probability=value[3],
            score_tensor=score_tensors[i], edge_probability=value[4], adduct_probability=value[5],
            probability_tensor=probability_tensors[i],
        ) for i, (meta, value) in enumerate(zip(metadata, values))]

    def _score_joint_ion_candidates_for_node(self, *, structure: FragmentTreeStructure, sample_tree_batch, batch_node_index: int, global_node_id: int, sample_id: int, keep_logit: Tensor, ion_logit: Tensor, unsaturation_logit: Tensor, radical_logit: Tensor, edge_probability: Tensor) -> List[FragmentIonCandidate]:
        device = keep_logit.device
        base_formula = structure.node_formula[global_node_id].to(device).float()
        main_adduct_index = int(sample_tree_batch.node_main_adduct_type_index[batch_node_index].detach().cpu().item())
        role_index = 0 if bool(sample_tree_batch.node_is_precursor_root[batch_node_index].detach().cpu().item()) else 1
        ion_indices = self._valid_candidate_indices(self.feature_model.ion_candidate_valid_mask_by_role_adduct, role_index=role_index, main_adduct_index=main_adduct_index, device=device)
        unsaturation_indices = self._valid_candidate_indices(self.feature_model.unsaturation_candidate_valid_mask_by_role_adduct, role_index=role_index, main_adduct_index=main_adduct_index, device=device)
        radical_indices = self._valid_candidate_indices(self.feature_model.radical_candidate_valid_mask_by_role_adduct, role_index=role_index, main_adduct_index=main_adduct_index, device=device)
        rows: List[Tuple[int, int, int, Tensor, Tensor]] = []
        for ion_index_tensor in ion_indices:
            ion_index = int(ion_index_tensor.detach().cpu().item())
            ion_delta = structure.ion_formula_delta[ion_index].to(device).float()
            for unsaturation_index_tensor in unsaturation_indices:
                unsaturation_index = int(unsaturation_index_tensor.detach().cpu().item())
                unsaturation_delta = structure.unsaturation_formula_delta[unsaturation_index].to(device).float()
                for radical_index_tensor in radical_indices:
                    radical_index = int(radical_index_tensor.detach().cpu().item())
                    radical_delta = structure.radical_formula_delta[radical_index].to(device).float()
                    delta_formula = ion_delta + unsaturation_delta + radical_delta
                    rows.append((ion_index, unsaturation_index, radical_index, base_formula + delta_formula, delta_formula))
        if len(rows) == 0:
            return []
        formula_tensor = torch.stack([row[3] for row in rows], dim=0)
        candidate_logit = torch.stack(
            [
                ion_logit[batch_node_index, row[0]]
                + unsaturation_logit[batch_node_index, row[1]]
                + radical_logit[batch_node_index, row[2]]
                for row in rows
            ],
            dim=0,
        )
        # Ion/adduct states of the same node are mutually exclusive.  This
        # suppresses isotope-like clusters caused by independent sigmoids.
        adduct_probability = torch.softmax(candidate_logit, dim=0)
        joint_probability = edge_probability * adduct_probability
        combined_score = torch.log(joint_probability.clamp_min(1e-12))
        top = torch.topk(joint_probability, k=min(self.max_fragment_ion_candidates, int(joint_probability.numel())))
        tensorizer = self.feature_model.formula_tensorizer
        candidates: List[FragmentIonCandidate] = []
        for selected in top.indices.detach().cpu().tolist():
            selected = int(selected)
            ion_index, unsaturation_index, radical_index, final_formula, _ = rows[selected]
            formula_cpu = final_formula.detach().cpu()
            score_value = float(combined_score[selected].detach().cpu().item())
            candidates.append(FragmentIonCandidate(sample_id=sample_id, batch_node_index=batch_node_index, global_node_id=global_node_id, ion_index=ion_index, unsaturation_index=unsaturation_index, radical_index=radical_index, formula=tensorizer.tensor_to_formula(formula_cpu), formula_tensor=formula_cpu, score=score_value, keep_logit=float(keep_logit[batch_node_index].detach().cpu().item()), candidate_logit=float(candidate_logit[selected].detach().cpu().item()), probability=float(joint_probability[selected].detach().cpu().item()), score_tensor=combined_score[selected], edge_probability=float(edge_probability.detach().cpu().item()), adduct_probability=float(adduct_probability[selected].detach().cpu().item()), probability_tensor=joint_probability[selected]))
        return candidates

    def _select_next_cleavage_candidates(self, *, features, sample_tree_batch, cleave_logit: Tensor) -> List[NextCleavageCandidate]:
        device = cleave_logit.device
        node_is_precursor_root = sample_tree_batch.node_is_precursor_root.to(device).bool()
        kept_sample_ids = sample_tree_batch.kept_sample_ids.to(device).long()
        graph_index_by_node = sample_tree_batch.batch.to(device).long()
        expandable_global_nodes = features.structure.edge_index[0].to(device).long().unique()
        candidates: List[NextCleavageCandidate] = []
        for graph_index in range(int(kept_sample_ids.numel())):
            node_index = (graph_index_by_node == graph_index).nonzero(as_tuple=False).view(-1)
            node_index = node_index[~node_is_precursor_root[node_index]]
            node_index = node_index[
                torch.isin(
                    sample_tree_batch.node_id_global[node_index].to(device).long(),
                    expandable_global_nodes,
                )
            ]
            if node_index.numel() == 0:
                continue
            selected_node_index = node_index[
                torch.topk(
                    cleave_logit[node_index],
                    k=min(self.max_next_cleavage_candidates, int(node_index.numel())),
                ).indices
            ]
            for batch_node_index_tensor in selected_node_index:
                batch_node_index = int(batch_node_index_tensor.detach().cpu().item())
                score = cleave_logit[batch_node_index]
                candidates.append(
                    NextCleavageCandidate(
                        sample_id=int(kept_sample_ids[graph_index].detach().cpu().item()),
                        batch_node_index=batch_node_index,
                        global_node_id=int(sample_tree_batch.node_id_global[batch_node_index].detach().cpu().item()),
                        score=float(score.detach().cpu().item()),
                        probability=float(torch.sigmoid(score).detach().cpu().item()),
                    )
                )
        return candidates

    @staticmethod
    def _valid_candidate_indices(mask_by_role_adduct: Tensor, *, role_index: int, main_adduct_index: int, device: torch.device) -> Tensor:
        mask = mask_by_role_adduct.to(device).bool()
        if main_adduct_index < 0 or main_adduct_index >= mask.size(1):
            raise IndexError(f"main_adduct_index={main_adduct_index} is out of range for mask shape {tuple(mask.shape)}.")
        return mask[int(role_index), int(main_adduct_index)].nonzero(as_tuple=False).view(-1)
