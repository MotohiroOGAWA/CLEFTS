from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ...libs.mmkit.mmkit import Formula
from ..input.fragment_tree_features import FragmentTreeFeatures
from ..input.fragment_tree_structure import FragmentTreeStructure
from .fragment_tree_feature_model import FragmentTreeFeatureModel


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
    edge_cleave_logit: Tensor
    ion_logit: Tensor
    unsaturation_logit: Tensor
    radical_logit: Tensor
    ion_valid_mask_by_role_adduct: Tensor
    unsaturation_valid_mask_by_role_adduct: Tensor
    radical_valid_mask_by_role_adduct: Tensor
    kept_candidates: List[FragmentIonCandidate]
    next_cleavage_candidates: List[NextCleavageCandidate]



class FragmentTreeCandidateSelector(nn.Module):
    """Select fragments, ion states, and next cleavage candidates directly."""

    def __init__(
        self,
        feature_model: FragmentTreeFeatureModel,
        *,
        max_fragment_ion_candidates: int = 30,
        max_next_cleavage_candidates: int = 3,
        max_edges_per_step: Optional[int] = 128,
        max_retained_edges: Optional[int] = 30,
        max_nodes_for_ion_candidates: Optional[int] = None,
        hidden_dim: Optional[int] = None,
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
        if max_nodes_for_ion_candidates is not None and max_nodes_for_ion_candidates <= 0:
            raise ValueError("max_nodes_for_ion_candidates must be positive or None.")
        self.feature_model = feature_model
        self.mol_encoder = feature_model.mol_encoder
        self.fragmenter = feature_model.fragmenter
        self.cleavage_edge_fnet = feature_model.cleavage_edge_fnet
        self.max_fragment_ion_candidates = int(max_fragment_ion_candidates)
        self.max_next_cleavage_candidates = int(max_next_cleavage_candidates)
        self.max_edges_per_step = max_edges_per_step
        self.max_retained_edges = max_retained_edges
        self.max_nodes_for_ion_candidates = max_nodes_for_ion_candidates
        tree_dim = int(feature_model.tree_encoder.hidden_dim)
        hidden_dim = int(hidden_dim or tree_dim)
        self.node_keep_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.node_cleave_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.edge_cleave_head = nn.Sequential(
            nn.Linear(tree_dim * 2 + feature_model.cleavage_edge_fnet.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ion_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.ion_flat_candidates)))
        self.unsaturation_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.unsaturation_flat_candidates)))
        self.radical_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.radical_flat_candidates)))

    def forward(self, data: Union[FragmentTreeStructure, FragmentTreeFeatures]) -> FragmentTreeCandidateSelectionOutput:
        feature_output = self.feature_model(data)
        features = feature_output.ft_features
        sample_tree_batch = feature_output.sample_tree_batch
        keep_logit = self.node_keep_head(sample_tree_batch.x).squeeze(-1)
        cleave_logit = self.node_cleave_head(sample_tree_batch.x).squeeze(-1)
        edge_cleave_logit = self._edge_cleave_logit(sample_tree_batch)
        ion_logit = self.ion_head(sample_tree_batch.x)
        unsaturation_logit = self.unsaturation_head(sample_tree_batch.x)
        radical_logit = self.radical_head(sample_tree_batch.x)
        return FragmentTreeCandidateSelectionOutput(
            features=features,
            sample_tree_batch=sample_tree_batch,
            keep_logit=keep_logit,
            cleave_logit=cleave_logit,
            edge_cleave_logit=edge_cleave_logit,
            ion_logit=ion_logit,
            unsaturation_logit=unsaturation_logit,
            radical_logit=radical_logit,
            ion_valid_mask_by_role_adduct=self.feature_model.ion_candidate_valid_mask_by_role_adduct,
            unsaturation_valid_mask_by_role_adduct=self.feature_model.unsaturation_candidate_valid_mask_by_role_adduct,
            radical_valid_mask_by_role_adduct=self.feature_model.radical_candidate_valid_mask_by_role_adduct,
            kept_candidates=self._select_fragment_ion_candidates(features=features, sample_tree_batch=sample_tree_batch, keep_logit=keep_logit, ion_logit=ion_logit, unsaturation_logit=unsaturation_logit, radical_logit=radical_logit),
            next_cleavage_candidates=self._select_next_cleavage_candidates(sample_tree_batch=sample_tree_batch, cleave_logit=cleave_logit),
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

    def generate_depth_limited_candidates(self, data: Union[FragmentTreeStructure, FragmentTreeFeatures], *, max_depth: int) -> FragmentTreeCandidateSelectionOutput:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        # Molecular and cleavage-event encoders are evaluated once.  The much
        # larger sample-tree encoder is then run on bounded edge windows.
        features = self.feature_model.build_features(data)
        output = self._forward_progressive(features)
        for _ in range(max_depth):
            if len(output.next_cleavage_candidates) == 0:
                break
            next_features = self._expand_features_for_next_cleavage(output)
            if next_features is None:
                break
            output = self._forward_progressive(next_features)
        return output

    def _forward_progressive(
        self,
        features: FragmentTreeFeatures,
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
            return self.forward(features)

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
            window_output = self.forward(window_features)
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
        return self.forward(self._with_sample_edges(features, final_pairs))

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
        """Score an edge by its child keep score and edge cleavage score."""
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
                edge_score[(sample_id, int(global_edge))] = float(
                    output.edge_cleave_logit[local_edge].detach().cpu().item()
                )

        ranked: dict[int, List[Tuple[float, int]]] = {}
        edge_dst = structure.edge_index[1].detach().cpu().long()
        for sample_id, edge_id in pairs:
            dst = int(edge_dst[edge_id].item())
            score = node_score.get((sample_id, dst), float("-inf"))
            score += edge_score.get((sample_id, edge_id), 0.0)
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
    ) -> Optional[FragmentTreeFeatures]:
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
        added = False
        for candidate in output.next_cleavage_candidates:
            outgoing_edges = (edge_src == int(candidate.global_node_id)).nonzero(as_tuple=False).view(-1)
            for edge_id_tensor in outgoing_edges.detach().cpu().tolist():
                pair = (int(candidate.sample_id), int(edge_id_tensor))
                if pair in existing_pairs:
                    continue
                existing_pairs.add(pair)
                sample_edge_pairs.append(pair)
                added = True

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

    def _select_fragment_ion_candidates(self, *, features, sample_tree_batch, keep_logit: Tensor, ion_logit: Tensor, unsaturation_logit: Tensor, radical_logit: Tensor) -> List[FragmentIonCandidate]:
        structure = features.structure
        device = keep_logit.device
        node_is_precursor_root = sample_tree_batch.node_is_precursor_root.to(device).bool()
        kept_sample_ids = sample_tree_batch.kept_sample_ids.to(device).long()
        graph_index_by_node = sample_tree_batch.batch.to(device).long()
        candidates: List[FragmentIonCandidate] = []

        for graph_index in range(int(kept_sample_ids.numel())):
            sample_node_index = (graph_index_by_node == graph_index).nonzero(as_tuple=False).view(-1)
            sample_node_index = sample_node_index[~node_is_precursor_root[sample_node_index]]
            if sample_node_index.numel() == 0:
                continue
            node_score = keep_logit[sample_node_index]
            k = int(node_score.numel())
            if self.max_nodes_for_ion_candidates is not None:
                k = min(int(self.max_nodes_for_ion_candidates), k)
            node_order = sample_node_index[torch.topk(node_score, k=k).indices]
            sample_candidates: List[FragmentIonCandidate] = []
            for batch_node_index_tensor in node_order:
                batch_node_index = int(batch_node_index_tensor.detach().cpu().item())
                sample_candidates.extend(
                    self._score_joint_ion_candidates_for_node(
                        structure=structure,
                        sample_tree_batch=sample_tree_batch,
                        batch_node_index=batch_node_index,
                        global_node_id=int(sample_tree_batch.node_id_global[batch_node_index].detach().cpu().item()),
                        sample_id=int(kept_sample_ids[graph_index].detach().cpu().item()),
                        keep_logit=keep_logit,
                        ion_logit=ion_logit,
                        unsaturation_logit=unsaturation_logit,
                        radical_logit=radical_logit,
                    )
                )
            sample_candidates.sort(key=lambda item: item.score, reverse=True)
            candidates.extend(sample_candidates[: self.max_fragment_ion_candidates])

        return candidates

    def _score_joint_ion_candidates_for_node(self, *, structure: FragmentTreeStructure, sample_tree_batch, batch_node_index: int, global_node_id: int, sample_id: int, keep_logit: Tensor, ion_logit: Tensor, unsaturation_logit: Tensor, radical_logit: Tensor) -> List[FragmentIonCandidate]:
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
        combined_score = keep_logit[batch_node_index] + candidate_logit
        top = torch.topk(combined_score, k=min(self.max_fragment_ion_candidates, int(combined_score.numel())))
        tensorizer = self.feature_model.formula_tensorizer
        candidates: List[FragmentIonCandidate] = []
        for selected in top.indices.detach().cpu().tolist():
            selected = int(selected)
            ion_index, unsaturation_index, radical_index, final_formula, _ = rows[selected]
            formula_cpu = final_formula.detach().cpu()
            score_value = float(combined_score[selected].detach().cpu().item())
            candidates.append(FragmentIonCandidate(sample_id=sample_id, batch_node_index=batch_node_index, global_node_id=global_node_id, ion_index=ion_index, unsaturation_index=unsaturation_index, radical_index=radical_index, formula=tensorizer.tensor_to_formula(formula_cpu), formula_tensor=formula_cpu, score=score_value, keep_logit=float(keep_logit[batch_node_index].detach().cpu().item()), candidate_logit=float(candidate_logit[selected].detach().cpu().item()), probability=float(torch.sigmoid(combined_score[selected]).detach().cpu().item()), score_tensor=combined_score[selected]))
        return candidates

    def _select_next_cleavage_candidates(self, *, sample_tree_batch, cleave_logit: Tensor) -> List[NextCleavageCandidate]:
        device = cleave_logit.device
        node_is_precursor_root = sample_tree_batch.node_is_precursor_root.to(device).bool()
        kept_sample_ids = sample_tree_batch.kept_sample_ids.to(device).long()
        graph_index_by_node = sample_tree_batch.batch.to(device).long()
        candidates: List[NextCleavageCandidate] = []
        for graph_index in range(int(kept_sample_ids.numel())):
            node_index = (graph_index_by_node == graph_index).nonzero(as_tuple=False).view(-1)
            node_index = node_index[~node_is_precursor_root[node_index]]
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

