from __future__ import annotations

from dataclasses import dataclass
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
        max_next_cleavage_candidates: int = 30,
        max_nodes_for_ion_candidates: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if max_fragment_ion_candidates <= 0:
            raise ValueError("max_fragment_ion_candidates must be positive.")
        if max_next_cleavage_candidates <= 0:
            raise ValueError("max_next_cleavage_candidates must be positive.")
        if max_nodes_for_ion_candidates is not None and max_nodes_for_ion_candidates <= 0:
            raise ValueError("max_nodes_for_ion_candidates must be positive or None.")
        self.feature_model = feature_model
        self.mol_encoder = feature_model.mol_encoder
        self.fragmenter = feature_model.fragmenter
        self.cleavage_edge_fnet = feature_model.cleavage_edge_fnet
        self.max_fragment_ion_candidates = int(max_fragment_ion_candidates)
        self.max_next_cleavage_candidates = int(max_next_cleavage_candidates)
        self.max_nodes_for_ion_candidates = max_nodes_for_ion_candidates
        tree_dim = int(feature_model.tree_encoder.dim)
        hidden_dim = int(hidden_dim or tree_dim)
        self.node_keep_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.node_cleave_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.ion_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.ion_flat_candidates)))
        self.unsaturation_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.unsaturation_flat_candidates)))
        self.radical_head = nn.Sequential(nn.Linear(tree_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(feature_model.radical_flat_candidates)))

    def forward(self, data: Union[FragmentTreeStructure, FragmentTreeFeatures]) -> FragmentTreeCandidateSelectionOutput:
        feature_output = self.feature_model(data)
        features = feature_output.ft_features
        sample_tree_batch = feature_output.sample_tree_batch
        keep_logit = self.node_keep_head(sample_tree_batch.x).squeeze(-1)
        cleave_logit = self.node_cleave_head(sample_tree_batch.x).squeeze(-1)
        ion_logit = self.ion_head(sample_tree_batch.x)
        unsaturation_logit = self.unsaturation_head(sample_tree_batch.x)
        radical_logit = self.radical_head(sample_tree_batch.x)
        return FragmentTreeCandidateSelectionOutput(
            features=features,
            sample_tree_batch=sample_tree_batch,
            keep_logit=keep_logit,
            cleave_logit=cleave_logit,
            ion_logit=ion_logit,
            unsaturation_logit=unsaturation_logit,
            radical_logit=radical_logit,
            ion_valid_mask_by_role_adduct=self.feature_model.ion_candidate_valid_mask_by_role_adduct,
            unsaturation_valid_mask_by_role_adduct=self.feature_model.unsaturation_candidate_valid_mask_by_role_adduct,
            radical_valid_mask_by_role_adduct=self.feature_model.radical_candidate_valid_mask_by_role_adduct,
            kept_candidates=self._select_fragment_ion_candidates(features=features, sample_tree_batch=sample_tree_batch, keep_logit=keep_logit, ion_logit=ion_logit, unsaturation_logit=unsaturation_logit, radical_logit=radical_logit),
            next_cleavage_candidates=self._select_next_cleavage_candidates(sample_tree_batch=sample_tree_batch, cleave_logit=cleave_logit),
        )

    def generate_depth_limited_candidates(self, data: Union[FragmentTreeStructure, FragmentTreeFeatures], *, max_depth: int) -> FragmentTreeCandidateSelectionOutput:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        output = self.forward(data)
        for _ in range(max_depth):
            output = self.forward(output.features)
        return output

    def _select_fragment_ion_candidates(self, *, features, sample_tree_batch, keep_logit: Tensor, ion_logit: Tensor, unsaturation_logit: Tensor, radical_logit: Tensor) -> List[FragmentIonCandidate]:
        structure = features.structure
        device = keep_logit.device
        node_is_precursor_root = sample_tree_batch.node_is_precursor_root.to(device).bool()
        non_precursor_node_index = (~node_is_precursor_root).nonzero(as_tuple=False).view(-1)
        if non_precursor_node_index.numel() == 0:
            return []
        node_score = keep_logit[non_precursor_node_index]
        k = int(node_score.numel()) if self.max_nodes_for_ion_candidates is None else min(int(self.max_nodes_for_ion_candidates), int(node_score.numel()))
        node_order = non_precursor_node_index[torch.topk(node_score, k=k).indices]
        kept_sample_ids = sample_tree_batch.kept_sample_ids.to(device).long()
        graph_index_by_node = sample_tree_batch.batch.to(device).long()
        candidates: List[FragmentIonCandidate] = []
        for batch_node_index_tensor in node_order:
            batch_node_index = int(batch_node_index_tensor.detach().cpu().item())
            graph_index = int(graph_index_by_node[batch_node_index].detach().cpu().item())
            candidates.extend(
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
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.max_fragment_ion_candidates]

    def _score_joint_ion_candidates_for_node(self, *, structure: FragmentTreeStructure, sample_tree_batch, batch_node_index: int, global_node_id: int, sample_id: int, keep_logit: Tensor, ion_logit: Tensor, unsaturation_logit: Tensor, radical_logit: Tensor) -> List[FragmentIonCandidate]:
        device = keep_logit.device
        base_formula = structure.node_formula[global_node_id].to(device).float()
        main_adduct_index = int(sample_tree_batch.node_main_adduct_type_index[batch_node_index].detach().cpu().item())
        role_index = int(bool(sample_tree_batch.node_is_precursor_root[batch_node_index].detach().cpu().item()))
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
        node_index = (~sample_tree_batch.node_is_precursor_root.to(device).bool()).nonzero(as_tuple=False).view(-1)
        if node_index.numel() == 0:
            return []
        selected_node_index = node_index[torch.topk(cleave_logit[node_index], k=min(self.max_next_cleavage_candidates, int(node_index.numel()))).indices]
        kept_sample_ids = sample_tree_batch.kept_sample_ids.to(device).long()
        graph_index_by_node = sample_tree_batch.batch.to(device).long()
        candidates: List[NextCleavageCandidate] = []
        for batch_node_index_tensor in selected_node_index:
            batch_node_index = int(batch_node_index_tensor.detach().cpu().item())
            graph_index = int(graph_index_by_node[batch_node_index].detach().cpu().item())
            score = cleave_logit[batch_node_index]
            candidates.append(NextCleavageCandidate(sample_id=int(kept_sample_ids[graph_index].detach().cpu().item()), batch_node_index=batch_node_index, global_node_id=int(sample_tree_batch.node_id_global[batch_node_index].detach().cpu().item()), score=float(score.detach().cpu().item()), probability=float(torch.sigmoid(score).detach().cpu().item())))
        return candidates

    @staticmethod
    def _valid_candidate_indices(mask_by_role_adduct: Tensor, *, role_index: int, main_adduct_index: int, device: torch.device) -> Tensor:
        mask = mask_by_role_adduct.to(device).bool()
        if main_adduct_index < 0 or main_adduct_index >= mask.size(1):
            raise IndexError(f"main_adduct_index={main_adduct_index} is out of range for mask shape {tuple(mask.shape)}.")
        return mask[int(role_index), int(main_adduct_index)].nonzero(as_tuple=False).view(-1)

