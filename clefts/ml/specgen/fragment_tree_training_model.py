from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..input.training_fragment_tree_structure import TrainingFragmentTreeStructure
from .fragment_tree_candidate_selector import FragmentTreeCandidateSelectionOutput


class FormulaGroupCoverageLoss(nn.Module):
    """Require at least one selected candidate for each target formula group."""

    def __init__(self, missing_formula_penalty: float = 32.0) -> None:
        super().__init__()
        self.missing_formula_penalty = float(missing_formula_penalty)

    def forward(
        self,
        candidate_logit: Tensor,
        candidate_formula: Tensor,
        target_formula: Tensor,
        *,
        candidate_sample_index: Optional[Tensor] = None,
        target_sample_index: Optional[Tensor] = None,
        target_group_index: Optional[Tensor] = None,
        target_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if candidate_formula.dim() != 2 or target_formula.dim() != 2:
            raise ValueError("candidate_formula and target_formula must be 2D.")
        if candidate_logit.dim() != 1:
            raise ValueError("candidate_logit must be 1D.")
        if candidate_formula.size(0) != candidate_logit.size(0):
            raise ValueError("candidate_logit and candidate_formula are misaligned.")
        if candidate_formula.size(1) != target_formula.size(1):
            raise ValueError("Formula tensor widths must match.")

        device = candidate_logit.device
        target_formula = target_formula.to(device)

        if candidate_sample_index is not None:
            candidate_sample_index = candidate_sample_index.to(device).long()
        if target_sample_index is not None:
            target_sample_index = target_sample_index.to(device).long()
        if target_group_index is None:
            target_group_index = torch.arange(
                target_formula.size(0), dtype=torch.long, device=device
            )
        else:
            target_group_index = target_group_index.to(device).long()
        if target_weight is not None:
            target_weight = target_weight.to(device).float()

        group_keys = self._make_group_keys(
            target_sample_index=target_sample_index,
            target_group_index=target_group_index,
            num_targets=int(target_formula.size(0)),
        )

        losses: List[Tensor] = []
        weights: List[Tensor] = []

        for group_key in group_keys:
            target_mask = self._group_mask(
                group_key=group_key,
                target_sample_index=target_sample_index,
                target_group_index=target_group_index,
                num_targets=int(target_formula.size(0)),
                device=device,
            )
            group_target_formula = target_formula[target_mask]

            candidate_mask = torch.zeros(
                (candidate_formula.size(0),), dtype=torch.bool, device=device
            )
            if candidate_sample_index is not None and target_sample_index is not None:
                sample_id = int(group_key[0])
                candidate_mask |= candidate_sample_index == sample_id
            else:
                candidate_mask |= True

            formula_mask = torch.zeros_like(candidate_mask)
            for formula in group_target_formula:
                formula_mask |= torch.all(candidate_formula.to(device) == formula, dim=1)
            candidate_mask &= formula_mask

            if candidate_mask.any():
                losses.append(
                    F.softplus(-torch.logsumexp(candidate_logit[candidate_mask], dim=0))
                )
            else:
                losses.append(candidate_logit.new_tensor(self.missing_formula_penalty))

            if target_weight is not None:
                weights.append(target_weight[target_mask].max())

        if len(losses) == 0:
            return candidate_logit.sum() * 0.0

        loss_tensor = torch.stack(losses)
        if target_weight is None:
            return loss_tensor.mean()

        weight_tensor = torch.stack(weights).clamp_min(0.0)
        if float(weight_tensor.sum().detach().cpu().item()) <= 0.0:
            return loss_tensor.mean()
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()

    @staticmethod
    def _make_group_keys(
        *,
        target_sample_index: Optional[Tensor],
        target_group_index: Tensor,
        num_targets: int,
    ) -> List[Tuple[int, int]]:
        keys: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for target_index in range(num_targets):
            sample_key = (
                int(target_sample_index[target_index].item())
                if target_sample_index is not None
                else 0
            )
            group_key = int(target_group_index[target_index].item())
            key = (sample_key, group_key)
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    @staticmethod
    def _group_mask(
        *,
        group_key: Tuple[int, int],
        target_sample_index: Optional[Tensor],
        target_group_index: Tensor,
        num_targets: int,
        device: torch.device,
    ) -> Tensor:
        mask = target_group_index == int(group_key[1])
        if target_sample_index is not None:
            mask &= target_sample_index == int(group_key[0])
        if mask.numel() != num_targets:
            raise ValueError("group mask is misaligned with target rows.")
        return mask.to(device)


class FragmentTreeSelectionTrainingLoss(nn.Module):
    """Loss for keep/cleave and grouped formula target training."""

    def __init__(self, missing_formula_penalty: float = 32.0) -> None:
        super().__init__()
        self.formula_group_loss = FormulaGroupCoverageLoss(missing_formula_penalty)

    def forward(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
    ) -> Tensor:
        device = output.keep_logit.device
        losses: List[Tensor] = []

        keep_target, keep_mask = self._build_keep_targets(output, target, device=device)
        if keep_mask.any():
            losses.append(
                F.binary_cross_entropy_with_logits(
                    output.keep_logit[keep_mask],
                    keep_target[keep_mask],
                )
            )

        cleave_target, cleave_mask = self._build_cleave_targets(output, target, device=device)
        if cleave_mask.any():
            losses.append(
                F.binary_cross_entropy_with_logits(
                    output.cleave_logit[cleave_mask],
                    cleave_target[cleave_mask],
                )
            )

        if target.target_formula.numel() > 0:
            losses.append(self._formula_group_loss(output, target, device=device))

        if len(losses) == 0:
            return output.keep_logit.sum() * 0.0
        return torch.stack(losses).mean()

    def _formula_group_loss(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
    ) -> Tensor:
        if len(output.kept_candidates) == 0:
            return output.keep_logit.new_tensor(
                self.formula_group_loss.missing_formula_penalty
            )

        candidate_logit = torch.stack(
            [
                candidate.score_tensor.to(device)
                if candidate.score_tensor is not None
                else output.keep_logit.new_tensor(float(candidate.score))
                for candidate in output.kept_candidates
            ],
            dim=0,
        )
        candidate_formula = torch.stack(
            [candidate.formula_tensor for candidate in output.kept_candidates], dim=0
        ).to(device)
        candidate_sample_index = torch.tensor(
            [candidate.sample_id for candidate in output.kept_candidates],
            dtype=torch.long,
            device=device,
        )

        target_group_index = getattr(target, "target_formula_group_index", None)
        if target_group_index is None:
            target_group_index = torch.arange(
                target.target_formula.size(0), dtype=torch.long, device=device
            )

        return self.formula_group_loss(
            candidate_logit,
            candidate_formula,
            target.target_formula.to(device),
            candidate_sample_index=candidate_sample_index,
            target_sample_index=target.target_sample_index.to(device),
            target_group_index=target_group_index.to(device),
            target_weight=target.target_intensity.to(device),
        )

    def _build_keep_targets(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        batch = output.sample_tree_batch
        keep_target = torch.zeros_like(output.keep_logit, dtype=torch.float32, device=device)
        keep_mask = ~batch.node_is_precursor_root.to(device).bool()

        if target.target_node_index.numel() == 0:
            return keep_target, keep_mask

        positive_pairs = self._target_sample_node_pairs(
            target_sample_index=target.target_sample_index.to(device),
            target_node_index=target.target_node_index.to(device),
        )
        self._apply_positive_node_pairs(
            target_tensor=keep_target,
            positive_pairs=positive_pairs,
            batch=batch,
            device=device,
        )
        return keep_target, keep_mask

    def _build_cleave_targets(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        batch = output.sample_tree_batch
        cleave_target = torch.zeros_like(output.cleave_logit, dtype=torch.float32, device=device)
        cleave_mask = ~batch.node_is_precursor_root.to(device).bool()

        if target.target_edge_index.numel() == 0:
            return cleave_target, cleave_mask

        target_edge_index = target.target_edge_index.to(device).long()
        target_sample_index = target_edge_index[0]
        target_global_edge_index = target_edge_index[1]
        target_global_src_node = target.edge_index.to(device).long()[0, target_global_edge_index]
        positive_pairs = self._target_sample_node_pairs(
            target_sample_index=target_sample_index,
            target_node_index=target_global_src_node,
        )
        self._apply_positive_node_pairs(
            target_tensor=cleave_target,
            positive_pairs=positive_pairs,
            batch=batch,
            device=device,
        )
        return cleave_target, cleave_mask

    @staticmethod
    def _target_sample_node_pairs(
        *,
        target_sample_index: Tensor,
        target_node_index: Tensor,
    ) -> set[Tuple[int, int]]:
        return {
            (int(sample_id), int(node_id))
            for sample_id, node_id in zip(
                target_sample_index.detach().cpu().tolist(),
                target_node_index.detach().cpu().tolist(),
            )
        }

    @staticmethod
    def _apply_positive_node_pairs(
        *,
        target_tensor: Tensor,
        positive_pairs: set[Tuple[int, int]],
        batch,
        device: torch.device,
    ) -> None:
        kept_sample_ids = batch.kept_sample_ids.to(device).long()
        graph_index_by_node = batch.batch.to(device).long()
        node_global_ids = batch.node_id_global.to(device).long()

        for batch_node_index in range(int(target_tensor.numel())):
            graph_index = int(graph_index_by_node[batch_node_index].item())
            sample_id = int(kept_sample_ids[graph_index].item())
            node_id = int(node_global_ids[batch_node_index].item())
            if (sample_id, node_id) in positive_pairs:
                target_tensor[batch_node_index] = 1.0


class FragmentTreeTrainingModel(nn.Module):
    """Training wrapper that combines candidate selection and supervised losses."""

    def __init__(self, candidate_selector: nn.Module, loss_fn: nn.Module | None = None) -> None:
        super().__init__()
        self.candidate_selector = candidate_selector
        self.loss_fn = loss_fn if loss_fn is not None else FragmentTreeSelectionTrainingLoss()

    def forward(self, batch: TrainingFragmentTreeStructure):
        output = self.candidate_selector(batch)
        loss = self.loss_fn(output, batch)
        return {"loss": loss, "candidate_output": output}
