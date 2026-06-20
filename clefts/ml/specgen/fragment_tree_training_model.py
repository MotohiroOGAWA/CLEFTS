from __future__ import annotations

from typing import List

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

    def forward(self, candidate_logit: Tensor, candidate_formula: Tensor, target_formula: Tensor) -> Tensor:
        if candidate_formula.dim() != 2 or target_formula.dim() != 2:
            raise ValueError("candidate_formula and target_formula must be 2D.")
        if candidate_logit.dim() != 1:
            raise ValueError("candidate_logit must be 1D.")
        if candidate_formula.size(0) != candidate_logit.size(0):
            raise ValueError("candidate_logit and candidate_formula are misaligned.")
        if candidate_formula.size(1) != target_formula.size(1):
            raise ValueError("Formula tensor widths must match.")
        losses: List[Tensor] = []
        for target in target_formula:
            mask = torch.all(candidate_formula == target, dim=1)
            if mask.any():
                losses.append(F.softplus(-torch.logsumexp(candidate_logit[mask], dim=0)))
            else:
                losses.append(candidate_logit.new_tensor(self.missing_formula_penalty))
        if len(losses) == 0:
            return candidate_logit.sum() * 0.0
        return torch.stack(losses).mean()


class FragmentTreeSelectionTrainingLoss(nn.Module):
    """Loss for direct keep/cleave/candidate selection training."""

    def __init__(self, missing_formula_penalty: float = 32.0) -> None:
        super().__init__()
        self.formula_group_loss = FormulaGroupCoverageLoss(missing_formula_penalty)

    def forward(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure) -> Tensor:
        device = output.keep_logit.device
        losses: List[Tensor] = []
        if target.target_node_keep.numel() > 0:
            losses.append(F.binary_cross_entropy_with_logits(output.keep_logit[: target.target_node_keep.numel()], target.target_node_keep.to(device).float()))
        if target.target_node_expand.numel() > 0:
            losses.append(F.binary_cross_entropy_with_logits(output.cleave_logit[: target.target_node_expand.numel()], target.target_node_expand.to(device).float()))
        if target.target_formula.numel() > 0:
            if len(output.kept_candidates) == 0:
                losses.append(output.keep_logit.new_tensor(self.formula_group_loss.missing_formula_penalty))
            else:
                candidate_score = torch.tensor([candidate.score for candidate in output.kept_candidates], dtype=torch.float32, device=device)
                candidate_formula = torch.stack([candidate.formula_tensor for candidate in output.kept_candidates], dim=0).to(device)
                losses.append(self.formula_group_loss(candidate_score, candidate_formula, target.target_formula.to(device)))
        if len(losses) == 0:
            return output.keep_logit.sum() * 0.0
        return torch.stack(losses).mean()


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
