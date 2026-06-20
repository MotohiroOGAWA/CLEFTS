from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...libs.mmkit.mmkit import Formula
from .fragment_tree_candidate_selector import FragmentIonCandidate


@dataclass(frozen=True)
class FormulaIntensityPrediction:
    sample_id: int
    formula: Formula
    formula_tensor: Tensor
    mz: float
    intensity: float
    score: float
    candidates: List[FragmentIonCandidate]


@dataclass(frozen=True)
class FormulaIntensityOutput:
    formula_predictions: List[FormulaIntensityPrediction]


class FragmentTreeFormulaIntensityPredictor(nn.Module):
    """Predict final intensities for formula groups."""

    def __init__(self, formula_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(formula_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, formula_tensor: Tensor, group_score: Tensor, group_count: Optional[Tensor] = None) -> Tensor:
        if formula_tensor.dim() != 2:
            raise ValueError("formula_tensor must be 2D.")
        if group_score.dim() == 1:
            group_score = group_score[:, None]
        if group_count is None:
            group_count = torch.ones_like(group_score)
        elif group_count.dim() == 1:
            group_count = group_count[:, None]
        x = torch.cat([formula_tensor.float(), group_score.float(), group_count.float()], dim=-1)
        return F.softplus(self.net(x).squeeze(-1))

    def predict_from_candidates(self, candidates: List[FragmentIonCandidate]) -> FormulaIntensityOutput:
        if len(candidates) == 0:
            return FormulaIntensityOutput(formula_predictions=[])
        grouped: Dict[Tuple[int, str], List[FragmentIonCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault((candidate.sample_id, str(candidate.formula)), []).append(candidate)

        formula_rows: List[Tensor] = []
        scores: List[float] = []
        counts: List[float] = []
        keys: List[Tuple[int, str]] = []
        groups: List[List[FragmentIonCandidate]] = []
        for key, group in sorted(grouped.items(), key=lambda item: item[0]):
            score_tensor = torch.tensor([candidate.score for candidate in group], dtype=torch.float32)
            formula_rows.append(group[0].formula_tensor.float())
            scores.append(float(torch.logsumexp(score_tensor, dim=0).item()))
            counts.append(float(len(group)))
            keys.append(key)
            groups.append(group)

        device = next(self.parameters()).device
        formula_tensor = torch.stack(formula_rows, dim=0).to(device)
        score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
        count_tensor = torch.tensor(counts, dtype=torch.float32, device=device)
        intensities = self(formula_tensor, score_tensor, count_tensor)

        predictions: List[FormulaIntensityPrediction] = []
        for index, ((sample_id, _), group) in enumerate(zip(keys, groups)):
            formula = group[0].formula
            charge = int(getattr(formula, "charge", 0))
            mz = float(formula.exact_mass) if charge == 0 else float(formula.exact_mass) / abs(charge)
            predictions.append(
                FormulaIntensityPrediction(
                    sample_id=int(sample_id),
                    formula=formula,
                    formula_tensor=formula_tensor[index].detach().cpu(),
                    mz=mz,
                    intensity=float(intensities[index].detach().cpu().item()),
                    score=float(score_tensor[index].detach().cpu().item()),
                    candidates=group,
                )
            )
        return FormulaIntensityOutput(formula_predictions=predictions)


FormulaIntensityPredictor = FragmentTreeFormulaIntensityPredictor
