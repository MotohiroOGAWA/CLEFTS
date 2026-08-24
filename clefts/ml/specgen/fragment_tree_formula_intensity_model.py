from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...libs.mmkit.mmkit import Formula
from .fragment_tree_candidate_selector import (
    FragmentIonCandidate,
    FragmentTreeCandidateSelectionOutput,
)
from .fragment_tree_feature_model import FragmentTreeFeatureModel


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


@dataclass(frozen=True)
class FormulaIntensityTrainingOutput:
    sample_index: Tensor
    formula_tensor: Tensor
    # Sum of non-negative molecular-node -> formula-node edge scores.
    # The historical field name is retained for checkpoint/API compatibility.
    logit: Tensor
    candidates: List[List[FragmentIonCandidate]]
    presence_logit: Optional[Tensor] = None
    abundance_logit: Optional[Tensor] = None


class FragmentTreeFormulaIntensityPredictor(nn.Module):
    """Predict final intensities for formula nodes built from selected candidates."""

    def __init__(
        self,
        feature_model: FragmentTreeFeatureModel | int | None = None,
        *,
        formula_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_encoder_layers: int = 2,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()

        if isinstance(feature_model, int):
            if formula_dim is not None:
                raise ValueError("Specify either feature_model or formula_dim, not both.")
            formula_dim = int(feature_model)
            feature_model = None

        self.feature_model = feature_model
        if feature_model is None:
            if formula_dim is None:
                raise ValueError("feature_model is required unless formula_dim is provided for legacy use.")
            self.formula_dim = int(formula_dim)
            self.fragment_dim = None
            self.hidden_dim = int(hidden_dim or 128)
            self.formula_node_input = None
            self.ion_embedding = None
            self.unsaturation_embedding = None
            self.radical_embedding = None
            self.formula_node_encoder = None
            self.formula_node_head = None
            self.formula_presence_head = None
        else:
            self.formula_dim = int(feature_model.formula_tensorizer.dim)
            self.fragment_dim = int(feature_model.tree_encoder.hidden_dim)
            self.hidden_dim = int(hidden_dim or self.fragment_dim)
            self.formula_node_input = nn.Sequential(
                nn.Linear(
                    self.fragment_dim + self.formula_dim + self.hidden_dim * 3,
                    self.hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.ion_embedding = nn.Embedding(
                len(feature_model.ion_flat_candidates),
                self.hidden_dim,
            )
            self.unsaturation_embedding = nn.Embedding(
                len(feature_model.unsaturation_flat_candidates),
                self.hidden_dim,
            )
            self.radical_embedding = nn.Embedding(
                len(feature_model.radical_flat_candidates),
                self.hidden_dim,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=max(1, int(num_attention_heads)),
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True,
            )
            self.formula_node_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=max(1, int(num_encoder_layers)),
            )
            self.formula_node_head = nn.Linear(self.hidden_dim, 1)
            self.formula_presence_head = nn.Linear(self.hidden_dim, 1)

        self.net = nn.Sequential(
            nn.Linear(self.formula_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
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

    def forward_candidate_output(
        self,
        candidate_output: FragmentTreeCandidateSelectionOutput,
    ) -> FormulaIntensityTrainingOutput:
        if self.formula_node_input is None:
            raise RuntimeError(
                "FragmentTreeFormulaIntensityPredictor requires feature_model "
                "to train formula-node intensities."
            )
        candidates = candidate_output.kept_candidates
        if len(candidates) == 0:
            device = candidate_output.keep_logit.device
            return FormulaIntensityTrainingOutput(
                sample_index=torch.empty((0,), dtype=torch.long, device=device),
                formula_tensor=torch.empty((0, self.formula_dim), device=device),
                logit=candidate_output.keep_logit.new_empty((0,)),
                candidates=[],
                presence_logit=candidate_output.keep_logit.new_empty((0,)),
                abundance_logit=candidate_output.keep_logit.new_empty((0,)),
            )

        grouped: Dict[Tuple[int, Tuple[float, ...]], List[FragmentIonCandidate]] = {}
        for candidate in candidates:
            formula_key = tuple(float(value) for value in candidate.formula_tensor.tolist())
            grouped.setdefault((int(candidate.sample_id), formula_key), []).append(candidate)

        device = candidate_output.keep_logit.device
        sample_ids: List[int] = []
        formula_rows: List[Tensor] = []
        formula_node_rows: List[Tensor] = []
        groups: List[List[FragmentIonCandidate]] = []
        for (sample_id, _), group in sorted(grouped.items(), key=lambda item: item[0]):
            candidate_repr = self._candidate_formula_node_repr(candidate_output, group)
            formula_node_rows.append(candidate_repr.mean(dim=0))
            sample_ids.append(int(sample_id))
            formula_rows.append(group[0].formula_tensor.to(device).float())
            groups.append(group)

        sample_index = torch.tensor(sample_ids, dtype=torch.long, device=device)
        formula_node_repr = torch.stack(formula_node_rows, dim=0)
        encoded = self._encode_formula_nodes_by_sample(formula_node_repr, sample_index)
        presence_logit = self.formula_presence_head(encoded).squeeze(-1)
        abundance_logit = self.formula_node_head(encoded).squeeze(-1)
        intensity = self.relative_intensity(presence_logit, abundance_logit, sample_index)
        return FormulaIntensityTrainingOutput(
            sample_index=sample_index,
            formula_tensor=torch.stack(formula_rows, dim=0),
            logit=intensity,
            candidates=groups,
            presence_logit=presence_logit,
            abundance_logit=abundance_logit,
        )


    @staticmethod
    def relative_intensity(
        presence_logit: Tensor, abundance_logit: Tensor, sample_index: Tensor, eps: float = 1e-12
    ) -> Tensor:
        """Normalize presence-gated abundance independently per spectrum."""
        intensity = torch.zeros_like(abundance_logit)
        for sample_id in sample_index.detach().cpu().unique(sorted=True).tolist():
            mask = sample_index == int(sample_id)
            # Subtracting a shared maximum preserves the requested ratio while
            # preventing exp overflow/underflow for extreme abundance logits.
            stable_abundance = abundance_logit[mask] - abundance_logit[mask].max()
            raw_intensity = torch.sigmoid(presence_logit[mask]) * torch.exp(stable_abundance)
            intensity[mask] = raw_intensity / (raw_intensity.sum() + float(eps))
        return intensity

    def _candidate_formula_node_repr(
        self,
        candidate_output: FragmentTreeCandidateSelectionOutput,
        candidates: List[FragmentIonCandidate],
    ) -> Tensor:
        device = candidate_output.keep_logit.device
        fragment_rows = []
        for candidate in candidates:
            fragment_emb = candidate_output.sample_tree_batch.x[candidate.batch_node_index]
            formula_tensor = candidate.formula_tensor.to(device).float()
            state_emb = torch.cat(
                [
                    self.ion_embedding(
                        torch.tensor(candidate.ion_index, dtype=torch.long, device=device)
                    ),
                    self.unsaturation_embedding(
                        torch.tensor(
                            candidate.unsaturation_index, dtype=torch.long, device=device
                        )
                    ),
                    self.radical_embedding(
                        torch.tensor(candidate.radical_index, dtype=torch.long, device=device)
                    ),
                ],
                dim=0,
            )
            fragment_rows.append(
                self.formula_node_input(torch.cat([fragment_emb, formula_tensor, state_emb], dim=0))
            )
        return torch.stack(fragment_rows, dim=0)

    def _encode_formula_nodes_by_sample(self, node_repr: Tensor, sample_index: Tensor) -> Tensor:
        if node_repr.numel() == 0:
            return node_repr
        encoded = torch.empty_like(node_repr)
        for sample_id in sample_index.detach().cpu().unique(sorted=True).tolist():
            mask = sample_index == int(sample_id)
            encoded[mask] = self.formula_node_encoder(node_repr[mask][None, :, :]).squeeze(0)
        return encoded

    def predict_from_candidate_output(
        self,
        candidate_output: FragmentTreeCandidateSelectionOutput,
    ) -> FormulaIntensityOutput:
        if self.formula_node_input is None:
            return self.predict_from_candidates(candidate_output.kept_candidates)

        training_output = self.forward_candidate_output(candidate_output)
        if training_output.logit.numel() == 0:
            return FormulaIntensityOutput(formula_predictions=[])

        intensities = training_output.logit
        predictions: List[FormulaIntensityPrediction] = []
        for index, group in enumerate(training_output.candidates):
            if len(group) == 0:
                continue
            formula = group[0].formula
            charge = int(getattr(formula, "charge", 0))
            mz = float(formula.exact_mass) if charge == 0 else float(formula.exact_mass) / abs(charge)
            predictions.append(
                FormulaIntensityPrediction(
                    sample_id=int(training_output.sample_index[index].detach().cpu().item()),
                    formula=formula,
                    formula_tensor=training_output.formula_tensor[index].detach().cpu(),
                    mz=mz,
                    intensity=float(intensities[index].detach().cpu().item()),
                    score=float(training_output.logit[index].detach().cpu().item()),
                    candidates=group,
                )
            )
        return FormulaIntensityOutput(formula_predictions=predictions)

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
