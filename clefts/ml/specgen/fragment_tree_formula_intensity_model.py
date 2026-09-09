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
from ..common.torch_utils.segment_ops import segment_logsumexp


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
    """Predict candidate intensities, then merge candidates with equal formulas."""

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
            self.candidate_edge_input = None
        else:
            self.formula_dim = int(feature_model.formula_tensorizer.dim)
            self.fragment_dim = int(feature_model.tree_encoder.hidden_dim)
            self.hidden_dim = int(hidden_dim or self.fragment_dim)
            self.formula_node_input = nn.Sequential(
                nn.Linear(
                    self.fragment_dim + self.hidden_dim * 4,
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
            edge_dim = int(feature_model.fragment_edge_encoder.feature_dim)
            self.candidate_edge_input = nn.Sequential(
                nn.LayerNorm(edge_dim + 2),
                nn.Linear(edge_dim + 2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
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
            nn.Linear(2, self.hidden_dim),
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
        # Keep formula_tensor in the public signature for compatibility, but
        # never expose formula identity/composition to the intensity network.
        x = torch.cat([group_score.float(), group_count.float()], dim=-1)
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

        device = candidate_output.keep_logit.device
        candidate_sample_index = torch.tensor(
            [int(candidate.sample_id) for candidate in candidates],
            dtype=torch.long,
            device=device,
        )
        candidate_repr = self._candidate_formula_node_repr(candidate_output, candidates)
        encoded = self._encode_formula_nodes_by_sample(candidate_repr, candidate_sample_index)
        candidate_presence_logit = self.formula_presence_head(encoded).squeeze(-1)
        candidate_abundance_logit = self.formula_node_head(encoded).squeeze(-1)
        selection_probability = torch.stack(
            [self._candidate_probability(candidate, device=device) for candidate in candidates]
        )
        candidate_intensity = self.relative_intensity(
            candidate_presence_logit,
            candidate_abundance_logit,
            candidate_sample_index,
            selection_probability=selection_probability,
        )

        # Formula identity is deliberately used only after intensity prediction.
        grouped_indexes: Dict[Tuple[int, Tuple[float, ...]], List[int]] = {}
        for index, candidate in enumerate(candidates):
            formula_key = tuple(float(value) for value in candidate.formula_tensor.tolist())
            grouped_indexes.setdefault((int(candidate.sample_id), formula_key), []).append(index)

        ordered_groups = sorted(grouped_indexes.items(), key=lambda item: item[0])
        group_ids = [0] * len(candidates)
        groups = []
        for group_id, (_, indexes) in enumerate(ordered_groups):
            groups.append([candidates[i] for i in indexes])
            for i in indexes:
                group_ids[i] = group_id
        inverse = torch.tensor(group_ids, dtype=torch.long, device=device)
        size = len(groups)
        def summed(values):
            return values.new_zeros(size).index_add(0, inverse, values)
        denominator = summed(selection_probability).clamp_min(1e-12)
        presence = summed(torch.sigmoid(candidate_presence_logit) * selection_probability) / denominator
        return FormulaIntensityTrainingOutput(
            sample_index=torch.tensor([key[0] for key, _ in ordered_groups], dtype=torch.long, device=device),
            formula_tensor=torch.stack([group[0].formula_tensor for group in groups]).to(device).float(),
            logit=summed(candidate_intensity), candidates=groups,
            presence_logit=torch.logit(presence.clamp(1e-6, 1.0 - 1e-6)),
            abundance_logit=summed(candidate_abundance_logit * selection_probability) / denominator,
        )


    @staticmethod
    def relative_intensity(
        presence_logit: Tensor,
        abundance_logit: Tensor,
        sample_index: Tensor,
        eps: float = 1e-12,
        selection_probability: Optional[Tensor] = None,
    ) -> Tensor:
        """Normalize selection- and presence-gated candidate abundance."""
        if abundance_logit.numel() == 0:
            return abundance_logit
        if selection_probability is None:
            selection_probability = torch.ones_like(abundance_logit)
        samples, groups = torch.unique(sample_index, sorted=True, return_inverse=True)
        maximum = abundance_logit.new_full((samples.numel(),), -torch.inf)
        maximum.scatter_reduce_(0, groups, abundance_logit.detach(), reduce='amax', include_self=True)
        log_raw = (selection_probability.clamp_min(float(eps)).log()
                   + F.logsigmoid(presence_logit) + abundance_logit - maximum[groups])
        return (log_raw - segment_logsumexp(log_raw, groups, samples.numel())[groups]).exp()

    @staticmethod
    def _candidate_probability(candidate: FragmentIonCandidate, *, device: torch.device) -> Tensor:
        probability_tensor = getattr(candidate, "probability_tensor", None)
        if probability_tensor is not None:
            return probability_tensor.to(device).float()
        return torch.tensor(float(candidate.probability), dtype=torch.float32, device=device)

    def _candidate_formula_node_repr(self, candidate_output, candidates) -> Tensor:
        batch = candidate_output.sample_tree_batch
        device = candidate_output.keep_logit.device
        metadata = torch.tensor([(c.batch_node_index, c.ion_index, c.unsaturation_index, c.radical_index)
                                 for c in candidates], dtype=torch.long, device=device)
        nodes, ions, unsaturations, radicals = metadata.unbind(dim=1)
        state_emb = torch.cat((self.ion_embedding(ions), self.unsaturation_embedding(unsaturations),
                               self.radical_embedding(radicals)), dim=1)
        edge_dst = batch.edge_index[1].to(device).long()
        edge_values = torch.cat((batch.edge_attr, candidate_output.edge_absolute_logit[:, None],
                                  candidate_output.edge_cleave_logit[:, None]), dim=1)
        sums = edge_values.new_zeros((batch.x.size(0), edge_values.size(1))).index_add(0, edge_dst, edge_values)
        count = torch.bincount(edge_dst, minlength=batch.x.size(0)).clamp_min(1)
        edge_repr = self.candidate_edge_input((sums / count[:, None])[nodes])
        return self.formula_node_input(torch.cat((batch.x[nodes], state_emb, edge_repr), dim=1))

    def _encode_formula_nodes_by_sample(self, node_repr: Tensor, sample_index: Tensor) -> Tensor:
        if node_repr.numel() == 0:
            return node_repr
        samples, groups, counts = torch.unique(sample_index, sorted=True, return_inverse=True, return_counts=True)
        order = torch.argsort(groups, stable=True)
        starts = counts.cumsum(0) - counts
        positions = torch.empty_like(groups)
        positions[order] = torch.arange(order.numel(), device=order.device) - starts[groups[order]]
        width = int(counts.max().item())
        padded = node_repr.new_zeros((samples.numel(), width, node_repr.size(1)))
        padded[groups, positions] = node_repr
        mask = torch.ones((samples.numel(), width), dtype=torch.bool, device=node_repr.device)
        mask[groups, positions] = False
        encoded = self.formula_node_encoder(padded, src_key_padding_mask=mask)
        return encoded[groups, positions]

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
