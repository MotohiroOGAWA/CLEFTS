from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...libs.mmkit.mmkit import Formula
from ...libs.msentity.msentity import MSDataset
from ...libs.msentity.msentity.core.PeakSeries import PeakSeries
from ..input.fragment_tree_features import FragmentTreeFeatures
from ..input.fragment_tree_structure import FragmentTreeStructure
from .fragment_tree_probability_model import (
    FragmentTreeProbabilityModel,
    FragmentTreeProbabilityOutput,
)
from .fragment_spectrum_generator import FragmentSpectrumGeneratorOutput


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
    probability: float


@dataclass(frozen=True)
class NextCleavageCandidate:
    sample_id: int
    batch_node_index: int
    global_node_id: int
    probability: float


@dataclass(frozen=True)
class FragmentTreeCandidateGeneratorOutput:
    features: FragmentTreeFeatures
    probability_output: FragmentTreeProbabilityOutput
    kept_candidates: List[FragmentIonCandidate]
    next_cleavage_candidates: List[NextCleavageCandidate]


class FragmentTreeCandidateGenerator(nn.Module):
    """Select emitted fragment-ion candidates and next cleavage candidates."""

    def __init__(
        self,
        probability_model: FragmentTreeProbabilityModel,
        *,
        max_fragment_ion_candidates: int = 30,
        max_next_cleavage_candidates: int = 30,
        min_node_keep_probability: float = 0.0,
    ) -> None:
        super().__init__()

        if max_fragment_ion_candidates <= 0:
            raise ValueError("max_fragment_ion_candidates must be positive.")
        if max_next_cleavage_candidates <= 0:
            raise ValueError("max_next_cleavage_candidates must be positive.")

        self.probability_model = probability_model
        self.mol_encoder = probability_model.mol_encoder
        self.fragmenter = probability_model.fragmenter
        self.cleavage_edge_fnet = probability_model.cleavage_edge_fnet
        self.max_fragment_ion_candidates = int(max_fragment_ion_candidates)
        self.max_next_cleavage_candidates = int(max_next_cleavage_candidates)
        self.min_node_keep_probability = float(min_node_keep_probability)

    def forward(
        self,
        data: Union[FragmentTreeStructure, FragmentTreeFeatures],
    ) -> FragmentTreeCandidateGeneratorOutput:
        probability_output = self.probability_model(data)
        features = probability_output.ft_features
        sample_tree_batch = probability_output.sample_tree_batch
        probabilities = probability_output.probabilities

        kept_candidates: List[FragmentIonCandidate] = []
        next_candidates: List[NextCleavageCandidate] = []

        graph_index_by_node = sample_tree_batch.batch.long()
        kept_sample_ids = sample_tree_batch.kept_sample_ids.long()
        node_is_precursor_root = sample_tree_batch.node_is_precursor_root.bool()

        for batch_node_index in range(int(sample_tree_batch.num_nodes)):
            if bool(node_is_precursor_root[batch_node_index].detach().cpu().item()):
                continue

            node_keep_probability = float(
                probabilities.p_stop[batch_node_index].detach().cpu().item()
            )
            if node_keep_probability < self.min_node_keep_probability:
                continue

            graph_index = int(graph_index_by_node[batch_node_index].detach().cpu().item())
            sample_id = int(kept_sample_ids[graph_index].detach().cpu().item())
            global_node_id = int(
                sample_tree_batch.node_id_global[batch_node_index].detach().cpu().item()
            )

            kept_candidates.extend(
                self._top_fragment_ion_candidates_for_node(
                    features=features,
                    probability_output=probability_output,
                    batch_node_index=batch_node_index,
                    global_node_id=global_node_id,
                    sample_id=sample_id,
                    node_keep_probability=node_keep_probability,
                )
            )

            next_candidates.append(
                NextCleavageCandidate(
                    sample_id=sample_id,
                    batch_node_index=batch_node_index,
                    global_node_id=global_node_id,
                    probability=float(
                        probabilities.p_expand[batch_node_index].detach().cpu().item()
                    ),
                )
            )

        kept_candidates.sort(key=lambda item: item.probability, reverse=True)
        kept_candidates = kept_candidates[: self.max_fragment_ion_candidates]

        next_candidates.sort(key=lambda item: item.probability, reverse=True)
        next_candidates = next_candidates[: self.max_next_cleavage_candidates]

        return FragmentTreeCandidateGeneratorOutput(
            features=features,
            probability_output=probability_output,
            kept_candidates=kept_candidates,
            next_cleavage_candidates=next_candidates,
        )

    def generate_depth_limited_candidates(
        self,
        data: Union[FragmentTreeStructure, FragmentTreeFeatures],
        *,
        max_depth: int,
    ) -> FragmentTreeCandidateGeneratorOutput:
        """Run candidate selection repeatedly up to max_depth.

        The current structure builder owns actual graph expansion. This method
        preserves a stable API for the depth loop and returns the last selection
        from the supplied tree/features.
        """

        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")

        output = self.forward(data)
        for _ in range(max_depth):
            output = self.forward(output.features)
        return output

    def _top_fragment_ion_candidates_for_node(
        self,
        *,
        features: FragmentTreeFeatures,
        probability_output: FragmentTreeProbabilityOutput,
        batch_node_index: int,
        global_node_id: int,
        sample_id: int,
        node_keep_probability: float,
    ) -> List[FragmentIonCandidate]:
        probabilities = probability_output.probabilities
        structure = features.structure

        ion_indices = probabilities.ion.valid_mask[batch_node_index].nonzero(as_tuple=False).view(-1)
        unsaturation_indices = probabilities.unsaturation.valid_mask[batch_node_index].nonzero(as_tuple=False).view(-1)
        radical_indices = probabilities.radical.valid_mask[batch_node_index].nonzero(as_tuple=False).view(-1)

        rows: List[Tuple[int, int, int, float]] = []
        for ion_index_tensor in ion_indices:
            ion_index = int(ion_index_tensor.detach().cpu().item())
            p_ion = float(probabilities.ion.prob[batch_node_index, ion_index].detach().cpu().item())
            for unsaturation_index_tensor in unsaturation_indices:
                unsaturation_index = int(unsaturation_index_tensor.detach().cpu().item())
                p_unsaturation = float(probabilities.unsaturation.prob[batch_node_index, unsaturation_index].detach().cpu().item())
                for radical_index_tensor in radical_indices:
                    radical_index = int(radical_index_tensor.detach().cpu().item())
                    p_radical = float(probabilities.radical.prob[batch_node_index, radical_index].detach().cpu().item())
                    rows.append((ion_index, unsaturation_index, radical_index, node_keep_probability * p_ion * p_unsaturation * p_radical))

        rows.sort(key=lambda row: row[3], reverse=True)
        rows = rows[: self.max_fragment_ion_candidates]

        base_formula = structure.node_formula[global_node_id]
        tensorizer = self.probability_model.formula_tensorizer
        candidates: List[FragmentIonCandidate] = []
        for ion_index, unsaturation_index, radical_index, probability in rows:
            formula_tensor = (
                base_formula
                + structure.ion_formula_delta[ion_index]
                + structure.unsaturation_formula_delta[unsaturation_index]
                + structure.radical_formula_delta[radical_index]
            )
            formula = tensorizer.tensor_to_formula(formula_tensor)
            candidates.append(
                FragmentIonCandidate(
                    sample_id=sample_id,
                    batch_node_index=batch_node_index,
                    global_node_id=global_node_id,
                    ion_index=ion_index,
                    unsaturation_index=unsaturation_index,
                    radical_index=radical_index,
                    formula=formula,
                    formula_tensor=formula_tensor,
                    probability=float(probability),
                )
            )
        return candidates


class FormulaGroupCoverageLoss(nn.Module):
    """Ensure at least one candidate survives for each target formula group."""

    def forward(
        self,
        candidate_logit: Tensor,
        candidate_formula: Tensor,
        target_formula: Tensor,
    ) -> Tensor:
        if candidate_formula.dim() != 2 or target_formula.dim() != 2:
            raise ValueError("candidate_formula and target_formula must be 2D.")
        if candidate_logit.dim() != 1:
            raise ValueError("candidate_logit must be 1D.")
        if candidate_formula.size(0) != candidate_logit.size(0):
            raise ValueError("candidate_logit and candidate_formula are misaligned.")
        if candidate_formula.size(1) != target_formula.size(1):
            raise ValueError("Formula tensor widths must match.")

        losses = []
        for target in target_formula:
            mask = torch.all(candidate_formula == target, dim=1)
            if mask.any():
                group_logit = torch.logsumexp(candidate_logit[mask], dim=0)
                losses.append(F.softplus(-group_logit))
            else:
                losses.append(candidate_logit.new_tensor(32.0))
        if len(losses) == 0:
            return candidate_logit.sum() * 0.0
        return torch.stack(losses).mean()


class FormulaIntensityPredictor(nn.Module):
    """Predict intensity after formula-level grouping."""

    def __init__(self, formula_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(formula_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, formula_tensor: Tensor, candidate_probability: Tensor) -> Tensor:
        if candidate_probability.dim() == 1:
            candidate_probability = candidate_probability[:, None]
        x = torch.cat([formula_tensor.float(), candidate_probability.float()], dim=-1)
        return F.softplus(self.net(x).squeeze(-1))


def fragment_spectrum_output_to_msdataset(
    output: FragmentSpectrumGeneratorOutput,
    *,
    metadata: Optional[pd.DataFrame] = None,
    mz_column: str = "mz",
    intensity_column: str = "intensity",
) -> MSDataset:
    """Convert generated spectra to an MSDataset."""

    spectra = output.spectra
    if metadata is None:
        metadata = pd.DataFrame({"sample_id": [s.sample_id for s in spectra]})
    else:
        metadata = metadata.reset_index(drop=True).copy()

    if len(metadata) != len(spectra):
        raise ValueError("metadata rows must match generated spectra count.")

    peak_rows: List[Tuple[float, float]] = []
    peak_metadata_rows: List[Dict[str, object]] = []
    offsets = [0]

    for spectrum in spectra:
        for peak in spectrum.peaks:
            peak_rows.append((float(peak.mz), float(peak.intensity)))
            peak_metadata_rows.append(
                {
                    "sample_id": int(peak.sample_id),
                    "formula": peak.formula,
                }
            )
        offsets.append(len(peak_rows))

    data = np.asarray(peak_rows, dtype=np.float64)
    if data.size == 0:
        data = np.empty((0, 2), dtype=np.float64)

    peak_metadata = pd.DataFrame(peak_metadata_rows)
    peak_series = PeakSeries(
        data=data,
        offsets=np.asarray(offsets, dtype=np.int64),
        metadata=peak_metadata,
        metadata_columns=["sample_id", "formula"],
        sort_by_mz=True,
    )

    return MSDataset(
        spectrum_metadata=metadata,
        peak_series=peak_series,
        columns=metadata.columns.tolist(),
        description="In silico spectra generated by CLEFTS",
    )
