from __future__ import annotations


from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...input.training_fragment_tree_structure import TrainingFragmentTreeStructure
from ...specgen.fragment_tree_candidate_selector import FragmentTreeCandidateSelectionOutput
from ...specgen.fragment_tree_formula_intensity_model import FragmentTreeFormulaIntensityPredictor
from . import metric_labels
from .depth_metrics import DepthEvaluation


class PairwiseEdgeIntensityRankingLoss(nn.Module):
    """Rank independently scored edges by observed peak intensity.

    For each intensity-sorted anchor (group, in the gradient path; edge, in
    the diagnostic path), up to ``top_n`` comparison partners are selected in
    three tiers, filled in order until ``top_n`` is reached:

    1. the ``nearest_lower_partners`` *nearest* lower-intensity partners;
    2. ``extended_lower_partners`` *more* lower-intensity partners, farther away in the sorted
       order;
    3. ``background_partners`` partners with no intensity at all (unassigned/background
       edges), individually compared against the anchor.

    Tiers 1/2 skip (without consuming budget) any candidate whose
    ``sqrt(intensity)`` gap to the anchor is ``<= intensity_threshold``.  This
    retains RankNet's ordering objective without constructing the quadratic
    set of every possible pair.  Each pair is weighted by a reciprocal-rank
    weight ``(1/rank_i) / sum(1/rank_n)`` computed over the sample's
    intensity-sorted groups (rank 1 = most intense), not by the raw intensity
    value itself; intensity is still used to sort groups and to gate
    near-equal-intensity pairs.
    """

    def __init__(
        self,
        top_n: int = 10,
        nearest_lower_partners: int = 1,
        extended_lower_partners: int = 3,
        background_partners: int = 10,
        intensity_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        if top_n < 1:
            raise ValueError("top_n must be positive.")
        if nearest_lower_partners < 0 or extended_lower_partners < 0 or background_partners < 0:
            raise ValueError("nearest_lower_partners, extended_lower_partners, and background_partners must be non-negative.")
        if intensity_threshold < 0:
            raise ValueError("intensity_threshold must be non-negative.")
        self.top_n = int(top_n)
        self.nearest_lower_partners = int(nearest_lower_partners)
        self.extended_lower_partners = int(extended_lower_partners)
        self.background_partners = int(background_partners)
        self.intensity_threshold = float(intensity_threshold)

    def _select_tiered_partners(
        self,
        *,
        sqrt_values: Tensor,
        anchor_index: int,
        remaining_budget: int,
    ) -> List[int]:
        """Tier 1+2 offsets ``j > 0`` such that ``sqrt_values[anchor_index + j]``
        is a selected lower-intensity partner, nearest-first.  ``sqrt_values``
        must already be sorted descending.  A candidate whose gap to the
        anchor is ``<= intensity_threshold`` is skipped without consuming
        budget, so the scan continues past it.
        """
        offsets: List[int] = []
        budget = min(self.nearest_lower_partners + self.extended_lower_partners, max(int(remaining_budget), 0))
        if budget <= 0:
            return offsets
        anchor_value = float(sqrt_values[anchor_index])
        n = int(sqrt_values.numel())
        j = 1
        while len(offsets) < budget and anchor_index + j < n:
            candidate_value = float(sqrt_values[anchor_index + j])
            if anchor_value - candidate_value > self.intensity_threshold:
                offsets.append(j)
            j += 1
        return offsets

    def build_pairs(self, output, target) -> Tuple[Tensor, Tensor]:
        """Build tiered pairs within the same sample and source fragment
        (diagnostic only; mirrors forward()'s tier1/tier2/tier3 selection)."""
        logit = output.edge_absolute_logit
        batch = output.sample_tree_batch
        edge_ids = batch.edge_id_global.to(logit.device).long()
        edge_graph = batch.batch[batch.edge_index[0]].to(logit.device).long()
        sample_ids = batch.kept_sample_ids.to(logit.device).long()[edge_graph]
        node_ids = getattr(batch, "node_id_global", None)
        if node_ids is None:
            # Lightweight/legacy batches still preserve source identity via
            # their local edge index.
            source_ids = batch.edge_index[0].to(logit.device).long()
        else:
            source_ids = node_ids.to(logit.device).long()[batch.edge_index[0]]
        target_pairs = target.target_edge_index.to(logit.device).long()
        target_groups = target.target_edge_group_index.to(logit.device).long()
        formula_peak = target.formula_peak_index.to(logit.device).long()
        peak_intensity = target.sample_peak_intensity.to(logit.device).float()
        better_rows: List[int] = []
        worse_rows: List[int] = []

        for sample_id in sample_ids.unique(sorted=True).tolist():
            rows = (target_pairs[0] == int(sample_id)).nonzero(as_tuple=False).flatten()
            sample_mask = sample_ids == int(sample_id)
            intensity_by_row: Dict[int, float] = {}
            alternative_rows: set[int] = set()
            # A formula/mz group can have several valid explanatory edges.
            # Treat the current highest-scoring alternative as the latent
            # representative instead of forcing every alternative positive.
            for group_id in target_groups[rows].unique(sorted=True).tolist():
                group_rows = rows[target_groups[rows] == int(group_id)]
                group_edge_ids = target_pairs[1, group_rows].unique()
                candidates = (
                    sample_mask
                    & torch.isin(edge_ids, group_edge_ids)
                ).nonzero(as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                alternative_rows.update(int(value) for value in candidates.tolist())
                representative = int(candidates[torch.argmax(logit[candidates])].item())
                intensity_by_row[representative] = float(
                    peak_intensity[formula_peak[int(group_id)]].item()
                )
            for source_id in source_ids[sample_mask].unique(sorted=True).tolist():
                source_rows = (sample_mask & (source_ids == int(source_id))).nonzero(as_tuple=False).flatten().tolist()
                real_rows = torch.tensor(
                    [index for index in source_rows if int(index) in intensity_by_row],
                    dtype=torch.long, device=logit.device,
                )
                background_rows = torch.tensor(
                    [index for index in source_rows if int(index) not in alternative_rows],
                    dtype=torch.long, device=logit.device,
                )
                if real_rows.numel() == 0:
                    continue
                sqrt_values = torch.sqrt(
                    logit.new_tensor([intensity_by_row[int(index)] for index in real_rows]).clamp_min(0.0)
                )
                order = torch.argsort(sqrt_values, descending=True, stable=True)
                ordered_real, ordered_values = real_rows[order], sqrt_values[order]
                hard_negatives = (
                    background_rows[
                        torch.topk(
                            logit[background_rows], k=min(self.background_partners, int(background_rows.numel()))
                        ).indices
                    ]
                    if background_rows.numel()
                    else background_rows
                )
                for i in range(int(ordered_real.numel())):
                    offsets = self._select_tiered_partners(
                        sqrt_values=ordered_values, anchor_index=i, remaining_budget=self.top_n,
                    )
                    for j in offsets:
                        better_rows.append(int(ordered_real[i]))
                        worse_rows.append(int(ordered_real[i + j]))
                    remaining_after_12 = self.top_n - len(offsets)
                    tier3_count = min(self.background_partners, remaining_after_12, int(hard_negatives.numel()))
                    for k in range(tier3_count):
                        better_rows.append(int(ordered_real[i]))
                        worse_rows.append(int(hard_negatives[k]))
        return (
            torch.tensor(better_rows, dtype=torch.long, device=logit.device),
            torch.tensor(worse_rows, dtype=torch.long, device=logit.device),
        )

    def forward(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
    ) -> Tensor:
        logit = output.edge_absolute_logit
        if logit.numel() == 0:
            return logit.sum() * 0.0
        if not hasattr(output.sample_tree_batch, "edge_id_global"):
            raise ValueError("sample_tree_batch must expose edge_id_global.")
        batch = output.sample_tree_batch
        edge_ids = batch.edge_id_global.to(logit.device).long()
        edge_graph = batch.batch[batch.edge_index[0]].to(logit.device).long()
        sample_ids = batch.kept_sample_ids.to(logit.device).long()[edge_graph]
        target_edges = target.target_edge_index.to(logit.device).long()
        target_groups = target.target_edge_group_index.to(logit.device).long()
        formula_peak = target.formula_peak_index.to(logit.device).long()
        intensities = target.sample_peak_intensity.to(logit.device).float().clamp_min(0)
        losses: List[Tensor] = []
        weights: List[Tensor] = []
        for sample_id in sample_ids.unique(sorted=True).tolist():
            rows = (target_edges[0] == int(sample_id)).nonzero(as_tuple=False).flatten()
            group_rows = []
            alternatives: set[int] = set()
            for group_id in target_groups[rows].unique(sorted=True).tolist():
                members = rows[target_groups[rows] == int(group_id)]
                ids = target_edges[1, members].unique()
                candidate = (sample_ids == int(sample_id)) & torch.isin(edge_ids, ids)
                indexes = candidate.nonzero(as_tuple=False).flatten()
                if not indexes.numel():
                    continue
                alternatives.update(int(v) for v in indexes.tolist())
                # logsumexp is log(sum(exp(score))): the requested sum of all
                # alternative absolute evidences without rewarding group size
                # through a forced all-positive label.
                evidence = torch.logsumexp(logit[indexes], dim=0)
                intensity = intensities[formula_peak[int(group_id)]]
                group_rows.append((evidence, intensity))
            group_rows.sort(key=lambda item: float(item[1]), reverse=True)
            # Reciprocal-rank weight (1/rank_i) / sum(1/rank_n), normalized
            # within this sample's intensity-sorted groups: rank 1 (the most
            # intense group) carries the most weight, regardless of which
            # tier a given partner came from.
            rank_weights = logit.new_empty((0,))
            sqrt_values = logit.new_empty((0,))
            if group_rows:
                ranks = torch.arange(
                    1, len(group_rows) + 1, dtype=torch.float32, device=logit.device
                )
                reciprocal_ranks = 1.0 / ranks
                rank_weights = reciprocal_ranks / reciprocal_ranks.sum()
                sqrt_values = torch.sqrt(
                    torch.stack([item[1] for item in group_rows]).clamp_min(0.0)
                )
            # Observed formula groups also compete against unassigned edges,
            # including edges at the same depth.  Bound this comparison count.
            background = torch.tensor([
                index for index in (sample_ids == int(sample_id)).nonzero(as_tuple=False).flatten().tolist()
                if int(index) not in alternatives
            ], dtype=torch.long, device=logit.device)
            hard_negatives = (
                background[torch.topk(logit[background], k=min(self.background_partners, int(background.numel()))).indices]
                if background.numel() and group_rows
                else background[:0]
            )
            for i in range(len(group_rows)):
                offsets = self._select_tiered_partners(
                    sqrt_values=sqrt_values, anchor_index=i, remaining_budget=self.top_n,
                )
                for j in offsets:
                    losses.append(F.softplus(-(group_rows[i][0] - group_rows[i + j][0])))
                    weights.append(rank_weights[i])
                if float(group_rows[i][1]) <= 0:
                    continue
                remaining_after_12 = self.top_n - len(offsets)
                tier3_count = min(self.background_partners, remaining_after_12, int(hard_negatives.numel()))
                for k in range(tier3_count):
                    losses.append(F.softplus(-(group_rows[i][0] - logit[hard_negatives[k]])))
                    weights.append(rank_weights[i])
        if not losses:
            return logit.sum() * 0.0
        loss_values = torch.stack(losses)
        loss_weights = torch.stack(weights)
        return (loss_values * loss_weights).sum() / loss_weights.sum().clamp_min(1e-12)

    @torch.no_grad()
    def metrics(self, output, target) -> Dict[str, float]:
        loss = float(self(output, target).detach().cpu().item())
        better, worse = self.build_pairs(output, target)
        accuracy = (
            float((output.edge_absolute_logit[better] > output.edge_absolute_logit[worse]).float().mean().item())
            if better.numel() else 0.0
        )
        return {"edge_ranking_loss": loss, "pairwise_ranking_accuracy": accuracy}


class FragmentEdgeAbsoluteRankerTrainingLoss(nn.Module):
    """Direct supervision for the cheap, condition-independent edge selector."""

    def __init__(self, adduct_labels: Optional[Dict[int, str]] = None) -> None:
        super().__init__()
        self.adduct_labels = dict(adduct_labels or {})

    @staticmethod
    def _tensorboard_label(value: str) -> str:
        return value.strip().replace("/", "∕").replace(" ", "_")

    def forward(self, edge_output, target: TrainingFragmentTreeStructure):
        logits = edge_output.absolute_score_logit
        finite = torch.isfinite(logits)
        target_pairs = target.target_edge_index.to(logits.device).long()
        target_groups = target.target_edge_group_index.to(logits.device).long()
        positive_ids = target_pairs[1].unique(sorted=True) if target_pairs.numel() else logits.new_empty((0,), dtype=torch.long)
        positive_ids = positive_ids[(positive_ids >= 0) & (positive_ids < logits.numel())]
        negative = finite.clone()
        if positive_ids.numel():
            negative[positive_ids] = False
        components = []
        # Multiple edges in one (sample, formula/mz) group are alternatives:
        # logsumexp is a smooth OR, so at least one explanation must score high.
        group_losses = []
        if target_pairs.numel():
            for sample_id in target_pairs[0].unique(sorted=True).tolist():
                sample_rows = target_pairs[0] == int(sample_id)
                for group_id in target_groups[sample_rows].unique(sorted=True).tolist():
                    ids = target_pairs[1, sample_rows & (target_groups == int(group_id))]
                    ids = ids[(ids >= 0) & (ids < logits.numel())]
                    ids = ids[finite[ids]].unique()
                    if ids.numel():
                        group_losses.append(F.softplus(-torch.logsumexp(logits[ids], dim=0)))
        if group_losses:
            components.append(torch.stack(group_losses).mean())
        if negative.any():
            components.append(
                F.binary_cross_entropy_with_logits(
                    logits[negative], torch.zeros_like(logits[negative])
                )
            )
        loss = torch.stack(components).mean() if components else logits[finite].sum() * 0.0
        return loss, self.metrics(edge_output, target)

    def metrics(self, edge_output, target: TrainingFragmentTreeStructure) -> Dict[str, float]:
        selected = set(edge_output.selected_edge_index.detach().cpu().tolist())
        target_edges = target.target_edge_index.detach().cpu().long()
        unique_edges = set(target_edges[1].tolist()) if target_edges.numel() else set()
        edge_recall = (
            len(unique_edges & selected) / len(unique_edges) if unique_edges else 1.0
        )

        group_total = 0
        group_kept = 0
        if target_edges.numel():
            groups = target.target_edge_group_index.detach().cpu().long()
            for sample_id in target_edges[0].unique(sorted=True).tolist():
                sample_mask = target_edges[0] == int(sample_id)
                for group_id in groups[sample_mask].unique(sorted=True).tolist():
                    mask = sample_mask & (groups == int(group_id))
                    group_edge_ids = set(target_edges[1, mask].tolist())
                    group_total += 1
                    group_kept += int(bool(group_edge_ids & selected))
        group_recall = group_kept / group_total if group_total else 1.0

        edge_dst = target.edge_index[1].detach().cpu().long()
        target_nodes = {int(edge_dst[edge]) for edge in unique_edges}
        selected_nodes = {int(edge_dst[edge]) for edge in selected}
        node_recall = (
            len(target_nodes & selected_nodes) / len(target_nodes) if target_nodes else 1.0
        )
        original = int(edge_output.original_edge_count)
        retained = int(edge_output.selected_edge_index.numel())
        metrics = {
            "target_edge_recall": float(edge_recall),
            "target_group_recall": float(group_recall),
            "target_node_recall": float(node_recall),
            "original_edge_count": float(original),
            "retained_edge_count": float(retained),
            "pruned_fraction": float(1.0 - retained / max(original, 1)),
            "over_limit": float(original > retained),
        }
        metrics.update(self._depth_metrics(target, target_edges, selected))
        # Condition slices are diagnostic only: the absolute_ranker remains
        # condition-independent, but every sample's target coverage is visible.
        if target_edges.numel():
            sample_ids = target_edges[0].unique(sorted=True)
            ce_values = target.sample_ce_value.detach().cpu().float()
            finite_ce = ce_values[torch.isfinite(ce_values)]
            ce_cuts = (
                torch.quantile(finite_ce, torch.tensor([0.25, 0.5, 0.75])).tolist()
                if finite_ce.numel()
                else [0.0, 0.0, 0.0]
            )
            sliced: Dict[str, List[float]] = {}
            for sample_id in sample_ids.tolist():
                sample_set = set(target_edges[1, target_edges[0] == sample_id].tolist())
                recall = len(sample_set & selected) / max(len(sample_set), 1)
                adduct = int(target.sample_adduct_type_index[sample_id].item())
                adduct_label = self._tensorboard_label(
                    self.adduct_labels.get(adduct, f"unknown-{adduct}")
                )
                ce = float(ce_values[sample_id].item())
                ce_bin = sum(ce > cut for cut in ce_cuts) if torch.isfinite(ce_values[sample_id]) else -1
                ce_label = {
                    -1: "non-finite",
                    0: "min-to-q1",
                    1: "q1-to-median",
                    2: "median-to-q3",
                    3: "q3-to-max",
                }[ce_bin]
                sliced.setdefault(
                    f"by_adduct/{adduct_label}/target_edge_recall", []
                ).append(recall)
                sliced.setdefault(
                    f"by_ce_range/{ce_label}/target_edge_recall", []
                ).append(recall)
            metrics.update(
                {name: float(sum(values) / len(values)) for name, values in sliced.items()}
            )
        return metrics

    @staticmethod
    def _depth_metrics(
        target: TrainingFragmentTreeStructure,
        target_edges: Tensor,
        selected: set[int],
    ) -> Dict[str, float]:
        """Natural absolute_ranker coverage for every cleavage depth."""
        if not target_edges.numel() or not target.edge_index.numel():
            return {}
        edge_index = target.edge_index.detach().cpu().long()
        edge_src, edge_dst = edge_index[0], edge_index[1]
        num_nodes = int(target.num_nodes)
        has_parent = torch.zeros((num_nodes,), dtype=torch.bool)
        has_parent[edge_dst] = True
        node_depth = torch.full((num_nodes,), -1, dtype=torch.long)
        node_depth[~has_parent] = 0
        # The stored fragment tree is a DAG. Repeated relaxation also handles
        # collated disconnected trees and chooses the shortest reachable depth.
        for _ in range(num_nodes):
            known = node_depth[edge_src] >= 0
            if not known.any():
                break
            candidate = node_depth[edge_src[known]] + 1
            destinations = edge_dst[known]
            changed = False
            for destination, depth in zip(destinations.tolist(), candidate.tolist()):
                current = int(node_depth[destination].item())
                if current < 0 or depth < current:
                    node_depth[destination] = int(depth)
                    changed = True
            if not changed:
                break
        edge_depth = node_depth[edge_src] + 1
        groups = target.target_edge_group_index.detach().cpu().long()
        result: Dict[str, float] = {}
        valid_target = (
            (target_edges[1] >= 0) & (target_edges[1] < edge_depth.numel())
        )
        for depth in edge_depth[target_edges[1, valid_target]].unique(sorted=True).tolist():
            if depth < 1:
                continue
            row_mask = valid_target.clone()
            row_mask[valid_target] = edge_depth[target_edges[1, valid_target]] == int(depth)
            depth_pairs = target_edges[:, row_mask]
            depth_edges = set(depth_pairs[1].tolist())
            prefix = f"by_depth/depth_{int(depth)}"
            result[f"{prefix}/target_edge_recall"] = (
                len(depth_edges & selected) / len(depth_edges) if depth_edges else 1.0
            )
            depth_nodes = {int(edge_dst[edge]) for edge in depth_edges}
            selected_nodes = {int(edge_dst[edge]) for edge in depth_edges & selected}
            result[f"{prefix}/target_node_recall"] = (
                len(depth_nodes & selected_nodes) / len(depth_nodes) if depth_nodes else 1.0
            )
            group_total = 0
            group_kept = 0
            for sample_id in depth_pairs[0].unique(sorted=True).tolist():
                sample_rows = row_mask & (target_edges[0] == int(sample_id))
                for group_id in groups[sample_rows].unique(sorted=True).tolist():
                    group_rows = sample_rows & (groups == int(group_id))
                    group_edge_ids = set(target_edges[1, group_rows].tolist())
                    group_total += 1
                    group_kept += int(bool(group_edge_ids & selected))
            result[f"{prefix}/target_group_recall"] = (
                group_kept / group_total if group_total else 1.0
            )
        return result


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
    """Loss for peak-wise fragment coverage, state prediction, and cleavage."""

    def __init__(self, intensity_alpha: float = 0.2, intensity_gamma: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < intensity_alpha <= 1.0:
            raise ValueError("intensity_alpha must be in (0, 1].")
        if intensity_gamma <= 0.0:
            raise ValueError("intensity_gamma must be positive.")
        self.intensity_alpha = float(intensity_alpha)
        self.intensity_gamma = float(intensity_gamma)

    def forward(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
    ) -> Tensor:
        device = output.keep_logit.device
        self._validate_state_targets(target)
        losses: List[Tensor] = []

        if target.target_node_index.numel() > 0:
            # Computed once and shared: each is an O(rows) Python loop with
            # per-row GPU-sync (.item()/.tolist()) calls, so recomputing them
            # separately in every peak-wise loss below measurably adds up.
            batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
            peak_keys = self._target_peak_keys(target, device=device)
            losses.append(self._peak_fragment_loss(
                output, target, device=device,
                peak_keys=peak_keys, batch_node_by_sample_node=batch_node_by_sample_node,
            ))
            losses.append(self._keep_negative_loss(output, target, device=device))
            losses.append(self._state_loss(
                output, target, device=device,
                peak_keys=peak_keys, batch_node_by_sample_node=batch_node_by_sample_node,
            ))
            precursor_loss = self._precursor_keep_loss(
                output, target, device=device,
                peak_keys=peak_keys, batch_node_by_sample_node=batch_node_by_sample_node,
            )
            if precursor_loss is not None:
                losses.append(precursor_loss)

        cleave_target, cleave_mask = self._build_cleave_targets(output, target, device=device)
        if cleave_mask.any():
            losses.append(self._expand_node_ranking_loss(
                output, target, cleave_target=cleave_target,
                cleave_mask=cleave_mask, device=device,
            ))

        edge_group_loss = self._edge_group_coverage_loss(output, target, device=device)
        edge_negative_loss = self._edge_negative_loss(output, target, device=device)
        losses.append(edge_group_loss)
        losses.append(edge_negative_loss)

        if len(losses) == 0:
            return output.keep_logit.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _validate_state_targets(target: TrainingFragmentTreeStructure) -> None:
        target_count = int(target.target_node_index.numel())
        state_tensors = (
            target.target_ion_index,
            target.target_unsaturation_index,
            target.target_radical_index,
            target.target_sample_index,
            target.target_peak_index,
            target.target_intensity,
        )
        if any(int(tensor.numel()) != target_count for tensor in state_tensors):
            raise ValueError("Target state tensors must align with target_node_index.")
        if target_count > 0:
            if (
                (target.target_ion_index < 0).any()
                or (target.target_unsaturation_index < 0).any()
                or (target.target_radical_index < 0).any()
            ):
                raise ValueError(
                    "Training structures must contain resolved ion, "
                    "unsaturation, and radical target indexes. Regenerate "
                    "the training structures with the current builder."
                )

    def _peak_fragment_loss(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
        peak_keys: Optional[List[Tuple[int, int]]] = None,
        batch_node_by_sample_node: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> Tensor:
        if batch_node_by_sample_node is None:
            batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
        if peak_keys is None:
            peak_keys = self._target_peak_keys(target, device=device)
        losses: List[Tensor] = []
        weights: List[Tensor] = []

        for peak_key in peak_keys:
            row_index = self._target_peak_mask(target, peak_key=peak_key, device=device)
            batch_node_indexes = self._unique_batch_node_indexes(
                target_sample_index=target.target_sample_index[row_index].to(device),
                target_node_index=target.target_node_index[row_index].to(device),
                batch_node_by_sample_node=batch_node_by_sample_node,
            )
            if batch_node_indexes.numel() == 0:
                continue
            peak_logit = output.keep_logit[batch_node_indexes]
            losses.append(F.softplus(-torch.logsumexp(peak_logit, dim=0)))
            peak_intensity = target.target_intensity[row_index].to(device).float().max()
            sample_mask = target.target_sample_index.to(device).long() == int(peak_key[0])
            sample_max = target.target_intensity.to(device).float()[sample_mask].max().clamp_min(1e-12)
            weights.append(self.intensity_weight(peak_intensity / sample_max))

        if len(losses) == 0:
            return output.keep_logit.sum() * 0.0
        return self._weighted_mean(losses, weights, output.keep_logit)

    def intensity_weight(self, normalized_intensity: Tensor) -> Tensor:
        normalized_intensity = normalized_intensity.float().clamp(0.0, 1.0)
        return self.intensity_alpha + (1.0 - self.intensity_alpha) * normalized_intensity.pow(
            self.intensity_gamma
        )

    def _precursor_keep_loss(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
        peak_keys: Optional[List[Tuple[int, int]]] = None,
        batch_node_by_sample_node: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> Optional[Tensor]:
        """Dedicated keep-loss for precursor-root target peaks.

        The precursor ion is usually the dominant peak in an observed
        spectrum, yet it is only one of many (sample, peak) rows inside
        ``_peak_fragment_loss``'s batch-wide weighted mean, so its gradient
        is diluted by however many fragment peaks the sample also has. This
        term averages only over precursor peaks so it carries a stable,
        undiluted share of the training signal regardless of fragment count.
        """
        batch = output.sample_tree_batch
        node_is_precursor_root = batch.node_is_precursor_root.to(device).bool()
        if batch_node_by_sample_node is None:
            batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
        if peak_keys is None:
            peak_keys = self._target_peak_keys(target, device=device)
        losses: List[Tensor] = []

        for peak_key in peak_keys:
            row_index = self._target_peak_mask(target, peak_key=peak_key, device=device)
            batch_node_indexes = self._unique_batch_node_indexes(
                target_sample_index=target.target_sample_index[row_index].to(device),
                target_node_index=target.target_node_index[row_index].to(device),
                batch_node_by_sample_node=batch_node_by_sample_node,
            )
            if batch_node_indexes.numel() == 0:
                continue
            precursor_indexes = batch_node_indexes[node_is_precursor_root[batch_node_indexes]]
            if precursor_indexes.numel() == 0:
                continue
            losses.append(F.softplus(-torch.logsumexp(output.keep_logit[precursor_indexes], dim=0)))

        if len(losses) == 0:
            return None
        return torch.stack(losses).mean()

    def _keep_negative_loss(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
    ) -> Tensor:
        batch = output.sample_tree_batch
        kept_sample_ids = batch.kept_sample_ids.to(device).long()
        graph_index_by_node = batch.batch.to(device).long()
        node_global_ids = batch.node_id_global.to(device).long()
        node_is_precursor_root = batch.node_is_precursor_root.to(device).bool()
        positive_pairs = self._target_sample_node_pairs(
            target_sample_index=target.target_sample_index.to(device).long(),
            target_node_index=target.target_node_index.to(device).long(),
        )
        sample_losses: List[Tensor] = []
        for graph_index in range(int(kept_sample_ids.numel())):
            sample_id = int(kept_sample_ids[graph_index].detach().cpu().item())
            node_mask = (graph_index_by_node == graph_index) & (~node_is_precursor_root)
            if not node_mask.any():
                continue
            node_indexes = node_mask.nonzero(as_tuple=False).view(-1)
            negative_indexes = []
            for batch_node_index in node_indexes.detach().cpu().tolist():
                batch_node_index = int(batch_node_index)
                node_id = int(node_global_ids[batch_node_index].detach().cpu().item())
                if (sample_id, node_id) not in positive_pairs:
                    negative_indexes.append(batch_node_index)
            if not negative_indexes:
                continue
            index_tensor = torch.tensor(negative_indexes, dtype=torch.long, device=device)
            sample_losses.append(
                F.binary_cross_entropy_with_logits(
                    output.keep_logit[index_tensor],
                    output.keep_logit.new_zeros((index_tensor.numel(),)),
                )
            )

        if len(sample_losses) == 0:
            return output.keep_logit.sum() * 0.0
        return torch.stack(sample_losses).mean()

    def _state_loss(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
        peak_keys: Optional[List[Tuple[int, int]]] = None,
        batch_node_by_sample_node: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> Tensor:
        batch = output.sample_tree_batch
        if batch_node_by_sample_node is None:
            batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
        if peak_keys is None:
            peak_keys = self._target_peak_keys(target, device=device)
        losses: List[Tensor] = []
        weights: List[Tensor] = []

        for peak_key in peak_keys:
            row_index = self._target_peak_mask(target, peak_key=peak_key, device=device)
            batch_node_indexes = self._unique_batch_node_indexes(
                target_sample_index=target.target_sample_index[row_index].to(device),
                target_node_index=target.target_node_index[row_index].to(device),
                batch_node_by_sample_node=batch_node_by_sample_node,
            )
            if batch_node_indexes.numel() == 0:
                continue

            fragment_prob = torch.softmax(output.keep_logit[batch_node_indexes], dim=0).detach()
            prob_by_batch_node = {
                int(batch_node): fragment_prob[index]
                for index, batch_node in enumerate(batch_node_indexes.detach().cpu().tolist())
            }
            peak_weight = target.target_intensity[row_index].to(device).float().max()
            seen_nodes: set[int] = set()

            for row in row_index.detach().cpu().tolist():
                row = int(row)
                key = (
                    int(target.target_sample_index[row].detach().cpu().item()),
                    int(target.target_node_index[row].detach().cpu().item()),
                )
                batch_node_index = batch_node_by_sample_node.get(key)
                if batch_node_index is None or batch_node_index in seen_nodes:
                    continue
                seen_nodes.add(batch_node_index)

                role_index = 0 if bool(batch.node_is_precursor_root[batch_node_index].detach().cpu().item()) else 1
                main_adduct_index = int(batch.node_main_adduct_type_index[batch_node_index].detach().cpu().item())
                node_weight = peak_weight * prob_by_batch_node[int(batch_node_index)]

                losses.append(
                    self._masked_cross_entropy(
                        output.ion_logit[batch_node_index],
                        int(target.target_ion_index[row].detach().cpu().item()),
                        output.ion_valid_mask_by_role_adduct[role_index, main_adduct_index],
                    )
                )
                weights.append(node_weight)

                losses.append(
                    self._masked_cross_entropy(
                        output.unsaturation_logit[batch_node_index],
                        int(target.target_unsaturation_index[row].detach().cpu().item()),
                        output.unsaturation_valid_mask_by_role_adduct[role_index, main_adduct_index],
                    )
                )
                weights.append(node_weight)

                losses.append(
                    self._masked_cross_entropy(
                        output.radical_logit[batch_node_index],
                        int(target.target_radical_index[row].detach().cpu().item()),
                        output.radical_valid_mask_by_role_adduct[role_index, main_adduct_index],
                    )
                )
                weights.append(node_weight)

        return self._weighted_mean(losses, weights, output.keep_logit)

    @staticmethod
    def _masked_cross_entropy(logit: Tensor, target_index: int, mask: Tensor) -> Tensor:
        if mask.numel() == 0:
            return F.cross_entropy(logit[None, :], logit.new_tensor([target_index], dtype=torch.long))
        mask = mask.to(logit.device).bool()
        if target_index < 0 or target_index >= int(logit.numel()):
            raise IndexError(f"target_index={target_index} is out of range for logits.")
        if not bool(mask[target_index].detach().cpu().item()):
            raise ValueError(f"target_index={target_index} is not valid for this node role/adduct.")
        masked_logit = logit.masked_fill(~mask, -torch.finfo(logit.dtype).max)
        return F.cross_entropy(masked_logit[None, :], logit.new_tensor([target_index], dtype=torch.long))

    @staticmethod
    def _target_peak_keys(
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
    ) -> List[Tuple[int, int]]:
        keys: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        sample_index = target.target_sample_index.to(device).long()
        peak_index = target.target_peak_index.to(device).long()
        for row in range(int(sample_index.numel())):
            key = (int(sample_index[row].item()), int(peak_index[row].item()))
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    @staticmethod
    def _target_peak_mask(
        target: TrainingFragmentTreeStructure,
        *,
        peak_key: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        sample_index = target.target_sample_index.to(device).long()
        peak_index = target.target_peak_index.to(device).long()
        return ((sample_index == int(peak_key[0])) & (peak_index == int(peak_key[1]))).nonzero(as_tuple=False).view(-1)

    @staticmethod
    def _batch_node_by_sample_node(
        output: FragmentTreeCandidateSelectionOutput,
        *,
        device: torch.device,
    ) -> Dict[Tuple[int, int], int]:
        batch = output.sample_tree_batch
        kept_sample_ids = batch.kept_sample_ids.to(device).long()
        graph_index_by_node = batch.batch.to(device).long()
        node_global_ids = batch.node_id_global.to(device).long()
        mapping: Dict[Tuple[int, int], int] = {}
        for batch_node_index in range(int(node_global_ids.numel())):
            graph_index = int(graph_index_by_node[batch_node_index].item())
            sample_id = int(kept_sample_ids[graph_index].item())
            node_id = int(node_global_ids[batch_node_index].item())
            mapping[(sample_id, node_id)] = int(batch_node_index)
        return mapping

    @staticmethod
    def _unique_batch_node_indexes(
        *,
        target_sample_index: Tensor,
        target_node_index: Tensor,
        batch_node_by_sample_node: Dict[Tuple[int, int], int],
    ) -> Tensor:
        device = target_sample_index.device
        indexes: List[int] = []
        seen: set[int] = set()
        for sample_id, node_id in zip(
            target_sample_index.detach().cpu().tolist(),
            target_node_index.detach().cpu().tolist(),
        ):
            batch_node_index = batch_node_by_sample_node.get((int(sample_id), int(node_id)))
            if batch_node_index is not None and batch_node_index not in seen:
                seen.add(batch_node_index)
                indexes.append(batch_node_index)
        return torch.tensor(indexes, dtype=torch.long, device=device)

    @staticmethod
    def _weighted_mean(losses: List[Tensor], weights: List[Tensor], like: Tensor) -> Tensor:
        if len(losses) == 0:
            return like.sum() * 0.0
        loss_tensor = torch.stack(losses)
        weight_tensor = torch.stack(weights).to(loss_tensor.device).float().clamp_min(0.0)
        if float(weight_tensor.sum().detach().cpu().item()) <= 0.0:
            return loss_tensor.mean()
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()

    @staticmethod
    def _balanced_node_bce_loss(
        *,
        logit: Tensor,
        target: Tensor,
        mask: Tensor,
        output: FragmentTreeCandidateSelectionOutput,
        device: torch.device,
    ) -> Tensor:
        batch = output.sample_tree_batch
        kept_sample_ids = batch.kept_sample_ids.to(device).long()
        graph_index_by_node = batch.batch.to(device).long()
        sample_losses: List[Tensor] = []
        for graph_index in range(int(kept_sample_ids.numel())):
            sample_mask = mask & (graph_index_by_node == graph_index)
            if not sample_mask.any():
                continue
            positive_mask = sample_mask & (target > 0.5)
            negative_mask = sample_mask & (target <= 0.5)
            component_losses: List[Tensor] = []
            if positive_mask.any():
                component_losses.append(
                    F.binary_cross_entropy_with_logits(
                        logit[positive_mask],
                        target[positive_mask],
                    )
                )
            if negative_mask.any():
                component_losses.append(
                    F.binary_cross_entropy_with_logits(
                        logit[negative_mask],
                        target[negative_mask],
                    )
                )
            if component_losses:
                sample_losses.append(torch.stack(component_losses).mean())

        if len(sample_losses) == 0:
            return logit.sum() * 0.0
        return torch.stack(sample_losses).mean()

    def _expand_node_ranking_loss(
        self, output, target, *, cleave_target: Tensor,
        cleave_mask: Tensor, device: torch.device,
    ) -> Tensor:
        """At least one stored path intermediate must beat every outsider."""
        batch = output.sample_tree_batch
        kept = batch.kept_sample_ids.to(device).long()
        graph_by_node = batch.batch.to(device).long()
        losses: List[Tensor] = []
        weights: List[Tensor] = []
        for graph_id, sample_id in enumerate(kept.tolist()):
            local = cleave_mask & (graph_by_node == graph_id)
            positive = local & (cleave_target > 0.5)
            negative = local & ~positive
            if not positive.any() or not negative.any():
                continue
            positive_score = torch.logsumexp(output.cleave_logit[positive], dim=0)
            hard_negative = output.cleave_logit[negative].max()
            losses.append(F.softplus(-(positive_score - hard_negative)))
            peak_start = int(target.sample_peak_ptr[sample_id])
            peak_end = int(target.sample_peak_ptr[sample_id + 1])
            sample_intensity = target.sample_peak_intensity[peak_start:peak_end].to(device).float()
            weights.append(sample_intensity.max().clamp_min(1e-12) if sample_intensity.numel() else output.cleave_logit.new_tensor(1.0))
        return self._weighted_mean(losses, weights, output.cleave_logit)

    def _edge_group_coverage_loss(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
    ) -> Tensor:
        if target.target_edge_index.numel() == 0 or output.edge_cleave_logit.numel() == 0:
            return output.edge_cleave_logit.sum() * 0.0

        batch_edge_by_sample_edge = self._batch_edge_by_sample_edge(output, device=device)
        target_edge_index = target.target_edge_index.to(device).long()
        target_edge_group_index = target.target_edge_group_index.to(device).long()
        if target_edge_index.size(1) != target_edge_group_index.numel():
            raise ValueError("target_edge_index and target_edge_group_index are misaligned.")

        losses: List[Tensor] = []
        seen: set[Tuple[int, int]] = set()
        for row in range(int(target_edge_group_index.numel())):
            group_key = (
                int(target_edge_index[0, row].detach().cpu().item()),
                int(target_edge_group_index[row].detach().cpu().item()),
            )
            if group_key in seen:
                continue
            seen.add(group_key)

            group_mask = (
                (target_edge_index[0] == int(group_key[0]))
                & (target_edge_group_index == int(group_key[1]))
            )
            batch_edge_indexes: List[int] = []
            for edge_id in target_edge_index[1, group_mask].detach().cpu().tolist():
                batch_edge_index = batch_edge_by_sample_edge.get((int(group_key[0]), int(edge_id)))
                if batch_edge_index is not None:
                    batch_edge_indexes.append(int(batch_edge_index))

            if not batch_edge_indexes:
                continue

            index_tensor = torch.tensor(
                sorted(set(batch_edge_indexes)),
                dtype=torch.long,
                device=device,
            )
            losses.append(F.softplus(-torch.logsumexp(output.edge_cleave_logit[index_tensor], dim=0)))

        if len(losses) == 0:
            return output.edge_cleave_logit.sum() * 0.0
        return torch.stack(losses).mean()

    def _edge_negative_loss(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
        *,
        device: torch.device,
    ) -> Tensor:
        if output.edge_cleave_logit.numel() == 0 or target.target_edge_index.numel() == 0:
            return output.edge_cleave_logit.sum() * 0.0

        batch = output.sample_tree_batch
        target_edge_index = target.target_edge_index.to(device).long()
        positive_pairs = {
            (int(sample_id), int(edge_id))
            for sample_id, edge_id in target_edge_index.detach().cpu().t().tolist()
        }

        edge_global_ids = batch.edge_id_global.to(device).long()
        edge_ptr = batch.edge_ptr.to(device).long()
        kept_sample_ids = batch.kept_sample_ids.to(device).long()
        sample_losses: List[Tensor] = []

        for graph_index in range(int(kept_sample_ids.numel())):
            sample_id = int(kept_sample_ids[graph_index].detach().cpu().item())
            start = int(edge_ptr[graph_index].detach().cpu().item())
            end = int(edge_ptr[graph_index + 1].detach().cpu().item())
            if end <= start:
                continue

            negative_indexes = []
            for batch_edge_index in range(start, end):
                edge_id = int(edge_global_ids[batch_edge_index].detach().cpu().item())
                if (sample_id, edge_id) not in positive_pairs:
                    negative_indexes.append(batch_edge_index)

            if not negative_indexes:
                continue

            index_tensor = torch.tensor(negative_indexes, dtype=torch.long, device=device)
            sample_losses.append(
                F.binary_cross_entropy_with_logits(
                    output.edge_cleave_logit[index_tensor],
                    output.edge_cleave_logit.new_zeros((index_tensor.numel(),)),
                )
            )

        if len(sample_losses) == 0:
            return output.edge_cleave_logit.sum() * 0.0
        return torch.stack(sample_losses).mean()

    @staticmethod
    def _batch_edge_by_sample_edge(
        output: FragmentTreeCandidateSelectionOutput,
        *,
        device: torch.device,
    ) -> Dict[Tuple[int, int], int]:
        batch = output.sample_tree_batch
        edge_global_ids = batch.edge_id_global.to(device).long()
        edge_ptr = batch.edge_ptr.to(device).long()
        kept_sample_ids = batch.kept_sample_ids.to(device).long()
        mapping: Dict[Tuple[int, int], int] = {}
        for graph_index in range(int(kept_sample_ids.numel())):
            sample_id = int(kept_sample_ids[graph_index].item())
            start = int(edge_ptr[graph_index].item())
            end = int(edge_ptr[graph_index + 1].item())
            for batch_edge_index in range(start, end):
                edge_id = int(edge_global_ids[batch_edge_index].item())
                mapping[(sample_id, edge_id)] = int(batch_edge_index)
        return mapping

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

        if target.target_expand_node_index.numel() == 0:
            return cleave_target, cleave_mask

        # Stored supervision already contains the intermediate nodes on every
        # assigned fragmentation path.  Use it directly; training must never
        # invoke Fragmenter/RDKit to create new cleavages.
        assignment_samples = target.target_sample_index.to(device).long()
        expand_nodes = target.target_expand_node_index.to(device).long()
        expand_ptr = target.terminal_expand_ptr.to(device).long()
        sample_parts: List[Tensor] = []
        node_parts: List[Tensor] = []
        for assignment_id in range(int(assignment_samples.numel())):
            start = int(expand_ptr[assignment_id].item())
            end = int(expand_ptr[assignment_id + 1].item())
            if end <= start:
                continue
            nodes = expand_nodes[start:end]
            node_parts.append(nodes)
            sample_parts.append(
                torch.full_like(nodes, int(assignment_samples[assignment_id].item()))
            )
        if not node_parts:
            return cleave_target, cleave_mask
        positive_pairs = self._target_sample_node_pairs(
            target_sample_index=torch.cat(sample_parts),
            target_node_index=torch.cat(node_parts),
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


class FragmentTreeIntensityTrainingLoss(nn.Module):
    """Formula-level Huber, spectrum cosine, and balanced presence losses."""

    def __init__(self, huber_weight: float = 1.0, cosine_weight: float = 1.0, presence_weight: float = 1.0) -> None:
        super().__init__()
        self.huber_weight = float(huber_weight)
        self.cosine_weight = float(cosine_weight)
        self.presence_weight = float(presence_weight)

    def forward(
        self,
        intensity_output,
        target: TrainingFragmentTreeStructure,
    ) -> Tensor:
        if intensity_output.logit.numel() == 0:
            return target.target_intensity.sum() * 0.0

        device = intensity_output.logit.device
        losses: List[Tensor] = []
        for sample_id in intensity_output.sample_index.detach().cpu().unique(sorted=True).tolist():
            sample_id = int(sample_id)
            pred_mask = intensity_output.sample_index == sample_id
            pred_index = pred_mask.nonzero(as_tuple=False).view(-1)
            target_weight = self._target_weight_for_predictions(
                predicted_formula=intensity_output.formula_tensor[pred_index],
                target=target,
                sample_id=sample_id,
                device=device,
            )
            target_intensity = target_weight / target_weight.sum().clamp_min(1e-12)
            predicted_intensity = intensity_output.logit[pred_index]
            huber_loss = F.smooth_l1_loss(predicted_intensity, target_intensity)
            cosine_loss = self.cosine_loss(predicted_intensity, target_intensity)
            sample_loss = self.huber_weight * huber_loss + self.cosine_weight * cosine_loss
            presence_logit = getattr(intensity_output, "presence_logit", None)
            if presence_logit is not None:
                presence_target = (target_weight > 0).to(presence_logit.dtype)
                sample_loss = sample_loss + self.presence_weight * self.balanced_presence_loss(
                    presence_logit[pred_index], presence_target
                )
            losses.append(sample_loss)

        if len(losses) == 0:
            return intensity_output.logit.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def cosine_loss(predicted: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
        return 1.0 - F.cosine_similarity(
            predicted.unsqueeze(0), target.unsqueeze(0), dim=1, eps=eps
        ).squeeze(0)

    @staticmethod
    def balanced_presence_loss(logit: Tensor, target: Tensor) -> Tensor:
        positive = target > 0.5
        negative = ~positive
        components: List[Tensor] = []
        if positive.any():
            components.append(F.binary_cross_entropy_with_logits(logit[positive], target[positive]))
        if negative.any():
            components.append(F.binary_cross_entropy_with_logits(logit[negative], target[negative]))
        if not components:
            return logit.sum() * 0.0
        return torch.stack(components).mean()

    @staticmethod
    def _target_weight_for_predictions(
        *,
        predicted_formula: Tensor,
        target: TrainingFragmentTreeStructure,
        sample_id: int,
        device: torch.device,
    ) -> Tensor:
        target_weight = predicted_formula.new_zeros((predicted_formula.size(0),))
        formula_peak_index = target.formula_peak_index.to(device).long()
        formula_sample_index = target.peak_sample_index.to(device).long()[formula_peak_index]
        target_mask = formula_sample_index == int(sample_id)
        if not target_mask.any():
            return target_weight

        target_formula = target.target_formula.to(device).float()[target_mask]
        target_intensity = target.sample_peak_intensity.to(device).float()[
            formula_peak_index[target_mask]
        ]
        for pred_index, formula in enumerate(predicted_formula):
            formula_mask = torch.all(target_formula == formula, dim=1)
            if formula_mask.any():
                target_weight[pred_index] = target_intensity[formula_mask].sum()
        return target_weight


class FragmentTreeTrainingModel(nn.Module):
    """Training wrapper that combines candidate selection and supervised losses."""

    def __init__(
        self,
        candidate_selector: nn.Module,
        loss_fn: nn.Module | None = None,
        *,
        intensity_predictor: FragmentTreeFormulaIntensityPredictor | None = None,
        intensity_loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.candidate_selector = candidate_selector
        self.intensity_predictor = intensity_predictor
        self.loss_fn = loss_fn if loss_fn is not None else FragmentTreeSelectionTrainingLoss()
        self.intensity_loss_fn = (
            intensity_loss_fn
            if intensity_loss_fn is not None
            else FragmentTreeIntensityTrainingLoss()
        )
        self._checkpoint_model_config: Dict[str, Any] = {}
        feature_model = candidate_selector.feature_model
        self.adduct_labels = {
            int(index): str(adduct)
            for index, adduct in feature_model.main_adduct_types.items()
        }
        self.ranking_loss_weight = float(getattr(candidate_selector, "ranking_loss_weight", 1.0))
        self.absolute_ranker_loss_fn = PairwiseEdgeIntensityRankingLoss(
            top_n=int(getattr(candidate_selector, "top_n", 10)),
            nearest_lower_partners=int(getattr(candidate_selector, "nearest_lower_partners", 1)),
            extended_lower_partners=int(getattr(candidate_selector, "extended_lower_partners", 3)),
            background_partners=int(getattr(candidate_selector, "background_partners", 10)),
            intensity_threshold=float(getattr(candidate_selector, "ranking_intensity_threshold", 0.05)),
        )
        self.register_buffer("training_phase", torch.tensor(1, dtype=torch.long))

    @property
    def absolute_ranker_only(self) -> bool:
        return False

    def update_training_phase(self, validation_summary: Dict[str, float]) -> bool:
        """Compatibility hook; both architectural stages train jointly."""
        return False

    def set_checkpoint_model_config(self, model_config: Dict[str, Any]) -> None:
        self._checkpoint_model_config = dict(model_config)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._checkpoint_model_config)

    def forward(self, batch: TrainingFragmentTreeStructure):
        output = self.candidate_selector(batch)
        absolute_ranker_loss = self.absolute_ranker_loss_fn(output, batch)
        selection_loss = self.loss_fn(output, batch)
        if self.intensity_predictor is None:
            intensity_output = None
            intensity_loss = selection_loss.detach() * 0.0
        else:
            intensity_output = self.intensity_predictor.forward_candidate_output(output)
            intensity_loss = self.intensity_loss_fn(intensity_output, batch)
        edge_total_loss = selection_loss + self.ranking_loss_weight * absolute_ranker_loss
        loss = edge_total_loss + intensity_loss
        absolute_ranker_metrics = self.absolute_ranker_loss_fn.metrics(output, batch)
        absolute_ranker_metrics.update(self._edge_retain_metrics(output, batch))
        absolute_ranker_metrics.update(self._selected_peak_metrics(output, batch))
        absolute_ranker_metrics.update(self._tree_edge_budget_metrics(output, batch))
        absolute_ranker_metrics.update(DepthEvaluation(batch).evaluate(output))
        if intensity_output is not None:
            absolute_ranker_metrics.update(
                self._intensity_similarity_metrics(intensity_output, batch)
            )
        absolute_ranker_metrics.update({
            "edge_retain_loss": float(selection_loss.detach().cpu().item()),
            "edge_total_loss": float(edge_total_loss.detach().cpu().item()),
        })
        with torch.no_grad():
            precursor_keep_loss = self.loss_fn._precursor_keep_loss(output, batch, device=selection_loss.device)
        if precursor_keep_loss is not None:
            absolute_ranker_metrics["precursor/keep_loss"] = float(
                precursor_keep_loss.detach().cpu().item()
            )
        return {
            "loss": loss,
            "selection_loss": selection_loss,
            "intensity_loss": intensity_loss,
            "absolute_ranker_loss": absolute_ranker_loss,
            "edge_retain_loss": selection_loss,
            "edge_ranking_loss": absolute_ranker_loss,
            "edge_total_loss": edge_total_loss,
            "candidate_output": output,
            "intensity_output": intensity_output,
            "absolute_ranker_metrics": absolute_ranker_metrics,
        }

    @torch.no_grad()
    def evaluate_depth_rollout(self, target) -> Dict[str, float]:
        """Follow model-chosen frontiers on the saved DAG, without target injection."""
        evaluator = DepthEvaluation(target)
        metrics = {}
        previous = None

        def observe(step, output):
            nonlocal previous
            metrics.update({f"rollout_step_{step}/{name}": value
                            for name, value in evaluator.evaluate(output, pending_only=True).items()})
            if previous is not None:
                metrics.update({f"rollout_step_{step}/{name}": value
                                for name, value in evaluator.after_expansion(previous, output).items()})
            previous = output

        max_depth = min(
            max(0, int(self.candidate_selector.fragmenter.tree_max_depth) - 1),
            len(self.candidate_selector.fragment_edge_encoder.max_edges_per_depth) - 1,
        )
        final = self.candidate_selector.generate_depth_limited_candidates(
            target, max_depth=max_depth, on_step=observe,
        )
        metrics.update({f"rollout_final/{name}": value
                        for name, value in evaluator.evaluate(final, pending_only=True).items()})
        return metrics

    def _grouped_metric_means(
        self,
        per_sample_values_by_metric: Dict[str, Dict[int, float]],
        target,
    ) -> Dict[str, float]:
        """Group per-sample metric values by adduct type and CE bucket.

        Returns flat ``<metric>@by_adduct:<label>`` and
        ``<metric>@by_ce_range:<label>`` keys, each the mean of the
        per-sample values whose sample falls in that group. The ``@scope``
        suffix separates ordinary, adduct, and CE TensorBoard cards
        (see ``log_distribution_cards``). Grouping is computed once here and shared by
        every caller's metric.
        """
        if not any(per_sample_values_by_metric.values()):
            return {}
        if not hasattr(target, "sample_adduct_type_index") or not hasattr(target, "sample_ce_value"):
            return {}
        adduct_index = target.sample_adduct_type_index.detach().cpu().long()
        ce_values = target.sample_ce_value.detach().cpu().float()
        finite_ce = ce_values[torch.isfinite(ce_values)]
        if finite_ce.numel():
            ce_q1, ce_median, ce_q3 = torch.quantile(
                finite_ce, torch.tensor([0.25, 0.5, 0.75])
            ).tolist()
        else:
            ce_q1 = ce_median = ce_q3 = 0.0
        grouped: Dict[str, float] = {}
        for metric_name, values_by_sample in per_sample_values_by_metric.items():
            by_adduct: Dict[str, List[float]] = {}
            by_ce: Dict[str, List[float]] = {}
            for sample_id, value in values_by_sample.items():
                adduct = int(adduct_index[sample_id].item())
                adduct_label = metric_labels.tensorboard_label(
                    self.adduct_labels.get(adduct, f"unknown-{adduct}")
                )
                by_adduct.setdefault(adduct_label, []).append(value)
                ce_label = metric_labels.ce_range_label(
                    float(ce_values[sample_id].item()), q1=ce_q1, median=ce_median, q3=ce_q3
                )
                by_ce.setdefault(ce_label, []).append(value)
            for label, values in by_adduct.items():
                grouped[f"{metric_name}@by_adduct:{label}"] = sum(values) / len(values)
            for label, values in by_ce.items():
                grouped[f"{metric_name}@by_ce_range:{label}"] = sum(values) / len(values)
        return grouped

    @torch.no_grad()
    def _tree_edge_budget_metrics(self, output, target) -> Dict[str, float]:
        """Report how the per-tree ``max_edges_per_tree`` cap behaved.

        Empty when the mechanism is disabled (no ``selected_edge_index``) or
        the structure predates ``tree_sample_ptr``. "Available" edges are
        counted per tree among edges actually referenced by some sample
        (``sample_edge_index``) -- edges no sample references have zero
        importance and are never worth the attention budget regardless of
        which tree they belong to, so they are not counted here either.
        """
        selected_edge_index = getattr(output, "selected_edge_index", None)
        if selected_edge_index is None or not hasattr(target, "tree_sample_ptr"):
            return {}
        device = selected_edge_index.device
        num_edges = int(target.num_edges)
        selected = selected_edge_index.to(device).long().unique()
        selected_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        if selected.numel():
            selected_mask[selected] = True

        sample_edge_index = target.sample_edge_index.to(device).long()
        valid = sample_edge_index[1] >= 0
        sample_ids = sample_edge_index[0, valid]
        edge_ids = sample_edge_index[1, valid]
        reference_counts = torch.zeros(num_edges, device=device, dtype=torch.float32)
        if edge_ids.numel():
            reference_counts.scatter_add_(0, edge_ids, torch.ones_like(edge_ids, dtype=torch.float32))
        selected_reference_counts = reference_counts[selected] if selected.numel() else reference_counts[:0]

        max_edges_per_tree = getattr(self.candidate_selector, "max_edges_per_tree", None)
        tree_sample_ptr = target.tree_sample_ptr.to(device).long()
        num_trees = int(tree_sample_ptr.numel()) - 1
        trees_over_budget = 0
        referenced_edge_count = 0
        if edge_ids.numel():
            sample_tree_id = torch.zeros(int(target.num_samples), dtype=torch.long, device=device)
            for tree_id in range(num_trees):
                start, end = int(tree_sample_ptr[tree_id]), int(tree_sample_ptr[tree_id + 1])
                sample_tree_id[start:end] = tree_id
            edge_to_tree = torch.full((num_edges,), -1, dtype=torch.long, device=device)
            edge_to_tree[edge_ids] = sample_tree_id[sample_ids]
            referenced_edges = edge_ids.unique()
            referenced_edge_count = int(referenced_edges.numel())
            for tree_id in range(num_trees):
                tree_size = int((edge_to_tree[referenced_edges] == tree_id).sum().item())
                if max_edges_per_tree is not None and tree_size > int(max_edges_per_tree):
                    trees_over_budget += 1

        target_edge_index = getattr(target, "target_edge_index", None)
        if target_edge_index is not None and target_edge_index.numel():
            target_edges = target_edge_index[1].to(device).long().unique()
            target_edges = target_edges[(target_edges >= 0) & (target_edges < num_edges)]
            target_edge_recall = (
                float(selected_mask[target_edges].float().mean().item())
                if target_edges.numel() else 1.0
            )
        else:
            target_edge_recall = 1.0

        return {
            "tree_edge_budget/selected_edge_count": float(selected.numel()),
            "tree_edge_budget/available_edge_count": float(referenced_edge_count),
            "tree_edge_budget/target_edge_recall": target_edge_recall,
            "tree_edge_budget/trees_over_budget_fraction": (
                trees_over_budget / max(num_trees, 1)
            ),
            "tree_edge_budget/mean_samples_per_selected_edge": (
                float(selected_reference_counts.mean().item())
                if selected_reference_counts.numel() else 0.0
            ),
        }

    @torch.no_grad()
    def _edge_retain_metrics(self, output, target) -> Dict[str, float]:
        logits = output.edge_absolute_logit
        sample_tree_batch = output.sample_tree_batch
        if logits.numel() == 0:
            return {"edge_retain_precision": 1.0, "edge_retain_recall": 1.0}
        edge_graph = sample_tree_batch.batch[
            sample_tree_batch.edge_index[0]
        ].to(logits.device).long()
        sample_ids = sample_tree_batch.kept_sample_ids.to(logits.device).long()[edge_graph]
        edge_ids = sample_tree_batch.edge_id_global.to(logits.device).long()
        positive_pairs = {
            (int(sample_id), int(edge_id))
            for sample_id, edge_id in target.target_edge_index.detach().cpu().t().tolist()
        }
        truth = torch.tensor(
            [
                (int(sample_id), int(edge_id)) in positive_pairs
                for sample_id, edge_id in zip(sample_ids.tolist(), edge_ids.tolist())
            ],
            dtype=torch.bool,
            device=logits.device,
        )
        predicted = logits > 0
        true_positive = int((predicted & truth).sum().item())
        metrics = {
            "edge_retain_accuracy": float((predicted == truth).float().mean().item()),
            "edge_retain_precision": true_positive / max(int(predicted.sum().item()), 1),
            "edge_retain_recall": true_positive / max(int(truth.sum().item()), 1),
        }
        per_sample: Dict[str, Dict[int, float]] = {
            "edge_retain_accuracy": {},
            "edge_retain_precision": {},
            "edge_retain_recall": {},
        }
        for sample_id in sample_ids.unique(sorted=True).tolist():
            sample_mask = sample_ids == int(sample_id)
            sample_predicted = predicted[sample_mask]
            sample_truth = truth[sample_mask]
            sample_true_positive = int((sample_predicted & sample_truth).sum().item())
            per_sample["edge_retain_accuracy"][int(sample_id)] = float(
                (sample_predicted == sample_truth).float().mean().item()
            )
            per_sample["edge_retain_precision"][int(sample_id)] = sample_true_positive / max(
                int(sample_predicted.sum().item()), 1
            )
            per_sample["edge_retain_recall"][int(sample_id)] = sample_true_positive / max(
                int(sample_truth.sum().item()), 1
            )
        metrics.update(self._grouped_metric_means(per_sample, target))
        return metrics

    @torch.no_grad()
    def _selected_peak_metrics(self, output, target) -> Dict[str, float]:
        selected_nodes = {
            (int(candidate.sample_id), int(candidate.global_node_id))
            for candidate in output.kept_candidates
        }
        target_nodes = set(zip(
            target.target_sample_index.detach().cpu().long().tolist(),
            target.target_node_index.detach().cpu().long().tolist(),
        ))
        batch = output.sample_tree_batch
        kept_samples = batch.kept_sample_ids.detach().cpu().long()
        graph_nodes = batch.batch.detach().cpu().long()
        global_nodes = batch.node_id_global.detach().cpu().long()
        node_is_precursor_root = batch.node_is_precursor_root.detach().cpu().bool()
        universe = {
            (int(kept_samples[int(graph_id)]), int(node_id))
            for graph_id, node_id in zip(graph_nodes.tolist(), global_nodes.tolist())
        }
        # (sample_id, global_node_id) pairs that represent the precursor ion
        # itself, so precursor coverage can be isolated from fragment peaks.
        precursor_pairs = {
            (int(kept_samples[int(graph_id)]), int(node_id))
            for graph_id, node_id, is_precursor in zip(
                graph_nodes.tolist(), global_nodes.tolist(), node_is_precursor_root.tolist()
            )
            if is_precursor
        }
        true_positive = len(selected_nodes & target_nodes)
        true_negative = len(universe - selected_nodes - target_nodes)
        covered = 0.0
        total = float(target.sample_peak_intensity.detach().cpu().clamp_min(0).sum().item())
        precursor_covered = 0.0
        precursor_total = 0.0
        sample_covered: Dict[int, float] = {}
        sample_total: Dict[int, float] = {}
        sample_precursor_covered: Dict[int, float] = {}
        sample_precursor_total: Dict[int, float] = {}
        sample_ids = target.target_sample_index.detach().cpu().long()
        peak_ids = target.target_peak_index.detach().cpu().long()
        node_ids = target.target_node_index.detach().cpu().long()
        for sample_id in range(int(target.num_samples)):
            start = int(target.sample_peak_ptr[sample_id])
            stop = int(target.sample_peak_ptr[sample_id + 1])
            for peak_id in range(start, stop):
                rows = (sample_ids == sample_id) & (peak_ids == peak_id)
                peak_node_ids = node_ids[rows].tolist()
                is_precursor_peak = any(
                    (sample_id, int(node_id)) in precursor_pairs for node_id in peak_node_ids
                )
                peak_intensity = float(
                    target.sample_peak_intensity[peak_id].detach().cpu().clamp_min(0).item()
                )
                is_hit = any(
                    (sample_id, int(node_id)) in selected_nodes for node_id in peak_node_ids
                )
                sample_total[sample_id] = sample_total.get(sample_id, 0.0) + peak_intensity
                if is_hit:
                    covered += peak_intensity
                    sample_covered[sample_id] = sample_covered.get(sample_id, 0.0) + peak_intensity
                if is_precursor_peak:
                    precursor_total += peak_intensity
                    sample_precursor_total[sample_id] = (
                        sample_precursor_total.get(sample_id, 0.0) + peak_intensity
                    )
                    if is_hit:
                        precursor_covered += peak_intensity
                        sample_precursor_covered[sample_id] = (
                            sample_precursor_covered.get(sample_id, 0.0) + peak_intensity
                        )

        precursor_target_nodes = target_nodes & precursor_pairs
        precursor_selected_nodes = selected_nodes & precursor_pairs
        precursor_true_positive = len(precursor_selected_nodes & precursor_target_nodes)
        precursor_true_negative = len(
            precursor_pairs - precursor_selected_nodes - precursor_target_nodes
        )
        metrics = {
            "peak_selection_accuracy": (true_positive + true_negative) / max(len(universe), 1),
            "peak_selection_precision": true_positive / max(len(selected_nodes), 1),
            "peak_selection_recall": true_positive / max(len(target_nodes), 1),
            "selected_peak_intensity_coverage": covered / max(total, 1e-12),
        }
        if precursor_pairs:
            precursor_selected_count = len(precursor_selected_nodes)
            metrics.update({
                "precursor/peak_selection_accuracy": (
                    (precursor_true_positive + precursor_true_negative)
                    / max(len(precursor_pairs), 1)
                ),
                "precursor/peak_selection_precision": (
                    precursor_true_positive / max(precursor_selected_count, 1)
                    if precursor_selected_count else 1.0 if not precursor_target_nodes else 0.0
                ),
                "precursor/peak_selection_recall": (
                    precursor_true_positive / max(len(precursor_target_nodes), 1)
                ),
                "precursor/selected_peak_intensity_coverage": (
                    precursor_covered / precursor_total if precursor_total > 0 else 1.0
                ),
            })

        # Per-sample breakdown, grouped by adduct type and CE bucket. Node
        # sets are partitioned by sample once, rather than rescanning each
        # full set per sample.
        universe_by_sample: Dict[int, set] = {}
        selected_by_sample: Dict[int, set] = {}
        target_by_sample: Dict[int, set] = {}
        precursor_by_sample: Dict[int, set] = {}
        precursor_target_by_sample: Dict[int, set] = {}
        precursor_selected_by_sample: Dict[int, set] = {}
        for pair in universe:
            universe_by_sample.setdefault(pair[0], set()).add(pair)
        for pair in selected_nodes:
            selected_by_sample.setdefault(pair[0], set()).add(pair)
        for pair in target_nodes:
            target_by_sample.setdefault(pair[0], set()).add(pair)
        for pair in precursor_pairs:
            precursor_by_sample.setdefault(pair[0], set()).add(pair)
        for pair in precursor_target_nodes:
            precursor_target_by_sample.setdefault(pair[0], set()).add(pair)
        for pair in precursor_selected_nodes:
            precursor_selected_by_sample.setdefault(pair[0], set()).add(pair)

        per_sample: Dict[str, Dict[int, float]] = {
            "peak_selection_accuracy": {},
            "peak_selection_precision": {},
            "peak_selection_recall": {},
            "selected_peak_intensity_coverage": {},
            "precursor/peak_selection_accuracy": {},
            "precursor/peak_selection_precision": {},
            "precursor/peak_selection_recall": {},
            "precursor/selected_peak_intensity_coverage": {},
        }
        for sample_id, sample_universe in universe_by_sample.items():
            sample_selected = selected_by_sample.get(sample_id, set())
            sample_target = target_by_sample.get(sample_id, set())
            sample_tp = len(sample_selected & sample_target)
            sample_tn = len(sample_universe - sample_selected - sample_target)
            per_sample["peak_selection_accuracy"][sample_id] = (
                (sample_tp + sample_tn) / max(len(sample_universe), 1)
            )
            per_sample["peak_selection_precision"][sample_id] = sample_tp / max(
                len(sample_selected), 1
            )
            per_sample["peak_selection_recall"][sample_id] = sample_tp / max(
                len(sample_target), 1
            )
        for sample_id, total_intensity in sample_total.items():
            per_sample["selected_peak_intensity_coverage"][sample_id] = (
                sample_covered.get(sample_id, 0.0) / max(total_intensity, 1e-12)
            )
        for sample_id, sample_precursor_pairs in precursor_by_sample.items():
            sample_precursor_selected = precursor_selected_by_sample.get(sample_id, set())
            sample_precursor_target = precursor_target_by_sample.get(sample_id, set())
            sample_precursor_tp = len(sample_precursor_selected & sample_precursor_target)
            sample_precursor_tn = len(
                sample_precursor_pairs - sample_precursor_selected - sample_precursor_target
            )
            per_sample["precursor/peak_selection_accuracy"][sample_id] = (
                (sample_precursor_tp + sample_precursor_tn) / max(len(sample_precursor_pairs), 1)
            )
            per_sample["precursor/peak_selection_precision"][sample_id] = (
                sample_precursor_tp / max(len(sample_precursor_selected), 1)
                if sample_precursor_selected
                else (1.0 if not sample_precursor_target else 0.0)
            )
            per_sample["precursor/peak_selection_recall"][sample_id] = sample_precursor_tp / max(
                len(sample_precursor_target), 1
            )
            precursor_total_intensity = sample_precursor_total.get(sample_id, 0.0)
            per_sample["precursor/selected_peak_intensity_coverage"][sample_id] = (
                sample_precursor_covered.get(sample_id, 0.0) / precursor_total_intensity
                if precursor_total_intensity > 0
                else 1.0
            )
        metrics.update(self._grouped_metric_means(per_sample, target))
        return metrics

    @torch.no_grad()
    def _intensity_similarity_metrics(self, intensity_output, target) -> Dict[str, float]:
        values: List[float] = []
        values_by_sample: Dict[int, float] = {}
        device = intensity_output.logit.device
        for sample_id in intensity_output.sample_index.detach().cpu().unique(sorted=True).tolist():
            mask = intensity_output.sample_index == int(sample_id)
            index = mask.nonzero(as_tuple=False).view(-1)
            truth = FragmentTreeIntensityTrainingLoss._target_weight_for_predictions(
                predicted_formula=intensity_output.formula_tensor[index],
                target=target, sample_id=int(sample_id), device=device,
            )
            prediction = intensity_output.logit[index].float().clamp_min(0)
            if truth.numel() and float(truth.norm().item()) > 0 and float(prediction.norm().item()) > 0:
                value = float(F.cosine_similarity(prediction[None], truth.float()[None]).item())
                values.append(value)
                values_by_sample[int(sample_id)] = value
        metrics = {"intensity_cosine_similarity": sum(values) / len(values) if values else 0.0}
        metrics.update(
            self._grouped_metric_means(
                {"intensity_cosine_similarity": values_by_sample}, target
            )
        )
        return metrics
