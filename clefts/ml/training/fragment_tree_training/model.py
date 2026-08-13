from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...input.training_fragment_tree_structure import TrainingFragmentTreeStructure
from ...specgen.fragment_tree_candidate_selector import FragmentTreeCandidateSelectionOutput
from ...specgen.fragment_tree_formula_intensity_model import FragmentTreeFormulaIntensityPredictor


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

    def forward(
        self,
        output: FragmentTreeCandidateSelectionOutput,
        target: TrainingFragmentTreeStructure,
    ) -> Tensor:
        device = output.keep_logit.device
        self._validate_state_targets(target)
        losses: List[Tensor] = []

        if target.target_node_index.numel() > 0:
            losses.append(self._peak_fragment_loss(output, target, device=device))
            losses.append(self._keep_negative_loss(output, target, device=device))
            losses.append(self._state_loss(output, target, device=device))

        cleave_target, cleave_mask = self._build_cleave_targets(output, target, device=device)
        if cleave_mask.any():
            losses.append(
                self._balanced_node_bce_loss(
                    logit=output.cleave_logit,
                    target=cleave_target,
                    mask=cleave_mask,
                    output=output,
                    device=device,
                )
            )

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
    ) -> Tensor:
        batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
        losses: List[Tensor] = []

        for peak_key in self._target_peak_keys(target, device=device):
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

        if len(losses) == 0:
            return output.keep_logit.sum() * 0.0
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
    ) -> Tensor:
        batch = output.sample_tree_batch
        batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
        losses: List[Tensor] = []
        weights: List[Tensor] = []

        for peak_key in self._target_peak_keys(target, device=device):
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


class FragmentTreeIntensityTrainingLoss(nn.Module):
    """Loss for per-formula intensities over selected formula nodes."""

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
            sample_target_mask = target.target_sample_index.to(device).long() == sample_id
            if sample_target_mask.any():
                max_target_intensity = target.target_intensity.to(device).float()[
                    sample_target_mask
                ].max().clamp_min(1e-12)
                target_intensity = (target_weight / max_target_intensity).clamp(0.0, 1.0)
            else:
                target_intensity = target_weight
            losses.append(
                F.smooth_l1_loss(
                    intensity_output.logit[pred_index],
                    target_intensity,
                )
            )

        if len(losses) == 0:
            return intensity_output.logit.sum() * 0.0
        return torch.stack(losses).mean()

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

    def set_checkpoint_model_config(self, model_config: Dict[str, Any]) -> None:
        self._checkpoint_model_config = dict(model_config)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._checkpoint_model_config)

    def forward(self, batch: TrainingFragmentTreeStructure):
        output = self.candidate_selector(batch)
        selection_loss = self.loss_fn(output, batch)
        if self.intensity_predictor is None:
            intensity_output = None
            intensity_loss = selection_loss.detach() * 0.0
        else:
            intensity_output = self.intensity_predictor.forward_candidate_output(output)
            intensity_loss = self.intensity_loss_fn(intensity_output, batch)
        loss = selection_loss + intensity_loss
        return {
            "loss": loss,
            "selection_loss": selection_loss,
            "intensity_loss": intensity_loss,
            "candidate_output": output,
            "intensity_output": intensity_output,
        }
