"""Scalar reference paths retained for value/gradient regression tests."""
from typing import *
import torch
import torch.nn.functional as F
from torch import Tensor
from clefts.ml.training.fragment_tree_training.model import *
from clefts.ml.specgen.fragment_tree_formula_intensity_model import *


class ScalarRanking(PairwiseEdgeIntensityRankingLoss):

    def forward(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure) -> Tensor:
        logit = output.edge_absolute_logit
        if logit.numel() == 0:
            return logit.sum() * 0.0
        if not hasattr(output.sample_tree_batch, 'edge_id_global'):
            raise ValueError('sample_tree_batch must expose edge_id_global.')
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
                alternatives.update((int(v) for v in indexes.tolist()))
                evidence = torch.logsumexp(logit[indexes], dim=0)
                intensity = intensities[formula_peak[int(group_id)]]
                group_rows.append((evidence, intensity))
            group_rows.sort(key=lambda item: float(item[1]), reverse=True)
            rank_weights = logit.new_empty((0,))
            sqrt_values = logit.new_empty((0,))
            if group_rows:
                ranks = torch.arange(1, len(group_rows) + 1, dtype=torch.float32, device=logit.device)
                reciprocal_ranks = 1.0 / ranks
                rank_weights = reciprocal_ranks / reciprocal_ranks.sum()
                sqrt_values = torch.sqrt(torch.stack([item[1] for item in group_rows]).clamp_min(0.0))
            background = torch.tensor([index for index in (sample_ids == int(sample_id)).nonzero(as_tuple=False).flatten().tolist() if int(index) not in alternatives], dtype=torch.long, device=logit.device)
            hard_negatives = background[torch.topk(logit[background], k=min(self.background_partners, int(background.numel()))).indices] if background.numel() and group_rows else background[:0]
            for i in range(len(group_rows)):
                offsets = self._select_tiered_partners(sqrt_values=sqrt_values, anchor_index=i, remaining_budget=self.top_n)
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

class ScalarSelection(FragmentTreeSelectionTrainingLoss):

    def forward(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure) -> Tensor:
        device = output.keep_logit.device
        self._validate_state_targets(target)
        losses: List[Tensor] = []
        if target.target_node_index.numel() > 0:
            batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
            peak_keys = self._target_peak_keys(target, device=device)
            losses.append(self._peak_fragment_loss(output, target, device=device, peak_keys=peak_keys, batch_node_by_sample_node=batch_node_by_sample_node))
            losses.append(self._keep_negative_loss(output, target, device=device))
            losses.append(self._state_loss(output, target, device=device, peak_keys=peak_keys, batch_node_by_sample_node=batch_node_by_sample_node))
            precursor_loss = self._precursor_keep_loss(output, target, device=device, peak_keys=peak_keys, batch_node_by_sample_node=batch_node_by_sample_node)
            if precursor_loss is not None:
                losses.append(precursor_loss)
        (cleave_target, cleave_mask) = self._build_cleave_targets(output, target, device=device)
        if cleave_mask.any():
            losses.append(self._expand_node_ranking_loss(output, target, cleave_target=cleave_target, cleave_mask=cleave_mask, device=device))
        edge_group_loss = self._edge_group_coverage_loss(output, target, device=device)
        edge_negative_loss = self._edge_negative_loss(output, target, device=device)
        losses.append(edge_group_loss)
        losses.append(edge_negative_loss)
        if len(losses) == 0:
            return output.keep_logit.sum() * 0.0
        return torch.stack(losses).mean()

    def _peak_fragment_loss(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure, *, device: torch.device, peak_keys: Optional[List[Tuple[int, int]]]=None, batch_node_by_sample_node: Optional[Dict[Tuple[int, int], int]]=None) -> Tensor:
        if batch_node_by_sample_node is None:
            batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
        if peak_keys is None:
            peak_keys = self._target_peak_keys(target, device=device)
        losses: List[Tensor] = []
        weights: List[Tensor] = []
        for peak_key in peak_keys:
            row_index = self._target_peak_mask(target, peak_key=peak_key, device=device)
            batch_node_indexes = self._unique_batch_node_indexes(target_sample_index=target.target_sample_index[row_index].to(device), target_node_index=target.target_node_index[row_index].to(device), batch_node_by_sample_node=batch_node_by_sample_node)
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

    def _precursor_keep_loss(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure, *, device: torch.device, peak_keys: Optional[List[Tuple[int, int]]]=None, batch_node_by_sample_node: Optional[Dict[Tuple[int, int], int]]=None) -> Optional[Tensor]:
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
            batch_node_indexes = self._unique_batch_node_indexes(target_sample_index=target.target_sample_index[row_index].to(device), target_node_index=target.target_node_index[row_index].to(device), batch_node_by_sample_node=batch_node_by_sample_node)
            if batch_node_indexes.numel() == 0:
                continue
            precursor_indexes = batch_node_indexes[node_is_precursor_root[batch_node_indexes]]
            if precursor_indexes.numel() == 0:
                continue
            losses.append(F.softplus(-torch.logsumexp(output.keep_logit[precursor_indexes], dim=0)))
        if len(losses) == 0:
            return None
        return torch.stack(losses).mean()

    def _keep_negative_loss(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure, *, device: torch.device) -> Tensor:
        batch = output.sample_tree_batch
        kept_sample_ids = batch.kept_sample_ids.to(device).long()
        graph_index_by_node = batch.batch.to(device).long()
        node_global_ids = batch.node_id_global.to(device).long()
        node_is_precursor_root = batch.node_is_precursor_root.to(device).bool()
        positive_pairs = self._target_sample_node_pairs(target_sample_index=target.target_sample_index.to(device).long(), target_node_index=target.target_node_index.to(device).long())
        sample_losses: List[Tensor] = []
        for graph_index in range(int(kept_sample_ids.numel())):
            sample_id = int(kept_sample_ids[graph_index].detach().cpu().item())
            node_mask = (graph_index_by_node == graph_index) & ~node_is_precursor_root
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
            sample_losses.append(F.binary_cross_entropy_with_logits(output.keep_logit[index_tensor], output.keep_logit.new_zeros((index_tensor.numel(),))))
        if len(sample_losses) == 0:
            return output.keep_logit.sum() * 0.0
        return torch.stack(sample_losses).mean()

    def _state_loss(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure, *, device: torch.device, peak_keys: Optional[List[Tuple[int, int]]]=None, batch_node_by_sample_node: Optional[Dict[Tuple[int, int], int]]=None) -> Tensor:
        batch = output.sample_tree_batch
        if batch_node_by_sample_node is None:
            batch_node_by_sample_node = self._batch_node_by_sample_node(output, device=device)
        if peak_keys is None:
            peak_keys = self._target_peak_keys(target, device=device)
        losses: List[Tensor] = []
        weights: List[Tensor] = []
        for peak_key in peak_keys:
            row_index = self._target_peak_mask(target, peak_key=peak_key, device=device)
            batch_node_indexes = self._unique_batch_node_indexes(target_sample_index=target.target_sample_index[row_index].to(device), target_node_index=target.target_node_index[row_index].to(device), batch_node_by_sample_node=batch_node_by_sample_node)
            if batch_node_indexes.numel() == 0:
                continue
            fragment_prob = torch.softmax(output.keep_logit[batch_node_indexes], dim=0).detach()
            prob_by_batch_node = {int(batch_node): fragment_prob[index] for (index, batch_node) in enumerate(batch_node_indexes.detach().cpu().tolist())}
            peak_weight = target.target_intensity[row_index].to(device).float().max()
            seen_nodes: set[int] = set()
            for row in row_index.detach().cpu().tolist():
                row = int(row)
                key = (int(target.target_sample_index[row].detach().cpu().item()), int(target.target_node_index[row].detach().cpu().item()))
                batch_node_index = batch_node_by_sample_node.get(key)
                if batch_node_index is None or batch_node_index in seen_nodes:
                    continue
                seen_nodes.add(batch_node_index)
                role_index = 0 if bool(batch.node_is_precursor_root[batch_node_index].detach().cpu().item()) else 1
                main_adduct_index = int(batch.node_main_adduct_type_index[batch_node_index].detach().cpu().item())
                node_weight = peak_weight * prob_by_batch_node[int(batch_node_index)]
                losses.append(self._masked_cross_entropy(output.ion_logit[batch_node_index], int(target.target_ion_index[row].detach().cpu().item()), output.ion_valid_mask_by_role_adduct[role_index, main_adduct_index]))
                weights.append(node_weight)
                losses.append(self._masked_cross_entropy(output.unsaturation_logit[batch_node_index], int(target.target_unsaturation_index[row].detach().cpu().item()), output.unsaturation_valid_mask_by_role_adduct[role_index, main_adduct_index]))
                weights.append(node_weight)
                losses.append(self._masked_cross_entropy(output.radical_logit[batch_node_index], int(target.target_radical_index[row].detach().cpu().item()), output.radical_valid_mask_by_role_adduct[role_index, main_adduct_index]))
                weights.append(node_weight)
        return self._weighted_mean(losses, weights, output.keep_logit)

    def _edge_group_coverage_loss(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure, *, device: torch.device) -> Tensor:
        if target.target_edge_index.numel() == 0 or output.edge_cleave_logit.numel() == 0:
            return output.edge_cleave_logit.sum() * 0.0
        batch_edge_by_sample_edge = self._batch_edge_by_sample_edge(output, device=device)
        target_edge_index = target.target_edge_index.to(device).long()
        target_edge_group_index = target.target_edge_group_index.to(device).long()
        if target_edge_index.size(1) != target_edge_group_index.numel():
            raise ValueError('target_edge_index and target_edge_group_index are misaligned.')
        losses: List[Tensor] = []
        seen: set[Tuple[int, int]] = set()
        for row in range(int(target_edge_group_index.numel())):
            group_key = (int(target_edge_index[0, row].detach().cpu().item()), int(target_edge_group_index[row].detach().cpu().item()))
            if group_key in seen:
                continue
            seen.add(group_key)
            group_mask = (target_edge_index[0] == int(group_key[0])) & (target_edge_group_index == int(group_key[1]))
            batch_edge_indexes: List[int] = []
            for edge_id in target_edge_index[1, group_mask].detach().cpu().tolist():
                batch_edge_index = batch_edge_by_sample_edge.get((int(group_key[0]), int(edge_id)))
                if batch_edge_index is not None:
                    batch_edge_indexes.append(int(batch_edge_index))
            if not batch_edge_indexes:
                continue
            index_tensor = torch.tensor(sorted(set(batch_edge_indexes)), dtype=torch.long, device=device)
            losses.append(F.softplus(-torch.logsumexp(output.edge_cleave_logit[index_tensor], dim=0)))
        if len(losses) == 0:
            return output.edge_cleave_logit.sum() * 0.0
        return torch.stack(losses).mean()

    def _edge_negative_loss(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure, *, device: torch.device) -> Tensor:
        if output.edge_cleave_logit.numel() == 0 or target.target_edge_index.numel() == 0:
            return output.edge_cleave_logit.sum() * 0.0
        batch = output.sample_tree_batch
        target_edge_index = target.target_edge_index.to(device).long()
        positive_pairs = {(int(sample_id), int(edge_id)) for (sample_id, edge_id) in target_edge_index.detach().cpu().t().tolist()}
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
            sample_losses.append(F.binary_cross_entropy_with_logits(output.edge_cleave_logit[index_tensor], output.edge_cleave_logit.new_zeros((index_tensor.numel(),))))
        if len(sample_losses) == 0:
            return output.edge_cleave_logit.sum() * 0.0
        return torch.stack(sample_losses).mean()

    def _build_cleave_targets(self, output: FragmentTreeCandidateSelectionOutput, target: TrainingFragmentTreeStructure, *, device: torch.device) -> Tuple[Tensor, Tensor]:
        batch = output.sample_tree_batch
        cleave_target = torch.zeros_like(output.cleave_logit, dtype=torch.float32, device=device)
        cleave_mask = ~batch.node_is_precursor_root.to(device).bool()
        if target.target_expand_node_index.numel() == 0:
            return (cleave_target, cleave_mask)
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
            sample_parts.append(torch.full_like(nodes, int(assignment_samples[assignment_id].item())))
        if not node_parts:
            return (cleave_target, cleave_mask)
        positive_pairs = self._target_sample_node_pairs(target_sample_index=torch.cat(sample_parts), target_node_index=torch.cat(node_parts))
        self._apply_positive_node_pairs(target_tensor=cleave_target, positive_pairs=positive_pairs, batch=batch, device=device)
        return (cleave_target, cleave_mask)

class ScalarIntensity(FragmentTreeFormulaIntensityPredictor):

    def forward_candidate_output(self, candidate_output: FragmentTreeCandidateSelectionOutput) -> FormulaIntensityTrainingOutput:
        if self.formula_node_input is None:
            raise RuntimeError('FragmentTreeFormulaIntensityPredictor requires feature_model to train formula-node intensities.')
        candidates = candidate_output.kept_candidates
        if len(candidates) == 0:
            device = candidate_output.keep_logit.device
            return FormulaIntensityTrainingOutput(sample_index=torch.empty((0,), dtype=torch.long, device=device), formula_tensor=torch.empty((0, self.formula_dim), device=device), logit=candidate_output.keep_logit.new_empty((0,)), candidates=[], presence_logit=candidate_output.keep_logit.new_empty((0,)), abundance_logit=candidate_output.keep_logit.new_empty((0,)))
        device = candidate_output.keep_logit.device
        candidate_sample_index = torch.tensor([int(candidate.sample_id) for candidate in candidates], dtype=torch.long, device=device)
        candidate_repr = self._candidate_formula_node_repr(candidate_output, candidates)
        encoded = self._encode_formula_nodes_by_sample(candidate_repr, candidate_sample_index)
        candidate_presence_logit = self.formula_presence_head(encoded).squeeze(-1)
        candidate_abundance_logit = self.formula_node_head(encoded).squeeze(-1)
        selection_probability = torch.stack([self._candidate_probability(candidate, device=device) for candidate in candidates])
        candidate_intensity = self.relative_intensity(candidate_presence_logit, candidate_abundance_logit, candidate_sample_index, selection_probability=selection_probability)
        grouped_indexes: Dict[Tuple[int, Tuple[float, ...]], List[int]] = {}
        for (index, candidate) in enumerate(candidates):
            formula_key = tuple((float(value) for value in candidate.formula_tensor.tolist()))
            grouped_indexes.setdefault((int(candidate.sample_id), formula_key), []).append(index)
        sample_ids: List[int] = []
        formula_rows: List[Tensor] = []
        formula_intensities: List[Tensor] = []
        formula_presence_logits: List[Tensor] = []
        formula_abundance_logits: List[Tensor] = []
        groups: List[List[FragmentIonCandidate]] = []
        for ((sample_id, _), indexes) in sorted(grouped_indexes.items(), key=lambda item: item[0]):
            index = torch.tensor(indexes, dtype=torch.long, device=device)
            group = [candidates[item] for item in indexes]
            sample_ids.append(int(sample_id))
            formula_rows.append(group[0].formula_tensor.to(device).float())
            formula_intensities.append(candidate_intensity[index].sum())
            group_probability = selection_probability[index]
            group_presence = (torch.sigmoid(candidate_presence_logit[index]) * group_probability).sum() / group_probability.sum().clamp_min(1e-12)
            formula_presence_logits.append(torch.logit(group_presence.clamp(1e-06, 1.0 - 1e-06)))
            formula_abundance_logits.append((candidate_abundance_logit[index] * group_probability).sum() / group_probability.sum().clamp_min(1e-12))
            groups.append(group)
        sample_index = torch.tensor(sample_ids, dtype=torch.long, device=device)
        return FormulaIntensityTrainingOutput(sample_index=sample_index, formula_tensor=torch.stack(formula_rows, dim=0), logit=torch.stack(formula_intensities), candidates=groups, presence_logit=torch.stack(formula_presence_logits), abundance_logit=torch.stack(formula_abundance_logits))

    @staticmethod
    def relative_intensity(presence_logit: Tensor, abundance_logit: Tensor, sample_index: Tensor, eps: float=1e-12, selection_probability: Optional[Tensor]=None) -> Tensor:
        """Normalize selection- and presence-gated candidate abundance."""
        intensity = torch.zeros_like(abundance_logit)
        if selection_probability is None:
            selection_probability = torch.ones_like(abundance_logit)
        for sample_id in sample_index.detach().cpu().unique(sorted=True).tolist():
            mask = sample_index == int(sample_id)
            stable_abundance = abundance_logit[mask] - abundance_logit[mask].max()
            log_raw = torch.log(selection_probability[mask].clamp_min(float(eps))) + F.logsigmoid(presence_logit[mask]) + stable_abundance
            intensity[mask] = torch.softmax(log_raw, dim=0)
        return intensity

    def _candidate_formula_node_repr(self, candidate_output: FragmentTreeCandidateSelectionOutput, candidates: List[FragmentIonCandidate]) -> Tensor:
        device = candidate_output.keep_logit.device
        fragment_rows = []
        batch = candidate_output.sample_tree_batch
        edge_dst = batch.edge_index[1].to(device).long()
        for candidate in candidates:
            fragment_emb = candidate_output.sample_tree_batch.x[candidate.batch_node_index]
            state_emb = torch.cat([self.ion_embedding(torch.tensor(candidate.ion_index, dtype=torch.long, device=device)), self.unsaturation_embedding(torch.tensor(candidate.unsaturation_index, dtype=torch.long, device=device)), self.radical_embedding(torch.tensor(candidate.radical_index, dtype=torch.long, device=device))], dim=0)
            incoming = (edge_dst == int(candidate.batch_node_index)).nonzero(as_tuple=False).flatten()
            if incoming.numel():
                edge_feature = batch.edge_attr[incoming].mean(dim=0)
                absolute_score = candidate_output.edge_absolute_logit[incoming].mean().view(1)
                competition_score = candidate_output.edge_cleave_logit[incoming].mean().view(1)
            else:
                edge_feature = batch.edge_attr.new_zeros((batch.edge_attr.size(-1),))
                absolute_score = edge_feature.new_zeros((1,))
                competition_score = edge_feature.new_zeros((1,))
            edge_repr = self.candidate_edge_input(torch.cat([edge_feature, absolute_score, competition_score], dim=0))
            fragment_rows.append(self.formula_node_input(torch.cat([fragment_emb, state_emb, edge_repr], dim=0)))
        return torch.stack(fragment_rows, dim=0)

    def _encode_formula_nodes_by_sample(self, node_repr: Tensor, sample_index: Tensor) -> Tensor:
        if node_repr.numel() == 0:
            return node_repr
        encoded = torch.empty_like(node_repr)
        for sample_id in sample_index.detach().cpu().unique(sorted=True).tolist():
            mask = sample_index == int(sample_id)
            encoded[mask] = self.formula_node_encoder(node_repr[mask][None, :, :]).squeeze(0)
        return encoded
