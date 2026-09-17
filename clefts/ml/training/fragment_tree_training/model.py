"""Batched teacher forcing and threshold-aligned action losses. No chemistry."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ...specgen.components.action.action_decoder import ActionPool, positive_mask


def absolute_filter_loss(logits: Tensor, positive: Tensor, eligible: Tensor,
                         threshold: float, negative_weight: float = 1.0) -> Tensor:
    shifted = logits - threshold
    pos = positive & eligible
    neg = ~positive & eligible
    return ((F.softplus(-shifted).masked_fill(~pos, 0).sum() / pos.sum().clamp_min(1))
            + negative_weight * F.softplus(shifted).masked_fill(~neg, 0).sum() / neg.sum().clamp_min(1))


def multi_positive_loss(logits: Tensor, positive: Tensor) -> Tensor:
    valid_rows = positive.any(dim=1)
    if logits.shape[0] == 0:
        return logits.sum() * 0
    safe_positive = positive.clone()
    safe_positive[~valid_rows, -1] = True
    numerator = torch.logsumexp(logits.masked_fill(~safe_positive, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    # Unsupervised states contribute no gradient, including all-empty batches.
    loss = denominator - numerator.masked_fill(~valid_rows, 0)
    return loss.masked_fill(~valid_rows, 0).sum() / valid_rows.sum().clamp_min(1)


@dataclass(frozen=True)
class ActionTrainingOutput:
    loss: Tensor
    absolute_loss: Tensor
    next_action_loss: Tensor
    absolute_logits: Tensor
    next_action_logits: Tensor
    next_positive_mask: Tensor
    pool: ActionPool
    metrics: dict[str, Tensor]
    downstream_output: object | None = None


class ActionFragmentTreeTrainingModel(nn.Module):
    def __init__(self, feature_model: nn.Module, absolute_weight: float = 1.0,
                 next_weight: float = 1.0, negative_weight: float = 1.0,
                 downstream_model: nn.Module | None = None, intensity_weight: float = 1.0) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.absolute_weight, self.next_weight, self.negative_weight = absolute_weight, next_weight, negative_weight
        self.intensity_weight = intensity_weight
        self.downstream_model = downstream_model

    def set_checkpoint_model_config(self, config: dict) -> None:
        self._checkpoint_model_config = dict(config)

    def get_params(self) -> dict:
        return dict(getattr(self,"_checkpoint_model_config",{}))

    def forward(self, data: object) -> ActionTrainingOutput:
        encoded = self.feature_model(data, training_pool=True)
        pool, conditions, absolute = encoded.pool, encoded.condition_h, encoded.absolute_logits
        s, a = absolute.shape
        eligible = data.action_tree_index[None, :] == data.sample_tree_index[:, None]
        positive = positive_mask(data.sample_positive_action_ptr, data.sample_positive_action_index, s, a)
        absolute_loss = absolute_filter_loss(absolute, positive, eligible, self.feature_model.threshold, self.negative_weight)
        p = data.teacher_state_sample_index.numel()
        sample = data.teacher_state_sample_index
        inverse = torch.full((s, a + 1), -1, dtype=torch.long, device=absolute.device)
        pool_ids = pool.action_index.masked_fill(~pool.valid, a)
        inverse.scatter_(1, pool_ids, torch.arange(pool.valid.shape[1], device=absolute.device).expand(s, -1))
        inverse[:, a] = -1
        counts = data.teacher_state_action_ptr[1:] - data.teacher_state_action_ptr[:-1]
        owner = torch.repeat_interleave(torch.arange(p, device=absolute.device), counts)
        position = torch.arange(owner.numel(), device=absolute.device) - data.teacher_state_action_ptr[owner]
        state = torch.full((p, self.feature_model.max_action_count), -1, dtype=torch.long, device=absolute.device)
        local = inverse[sample[owner], data.teacher_state_action_index]
        state[owner, position] = local
        state = state.masked_fill(state < 0, pool.valid.shape[1]).sort(dim=1).values
        state = state.masked_fill(state == pool.valid.shape[1], -1)
        logits, expansion = self.feature_model.decoder.score_states(pool, conditions, state, sample)
        next_counts = data.teacher_positive_action_ptr[1:] - data.teacher_positive_action_ptr[:-1]
        next_owner = torch.repeat_interleave(torch.arange(p, device=absolute.device), next_counts)
        next_local = inverse[sample[next_owner], data.teacher_positive_action_index]
        next_positive = torch.zeros_like(logits, dtype=torch.bool)
        next_positive[next_owner, next_local] = True
        next_positive[:, -1] = data.teacher_positive_eos
        if torch.any(next_positive & ~torch.isfinite(logits)):
            raise ValueError("Teacher positive is not a valid candidate under the shared compatibility rules")
        next_loss = multi_positive_loss(logits, next_positive)
        metrics = {"positive_action_recall@K": pool.positive_recall.mean(),
                   "positive_fraction_above_threshold": ((absolute >= self.feature_model.threshold) & positive).sum() / positive.sum().clamp_min(1),
                   "negative_fraction_below_threshold": ((absolute < self.feature_model.threshold) & ~positive & eligible).sum() / (~positive & eligible).sum().clamp_min(1),
                   "actions_before_filter": eligible.sum(dim=1).float().mean(),
                   "actions_after_filter": pool.valid.sum(dim=1).float().mean(),
                   "valid_candidate_count": torch.isfinite(logits[:, :-1]).sum(dim=1).float().mean() if p else absolute.new_zeros(()),
                   "normalized_replacement_rate": ((expansion.child_action_count <= counts[expansion.parent_state_index]) & expansion.valid).float().mean() if expansion.valid.numel() else absolute.new_zeros(())}
        if p:
            prediction = logits.argmax(dim=1)
            metrics["positive_set_top1_accuracy"] = next_positive.gather(1, prediction[:, None]).float().mean()
            top = logits.topk(min(5, logits.shape[1]), dim=1).indices
            metrics["positive_set_top5_recall"] = next_positive.gather(1, top).any(dim=1).float().mean()
            eos_prediction, eos_target = prediction == logits.shape[1] - 1, data.teacher_positive_eos
            correct = (eos_prediction & eos_target).sum()
            metrics["eos_precision"] = correct / eos_prediction.sum().clamp_min(1)
            metrics["eos_recall"] = correct / eos_target.sum().clamp_min(1)
        for k in (16, 32, 64, 128):
            ids = absolute.masked_fill(~eligible | (absolute < self.feature_model.threshold), -torch.inf).topk(min(k, a), dim=1).indices
            found = positive.gather(1, ids) & torch.isfinite(absolute.masked_fill(~eligible | (absolute < self.feature_model.threshold), -torch.inf).gather(1, ids))
            metrics[f"positive_action_recall@{k}"] = (found.sum(dim=1) / positive.sum(dim=1).clamp_min(1)).mean()
        downstream_output = None
        loss = self.absolute_weight * absolute_loss + self.next_weight * next_loss
        if self.downstream_model is not None:
            if data.downstream is None:
                raise ValueError("Downstream training needs stored materialized graphs")
            downstream_output = (self.downstream_model(data.downstream, action_h=encoded.action_h, condition_h=conditions)
                                 if getattr(self.downstream_model,"requires_action_features",False) else self.downstream_model(data.downstream))
            downstream_loss = downstream_output["loss"] if isinstance(downstream_output, dict) else downstream_output.loss
            loss = loss + self.intensity_weight * downstream_loss
        return ActionTrainingOutput(loss, absolute_loss, next_loss, absolute, logits, next_positive, pool, metrics, downstream_output)
