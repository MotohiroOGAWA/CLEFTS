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


def multi_positive_loss(logits: Tensor, positive: Tensor, target_weight: Tensor | None = None,
                        negative_weight: float = 0.2, minimum_positive_weight: float = 0.05) -> Tensor:
    """Independent weighted BCE; chemically invalid candidates have no loss."""
    valid=torch.isfinite(logits)
    if torch.any(positive & ~valid):raise ValueError("Invalid teacher-positive branch")
    safe=logits.masked_fill(~valid,0)
    weights=torch.ones_like(safe) if target_weight is None else target_weight
    weights=weights+minimum_positive_weight
    pos=F.softplus(-safe)*weights*positive
    neg=F.softplus(safe)*(valid & ~positive)
    return pos.sum()/(weights*positive).sum().clamp_min(1e-8)+negative_weight*neg.sum()/(valid & ~positive).sum().clamp_min(1)


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
    absolute_intensity_loss: Tensor | None = None


class ActionFragmentTreeTrainingModel(nn.Module):
    def __init__(self, feature_model: nn.Module, absolute_weight: float = 1.0,
                 next_weight: float = 1.0, negative_weight: float = 0.2,
                 downstream_model: nn.Module | None = None, intensity_weight: float = 1.0,
                 absolute_intensity_weight: float = 1.0, minimum_positive_weight: float = 0.05) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.absolute_weight, self.next_weight, self.negative_weight = absolute_weight, next_weight, negative_weight
        self.minimum_positive_weight=minimum_positive_weight
        self.intensity_weight = intensity_weight
        self.absolute_intensity_weight = absolute_intensity_weight
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
        precursor_owner=torch.repeat_interleave(torch.arange(s,device=absolute.device),data.sample_precursor_row_ptr[1:]-data.sample_precursor_row_ptr[:-1])
        precursor_owner=torch.repeat_interleave(precursor_owner,data.precursor_row_action_ptr[1:]-data.precursor_row_action_ptr[:-1])
        eligible[precursor_owner,data.precursor_row_action_index]=positive[precursor_owner,data.precursor_row_action_index]
        # Attribute observed peak salience to stored teacher states/actions.
        # Max aggregation avoids counting ambiguous assignments repeatedly.
        state_target=absolute.new_zeros(data.teacher_node_sample_index.numel())
        if data.downstream is not None and getattr(data.downstream,'target_intensity',None) is not None:
            downstream=data.downstream
            nodes=absolute.new_zeros(len(downstream.decoded.compounds))
            nodes.scatter_reduce_(0,downstream.ion_node_index,downstream.target_intensity[downstream.ion_formula_index],reduce='amax',include_self=True)
            mapped=data.state_fragment_node_index
            if mapped.numel():
                state_target=torch.where(mapped>=0,nodes[mapped.clamp_min(0)],state_target)
            state_observed=state_target.clone()
            for _ in range(self.feature_model.decoder.max_decode_steps+self.feature_model.max_action_count+1):
                updated=state_target.clone()
                updated.scatter_reduce_(0,data.transition_parent_state_index,state_target[data.transition_child_state_index],reduce='amax',include_self=True)
                if torch.equal(updated,state_target):break
                state_target=updated
        if data.downstream is None or getattr(data.downstream,'target_intensity',None) is None:state_observed=state_target.clone()
        counts_all=data.teacher_node_action_ptr[1:]-data.teacher_node_action_ptr[:-1]
        owner_all=torch.repeat_interleave(torch.arange(state_target.numel(),device=absolute.device),counts_all)
        target=absolute.new_zeros((s,a))
        target.view(-1).scatter_reduce_(0,data.teacher_node_sample_index[owner_all]*a+data.teacher_node_action_index,state_target[owner_all],reduce='amax',include_self=True)
        absolute_loss=absolute_filter_loss(absolute,positive,eligible & torch.isfinite(absolute),self.feature_model.threshold,self.negative_weight)
        absolute_intensity_loss=multi_positive_loss(absolute.masked_fill(~eligible,-torch.inf),positive & eligible & torch.isfinite(absolute),target,self.negative_weight)
        p = data.teacher_node_sample_index.numel()
        sample = data.teacher_node_sample_index
        inverse = torch.full((s, a + 1), -1, dtype=torch.long, device=absolute.device)
        pool_ids = pool.action_index.masked_fill(~pool.valid, a)
        inverse.scatter_(1, pool_ids, torch.arange(pool.valid.shape[1], device=absolute.device).expand(s, -1))
        inverse[:, a] = -1
        counts = data.teacher_node_action_ptr[1:] - data.teacher_node_action_ptr[:-1]
        owner = torch.repeat_interleave(torch.arange(p, device=absolute.device), counts)
        position = torch.arange(owner.numel(), device=absolute.device) - data.teacher_node_action_ptr[owner]
        state = torch.full((p, self.feature_model.max_action_count), -1, dtype=torch.long, device=absolute.device)
        local = inverse[sample[owner], data.teacher_node_action_index]
        if torch.any(local<0):raise ValueError("Teacher state action missing from training pool")
        state[owner, position] = local
        state = state.masked_fill(state < 0, pool.valid.shape[1]).sort(dim=1).values
        state = state.masked_fill(state == pool.valid.shape[1], -1)
        logits, expansion = self.feature_model.decoder.score_states(pool, conditions, state, sample)
        next_counts = data.teacher_positive_action_ptr[1:] - data.teacher_positive_action_ptr[:-1]
        next_owner = torch.repeat_interleave(torch.arange(p, device=absolute.device), next_counts)
        next_local = inverse[sample[next_owner], data.teacher_positive_action_index]
        if torch.any(next_local<0):raise ValueError("Teacher-positive action missing from training pool")
        next_positive = torch.zeros_like(logits, dtype=torch.bool)
        next_positive[next_owner, next_local] = True

        if torch.any(next_positive & ~torch.isfinite(logits)):
            raise ValueError("Teacher positive is not a valid candidate under the shared compatibility rules")
        next_weights=absolute.new_zeros(logits.shape)
        next_weights[next_owner,next_local]=data.teacher_positive_action_weight
        next_loss=multi_positive_loss(logits,next_positive,next_weights,self.negative_weight,self.minimum_positive_weight)
        finite_eligible=eligible & torch.isfinite(absolute)
        # Recall is measured before forced-positive insertion, matching inference.
        kept=torch.isfinite(absolute) & (absolute>self.feature_model.threshold)
        ids=absolute.masked_fill(~kept,-torch.inf).argsort(dim=1,descending=True)[:,:min(self.feature_model.top_k,a)]
        selected=torch.zeros_like(kept)
        selected.scatter_(1,ids,kept.gather(1,ids))
        metrics = {"intensity_recall_at_filter":((target*selected).sum(1)/target.sum(1).clamp_min(1e-8)).mean(),"absolute_intensity_loss":absolute_intensity_loss,"positive_action_recall@K": pool.positive_recall.mean(),
                   "positive_fraction_above_threshold": ((absolute > self.feature_model.threshold) & positive).sum() / positive.sum().clamp_min(1),
                   "negative_fraction_below_threshold": ((absolute <= self.feature_model.threshold) & ~positive & finite_eligible).sum() / (~positive & finite_eligible).sum().clamp_min(1),
                   "actions_before_filter": eligible.sum(dim=1).float().mean(),
                   "actions_after_filter": selected.sum(dim=1).float().mean(),
                   "training_pool_actions": pool.valid.sum(dim=1).float().mean(),
                   "valid_candidate_count": torch.isfinite(logits).sum(dim=1).float().mean() if p else absolute.new_zeros(()),
                   "normalized_replacement_rate": ((expansion.child_action_count <= counts[expansion.parent_state_index]) & expansion.valid).float().mean() if expansion.valid.numel() else absolute.new_zeros(())}
        predicted=(logits.sigmoid()>=self.feature_model.decoder.prediction_threshold)&torch.isfinite(logits)
        correct=(predicted & next_positive).sum()
        metrics.update(branch_positive_recall=correct/next_positive.sum().clamp_min(1),
            branch_positive_precision=correct/predicted.sum().clamp_min(1),
            mean_positive_actions_per_node=next_positive.sum().float()/max(p,1),
            mean_valid_actions_per_node=torch.isfinite(logits).sum().float()/max(p,1),
            mean_predicted_actions_per_node=predicted.sum().float()/max(p,1))
        # The decode safety limit is often much larger than the fragmenter's
        # configured tree depth and must not create meaningless empty charts.
        maximum_reported_depth=min(self.feature_model.decoder.max_decode_steps,self.feature_model.max_action_count)
        for depth in range(maximum_reported_depth+1):
            mask=data.teacher_node_ms2_depth==depth
            metrics[f'recall_depth_{depth}']=(predicted[mask]&next_positive[mask]).sum()/next_positive[mask].sum().clamp_min(1)
        for k in (16, 32, 64, 128):
            ids = absolute.masked_fill(~eligible | (absolute <= self.feature_model.threshold), -torch.inf).topk(min(k, a), dim=1).indices
            found = positive.gather(1, ids) & torch.isfinite(absolute.masked_fill(~eligible | (absolute <= self.feature_model.threshold), -torch.inf).gather(1, ids))
            metrics[f"positive_action_recall@{k}"] = (found.sum(dim=1) / positive.sum(dim=1).clamp_min(1)).mean()
        downstream_output = None
        loss = self.absolute_weight * (absolute_loss+self.absolute_intensity_weight*absolute_intensity_loss) + self.next_weight * next_loss
        if self.downstream_model is not None:
            if data.downstream is None:
                raise ValueError("Downstream training needs stored materialized graphs")
            downstream_output = (self.downstream_model(data.downstream, action_h=encoded.action_h, condition_h=conditions,source_embeddings=encoded.source_embeddings)
                                 if getattr(self.downstream_model,"requires_action_features",False) else self.downstream_model(data.downstream))
            downstream_loss = downstream_output["loss"] if isinstance(downstream_output, dict) else downstream_output.loss
            loss = loss + self.intensity_weight * downstream_loss
        if downstream_output is not None and getattr(data.downstream,'target_intensity',None) is not None:
            predicted=downstream_output.intensity
            observed=data.downstream.target_intensity
            group=data.downstream.formula_sample_index
            target_max=absolute.new_zeros(s).scatter_reduce_(0,group,observed,reduce='amax',include_self=True)
            observed=observed/target_max[group].clamp_min(1e-8)
            dot=absolute.new_zeros(s).scatter_add_(0,group,predicted*observed)
            pp=absolute.new_zeros(s).scatter_add_(0,group,predicted.square())
            tt=absolute.new_zeros(s).scatter_add_(0,group,observed.square())
            valid_spectrum=tt>0
            cosine=dot/(pp*tt).sqrt().clamp_min(1e-8)
            metrics['teacher_spectrum_cosine_similarity']=cosine[valid_spectrum].mean() if torch.any(valid_spectrum) else absolute.new_zeros(())
            metrics['intensity_mae']=(predicted-observed).abs().mean() if predicted.numel() else absolute.new_zeros(())
        return ActionTrainingOutput(loss, absolute_loss, next_loss, absolute, logits, next_positive, pool, metrics, downstream_output,absolute_intensity_loss)
