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


def multi_positive_loss(logits: Tensor, positive: Tensor, target_weight: Tensor | None = None) -> Tensor:
    valid_rows = positive.any(dim=1)
    if logits.shape[0] == 0:
        return logits.sum() * 0
    safe_positive = positive.clone()
    safe_positive[~valid_rows, -1] = True
    if target_weight is not None:
        weights=target_weight.masked_fill(~positive,0)
        weights=weights/weights.sum(dim=1,keepdim=True).clamp_min(1e-8)
        safe_logits=logits.masked_fill(~valid_rows[:,None],0)
        log_probs=F.log_softmax(safe_logits,dim=1).masked_fill(~positive,0)
        return -(weights*log_probs).sum(dim=1).sum()/valid_rows.sum().clamp_min(1)
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
    absolute_intensity_loss: Tensor | None = None


class ActionFragmentTreeTrainingModel(nn.Module):
    def __init__(self, feature_model: nn.Module, absolute_weight: float = 1.0,
                 next_weight: float = 1.0, negative_weight: float = 1.0,
                 downstream_model: nn.Module | None = None, intensity_weight: float = 1.0,
                 absolute_intensity_weight: float = 1.0) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.absolute_weight, self.next_weight, self.negative_weight = absolute_weight, next_weight, negative_weight
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
        # Attribute observed peak salience to stored teacher states/actions.
        # Max aggregation avoids counting ambiguous assignments repeatedly.
        state_target=absolute.new_zeros(data.teacher_state_sample_index.numel())
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
        counts_all=data.teacher_state_action_ptr[1:]-data.teacher_state_action_ptr[:-1]
        owner_all=torch.repeat_interleave(torch.arange(state_target.numel(),device=absolute.device),counts_all)
        target=absolute.new_zeros((s,a))
        target.view(-1).scatter_reduce_(0,data.teacher_state_sample_index[owner_all]*a+data.teacher_state_action_index,state_target[owner_all],reduce='amax',include_self=True)
        absolute_loss=absolute_filter_loss(absolute,positive,eligible & torch.isfinite(absolute),self.feature_model.threshold,self.negative_weight)
        row_logits=encoded.precursor_absolute_logits
        row_target=target[encoded.precursor_sample_index].masked_fill(~torch.isfinite(row_logits),0)
        # Root/precursor without further cleavage is an explicit ranked outcome.
        row_eos=absolute.new_zeros(encoded.precursor_sample_index.numel())
        for row in range(row_eos.numel()):
            lo,hi=data.precursor_row_action_ptr[row:row+2]
            seed=data.precursor_row_action_index[lo:hi]
            same=(data.teacher_state_sample_index==encoded.precursor_sample_index[row]) & (counts_all==seed.numel())
            contains=absolute.new_zeros(state_target.numel(),dtype=torch.long)
            contains.scatter_add_(0,owner_all,torch.isin(data.teacher_state_action_index,seed).long())
            match=same & (contains==seed.numel())
            if torch.any(match):row_eos[row]=state_observed[match].max()
        rank_logits=torch.cat((row_logits,encoded.precursor_eos_logits[:,None]),dim=1)
        rank_target=torch.cat((row_target,row_eos[:,None]),dim=1)
        rank_positive=rank_target>0
        absolute_intensity_loss=multi_positive_loss(rank_logits,rank_positive,rank_target)
        row_positive=positive[encoded.precursor_sample_index] & torch.isfinite(row_logits)
        row_positive=torch.cat((row_positive,(row_eos>0)[:,None]),dim=1)
        absolute_loss=absolute_filter_loss(rank_logits,row_positive,torch.isfinite(rank_logits),self.feature_model.threshold,self.negative_weight)

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
        # Teacher prefixes used only to construct a non-root precursor are not
        # MS2 decoding states. Decoding/training both start at the precursor.
        ready=torch.zeros(p,dtype=torch.bool,device=absolute.device)
        for row in range(encoded.precursor_sample_index.numel()):
            lo,hi=data.precursor_row_action_ptr[row:row+2]
            seed=data.precursor_row_action_index[lo:hi]
            present=torch.zeros(p,dtype=torch.long,device=absolute.device)
            present.scatter_add_(0,owner,torch.isin(data.teacher_state_action_index,seed).long())
            ready |= (sample==encoded.precursor_sample_index[row]) & (present==seed.numel())
        next_positive &= ready[:,None]
        if torch.any(next_positive & ~torch.isfinite(logits)):
            raise ValueError("Teacher positive is not a valid candidate under the shared compatibility rules")
        next_weights=absolute.new_zeros(logits.shape)
        edge_parent=data.transition_parent_state_index
        edge_child=data.transition_child_state_index
        edge_action=data.transition_added_action_index
        if edge_parent.numel():
            edge_local=inverse[sample[edge_parent],edge_action]
            okay=edge_local>=0
            flat=edge_parent[okay]*logits.shape[1]+edge_local[okay]
            next_weights.view(-1).scatter_reduce_(0,flat,state_target[edge_child[okay]],reduce='amax',include_self=True)
        next_weights[:,-1]=state_target*data.teacher_positive_eos
        # Preserve supervision for low-intensity assigned fragments.
        next_weights=torch.where(next_positive,next_weights.clamp_min(0.05),next_weights)
        next_loss=multi_positive_loss(logits,next_positive,next_weights)
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
                   "valid_candidate_count": torch.isfinite(logits[:, :-1]).sum(dim=1).float().mean() if p else absolute.new_zeros(()),
                   "normalized_replacement_rate": ((expansion.child_action_count <= counts[expansion.parent_state_index]) & expansion.valid).float().mean() if expansion.valid.numel() else absolute.new_zeros(())}
        if p and torch.any(ready):
            supervised=ready & next_positive.any(dim=1)
            prediction = logits.argmax(dim=1)
            metrics["positive_set_top1_accuracy"] = next_positive.gather(1, prediction[:, None]).squeeze(1)[supervised].float().mean()
            top = logits.topk(min(5, logits.shape[1]), dim=1).indices
            metrics["positive_set_top5_recall"] = next_positive.gather(1, top).any(dim=1)[supervised].float().mean()
            eos_prediction, eos_target = (prediction == logits.shape[1] - 1) & supervised, data.teacher_positive_eos & supervised
            correct = (eos_prediction & eos_target).sum()
            metrics["eos_precision"] = correct / eos_prediction.sum().clamp_min(1)
            metrics["eos_recall"] = correct / eos_target.sum().clamp_min(1)
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
            metrics['spectrum_cosine_similarity']=cosine[valid_spectrum].mean() if torch.any(valid_spectrum) else absolute.new_zeros(())
            metrics['intensity_mae']=(predicted-observed).abs().mean() if predicted.numel() else absolute.new_zeros(())
        return ActionTrainingOutput(loss, absolute_loss, next_loss, absolute, logits, next_positive, pool, metrics, downstream_output,absolute_intensity_loss)
