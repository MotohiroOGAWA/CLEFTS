"""Prepared-tensor branch MIL and physical-ion intensity training."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from ...specgen.components.action.action_decoder import ActionPool


def smooth_max_mil(path_scores: Tensor, peak_path_ptr: Tensor, temperature: float = 0.1) -> Tensor:
    """Normalized smooth maximum for alternative pathways of each peak."""
    if not 0 < temperature or not math.isfinite(temperature):
        raise ValueError("branch_mil_temperature must be positive and finite")
    counts=peak_path_ptr[1:]-peak_path_ptr[:-1]
    if torch.any(counts<=0):raise ValueError("Every supervised peak needs at least one candidate path")
    owner=torch.repeat_interleave(torch.arange(counts.numel(),device=path_scores.device),counts)
    scaled=path_scores/temperature
    maximum=scaled.new_full((counts.numel(),),-torch.inf)
    maximum.scatter_reduce_(0,owner,scaled,reduce='amax',include_self=True)
    total=scaled.new_zeros(counts.numel()).scatter_add_(0,owner,(scaled-maximum[owner]).exp())
    return temperature*(maximum+total.log()-counts.to(path_scores.dtype).log())


def branch_mil_loss(logits: Tensor, data: object, inverse_action: Tensor,
                    temperature: float = 0.1, negative_weight: float = 0.2) -> tuple[Tensor,Tensor,Tensor,Tensor]:
    """Return total, positive MIL, weak-negative loss and path log scores."""
    step_state=data.teacher_path_step_state_index
    step_action=data.teacher_path_step_action_index
    if step_state.numel():
        owner=torch.repeat_interleave(torch.arange(data.teacher_path_step_ptr.numel()-1,device=logits.device),
                                      data.teacher_path_step_ptr[1:]-data.teacher_path_step_ptr[:-1])
        group=data.teacher_node_branch_group_index[step_state]
        local=inverse_action[group,step_action]
        if torch.any(local<0):raise ValueError("Prepared positive action is absent from its branch group")
        step_log_probability=F.logsigmoid(logits[step_state,local])
        path_scores=logits.new_zeros(data.teacher_path_step_ptr.numel()-1).scatter_add_(0,owner,step_log_probability)
        positive=-smooth_max_mil(path_scores,data.teacher_peak_path_ptr,temperature).mean()
    else:
        path_scores=logits.new_empty(0);positive=logits.sum()*0
    negative_counts=data.weak_negative_action_ptr[1:]-data.weak_negative_action_ptr[:-1]
    negative_owner=torch.repeat_interleave(torch.arange(negative_counts.numel(),device=logits.device),negative_counts)
    if negative_owner.numel():
        group=data.teacher_node_branch_group_index[negative_owner]
        local=inverse_action[group,data.weak_negative_action_index]
        valid=local>=0
        negative=F.softplus(logits[negative_owner[valid],local[valid]]).mean() if torch.any(valid) else logits.sum()*0
    else:negative=logits.sum()*0
    return positive+negative_weight*negative,positive,negative,path_scores


@dataclass(frozen=True)
class ActionTrainingOutput:
    loss: Tensor
    branch_loss: Tensor
    branch_logits: Tensor
    branch_positive_mask: Tensor
    pool: ActionPool
    metrics: dict[str,Tensor]
    downstream_output: object|None=None


class ActionFragmentTreeTrainingModel(nn.Module):
    def __init__(self,feature_model:nn.Module,negative_weight:float=.2,
                 downstream_model:nn.Module|None=None,intensity_weight:float=1.,
                 branch_mil_temperature:float=.1,branch_weight:float=1.)->None:
        super().__init__();self.feature_model=feature_model;self.downstream_model=downstream_model
        self.negative_weight=negative_weight;self.intensity_weight=intensity_weight
        self.branch_mil_temperature=branch_mil_temperature;self.branch_weight=branch_weight

    def set_checkpoint_model_config(self,config:dict)->None:self._checkpoint_model_config=dict(config)
    def get_params(self)->dict:return dict(getattr(self,'_checkpoint_model_config',{}))

    def forward(self,data:object)->ActionTrainingOutput:
        encoded=self.feature_model(data,training_pool=True);pool=encoded.pool
        groups=encoded.branch_main_adduct_h.shape[0];actions=encoded.action_h.shape[0]
        inverse=torch.full((groups,actions+1),-1,dtype=torch.long,device=encoded.action_h.device)
        ids=pool.action_index.masked_fill(~pool.valid,actions)
        inverse.scatter_(1,ids,torch.arange(pool.valid.shape[1],device=ids.device).expand_as(ids));inverse[:,actions]=-1
        state_count=data.teacher_node_branch_group_index.numel();owners=data.teacher_node_branch_group_index
        counts=data.teacher_node_action_ptr[1:]-data.teacher_node_action_ptr[:-1]
        row=torch.repeat_interleave(torch.arange(state_count,device=ids.device),counts)
        position=torch.arange(row.numel(),device=ids.device)-data.teacher_node_action_ptr[row]
        state=torch.full((state_count,self.feature_model.max_action_count),-1,dtype=torch.long,device=ids.device)
        local=inverse[owners[row],data.teacher_node_action_index]
        if torch.any(local<0):raise ValueError("Prepared state action is absent from its branch group")
        state[row,position]=local
        sentinel=pool.valid.shape[1]
        state=state.masked_fill(state<0,sentinel).sort(1).values
        state=state.masked_fill(state==sentinel,-1)
        logits,_=self.feature_model.decoder.score_states(pool,encoded.branch_main_adduct_h,state,owners)
        branch_loss,positive_loss,negative_loss,path_scores=branch_mil_loss(
            logits,data,inverse,self.branch_mil_temperature,self.negative_weight)
        positive=torch.zeros_like(logits,dtype=torch.bool)
        next_counts=data.teacher_positive_action_ptr[1:]-data.teacher_positive_action_ptr[:-1]
        next_owner=torch.repeat_interleave(torch.arange(state_count,device=ids.device),next_counts)
        if next_owner.numel():
            next_local=inverse[owners[next_owner],data.teacher_positive_action_index]
            positive[next_owner,next_local]=True
        predicted=(logits.sigmoid()>=.5)&torch.isfinite(logits)
        metrics={'branch_loss':branch_loss,'branch_positive_mil_loss':positive_loss,'branch_negative_loss':negative_loss,
                 'num_branch_groups':logits.new_tensor(float(groups)),'num_teacher_paths':logits.new_tensor(float(path_scores.numel()))}
        max_depth=int(data.teacher_node_ms2_depth.max().item()) if data.teacher_node_ms2_depth.numel() else -1
        for depth in range(max_depth+1):
            mask=data.teacher_node_ms2_depth==depth
            metrics[f'branch_recall_depth_{depth}']=(predicted[mask]&positive[mask]).sum()/positive[mask].sum().clamp_min(1)
            metrics[f'num_training_states_depth_{depth}']=mask.sum().to(logits.dtype)
        downstream_output=None;loss=self.branch_weight*branch_loss
        if self.downstream_model is not None:
            if data.downstream is None:raise ValueError("Intensity training requires prepared physical-ion tensors")
            downstream_output=self.downstream_model(data.downstream,action_h=encoded.action_h,
                main_adduct_h=encoded.sample_main_adduct_h,collision_energy_h=encoded.collision_energy_h,
                source_embeddings=encoded.source_embeddings)
            loss=loss+self.intensity_weight*downstream_output.loss
            metrics.update(downstream_output.metrics)
        return ActionTrainingOutput(loss,branch_loss,logits,positive,pool,metrics,downstream_output)
