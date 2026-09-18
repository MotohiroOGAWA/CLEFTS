"""Selection features have no dependency on materialized target graphs."""
from __future__ import annotations
import torch
from dataclasses import dataclass, replace
from torch import Tensor, nn
from .components.condition.condition_encoder import MS2ConditionEncoder
from .components.action.action_encoder import ActionEncoder
from .components.action.action_condition_scorer import ActionConditionScorer
from .components.action.action_decoder import ActionPool, ActionSequenceDecoder, build_action_pool


@dataclass(frozen=True)
class SourceActionFeatures:
    action_h: Tensor
    condition_h: Tensor
    absolute_logits: Tensor
    pool: ActionPool
    precursor_absolute_logits: Tensor | None = None
    precursor_eos_logits: Tensor | None = None
    precursor_sample_index: Tensor | None = None
    source_embeddings: Tensor | None = None


class SourceActionFeatureModel(nn.Module):
    def __init__(self, mol_encoder: nn.Module, category_sizes: tuple[int, int, int],
                 condition_feature_dim: int, hidden_dim: int = 128, condition_dim: int = 128,
                 max_action_count: int = 3, action_prefilter_top_k: int = 64,
                 action_prefilter_max_k: int = 128, action_prefilter_threshold_logit: float = 1.0,
                 beam_size: int = 32, num_heads: int = 4, max_decode_steps: int = 16,
                 state_num_layers: int = 2, state_dropout: float = 0.0,
                 condition_encoder: nn.Module | None = None, max_roles: int = 64) -> None:
        super().__init__()
        if not 1 <= action_prefilter_top_k <= action_prefilter_max_k:
            raise ValueError("Require 1 <= action_prefilter_top_k <= action_prefilter_max_k")
        self.mol_encoder = mol_encoder
        self.action_encoder = ActionEncoder(mol_encoder.node_dim, mol_encoder.graph_dim, hidden_dim, category_sizes, num_heads,max_roles)
        self.condition_encoder = condition_encoder or nn.Sequential(nn.Linear(condition_feature_dim, condition_dim), nn.GELU(), nn.Linear(condition_dim, condition_dim))
        self.scorer = ActionConditionScorer(hidden_dim, condition_dim, hidden_dim)
        self.decoder = ActionSequenceDecoder(hidden_dim, condition_dim, max_action_count, beam_size, num_heads, max_decode_steps, state_num_layers, state_dropout)
        self.top_k, self.max_k, self.threshold = action_prefilter_top_k, action_prefilter_max_k, action_prefilter_threshold_logit
        self.max_action_count = max_action_count

    def freeze_mol_encoder(self) -> None:
        self._freeze_mol_encoder = True
        self.mol_encoder.eval()
        for parameter in self.mol_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_mol_encoder", False):
            self.mol_encoder.eval()
        return self

    def encode_static(self,data,max_graphs=128):
        """Bound source encoding, then share action tokens across prediction chunks."""
        from torch_geometric.data import Batch
        device=next(self.parameters()).device
        graphs=data.source_graph.to_data_list()
        atoms=[];molecules=[]
        for start in range(0,len(graphs),max_graphs):
            encoded=self.mol_encoder(Batch.from_data_list(graphs[start:start+max_graphs]).to(device))
            atoms.append(encoded.x);molecules.append(encoded.embeddings)
        source_x=torch.cat(atoms);source_h=torch.cat(molecules)
        kwargs={name:getattr(data,name).to(device) for name in ('action_type','action_tree_index','action_source_atom_ptr','action_source_atom_index','action_static_features','action_source_atom_features')}
        actions=self.action_encoder(source_atom_h=source_x,source_mol_h=source_h,**kwargs)
        return actions,source_h

    def forward(self, data: object, training_pool: bool = False, static_features: tuple[Tensor, Tensor] | None = None) -> SourceActionFeatures:
        if data.max_action_role_count > self.action_encoder.roles.num_embeddings:
            raise ValueError("Configure max_roles for the largest SMARTS query in the action universe")
        source = self.mol_encoder(data.source_graph) if static_features is None else None
        action = self.action_encoder(source_atom_h=source.x, source_mol_h=source.embeddings,
            action_type=data.action_type, action_tree_index=data.action_tree_index,
            action_source_atom_ptr=data.action_source_atom_ptr, action_source_atom_index=data.action_source_atom_index,
            action_static_features=data.action_static_features,action_source_atom_features=data.action_source_atom_features) if static_features is None else static_features[0]
        source_embeddings = source.embeddings if source is not None else static_features[1]
        condition = (self.condition_encoder(adduct_idx=data.condition_features[:, 0].long(), collision_energy=data.condition_features[:, 1])
                     if isinstance(self.condition_encoder, MS2ConditionEncoder) else self.condition_encoder(data.condition_features))
        counts = data.precursor_row_action_ptr[1:]-data.precursor_row_action_ptr[:-1]
        row_sample = torch.repeat_interleave(torch.arange(data.num_samples,device=condition.device),data.sample_precursor_row_ptr[1:]-data.sample_precursor_row_ptr[:-1])
        row = torch.repeat_interleave(torch.arange(counts.numel(),device=condition.device),counts)
        position = torch.arange(row.numel(),device=condition.device)-data.precursor_row_action_ptr[row]
        state = torch.full((counts.numel(),self.max_action_count),-1,dtype=torch.long,device=condition.device)
        state[row,position]=data.precursor_row_action_index
        # Static action tokens are shared by all CE/adduct conditions.
        tokens=torch.cat((action,action.new_zeros((1,action.shape[-1]))))[None].expand(data.num_samples,-1,-1)
        precursor_h=self.decoder.state_encoder(tokens,state,row_sample)
        row_absolute=self.scorer(action,condition[row_sample],precursor_h)
        mask=torch.zeros_like(row_absolute,dtype=torch.bool)
        mask[data.precursor_next_index[0],data.precursor_next_index[1]]=True
        row_absolute=row_absolute.masked_fill(~mask,-torch.inf)
        absolute=condition.new_full((data.num_samples,action.shape[0]),-torch.inf)
        absolute.scatter_reduce_(0,row_sample[:,None].expand_as(row_absolute),row_absolute,reduce='amax',include_self=True)
        eos=self.scorer.base(precursor_h).squeeze(-1)+(self.scorer.condition_q(condition[row_sample])*self.scorer.action_q(precursor_h)).sum(-1)*self.scorer.scale
        # Mandatory precursor actions are not filtered by a learned score.
        owners=torch.repeat_interleave(row_sample,counts)
        mandatory=torch.zeros_like(absolute,dtype=torch.bool)
        mandatory[owners,data.precursor_row_action_index]=True
        absolute=absolute.masked_fill(mandatory,self.threshold+1)

        pool = build_action_pool(action, absolute, data, top_k=self.top_k, max_k=self.max_k,
                                 threshold=self.threshold, training=training_pool, max_action_count=self.max_action_count)
        pool=replace(pool,precursor_eos_logits=eos)
        return SourceActionFeatures(action, condition, absolute, pool,row_absolute,eos,row_sample,source_embeddings)
