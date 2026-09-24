"""Prepared physical-ion candidates and CE-conditioned intensity model."""
from __future__ import annotations
from dataclasses import dataclass,replace
from collections.abc import Sequence
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from torch_geometric.data import Batch,Data
from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.domain.fragment.ion_tree import FragmentIonTree
from clefts.libs.mmkit.mmkit import Adduct
from ..common.layers.graphormer import GraphormerEncoder
from ..mol.formula_encoder import FormulaTensorizer
from .materialization import DecodedFragmentTreeBatch


@dataclass(frozen=True)
class PostMaterializationBatch:
    decoded: DecodedFragmentTreeBatch
    tree_graph: Batch
    physical_candidate_node_index: Tensor
    physical_candidate_sample_index: Tensor
    physical_candidate_formula_tensor: Tensor
    physical_candidate_mz: Tensor
    physical_candidate_charge: Tensor
    physical_candidate_explanation_ptr: Tensor
    explanation_ion_type_index: Tensor
    explanation_unsaturation: Tensor
    explanation_radical: Tensor
    target_intensity: Tensor|None=None
    physical_candidate_is_positive: Tensor|None=None
    physical_candidate_adduct: tuple[str,...]=()
    ion_type_strs: tuple[str,...]=()
    unique_node_graph: Batch|None=None
    node_graph_inverse: Tensor|None=None
    unique_source_index: Tensor|None=None
    unique_smiles: tuple[str,...]=()
    node_smiles: tuple[str,...]=()

    @property
    def formula_sample_index(self):return self.physical_candidate_sample_index
    @property
    def formula_tensor(self):return self.physical_candidate_formula_tensor
    @property
    def formula_mz(self):return self.physical_candidate_mz
    @property
    def ion_node_index(self):return self.physical_candidate_node_index
    @property
    def ion_formula_index(self):return torch.arange(self.physical_candidate_node_index.numel(),device=self.physical_candidate_node_index.device)
    @property
    def ion_adduct(self):return self.physical_candidate_adduct

    def to(self,device:str|torch.device):
        names=('physical_candidate_node_index','physical_candidate_sample_index','physical_candidate_formula_tensor',
               'physical_candidate_mz','physical_candidate_charge','physical_candidate_explanation_ptr',
               'explanation_ion_type_index','explanation_unsaturation','explanation_radical','target_intensity',
               'physical_candidate_is_positive','node_graph_inverse','unique_source_index')
        return replace(self,decoded=self.decoded.to(device),tree_graph=self.tree_graph.to(device),
            unique_node_graph=self.unique_node_graph.to(device) if self.unique_node_graph is not None else None,
            **{name:getattr(self,name).to(device) for name in names if getattr(self,name) is not None})


@dataclass(frozen=True)
class PostMaterializationOutput:
    intensity:Tensor
    sample_index:Tensor
    formula_tensor:Tensor
    mz:Tensor
    loss:Tensor
    confidence:Tensor
    ion_probability:Tensor
    metrics:dict[str,Tensor]


def normalized_attention_pool(values:Tensor,group:Tensor,count:int,score_layer:nn.Module)->Tensor:
    """Attention pooling whose weights sum to one inside every candidate."""
    if count==0:return values.new_zeros((0,values.shape[-1]))
    score=score_layer(values).squeeze(-1)
    maximum=score.new_full((count,),-torch.inf).scatter_reduce_(0,group,score,reduce='amax',include_self=True)
    weight=(score-maximum[group]).exp()
    denominator=weight.new_zeros(count).scatter_add_(0,group,weight).clamp_min(1e-12)
    return values.new_zeros((count,values.shape[-1])).index_add_(0,group,values*(weight/denominator[group])[:,None])


class PostMaterializationFragmentTreeModel(nn.Module):
    requires_action_features=True
    def __init__(self,mol_encoder:nn.Module,action_dim:int,
                 hidden_dim:int=128,num_heads:int=4,num_layers:int=2,max_action_count:int=3,
                 cosine_loss_weight:float=.5,ion_loss_weight:float=.5,ion_prediction_threshold:float=.5,
                 intensity_power:float=.5,precursor_free_loss_weight:float=.5,dropout:float=0.,
                 ion_type_count:int=1,max_unsaturation:int=0,ion_embedding_dim:int=32,
                 unsaturation_embedding_dim:int=16,radical_embedding_dim:int=8,state_hidden_dim:int|None=None,
                 main_adduct_dim:int|None=None,collision_energy_dim:int=16,
                 equivalent_state_aggregation:str='attention',peak_intensity_threshold:float=0.)->None:
        super().__init__()
        if equivalent_state_aggregation!='attention':raise ValueError("equivalent_state_aggregation must be 'attention'")
        if intensity_power<=0:raise ValueError('intensity_power must be positive')
        if not 0<=peak_intensity_threshold<=1:raise ValueError('peak_intensity_threshold must be in [0,1]')
        self.intensity_power=intensity_power;self.precursor_free_loss_weight=precursor_free_loss_weight
        self.cosine_loss_weight=cosine_loss_weight;self.ion_loss_weight=ion_loss_weight
        self.ion_prediction_threshold=ion_prediction_threshold
        self.peak_intensity_threshold=peak_intensity_threshold
        self.mol_encoder=mol_encoder
        state_hidden_dim=state_hidden_dim or hidden_dim;main_adduct_dim=main_adduct_dim or 128
        self.edge_encoder=nn.Sequential(nn.Linear(action_dim+mol_encoder.graph_dim*2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
        self.tree_encoder=GraphormerEncoder(node_dim=mol_encoder.graph_dim,hidden_dim=hidden_dim,edge_dim=hidden_dim,
            num_heads=num_heads,num_layers=num_layers,max_spatial_dist=max_action_count+1,max_edge_dist=max_action_count+1,
            undirected_for_spd=False,undirected_for_path=False,dropout=dropout)
        self.ion_embedding=nn.Embedding(max(1,ion_type_count),ion_embedding_dim)
        self.unsaturation_embedding=nn.Embedding(max_unsaturation+1,unsaturation_embedding_dim)
        self.radical_embedding=nn.Embedding(2,radical_embedding_dim)
        self.hydrogen_shift_encoder=nn.Sequential(nn.Linear(ion_embedding_dim+unsaturation_embedding_dim+radical_embedding_dim,state_hidden_dim),nn.GELU(),nn.Linear(state_hidden_dim,hidden_dim))
        self.explanation_attention=nn.Linear(hidden_dim,1)
        self.main_adduct_projection=nn.Linear(main_adduct_dim,hidden_dim)
        self.collision_energy_projection=nn.Linear(collision_energy_dim,hidden_dim)
        self.intensity_head=nn.Sequential(nn.Linear(hidden_dim*4,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
        self.ion_score=nn.Linear(hidden_dim*2,1)

    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.mol_encoder.parameters()):self.mol_encoder.eval()
        return self

    def forward(self,data:PostMaterializationBatch,*,action_h:Tensor,main_adduct_h:Tensor,
                collision_energy_h:Tensor,source_embeddings:Tensor|None=None,molecular_embeddings:Tensor|None=None):
        if molecular_embeddings is None:
            if data.node_graph_inverse is not None and data.unique_source_index is not None:
                if source_embeddings is None or data.unique_node_graph is None:
                    raise ValueError('Deduplicated prepared graphs require source embeddings and unique graph tensors')
                unique=source_embeddings.new_zeros((data.unique_source_index.numel(),source_embeddings.shape[1]))
                source_mask=data.unique_source_index>=0
                unique[source_mask]=source_embeddings[data.unique_source_index[source_mask]]
                graph_mask=~source_mask
                if torch.any(graph_mask):
                    graph_h=self.mol_encoder(data.unique_node_graph).embeddings
                    unique[graph_mask]=graph_h[:int(graph_mask.sum())]
                molecular=unique[data.node_graph_inverse]
            else:
                molecular=self.mol_encoder(data.decoded.node_graph).embeddings
        else:molecular=molecular_embeddings
        edges=data.decoded.edge_index;added=data.decoded.edge_added_action_index
        action=action_h.new_zeros((added.numel(),action_h.shape[1]));normal=added>=0
        action[normal]=action_h[added[normal]]
        edge_h=self.edge_encoder(torch.cat((action,molecular[edges[0]],molecular[edges[1]]),1)) if edges.numel() else molecular.new_zeros((0,self.tree_encoder.hidden_dim))
        tree=data.tree_graph.clone();tree.x=molecular;tree.edge_attr=edge_h
        node_h,_=self.tree_encoder(tree)
        explanation=torch.cat((self.ion_embedding(data.explanation_ion_type_index),
            self.unsaturation_embedding(data.explanation_unsaturation),self.radical_embedding(data.explanation_radical)),1)
        explanation=self.hydrogen_shift_encoder(explanation)
        counts=data.physical_candidate_explanation_ptr[1:]-data.physical_candidate_explanation_ptr[:-1]
        owner=torch.repeat_interleave(torch.arange(counts.numel(),device=node_h.device),counts)
        physical=normalized_attention_pool(explanation,owner,counts.numel(),self.explanation_attention)
        sample=data.physical_candidate_sample_index;fragment=node_h[data.physical_candidate_node_index]
        adduct=self.main_adduct_projection(main_adduct_h[sample]);energy=self.collision_energy_projection(collision_energy_h[sample])
        intensity=F.softplus(self.intensity_head(torch.cat((fragment,physical,adduct,energy),1)).squeeze(-1))
        ion_logit=self.ion_score(torch.cat((fragment,physical),1)).squeeze(-1)
        sample_count=main_adduct_h.shape[0]
        maximum=intensity.new_zeros(sample_count).scatter_reduce_(0,sample,intensity,reduce='amax',include_self=True)
        normalized=intensity/maximum[sample].clamp_min(1e-8)
        node_group=data.decoded.node_sample_index
        roots=node_group.new_full((data.tree_graph.num_graphs,),node_group.numel())
        roots.scatter_reduce_(0,node_group,torch.arange(node_group.numel(),device=node_group.device),reduce='amin',include_self=True)
        candidate_group=node_group[data.physical_candidate_node_index]
        precursor=data.physical_candidate_node_index==roots[candidate_group]

        def spectrum_loss(keep):
            total=intensity.sum()*0
            if data.target_intensity is not None and torch.any(keep):
                masked=intensity*keep;peak_max=masked.new_zeros(sample_count).scatter_reduce_(0,sample,masked,reduce='amax',include_self=True)
                pred=(masked/peak_max[sample].clamp_min(1e-8))[keep].pow(self.intensity_power)
                target_raw=data.target_intensity.clamp_min(0)*keep
                target_max=target_raw.new_zeros(sample_count).scatter_reduce_(0,sample,target_raw,reduce='amax',include_self=True)
                target=(target_raw/target_max[sample].clamp_min(1e-8))[keep].pow(self.intensity_power)
                total=F.mse_loss(pred,target);group=sample[keep]
                dot=pred.new_zeros(sample_count).scatter_add_(0,group,pred*target)
                pn=pred.new_zeros(sample_count).scatter_add_(0,group,pred.square())
                tn=pred.new_zeros(sample_count).scatter_add_(0,group,target.square());observed=tn>0
                if torch.any(observed):total=total+self.cosine_loss_weight*(1-dot[observed]/(pn[observed]*tn[observed]).sqrt().clamp_min(1e-8)).mean()
            return total
        all_candidates=torch.ones_like(precursor);loss=spectrum_loss(all_candidates)
        if self.precursor_free_loss_weight>0:loss=loss+self.precursor_free_loss_weight*spectrum_loss(~precursor)
        if data.physical_candidate_is_positive is not None:
            pos=data.physical_candidate_is_positive;neg=~pos
            auxiliary=(F.softplus(-ion_logit[pos]).mean() if torch.any(pos) else ion_logit.sum()*0)+(F.softplus(ion_logit[neg]).mean() if torch.any(neg) else ion_logit.sum()*0)
            loss=loss+self.ion_loss_weight*auxiliary
        def cosine_metric(keep):
            if data.target_intensity is None or not torch.any(keep):return loss*0
            pred=normalized[keep];target_raw=data.target_intensity.clamp_min(0)*keep
            target_max=target_raw.new_zeros(sample_count).scatter_reduce_(0,sample,target_raw,reduce='amax',include_self=True)
            target=(target_raw/target_max[sample].clamp_min(1e-8))[keep];group=sample[keep]
            dot=pred.new_zeros(sample_count).scatter_add_(0,group,pred*target)
            pn=pred.new_zeros(sample_count).scatter_add_(0,group,pred.square());tn=pred.new_zeros(sample_count).scatter_add_(0,group,target.square())
            observed=tn>0;value=dot/(pn*tn).sqrt().clamp_min(1e-8)
            return value[observed].mean() if torch.any(observed) else loss*0
        target_max=data.target_intensity.new_zeros(sample_count).scatter_reduce_(0,sample,data.target_intensity.clamp_min(0),reduce='amax',include_self=True) if data.target_intensity is not None else None
        normalized_target=data.target_intensity.clamp_min(0)/target_max[sample].clamp_min(1e-8) if data.target_intensity is not None else None
        metrics={'intensity_loss':loss,'intensity_mae':(normalized-normalized_target).abs().mean() if normalized_target is not None and normalized.numel() else loss*0,
            'full_spectrum_cosine':cosine_metric(all_candidates),'precursor_free_cosine':cosine_metric(~precursor),
            'num_physical_ion_candidates':intensity.new_tensor(float(intensity.numel())),
            'num_ion_explanations':intensity.new_tensor(float(explanation.shape[0])),
            'mean_explanations_per_physical_candidate':counts.float().mean() if counts.numel() else loss*0,
            'max_explanations_per_physical_candidate':counts.max().to(intensity.dtype) if counts.numel() else loss*0}
        return PostMaterializationOutput(normalized,sample,data.physical_candidate_formula_tensor,data.physical_candidate_mz,
            loss,ion_logit.sigmoid(),ion_logit.sigmoid(),metrics)


def prepare_post_materialization(decoded:DecodedFragmentTreeBatch,fragmenter:Fragmenter,
                                 precursor_types:Sequence[Adduct],tensorizer:FormulaTensorizer,
                                 ion_positive_sets:Sequence[set[tuple[int,str]]]|None=None,
                                 sample_branch_group_index:Tensor|None=None,
                                 ion_type_strs:Sequence[str]|None=None)->PostMaterializationBatch:
    """Run all formula/ion chemistry and physical-candidate grouping."""
    sample_count=len(precursor_types)
    sample_branch_group_index=torch.as_tensor(sample_branch_group_index if sample_branch_group_index is not None else torch.arange(sample_count),dtype=torch.long)
    ion_vocab=list(dict.fromkeys(ion_type_strs or [str(rule.ion_shift) for adduct_rule in fragmenter.adduct_rule_set.adduct_rules for rule in adduct_rule.ion_shifts]))
    ion_to_index={value:i for i,value in enumerate(ion_vocab)}
    graphs=[];nodes=[];samples=[];formulas=[];mzs=[];charges=[];explanation_rows=[];adducts=[];positives=[]
    node_offset=0
    for branch_group,tree in enumerate(decoded.trees):
        compounds={i:decoded.compounds[node_offset+i] for i in range(tree.num_nodes)};builder=fragmenter.fragment_ion_tree_builder
        ion_tree=FragmentIonTree.from_fragment_tree(tree,fragment_ion_adduct_rule_set=fragmenter.adduct_rule_set,
            hydrogen_state_candidate_store=builder._build_hydrogen_state_candidate_store(),
            ion_shift_candidate_store=builder._build_ion_shift_candidate_store(fragment_tree=tree,fragment_compound_by_index=compounds))
        object.__setattr__(ion_tree,'_fragment_compound_by_index',compounds)
        group_samples=(sample_branch_group_index==branch_group).nonzero().flatten().tolist()
        if not group_samples:node_offset+=tree.num_nodes;continue
        main=fragmenter._resolve_main_adduct_type(precursor_types[group_samples[0]])
        formula_group=ion_tree.get_formula_candidate_group(main);rule_index=ion_tree.get_adduct_rule_index_by_adduct_type(main)
        hydrogen_states=ion_tree.hydrogen_state_candidate_store.get_candidate_states(rule_index)
        local={}
        for formula_index,formula in enumerate(formula_group.formulas):
            for entry in range(formula_group.indptr[formula_index],formula_group.indptr[formula_index+1]):
                node=int(formula_group.node_indices[entry]);net=formula_group.candidate_adducts[formula_group.candidate_adduct_indices[entry]]
                shift=int(formula_group.shift_rule_indices[entry]);hydrogen=int(formula_group.hydrogen_candidate_indices[entry])
                ion=str(ion_tree.ion_shift_candidate_store.get_ion_shift_adduct_type(shift));u,r=map(int,hydrogen_states[hydrogen])
                key=(node,str(formula.normalized),str(net),int(net.charge))
                local.setdefault(key,dict(node=node,formula=formula,net=net,explanations=[]))['explanations'].append((ion_to_index[ion],u,r))
        edge_store=tree.edge_store
        graphs.append(Data(x=torch.zeros((tree.num_nodes,1)),edge_index=torch.stack((torch.as_tensor(edge_store.source_indices),torch.as_tensor(edge_store.target_indices)))))
        for sample in group_samples:
            for item in local.values():
                nodes.append(node_offset+item['node']);samples.append(sample);formulas.append(item['formula']);mzs.append(item['formula'].exact_mass)
                charges.append(item['net'].charge);adducts.append(str(item['net']));explanation_rows.append(tuple(dict.fromkeys(item['explanations'])))
                if ion_positive_sets is not None:positives.append((node_offset+item['node'],str(item['net'])) in ion_positive_sets[sample])
        node_offset+=tree.num_nodes
    explanation_ptr,flat=__import__('clefts.ml.input.source_action_structure',fromlist=['csr']).csr(explanation_rows)
    return PostMaterializationBatch(decoded,Batch.from_data_list(graphs),torch.tensor(nodes,dtype=torch.long),torch.tensor(samples,dtype=torch.long),
        tensorizer.formulas_to_tensor(formulas),torch.tensor(mzs,dtype=torch.float64),torch.tensor(charges,dtype=torch.long),explanation_ptr,
        torch.tensor([x[0] for x in flat.reshape(-1,3).tolist()],dtype=torch.long) if flat.numel() else torch.empty(0,dtype=torch.long),
        torch.tensor([x[1] for x in flat.reshape(-1,3).tolist()],dtype=torch.long) if flat.numel() else torch.empty(0,dtype=torch.long),
        torch.tensor([x[2] for x in flat.reshape(-1,3).tolist()],dtype=torch.long) if flat.numel() else torch.empty(0,dtype=torch.long),
        physical_candidate_is_positive=torch.tensor(positives,dtype=torch.bool) if ion_positive_sets is not None else None,
        physical_candidate_adduct=tuple(adducts),ion_type_strs=tuple(ion_vocab),node_smiles=tuple(c.smiles for c in decoded.compounds))


def collate_post_materialization(items:Sequence[PostMaterializationBatch],action_offsets:Sequence[int])->PostMaterializationBatch:
    molecules=[];trees=[];compounds=[];metadata=[];edges=[];node_groups=[];edge_actions=[];terminals=[];scores=[]
    nodes=[];samples=[];formulas=[];mzs=[];charges=[];explanations=[];targets=[];positives=[];adducts=[];node_smiles=[]
    node_offset=sample_offset=0
    ion_vocab=list(dict.fromkeys(value for item in items for value in item.ion_type_strs));ion_index={v:i for i,v in enumerate(ion_vocab)}
    for item,action_offset in zip(items,action_offsets):
        d=item.decoded;molecules.extend(d.node_graph.to_data_list());trees.extend(item.tree_graph.to_data_list());compounds.extend(d.compounds);metadata.extend(d.trees)
        edges.append(d.edge_index+node_offset);node_groups.append(d.node_sample_index+len(metadata)-len(d.trees));edge_actions.append(torch.where(d.edge_added_action_index>=0,d.edge_added_action_index+action_offset,d.edge_added_action_index))
        terminals.append(d.terminal_node_index+node_offset);scores.append(d.node_log_score)
        nodes.append(item.physical_candidate_node_index+node_offset);samples.append(item.physical_candidate_sample_index+sample_offset);formulas.append(item.physical_candidate_formula_tensor);mzs.append(item.physical_candidate_mz);charges.append(item.physical_candidate_charge)
        for candidate in range(item.physical_candidate_node_index.numel()):
            start,stop=item.physical_candidate_explanation_ptr[candidate:candidate+2]
            explanations.append([(ion_index[item.ion_type_strs[int(i)]],int(u),int(r)) for i,u,r in zip(item.explanation_ion_type_index[start:stop],item.explanation_unsaturation[start:stop],item.explanation_radical[start:stop])])
        if item.target_intensity is not None:targets.append(item.target_intensity)
        if item.physical_candidate_is_positive is not None:positives.append(item.physical_candidate_is_positive)
        adducts.extend(item.physical_candidate_adduct);node_smiles.extend(item.node_smiles);node_offset+=len(d.compounds);sample_offset+=int(item.physical_candidate_sample_index.max())+1 if item.physical_candidate_sample_index.numel() else 0
    from ..input.source_action_structure import csr
    ptr,flat=csr(explanations);triples=flat.reshape(-1,3) if flat.numel() else torch.empty((0,3),dtype=torch.long)
    decoded=DecodedFragmentTreeBatch(Batch.from_data_list(molecules),torch.cat(edges,1),torch.cat(node_groups),torch.cat(edge_actions),torch.cat(terminals),torch.cat(scores),tuple(metadata),tuple(compounds),sum(i.decoded.rdkit_run_count for i in items),sum(i.decoded.failed_effect_count for i in items))
    return PostMaterializationBatch(decoded,Batch.from_data_list(trees),torch.cat(nodes),torch.cat(samples),torch.cat(formulas),torch.cat(mzs),torch.cat(charges),ptr,triples[:,0],triples[:,1],triples[:,2],torch.cat(targets) if targets else None,torch.cat(positives) if positives else None,tuple(adducts),tuple(ion_vocab),node_smiles=tuple(node_smiles))


def deduplicate_molecular_graphs(data:PostMaterializationBatch,source_smiles=()):
    graphs=data.decoded.node_graph.to_data_list();source_index={s:i for i,s in enumerate(source_smiles)};keys={};unique=[];inverse=[];shared=[]
    if len(data.node_smiles)!=len(graphs):raise ValueError('Molecular identities must be prepared before training')
    for graph,key in zip(graphs,data.node_smiles):
        if key not in keys:
            keys[key]=len(shared);shared.append(source_index.get(key,-1))
            if key not in source_index:unique.append(graph)
        inverse.append(keys[key])
    unique_batch=Batch.from_data_list(unique if unique else [graphs[0]])
    return replace(data,unique_node_graph=unique_batch,node_graph_inverse=torch.tensor(inverse),unique_source_index=torch.tensor(shared),unique_smiles=tuple(keys))
