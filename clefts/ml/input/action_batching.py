"""Slice prepared tensor records before transfer; no chemistry is performed."""
from dataclasses import replace
import torch
from torch_geometric.data import Batch
from .source_action_structure import csr, coo


def select_samples(data, indices):
    indices=torch.as_tensor(indices,dtype=torch.long,device=data.device)
    if data.device.type!='cpu':
        raise ValueError('Slice structures before transfer to CUDA')
    remap=torch.full((data.num_samples,),-1,dtype=torch.long)
    remap[indices]=torch.arange(indices.numel())
    state_ids=(remap[data.teacher_node_sample_index]>=0).nonzero().flatten()
    state_map=torch.full((data.teacher_node_sample_index.numel(),),-1,dtype=torch.long)
    state_map[state_ids]=torch.arange(state_ids.numel())
    def rows(ptr,values,ids):
        return csr([values[ptr[i]:ptr[i+1]].tolist() for i in ids.tolist()])
    state_ptr,state_index=rows(data.teacher_node_action_ptr,data.teacher_node_action_index,state_ids)
    next_ptr,next_index=rows(data.teacher_positive_action_ptr,data.teacher_positive_action_index,state_ids)
    positive_ptr,positive_index=rows(data.sample_positive_action_ptr,data.sample_positive_action_index,indices)
    seed_ids=torch.cat([torch.arange(data.sample_precursor_row_ptr[i],data.sample_precursor_row_ptr[i+1]) for i in indices.tolist()])
    seed_ptr,seed_index=rows(data.precursor_row_action_ptr,data.precursor_row_action_index,seed_ids)
    seed_counts=data.sample_precursor_row_ptr[indices+1]-data.sample_precursor_row_ptr[indices]
    seed_map=torch.full((data.precursor_row_action_ptr.numel()-1,),-1,dtype=torch.long)
    seed_map[seed_ids]=torch.arange(seed_ids.numel())
    pair=data.precursor_next_index
    okay=seed_map[pair[0]]>=0
    precursor_next=torch.stack((seed_map[pair[0,okay]],pair[1,okay]))
    transitions=state_map[data.transition_parent_state_index]>=0
    state_nodes=data.state_fragment_node_index[state_ids] if data.state_fragment_node_index.numel() else data.state_fragment_node_index
    downstream=data.downstream
    annotations=tuple(data.sample_annotations[i] for i in indices.tolist()) if data.sample_annotations else ()
    if downstream is not None:
        decoded=downstream.decoded
        node_ids=(remap[decoded.node_sample_index]>=0).nonzero().flatten()
        node_map=torch.full((len(decoded.compounds),),-1,dtype=torch.long)
        node_map[node_ids]=torch.arange(node_ids.numel())
        edges=(node_map[decoded.edge_index[0]]>=0)&(node_map[decoded.edge_index[1]]>=0)
        edge_map=torch.full((edges.numel(),),-1,dtype=torch.long);edge_map[edges]=torch.arange(int(edges.sum()))
        seed_pair=decoded.edge_seed_action_index
        keep_seed=edge_map[seed_pair[0]]>=0
        seed_pair=torch.stack((edge_map[seed_pair[0,keep_seed]],seed_pair[1,keep_seed]))
        terminals=decoded.terminal_node_index[node_map[decoded.terminal_node_index]>=0]
        graphs=decoded.node_graph.to_data_list()
        tree_graphs=downstream.tree_graph.to_data_list()
        subset=replace(decoded,node_graph=Batch.from_data_list([graphs[i] for i in node_ids.tolist()]),edge_index=node_map[decoded.edge_index[:,edges]],
            node_sample_index=remap[decoded.node_sample_index[node_ids]],edge_added_action_index=decoded.edge_added_action_index[edges],terminal_node_index=node_map[terminals],
            node_log_score=decoded.node_log_score[node_ids],trees=tuple(decoded.trees[i] for i in indices.tolist()),compounds=tuple(decoded.compounds[i] for i in node_ids.tolist()),
            edge_seed_action_index=seed_pair,materialized_state_index=torch.empty(0,dtype=torch.long),materialized_node_index=torch.empty(0,dtype=torch.long))
        formula_ids=(remap[downstream.formula_sample_index]>=0).nonzero().flatten()
        formula_map=torch.full((downstream.formula_sample_index.numel(),),-1,dtype=torch.long)
        formula_map[formula_ids]=torch.arange(formula_ids.numel())
        ions=formula_map[downstream.ion_formula_index]>=0
        downstream=replace(downstream,decoded=subset,tree_graph=Batch.from_data_list([tree_graphs[i] for i in indices.tolist()]),
            ion_node_index=node_map[downstream.ion_node_index[ions]],ion_features=downstream.ion_features[ions],formula_sample_index=remap[downstream.formula_sample_index[formula_ids]],
            formula_tensor=downstream.formula_tensor[formula_ids],formula_mz=downstream.formula_mz[formula_ids],ion_formula_index=formula_map[downstream.ion_formula_index[ions]],
            target_intensity=downstream.target_intensity[formula_ids] if downstream.target_intensity is not None else None,
            ion_is_positive=downstream.ion_is_positive[ions] if downstream.ion_is_positive is not None else None,
            ion_adduct=tuple(adduct for adduct,keep in zip(downstream.ion_adduct,ions.tolist()) if keep) if downstream.ion_adduct else (),
            unique_node_graph=None,node_graph_inverse=None,unique_source_index=None,node_smiles=tuple(downstream.node_smiles[i] for i in node_ids.tolist()))
        if state_nodes.numel():state_nodes=torch.where(state_nodes>=0,node_map[state_nodes.clamp_min(0)],state_nodes)
        annotations=tuple({**sample,'peaks':[{**peak,'matches':[
            {**match,'nodeIndices':[int(node_map[index]) for index in match['nodeIndices'] if node_map[index]>=0]}
            for match in peak['matches']]} for peak in sample['peaks']]} for sample in annotations)
    return replace(data,sample_tree_index=data.sample_tree_index[indices],condition_features=data.condition_features[indices],
        teacher_node_sample_index=remap[data.teacher_node_sample_index[state_ids]],teacher_node_action_ptr=state_ptr,teacher_node_action_index=state_index,
        teacher_node_precursor_row_index=seed_map[data.teacher_node_precursor_row_index[state_ids]],
        teacher_node_parent_index=torch.where(data.teacher_node_parent_index[state_ids]>=0,state_map[data.teacher_node_parent_index[state_ids].clamp_min(0)],data.teacher_node_parent_index[state_ids]),
        teacher_node_added_action_index=data.teacher_node_added_action_index[state_ids],teacher_node_ms2_depth=data.teacher_node_ms2_depth[state_ids],
        teacher_positive_action_weight=torch.cat([data.teacher_positive_action_weight[data.teacher_positive_action_ptr[i]:data.teacher_positive_action_ptr[i+1]] for i in state_ids.tolist()]) if state_ids.numel() else torch.empty(0),
        teacher_positive_action_ptr=next_ptr,teacher_positive_action_index=next_index,teacher_node_observed=data.teacher_node_observed[state_ids],
        sample_positive_action_ptr=positive_ptr,sample_positive_action_index=positive_index,
        sample_precursor_row_ptr=torch.cat((torch.zeros(1,dtype=torch.long),seed_counts.cumsum(0))),precursor_row_action_ptr=seed_ptr,precursor_row_action_index=seed_index,precursor_next_index=precursor_next,
        transition_parent_state_index=state_map[data.transition_parent_state_index[transitions]],transition_child_state_index=state_map[data.transition_child_state_index[transitions]],transition_added_action_index=data.transition_added_action_index[transitions],
        state_fragment_node_index=state_nodes,downstream=downstream,sample_annotations=annotations)
