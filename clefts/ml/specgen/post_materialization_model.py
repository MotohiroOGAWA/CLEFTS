"""Shared stored-graph/inference-graph downstream boundary."""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.domain.fragment.ion_tree import FragmentIonTree
from clefts.libs.mmkit.mmkit import Adduct, Compound
from ..common.layers.graphormer import GraphormerEncoder
from ..mol.formula_encoder import FormulaTensorizer
from .fragment_tree_formula_intensity_model import FragmentTreeFormulaIntensityPredictor
from .action_materialization import DecodedFragmentTreeBatch


@dataclass(frozen=True)
class PostMaterializationBatch:
    decoded: DecodedFragmentTreeBatch
    tree_graph: Batch
    ion_node_index: Tensor
    ion_features: Tensor
    formula_sample_index: Tensor
    formula_tensor: Tensor
    formula_mz: Tensor
    ion_formula_index: Tensor
    target_intensity: Tensor | None = None

    def to(self, device: str | torch.device) -> PostMaterializationBatch:
        return replace(self, decoded=self.decoded.to(device), tree_graph=self.tree_graph.to(device),
            **{name: getattr(self,name).to(device) for name in ("ion_node_index","ion_features","formula_sample_index","formula_tensor","formula_mz","ion_formula_index","target_intensity") if getattr(self,name) is not None})


@dataclass(frozen=True)
class PostMaterializationOutput:
    intensity: Tensor
    sample_index: Tensor
    formula_tensor: Tensor
    mz: Tensor
    loss: Tensor


class PostMaterializationFragmentTreeModel(nn.Module):
    requires_action_features = True

    def __init__(self, mol_encoder: nn.Module, action_dim: int, condition_dim: int,
                 formula_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, max_action_count: int = 3) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.edge_encoder = nn.Sequential(nn.Linear(action_dim + mol_encoder.graph_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim))
        self.tree_encoder = GraphormerEncoder(node_dim=mol_encoder.graph_dim,hidden_dim=hidden_dim,edge_dim=hidden_dim,
            condition_dim=condition_dim,condition_token_count=1,num_heads=num_heads,num_layers=num_layers,
            max_spatial_dist=max_action_count+1,max_edge_dist=max_action_count+1,undirected_for_spd=False,undirected_for_path=False,dropout=0.)
        self.ion_encoder = nn.Linear(3,hidden_dim)
        self.ion_score = nn.Linear(hidden_dim,1)
        self.formula_intensity = FragmentTreeFormulaIntensityPredictor(formula_dim=formula_dim,hidden_dim=hidden_dim)

    def forward(self, data: PostMaterializationBatch, *, action_h: Tensor, condition_h: Tensor) -> PostMaterializationOutput:
        # Target graphs are consumed only here, after action selection.
        molecular = self.mol_encoder(data.decoded.node_graph).embeddings
        edges = data.decoded.edge_index
        added = data.decoded.edge_added_action_index
        action = action_h[added] if added.numel() else action_h.new_empty((0,action_h.shape[1]))
        edge_h = self.edge_encoder(torch.cat((action,molecular[edges[0]],molecular[edges[1]]),dim=1))
        tree = data.tree_graph.clone()
        tree.x = molecular
        tree.edge_attr = edge_h
        node_h, _ = self.tree_encoder(tree,condition_repr=condition_h)
        ion_h = node_h[data.ion_node_index] + self.ion_encoder(data.ion_features)
        ion_score = F.softplus(self.ion_score(ion_h).squeeze(-1))
        score = ion_score.new_zeros(data.formula_tensor.shape[0]).scatter_add_(0,data.ion_formula_index,ion_score)
        count = ion_score.new_zeros(data.formula_tensor.shape[0]).scatter_add_(0,data.ion_formula_index,torch.ones_like(ion_score))
        intensity = self.formula_intensity(data.formula_tensor,score,count)
        loss = intensity.sum()*0 if data.target_intensity is None or intensity.numel()==0 else F.mse_loss(torch.log1p(intensity),torch.log1p(data.target_intensity.clamp_min(0)))
        return PostMaterializationOutput(intensity,data.formula_sample_index,data.formula_tensor,data.formula_mz,loss)


def prepare_post_materialization(decoded: DecodedFragmentTreeBatch, fragmenter: Fragmenter,
                                 precursor_types: Sequence[Adduct], tensorizer: FormulaTensorizer) -> PostMaterializationBatch:
    """Ion/formula preparation is outside all neural forwards."""
    graphs, node_ids, ion_features, formulas, formula_samples, mzs, formula_ids = [], [], [], [], [], [], []
    node_offset = 0
    terminal = set(decoded.terminal_node_index.tolist())
    for sample, tree in enumerate(decoded.trees):
        compounds = {i:decoded.compounds[node_offset+i] for i in range(tree.num_nodes)}
        builder = fragmenter.fragment_ion_tree_builder
        ion_tree = FragmentIonTree.from_fragment_tree(tree,fragment_ion_adduct_rule_set=fragmenter.adduct_rule_set,
            hydrogen_state_candidate_store=builder._build_hydrogen_state_candidate_store(),
            ion_shift_candidate_store=builder._build_ion_shift_candidate_store(fragment_tree=tree,fragment_compound_by_index=compounds))
        object.__setattr__(ion_tree,"_fragment_compound_by_index",compounds)
        main = fragmenter._resolve_main_adduct_type(precursor_types[sample])
        group = ion_tree.get_formula_candidate_group(main)
        context = fragmenter._build_precursor_assignment_context(ion_tree,precursor_types[sample],dict(compounds))
        from clefts.domain.fragment.pathway.build_pathway import build_pathway_items_for_node
        reachable = {index for index in range(tree.num_nodes) if build_pathway_items_for_node(tree,index,context.precursor_adduct_types,
            max_action_count=fragmenter.tree_max_action_count,precursor_candidate_max_action_count=fragmenter.precursor_candidate_max_action_count)}
        local_edges = tree.edge_store
        graphs.append(Data(x=torch.zeros((tree.num_nodes,1)),edge_index=torch.stack((torch.as_tensor(local_edges.source_indices),torch.as_tensor(local_edges.target_indices)))))
        for formula_index, formula in enumerate(group.formulas):
            selected=[]
            for entry in range(group.indptr[formula_index],group.indptr[formula_index+1]):
                node = int(group.node_indices[entry])
                if node not in reachable or node_offset+node not in terminal:
                    continue
                adduct = group.candidate_adducts[group.candidate_adduct_indices[entry]]
                if node == 0 and adduct.apply_to_formula(compounds[0].formula).normalized != precursor_types[sample].apply_to_formula(compounds[0].formula).normalized:
                    continue
                selected.append((node,float(group.hydrogen_candidate_indices[entry]),float(group.shift_rule_indices[entry]),float(adduct.charge)))
            if not selected:
                continue
            new_index=len(formulas)
            formulas.append(formula)
            formula_samples.append(sample)
            mzs.append(formula.exact_mass)
            for node,hydrogen,shift,charge in selected:
                node_ids.append(node_offset+node)
                ion_features.append((hydrogen,shift,charge))
                formula_ids.append(new_index)
        node_offset += tree.num_nodes
    return PostMaterializationBatch(decoded,Batch.from_data_list(graphs),torch.tensor(node_ids,dtype=torch.long),
        torch.tensor(ion_features,dtype=torch.float32).reshape(-1,3),torch.tensor(formula_samples,dtype=torch.long),
        tensorizer.formulas_to_tensor(formulas),torch.tensor(mzs,dtype=torch.float64),torch.tensor(formula_ids,dtype=torch.long))


def collate_post_materialization(items: Sequence[PostMaterializationBatch], action_offsets: Sequence[int]) -> PostMaterializationBatch:
    molecules=[];trees=[];compounds=[];metadata_trees=[];edges=[];samples=[];actions=[];terminals=[];scores=[]
    ion_nodes=[];ion_features=[];formula_samples=[];formulas=[];mzs=[];formula_ids=[];targets=[]
    node_offset=sample_offset=formula_offset=0
    for item,action_offset in zip(items,action_offsets):
        decoded=item.decoded
        molecules.extend(decoded.node_graph.to_data_list());trees.extend(item.tree_graph.to_data_list())
        compounds.extend(decoded.compounds);metadata_trees.extend(decoded.trees)
        edges.append(decoded.edge_index+node_offset);samples.append(decoded.node_sample_index+sample_offset)
        actions.append(decoded.edge_added_action_index+action_offset);terminals.append(decoded.terminal_node_index+node_offset)
        scores.append(decoded.node_log_score);ion_nodes.append(item.ion_node_index+node_offset)
        ion_features.append(item.ion_features);formula_samples.append(item.formula_sample_index+sample_offset)
        formulas.append(item.formula_tensor);mzs.append(item.formula_mz);formula_ids.append(item.ion_formula_index+formula_offset)
        if item.target_intensity is not None: targets.append(item.target_intensity)
        node_offset+=len(decoded.compounds);sample_offset+=len(decoded.trees);formula_offset+=item.formula_tensor.shape[0]
    if targets and len(targets)!=len(items):
        raise ValueError("Cannot mix supervised and inference downstream batches")
    decoded=DecodedFragmentTreeBatch(Batch.from_data_list(molecules),torch.cat(edges,dim=1),torch.cat(samples),torch.cat(actions),
        torch.cat(terminals),torch.cat(scores),tuple(metadata_trees),tuple(compounds),sum(i.decoded.rdkit_run_count for i in items),sum(i.decoded.failed_effect_count for i in items))
    return PostMaterializationBatch(decoded,Batch.from_data_list(trees),torch.cat(ion_nodes),torch.cat(ion_features),
        torch.cat(formula_samples),torch.cat(formulas),torch.cat(mzs),torch.cat(formula_ids),torch.cat(targets) if targets else None)
