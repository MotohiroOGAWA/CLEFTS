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
from .materialization import DecodedFragmentTreeBatch


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
    # True for a candidate ion whose (node, net adduct) was matched to an
    # observed peak during dataset preparation; None when unavailable (e.g.
    # inference). Multiple mass-degenerate candidates at the same node can
    # be positive together: they are indistinguishable from MS data alone.
    ion_is_positive: Tensor | None = None
    # str(Adduct) for each candidate ion, parallel to ion_node_index/ion_features;
    # display-only (e.g. the fragment-tree node inspector), never read by the model.
    ion_adduct: tuple[str,...] = ()
    unique_node_graph: Batch | None = None
    node_graph_inverse: Tensor | None = None
    unique_source_index: Tensor | None = None
    unique_smiles: tuple[str,...] = ()
    node_smiles: tuple[str,...] = ()

    def to(self, device: str | torch.device) -> PostMaterializationBatch:
        return replace(self, decoded=self.decoded.to(device), tree_graph=self.tree_graph.to(device),unique_node_graph=self.unique_node_graph.to(device) if self.unique_node_graph is not None else None,
            **{name: getattr(self,name).to(device) for name in ("ion_node_index","ion_features","formula_sample_index","formula_tensor","formula_mz","ion_formula_index","target_intensity","ion_is_positive","node_graph_inverse","unique_source_index") if getattr(self,name) is not None})


@dataclass(frozen=True)
class PostMaterializationOutput:
    intensity: Tensor
    sample_index: Tensor
    formula_tensor: Tensor
    mz: Tensor
    loss: Tensor
    # Highest ion-score sigmoid confidence among the candidates explaining this
    # formula. Never used to gate the loss or the intensity itself (so training
    # dynamics/metrics stay unaffected); callers producing a spectrum for
    # display can drop low-confidence peaks with it instead.
    confidence: Tensor
    # sigmoid(ion_logit) per candidate ion, parallel to the input batch's
    # ion_node_index/ion_adduct; display-only, same reasoning as confidence.
    ion_probability: Tensor


class PostMaterializationFragmentTreeModel(nn.Module):
    requires_action_features = True

    def __init__(self, mol_encoder: nn.Module, action_dim: int, condition_dim: int,
                 formula_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, max_action_count: int = 3,
                 cosine_loss_weight: float = 0.5, ion_loss_weight: float = 0.5,
                 ion_prediction_threshold: float = 0.5, intensity_power: float = 0.5,
                 precursor_free_loss_weight: float = 0.5, dropout: float = 0.) -> None:
        super().__init__()
        if not 0<=cosine_loss_weight or not torch.isfinite(torch.tensor(cosine_loss_weight)):raise ValueError("cosine_loss_weight must be non-negative and finite")
        if not 0<=ion_loss_weight or not torch.isfinite(torch.tensor(ion_loss_weight)):raise ValueError("ion_loss_weight must be non-negative and finite")
        if not 0<ion_prediction_threshold<1:raise ValueError("ion_prediction_threshold must be between zero and one")
        if not 0<intensity_power or not torch.isfinite(torch.tensor(intensity_power)):raise ValueError("intensity_power must be positive and finite")
        if not 0<=precursor_free_loss_weight or not torch.isfinite(torch.tensor(precursor_free_loss_weight)):raise ValueError("precursor_free_loss_weight must be non-negative and finite")
        if not 0<=dropout<1:raise ValueError("dropout must be in [0,1)")
        self.intensity_power=intensity_power
        self.precursor_free_loss_weight=precursor_free_loss_weight
        self.cosine_loss_weight=cosine_loss_weight
        self.ion_loss_weight=ion_loss_weight
        self.ion_prediction_threshold=ion_prediction_threshold
        self.mol_encoder = mol_encoder
        self.edge_encoder = nn.Sequential(nn.Linear(action_dim + mol_encoder.graph_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim))
        self.tree_encoder = GraphormerEncoder(node_dim=mol_encoder.graph_dim,hidden_dim=hidden_dim,edge_dim=hidden_dim,
            condition_dim=condition_dim,condition_token_count=1,num_heads=num_heads,num_layers=num_layers,
            max_spatial_dist=max_action_count+1,max_edge_dist=max_action_count+1,undirected_for_spd=False,undirected_for_path=False,dropout=dropout)
        self.ion_encoder = nn.Linear(3,hidden_dim)
        self.ion_score = nn.Linear(hidden_dim,1)
        self.formula_intensity = FragmentTreeFormulaIntensityPredictor(formula_dim=formula_dim,hidden_dim=hidden_dim)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.mol_encoder.parameters()):self.mol_encoder.eval()
        return self

    def forward(self, data: PostMaterializationBatch, *, action_h: Tensor, condition_h: Tensor, source_embeddings: Tensor | None = None, molecular_embeddings: Tensor | None = None) -> PostMaterializationOutput:
        # Target graphs are consumed only here, after action selection.
        if molecular_embeddings is not None:return self._forward_embeddings(data,molecular_embeddings,action_h,condition_h)
        if data.unique_node_graph is None:
            molecular=self.mol_encoder(data.decoded.node_graph).embeddings
        else:
            if source_embeddings is not None and data.unique_source_index is not None:
                shared=data.unique_source_index>=0
                molecular=source_embeddings.new_zeros((data.unique_source_index.numel(),self.mol_encoder.graph_dim))
                molecular[shared]=source_embeddings[data.unique_source_index[shared]]
                if torch.any(~shared):
                    # Graph lists are built on CPU in preparation, not in forward.
                    molecular[~shared]=self.mol_encoder(data.unique_node_graph).embeddings
            else:
                molecular=self.mol_encoder(data.decoded.node_graph).embeddings
                return self._forward_embeddings(data,molecular,action_h,condition_h)
            molecular=molecular[data.node_graph_inverse]
        return self._forward_embeddings(data,molecular,action_h,condition_h)

    def _forward_embeddings(self,data,molecular,action_h,condition_h):
        edges = data.decoded.edge_index
        added = data.decoded.edge_added_action_index
        action=action_h.new_zeros((added.numel(),action_h.shape[1]))
        normal=added>=0
        action[normal]=action_h[added[normal]]
        seeds=data.decoded.edge_seed_action_index
        if seeds.numel():
            seed_h=action_h.new_zeros(action.shape).index_add_(0,seeds[0],action_h[seeds[1]])
            count=action_h.new_zeros(added.shape).index_add_(0,seeds[0],torch.ones_like(seeds[0],dtype=action_h.dtype))
            action=action+seed_h/count[:,None].clamp_min(1)
        edge_h = self.edge_encoder(torch.cat((action,molecular[edges[0]],molecular[edges[1]]),dim=1))
        tree = data.tree_graph.clone()
        tree.x = molecular
        tree.edge_attr = edge_h
        node_h, _ = self.tree_encoder(tree,condition_repr=condition_h)
        ion_h = node_h[data.ion_node_index] + self.ion_encoder(data.ion_features)
        ion_logit = self.ion_score(F.gelu(ion_h)).squeeze(-1)
        ion_score = F.softplus(ion_logit)
        num_formulas = data.formula_tensor.shape[0]
        score = ion_score.new_zeros(num_formulas).scatter_add_(0,data.ion_formula_index,ion_score)
        count = ion_score.new_zeros(num_formulas).scatter_add_(0,data.ion_formula_index,torch.ones_like(ion_score))
        # The most confident candidate ion explaining each formula. Kept out of
        # the loss/intensity path entirely (that stays exactly as trained) so a
        # caller building a spectrum for display can drop unconfident peaks
        # (e.g. below self.ion_prediction_threshold) without changing what the
        # model was actually optimized and validated against.
        confidence = ion_logit.new_zeros(num_formulas).scatter_reduce_(0,data.ion_formula_index,ion_logit.sigmoid(),reduce='amax',include_self=True)
        intensity = self.formula_intensity(data.formula_tensor,score,count)
        sample_count=data.tree_graph.num_graphs
        # Displayed/predicted spectrum always includes the precursor peak and is
        # never power-transformed; only the training losses below are.
        maximum=intensity.new_zeros(sample_count).scatter_reduce_(0,data.formula_sample_index,intensity,reduce='amax',include_self=True)
        normalized_intensity=intensity/maximum[data.formula_sample_index].clamp_min(1e-8)

        # Root (source/precursor) node per sample: the first node materialized
        # for that sample. A formula is "the precursor formula" if one of its
        # candidate ions sits on that node (prepare_post_materialization already
        # only keeps a node-0 candidate there when it matches the true precursor).
        node_sample = data.decoded.node_sample_index
        root_node = node_sample.new_full((sample_count,), node_sample.numel())
        root_node.scatter_reduce_(0,node_sample,torch.arange(node_sample.numel(),device=node_sample.device),reduce='amin',include_self=True)
        ion_is_root = data.ion_node_index==root_node[node_sample[data.ion_node_index]]
        formula_is_precursor = ion_score.new_zeros(num_formulas).scatter_add_(0,data.ion_formula_index,ion_is_root.to(ion_score.dtype))>0

        def spectrum_loss(keep_formula,keep_ion):
            """Every loss term recomputed over only keep_formula/keep_ion, each
            re-normalized to its own max=1. Used once over everything, and once
            with the precursor excluded so its usually-dominant magnitude cannot
            starve gradient for the rest of the spectrum; the model itself is
            never asked to predict anything but the full, precursor-included
            spectrum (see normalized_intensity above)."""
            total = intensity.sum()*0
            if data.target_intensity is not None and intensity.numel() and torch.any(keep_formula):
                masked=intensity*keep_formula
                peak_max=masked.new_zeros(sample_count).scatter_reduce_(0,data.formula_sample_index,masked,reduce='amax',include_self=True)
                pred=(masked/peak_max[data.formula_sample_index].clamp_min(1e-8))[keep_formula]
                masked_target=data.target_intensity.clamp_min(0)*keep_formula
                target_max=masked_target.new_zeros(sample_count).scatter_reduce_(0,data.formula_sample_index,masked_target,reduce='amax',include_self=True)
                target=(masked_target/target_max[data.formula_sample_index].clamp_min(1e-8))[keep_formula]
                # A square-root-like power transform is standard in spectral
                # similarity: it keeps the model outputting ordinary intensities,
                # but the loss itself weighs small peaks much more relative to
                # the base peak than a raw linear/log1p comparison would.
                pred=pred.clamp_min(0).pow(self.intensity_power);target=target.pow(self.intensity_power)
                total=total+F.mse_loss(pred,target)
                group=data.formula_sample_index[keep_formula]
                dot=pred.new_zeros(sample_count).scatter_add_(0,group,pred*target)
                pred_norm=pred.new_zeros(sample_count).scatter_add_(0,group,pred.square())
                target_norm=pred.new_zeros(sample_count).scatter_add_(0,group,target.square())
                observed=target_norm>0
                cosine=dot/(pred_norm*target_norm).clamp_min(1e-16).sqrt()
                if torch.any(observed):total=total+self.cosine_loss_weight*(1-cosine[observed]).mean()
            if data.ion_is_positive is not None and ion_logit.numel():
                # Positive and negative candidate ions are mean-normalized
                # separately (like multi_positive_loss in
                # fragment_tree_training/model.py) so an imbalanced count on
                # either side cannot let the model coast by scoring every
                # candidate the same way.
                positive=data.ion_is_positive & keep_ion
                negative=(~data.ion_is_positive) & keep_ion
                positive_loss=F.softplus(-ion_logit[positive])
                negative_loss=F.softplus(ion_logit[negative])
                ion_loss=(positive_loss.mean() if positive_loss.numel() else ion_logit.sum()*0) \
                    + (negative_loss.mean() if negative_loss.numel() else ion_logit.sum()*0)
                total=total+self.ion_loss_weight*ion_loss
            return total

        keep_all_formula=formula_is_precursor.new_ones(num_formulas)
        keep_all_ion=ion_logit.new_ones(ion_logit.shape[0],dtype=torch.bool)
        loss=spectrum_loss(keep_all_formula,keep_all_ion)
        if self.precursor_free_loss_weight>0 and num_formulas:
            keep_formula=~formula_is_precursor
            keep_ion=~formula_is_precursor[data.ion_formula_index] if data.ion_formula_index.numel() else keep_all_ion
            loss=loss+self.precursor_free_loss_weight*spectrum_loss(keep_formula,keep_ion)
        return PostMaterializationOutput(normalized_intensity,data.formula_sample_index,data.formula_tensor,data.formula_mz,loss,confidence,ion_logit.sigmoid())


def prepare_post_materialization(decoded: DecodedFragmentTreeBatch, fragmenter: Fragmenter,
                                 precursor_types: Sequence[Adduct], tensorizer: FormulaTensorizer,
                                 ion_positive_sets: Sequence[set[tuple[int, str]]] | None = None) -> PostMaterializationBatch:
    """Ion/formula preparation is outside all neural forwards.

    ion_positive_sets, when given, has one set per sample of (node_index, str(adduct))
    pairs already confirmed by an observed peak during dataset preparation (see
    ActionStructureBuilder.build). A candidate ion is positive when its own
    (node, net adduct) is in that set; nothing is asserted for inference callers,
    which omit this argument entirely.
    """
    graphs, node_ids, ion_features, formulas, formula_samples, mzs, formula_ids, ion_adduct = [], [], [], [], [], [], [], []
    ion_is_positive = [] if ion_positive_sets is not None else None
    node_offset = 0
    # Every generated node can emit ions, including observed intermediates
    # that also have positive outgoing branches. Terminal only stops expansion.
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
                if node not in reachable:
                    continue
                adduct = group.candidate_adducts[group.candidate_adduct_indices[entry]]
                if node == 0 and adduct.apply_to_formula(compounds[0].formula).normalized != precursor_types[sample].apply_to_formula(compounds[0].formula).normalized:
                    continue
                selected.append((node,float(group.hydrogen_candidate_indices[entry]),float(group.shift_rule_indices[entry]),float(adduct.charge),adduct))
            if not selected:
                continue
            new_index=len(formulas)
            formulas.append(formula)
            formula_samples.append(sample)
            mzs.append(formula.exact_mass)
            for node,hydrogen,shift,charge,adduct in selected:
                node_ids.append(node_offset+node)
                ion_features.append((hydrogen,shift,charge))
                formula_ids.append(new_index)
                ion_adduct.append(str(adduct))
                if ion_is_positive is not None:
                    ion_is_positive.append((node,str(adduct)) in ion_positive_sets[sample])
        node_offset += tree.num_nodes
    return PostMaterializationBatch(decoded,Batch.from_data_list(graphs),torch.tensor(node_ids,dtype=torch.long),
        torch.tensor(ion_features,dtype=torch.float32).reshape(-1,3),torch.tensor(formula_samples,dtype=torch.long),
        tensorizer.formulas_to_tensor(formulas),torch.tensor(mzs,dtype=torch.float64),torch.tensor(formula_ids,dtype=torch.long),
        ion_is_positive=torch.tensor(ion_is_positive,dtype=torch.bool) if ion_is_positive is not None else None,
        ion_adduct=tuple(ion_adduct),
        node_smiles=tuple(compound.smiles for compound in decoded.compounds))


def collate_post_materialization(items: Sequence[PostMaterializationBatch], action_offsets: Sequence[int]) -> PostMaterializationBatch:
    molecules=[];trees=[];compounds=[];metadata_trees=[];edges=[];samples=[];actions=[];terminals=[];scores=[]
    ion_nodes=[];ion_features=[];formula_samples=[];formulas=[];mzs=[];formula_ids=[];targets=[];ion_positives=[];ion_adducts=[]
    node_offset=sample_offset=formula_offset=0
    node_smiles=[]
    seed_actions=[];edge_offset=0
    for item,action_offset in zip(items,action_offsets):
        decoded=item.decoded
        node_smiles.extend(item.node_smiles)
        molecules.extend(decoded.node_graph.to_data_list());trees.extend(item.tree_graph.to_data_list())
        compounds.extend(decoded.compounds);metadata_trees.extend(decoded.trees)
        edges.append(decoded.edge_index+node_offset);samples.append(decoded.node_sample_index+sample_offset)
        actions.append(torch.where(decoded.edge_added_action_index>=0,decoded.edge_added_action_index+action_offset,decoded.edge_added_action_index))
        seed_actions.append(decoded.edge_seed_action_index+torch.tensor([[edge_offset],[action_offset]]));edge_offset+=decoded.edge_index.shape[1];terminals.append(decoded.terminal_node_index+node_offset)
        scores.append(decoded.node_log_score);ion_nodes.append(item.ion_node_index+node_offset)
        ion_features.append(item.ion_features);formula_samples.append(item.formula_sample_index+sample_offset)
        formulas.append(item.formula_tensor);mzs.append(item.formula_mz);formula_ids.append(item.ion_formula_index+formula_offset)
        if item.target_intensity is not None: targets.append(item.target_intensity)
        if item.ion_is_positive is not None: ion_positives.append(item.ion_is_positive)
        ion_adducts.append(item.ion_adduct)
        node_offset+=len(decoded.compounds);sample_offset+=len(decoded.trees);formula_offset+=item.formula_tensor.shape[0]
    if targets and len(targets)!=len(items):
        raise ValueError("Cannot mix supervised and inference downstream batches")
    if ion_positives and len(ion_positives)!=len(items):
        raise ValueError("Cannot mix downstream batches with and without ion positive/negative labels")
    decoded=DecodedFragmentTreeBatch(Batch.from_data_list(molecules),torch.cat(edges,dim=1),torch.cat(samples),torch.cat(actions),
        torch.cat(terminals),torch.cat(scores),tuple(metadata_trees),tuple(compounds),sum(i.decoded.rdkit_run_count for i in items),sum(i.decoded.failed_effect_count for i in items),edge_seed_action_index=torch.cat(seed_actions,dim=1))
    return PostMaterializationBatch(decoded,Batch.from_data_list(trees),torch.cat(ion_nodes),torch.cat(ion_features),
        torch.cat(formula_samples),torch.cat(formulas),torch.cat(mzs),torch.cat(formula_ids),torch.cat(targets) if targets else None,
        ion_is_positive=torch.cat(ion_positives) if ion_positives else None,ion_adduct=sum(ion_adducts,()),node_smiles=tuple(node_smiles))


def deduplicate_molecular_graphs(data: PostMaterializationBatch, source_smiles=()):
    """Called during CPU preparation/loading; reuse graph embeddings across CE/adducts."""
    graphs=data.decoded.node_graph.to_data_list()
    source_index={smiles:i for i,smiles in enumerate(source_smiles)}
    keys={};unique=[];inverse=[];shared=[]
    if len(data.node_smiles)!=len(graphs):raise ValueError('Molecular identities must be prepared before training')
    for graph,key in zip(graphs,data.node_smiles):
        if key not in keys:
            keys[key]=len(shared);shared.append(source_index.get(key,-1))
            if key not in source_index:unique.append(graph)
        inverse.append(keys[key])
    # A source-only tree needs no additional molecular encoding.
    unique_batch=Batch.from_data_list(unique) if unique else Batch.from_data_list([graphs[0]])
    return replace(data,unique_node_graph=unique_batch,node_graph_inverse=torch.tensor(inverse,dtype=torch.long),unique_source_index=torch.tensor(shared,dtype=torch.long),unique_smiles=tuple(keys))
