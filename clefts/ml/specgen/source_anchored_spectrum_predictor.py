"""Source -> actions -> Torch decode -> selected molecules -> tree/intensity."""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.libs.mmkit.mmkit import Adduct, Compound
from ..input.source_action_structure import SourceActionStructure, prepare_source_actions
from ..mol.mol_encoder import MolEncoder
from ..mol.formula_encoder import FormulaTensorizer
from .components.condition.condition_encoder import MS2ConditionEncoder
from .components.action.action_decoder import ActionDecoderOutput
from .source_action_feature_model import SourceActionFeatureModel, SourceActionFeatures
from .materialization import materialize_action_states, DecodedFragmentTreeBatch
from .post_materialization_model import PostMaterializationFragmentTreeModel, prepare_post_materialization, PostMaterializationOutput, PostMaterializationBatch


@dataclass(frozen=True)
class SourceAnchoredSelectionOutput:
    features: SourceActionFeatures
    decoded: ActionDecoderOutput


@dataclass(frozen=True)
class SourceAnchoredSpectrumOutput:
    selection: SourceAnchoredSelectionOutput
    fragments: DecodedFragmentTreeBatch
    spectra: PostMaterializationOutput
    # ion_node_index/ion_formula_index link each candidate ion to its
    # fragment-tree node and formula; consumers that need that mapping (e.g.
    # a fragment-tree viewer) read it from here instead of re-deriving it.
    downstream: PostMaterializationBatch
    # Grouped preparation may reorder records; this maps internal -> input.
    sample_input_index: tuple[int, ...]


class SourceAnchoredFragmentSpectrumGenerator(nn.Module):
    architecture = "source-anchored-action-autoregressive-v1"

    def __init__(self, fragmenter_params: dict, mol_encoder_params: dict,
                 action_model_params: dict | None = None, post_model_params: dict | None = None,
                 max_samples: int = 128,
                 adduct_type_strs: Sequence[str] | None = None, fine_tuning: dict | None = None) -> None:
        super().__init__()
        if type(max_samples) is not int or max_samples<1:raise ValueError("max_samples must be positive")
        self.max_samples=max_samples
        self.fragmenter = Fragmenter.from_dict(fragmenter_params)
        self.adduct_type_strs = tuple(dict.fromkeys(str(Adduct.parse(value)) for value in
            (adduct_type_strs or (str(adduct) for adduct in self.fragmenter.adduct_types))))
        params = dict(action_model_params or {})
        hidden = params.get("hidden_dim",128)
        condition_dim = params.get("condition_dim",128)
        self.mol_encoder = MolEncoder(**mol_encoder_params)
        condition = MS2ConditionEncoder(adduct_type_strs=self.adduct_type_strs,adduct_embedding_dim=32,
            ce_feature_dim=16,feature_dim=condition_dim)
        patterns=self.fragmenter.fragment_ion_tree_builder.cleavage_patterns
        params.setdefault("max_roles",max(64,max((pattern.reactant_query.GetNumAtoms() for pattern in patterns),default=0)))
        sizes=(max((pattern.pattern_id for pattern in patterns),default=0)+1,
               max((reaction.id for pattern in patterns for reaction in pattern.cleavage_reactions),default=0)+1,
               max((len(reaction.prod_temp) for pattern in patterns for reaction in pattern.cleavage_reactions),default=1))
        self.feature_model=SourceActionFeatureModel(self.mol_encoder,sizes,2,
            max_action_count=self.fragmenter.tree_max_action_count,condition_encoder=condition,**params)
        self.tensorizer=FormulaTensorizer.from_symbols_and_adducts(symbols=self.mol_encoder.symbols,adducts=self.fragmenter.adduct_types)
        post_params=dict(post_model_params or {})
        post_params.setdefault("hidden_dim",hidden)
        self.post_model=PostMaterializationFragmentTreeModel(self.mol_encoder,hidden,condition_dim,self.tensorizer.dim,
            max_action_count=self.fragmenter.tree_max_action_count,**post_params)
        if fine_tuning:
            from .fine_tuning import install_expansion
            install_expansion(self, fine_tuning)

    def forward(self, data: SourceActionStructure) -> SourceAnchoredSelectionOutput:
        """Neural forward, including validation, never invokes RDKit."""
        features=self.feature_model(data)
        return SourceAnchoredSelectionOutput(features,self.feature_model.decoder(features.pool,features.condition_h))

    def prepare(self, sources: Sequence[Compound], precursor_types: Sequence[Adduct], collision_energy: Sequence[float]) -> tuple[SourceActionStructure, tuple[Compound,...], tuple[tuple,...], tuple[Adduct,...], tuple[int,...]]:
        if not sources or not len(sources)==len(precursor_types)==len(collision_energy):
            raise ValueError("sources, precursor_types and collision_energy need equal nonzero lengths")
        grouped: dict[str,list[int]]={}
        for i,source in enumerate(sources):
            grouped.setdefault(source.smiles,[]).append(i)
        structures,unique_sources,universes,ordered_adducts,input_indices=[],[],[],[],[]
        for rows in grouped.values():
            source=sources[rows[0]]
            actions=self.fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source)
            conditions=torch.tensor([[self.adduct_type_strs.index(str(precursor_types[i])),collision_energy[i]] for i in rows],dtype=torch.float32)
            # Bounded by precursor_candidate_max_action_count (not the full
            # tree_max_action_count), so resolving a given precursor stays a
            # small, fixed-cost preparation step, never an exhaustive search.
            precursor_tree=self.fragmenter.build_fragment_ion_tree(source,
                max_action_count=self.fragmenter.precursor_candidate_max_action_count,_include_fragment_compound_cache=True)
            precursor_cache={}
            for i in rows:
                key=str(precursor_types[i])
                if key not in precursor_cache:
                    sequences={pa.action_sequence for pa in self.fragmenter.resolve_precursor_actions(precursor_tree,precursor_types[i])}
                    precursor_cache[key]=tuple(sorted(sequences,key=lambda seq:(seq is not None,seq.key if seq else ())))
            precursor_sequences=[precursor_cache[str(precursor_types[i])] for i in rows]
            structures.append(prepare_source_actions(source=source,actions=actions,graph_builder=self.mol_encoder.graph_builder,
                condition_features=conditions,max_action_count=self.feature_model.max_action_count,
                precursor_sequences=precursor_sequences))
            unique_sources.append(source)
            universes.append(actions)
            ordered_adducts.extend(precursor_types[i] for i in rows)
            input_indices.extend(rows)
        data=SourceActionStructure.from_structures(structures)
        return data,tuple(unique_sources),tuple(universes),tuple(ordered_adducts),tuple(input_indices)

    @torch.no_grad()
    def predict_batches(self,sources,precursor_types,collision_energy,*,max_samples=None):
        limit=self.max_samples if max_samples is None else max_samples
        if self.training:raise ValueError('Call eval() before predict_batches()')
        if type(limit) is not int or limit<1:raise ValueError('max_samples must be positive')
        if not sources or not len(sources)==len(precursor_types)==len(collision_energy):raise ValueError('Input lengths must match and be nonzero')
        # Keep every condition of a shared source in one group. Packs bound the
        # source/action tensors too, rather than merely slicing final scores.
        groups={}
        for i,source in enumerate(sources):groups.setdefault(source.smiles,[]).append(i)
        packs=[];current=[]
        for rows in groups.values():
            if current and len(current)+len(rows)>limit:packs.append(current);current=[]
            current.extend(rows)
            if len(current)>=limit:packs.append(current);current=[]
        if current:packs.append(current)
        for rows in packs:
            for output in self._predict_group_batches([sources[i] for i in rows],[precursor_types[i] for i in rows],[collision_energy[i] for i in rows],max_samples=limit):
                yield replace(output,sample_input_index=tuple(rows[i] for i in output.sample_input_index))

    @torch.no_grad()
    def _predict_group_batches(self, sources, precursor_types, collision_energy, *, max_samples=None):
        """Yield bounded outputs, with global input indices and shared static tokens."""
        if self.training:raise ValueError("Call eval() before predict_batches()")
        limit=self.max_samples if max_samples is None else max_samples
        if type(limit) is not int or limit<1:raise ValueError("max_samples must be positive")
        data,unique_sources,actions,adducts,indices=self.prepare(sources,precursor_types,collision_energy)
        from ..input.action_batching import select_samples
        from .post_materialization_model import deduplicate_molecular_graphs
        device=next(self.parameters()).device
        static=self.feature_model.encode_static(data,max_graphs=limit)
        effect_cache={}
        molecular_cache={source.smiles:static[1][i] for i,source in enumerate(unique_sources)}
        for start in range(0,data.num_samples,limit):
            stop=min(start+limit,data.num_samples)
            chunk=select_samples(data,range(start,stop)).to(device)
            features=self.feature_model(chunk,static_features=static)
            selection=SourceAnchoredSelectionOutput(features,self.feature_model.decoder(features.pool,features.condition_h))
            decoded=materialize_action_states(selection.decoded,unique_sources,actions,chunk.sample_tree_index,self.mol_encoder.graph_builder,effect_cache=effect_cache)
            downstream=prepare_post_materialization(decoded,self.fragmenter,adducts[start:stop],self.tensorizer)
            downstream=deduplicate_molecular_graphs(downstream,tuple(source.smiles for source in unique_sources))
            graph_keys=[key for key,index in zip(downstream.unique_smiles,downstream.unique_source_index.tolist()) if index<0]
            graphs=downstream.unique_node_graph.to_data_list()
            missing=[i for i,key in enumerate(graph_keys) if key not in molecular_cache]
            if missing:
                from torch_geometric.data import Batch
                # Also bound molecular graph encoding for large selected trees.
                for offset in range(0,len(missing),limit):
                    ids=missing[offset:offset+limit]
                    embeddings=self.mol_encoder(Batch.from_data_list([graphs[i] for i in ids]).to(device)).embeddings
                    molecular_cache.update((graph_keys[i],embeddings[j]) for j,i in enumerate(ids))
            molecular=torch.stack([molecular_cache[key] for key in downstream.unique_smiles])[downstream.node_graph_inverse.to(device)]
            downstream=downstream.to(device)
            spectra=self.post_model(downstream,action_h=features.action_h,condition_h=features.condition_h,source_embeddings=features.source_embeddings,molecular_embeddings=molecular)
            yield SourceAnchoredSpectrumOutput(selection,decoded,spectra,downstream,indices[start:stop])

    @torch.no_grad()
    def predict(self, sources, precursor_types, collision_energy):
        if len(sources)>self.max_samples:
            raise ValueError("Use predict_batches() for inputs exceeding max_samples")
        return next(self.predict_batches(sources,precursor_types,collision_energy))
