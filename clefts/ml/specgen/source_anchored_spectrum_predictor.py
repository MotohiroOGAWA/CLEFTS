"""Source -> actions -> Torch decode -> selected molecules -> tree/intensity."""
from __future__ import annotations
from dataclasses import dataclass
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
from .action_materialization import materialize_action_states, DecodedFragmentTreeBatch
from .post_materialization_model import PostMaterializationFragmentTreeModel, prepare_post_materialization, PostMaterializationOutput


@dataclass(frozen=True)
class SourceAnchoredSelectionOutput:
    features: SourceActionFeatures
    decoded: ActionDecoderOutput


@dataclass(frozen=True)
class SourceAnchoredSpectrumOutput:
    selection: SourceAnchoredSelectionOutput
    fragments: DecodedFragmentTreeBatch
    spectra: PostMaterializationOutput
    # Grouped preparation may reorder records; this maps internal -> input.
    sample_input_index: tuple[int, ...]


class SourceAnchoredFragmentSpectrumGenerator(nn.Module):
    architecture = "source-anchored-action-autoregressive-v1"

    def __init__(self, fragmenter_params: dict, mol_encoder_params: dict,
                 action_model_params: dict | None = None, post_model_params: dict | None = None,
                 adduct_type_strs: Sequence[str] | None = None) -> None:
        super().__init__()
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
            structures.append(prepare_source_actions(source=source,actions=actions,graph_builder=self.mol_encoder.graph_builder,
                condition_features=conditions,max_action_count=self.feature_model.max_action_count))
            unique_sources.append(source)
            universes.append(actions)
            ordered_adducts.extend(precursor_types[i] for i in rows)
            input_indices.extend(rows)
        data=SourceActionStructure.from_structures(structures)
        return data,tuple(unique_sources),tuple(universes),tuple(ordered_adducts),tuple(input_indices)

    @torch.no_grad()
    def predict(self, sources: Sequence[Compound], precursor_types: Sequence[Adduct], collision_energy: Sequence[float]) -> SourceAnchoredSpectrumOutput:
        if self.training:
            raise ValueError("Call eval() before predict(); training uses precomputed schema v4 data")
        data,sources,actions,adducts,indices=self.prepare(sources,precursor_types,collision_energy)
        device=next(self.parameters()).device
        data=data.to(device)
        selection=self(data)
        decoded=materialize_action_states(selection.decoded,sources,actions,data.sample_tree_index,self.mol_encoder.graph_builder)
        downstream=prepare_post_materialization(decoded,self.fragmenter,adducts,self.tensorizer).to(device)
        spectra=self.post_model(downstream,action_h=selection.features.action_h,condition_h=selection.features.condition_h)
        return SourceAnchoredSpectrumOutput(selection,decoded,spectra,indices)
