"""Preparation-only action teachers from original spectra and Source chemistry."""
from __future__ import annotations
from dataclasses import replace
from collections.abc import Sequence
from types import SimpleNamespace
import torch
from clefts.libs.mmkit.mmkit import Compound, Adduct
from clefts.domain.fragment.cleavage import CleavageActionSequence
from .source_action_structure import SourceActionStructure, prepare_source_actions, UnresolvedPrecursorError
from ..specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from ..specgen.components.action.action_decoder import ActionDecoderOutput
from ..specgen.materialization import materialize_action_states
from ..specgen.post_materialization_model import prepare_post_materialization


def _walk_pathway_chains(tree, fragmenter, pathway):
    """Retain every supported transition history, including precursor provenance."""
    states=[(0, None, ((None,None),))]
    for node in pathway.nodes[1:]:
        next_states=[]
        for index,state,chain in states:
            for edge in tree.get_out_edges(index):
                if tree.get_node(edge.target_index).smiles!=node.smiles:continue
                for transition in edge.transitions:
                    if transition.parent_action_sequence!=state:continue
                    action=transition.added_action
                    if state is not None and action is not None:
                        if not set(action.source_atom_maps)<=state.retained_atom_maps or any(previous.changed_bond_maps & action.matched_bond_maps for previous in state.actions):continue
                    if node.is_precursor and len(transition.action_sequence.actions)>fragmenter.precursor_candidate_max_action_count:continue
                    next_states.append((edge.target_index,transition.action_sequence,(*chain,(transition.action_sequence,action))))
        states=next_states
    return tuple(dict.fromkeys(chain for _,_,chain in states))


def _walk_pathway(tree,fragmenter,pathway):
    return {chain[-1][0] for chain in _walk_pathway_chains(tree,fragmenter,pathway)}


class ActionStructureBuilder:
    def __init__(self, generator: SourceAnchoredFragmentSpectrumGenerator, *, max_node: int = -1, max_edge: int = -1) -> None:
        self.generator = generator
        self.max_node = max_node
        self.max_edge = max_edge

    def build(self, source: Compound, precursor_types: Sequence[Adduct], collision_energy: Sequence[float],
              peaks_mz: Sequence[Sequence[float]], peaks_intensity: Sequence[Sequence[float]]) -> tuple[SourceActionStructure, tuple[int, ...]]:
        """Return the built structure plus which input sample indices it kept.

        A sample whose precursor ion is unreachable under the configured
        cleavage patterns (e.g. an ion shift the patterns cannot produce) is
        dropped rather than failing the whole group: real MS2 fragmentation
        never happens on an unreachable precursor, so such a sample has
        nothing to supervise. The caller uses the returned indices to report
        which of its original records were actually kept.
        """
        if not precursor_types or not len(precursor_types)==len(collision_energy)==len(peaks_mz)==len(peaks_intensity):
            raise ValueError("Every spectrum needs precursor, CE, peaks and intensities")
        for mz,intensity in zip(peaks_mz,peaks_intensity):
            if len(mz)!=len(intensity):
                raise ValueError("Peak m/z and intensity lengths differ")
        generator=self.generator
        fragmenter=generator.fragmenter
        actions=fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source)
        # Full chemistry is permitted here, never inside neural forward.
        tree=fragmenter.build_fragment_ion_tree(source,max_node=self.max_node,max_edge=self.max_edge,_include_fragment_compound_cache=True)
        # Recorded once per sample, from the SAME already-built tree: which
        # Source action set(s) already select this sample's precursor ion.
        # Real MS2 fragmentation always happens on that selected ion, so
        # prepare_source_actions requires every other teacher state to be
        # consistent with one of these alternatives. Computed before anything
        # else so unreachable samples can be dropped up front.
        precursor_sequences_all=[tuple(pa.action_sequence for pa in fragmenter.resolve_precursor_actions(tree,precursor_type))
                                 for precursor_type in precursor_types]
        kept=tuple(index for index,sequences in enumerate(precursor_sequences_all) if sequences)
        if not kept:
            raise UnresolvedPrecursorError("No sample in this group has a valid precursor action sequence from Source")
        if len(kept)!=len(precursor_types):
            precursor_types=[precursor_types[index] for index in kept]
            collision_energy=[collision_energy[index] for index in kept]
            peaks_mz=[peaks_mz[index] for index in kept]
            peaks_intensity=[peaks_intensity[index] for index in kept]
            precursor_sequences=[precursor_sequences_all[index] for index in kept]
        else:
            precursor_sequences=precursor_sequences_all
        assignments=fragmenter.assign_fragment_pathways_to_peak_sets(tree,zip(precursor_types,peaks_mz))
        teacher_paths=[]
        for (_,groups),intensities in zip(assignments,peaks_intensity):
            paths=[]
            for group,intensity in zip(groups,intensities):
                for pathway in group:
                    paths.extend((chain,float(intensity)) for chain in _walk_pathway_chains(tree,fragmenter,pathway))
            teacher_paths.append(paths)
        conditions=torch.tensor([[generator.adduct_type_strs.index(str(adduct)),ce]
                                 for adduct,ce in zip(precursor_types,collision_energy)],dtype=torch.float32)
        data=prepare_source_actions(source=source,actions=actions,graph_builder=generator.mol_encoder.graph_builder,
            condition_features=conditions,max_action_count=fragmenter.tree_max_action_count,teacher_pathways=teacher_paths,
            precursor_sequences=precursor_sequences)
        # Materialization adds its own Source bookkeeping roots. Teacher roots
        # are precursor seeds at MS2 depth zero, including nonempty seeds.
        rows=[tuple(data.teacher_node_action_index[a:b].tolist()) for a,b in zip(data.teacher_node_action_ptr[:-1],data.teacher_node_action_ptr[1:])]
        samples=data.teacher_node_sample_index.tolist()
        dense_rows=[];dense_sample=[];parents=[];added=[];terminal=[];teacher_dense={}
        width=fragmenter.tree_max_action_count
        for sample in range(len(precursor_types)):
            source_row=len(dense_rows)
            dense_rows.append(tuple([-1]*width));dense_sample.append(sample);parents.append(-1);added.append(-1);terminal.append(False)
            for node,(owner,ids) in enumerate(zip(samples,rows)):
                if owner!=sample:continue
                parent=int(data.teacher_node_parent_index[node])
                teacher_dense[node]=len(dense_rows)
                dense_rows.append((*ids,*([-1]*(width-len(ids)))))
                dense_sample.append(sample);parents.append(teacher_dense[parent] if parent>=0 else source_row)
                added.append(int(data.teacher_node_added_action_index[node]));terminal.append(True)
        synthetic=ActionDecoderOutput(torch.tensor(dense_sample),torch.tensor(dense_rows,dtype=torch.long),
            torch.zeros(len(dense_rows)),torch.tensor(terminal),torch.tensor(parents),torch.tensor(added),
            SimpleNamespace(action_index=torch.arange(len(actions)).expand(len(precursor_types),-1)))
        decoded=materialize_action_states(synthetic,(source,),(actions,),data.sample_tree_index,generator.mol_encoder.graph_builder)
        downstream=prepare_post_materialization(decoded,fragmenter,precursor_types,generator.tensorizer)
        target=[]
        for sample,mz in zip(downstream.formula_sample_index.tolist(),downstream.formula_mz.tolist()):
            matches=[float(intensity) for observed,intensity in zip(peaks_mz[sample],peaks_intensity[sample])
                     if fragmenter.mass_tolerance.within(observed,mz)]
            target.append(max(matches,default=0.))
        node_by_state={(dense_sample[row],tuple(i for i in dense_rows[row] if i>=0)):node
                       for row,node in zip(decoded.materialized_state_index.tolist(),decoded.materialized_node_index.tolist())}
        state_nodes=torch.tensor([node_by_state.get((sample,tuple(row)),-1) for sample,row in zip(samples,rows)],dtype=torch.long)
        annotations=[]
        for sample,((_,groups),adduct,energy,mzs,intensities) in enumerate(zip(assignments,precursor_types,collision_energy,peaks_mz,peaks_intensity)):
            main=fragmenter._resolve_main_adduct_type(adduct)
            precursor_mz=adduct.apply_to_formula(source.formula).normalized.exact_mass
            peaks=[]
            for peak_index,(mz,intensity,group) in enumerate(zip(mzs,intensities,groups)):
                matches=[]
                for pathway in group:
                    sequences=_walk_pathway(tree,fragmenter,pathway)
                    local_nodes=sorted({node_by_state[(sample,tuple(sorted(actions.index(action) for action in seq.actions)) if seq else ())]
                                        for seq in sequences if (sample,tuple(sorted(actions.index(action) for action in seq.actions)) if seq else ()) in node_by_state})
                    if not local_nodes:continue
                    match=dict(nodeIndices=local_nodes,smiles=pathway.terminal_node.smiles,
                               formula=str(pathway.formula),theoreticalMz=float(pathway.formula.exact_mass),
                               massErrorPpm=(float(mz)-pathway.formula.exact_mass)/pathway.formula.exact_mass*1e6,
                               adduct=str(pathway.adduct),hydrogenShift=pathway.adduct.element_diff.get('H',0)-main.element_diff.get('H',0))
                    if match not in matches:matches.append(match)
                peaks.append(dict(index=peak_index,mz=float(mz),intensity=float(intensity),
                                  precursor=bool(fragmenter.mass_tolerance.within(float(mz),precursor_mz)),matches=matches))
            def score(selected):
                total=sum(peak['intensity'] for peak in selected)
                return sum(peak['intensity'] for peak in selected if peak['matches'])/total if total>0 else None
            annotations.append(dict(adduct=str(adduct),mainAdduct=str(main),collisionEnergy=float(energy),
                                    precursorMz=float(precursor_mz),peaks=peaks,
                                    assignmentScore=score(peaks),
                                    assignmentScoreWithoutPrecursor=score([peak for peak in peaks if not peak['precursor']])))
        return replace(data,state_fragment_node_index=state_nodes,sample_annotations=tuple(annotations),
                       downstream=replace(downstream,target_intensity=torch.tensor(target))),kept
