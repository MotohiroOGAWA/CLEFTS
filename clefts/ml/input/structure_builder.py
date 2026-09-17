"""Preparation-only action teachers from original spectra and Source chemistry."""
from __future__ import annotations
from dataclasses import replace
from collections.abc import Sequence
from types import SimpleNamespace
import torch
from clefts.libs.mmkit.mmkit import Compound, Adduct
from clefts.domain.fragment.cleavage import CleavageActionSequence
from .source_action_structure import SourceActionStructure, prepare_source_actions
from ..specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from ..specgen.components.action.action_decoder import ActionDecoderOutput
from ..specgen.materialization import materialize_action_states
from ..specgen.post_materialization_model import prepare_post_materialization


def _walk_pathway(tree, fragmenter, pathway) -> set:
    """Follow a domain FragmentPathway (always rooted at Source) through tree
    edges/transitions, returning the resulting CleavageActionSequence set."""
    states = {(0, None)}
    for node in pathway.nodes[1:]:
        next_states = set()
        for index, state in states:
            for edge in tree.get_out_edges(index):
                if tree.get_node(edge.target_index).smiles != node.smiles:
                    continue
                for transition in edge.transitions:
                    if transition.parent_action_sequence != state:
                        continue
                    if node.is_precursor and len(transition.action_sequence.actions) > fragmenter.precursor_candidate_max_action_count:
                        continue
                    next_states.add((edge.target_index, transition.action_sequence))
        states = next_states
    return {state for _, state in states}


class ActionStructureBuilder:
    def __init__(self, generator: SourceAnchoredFragmentSpectrumGenerator, *, max_node: int = -1, max_edge: int = -1) -> None:
        self.generator = generator
        self.max_node = max_node
        self.max_edge = max_edge

    def build(self, source: Compound, precursor_types: Sequence[Adduct], collision_energy: Sequence[float],
              peaks_mz: Sequence[Sequence[float]], peaks_intensity: Sequence[Sequence[float]]) -> SourceActionStructure:
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
        assignments=fragmenter.assign_fragment_pathways_to_peak_sets(tree,zip(precursor_types,peaks_mz))
        targets=[]
        for _,groups in assignments:
            sequences=set()
            for group in groups:
                for pathway in group:
                    sequences.update(_walk_pathway(tree,fragmenter,pathway))
            targets.append(tuple(sorted(sequences,key=lambda seq:seq.key if seq else ())))
        # Recorded once per sample, from the SAME already-built tree: which
        # Source action set(s) already select this sample's precursor ion.
        # Real MS2 fragmentation always happens on that selected ion, so
        # prepare_source_actions requires every other teacher state to be
        # consistent with one of these alternatives.
        precursor_sequences=[tuple(pa.action_sequence for pa in fragmenter.resolve_precursor_actions(tree,precursor_type))
                             for precursor_type in precursor_types]
        conditions=torch.tensor([[generator.adduct_type_strs.index(str(adduct)),ce]
                                 for adduct,ce in zip(precursor_types,collision_energy)],dtype=torch.float32)
        data=prepare_source_actions(source=source,actions=actions,graph_builder=generator.mol_encoder.graph_builder,
            condition_features=conditions,max_action_count=fragmenter.tree_max_action_count,target_sequences=targets,
            precursor_sequences=precursor_sequences)
        # Reconstruct a preparation-only DAG from all teacher states. Parent
        # traversal includes all valid orders for supervision; materialization
        # chooses a representative ancestor chain for graph presentation.
        rows=[]
        for start,stop in zip(data.teacher_state_action_ptr[:-1],data.teacher_state_action_ptr[1:]):
            rows.append(tuple(data.teacher_state_action_index[start:stop].tolist()))
        state_by_row=[CleavageActionSequence(actions[i] for i in row) if row else None for row in rows]
        samples=data.teacher_state_sample_index.tolist()
        dense_rows, dense_sample, parents, added, terminal=[],[],[],[],[]
        for sample in range(len(precursor_types)):
            known={state_by_row[i]:i for i,s in enumerate(samples) if s==sample}
            if None not in known:
                continue
            queue=[(None,-1,-1)]
            visited={None}
            while queue:
                state,parent,action=queue.pop(0)
                row_index=len(dense_rows)
                ids=tuple(actions.index(a) for a in state.actions) if state else ()
                dense_rows.append((*ids,*([-1]*(fragmenter.tree_max_action_count-len(ids)))))
                dense_sample.append(sample)
                parents.append(parent)
                added.append(action)
                terminal.append(bool(data.teacher_positive_eos[known[state]]))
                original=known[state]
                start,stop=data.teacher_positive_action_ptr[original:original+2].tolist()
                # prepare_source_actions already guarantees every teacher state
                # here is precursor-consistent, so any traversal order yields a
                # representative chain that passes through the precursor node.
                for action_index in data.teacher_positive_action_index[start:stop].tolist():
                    child=CleavageActionSequence((*(state.actions if state else ()),actions[action_index]))
                    if child in known and child not in visited:
                        visited.add(child)
                        queue.append((child,row_index,action_index))
        # Even records with no assignments have a Source graph, but no terminal
        # targets. BOS is inserted solely as a materialization bookkeeping root.
        present=set(dense_sample)
        for sample in range(len(precursor_types)):
            if sample not in present:
                dense_rows.append(tuple([-1]*fragmenter.tree_max_action_count))
                dense_sample.append(sample);parents.append(-1);added.append(-1);terminal.append(False)
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
        return replace(data,state_fragment_node_index=state_nodes,
                       downstream=replace(downstream,target_intensity=torch.tensor(target)))
