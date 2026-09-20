"""Schema v6 Source/action tensors. RDKit is confined to preparation."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from collections.abc import Iterable, Sequence
import torch
from torch import Tensor
from torch_geometric.data import Batch
from clefts.domain.fragment.cleavage import CleavageAction, CleavageActionSequence
from clefts.domain.fragment.cleavage.CleavageActionRelations import CleavageActionRelations
from clefts.libs.mmkit.mmkit import Compound
from ..mol.graph_builder import MolGraphBuilder

SCHEMA = "clefts.fragment-tree-training"
SCHEMA_VERSION = 6
FRAGMENTATION_SCHEMA = "source-anchored-branching-v1"


class UnresolvedPrecursorError(ValueError):
    """A sample's precursor ion has no valid cleavage action sequence from Source.

    Not every observed precursor is reachable under a given cleavage pattern
    set (e.g. an ion shift the configured patterns cannot produce); such a
    sample cannot be supervised and must be dropped, not treated as a fatal
    error for the rest of its group.
    """


def coo(pairs: Sequence[tuple[int, int]]) -> Tensor:
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()


def csr(rows: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
    counts = torch.tensor([len(row) for row in rows], dtype=torch.long)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))), torch.tensor([v for row in rows for v in row], dtype=torch.long)


@dataclass(frozen=True)
class SourceActionStructure:
    source_graph: Batch
    source_graph_offset: Tensor
    action_tree_index: Tensor
    action_type: Tensor
    action_source_atom_ptr: Tensor
    action_source_atom_index: Tensor
    action_static_features: Tensor
    action_conflict_index: Tensor
    action_precedence_index: Tensor
    action_dominance_index: Tensor
    action_retained_index: Tensor
    sample_tree_index: Tensor
    condition_features: Tensor
    teacher_node_sample_index: Tensor
    teacher_node_action_ptr: Tensor
    teacher_node_action_index: Tensor
    teacher_positive_action_ptr: Tensor
    teacher_positive_action_index: Tensor
    teacher_node_observed: Tensor
    sample_positive_action_ptr: Tensor
    sample_positive_action_index: Tensor
    # Every alternative PrecursorAction.action_sequence for this sample's
    # precursor type, recorded as its own row (never merged/unioned together:
    # alternatives are mutually exclusive ways of reaching the precursor, not
    # actions that can all be applied at once). A sample whose precursor is
    # Source itself has an explicit empty row here.
    sample_precursor_row_ptr: Tensor
    precursor_row_action_ptr: Tensor
    precursor_row_action_index: Tensor
    # Downstream precomputed graphs are optional for scorer-only training.
    source_smiles: tuple[str,...] = field(default=(),kw_only=True)
    precursor_next_index: Tensor = field(default_factory=lambda: torch.empty((2,0),dtype=torch.long),kw_only=True)
    max_action_role_count: int = field(default=0,kw_only=True)
    source_atom_capacity: int = field(default=0,kw_only=True)
    action_source_atom_features: Tensor | None = field(default=None,kw_only=True)
    sample_annotations: tuple[dict, ...] = field(default=(),kw_only=True)
    downstream: object | None = None
    transition_parent_state_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    transition_child_state_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    transition_added_action_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    state_fragment_node_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)

    teacher_node_precursor_row_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    teacher_node_parent_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    teacher_node_added_action_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    teacher_node_ms2_depth: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    teacher_positive_action_weight: Tensor = field(default_factory=lambda: torch.empty(0),kw_only=True)

    @property
    def num_samples(self) -> int:
        return self.sample_tree_index.numel()

    @property
    def device(self) -> torch.device:
        return self.action_type.device

    def to(self, device: str | torch.device) -> SourceActionStructure:
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                               for f in fields(self) if hasattr(getattr(self, f.name), "to")})

    def save(self, path: str | Path) -> None:
        torch.save(dict(schema=SCHEMA, schema_version=SCHEMA_VERSION,
                        fragmentation_schema=FRAGMENTATION_SCHEMA, structure=self.to("cpu")), path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu", *, max_action_count: int = 3) -> SourceActionStructure:
        payload = torch.load(path, map_location="cpu",weights_only=False)
        if not isinstance(payload, dict) or (payload.get("schema"), payload.get("schema_version"), payload.get("fragmentation_schema")) != (SCHEMA, SCHEMA_VERSION, FRAGMENTATION_SCHEMA):
            raise ValueError("Action training requires schema v6; regenerate from original MSDataset")
        result = payload["structure"]
        if not isinstance(result, cls):
            raise TypeError("schema v6 structure must be SourceActionStructure")
        if result.downstream is not None and not hasattr(result.downstream.decoded,"edge_seed_action_index"):
            object.__setattr__(result.downstream.decoded,"edge_seed_action_index",torch.empty((2,0),dtype=torch.long))
        if result.downstream is not None and not result.downstream.node_smiles:
            result=replace(result,downstream=replace(result.downstream,node_smiles=tuple(compound.smiles for compound in result.downstream.decoded.compounds)))
        if not result.source_smiles and result.downstream is not None:
            keys={}
            for sample,tree in enumerate(result.sample_tree_index.tolist()):
                node=int((result.downstream.decoded.node_sample_index==sample).nonzero()[0])
                keys[tree]=result.downstream.node_smiles[node]
            result=replace(result,source_smiles=tuple(keys[i] for i in range(result.source_graph.num_graphs)))
        return result.to(device)

    @classmethod
    def from_structures(cls, structures: Sequence[SourceActionStructure]) -> SourceActionStructure:
        if not structures:
            raise ValueError("Cannot collate an empty batch")
        graphs, action_types, atom_rows, retained, sample_trees, conditions = [], [], [], [], [], []
        relations = {name: [] for name in ("action_conflict_index", "action_precedence_index", "action_dominance_index")}
        state_samples, state_rows, next_rows, eos, positives, static, tree_indices = [], [], [], [], [], [], []
        precursor_rows, sample_precursor_row_counts = [], []
        precursor_next, precursor_offset = [], 0
        atom_offset = action_offset = tree_offset = sample_offset = state_offset = node_offset = 0
        downstream_items, action_offsets, parents, children, added_actions, state_nodes = [], [], [], [], [], []
        annotations=[]
        seed_owners=[]; node_parents=[]; node_added=[]
        for item in structures:
            action_offsets.append(action_offset)
            seed_owners.append(item.teacher_node_precursor_row_index+precursor_offset)
            node_parents.append(torch.where(item.teacher_node_parent_index>=0,item.teacher_node_parent_index+state_offset,item.teacher_node_parent_index))
            node_added.append(torch.where(item.teacher_node_added_action_index>=0,item.teacher_node_added_action_index+action_offset,item.teacher_node_added_action_index))
            parents.append(item.transition_parent_state_index+state_offset)
            children.append(item.transition_child_state_index+state_offset)
            added_actions.append(item.transition_added_action_index+action_offset)
            state_nodes.append(torch.where(item.state_fragment_node_index>=0,item.state_fragment_node_index+node_offset,item.state_fragment_node_index))
            for sample in getattr(item,'sample_annotations',()):
                annotations.append({**sample,'peaks':[{**peak,'matches':[
                    {**match,'nodeIndices':[index+node_offset for index in match['nodeIndices']]}
                    for match in peak['matches']]} for peak in sample['peaks']]})
            state_offset+=item.teacher_node_sample_index.numel()
            if item.downstream is not None:
                downstream_items.append(item.downstream)
                node_offset+=len(item.downstream.decoded.compounds)
            graphs.extend(item.source_graph.to_data_list())
            action_types.append(item.action_type)
            static.append(item.action_static_features)
            tree_indices.append(item.action_tree_index + tree_offset)
            for start, stop in zip(item.action_source_atom_ptr[:-1], item.action_source_atom_ptr[1:]):
                atom_rows.append((item.action_source_atom_index[start:stop] + atom_offset).tolist())
            retained.append(item.action_retained_index + torch.tensor([[action_offset], [atom_offset]], device=item.action_retained_index.device))
            for name in relations:
                relations[name].append(getattr(item, name) + action_offset)
            sample_trees.append(item.sample_tree_index + tree_offset)
            conditions.append(item.condition_features)
            state_samples.append(item.teacher_node_sample_index + sample_offset)
            for ptr, data, dest in ((item.teacher_node_action_ptr, item.teacher_node_action_index, state_rows),
                                    (item.teacher_positive_action_ptr, item.teacher_positive_action_index, next_rows),
                                    (item.sample_positive_action_ptr, item.sample_positive_action_index, positives)):
                for start, stop in zip(ptr[:-1], ptr[1:]):
                    dest.append((data[start:stop] + action_offset).tolist())
            for start, stop in zip(item.precursor_row_action_ptr[:-1], item.precursor_row_action_ptr[1:]):
                precursor_rows.append((item.precursor_row_action_index[start:stop] + action_offset).tolist())
            precursor_next.append(item.precursor_next_index + torch.tensor([[precursor_offset],[action_offset]]))
            precursor_offset += item.precursor_row_action_ptr.numel()-1
            sample_precursor_row_counts.extend((item.sample_precursor_row_ptr[1:] - item.sample_precursor_row_ptr[:-1]).tolist())
            eos.append(item.teacher_node_observed)
            atom_offset += item.source_graph.num_nodes
            action_offset += item.action_type.shape[0]
            tree_offset += item.source_graph.num_graphs
            sample_offset += item.sample_tree_index.numel()
        graph = Batch.from_data_list(graphs)
        atom_ptr, atom_index = csr(atom_rows)
        state_ptr, state_index = csr(state_rows)
        next_ptr, next_index = csr(next_rows)
        positive_ptr, positive_index = csr(positives)
        precursor_row_action_ptr, precursor_row_action_index = csr(precursor_rows)
        sample_precursor_row_ptr = torch.cat((torch.zeros(1, dtype=torch.long),
            torch.tensor(sample_precursor_row_counts, dtype=torch.long).cumsum(0)))
        downstream=None
        if downstream_items:
            if len(downstream_items)!=len(structures):
                raise ValueError("Cannot mix action-only and downstream structures")
            from ..specgen.post_materialization_model import collate_post_materialization
            downstream=collate_post_materialization(downstream_items,action_offsets)
        return cls(graph, graph.ptr, torch.cat(tree_indices), torch.cat(action_types), atom_ptr, atom_index,
                   torch.cat(static), *(torch.cat(relations[name], dim=1) for name in relations),
                   torch.cat(retained, dim=1), torch.cat(sample_trees), torch.cat(conditions),
                   torch.cat(state_samples), state_ptr, state_index, next_ptr, next_index, torch.cat(eos),
                   positive_ptr, positive_index, sample_precursor_row_ptr, precursor_row_action_ptr, precursor_row_action_index,
                   downstream,source_smiles=tuple(key for item in structures for key in item.source_smiles),precursor_next_index=torch.cat(precursor_next,dim=1),transition_parent_state_index=torch.cat(parents),
                   transition_child_state_index=torch.cat(children),transition_added_action_index=torch.cat(added_actions),
                   state_fragment_node_index=torch.cat(state_nodes),
                   teacher_node_precursor_row_index=torch.cat(seed_owners),teacher_node_parent_index=torch.cat(node_parents),
                   teacher_node_added_action_index=torch.cat(node_added),teacher_node_ms2_depth=torch.cat([i.teacher_node_ms2_depth for i in structures]),
                   teacher_positive_action_weight=torch.cat([i.teacher_positive_action_weight for i in structures]),
                   sample_annotations=tuple(annotations) if len(annotations)==sum(item.num_samples for item in structures) else (),
                   max_action_role_count=max(item.max_action_role_count for item in structures),
                   source_atom_capacity=max(item.source_atom_capacity for item in structures),
                   action_source_atom_features=torch.cat([item.action_source_atom_features for item in structures]))


def prepare_source_actions(*, source: Compound, actions: tuple[CleavageAction, ...],
                           graph_builder: MolGraphBuilder, condition_features: Tensor,
                           max_action_count: int, target_sequences: Sequence[Sequence[CleavageActionSequence | None]] | None = None,
                           precursor_sequences: Sequence[Iterable[CleavageActionSequence | None]] | None = None,
                           teacher_pathways=None) -> SourceActionStructure:
    """Pack Source primitives and observed positive pathways as sparse tensors.

    Precursor alternatives remain independent rows. Teacher nodes are keyed by
    (sample, precursor row, normalized state); their parent is representative,
    while transition tensors retain every pathway-supported positive edge.
    Inference passes no pathways and receives no teacher nodes.
    """
    relations = CleavageActionRelations.from_actions(actions)
    maps = {atom.GetAtomMapNum(): index for index, atom in enumerate(graph_builder.graph_atoms(source))}
    atom_ptr, atom_index = csr([tuple(maps[v] for v in a.source_atom_maps if v in maps) for a in actions])
    role_features=torch.tensor([[float(v in a.retained_atom_maps),float(v in a.discarded_atom_maps),
        sum(v in edge for edge in a.matched_bond_maps),sum(v in edge for edge in a.cut_bond_maps),
        sum(v in edge for edge in a.changed_bond_maps),sum(v in (u,w) for u,w,_ in a.bond_updates)]
        for a in actions for v in a.source_atom_maps if v in maps],dtype=torch.float32).reshape(-1,6)
    retained = coo([(i, maps[v]) for i, a in enumerate(actions) for v in sorted(a.retained_atom_maps) if v in maps])
    samples, states, positive_next, eos, sample_positive = [], [], [], [], []
    transition_parents,transition_children,transition_actions=[],[],[]
    index = {a: i for i, a in enumerate(actions)}
    if precursor_sequences is not None and len(precursor_sequences) != condition_features.shape[0]:
        raise ValueError("precursor_sequences must have one row per condition")
    # Recorded independently of target_sequences: inference needs this to seed
    # beam decoding from the precursor state even when there is no teacher DAG.
    alternatives = [tuple(rows) for rows in precursor_sequences] if precursor_sequences is not None else [(None,)] * condition_features.shape[0]
    if any(not rows for rows in alternatives):
        raise UnresolvedPrecursorError("No valid precursor action sequence for a sample; cannot decode from Source")
    precursor_rows = [tuple(sorted(index[a] for a in seq.actions)) if seq is not None else () for rows in alternatives for seq in rows]
    if any(len(row)>max_action_count for row in precursor_rows):raise ValueError("Precursor exceeds Fragmenter max_action_count")
    sample_precursor_row_counts = [len(rows) for rows in alternatives]
    # Chemistry relations are prepared once. Neural filtering applies this sparse
    # table on the device; there is no RDKit or Python chemistry in forward.
    precursor_next = [] # No valid/negative outgoing edges are stored.
    node_seeds=[]; node_parents=[]; node_added=[]; depths=[]; weights=[]
    # Path records contain actual chemistry transitions, never permutations of
    # a terminal action set. Each precursor alternative has independent nodes.
    if target_sequences is not None and teacher_pathways is None:
        raise ValueError("Branching teachers require observed teacher_pathways, not terminal action sets")
    if teacher_pathways is not None:
        if len(teacher_pathways)!=condition_features.shape[0]:
            raise ValueError("teacher_pathways must have one list per condition")
        seed_offset=0
        for sample, paths in enumerate(teacher_pathways):
            positive_actions=set()
            for chain,_ in paths:
                if not any(state==seed for state,_ in chain for seed in alternatives[sample]):
                    raise ValueError("Observed teacher pathway has no resolved precursor seed")
            for seed_local, seed in enumerate(alternatives[sample]):
                known={}; outgoing={}; salience={}
                def add(state, parent=-1, action=-1):
                    if state in known:return known[state]
                    row=len(states);known[state]=row
                    samples.append(sample);states.append(tuple(sorted(index[a] for a in state.actions)) if state else ())
                    positive_next.append([]);eos.append(False)
                    node_seeds.append(seed_offset+seed_local);node_parents.append(parent);node_added.append(action)
                    depths.append(0 if parent<0 else depths[parent]+1)
                    outgoing[row]={};salience[row]=0.
                    return row
                add(seed)
                edges=set()
                for chain, intensity in paths:
                    # chain: (state, added_action) rooted at Source. Only the
                    # suffix starting at this exact precursor belongs to MS2.
                    starts=[i for i,(state,_) in enumerate(chain) if state==seed]
                    for start in starts:
                        parent=add(seed)
                        for child, action in chain[start+1:]:
                            if action is None:continue
                            assert action in index, "All teacher-positive actions must be retained"
                            child_row=add(child,parent,index[action])
                            if child_row==parent:continue
                            edges.add((parent,child_row,index[action]))
                            outgoing[parent].setdefault(index[action],set()).add(child_row)
                            positive_actions.add(index[action]);parent=child_row
                        salience[parent]=max(salience[parent],float(intensity));eos[parent]=True
                # Monotone max propagation also supports a normalized DAG.
                for _ in range(len(known)):
                    changed=False
                    for parent,child,action in sorted(edges,reverse=True):
                        value=max(salience[parent],salience[child])
                        changed |= value!=salience[parent];salience[parent]=value
                    if not changed:break
                for row in known.values():
                    positive_next[row]=sorted(outgoing[row])
                    weights.extend(max(salience[c] for c in outgoing[row][a]) for a in positive_next[row])
                for parent,child,action in sorted(edges):
                    transition_parents.append(parent);transition_children.append(child);transition_actions.append(action)
            sample_positive.append(sorted(positive_actions));seed_offset+=len(alternatives[sample])
    else:
        sample_positive = [() for _ in range(condition_features.shape[0])]
    state_ptr, state_index = csr(states)
    next_ptr, next_index = csr(positive_next)
    positive_ptr, positive_index = csr(sample_positive)
    precursor_row_action_ptr, precursor_row_action_index = csr(precursor_rows)
    sample_precursor_row_ptr = torch.cat((torch.zeros(1, dtype=torch.long),
        torch.tensor(sample_precursor_row_counts, dtype=torch.long).cumsum(0)))
    graph = Batch.from_data_list([graph_builder.build(source)])
    static = torch.tensor([[len(a.matched_bond_maps), len(a.cut_bond_maps), len(a.changed_bond_maps),
                            len(a.bond_updates), len(a.retained_atom_maps), len(a.discarded_atom_maps)] for a in actions], dtype=torch.float32).reshape(-1, 6).log1p()
    return SourceActionStructure(graph, graph.ptr, torch.zeros(len(actions), dtype=torch.long),
        torch.tensor([(a.cleavage_pattern_id, a.reaction_id, a.product_molecule_id) for a in actions], dtype=torch.long).reshape(-1, 3),
        atom_ptr, atom_index, static, coo(relations.conflict_pairs), coo(relations.precedence_pairs),
        coo(relations.dominance_pairs), retained, torch.zeros(condition_features.shape[0], dtype=torch.long), condition_features,
        torch.tensor(samples, dtype=torch.long), state_ptr, state_index, next_ptr, next_index,
        torch.tensor(eos, dtype=torch.bool), positive_ptr, positive_index,
        sample_precursor_row_ptr, precursor_row_action_ptr, precursor_row_action_index,
        teacher_node_precursor_row_index=torch.tensor(node_seeds,dtype=torch.long),
        teacher_node_parent_index=torch.tensor(node_parents,dtype=torch.long),teacher_node_added_action_index=torch.tensor(node_added,dtype=torch.long),
        teacher_node_ms2_depth=torch.tensor(depths,dtype=torch.long),teacher_positive_action_weight=torch.tensor(weights,dtype=torch.float32),
        source_smiles=(source.smiles,),precursor_next_index=coo(precursor_next),transition_parent_state_index=torch.tensor(transition_parents,dtype=torch.long),
        transition_child_state_index=torch.tensor(transition_children,dtype=torch.long),
        transition_added_action_index=torch.tensor(transition_actions,dtype=torch.long),
        state_fragment_node_index=torch.full((len(states),),-1,dtype=torch.long),
        source_atom_capacity=len(maps),action_source_atom_features=role_features,
        max_action_role_count=max((len(a.source_atom_maps) for a in actions),default=0))


def make_structure_file_stem(smiles: str, *, index: int) -> str:
    """Make a stable readable file stem for one SMILES group."""
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    return f"smiles_{index:06d}_{digest}"


def save_fragment_tree_structure(*, structure: SourceActionStructure, output_file: str | Path,
                                 metadata: dict[str, object] | None = None) -> None:
    """Save one structure and a small metadata sidecar into a torch file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, schema_version=SCHEMA_VERSION, fragmentation_schema=FRAGMENTATION_SCHEMA,
                    structure=structure.to("cpu"), metadata=dict(metadata or {})), output_path)
