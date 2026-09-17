"""Schema v4 Source/action tensors. RDKit is confined to preparation."""
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
from clefts.domain.fragment.cleavage.CleavageActionSearch import CleavageActionSearch
from clefts.libs.mmkit.mmkit import Compound
from ..mol.graph_builder import MolGraphBuilder

SCHEMA = "clefts.fragment-tree-training"
SCHEMA_VERSION = 5
FRAGMENTATION_SCHEMA = "source-anchored-action-autoregressive-v1"


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
    teacher_state_sample_index: Tensor
    teacher_state_action_ptr: Tensor
    teacher_state_action_index: Tensor
    teacher_positive_action_ptr: Tensor
    teacher_positive_action_index: Tensor
    teacher_positive_eos: Tensor
    sample_positive_action_ptr: Tensor
    sample_positive_action_index: Tensor
    # Every alternative PrecursorAction.action_sequence for this sample's
    # precursor type, recorded as its own row (never merged/unioned together:
    # alternatives are mutually exclusive ways of reaching the precursor, not
    # actions that can all be applied at once). A sample whose precursor is
    # Source itself has zero rows here.
    sample_precursor_row_ptr: Tensor
    precursor_row_action_ptr: Tensor
    precursor_row_action_index: Tensor
    # Downstream precomputed graphs are optional for scorer-only training.
    max_action_role_count: int = field(default=0,kw_only=True)
    source_atom_capacity: int = field(default=0,kw_only=True)
    action_source_atom_features: Tensor | None = field(default=None,kw_only=True)
    sample_annotations: tuple[dict, ...] = field(default=(),kw_only=True)
    downstream: object | None = None
    transition_parent_state_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    transition_child_state_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    transition_added_action_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)
    state_fragment_node_index: Tensor = field(default_factory=lambda: torch.empty(0,dtype=torch.long),kw_only=True)

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
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> SourceActionStructure:
        payload = torch.load(path, map_location=device)
        if not isinstance(payload, dict) or (payload.get("schema"), payload.get("schema_version"), payload.get("fragmentation_schema")) != (SCHEMA, SCHEMA_VERSION, FRAGMENTATION_SCHEMA):
            raise ValueError("Action training requires schema v4; regenerate from original MSDataset")
        result = payload["structure"]
        if not isinstance(result, cls):
            raise TypeError("schema v4 structure must be SourceActionStructure")
        return result

    @classmethod
    def from_structures(cls, structures: Sequence[SourceActionStructure]) -> SourceActionStructure:
        if not structures:
            raise ValueError("Cannot collate an empty batch")
        graphs, action_types, atom_rows, retained, sample_trees, conditions = [], [], [], [], [], []
        relations = {name: [] for name in ("action_conflict_index", "action_precedence_index", "action_dominance_index")}
        state_samples, state_rows, next_rows, eos, positives, static, tree_indices = [], [], [], [], [], [], []
        precursor_rows, sample_precursor_row_counts = [], []
        atom_offset = action_offset = tree_offset = sample_offset = state_offset = node_offset = 0
        downstream_items, action_offsets, parents, children, added_actions, state_nodes = [], [], [], [], [], []
        annotations=[]
        for item in structures:
            action_offsets.append(action_offset)
            parents.append(item.transition_parent_state_index+state_offset)
            children.append(item.transition_child_state_index+state_offset)
            added_actions.append(item.transition_added_action_index+action_offset)
            state_nodes.append(torch.where(item.state_fragment_node_index>=0,item.state_fragment_node_index+node_offset,item.state_fragment_node_index))
            for sample in getattr(item,'sample_annotations',()):
                annotations.append({**sample,'peaks':[{**peak,'matches':[
                    {**match,'nodeIndices':[index+node_offset for index in match['nodeIndices']]}
                    for match in peak['matches']]} for peak in sample['peaks']]})
            state_offset+=item.teacher_state_sample_index.numel()
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
            state_samples.append(item.teacher_state_sample_index + sample_offset)
            for ptr, data, dest in ((item.teacher_state_action_ptr, item.teacher_state_action_index, state_rows),
                                    (item.teacher_positive_action_ptr, item.teacher_positive_action_index, next_rows),
                                    (item.sample_positive_action_ptr, item.sample_positive_action_index, positives)):
                for start, stop in zip(ptr[:-1], ptr[1:]):
                    dest.append((data[start:stop] + action_offset).tolist())
            for start, stop in zip(item.precursor_row_action_ptr[:-1], item.precursor_row_action_ptr[1:]):
                precursor_rows.append((item.precursor_row_action_index[start:stop] + action_offset).tolist())
            sample_precursor_row_counts.extend((item.sample_precursor_row_ptr[1:] - item.sample_precursor_row_ptr[:-1]).tolist())
            eos.append(item.teacher_positive_eos)
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
                   downstream,transition_parent_state_index=torch.cat(parents),
                   transition_child_state_index=torch.cat(children),transition_added_action_index=torch.cat(added_actions),
                   state_fragment_node_index=torch.cat(state_nodes),
                   sample_annotations=tuple(annotations) if len(annotations)==sum(item.num_samples for item in structures) else (),
                   max_action_role_count=max(item.max_action_role_count for item in structures),
                   source_atom_capacity=max(item.source_atom_capacity for item in structures),
                   action_source_atom_features=torch.cat([item.action_source_atom_features for item in structures]))


def prepare_source_actions(*, source: Compound, actions: tuple[CleavageAction, ...],
                           graph_builder: MolGraphBuilder, condition_features: Tensor,
                           max_action_count: int, target_sequences: Sequence[Sequence[CleavageActionSequence | None]] | None = None,
                           precursor_sequences: Sequence[Iterable[CleavageActionSequence | None]] | None = None) -> SourceActionStructure:
    """Inference prepares primitives only; teacher enumeration is preparation-only.

    precursor_sequences records, per sample, every PrecursorAction.action_sequence
    alternative for that sample's precursor type (Fragmenter.resolve_precursor_actions,
    computed once against the same exhaustively built fragment tree). Real MS2
    fragmentation always happens on the selected precursor ion, never on the
    bare Source, so every teacher state other than the empty <BOS> prefix must
    itself be a subset or superset of one of these alternatives.
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
    precursor_rows = [tuple(sorted(index[a] for a in seq.actions)) for sample_alternatives in
                      (precursor_sequences or ()) for seq in sample_alternatives if seq is not None]
    sample_precursor_row_counts = [sum(1 for seq in sample_alternatives if seq is not None)
                                   for sample_alternatives in precursor_sequences] if precursor_sequences is not None \
                                   else [0] * condition_features.shape[0]
    if target_sequences is not None:
        if len(target_sequences) != condition_features.shape[0]:
            raise ValueError("target_sequences must have one row per condition")
        search = CleavageActionSearch(actions, max_action_count=max_action_count)
        candidates = search.enumerate()
        known = {None, *(c.action_sequence for c in candidates)}
        transitions: set[tuple[CleavageActionSequence | None, CleavageActionSequence, CleavageAction]] = set()
        # Enumerate every valid outgoing transition, including replacement, not
        # just one arbitrary permutation of a target set.
        for parent in known:
            if parent is not None and len(parent.actions) >= max_action_count:
                continue
            for action in actions:
                if parent is not None and action in parent.actions:
                    continue
                raw = (*(parent.actions if parent else ()), action)
                if relations.rejection(tuple(index[a] for a in raw)):
                    continue
                child = CleavageActionSequence(raw)
                if child != parent and child.retained_atom_maps and len(child.actions) <= max_action_count and child in known:
                    transitions.add((parent, child, action))
        for sample, targets in enumerate(target_sequences):
            terminal = set(targets)
            if not terminal <= known:
                raise ValueError("Teacher target is not a valid searchable Source action state")
            required = frozenset(precursor_sequences[sample]) if precursor_sequences is not None else frozenset((None,))
            if not required <= known:
                raise ValueError("Precursor action sequence is not a valid searchable Source action state")
            root_is_precursor = None in required
            required_sets = frozenset(frozenset(index[a] for a in req.actions) for req in required if req is not None)
            if root_is_precursor:
                # Source itself is a valid precursor selection: the empty
                # action set trivially satisfies (is a subset of) every state.
                required_sets = required_sets | frozenset({frozenset()})

            def contains_a_precursor(state: CleavageActionSequence | None) -> bool:
                """state already fully applied one recorded precursor action set."""
                if state is None:
                    return root_is_precursor
                actions_set = frozenset(index[a] for a in state.actions)
                return any(required_set <= actions_set for required_set in required_sets)

            if not all(contains_a_precursor(target) for target in terminal):
                raise ValueError("Teacher target does not contain a recorded precursor action set")
            ancestors = set(terminal)
            while True:
                updated = ancestors | {parent for parent, child, _ in transitions if child in ancestors}
                if updated == ancestors:
                    break
                ancestors = updated
            if not root_is_precursor:
                # Real MS2 fragmentation happens on the selected precursor ion:
                # keep only <BOS> (the DAG's entry point) and states already on
                # a path toward, through, or beyond a recorded precursor set.
                def on_precursor_path(state: CleavageActionSequence | None) -> bool:
                    if state is None:
                        return True
                    actions_set = frozenset(index[a] for a in state.actions)
                    return contains_a_precursor(state) or any(actions_set <= required_set for required_set in required_sets)
                ancestors = {state for state in ancestors if on_precursor_path(state)}
            ordered = sorted(ancestors, key=lambda seq: (len(seq.actions), seq.key) if seq else (0, ()))
            row_by_state={state:len(states)+i for i,state in enumerate(ordered)}
            for parent,child,action in sorted(transitions,key=lambda t: ((t[0].key if t[0] else ()),t[1].key,t[2].key)):
                if parent in ancestors and child in ancestors:
                    transition_parents.append(row_by_state[parent]);transition_children.append(row_by_state[child]);transition_actions.append(index[action])
            positive_actions = set()
            for state in ordered:
                next_actions = sorted({index[action] for parent, child, action in transitions if parent == state and child in ancestors})
                positive_actions.update(next_actions)
                if state:
                    positive_actions.update(index[a] for a in state.actions)
                samples.append(sample)
                states.append(tuple(sorted(index[a] for a in state.actions)) if state else ())
                positive_next.append(next_actions)
                eos.append(state in terminal)
            sample_positive.append(sorted(positive_actions))
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
        transition_parent_state_index=torch.tensor(transition_parents,dtype=torch.long),
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
