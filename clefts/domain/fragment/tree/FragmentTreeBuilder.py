from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any

from rdkit import Chem
from ....libs.mmkit.mmkit import Compound
from ..cleavage._CleavagePattern import _CleavagePattern
from ..cleavage.CleavagePatternSet import CleavagePatternSet
from ..cleavage.CleavageAction import CleavageAction, mapped_source
from ..cleavage.CleavageActionGenerator import create_cleavage_actions
from ..cleavage.CleavageActionSequence import CleavageActionSequence
from ..cleavage.CleavageActionResult import CleavageActionResult
from ..cleavage.CleavageActionSearch import CleavageActionSearch, _ActionSequenceCandidate
from .CleavageActionTransition import CleavageActionTransition
from .FragmentEdge import FragmentEdge
from .FragmentNode import FragmentNode
from .FragmentTree import FragmentTree


@dataclass(frozen=True)
class _FragmentExpansionState:
    node_index: int
    action_sequence: CleavageActionSequence | None


@dataclass(frozen=True)
class FragmentTreeBuilder:
    """Build an in-memory tree from combinations anchored to Original Source.

    Primitive matching happens once per pattern on Source. Edges record action
    state transitions; generated fragments are never used as reaction inputs.
    Database serialization of these transitions belongs to a later stage.
    """
    max_action_count: int
    cleavage_pattern_set: CleavagePatternSet
    only_add_min_action_count: bool = True

    def __post_init__(self) -> None:
        if type(self.max_action_count) is not int or self.max_action_count <= 0:
            raise ValueError("max_action_count must be a positive integer.")
        if not isinstance(self.cleavage_pattern_set, CleavagePatternSet):
            raise TypeError("cleavage_pattern_set must be a CleavagePatternSet.")
        if not isinstance(self.only_add_min_action_count, bool):
            raise TypeError("only_add_min_action_count must be a bool.")

    @property
    def cleavage_patterns(self) -> tuple[_CleavagePattern, ...]:
        return self.cleavage_pattern_set.patterns

    @property
    def name(self) -> str:
        return self.cleavage_pattern_set.name

    def create_cleavage_actions(self, source_compound: Compound) -> tuple[CleavageAction, ...]:
        if not isinstance(source_compound, Compound):
            raise TypeError("source_compound must be a Compound.")
        return create_cleavage_actions(source_compound, self.cleavage_pattern_set)

    def _action_limit(self, max_action_count: int | None) -> int:
        if max_action_count is None:
            return self.max_action_count
        if type(max_action_count) is not int or max_action_count < 0:
            raise ValueError("max_action_count must be None or a non-negative integer.")
        return max_action_count

    def _search(
        self,
        source_compound: Compound,
        *,
        seed_action_sequences: Sequence[CleavageActionSequence] | None,
        max_action_count: int | None,
    ) -> tuple[CleavageActionSearch, tuple[_ActionSequenceCandidate, ...]]:
        if not isinstance(source_compound, Compound):
            raise TypeError("source_compound must be a Compound.")
        limit = self._action_limit(max_action_count)
        seeds = tuple(seed_action_sequences or ())
        if any(not isinstance(seed, CleavageActionSequence) for seed in seeds):
            raise TypeError("seed_action_sequences must contain CleavageActionSequence instances")
        if any(len(seed.actions) > limit for seed in seeds):
            raise ValueError("Seed exceeds max_action_count")
        universe = frozenset(a.GetAtomMapNum() for a in mapped_source(source_compound).GetAtoms())
        for seed in seeds:
            if any(a.retained_atom_maps | a.discarded_atom_maps != universe for a in seed.actions):
                raise ValueError("Seed action universe differs from Original Source")
        actions = self.create_cleavage_actions(source_compound)
        search = CleavageActionSearch(actions, max_action_count=limit,
                                      seed_action_sequences=seeds)
        return search, search.enumerate()

    def _materialize(
        self,
        source_compound: Compound,
        search: CleavageActionSearch,
        candidates: tuple[_ActionSequenceCandidate, ...],
    ) -> dict[CleavageActionSequence, CleavageActionResult]:
        """Compile/run each unique effect once, retaining every action history."""
        results: dict[CleavageActionSequence, CleavageActionResult] = {}
        effect_cache: dict[tuple[object, ...], tuple[Compound, str] | None] = {}
        visited_effect_keys: set[tuple[object, ...]] = set()
        for candidate in candidates:
            sequence = candidate.action_sequence
            if sequence in results:
                continue
            effect = sequence.effect_key
            if effect in visited_effect_keys:
                search.stats.num_duplicate_effect_pruned += 1
                cached = effect_cache[effect]
            else:
                visited_effect_keys.add(effect)
                try:
                    reaction = sequence.compile(source_compound)
                    search.stats.num_compiled_sequences += 1
                    search.stats.num_rdkit_run_reactants += 1
                    products = reaction.run(source_compound)
                    if len(products) != 1:
                        raise ValueError("Source-specific reaction must generate exactly one target")
                    cached = (Compound(products[0]), reaction.smirks)
                    search.stats.num_generated_fragments += 1
                except Chem.rdchem.MolSanitizeException:
                    # A graph-valid collection can still violate chemical valence.
                    if candidate.is_seed:
                        raise
                    cached = None
                effect_cache[effect] = cached
            if cached is not None:
                target, smirks = cached
                results[sequence] = CleavageActionResult(sequence, target, smirks)
        return results

    def cleave_all(
        self,
        source_compound: Compound,
        *,
        seed_action_sequences: Sequence[CleavageActionSequence] | None = None,
        max_action_count: int | None = None,
    ) -> tuple[CleavageActionResult, ...]:
        """Enumerate unordered histories, then react once per unique Source effect."""
        search, candidates = self._search(source_compound,
            seed_action_sequences=seed_action_sequences, max_action_count=max_action_count)
        results = self._materialize(source_compound, search, candidates)
        return tuple(sorted(results.values(), key=lambda r: (len(r.action_sequence.actions),
                                                            r.action_sequence.key)))

    def build(
        self,
        compound: Compound,
        *,
        seed_action_sequences: Sequence[CleavageActionSequence] | None = None,
        max_node: int = -1,
        max_edge: int = -1,
        max_action_count: int | None = None,
        print_info: bool = False,
    ) -> FragmentTree:
        return self._build_result(compound, seed_action_sequences=seed_action_sequences,
            max_node=max_node, max_edge=max_edge, max_action_count=max_action_count,
            print_info=print_info)["fragment_tree"]

    def _build_result(
        self,
        compound: Compound,
        *,
        seed_action_sequences: Sequence[CleavageActionSequence] | None = None,
        max_node: int = -1,
        max_edge: int = -1,
        max_action_count: int | None = None,
        print_info: bool = False,
    ) -> dict[str, Any]:
        if type(max_node) is not int or not (max_node == -1 or max_node > 0):
            raise ValueError("max_node must be -1 or a positive integer.")
        if type(max_edge) is not int or max_edge < -1:
            raise ValueError("max_edge must be -1 or a non-negative integer.")
        started = time.monotonic()
        source_compound = compound
        search, candidates = self._search(source_compound,
            seed_action_sequences=seed_action_sequences, max_action_count=max_action_count)
        results = self._materialize(source_compound, search, candidates)
        state = _FragmentTreeBuildState(source_compound.smiles, max_node=max_node, max_edge=max_edge,
                                        only_add_min_action_count=self.only_add_min_action_count)
        compounds = {0: source_compound.copy()}
        expansion_by_sequence: dict[CleavageActionSequence | None, _FragmentExpansionState] = {
            None: _FragmentExpansionState(0, None)}
        processed_states: set[tuple[int, tuple[tuple[object, ...], ...] | None]] = set()
        for candidate in candidates:
            sequence = candidate.action_sequence
            if sequence not in results:
                continue
            parent_sequence = candidate.parent_action_sequence
            added_action = candidate.added_action
            parent = expansion_by_sequence.get(parent_sequence)
            if parent is None:
                # Canonical combination enumeration is not a chemical execution
                # order. A primitive target can fail valence while its combined
                # target is valid. Link to an already generated alternative
                # predecessor without re-enumerating or executing the effect.
                for added in sequence.actions:
                    remaining = tuple(a for a in sequence.actions if a != added)
                    predecessor = CleavageActionSequence(remaining) if remaining else None
                    alternate = expansion_by_sequence.get(predecessor)
                    if alternate is not None and (predecessor is not None or not seed_action_sequences):
                        normalized = CleavageActionSequence((*(predecessor.actions if predecessor else ()), added))
                        if normalized == sequence:
                            parent = alternate
                            parent_sequence = predecessor
                            added_action = added
                            break
                if parent is None:
                    continue
            result = results[sequence]
            target_index = state.get_or_create_node_index(result.compound.smiles, len(sequence.actions))
            child = _FragmentExpansionState(target_index, sequence)
            expansion_by_sequence[sequence] = child
            compounds.setdefault(target_index, result.compound)
            processed_states.add((target_index, sequence.key))
            transition = CleavageActionTransition(added_action, sequence,
                                                   parent_sequence, candidate.is_seed)
            state.add_fragment_edge(parent.node_index, target_index, transition)
        # Combination enumeration visits each unordered collection once. Tree
        # presentation may retain several valid one-action predecessors for the
        # same already computed history, without executing its effect again.
        for sequence, child in expansion_by_sequence.items():
            if sequence is None:
                continue
            for added in sequence.actions:
                remaining = tuple(a for a in sequence.actions if a != added)
                predecessor = CleavageActionSequence(remaining) if remaining else None
                if predecessor is None and seed_action_sequences:
                    continue
                parent = expansion_by_sequence.get(predecessor)
                if parent is None:
                    continue
                if CleavageActionSequence((*(predecessor.actions if predecessor else ()), added)) != sequence:
                    continue
                state.add_fragment_edge(parent.node_index, child.node_index,
                    CleavageActionTransition(added, sequence, predecessor))
        if not seed_action_sequences:
            processed_states.add((0, None))
        stats = search.stats.to_dict()
        if print_info:
            print(f"Source action search: {stats}; nodes={len(state.nodes)}, "
                  f"edges={len(state.edges)}, elapsed={time.monotonic() - started:.2f}s")
        return {"fragment_tree": state.to_fragment_tree(),
                "fragment_compound_by_index": compounds,
                "expansion_states": tuple(expansion_by_sequence.values()),
                "processed_expansion_states": frozenset(processed_states),
                "search_stats": stats}

    def to_dict(self) -> dict[str, Any]:
        return {"max_action_count": self.max_action_count,
                "cleavage_pattern_set": self.cleavage_pattern_set.to_dict(),
                "only_add_min_action_count": self.only_add_min_action_count}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FragmentTreeBuilder:
        return cls(max_action_count=data["max_action_count"],
                   cleavage_pattern_set=CleavagePatternSet.from_dict(data["cleavage_pattern_set"]),
                   only_add_min_action_count=data.get("only_add_min_action_count", True))

    def copy(self) -> FragmentTreeBuilder:
        return FragmentTreeBuilder(self.max_action_count, self.cleavage_pattern_set.copy(),
                                   self.only_add_min_action_count)


class _FragmentTreeBuildState:
    """Chemical nodes and in-memory edges, independent of expansion histories."""

    def __init__(
        self,
        root_smiles: str,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        only_add_min_action_count: bool = True,
    ) -> None:
        self.root_smiles = root_smiles
        self.max_node = max_node
        self.max_edge = max_edge
        self.only_add_min_action_count = only_add_min_action_count
        self.nodes: dict[int, FragmentNode] = {}
        self.edges: dict[tuple[int, int], FragmentEdge] = {}
        self.smiles_to_node_index: dict[str, int] = {}
        self.node_action_counts: dict[int, int] = {}
        self.get_or_create_node_index(root_smiles, 0)

    def get_or_create_node_index(self, smiles: str, action_count: int) -> int:
        if smiles in self.smiles_to_node_index:
            index = self.smiles_to_node_index[smiles]
            self.node_action_counts[index] = min(self.node_action_counts[index], action_count)
            return index
        if self.max_node >= 0 and len(self.nodes) >= self.max_node:
            raise ValueError(f"Fragment tree node limit exceeded: max_node={self.max_node}")
        index = len(self.nodes)
        self.nodes[index] = FragmentNode(index, -1, smiles)
        self.smiles_to_node_index[smiles] = index
        self.node_action_counts[index] = action_count
        return index

    def add_fragment_edge(
        self,
        source_index: int,
        target_index: int,
        transition: CleavageActionTransition,
    ) -> None:
        key = (source_index, target_index)
        # Minimum action count filters edge presentation only. Every distinct
        # action history remains eligible for expansion, even for a merged node.
        if (self.only_add_min_action_count and not transition.is_seed
                and len(transition.action_sequence.actions) > self.node_action_counts[target_index]):
            return
        if key in self.edges:
            self.edges[key] = self.edges[key].with_transition(transition)
            return
        if self.max_edge >= 0 and len(self.edges) >= self.max_edge:
            raise ValueError(f"Fragment tree edge limit exceeded: max_edge={self.max_edge}")
        self.edges[key] = FragmentEdge(len(self.edges), -1, source_index, target_index,
                                       self.nodes[source_index].id, self.nodes[target_index].id,
                                       transitions=(transition,))

    def to_fragment_tree(self) -> FragmentTree:
        return FragmentTree.from_nodes_and_edges(smiles=self.root_smiles,
            nodes=tuple(self.nodes.values()), edges=tuple(self.edges.values()))
