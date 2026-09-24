from __future__ import annotations

from collections.abc import Sequence, Iterable
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
from ..cleavage.CleavageActionSearch import (CleavageActionSearch, FragmentTreeLimitExceeded,
                                              _ActionSequenceCandidate, validate_search_limits)
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
        max_cleavage_combinations: int = -1,
    ) -> tuple[CleavageActionSearch, Iterable[_ActionSequenceCandidate]]:
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
                                      seed_action_sequences=seeds,
                                      max_cleavage_combinations=max_cleavage_combinations)
        return search, search.iter_candidates()

    def _materialize(self, source_compound, search, candidates):
        return {candidate.action_sequence: result for candidate, result in
                self._materialize_iter(source_compound, search, candidates)}

    def _materialize_iter(
        self,
        source_compound: Compound,
        search: CleavageActionSearch,
        candidates: Iterable[_ActionSequenceCandidate],
        *,
        max_unique_fragment_smiles: int = -1,
    ):
        """Compile/run each unique effect once, retaining every action history.

        Every newly generated fragment's canonical SMILES joins a set that
        excludes the Source SMILES. Once that set grows beyond
        max_unique_fragment_smiles the source is abandoned immediately, before
        the next candidate is requested from the lazy combination search.
        """
        results: dict[CleavageActionSequence, CleavageActionResult] = {}
        unique_fragment_smiles: set[str] = set()
        source_smiles = source_compound.smiles
        effect_cache: dict[tuple[object, ...], tuple[Compound, str] | None] = {}
        visited_effect_keys: set[tuple[object, ...]] = set()
        for candidate in candidates:
            sequence = candidate.action_sequence
            if sequence in results:
                yield candidate, results[sequence]
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
                    smiles = cached[0].smiles
                    if smiles != source_smiles and smiles not in unique_fragment_smiles:
                        unique_fragment_smiles.add(smiles)
                        search.stats.num_unique_fragment_smiles = len(unique_fragment_smiles)
                        if 0 <= max_unique_fragment_smiles < len(unique_fragment_smiles):
                            raise FragmentTreeLimitExceeded("max_unique_fragment_smiles", max_unique_fragment_smiles)
                except Chem.rdchem.MolSanitizeException:
                    # A graph-valid collection can still violate chemical valence.
                    if candidate.is_seed:
                        raise
                    cached = None
                effect_cache[effect] = cached
            if cached is not None:
                target, smirks = cached
                results[sequence] = CleavageActionResult(sequence, target, smirks)
                yield candidate, results[sequence]

    def cleave_all(
        self,
        source_compound: Compound,
        *,
        seed_action_sequences: Sequence[CleavageActionSequence] | None = None,
        max_action_count: int | None = None,
    ) -> tuple[CleavageActionResult, ...]:
        """Iterate normalized unordered histories and react once per unique Source effect."""
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
        max_unique_fragment_smiles: int = -1,
        max_cleavage_combinations: int = -1,
        max_action_count: int | None = None,
        print_info: bool = False,
    ) -> FragmentTree:
        return self._build_result(compound, seed_action_sequences=seed_action_sequences,
            max_unique_fragment_smiles=max_unique_fragment_smiles,
            max_cleavage_combinations=max_cleavage_combinations, max_action_count=max_action_count,
            print_info=print_info)["fragment_tree"]

    def _build_result(
        self,
        compound: Compound,
        *,
        seed_action_sequences: Sequence[CleavageActionSequence] | None = None,
        max_unique_fragment_smiles: int = -1,
        max_cleavage_combinations: int = -1,
        max_action_count: int | None = None,
        print_info: bool = False,
    ) -> dict[str, Any]:
        """Build one source tree; -1 disables either search limit.

        max_cleavage_combinations bounds search.stats.num_raw_combinations and
        is enforced inside CleavageActionSearch before RDKit runs.
        max_unique_fragment_smiles bounds distinct generated fragment SMILES,
        excluding the Source itself, and is enforced as each fragment is
        materialized. A FragmentTreeLimitExceeded carries the statistics
        observed up to that point in ``stats``.
        """
        validate_search_limits(max_unique_fragment_smiles, max_cleavage_combinations)
        started = time.monotonic()
        source_compound = compound
        search, candidates = self._search(source_compound,
            seed_action_sequences=seed_action_sequences, max_action_count=max_action_count,
            max_cleavage_combinations=max_cleavage_combinations)
        try:
            return self._build_from_candidates(source_compound, search, candidates,
                seed_action_sequences=seed_action_sequences,
                max_unique_fragment_smiles=max_unique_fragment_smiles,
                print_info=print_info, started=started)
        except FragmentTreeLimitExceeded as error:
            error.stats = search.stats.to_dict()
            raise

    def _build_from_candidates(self, source_compound, search, candidates, *, seed_action_sequences,
                               max_unique_fragment_smiles, print_info, started) -> dict[str, Any]:
        state = _FragmentTreeBuildState(source_compound.smiles,
                                        only_add_min_action_count=self.only_add_min_action_count)
        compounds = {0: source_compound.copy()}
        expansion_by_sequence: dict[CleavageActionSequence | None, _FragmentExpansionState] = {
            None: _FragmentExpansionState(0, None)}
        processed_states: set[tuple[int, tuple[tuple[object, ...], ...] | None]] = set()
        pending_predecessors = {}
        for candidate, result in self._materialize_iter(source_compound, search, candidates,
                max_unique_fragment_smiles=max_unique_fragment_smiles):
            sequence = candidate.action_sequence
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
            target_index = state.get_or_create_node_index(result.compound.smiles, len(sequence.actions))
            child = _FragmentExpansionState(target_index, sequence)
            expansion_by_sequence[sequence] = child
            compounds.setdefault(target_index, result.compound)
            processed_states.add((target_index, sequence.key))
            transition = CleavageActionTransition(added_action, sequence,
                                                   parent_sequence, candidate.is_seed)
            state.add_fragment_edge(parent.node_index, target_index, transition)
            # Add alternative routes immediately, including routes whose predecessor
            # arrives later. Count distinct transitions rather than node pairs.
            for target_sequence, added in pending_predecessors.pop(sequence, []):
                destination = expansion_by_sequence[target_sequence]
                state.add_fragment_edge(child.node_index, destination.node_index,
                    CleavageActionTransition(added, target_sequence, sequence))
            for added in sequence.actions:
                remaining = tuple(a for a in sequence.actions if a != added)
                predecessor = CleavageActionSequence(remaining) if remaining else None
                if predecessor is None and seed_action_sequences:
                    continue
                if CleavageActionSequence((*(predecessor.actions if predecessor else ()), added)) != sequence:
                    continue
                alternate = expansion_by_sequence.get(predecessor)
                if alternate is None:
                    pending_predecessors.setdefault(predecessor, []).append((sequence, added))
                else:
                    state.add_fragment_edge(alternate.node_index, child.node_index,
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
        only_add_min_action_count: bool = True,
    ) -> None:
        self.root_smiles = root_smiles
        self.only_add_min_action_count = only_add_min_action_count
        self.nodes: dict[int, FragmentNode] = {}
        self.edges: dict[tuple[int, int], FragmentEdge] = {}
        self.transition_count = 0
        self.smiles_to_node_index: dict[str, int] = {}
        self.node_action_counts: dict[int, int] = {}
        self.get_or_create_node_index(root_smiles, 0)

    def get_or_create_node_index(self, smiles: str, action_count: int) -> int:
        if smiles in self.smiles_to_node_index:
            index = self.smiles_to_node_index[smiles]
            self.node_action_counts[index] = min(self.node_action_counts[index], action_count)
            return index
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
        if key in self.edges and transition in self.edges[key].transitions:
            return
        self.transition_count += 1
        if key in self.edges:
            self.edges[key] = self.edges[key].with_transition(transition)
            return
        self.edges[key] = FragmentEdge(len(self.edges), -1, source_index, target_index,
                                       self.nodes[source_index].id, self.nodes[target_index].id,
                                       transitions=(transition,))

    def to_fragment_tree(self) -> FragmentTree:
        return FragmentTree.from_nodes_and_edges(smiles=self.root_smiles,
            nodes=tuple(self.nodes.values()), edges=tuple(self.edges.values()))
