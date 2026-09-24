"""Pure Python combination search; RDKit is used only after normalization."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import heapq

from .CleavageAction import CleavageAction
from .CleavageActionSequence import CleavageActionSequence


@dataclass
class CleavageActionSearchStats:
    num_primitive_actions: int = 0
    num_raw_combinations: int = 0
    num_hard_conflict_pruned: int = 0
    num_invalidated_action_pruned: int = 0
    num_redundancy_pruned: int = 0
    num_duplicate_sequence_pruned: int = 0
    num_duplicate_effect_pruned: int = 0
    num_compiled_sequences: int = 0
    num_rdkit_run_reactants: int = 0
    num_generated_fragments: int = 0
    num_unique_fragment_smiles: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(vars(self))


class FragmentTreeLimitExceeded(ValueError):
    """A single source tree exceeds a configured search limit.

    ``stats`` holds the search statistics observed up to the moment the limit
    was hit (the builder attaches them), so a skipped source stays diagnosable.
    """

    def __init__(self, limit_name: str, limit: int, stats: dict[str, int] | None = None) -> None:
        self.limit_name = limit_name
        self.limit = limit
        self.stats = stats
        super().__init__(f"{limit_name} exceeded: limit={limit}, observed>{limit}")

    def __reduce__(self):
        return type(self), (self.limit_name, self.limit, self.stats)


def validate_search_limits(max_unique_fragment_smiles: int = -1, max_cleavage_combinations: int = -1) -> None:
    """Both limits are -1 (unlimited) or a positive integer."""
    for name, value in (("max_unique_fragment_smiles", max_unique_fragment_smiles),
                        ("max_cleavage_combinations", max_cleavage_combinations)):
        if type(value) is not int or not (value == -1 or value > 0):
            raise ValueError(f"{name} must be -1 or a positive integer.")


# Compatibility for legacy imports; chemistry rules live in one public utility.
from .CleavageActionRelations import CleavageActionRelations as _ActionRelations


@dataclass(frozen=True)
class _ActionSequenceCandidate:
    parent_action_sequence: CleavageActionSequence | None
    action_sequence: CleavageActionSequence
    added_action: CleavageAction | None
    is_seed: bool = False


class CleavageActionSearch:
    """Explore unordered collections using increasing primitive indices.

    A cursor is separate from the normalized history, since normalization may
    remove an action. Different cursors can expose different remaining actions;
    histories are deduplicated independently from these enumeration positions.
    For a repeated history only its smallest (most permissive) cursor expands.
    Seed actions are fixed anchors; added indices increase from -1, allowing
    additions whose index is lower than an action already in the seed.
    """

    def __init__(
        self,
        actions: tuple[CleavageAction, ...],
        *,
        max_action_count: int,
        seed_action_sequences: Sequence[CleavageActionSequence] | None = None,
        max_cleavage_combinations: int = -1,
    ) -> None:
        validate_search_limits(max_cleavage_combinations=max_cleavage_combinations)
        seeds = tuple(sorted(set(seed_action_sequences or ()), key=lambda s: s.key))
        if any(len(seed.actions) > max_action_count for seed in seeds):
            raise ValueError("Seed exceeds max_action_count")
        self.actions = tuple(sorted(set(actions) | {a for seed in seeds for a in seed.actions},
                                    key=lambda a: a.key))
        self.index_by_action = {a: i for i, a in enumerate(self.actions)}
        self.relations = _ActionRelations.from_actions(self.actions)
        self.max_action_count = max_action_count
        self.seeds = seeds
        self.max_cleavage_combinations = max_cleavage_combinations
        self.stats = CleavageActionSearchStats(num_primitive_actions=len(actions))
        self.visited_sequence_keys: set[tuple[tuple[object, ...], ...]] = set()

    def enumerate(self) -> tuple[_ActionSequenceCandidate, ...]:
        return tuple(self.iter_candidates())

    def iter_candidates(self):
        """Yield each connection lazily so tree limits can stop combination search.

        Raises FragmentTreeLimitExceeded as soon as num_raw_combinations
        exceeds max_cleavage_combinations. The count spans every size, from
        one action up to max_action_count, and is checked before a candidate
        is yielded, so no further candidate reaches RDKit materialization.
        """
        connections: set[_ActionSequenceCandidate] = set()
        # Heap prioritizes the normalized action count, with a stable serial tie-break.
        frontier: list[tuple[int, int, CleavageActionSequence | None, int]] = []
        minimum_cursor: dict[CleavageActionSequence | None, int] = {}
        serial = 0
        for seed in self.seeds:
            rejection = self.relations.rejection(tuple(self.index_by_action[a] for a in seed.actions))
            if rejection:
                raise ValueError(f"Invalid seed action sequence: {rejection}")
            self.visited_sequence_keys.add(seed.key)
            candidate = _ActionSequenceCandidate(None, seed, None, True)
            connections.add(candidate)
            yield candidate
            heapq.heappush(frontier, (len(seed.actions), serial, seed, -1))
            minimum_cursor[seed] = -1
            serial += 1
        if not self.seeds:
            frontier.append((0, serial, None, -1))
            minimum_cursor[None] = -1
            serial += 1
        while frontier:
            _, _, parent, cursor = heapq.heappop(frontier)
            if minimum_cursor[parent] != cursor:
                continue
            previous = parent.actions if parent else ()
            if len(previous) >= self.max_action_count:
                continue
            for i in range(cursor + 1, len(self.actions)):
                action = self.actions[i]
                if action in previous:
                    continue
                self.stats.num_raw_combinations += 1
                if 0 <= self.max_cleavage_combinations < self.stats.num_raw_combinations:
                    raise FragmentTreeLimitExceeded("max_cleavage_combinations", self.max_cleavage_combinations)
                indices = tuple(self.index_by_action[a] for a in previous) + (i,)
                rejection = self.relations.rejection(indices)
                if rejection:
                    if rejection == "hard_conflict":
                        self.stats.num_hard_conflict_pruned += 1
                    else:
                        self.stats.num_invalidated_action_pruned += 1
                    continue
                sequence = CleavageActionSequence((*previous, action))
                if len(sequence.actions) < len(previous) + 1:
                    self.stats.num_redundancy_pruned += 1
                if not sequence.retained_atom_maps or len(sequence.actions) > self.max_action_count:
                    continue
                if sequence == parent:
                    self.stats.num_duplicate_sequence_pruned += 1
                    continue
                candidate = _ActionSequenceCandidate(parent, sequence, action)
                if sequence.key in self.visited_sequence_keys:
                    self.stats.num_duplicate_sequence_pruned += 1
                else:
                    self.visited_sequence_keys.add(sequence.key)
                if candidate not in connections:
                    connections.add(candidate)
                    yield candidate
                if sequence not in minimum_cursor or i < minimum_cursor[sequence]:
                    minimum_cursor[sequence] = i
                    heapq.heappush(frontier, (len(sequence.actions), serial, sequence, i))
                    serial += 1
