"""Source chemistry relations; computed once, before neural execution."""
from __future__ import annotations
from dataclasses import dataclass
from .CleavageAction import CleavageAction

def action_dominates(b: CleavageAction, a: CleavageAction) -> bool:
    """Exact redundancy rule, shared by domain normalization and ML tables."""
    if a == b or not b.retained_atom_maps <= a.retained_atom_maps:
        return False
    if any({u, v} <= b.retained_atom_maps for u, v, _ in a.bond_updates):
        return False
    surviving = frozenset(edge for edge in a.cut_bond_maps if set(edge) <= b.retained_atom_maps)
    if not surviving <= b.cut_bond_maps:
        return False
    if b.retained_atom_maps < a.retained_atom_maps:
        return True
    b_cuts = frozenset(edge for edge in b.cut_bond_maps if set(edge) <= b.retained_atom_maps)
    return surviving < b_cuts or (surviving == b_cuts and b.key < a.key)


@dataclass(frozen=True)
class CleavageActionRelations:
    conflict_mask: tuple[int, ...]
    must_precede_mask: tuple[int, ...]
    redundancy_hint_mask: tuple[int, ...]
    conflict_pairs: tuple[tuple[int, int], ...]
    precedence_pairs: tuple[tuple[int, int], ...]
    dominance_pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_actions(cls, actions: tuple[CleavageAction, ...]) -> CleavageActionRelations:
        conflicts = [0] * len(actions)
        precedence = [0] * len(actions)
        hints = [0] * len(actions)
        for i, a in enumerate(actions):
            for j, b in enumerate(actions):
                if i == j:
                    continue
                if a.changed_bond_maps & b.changed_bond_maps:
                    conflicts[i] |= 1 << j
                # If A invalidates B's original match, B must precede A.
                if (not set(b.source_atom_maps) <= a.retained_atom_maps
                        or a.changed_bond_maps & b.matched_bond_maps):
                    precedence[j] |= 1 << i
                if set(a.source_atom_maps) <= b.discarded_atom_maps:
                    hints[i] |= 1 << j
        return cls(tuple(conflicts), tuple(precedence), tuple(hints),
                   tuple((i, j) for i, mask in enumerate(conflicts) for j in range(len(actions)) if mask & (1 << j)),
                   tuple((i, j) for i, mask in enumerate(precedence) for j in range(len(actions)) if mask & (1 << j)),
                   tuple((j, i) for i, a in enumerate(actions) for j, b in enumerate(actions)
                         if i != j and action_dominates(b, a)))

    def rejection(self, indices: tuple[int, ...]) -> str | None:
        selected = sum(1 << i for i in indices)
        if any(self.conflict_mask[i] & selected for i in indices):
            return "hard_conflict"
        incoming = {i: 0 for i in indices}
        for i in indices:
            for j in indices:
                if self.must_precede_mask[i] & (1 << j):
                    incoming[j] += 1
        pending = [i for i, degree in incoming.items() if degree == 0]
        removed = 0
        while pending:
            i = pending.pop()
            removed += 1
            for j in indices:
                if self.must_precede_mask[i] & (1 << j):
                    incoming[j] -= 1
                    if incoming[j] == 0:
                        pending.append(j)
        return "precedence_cycle" if removed != len(indices) else None


