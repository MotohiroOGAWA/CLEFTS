"""Order-independent, validated combinations of concrete cleavage actions."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from .CleavageAction import CleavageAction


@dataclass(frozen=True, init=False)
class CleavageActionSequence:
    actions: tuple[CleavageAction, ...]

    def __init__(self, actions):
        unique = tuple(sorted(set(actions), key=lambda a: a.key))
        if not unique:
            raise ValueError("At least one CleavageAction is required")
        universe = unique[0].retained_atom_maps | unique[0].discarded_atom_maps
        for a in unique:
            if a.retained_atom_maps | a.discarded_atom_maps != universe:
                raise ValueError("Actions must describe the same Source atom universe")
        for a, b in combinations(unique, 2):
            shared = a.changed_bond_maps & b.changed_bond_maps
            if shared:
                raise ValueError(f"Actions {a.key} and {b.key} share changed Source bonds: {sorted(shared)}")
        # Retained-set inclusion alone cannot remove additional cuts between surviving atoms.
        kept = []
        for a in unique:
            redundant = False
            for b in unique:
                if a == b or not b.retained_atom_maps <= a.retained_atom_maps:
                    continue
                if any({u, v} <= b.retained_atom_maps for u, v, _ in a.bond_updates):
                    continue
                surviving_cuts = frozenset(edge for edge in a.cut_bond_maps
                                           if set(edge) <= b.retained_atom_maps)
                if not surviving_cuts <= b.cut_bond_maps:
                    continue
                if b.retained_atom_maps < a.retained_atom_maps:
                    redundant = True
                    break
                # Equal retained sets: deterministic tie-break, preserving extra edits.
                b_cuts = frozenset(edge for edge in b.cut_bond_maps if set(edge) <= b.retained_atom_maps)
                if surviving_cuts < b_cuts or (surviving_cuts == b_cuts and b.key < a.key):
                    redundant = True
                    break
            if not redundant:
                kept.append(a)
        object.__setattr__(self, "actions", tuple(kept))

    @classmethod
    def from_actions(cls, actions):
        return cls(actions)

    @property
    def retained_atom_maps(self) -> frozenset[int]:
        return frozenset.intersection(*(a.retained_atom_maps for a in self.actions))

    @property
    def cut_bond_maps(self) -> frozenset[tuple[int, int]]:
        return frozenset().union(*(a.cut_bond_maps for a in self.actions))

    @property
    def bond_updates(self):
        return frozenset().union(*(a.bond_updates for a in self.actions))

    def compile(self, source):
        from .CompositeCleavageReaction import CompositeCleavageReaction
        return CompositeCleavageReaction.from_sequence(source=source, action_sequence=self)
