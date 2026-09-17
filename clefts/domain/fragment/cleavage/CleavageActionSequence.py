"""Order-independent, validated combinations of concrete cleavage actions."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from dataclasses import dataclass
from rdkit import Chem
from clefts.libs.mmkit.mmkit import Compound
from itertools import combinations
from .CleavageAction import CleavageAction
from .CleavageActionRelations import action_dominates

if TYPE_CHECKING:
    from .CompositeCleavageReaction import CompositeCleavageReaction


@dataclass(frozen=True, init=False)
class CleavageActionSequence:
    actions: tuple[CleavageAction, ...]

    def __init__(self, actions: Iterable[CleavageAction]) -> None:
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
        kept = [a for a in unique if not any(action_dominates(b, a) for b in unique)]
        object.__setattr__(self, "actions", tuple(kept))

    @classmethod
    def from_actions(cls, actions: Iterable[CleavageAction]) -> CleavageActionSequence:
        return cls(actions)

    @property
    def retained_atom_maps(self) -> frozenset[int]:
        return frozenset.intersection(*(a.retained_atom_maps for a in self.actions))

    @property
    def cut_bond_maps(self) -> frozenset[tuple[int, int]]:
        return frozenset().union(*(a.cut_bond_maps for a in self.actions))

    @property
    def bond_updates(self) -> frozenset[tuple[int, int, Chem.BondType]]:
        return frozenset().union(*(a.bond_updates for a in self.actions))

    @property
    def key(self) -> tuple[tuple[object, ...], ...]:
        """Canonical action history, distinct from the final effect."""
        return tuple(action.key for action in self.actions)

    @property
    def effective_cut_bond_maps(self) -> frozenset[tuple[int, int]]:
        retained = self.retained_atom_maps
        return frozenset(edge for edge in self.cut_bond_maps if set(edge) <= retained)

    @property
    def effective_bond_updates(self) -> frozenset[tuple[int, int, Chem.BondType]]:
        retained = self.retained_atom_maps
        return frozenset((u, v, order) for u, v, order in self.bond_updates
                         if {u, v} <= retained)

    @property
    def effect_key(self) -> tuple[
        tuple[int, ...], tuple[tuple[int, int], ...],
        tuple[tuple[int, int, Chem.BondType], ...],
    ]:
        return (tuple(sorted(self.retained_atom_maps)),
                tuple(sorted(self.effective_cut_bond_maps)),
                tuple(sorted(self.effective_bond_updates)))

    def compile(self, source: Compound | Chem.Mol) -> CompositeCleavageReaction:
        from .CompositeCleavageReaction import CompositeCleavageReaction
        return CompositeCleavageReaction.from_sequence(source=source, action_sequence=self)
