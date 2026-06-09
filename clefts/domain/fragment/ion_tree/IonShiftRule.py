from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Set

from rdkit import Chem

from ....libs.mmkit.mmkit import Adduct


@dataclass(frozen=True)
class IonShiftRule:
    """Ion shift rule for a group of atoms."""

    ion_shift: Adduct
    atoms: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize fields."""

        if not isinstance(self.ion_shift, Adduct):
            raise TypeError(
                "ion_shift must be an Adduct instance. "
                f"Got {type(self.ion_shift).__name__}."
            )

        if isinstance(self.atoms, str):
            raise TypeError(
                "atoms must be an iterable of atom symbols, not a single string."
            )

        try:
            atoms = tuple(str(atom) for atom in self.atoms)
        except TypeError as exc:
            raise TypeError(
                "atoms must be an iterable of atom symbols."
            ) from exc

        atoms = tuple(atom.strip() for atom in atoms)

        for atom in atoms:
            if not atom:
                raise ValueError("Atom symbol must not be empty.")

            if not self._is_valid_atom_symbol(atom):
                raise ValueError(f"Invalid atom symbol: {atom!r}")

        # Remove duplicate atoms while preserving order.
        atoms = tuple(dict.fromkeys(atoms))

        object.__setattr__(self, "atoms", atoms)

    @staticmethod
    def _is_valid_atom_symbol(symbol: str) -> bool:
        periodic_table = Chem.GetPeriodicTable()

        try:
            atomic_number = periodic_table.GetAtomicNumber(symbol)
        except RuntimeError:
            return False

        return atomic_number > 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IonShiftRule":
        return cls(
            ion_shift=Adduct.parse(data["ion_shift"]),
            atoms=tuple(str(atom) for atom in data.get("atoms", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ion_shift": str(self.ion_shift),
        }

        if self.atoms:
            data["atoms"] = list(self.atoms)

        return data

    def is_applicable_to_atoms(self, atom_symbols: Set[str]) -> bool:
        """Return True if this ion shift can be applied to the fragment atoms.

        If atoms is empty, the rule is treated as a generic rule.
        """

        if not self.atoms:
            return True

        return bool(set(self.atoms) & atom_symbols)
    
    def copy(self) -> "IonShiftRule":
        """Return a deep copy of this IonShiftRule."""
        return IonShiftRule(
            ion_shift=self.ion_shift.copy(),
            atoms=tuple(self.atoms),
        )