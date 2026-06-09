from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Set

from ....libs.mmkit.mmkit import Adduct, Compound

from .IonShiftRule import IonShiftRule


@dataclass(frozen=True)
class FragmentIonAdductRule:
    name: str
    adduct_type: Adduct
    radical: bool
    unsaturation: int
    ion_shifts: Tuple[IonShiftRule, ...]
    radical_atoms: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")

        if not self.name:
            raise ValueError("name must not be empty")

        if not isinstance(self.adduct_type, Adduct):
            raise TypeError("adduct_type must be an Adduct")

        if not isinstance(self.radical, bool):
            raise TypeError("radical must be a boolean")

        if not isinstance(self.unsaturation, int):
            raise TypeError("unsaturation must be an integer")

        if self.unsaturation < 0:
            raise ValueError("unsaturation must be non-negative")

        if not isinstance(self.ion_shifts, tuple):
            raise TypeError("ion_shifts must be a tuple")

        if not all(isinstance(ion_shift, IonShiftRule) for ion_shift in self.ion_shifts):
            raise TypeError("ion_shifts must be a tuple of IonShiftRule")

        if not isinstance(self.radical_atoms, tuple):
            raise TypeError("radical_atoms must be a tuple")

        for atom in self.radical_atoms:
            if not isinstance(atom, str):
                raise TypeError("radical_atoms must contain only strings")
            if not atom:
                raise ValueError("radical atom symbol must not be empty")

        # Remove duplicated radical atoms while preserving order.
        unique_radical_atoms = tuple(dict.fromkeys(self.radical_atoms))
        if unique_radical_atoms != self.radical_atoms:
            object.__setattr__(self, "radical_atoms", unique_radical_atoms)

        if not self.radical and self.radical_atoms:
            raise ValueError(
                "radical_atoms must be empty when radical is False"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragmentIonAdductRule":
        return cls(
            name=str(data["name"]),
            adduct_type=Adduct.parse(data["adduct_type"]),
            radical=bool(data["radical"]),
            unsaturation=int(data["unsaturation"]),
            radical_atoms=tuple(
                str(atom)
                for atom in data.get("radical_atoms", [])
            ),
            ion_shifts=tuple(
                IonShiftRule.from_dict(ion_shift_data)
                for ion_shift_data in data.get("ion_shifts", [])
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "adduct_type": str(self.adduct_type),
            "radical": self.radical,
            "unsaturation": self.unsaturation,
            "ion_shifts": [
                ion_shift.to_dict()
                for ion_shift in self.ion_shifts
            ],
        }

        if self.radical_atoms:
            data["radical_atoms"] = list(self.radical_atoms)

        return data

    def is_radical_applicable_to_atoms(
        self,
        atom_symbols: Set[str],
    ) -> bool:
        """Return whether radical formation is applicable to the fragment atoms.

        radical_atoms = () means radical is generic and can be applied
        to any fragment.
        """

        if not self.radical:
            return False

        if not self.radical_atoms:
            return True

        return bool(set(self.radical_atoms) & atom_symbols)

    def get_applicable_ion_shifts(
        self,
        atom_symbols: Set[str],
    ) -> Tuple[IonShiftRule, ...]:
        return tuple(
            ion_shift
            for ion_shift in self.ion_shifts
            if ion_shift.is_applicable_to_atoms(atom_symbols)
        )

    def is_applicable_to_atoms(
        self,
        atom_symbols: Set[str],
    ) -> bool:
        """Return whether this adduct rule can generate any candidate."""

        if self.radical and not self.is_radical_applicable_to_atoms(atom_symbols):
            return False

        return bool(self.get_applicable_ion_shifts(atom_symbols))

    def get_ion_state(self) -> tuple[int, int]:
        """Return ion state as (unsaturation, radical)."""

        return (
            int(self.unsaturation),
            int(self.radical),
        )

    def is_ion_state_applicable_to_compound(
        self,
        compound: Compound,
    ) -> bool:
        """Return whether this radical/unsaturation state is applicable.

        For now, unsaturation/radical values in this rule are treated as
        candidate states.

        If radical_atoms or other structural conditions are introduced,
        they should be checked here.
        """

        if not isinstance(compound, Compound):
            raise TypeError("compound must be a Compound.")

        return True

    def is_ion_shift_applicable_to_compound(
        self,
        *,
        compound: Compound,
        ion_shift_rule: IonShiftRule,
    ) -> bool:
        """Return whether one IonShiftRule is applicable to a compound.

        Current rule
        ------------
        1. If the fragment compound already has charge, return False.
        2. If the fragment is neutral:
            - generic ion shift is applicable.
            - atom-specific ion shift is applicable when the fragment
              contains at least one specified atom.
        """

        if not isinstance(compound, Compound):
            raise TypeError("compound must be a Compound.")

        if not isinstance(ion_shift_rule, IonShiftRule):
            raise TypeError("ion_shift_rule must be an IonShiftRule.")

        if compound.charge != 0:
            return False

        atom_symbols = self._get_atom_symbols(compound)

        return ion_shift_rule.is_applicable_to_atoms(atom_symbols)

    def copy(self) -> FragmentIonAdductRule:
        return FragmentIonAdductRule(
            name=self.name,
            adduct_type=self.adduct_type.copy(),
            radical=self.radical,
            unsaturation=self.unsaturation,
            radical_atoms=tuple(self.radical_atoms),
            ion_shifts=tuple(ion_shift.copy() for ion_shift in self.ion_shifts),
        )

    @staticmethod
    def _get_atom_symbols(compound: Compound) -> set[str]:
        mol = compound.mol

        return {
            atom.GetSymbol()
            for atom in mol.GetAtoms()
        }