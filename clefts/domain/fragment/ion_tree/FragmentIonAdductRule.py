from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Set, Optional

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

    _precursor_delta_h_state_by_adduct: Dict[Adduct, Tuple[int, int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

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

        object.__setattr__(
            self,
            "_precursor_delta_h_state_by_adduct",
            self._build_precursor_delta_h_state_by_adduct(),
        )
    
    @property
    def unsaturation_candidates(self) -> Tuple[int, ...]:
        """Return unsaturation candidates for this adduct rule.

        For now, only the specified unsaturation value is returned.
        """
        return tuple(u for u in range(self.unsaturation + 1))
    
    @property
    def unsaturation_adduct_candidates(self) -> Tuple[Adduct, ...]:
        """Return adduct candidates for this adduct rule.

        For now, only the specified unsaturation value is returned.
        """
        return tuple(Adduct.from_dict({"H": cnt * -2}) for cnt in self.unsaturation_candidates)
    
    @property
    def radical_candidates(self) -> Tuple[int, ...]:
        """Return radical delta H candidates for this adduct rule.

        For now, only the specified radical value is returned.
        """
        return (0, 1) if self.radical else (0,)
    
    @property
    def radical_adduct_candidates(self) -> Tuple[Adduct, ...]:
        """Return radical adduct candidates for this adduct rule.

        For now, only the specified radical value is returned.
        """
        return tuple(Adduct.from_dict({"H": -cnt}) for cnt in self.radical_candidates)
    
    @property
    def ion_shift_adduct_candidates(self) -> Tuple[Adduct, ...]:
        """Return ion shift adduct candidates for this adduct rule.

        For now, only the specified ion shift values are returned.
        """
        return tuple(ion_shift.ion_shift for ion_shift in self.ion_shifts)

    def get_precursor_delta_h_state_by_adduct(
        self,
        precursor_adduct: Adduct,
    ) -> Optional[Tuple[int, int]]:
        """Return precursor ion state from a precursor adduct.

        This method returns the ion state corresponding to the given
        precursor adduct.

        The returned ion state is:

            (unsaturation, radical)

        This method uses the lookup table built from self.adduct_type and
        possible H-loss states. It does not use ion_shifts.

        Example
        -------
        If self.adduct_type is [M+H]+, the lookup table contains:

            [M+H]+  -> (0, 0)
            [M]+    -> (0, 1)
            [M-H]+  -> (1, 0)
            [M-2H]+ -> (1, 1)

        Parameters
        ----------
        precursor_adduct:
            Observed or specified precursor adduct.

        Returns
        -------
        Optional[Tuple[int, int]]
            The corresponding (unsaturation, radical) state.
            Returns None if precursor_adduct is not registered in the lookup
            table.
        """

        if not isinstance(precursor_adduct, Adduct):
            raise TypeError("precursor_adduct must be an Adduct.")

        return self._precursor_delta_h_state_by_adduct.get(precursor_adduct, None)

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
    
    def _build_precursor_delta_h_state_by_adduct(
        self,
    ) -> Dict[Adduct, Tuple[int, int]]:
        """Build lookup table from precursor adduct type to delta H state.

        This table is based only on H loss from self.adduct_type.

        For each state:

            delta_H = -2 * unsaturation - radical

        Therefore, if self.adduct_type is [M+H]+:

            unsaturation=0, radical=0 -> delta_H= 0 -> [M+H]+
            unsaturation=0, radical=1 -> delta_H=-1 -> [M]+
            unsaturation=1, radical=0 -> delta_H=-2 -> [M-H]+
            unsaturation=1, radical=1 -> delta_H=-3 -> [M-2H]+
        """

        table: Dict[Adduct, Tuple[int, int]] = {}

        for radical_delta_h in self.radical_candidates:
            for unsaturation in self.unsaturation_candidates:
                state = (
                    int(unsaturation),
                    int(radical_delta_h != 0),
                )

                delta_h = -2 * int(unsaturation) - int(radical_delta_h)
                delta_h_adduct = Adduct.from_dict({"H": delta_h})
                adduct = self.adduct_type.add_prefer_self(delta_h_adduct)

                if adduct not in table:
                    table[adduct] = state

        return table