from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from .....libs.mmkit.mmkit import Adduct

if TYPE_CHECKING:
    from ..FragmentIonAdductRuleSet import FragmentIonAdductRuleSet


@dataclass(frozen=True)
class _FragmentIonShiftCandidateStore:
    """Ion shift candidates that can be applied to fragment nodes.

    Actual IonShiftRule objects are not duplicated.

    adduct_types:
        Adduct types in adduct-rule order.

    adduct_shift_rule_indptr:
        Pointer array from adduct rule index to shift rule range.

    shift_rule_indices:
        Shape is (num_shift_rules, 2).
        Columns are [adduct_rule_index, ion_shift_index].

    node_shift_rule_mask:
        Shape is (num_nodes, num_shift_rules).
        True means the shift rule can be applied to the node.
    """

    adduct_types: Tuple[Adduct, ...]
    adduct_shift_rule_indptr: np.ndarray
    shift_rule_indices: np.ndarray
    node_shift_rule_mask: np.ndarray

    def __post_init__(self) -> None:
        adduct_types = tuple(
            self._normalize_adduct_type(adduct_type)
            for adduct_type in self.adduct_types
        )

        adduct_shift_rule_indptr = np.asarray(
            self.adduct_shift_rule_indptr,
            dtype=np.int64,
        )

        shift_rule_indices = np.asarray(
            self.shift_rule_indices,
            dtype=np.int64,
        )

        node_shift_rule_mask = np.asarray(
            self.node_shift_rule_mask,
            dtype=bool,
        )

        if len(set(str(adduct_type) for adduct_type in adduct_types)) != len(
            adduct_types
        ):
            raise ValueError("adduct_types must be unique.")

        if adduct_shift_rule_indptr.ndim != 1:
            raise ValueError("adduct_shift_rule_indptr must be a 1D array.")

        if len(adduct_shift_rule_indptr) != len(adduct_types) + 1:
            raise ValueError(
                "adduct_shift_rule_indptr must have length "
                "num_adduct_types + 1."
            )

        if len(adduct_shift_rule_indptr) == 0:
            raise ValueError("adduct_shift_rule_indptr must not be empty.")

        if adduct_shift_rule_indptr[0] != 0:
            raise ValueError("adduct_shift_rule_indptr[0] must be 0.")

        if np.any(
            adduct_shift_rule_indptr[1:]
            < adduct_shift_rule_indptr[:-1]
        ):
            raise ValueError(
                "adduct_shift_rule_indptr must be non-decreasing."
            )

        if shift_rule_indices.ndim != 2:
            raise ValueError("shift_rule_indices must be a 2D array.")

        if shift_rule_indices.shape[1] != 2:
            raise ValueError(
                "shift_rule_indices must have shape "
                "(num_shift_rules, 2)."
            )

        if np.any(shift_rule_indices < 0):
            raise ValueError(
                "shift_rule_indices must contain non-negative integers."
            )

        if adduct_shift_rule_indptr[-1] != shift_rule_indices.shape[0]:
            raise ValueError(
                "adduct_shift_rule_indptr[-1] must equal "
                "the number of shift rules."
            )

        if len(adduct_types) == 0 and shift_rule_indices.shape[0] != 0:
            raise ValueError(
                "shift_rule_indices must be empty when adduct_types is empty."
            )

        if len(adduct_types) > 0 and shift_rule_indices.shape[0] > 0:
            adduct_rule_indices = shift_rule_indices[:, 0]

            if np.any(adduct_rule_indices >= len(adduct_types)):
                raise ValueError(
                    "shift_rule_indices contains invalid adduct_rule_index."
                )

        if node_shift_rule_mask.ndim != 2:
            raise ValueError("node_shift_rule_mask must be a 2D array.")

        if node_shift_rule_mask.shape[1] != shift_rule_indices.shape[0]:
            raise ValueError(
                "node_shift_rule_mask.shape[1] must match "
                "the number of shift rules."
            )

        self._validate_shift_rule_ranges(
            adduct_shift_rule_indptr=adduct_shift_rule_indptr,
            shift_rule_indices=shift_rule_indices,
            num_adduct_types=len(adduct_types),
        )

        object.__setattr__(self, "adduct_types", adduct_types)
        object.__setattr__(
            self,
            "adduct_shift_rule_indptr",
            adduct_shift_rule_indptr,
        )
        object.__setattr__(
            self,
            "shift_rule_indices",
            shift_rule_indices,
        )
        object.__setattr__(
            self,
            "node_shift_rule_mask",
            node_shift_rule_mask,
        )

    @classmethod
    def from_adduct_rule_set(
        cls,
        *,
        adduct_rule_set: FragmentIonAdductRuleSet,
        node_atom_symbols: Sequence[Collection[str]],
        node_charges: Sequence[int] | None = None,
    ) -> _FragmentIonShiftCandidateStore:
        """Create ion shift candidates from an adduct rule set.

        Parameters
        ----------
        adduct_rule_set:
            Fragment ion adduct rule set.

        node_atom_symbols:
            Atom symbols contained in each fragment node.

        node_charges:
            Formal charge for each fragment node.
            If charge != 0, all ion shifts are disabled for that node.
            If None, all nodes are treated as neutral.
        """

        if node_charges is None:
            node_charges = tuple(0 for _ in node_atom_symbols)

        if len(node_charges) != len(node_atom_symbols):
            raise ValueError(
                "node_charges must have the same length as "
                "node_atom_symbols."
            )

        normalized_node_charges = tuple(int(charge) for charge in node_charges)

        adduct_types: list[Adduct] = []
        shift_rule_positions: list[tuple[int, int]] = []
        indptr: list[int] = [0]

        for adduct_rule_index, adduct_rule in enumerate(
            adduct_rule_set.adduct_rules
        ):
            adduct_types.append(adduct_rule.adduct_type)

            for ion_shift_index, _ion_shift_rule in enumerate(
                adduct_rule.ion_shifts
            ):
                shift_rule_positions.append(
                    (adduct_rule_index, ion_shift_index)
                )

            indptr.append(len(shift_rule_positions))

        if shift_rule_positions:
            shift_rule_indices = np.asarray(
                shift_rule_positions,
                dtype=np.int64,
            )
        else:
            shift_rule_indices = np.empty((0, 2), dtype=np.int64)

        num_nodes = len(node_atom_symbols)
        num_shift_rules = len(shift_rule_positions)

        node_shift_rule_mask = np.zeros(
            (num_nodes, num_shift_rules),
            dtype=bool,
        )

        for shift_rule_index, (
            adduct_rule_index,
            ion_shift_index,
        ) in enumerate(shift_rule_positions):
            ion_shift_rule = (
                adduct_rule_set
                .adduct_rules[adduct_rule_index]
                .ion_shifts[ion_shift_index]
            )

            allowed_atoms = cls._get_allowed_atoms(ion_shift_rule)

            for node_index, atom_symbols in enumerate(node_atom_symbols):
                if normalized_node_charges[node_index] != 0:
                    continue

                node_shift_rule_mask[node_index, shift_rule_index] = (
                    cls._is_applicable_to_node(
                        allowed_atoms=allowed_atoms,
                        node_atom_symbols=atom_symbols,
                    )
                )

        return cls(
            adduct_types=tuple(adduct_types),
            adduct_shift_rule_indptr=np.asarray(indptr, dtype=np.int64),
            shift_rule_indices=shift_rule_indices,
            node_shift_rule_mask=node_shift_rule_mask,
        )

    @classmethod
    def empty(
        cls,
        *,
        num_nodes: int,
    ) -> _FragmentIonShiftCandidateStore:
        """Create an empty store with no adduct types and no shift rules."""

        if not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be an int.")

        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")

        return cls(
            adduct_types=tuple(),
            adduct_shift_rule_indptr=np.zeros(1, dtype=np.int64),
            shift_rule_indices=np.empty((0, 2), dtype=np.int64),
            node_shift_rule_mask=np.zeros((num_nodes, 0), dtype=bool),
        )

    @staticmethod
    def _normalize_adduct_type(
        adduct_type: Adduct | str,
    ) -> Adduct:
        if isinstance(adduct_type, Adduct):
            return adduct_type

        if isinstance(adduct_type, str):
            return Adduct.parse(adduct_type)

        raise TypeError(
            "adduct_type must be an Adduct or str. "
            f"Got {type(adduct_type).__name__}."
        )

    @staticmethod
    def _get_allowed_atoms(ion_shift_rule: object) -> tuple[str, ...]:
        atoms = getattr(ion_shift_rule, "atoms", None)

        if atoms is None:
            return tuple()

        return tuple(str(atom) for atom in atoms)

    @staticmethod
    def _is_applicable_to_node(
        *,
        allowed_atoms: tuple[str, ...],
        node_atom_symbols: Collection[str],
    ) -> bool:
        if len(allowed_atoms) == 0:
            return True

        return bool(set(allowed_atoms) & set(node_atom_symbols))

    @staticmethod
    def _validate_shift_rule_ranges(
        *,
        adduct_shift_rule_indptr: np.ndarray,
        shift_rule_indices: np.ndarray,
        num_adduct_types: int,
    ) -> None:
        for adduct_rule_index in range(num_adduct_types):
            start = int(adduct_shift_rule_indptr[adduct_rule_index])
            end = int(adduct_shift_rule_indptr[adduct_rule_index + 1])

            if start == end:
                continue

            actual_adduct_rule_indices = shift_rule_indices[start:end, 0]

            if not np.all(actual_adduct_rule_indices == adduct_rule_index):
                raise ValueError(
                    "adduct_shift_rule_indptr range does not match "
                    "shift_rule_indices[:, 0]. "
                    f"adduct_rule_index={adduct_rule_index}"
                )

            expected_ion_shift_indices = np.arange(
                end - start,
                dtype=np.int64,
            )

            actual_ion_shift_indices = shift_rule_indices[start:end, 1]

            if not np.array_equal(
                actual_ion_shift_indices,
                expected_ion_shift_indices,
            ):
                raise ValueError(
                    "ion_shift_index values must be contiguous within "
                    "each adduct rule range."
                )

    @property
    def num_nodes(self) -> int:
        return int(self.node_shift_rule_mask.shape[0])

    @property
    def num_adduct_types(self) -> int:
        return len(self.adduct_types)

    @property
    def num_shift_rules(self) -> int:
        return int(self.shift_rule_indices.shape[0])

    def get_adduct_type(
        self,
        adduct_rule_index: int,
    ) -> Adduct:
        """Return adduct type for one adduct_rule_index."""

        self._validate_adduct_rule_index(adduct_rule_index)
        return self.adduct_types[adduct_rule_index]

    def get_adduct_type_for_shift_rule(
        self,
        shift_rule_index: int,
    ) -> Adduct:
        """Return adduct type for one shift_rule_index."""

        adduct_rule_index = self.get_adduct_rule_index(shift_rule_index)
        return self.adduct_types[adduct_rule_index]

    def get_shift_rule_position(
        self,
        shift_rule_index: int,
    ) -> tuple[int, int]:
        """Return (adduct_rule_index, ion_shift_index)."""

        self._validate_shift_rule_index(shift_rule_index)

        return (
            int(self.shift_rule_indices[shift_rule_index, 0]),
            int(self.shift_rule_indices[shift_rule_index, 1]),
        )

    def get_adduct_rule_index(
        self,
        shift_rule_index: int,
    ) -> int:
        """Return adduct_rule_index for one shift_rule_index."""

        self._validate_shift_rule_index(shift_rule_index)
        return int(self.shift_rule_indices[shift_rule_index, 0])

    def get_ion_shift_index(
        self,
        shift_rule_index: int,
    ) -> int:
        """Return ion_shift_index for one shift_rule_index."""

        self._validate_shift_rule_index(shift_rule_index)
        return int(self.shift_rule_indices[shift_rule_index, 1])

    def get_shift_rule_range_for_adduct_rule(
        self,
        adduct_rule_index: int,
    ) -> slice:
        """Return shift-rule range for one adduct rule."""

        self._validate_adduct_rule_index(adduct_rule_index)

        start = int(self.adduct_shift_rule_indptr[adduct_rule_index])
        end = int(self.adduct_shift_rule_indptr[adduct_rule_index + 1])

        return slice(start, end)

    def get_shift_rule_indices_for_adduct_rule(
        self,
        adduct_rule_index: int,
    ) -> np.ndarray:
        """Return all shift_rule_indices belonging to one adduct rule."""

        shift_rule_range = self.get_shift_rule_range_for_adduct_rule(
            adduct_rule_index
        )

        return np.arange(
            shift_rule_range.start,
            shift_rule_range.stop,
            dtype=np.int64,
        )

    def get_adduct_rule_index_by_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> int:
        """Return adduct_rule_index for one adduct type."""

        key = str(self._normalize_adduct_type(adduct_type))

        for index, current_adduct_type in enumerate(self.adduct_types):
            if str(current_adduct_type) == key:
                return index

        raise KeyError(f"Unknown adduct_type: {key}")

    def get_shift_rule_indices_for_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> np.ndarray:
        """Return all shift_rule_indices belonging to one adduct type."""

        adduct_rule_index = self.get_adduct_rule_index_by_adduct_type(
            adduct_type
        )

        return self.get_shift_rule_indices_for_adduct_rule(
            adduct_rule_index
        )

    def get_node_shift_rule_mask(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return all shift-rule applicability values for one node."""

        self._validate_node_index(node_index)
        return self.node_shift_rule_mask[node_index]

    def get_node_adduct_type_mask(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return adduct-type applicability values for one node.

        True means the node has at least one applicable shift rule
        belonging to that adduct type.
        """

        self._validate_node_index(node_index)

        mask = np.zeros(self.num_adduct_types, dtype=bool)

        for adduct_rule_index in range(self.num_adduct_types):
            shift_rule_range = self.get_shift_rule_range_for_adduct_rule(
                adduct_rule_index
            )

            if shift_rule_range.start == shift_rule_range.stop:
                continue

            mask[adduct_rule_index] = bool(
                np.any(
                    self.node_shift_rule_mask[
                        node_index,
                        shift_rule_range,
                    ]
                )
            )

        return mask

    def is_shift_applicable(
        self,
        *,
        node_index: int,
        shift_rule_index: int,
    ) -> bool:
        """Return whether one shift rule is applicable to one node."""

        self._validate_node_index(node_index)
        self._validate_shift_rule_index(shift_rule_index)

        return bool(
            self.node_shift_rule_mask[node_index, shift_rule_index]
        )

    def is_adduct_type_applicable(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> bool:
        """Return whether one adduct type has any applicable shift."""

        self._validate_node_index(node_index)
        self._validate_adduct_rule_index(adduct_rule_index)

        shift_rule_range = self.get_shift_rule_range_for_adduct_rule(
            adduct_rule_index
        )

        if shift_rule_range.start == shift_rule_range.stop:
            return False

        return bool(
            np.any(
                self.node_shift_rule_mask[
                    node_index,
                    shift_rule_range,
                ]
            )
        )

    def get_applicable_shift_rule_indices(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return applicable shift_rule_indices for one node."""

        self._validate_node_index(node_index)

        return np.where(
            self.node_shift_rule_mask[node_index]
        )[0]

    def get_applicable_shift_rule_indices_for_adduct_rule(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> np.ndarray:
        """Return applicable shift rules for one node and one adduct rule."""

        self._validate_node_index(node_index)
        self._validate_adduct_rule_index(adduct_rule_index)

        shift_rule_range = self.get_shift_rule_range_for_adduct_rule(
            adduct_rule_index
        )

        local_indices = np.where(
            self.node_shift_rule_mask[node_index, shift_rule_range]
        )[0]

        return local_indices + shift_rule_range.start

    def get_applicable_shift_rule_indices_for_adduct_type(
        self,
        *,
        node_index: int,
        adduct_type: Adduct | str,
    ) -> np.ndarray:
        """Return applicable shift rules for one node and one adduct type."""

        adduct_rule_index = self.get_adduct_rule_index_by_adduct_type(
            adduct_type
        )

        return self.get_applicable_shift_rule_indices_for_adduct_rule(
            node_index=node_index,
            adduct_rule_index=adduct_rule_index,
        )

    def copy(self) -> _FragmentIonShiftCandidateStore:
        return _FragmentIonShiftCandidateStore(
            adduct_types=tuple(self.adduct_types),
            adduct_shift_rule_indptr=self.adduct_shift_rule_indptr.copy(),
            shift_rule_indices=self.shift_rule_indices.copy(),
            node_shift_rule_mask=self.node_shift_rule_mask.copy(),
        )

    def _validate_node_index(self, node_index: int) -> None:
        if not isinstance(node_index, int):
            raise TypeError("node_index must be an int.")

        if not 0 <= node_index < self.num_nodes:
            raise IndexError(
                f"node_index must be in [0, {self.num_nodes}). "
                f"Got {node_index}."
            )

    def _validate_shift_rule_index(
        self,
        shift_rule_index: int,
    ) -> None:
        if not isinstance(shift_rule_index, int):
            raise TypeError("shift_rule_index must be an int.")

        if not 0 <= shift_rule_index < self.num_shift_rules:
            raise IndexError(
                "shift_rule_index must be in "
                f"[0, {self.num_shift_rules}). "
                f"Got {shift_rule_index}."
            )

    def _validate_adduct_rule_index(
        self,
        adduct_rule_index: int,
    ) -> None:
        if not isinstance(adduct_rule_index, int):
            raise TypeError("adduct_rule_index must be an int.")

        if not 0 <= adduct_rule_index < self.num_adduct_types:
            raise IndexError(
                "adduct_rule_index must be in "
                f"[0, {self.num_adduct_types}). "
                f"Got {adduct_rule_index}."
            )