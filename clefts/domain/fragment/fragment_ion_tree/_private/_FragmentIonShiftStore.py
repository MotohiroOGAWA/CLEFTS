from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _FragmentIonShiftStore:
    """Array-backed store for node-wise IonShiftRule applicability.

    This store does not duplicate IonShiftRule objects.

    Actual rules are stored in FragmentIonAdductRuleSet:

        rule_set.adduct_rules[adduct_rule_index].ion_shifts[ion_shift_index]

    shift_rule_indices:
        2D int array with shape (num_shift_rules, 2).

        Column 0:
            adduct_rule_index in FragmentIonAdductRuleSet.adduct_rules.

        Column 1:
            ion_shift_index in FragmentIonAdductRule.ion_shifts.

    node_shift_rule_mask:
        2D bool array with shape (num_nodes, num_shift_rules).

        Axis 0:
            FragmentTree node_index.

        Axis 1:
            shift_rule_index.

    Examples
    --------
    shift_rule_indices[3] = [1, 1]

    means:

        rule_set.adduct_rules[1].ion_shifts[1]

    node_shift_rule_mask[5, 3] == True

    means:

        The shift rule 3 can be applied to FragmentTree node 5.
    """

    shift_rule_indices: np.ndarray
    node_shift_rule_mask: np.ndarray

    def __post_init__(self) -> None:
        shift_rule_indices = np.asarray(
            self.shift_rule_indices,
            dtype=np.int64,
        )

        node_shift_rule_mask = np.asarray(
            self.node_shift_rule_mask,
            dtype=bool,
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

        if node_shift_rule_mask.ndim != 2:
            raise ValueError("node_shift_rule_mask must be a 2D array.")

        if node_shift_rule_mask.shape[1] != shift_rule_indices.shape[0]:
            raise ValueError(
                "node_shift_rule_mask.shape[1] must match "
                "the number of shift rules."
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

    @staticmethod
    def empty(
        *,
        num_nodes: int,
        num_shift_rules: int,
    ) -> "_FragmentIonShiftStore":
        if not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be an int.")

        if not isinstance(num_shift_rules, int):
            raise TypeError("num_shift_rules must be an int.")

        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")

        if num_shift_rules < 0:
            raise ValueError("num_shift_rules must be non-negative.")

        return _FragmentIonShiftStore(
            shift_rule_indices=np.empty((num_shift_rules, 2), dtype=np.int64),
            node_shift_rule_mask=np.zeros(
                (num_nodes, num_shift_rules),
                dtype=bool,
            ),
        )

    @property
    def num_nodes(self) -> int:
        return int(self.node_shift_rule_mask.shape[0])

    @property
    def num_shift_rules(self) -> int:
        return int(self.shift_rule_indices.shape[0])

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

    def get_node_shift_rule_mask(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return all shift-rule applicability values for one node."""

        self._validate_node_index(node_index)
        return self.node_shift_rule_mask[node_index]

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

    def get_applicable_shift_rule_indices(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return applicable shift_rule_indices for one node."""

        self._validate_node_index(node_index)

        return np.where(
            self.node_shift_rule_mask[node_index]
        )[0]

    def get_shift_rule_indices_for_adduct_rule(
        self,
        adduct_rule_index: int,
    ) -> np.ndarray:
        """Return all shift_rule_indices belonging to one adduct rule."""

        self._validate_adduct_rule_index(adduct_rule_index)

        return np.where(
            self.shift_rule_indices[:, 0] == adduct_rule_index
        )[0]

    def get_applicable_shift_rule_indices_for_adduct_rule(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> np.ndarray:
        """Return applicable shift rules for one node and one adduct rule.

        This combines:

        - FragmentTree node applicability
        - membership in the specified FragmentIonAdductRule
        """

        self._validate_node_index(node_index)
        self._validate_adduct_rule_index(adduct_rule_index)

        node_mask = self.node_shift_rule_mask[node_index]

        adduct_rule_mask = (
            self.shift_rule_indices[:, 0] == adduct_rule_index
        )

        return np.where(node_mask & adduct_rule_mask)[0]

    def copy(self) -> "_FragmentIonShiftStore":
        return _FragmentIonShiftStore(
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

        if adduct_rule_index < 0:
            raise ValueError("adduct_rule_index must be non-negative.")