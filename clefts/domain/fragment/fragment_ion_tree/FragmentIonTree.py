from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..fragment_tree.FragmentTree import FragmentTree
from .IonShiftRule import IonShiftRule
from .FragmentIonAdductRule import FragmentIonAdductRule
from .FragmentIonAdductRuleSet import FragmentIonAdductRuleSet
from ._private._FragmentIonStateStore import _FragmentIonStateStore
from ._private._FragmentIonShiftStore import _FragmentIonShiftStore


@dataclass(frozen=True, kw_only=True)
class FragmentIonTree(FragmentTree):
    """FragmentTree with fragment ion candidate stores.

    This class keeps the original FragmentTree topology unchanged.

    ion_state_store:
        Node-wise unsaturation/radical candidates.

    ion_shift_store:
        Node-wise IonShiftRule applicability.
    """

    fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet
    ion_state_store: _FragmentIonStateStore
    ion_shift_store: _FragmentIonShiftStore

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(
            self.fragment_ion_adduct_rule_set,
            FragmentIonAdductRuleSet,
        ):
            raise TypeError(
                "fragment_ion_adduct_rule_set must be a "
                "FragmentIonAdductRuleSet."
            )

        if not isinstance(self.ion_state_store, _FragmentIonStateStore):
            raise TypeError(
                "ion_state_store must be a _FragmentIonStateStore."
            )

        if not isinstance(self.ion_shift_store, _FragmentIonShiftStore):
            raise TypeError(
                "ion_shift_store must be a _FragmentIonShiftStore."
            )

        if self.ion_state_store.num_nodes != self.num_nodes:
            raise ValueError(
                "ion_state_store.num_nodes must match FragmentTree.num_nodes."
            )

        if self.ion_shift_store.num_nodes != self.num_nodes:
            raise ValueError(
                "ion_shift_store.num_nodes must match FragmentTree.num_nodes."
            )

        self._validate_shift_store_rule_indices()

    @classmethod
    def from_fragment_tree(
        cls,
        fragment_tree: FragmentTree,
        *,
        fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet,
        ion_state_store: _FragmentIonStateStore | None = None,
        ion_shift_store: _FragmentIonShiftStore | None = None,
    ) -> "FragmentIonTree":
        """Create FragmentIonTree from FragmentTree and rule set."""

        if not isinstance(fragment_tree, FragmentTree):
            raise TypeError("fragment_tree must be a FragmentTree.")

        if not isinstance(
            fragment_ion_adduct_rule_set,
            FragmentIonAdductRuleSet,
        ):
            raise TypeError(
                "fragment_ion_adduct_rule_set must be a "
                "FragmentIonAdductRuleSet."
            )

        if ion_state_store is None:
            ion_state_store = (
                fragment_ion_adduct_rule_set.build_ion_state_store(
                    fragment_tree
                )
            )

        if ion_shift_store is None:
            ion_shift_store = (
                fragment_ion_adduct_rule_set.build_ion_shift_store(
                    fragment_tree
                )
            )

        return cls(
            smiles=fragment_tree.smiles,
            node_store=fragment_tree.node_store,
            edge_store=fragment_tree.edge_store,
            adjacency=fragment_tree.adjacency,
            depths=fragment_tree.depths,
            fragment_ion_adduct_rule_set=fragment_ion_adduct_rule_set,
            ion_state_store=ion_state_store,
            ion_shift_store=ion_shift_store,
        )

    def _validate_shift_store_rule_indices(self) -> None:
        num_adduct_rules = len(
            self.fragment_ion_adduct_rule_set.adduct_rules
        )

        for shift_rule_index in range(self.ion_shift_store.num_shift_rules):
            adduct_rule_index, ion_shift_index = (
                self.ion_shift_store.get_shift_rule_position(
                    shift_rule_index
                )
            )

            if not 0 <= adduct_rule_index < num_adduct_rules:
                raise ValueError(
                    "shift_rule_indices contains invalid "
                    "adduct_rule_index. "
                    f"shift_rule_index={shift_rule_index}, "
                    f"adduct_rule_index={adduct_rule_index}"
                )

            adduct_rule = (
                self.fragment_ion_adduct_rule_set
                .adduct_rules[adduct_rule_index]
            )

            if not 0 <= ion_shift_index < len(adduct_rule.ion_shifts):
                raise ValueError(
                    "shift_rule_indices contains invalid ion_shift_index. "
                    f"shift_rule_index={shift_rule_index}, "
                    f"adduct_rule_index={adduct_rule_index}, "
                    f"ion_shift_index={ion_shift_index}"
                )

    @property
    def num_adduct_rules(self) -> int:
        return len(self.fragment_ion_adduct_rule_set.adduct_rules)

    @property
    def num_ion_states(self) -> int:
        return self.ion_state_store.num_states

    @property
    def num_shift_rules(self) -> int:
        return self.ion_shift_store.num_shift_rules

    @property
    def ion_states(self) -> np.ndarray:
        return self.ion_state_store.ion_states

    @property
    def node_state_indptr(self) -> np.ndarray:
        return self.ion_state_store.node_state_indptr

    @property
    def ion_state_delta_h(self) -> np.ndarray:
        return self.ion_state_store.delta_h

    @property
    def shift_rule_indices(self) -> np.ndarray:
        return self.ion_shift_store.shift_rule_indices

    @property
    def node_shift_rule_mask(self) -> np.ndarray:
        return self.ion_shift_store.node_shift_rule_mask

    def get_node_ion_states(
        self,
        node_index: int,
    ) -> np.ndarray:
        return self.ion_state_store.get_node_states(node_index)

    def get_node_ion_state_delta_h(
        self,
        node_index: int,
    ) -> np.ndarray:
        return self.ion_state_store.get_node_delta_h(node_index)

    def get_node_shift_rule_mask(
        self,
        node_index: int,
    ) -> np.ndarray:
        return self.ion_shift_store.get_node_shift_rule_mask(node_index)

    def get_applicable_shift_rule_indices(
        self,
        node_index: int,
    ) -> np.ndarray:
        return self.ion_shift_store.get_applicable_shift_rule_indices(
            node_index
        )

    def get_applicable_shift_rule_indices_for_adduct_rule(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> np.ndarray:
        self._validate_adduct_rule_index(adduct_rule_index)

        return (
            self.ion_shift_store
            .get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=node_index,
                adduct_rule_index=adduct_rule_index,
            )
        )

    def get_applicable_adduct_rule_indices(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return adduct_rule_indices with at least one applicable shift.

        This means:
        - the FragmentTree node has at least one applicable IonShiftRule
          belonging to the adduct rule.

        It does not mean the precursor adduct_type itself is invalid or valid.
        """

        self._validate_node_index(node_index)

        applicable_shift_rule_indices = (
            self.get_applicable_shift_rule_indices(node_index)
        )

        if len(applicable_shift_rule_indices) == 0:
            return np.empty((0,), dtype=np.int64)

        adduct_rule_indices = self.shift_rule_indices[
            applicable_shift_rule_indices,
            0,
        ]

        return np.unique(adduct_rule_indices).astype(
            np.int64,
            copy=False,
        )

    def has_applicable_shift_for_adduct_rule(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> bool:
        """Return whether one node has any applicable shift for an adduct rule."""

        shift_rule_indices = (
            self.get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=node_index,
                adduct_rule_index=adduct_rule_index,
            )
        )

        return len(shift_rule_indices) > 0

    def get_shift_rule_position(
        self,
        shift_rule_index: int,
    ) -> tuple[int, int]:
        """Return (adduct_rule_index, ion_shift_index)."""

        return self.ion_shift_store.get_shift_rule_position(
            shift_rule_index
        )

    def get_adduct_rule(
        self,
        adduct_rule_index: int,
    ) -> FragmentIonAdductRule:
        self._validate_adduct_rule_index(adduct_rule_index)

        return self.fragment_ion_adduct_rule_set.adduct_rules[
            adduct_rule_index
        ]

    def get_ion_shift_rule(
        self,
        shift_rule_index: int,
    ) -> IonShiftRule:
        adduct_rule_index, ion_shift_index = (
            self.ion_shift_store.get_shift_rule_position(shift_rule_index)
        )

        return (
            self.fragment_ion_adduct_rule_set
            .adduct_rules[adduct_rule_index]
            .ion_shifts[ion_shift_index]
        )

    def get_applicable_ion_shift_rules_for_adduct_rule(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> Tuple[IonShiftRule, ...]:
        shift_rule_indices = (
            self.get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=node_index,
                adduct_rule_index=adduct_rule_index,
            )
        )

        return tuple(
            self.get_ion_shift_rule(int(shift_rule_index))
            for shift_rule_index in shift_rule_indices
        )
    
    def _validate_node_index(
        self,
        node_index: int,
    ) -> None:
        if not isinstance(node_index, int):
            raise TypeError("node_index must be an int.")

        if not 0 <= node_index < self.num_nodes:
            raise IndexError(
                f"node_index must be in [0, {self.num_nodes}). "
                f"Got {node_index}."
            )

    def _validate_adduct_rule_index(
        self,
        adduct_rule_index: int,
    ) -> None:
        if not isinstance(adduct_rule_index, int):
            raise TypeError("adduct_rule_index must be an int.")

        if not 0 <= adduct_rule_index < self.num_adduct_rules:
            raise IndexError(
                "adduct_rule_index must be in "
                f"[0, {self.num_adduct_rules}). "
                f"Got {adduct_rule_index}."
            )