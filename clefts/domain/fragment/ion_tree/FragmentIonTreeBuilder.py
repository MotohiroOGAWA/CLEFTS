from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....libs.mmkit.mmkit import Compound
from ..tree.FragmentTreeBuilder import FragmentTreeBuilder
from ..tree.FragmentTree import FragmentTree
from .FragmentIonTree import FragmentIonTree
from .FragmentIonAdductRuleSet import FragmentIonAdductRuleSet
from ._private._FragmentIonStateStore import _FragmentIonStateStore
from ._private._FragmentIonShiftStore import _FragmentIonShiftStore


@dataclass(frozen=True, kw_only=True)
class FragmentIonTreeBuilder(FragmentTreeBuilder):
    """Build FragmentIonTree directly from Compound.

    This builder reuses FragmentTreeBuilder._build_result().
    It avoids reconstructing compounds from fragment SMILES when
    ion_state_store and ion_shift_store can be built from
    fragment_compound_by_index.
    """

    fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet

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

    def build(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentIonTree:
        """Build FragmentIonTree from Compound."""
        result = self._build_result(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info,
        )

        fragment_tree: FragmentTree = result["fragment_tree"]
        fragment_compound_by_index: dict[int, Compound] = result[
            "fragment_compound_by_index"
        ]

        ion_state_store = self._build_ion_state_store(
            fragment_tree=fragment_tree,
            fragment_compound_by_index=fragment_compound_by_index,
        )

        ion_shift_store = self._build_ion_shift_store(
            fragment_tree=fragment_tree,
            fragment_compound_by_index=fragment_compound_by_index,
        )

        return FragmentIonTree.from_fragment_tree(
            fragment_tree,
            fragment_ion_adduct_rule_set=self.fragment_ion_adduct_rule_set,
            ion_state_store=ion_state_store,
            ion_shift_store=ion_shift_store,
        )

    def _build_ion_state_store(
        self,
        *,
        fragment_tree: FragmentTree,
        fragment_compound_by_index: dict[int, Compound],
    ) -> _FragmentIonStateStore:
        return self.fragment_ion_adduct_rule_set.build_ion_state_store(
            fragment_tree,
            fragment_compound_by_index=fragment_compound_by_index,
        )

    def _build_ion_shift_store(
        self,
        *,
        fragment_tree: FragmentTree,
        fragment_compound_by_index: dict[int, Compound],
    ) -> _FragmentIonShiftStore:
        return self.fragment_ion_adduct_rule_set.build_ion_shift_store(
            fragment_tree,
            fragment_compound_by_index=fragment_compound_by_index,
        )

    def copy(self) -> "FragmentIonTreeBuilder":
        return FragmentIonTreeBuilder(
            max_depth=self.max_depth,
            cleavage_pattern_set=self.cleavage_pattern_set.copy(),
            only_add_min_depth=self.only_add_min_depth,
            min_depth_only_from=self.min_depth_only_from,
            fragment_ion_adduct_rule_set=self.fragment_ion_adduct_rule_set.copy(),
        )