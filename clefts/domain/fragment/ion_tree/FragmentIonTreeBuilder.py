from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ....libs.mmkit.mmkit import Compound
from ..cleavage.CleavagePatternSet import CleavagePatternSet
from ..tree.FragmentTree import FragmentTree
from ..tree.FragmentTreeBuilder import FragmentTreeBuilder
from .FragmentIonAdductRuleSet import FragmentIonAdductRuleSet
from .FragmentIonTree import FragmentIonTree
from ._private._FragmentHydrogenStateCandidateStore import (
    _FragmentHydrogenStateCandidateStore,
)
from ._private._FragmentIonShiftCandidateStore import (
    _FragmentIonShiftCandidateStore,
)


@dataclass(frozen=True, kw_only=True)
class FragmentIonTreeBuilder(FragmentTreeBuilder):
    """Build FragmentIonTree directly from Compound.

    This builder first builds a FragmentTree, then creates fragment-ion
    candidate stores from FragmentIonAdductRuleSet.

    Hydrogen state candidates are generated from adduct-rule settings.
    Ion shift candidates are generated from adduct-rule settings and
    fragment-node atom symbols.
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
        max_depth: Optional[int] = None,
        print_info: bool = False,
        _include_fragment_compound_cache: bool = False,
    ) -> FragmentIonTree:
        """Build FragmentIonTree from Compound."""

        result = self._build_result(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            max_depth=max_depth,
            print_info=print_info,
        )

        fragment_tree: FragmentTree = result["fragment_tree"]
        fragment_compound_by_index: dict[int, Compound] = result[
            "fragment_compound_by_index"
        ]

        hydrogen_state_candidate_store = (
            self._build_hydrogen_state_candidate_store()
        )

        ion_shift_candidate_store = self._build_ion_shift_candidate_store(
            fragment_tree=fragment_tree,
            fragment_compound_by_index=fragment_compound_by_index,
        )

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            fragment_tree,
            fragment_ion_adduct_rule_set=self.fragment_ion_adduct_rule_set,
            hydrogen_state_candidate_store=hydrogen_state_candidate_store,
            ion_shift_candidate_store=ion_shift_candidate_store,
        )

        if _include_fragment_compound_cache:
            object.__setattr__(
                fragment_ion_tree,
                "_fragment_compound_by_index",
                fragment_compound_by_index,
            )

        return fragment_ion_tree

    def build_fragment_tree(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        max_depth: Optional[int] = None,
        print_info: bool = False,
    ) -> FragmentTree:
        """Build only FragmentTree."""

        return super().build(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            max_depth=max_depth,
            print_info=print_info,
        )

    def _build_hydrogen_state_candidate_store(
        self,
    ) -> _FragmentHydrogenStateCandidateStore:
        """Build hydrogen state candidates from adduct rules."""

        return _FragmentHydrogenStateCandidateStore.from_adduct_rule_set(
            self.fragment_ion_adduct_rule_set
        )

    def _build_ion_shift_candidate_store(
        self,
        *,
        fragment_tree: FragmentTree,
        fragment_compound_by_index: dict[int, Compound],
    ) -> _FragmentIonShiftCandidateStore:
        """Build ion shift candidates from adduct rules and node atoms."""

        node_atom_symbols = self._build_node_atom_symbols(
            fragment_tree=fragment_tree,
            fragment_compound_by_index=fragment_compound_by_index,
        )

        node_charges = self._build_node_charges(
            fragment_tree=fragment_tree,
            fragment_compound_by_index=fragment_compound_by_index,
        )

        return _FragmentIonShiftCandidateStore.from_adduct_rule_set(
            adduct_rule_set=self.fragment_ion_adduct_rule_set,
            node_atom_symbols=node_atom_symbols,
            node_charges=node_charges,
        )

    @staticmethod
    def _build_node_atom_symbols(
        *,
        fragment_tree: FragmentTree,
        fragment_compound_by_index: dict[int, Compound],
    ) -> tuple[frozenset[str], ...]:
        """Return atom symbols contained in each fragment node."""

        node_atom_symbols: list[frozenset[str]] = []

        for node_index in range(fragment_tree.num_nodes):
            compound = fragment_compound_by_index[node_index]
            node_atom_symbols.append(
                FragmentIonTreeBuilder._get_compound_atom_symbols(compound)
            )

        return tuple(node_atom_symbols)

    @staticmethod
    def _build_node_charges(
        *,
        fragment_tree: FragmentTree,
        fragment_compound_by_index: dict[int, Compound],
    ) -> tuple[int, ...]:
        """Return formal charges for each fragment node."""

        node_charges: list[int] = []

        for node_index in range(fragment_tree.num_nodes):
            compound = fragment_compound_by_index[node_index]
            node_charges.append(compound.charge)

        return tuple(node_charges)

    @staticmethod
    def _get_compound_atom_symbols(
        compound: Compound,
    ) -> frozenset[str]:
        """Return atom symbols contained in one Compound.

        This assumes mmkit Compound exposes an RDKit-like mol object.
        If your Compound API is different, only this method needs to be
        adjusted.
        """

        if hasattr(compound, "mol"):
            mol = compound.mol
        elif hasattr(compound, "rdmol"):
            mol = compound.rdmol
        elif hasattr(compound, "to_mol"):
            mol = compound.to_mol()
        else:
            raise TypeError(
                "Compound must expose mol, rdmol, or to_mol() to collect "
                "atom symbols."
            )

        return frozenset(
            atom.GetSymbol()
            for atom in mol.GetAtoms()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "fragment_ion_adduct_rule_set": (
                self.fragment_ion_adduct_rule_set.to_dict()
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> FragmentIonTreeBuilder:
        return cls(
            max_depth=data["max_depth"],
            cleavage_pattern_set=CleavagePatternSet.from_dict(
                data["cleavage_pattern_set"]
            ),
            only_add_min_depth=data["only_add_min_depth"],
            min_depth_only_from=data["min_depth_only_from"],
            fragment_ion_adduct_rule_set=FragmentIonAdductRuleSet.from_dict(
                data["fragment_ion_adduct_rule_set"]
            ),
        )

    def copy(self) -> FragmentIonTreeBuilder:
        return FragmentIonTreeBuilder(
            max_depth=self.max_depth,
            cleavage_pattern_set=self.cleavage_pattern_set.copy(),
            only_add_min_depth=self.only_add_min_depth,
            min_depth_only_from=self.min_depth_only_from,
            fragment_ion_adduct_rule_set=(
                self.fragment_ion_adduct_rule_set.copy()
            ),
        )