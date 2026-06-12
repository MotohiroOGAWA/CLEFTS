from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict
import numpy as np

from ....libs.mmkit.mmkit import Adduct

from ..tree.FragmentTree import FragmentTree
from .FragmentIonAdductRule import FragmentIonAdductRule
from .FragmentIonAdductRuleSet import FragmentIonAdductRuleSet
from .IonShiftRule import IonShiftRule
from ._private._FragmentHydrogenStateCandidateStore import _FragmentHydrogenStateCandidateStore
from ._private._FragmentIonShiftCandidateStore import _FragmentIonShiftCandidateStore
from ._private._FragmentIonFormulaCandidateGroup import _FragmentIonFormulaCandidateGroup


@dataclass(frozen=True, kw_only=True)
class FragmentIonTree(FragmentTree):
    """FragmentTree with fragment ion candidate stores.

    This class keeps the original FragmentTree topology unchanged.

    hydrogen_state_candidate_store:
        Hydrogen state candidates generated from adduct-rule settings.
        These candidates are grouped by adduct type, not by fragment node.

    ion_shift_candidate_store:
        Ion shift candidates and node-wise applicability.
    """

    fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet
    hydrogen_state_candidate_store: _FragmentHydrogenStateCandidateStore
    ion_shift_candidate_store: _FragmentIonShiftCandidateStore
    formula_candidate_groups: Dict[Adduct, _FragmentIonFormulaCandidateGroup] = field(default_factory=dict)

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

        if not isinstance(
            self.hydrogen_state_candidate_store,
            _FragmentHydrogenStateCandidateStore,
        ):
            raise TypeError(
                "hydrogen_state_candidate_store must be a "
                "_FragmentHydrogenStateCandidateStore."
            )
        self._validate_hydrogen_state_candidate_store()


        if not isinstance(
            self.ion_shift_candidate_store,
            _FragmentIonShiftCandidateStore,
        ):
            raise TypeError(
                "ion_shift_candidate_store must be a "
                "_FragmentIonShiftCandidateStore."
            )
        if self.ion_shift_candidate_store.num_nodes != self.num_nodes:
            raise ValueError(
                "ion_shift_candidate_store.num_nodes must match "
                "FragmentTree.num_nodes."
            )
        self._validate_ion_shift_candidate_store()


        if not isinstance(self.formula_candidate_groups, dict):
            raise TypeError(
                "formula_candidate_groups must be a dict."
            )
        self._validate_formula_candidate_groups()

    @classmethod
    def from_fragment_tree(
        cls,
        fragment_tree: FragmentTree,
        *,
        fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet,
        hydrogen_state_candidate_store: (
            _FragmentHydrogenStateCandidateStore | None
        ) = None,
        ion_shift_candidate_store: _FragmentIonShiftCandidateStore | None = None,
    ) -> "FragmentIonTree":
        """Create FragmentIonTree from FragmentTree and candidate stores.

        Notes
        -----
        hydrogen_state_candidate_store can be generated from the rule set.

        ion_shift_candidate_store must usually be passed by the builder,
        because ion shift applicability depends on fragment-node atom symbols.
        """

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

        if hydrogen_state_candidate_store is None:
            hydrogen_state_candidate_store = (
                _FragmentHydrogenStateCandidateStore.from_adduct_rule_set(
                    fragment_ion_adduct_rule_set
                )
            )

        if ion_shift_candidate_store is None:
            raise ValueError(
                "ion_shift_candidate_store must be provided. "
                "It depends on fragment-node atom symbols and should be "
                "built by FragmentIonTreeBuilder."
            )

        return cls(
            smiles=fragment_tree.smiles,
            node_store=fragment_tree.node_store,
            edge_store=fragment_tree.edge_store,
            adjacency=fragment_tree.adjacency,
            depths=fragment_tree.depths,
            fragment_ion_adduct_rule_set=fragment_ion_adduct_rule_set,
            hydrogen_state_candidate_store=hydrogen_state_candidate_store,
            ion_shift_candidate_store=ion_shift_candidate_store,
        )

    @property
    def num_adduct_rules(self) -> int:
        return len(self.fragment_ion_adduct_rule_set.adduct_rules)

    @property
    def num_hydrogen_state_candidates(self) -> int:
        return self.hydrogen_state_candidate_store.num_candidate_states

    @property
    def num_shift_rules(self) -> int:
        return self.ion_shift_candidate_store.num_shift_rules

    @property
    def declared_hydrogen_states(self) -> np.ndarray:
        return self.hydrogen_state_candidate_store.declared_states

    @property
    def hydrogen_state_candidates(self) -> np.ndarray:
        return self.hydrogen_state_candidate_store.candidate_states

    @property
    def adduct_state_indptr(self) -> np.ndarray:
        return self.hydrogen_state_candidate_store.adduct_state_indptr

    @property
    def hydrogen_state_candidate_delta_h(self) -> np.ndarray:
        return self.hydrogen_state_candidate_store.candidate_delta_h

    @property
    def shift_rule_indices(self) -> np.ndarray:
        return self.ion_shift_candidate_store.shift_rule_indices

    @property
    def shift_rule_adduct_types(self) -> Tuple[str, ...]:
        return self.ion_shift_candidate_store.adduct_types

    @property
    def node_shift_rule_mask(self) -> np.ndarray:
        return self.ion_shift_candidate_store.node_shift_rule_mask

    def get_hydrogen_state_candidates_for_adduct_rule(
        self,
        adduct_rule_index: int,
    ) -> np.ndarray:
        """Return hydrogen state candidates for one adduct rule."""

        self._validate_adduct_rule_index(adduct_rule_index)

        return self.hydrogen_state_candidate_store.get_candidate_states(
            adduct_rule_index
        )

    def get_hydrogen_state_candidate_delta_h_for_adduct_rule(
        self,
        adduct_rule_index: int,
    ) -> np.ndarray:
        """Return hydrogen delta candidates for one adduct rule."""

        self._validate_adduct_rule_index(adduct_rule_index)

        return self.hydrogen_state_candidate_store.get_candidate_delta_h(
            adduct_rule_index
        )

    def get_hydrogen_state_candidates_for_adduct_type(
        self,
        adduct_type: str,
    ) -> np.ndarray:
        """Return hydrogen state candidates for one adduct type."""

        return (
            self.hydrogen_state_candidate_store
            .get_candidate_states_by_adduct_type(adduct_type)
        )

    def get_hydrogen_state_candidate_delta_h_for_adduct_type(
        self,
        adduct_type: str,
    ) -> np.ndarray:
        """Return hydrogen delta candidates for one adduct type."""

        return (
            self.hydrogen_state_candidate_store
            .get_candidate_delta_h_by_adduct_type(adduct_type)
        )

    def get_hydrogen_state_candidate_delta_h_adduct_for_adduct_type(
        self,
        adduct_type: str,
    ) -> Tuple[Adduct, ...]:
        """Return hydrogen delta candidates for one adduct type."""

        delta_h = self.hydrogen_state_candidate_store.get_candidate_delta_h_by_adduct_type(adduct_type)
        adducts = tuple(Adduct.from_dict({"H": int(delta_h_i)}) for delta_h_i in delta_h)
        return adducts

    def get_adduct_rule_index_by_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> int:
        """Return adduct rule index by main adduct type."""

        if isinstance(adduct_type, Adduct):
            target_key = str(adduct_type)
        elif isinstance(adduct_type, str):
            target_key = str(Adduct.parse(adduct_type))
        else:
            raise TypeError(
                "adduct_type must be an Adduct or str. "
                f"Got {type(adduct_type).__name__}."
            )

        for adduct_rule_index, adduct_rule in enumerate(
            self.fragment_ion_adduct_rule_set.adduct_rules
        ):
            if str(adduct_rule.adduct_type) == target_key:
                return adduct_rule_index

        raise KeyError(f"Unknown adduct_type: {target_key}")

    def get_node_shift_rule_mask(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return all shift-rule applicability values for one node."""

        return self.ion_shift_candidate_store.get_node_shift_rule_mask(
            node_index
        )

    def get_node_adduct_type_mask(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return adduct-type applicability values for one node.

        True means the node has at least one applicable ion shift rule
        belonging to that adduct rule.
        """

        return self.ion_shift_candidate_store.get_node_adduct_type_mask(
            node_index
        )

    def get_applicable_shift_rule_indices(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return applicable shift rule indices for one node."""

        return (
            self.ion_shift_candidate_store
            .get_applicable_shift_rule_indices(node_index)
        )

    def get_applicable_shift_rule_indices_for_adduct_rule(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> np.ndarray:
        """Return applicable shift rules for one node and one adduct rule."""

        self._validate_adduct_rule_index(adduct_rule_index)

        return (
            self.ion_shift_candidate_store
            .get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=node_index,
                adduct_rule_index=adduct_rule_index,
            )
        )

    def get_applicable_shift_rule_indices_for_adduct_type(
        self,
        *,
        node_index: int,
        adduct_type: str,
    ) -> np.ndarray:
        """Return applicable shift rules for one node and one adduct type."""

        return (
            self.ion_shift_candidate_store
            .get_applicable_shift_rule_indices_for_adduct_type(
                node_index=node_index,
                adduct_type=adduct_type,
            )
        )

    def get_applicable_adduct_rule_indices(
        self,
        node_index: int,
    ) -> np.ndarray:
        """Return adduct rules with at least one applicable ion shift.

        This means the FragmentTree node has at least one applicable
        IonShiftRule belonging to the adduct rule.
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

    def is_adduct_type_applicable(
        self,
        *,
        node_index: int,
        adduct_rule_index: int,
    ) -> bool:
        return self.ion_shift_candidate_store.is_adduct_type_applicable(
            node_index=node_index,
            adduct_rule_index=adduct_rule_index,
        )

    def get_shift_rule_position(
        self,
        shift_rule_index: int,
    ) -> tuple[int, int]:
        """Return (adduct_rule_index, ion_shift_index)."""

        return self.ion_shift_candidate_store.get_shift_rule_position(
            shift_rule_index
        )

    def get_shift_rule_adduct_type(
        self,
        shift_rule_index: int,
    ) -> Adduct:
        """Return adduct type for one shift rule."""

        return self.ion_shift_candidate_store.get_adduct_type_for_shift_rule(
            shift_rule_index
        )

    def get_adduct_rule(
        self,
        adduct_rule_index: int,
    ) -> FragmentIonAdductRule:
        """Return one FragmentIonAdductRule."""

        self._validate_adduct_rule_index(adduct_rule_index)

        return self.fragment_ion_adduct_rule_set.adduct_rules[
            adduct_rule_index
        ]

    def get_ion_shift_rule(
        self,
        shift_rule_index: int,
    ) -> IonShiftRule:
        """Return one IonShiftRule."""

        adduct_rule_index, ion_shift_index = (
            self.ion_shift_candidate_store.get_shift_rule_position(
                shift_rule_index
            )
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
        """Return applicable IonShiftRule objects for one node and adduct rule."""

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

    def get_applicable_ion_shift_rules_for_adduct_type(
        self,
        *,
        node_index: int,
        adduct_type: str,
    ) -> Tuple[IonShiftRule, ...]:
        """Return applicable IonShiftRule objects for one node and adduct type."""

        shift_rule_indices = (
            self.get_applicable_shift_rule_indices_for_adduct_type(
                node_index=node_index,
                adduct_type=adduct_type,
            )
        )

        return tuple(
            self.get_ion_shift_rule(int(shift_rule_index))
            for shift_rule_index in shift_rule_indices
        )

    def get_formula_candidate_group(
        self,
        adduct_type: Adduct | str,
    ) -> _FragmentIonFormulaCandidateGroup:
        """Return formula candidate group for one main adduct type."""

        return self._ensure_formula_candidate_group(adduct_type)

    def get_node_formula_candidates(
        self,
        *,
        adduct_type: Adduct | str,
        formula: object,
    ) -> tuple[tuple[int, Adduct], ...]:
        """Return (node_index, candidate_adduct) candidates by formula."""

        group = self._ensure_formula_candidate_group(adduct_type)

        return group.get_candidates_by_formula(formula)

    def copy(self) -> FragmentIonTree:
        """Return a copy of this FragmentIonTree.

        formula_candidate_groups is a lazy cache, so it is intentionally
        not copied.
        """
        return self.from_fragment_tree(
            fragment_tree=super().copy(),
            fragment_ion_adduct_rule_set=self.fragment_ion_adduct_rule_set,
            hydrogen_state_candidate_store=self.hydrogen_state_candidate_store.copy(),
            ion_shift_candidate_store=self.ion_shift_candidate_store.copy(),
        )

    def _validate_hydrogen_state_candidate_store(self) -> None:
        """Validate hydrogen candidate store against the adduct rule set."""

        store = self.hydrogen_state_candidate_store

        if store.num_adduct_types != self.num_adduct_rules:
            raise ValueError(
                "hydrogen_state_candidate_store.num_adduct_types must "
                "match the number of adduct rules."
            )

        for adduct_rule_index, adduct_rule in enumerate(
            self.fragment_ion_adduct_rule_set.adduct_rules
        ):
            expected_adduct_type = adduct_rule.adduct_type
            actual_adduct_type = store.adduct_types[adduct_rule_index]

            if actual_adduct_type != expected_adduct_type:
                raise ValueError(
                    "hydrogen_state_candidate_store.adduct_types does not "
                    "match fragment_ion_adduct_rule_set.adduct_rules. "
                    f"adduct_rule_index={adduct_rule_index}, "
                    f"expected={expected_adduct_type}, "
                    f"actual={actual_adduct_type}"
                )
            
    def _validate_ion_shift_candidate_store(self) -> None:
        """Validate ion shift candidate store against the adduct rule set."""

        store = self.ion_shift_candidate_store

        if store.num_nodes != self.num_nodes:
            raise ValueError(
                "ion_shift_candidate_store.num_nodes must match "
                "FragmentIonTree.num_nodes."
            )

        if store.num_adduct_types != self.num_adduct_rules:
            raise ValueError(
                "ion_shift_candidate_store.num_adduct_types must match "
                "the number of adduct rules."
            )

        for adduct_rule_index, adduct_rule in enumerate(
            self.fragment_ion_adduct_rule_set.adduct_rules
        ):
            expected_adduct_type = str(adduct_rule.adduct_type)
            actual_adduct_type = str(
                store.get_adduct_type(adduct_rule_index)
            )

            if actual_adduct_type != expected_adduct_type:
                raise ValueError(
                    "ion_shift_candidate_store.adduct_types does not match "
                    "fragment_ion_adduct_rule_set.adduct_rules. "
                    f"adduct_rule_index={adduct_rule_index}, "
                    f"expected={expected_adduct_type}, "
                    f"actual={actual_adduct_type}"
                )

        for shift_rule_index in range(store.num_shift_rules):
            adduct_rule_index, ion_shift_index = store.get_shift_rule_position(
                shift_rule_index
            )

            if not 0 <= adduct_rule_index < self.num_adduct_rules:
                raise ValueError(
                    "shift_rule_indices contains invalid adduct_rule_index. "
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

            expected_adduct_type = str(adduct_rule.adduct_type)
            actual_adduct_type = str(
                store.get_adduct_type_for_shift_rule(shift_rule_index)
            )

            if actual_adduct_type != expected_adduct_type:
                raise ValueError(
                    "ion_shift_candidate_store.adduct_types does not match "
                    "the adduct type of the referenced adduct rule. "
                    f"shift_rule_index={shift_rule_index}, "
                    f"expected={expected_adduct_type}, "
                    f"actual={actual_adduct_type}"
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


    def _validate_formula_candidate_groups(self) -> None:
        """Validate existing formula candidate groups.

        This does not require all adduct types to be present because
        groups are built lazily per adduct type.
        """

        groups = self.formula_candidate_groups

        for adduct_type, group in groups.items():
            if not isinstance(group, _FragmentIonFormulaCandidateGroup):
                raise TypeError(
                    "formula_candidate_groups values must be "
                    "_FragmentIonFormulaCandidateGroup."
                )

            adduct_rule_index = self._get_adduct_rule_index_by_adduct_type(
                adduct_type
            )

            expected_adduct_type = (
                self.fragment_ion_adduct_rule_set
                .adduct_rules[adduct_rule_index]
                .adduct_type
            )

            if str(group.adduct_type) != str(expected_adduct_type):
                raise ValueError(
                    "formula candidate group adduct_type does not match "
                    "the corresponding adduct rule. "
                    f"expected={expected_adduct_type}, "
                    f"actual={group.adduct_type}"
                )

            if group.num_candidates > 0:
                if np.any(group.node_indices >= self.num_nodes):
                    raise ValueError(
                        "formula candidate group contains out-of-range "
                        "node indices."
                    )
        
    def _ensure_formula_candidate_group(
        self,
        adduct_type: Adduct | str,
    ) -> _FragmentIonFormulaCandidateGroup:
        """Build one formula candidate group lazily."""

        adduct_rule_index = self.get_adduct_rule_index_by_adduct_type(adduct_type)
        adduct_rule = self.fragment_ion_adduct_rule_set.adduct_rules[adduct_rule_index]

        normalized_adduct_type = adduct_rule.adduct_type
        groups = self.formula_candidate_groups

        if normalized_adduct_type in groups:
            return groups[normalized_adduct_type]

        group = _FragmentIonFormulaCandidateGroup.from_fragment_ion_tree(
            fragment_ion_tree=self,
            adduct_rule_index=adduct_rule_index,
        )

        groups[normalized_adduct_type] = group

        return group