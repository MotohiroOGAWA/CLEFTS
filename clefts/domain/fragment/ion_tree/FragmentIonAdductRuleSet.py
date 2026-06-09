from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import json

from ....libs.mmkit.mmkit import Adduct, Compound

from ..tree.FragmentTree import FragmentTree
from .FragmentIonAdductRule import FragmentIonAdductRule
from ._private._FragmentIonStateStore import _FragmentIonStateStore
from ._private._FragmentIonShiftStore import _FragmentIonShiftStore


@dataclass(frozen=True)
class FragmentIonAdductRuleSet:
    """Collection of FragmentIonAdductRule.

    This class corresponds to one JSON file and one ML model setting.

    Examples
    --------
    A positive rule set may contain:

    - [M+H]+
    - [M+Na]+
    - [M+NH4]+
    - [2M+H]+

    In that case, the ML model predicts one of these 4 adduct classes.
    """

    name: str
    adduct_rules: Tuple[FragmentIonAdductRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        if not isinstance(self.adduct_rules, tuple):
            raise TypeError("adduct_rules must be a tuple.")

        for rule in self.adduct_rules:
            if not isinstance(rule, FragmentIonAdductRule):
                raise TypeError(
                    "adduct_rules must contain only "
                    "FragmentIonAdductRule instances."
                )

        adduct_type_keys = [
            self._adduct_key(rule.adduct_type)
            for rule in self.adduct_rules
        ]

        duplicated_adduct_types = {
            adduct_type
            for adduct_type in adduct_type_keys
            if adduct_type_keys.count(adduct_type) > 1
        }

        if duplicated_adduct_types:
            raise ValueError(
                "adduct_rules contains duplicated adduct_type values: "
                f"{sorted(duplicated_adduct_types)}"
            )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "FragmentIonAdductRuleSet":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")

        return cls(
            name=data.get("name", ""),
            adduct_rules=tuple(
                FragmentIonAdductRule.from_dict(rule_data)
                for rule_data in data.get("adduct_rules", [])
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "adduct_rules": [
                rule.to_dict()
                for rule in self.adduct_rules
            ],
        }

    def to_json(self, path: str | Path) -> None:
        path = Path(path)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "FragmentIonAdductRuleSet":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(
            data,
        )


    @property
    def rules(self) -> Tuple[FragmentIonAdductRule, ...]:
        """Alias for adduct_rules."""
        return self.adduct_rules

    @property
    def num_classes(self) -> int:
        """Number of adduct classes for the ML model."""
        return len(self.adduct_rules)

    @property
    def adduct_types(self) -> Tuple[Adduct, ...]:
        """Adduct types in class-index order."""
        return tuple(
            rule.adduct_type
            for rule in self.adduct_rules
        )

    @property
    def adduct_type_to_index(self) -> Dict[str, int]:
        """Mapping from adduct type string to class index."""
        return {
            self._adduct_key(rule.adduct_type): index
            for index, rule in enumerate(self.adduct_rules)
        }

    @property
    def index_to_adduct_type(self) -> Dict[int, Adduct]:
        """Mapping from class index to Adduct."""
        return {
            index: rule.adduct_type
            for index, rule in enumerate(self.adduct_rules)
        }

    def get_rule_by_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> FragmentIonAdductRule:
        """Return rule corresponding to the given adduct type."""

        key = self._adduct_key(adduct_type)

        for rule in self.adduct_rules:
            if self._adduct_key(rule.adduct_type) == key:
                return rule

        raise KeyError(f"Unknown adduct_type: {key}")

    def get_rule_by_index(
        self,
        index: int,
    ) -> FragmentIonAdductRule:
        """Return rule corresponding to the ML class index."""

        if not isinstance(index, int):
            raise TypeError("index must be an int.")

        if not 0 <= index < self.num_classes:
            raise IndexError(
                f"index must be in [0, {self.num_classes}). "
                f"Got {index}."
            )

        return self.adduct_rules[index]

    def encode_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> int:
        """Convert adduct type to ML class index."""

        key = self._adduct_key(adduct_type)

        try:
            return self.adduct_type_to_index[key]
        except KeyError as exc:
            raise KeyError(f"Unknown adduct_type: {key}") from exc

    def decode_adduct_index(
        self,
        index: int,
    ) -> Adduct:
        """Convert ML class index to Adduct."""

        return self.get_rule_by_index(index).adduct_type

    def build_ion_state_store(
        self,
        fragment_tree: FragmentTree,
        *,
        fragment_compound_by_index: dict[int, Compound] | None = None,
    ) -> _FragmentIonStateStore:
        """Build node-wise unsaturation/radical state store."""

        if not isinstance(fragment_tree, FragmentTree):
            raise TypeError("fragment_tree must be a FragmentTree.")

        ion_states: list[tuple[int, int]] = []
        node_state_indptr: list[int] = [0]

        for node_index in range(fragment_tree.num_nodes):
            compound = self._get_fragment_compound(
                fragment_tree,
                node_index,
                fragment_compound_by_index=fragment_compound_by_index,
            )

            node_states: set[tuple[int, int]] = set()

            for adduct_rule in self.adduct_rules:
                if adduct_rule.is_ion_state_applicable_to_compound(
                    compound
                ):
                    node_states.add(adduct_rule.get_ion_state())

            ion_states.extend(sorted(node_states))
            node_state_indptr.append(len(ion_states))

        if ion_states:
            ion_state_array = np.asarray(ion_states, dtype=np.int16)
        else:
            ion_state_array = np.empty((0, 2), dtype=np.int16)

        return _FragmentIonStateStore(
            ion_states=ion_state_array,
            node_state_indptr=np.asarray(
                node_state_indptr,
                dtype=np.int64,
            ),
        )

    def build_ion_shift_store(
        self,
        fragment_tree: FragmentTree,
        *,
        fragment_compound_by_index: dict[int, Compound] | None = None,
    ) -> _FragmentIonShiftStore:
        """Build node-wise IonShiftRule applicability store."""

        if not isinstance(fragment_tree, FragmentTree):
            raise TypeError("fragment_tree must be a FragmentTree.")

        shift_rule_indices = self._build_shift_rule_indices()

        node_shift_rule_mask = np.zeros(
            (fragment_tree.num_nodes, len(shift_rule_indices)),
            dtype=bool,
        )

        for node_index in range(fragment_tree.num_nodes):
            compound = self._get_fragment_compound(
                fragment_tree,
                node_index,
                fragment_compound_by_index=fragment_compound_by_index,
            )

            for shift_rule_index, (
                adduct_rule_index,
                ion_shift_index,
            ) in enumerate(shift_rule_indices):
                adduct_rule = self.adduct_rules[int(adduct_rule_index)]
                ion_shift_rule = adduct_rule.ion_shifts[int(ion_shift_index)]

                node_shift_rule_mask[
                    node_index,
                    shift_rule_index,
                ] = adduct_rule.is_ion_shift_applicable_to_compound(
                    compound=compound,
                    ion_shift_rule=ion_shift_rule,
                )

        return _FragmentIonShiftStore(
            shift_rule_indices=shift_rule_indices,
            node_shift_rule_mask=node_shift_rule_mask,
        )
    
    def copy(self) -> FragmentIonAdductRuleSet:
        """Return a deep copy of this FragmentIonAdductRuleSet."""
        return FragmentIonAdductRuleSet(
            name=self.name,
            adduct_rules=tuple(
                rule.copy()
                for rule in self.adduct_rules
            ),
        )

    def _build_shift_rule_indices(self) -> np.ndarray:
        """Build shift_rule_index -> rule position table.

        Returns
        -------
        np.ndarray
            Shape is (num_shift_rules, 2).

            Column 0:
                adduct_rule_index.

            Column 1:
                ion_shift_index.
        """

        shift_rule_indices: list[tuple[int, int]] = []

        for adduct_rule_index, adduct_rule in enumerate(self.adduct_rules):
            for ion_shift_index, _ in enumerate(adduct_rule.ion_shifts):
                shift_rule_indices.append(
                    (
                        adduct_rule_index,
                        ion_shift_index,
                    )
                )

        if not shift_rule_indices:
            return np.empty((0, 2), dtype=np.int64)

        return np.asarray(shift_rule_indices, dtype=np.int64)

    @staticmethod
    def _adduct_key(adduct_type: Adduct | str) -> str:
        if isinstance(adduct_type, Adduct):
            return str(adduct_type)

        if isinstance(adduct_type, str):
            return str(Adduct.parse(adduct_type))

        raise TypeError(
            "adduct_type must be an Adduct or str. "
            f"Got {type(adduct_type).__name__}."
        )

    def _get_fragment_compound(
        self,
        fragment_tree: FragmentTree,
        node_index: int,
        *,
        fragment_compound_by_index: dict[int, Compound] | None = None,
    ) -> Compound:
        if fragment_compound_by_index is not None:
            compound = fragment_compound_by_index.get(node_index)

            if compound is not None:
                return compound

        node = fragment_tree.get_node(node_index)
        return Compound.from_smiles(node.smiles)