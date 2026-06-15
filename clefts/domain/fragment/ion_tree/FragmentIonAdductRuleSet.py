from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import json

from ....libs.mmkit.mmkit import Adduct
from .FragmentIonAdductRule import FragmentIonAdductRule


@dataclass(frozen=True)
class FragmentIonAdductRuleSet:
    """Collection of fragment ion adduct rules.

    This class corresponds to one JSON file and one ML model setting.
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
    ) -> FragmentIonAdductRuleSet:
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

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> FragmentIonAdductRuleSet:
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_json(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
            )

    @property
    def rules(self) -> Tuple[FragmentIonAdductRule, ...]:
        """Alias for adduct_rules."""

        return self.adduct_rules

    @property
    def num_classes(self) -> int:
        """Number of adduct classes."""

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
        """Mapping from class index to adduct type."""

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
        """Return rule corresponding to the class index."""

        if not isinstance(index, int):
            raise TypeError("index must be an int.")

        if not 0 <= index < self.num_classes:
            raise IndexError(
                f"index must be in [0, {self.num_classes}). "
                f"Got {index}."
            )

        return self.adduct_rules[index]

    def get_ion_shift_adducts_by_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> Tuple[Adduct, ...]:
        """Return ion shift adducts for the given adduct type."""

        rule = self.get_rule_by_adduct_type(adduct_type)

        return tuple(
            ion_shift_rule.ion_shift
            for ion_shift_rule in rule.ion_shifts
        )

    def get_ion_shift_adducts_by_index(
        self,
        index: int,
    ) -> Tuple[Adduct, ...]:
        """Return ion shift adducts for the class index."""

        rule = self.get_rule_by_index(index)

        return tuple(
            ion_shift_rule.ion_shift
            for ion_shift_rule in rule.ion_shifts
        )

    def encode_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> int:
        """Convert adduct type to class index."""

        key = self._adduct_key(adduct_type)

        try:
            return self.adduct_type_to_index[key]
        except KeyError as exc:
            raise KeyError(f"Unknown adduct_type: {key}") from exc

    def decode_adduct_index(
        self,
        index: int,
    ) -> Adduct:
        """Convert class index to adduct type."""

        return self.get_rule_by_index(index).adduct_type

    def copy(self) -> FragmentIonAdductRuleSet:
        """Return a deep copy."""

        return FragmentIonAdductRuleSet(
            name=self.name,
            adduct_rules=tuple(
                rule.copy()
                for rule in self.adduct_rules
            ),
        )

    @staticmethod
    def _adduct_key(
        adduct_type: Adduct | str,
    ) -> str:
        if isinstance(adduct_type, Adduct):
            return adduct_type

        if isinstance(adduct_type, str):
            return Adduct.parse(adduct_type)

        raise TypeError(
            "adduct_type must be an Adduct or str. "
            f"Got {type(adduct_type).__name__}."
        )