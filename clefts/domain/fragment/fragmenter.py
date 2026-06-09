from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
import json
from pathlib import Path

from ...libs.mmkit.mmkit import Adduct, Compound

from .tree import *
from .ion_tree import *
from .pathway import *


@dataclass(frozen=True)
class Fragmenter:
    fragment_ion_tree_builder: FragmentIonTreeBuilder

    @property
    def adduct_types(self) -> Tuple[Adduct, ...]:
        return self.adduct_rule_set.adduct_types
    
    @property
    def adduct_rule_set(self) -> FragmentIonAdductRuleSet:
        return self.fragment_ion_tree_builder.fragment_ion_adduct_rule_set
    
    @property
    def tree_max_depth(self) -> int:
        return self.fragment_ion_tree_builder.max_depth
    
    @property
    def cleavage_pattern_set(self) -> Any:
        return self.fragment_ion_tree_builder.cleavage_pattern_set
    
    @property
    def name(self) -> str:
        return self.fragment_ion_tree_builder.name
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_ion_tree_builder": self.fragment_ion_tree_builder.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> Fragmenter:
        return cls(
            fragment_ion_tree_builder=FragmentIonTreeBuilder.from_dict(
                data["fragment_ion_tree_builder"]
            )
        )

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
    ) -> Fragmenter:
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def build_fragment_tree(
        self,
        compound: Compound,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentTree:
        return self.fragment_ion_tree_builder.build_fragment_tree(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info,
        )
    
    def build_fragment_ion_tree(
        self,
        compound: Compound,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentIonTree:
        return self.fragment_ion_tree_builder.build(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info,
        )

    def copy(self) -> Fragmenter:
        """Return a copy of this fragmenter."""
        return Fragmenter(
            fragment_ion_tree_builder=self.fragment_ion_tree_builder.copy(),
        )