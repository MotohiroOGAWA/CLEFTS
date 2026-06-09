from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
import json


@dataclass(frozen=True)
class CleavageStep:
    cleavage_pattern_id: int
    reaction_id: int
    product_molecule_id: int
    reactant_indices: Tuple[int, ...]
    product_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reactant_indices, tuple):
            raise TypeError("reactant_indices must be a tuple.")
        if not isinstance(self.product_indices, tuple):
            raise TypeError("product_indices must be a tuple.")
        if not all(isinstance(idx, int) for idx in self.reactant_indices):
            raise TypeError("reactant_indices must contain only integers.")
        if not all(isinstance(idx, int) for idx in self.product_indices):
            raise TypeError("product_indices must contain only integers.")
        if not isinstance(self.cleavage_pattern_id, int):
            raise TypeError("cleavage_pattern_id must be an integer.")
        if not isinstance(self.reaction_id, int):
            raise TypeError("reaction_id must be an integer.")
        if not isinstance(self.product_molecule_id, int):
            raise TypeError("product_molecule_id must be an integer.")

    def __repr__(self) -> str:
        return (
            "CleavageStep("
            f"cleavage_pattern_id={self.cleavage_pattern_id}, "
            f"reaction_id={self.reaction_id}, "
            f"product_molecule_id={self.product_molecule_id}, "
            f"reactant_indices={self.reactant_indices}, "
            f"product_indices={self.product_indices}"
            ")"
        )

    def __str__(self) -> str:
        return self.to_json_str()

    def to_list(self) -> list[Any]:
        return [
            self.cleavage_pattern_id,
            self.reaction_id,
            self.product_molecule_id,
            list(self.reactant_indices),
            list(self.product_indices),
        ]

    @classmethod
    def from_list(cls, data: list[Any]) -> CleavageStep:
        if len(data) != 5:
            raise ValueError(
                "CleavageStep list must have 5 elements: "
                "[cleavage_pattern_id, reaction_id, product_molecule_id, "
                "reactant_indices, product_indices]."
            )

        return cls(
            cleavage_pattern_id=int(data[0]),
            reaction_id=int(data[1]),
            product_molecule_id=int(data[2]),
            reactant_indices=tuple(int(idx) for idx in data[3]),
            product_indices=tuple(int(idx) for idx in data[4]),
        )

    def to_json_str(self) -> str:
        return json.dumps(
            self.to_list(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def parse(cls, text: str) -> CleavageStep:
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                f"CleavageStep JSON must be a list: {text}"
            )

        return cls.from_list(data)

    @classmethod
    def from_json_str(cls, text: str) -> CleavageStep:
        return cls.parse(text)