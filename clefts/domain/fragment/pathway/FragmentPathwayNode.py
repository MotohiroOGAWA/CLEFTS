from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import json

from ....libs.mmkit.mmkit import Compound, Adduct


@dataclass(frozen=True)
class FragmentPathwayNode:
    smiles: str
    precursor_adduct_type: Optional[Adduct] = None

    def __post_init__(self) -> None:
        if not isinstance(self.smiles, str):
            raise TypeError("smiles must be a string.")

        if (
            self.precursor_adduct_type is not None
            and not isinstance(self.precursor_adduct_type, Adduct)
        ):
            raise TypeError(
                "precursor_adduct_type must be an Adduct instance or None."
            )

    @property
    def is_precursor(self) -> bool:
        return self.precursor_adduct_type is not None

    def __repr__(self) -> str:
        return (
            "FragmentPathwayNode("
            f"smiles={self.smiles!r}, "
            f"precursor_adduct_type={self.precursor_adduct_type}"
            ")"
        )

    def __str__(self) -> str:
        return self.to_json_str()

    def to_list(self) -> list[Any]:
        return [
            self.smiles,
            None if self.precursor_adduct_type is None
            else str(self.precursor_adduct_type),
        ]

    @classmethod
    def from_list(cls, data: list[Any]) -> FragmentPathwayNode:
        if not isinstance(data, list):
            raise TypeError("FragmentPathwayNode data must be a list.")

        if len(data) != 2:
            raise ValueError(
                "FragmentPathwayNode list must have 2 elements: "
                "[smiles, precursor_adduct_type]."
            )

        smiles, precursor_adduct_text = data

        precursor_adduct_type = (
            None
            if precursor_adduct_text is None
            else Adduct.parse(str(precursor_adduct_text))
        )

        return cls(
            smiles=str(smiles),
            precursor_adduct_type=precursor_adduct_type,
        )

    def to_json_str(self) -> str:
        return json.dumps(
            self.to_list(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def parse(cls, text: str) -> FragmentPathwayNode:
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                f"FragmentPathwayNode JSON must be a list: {text}"
            )

        return cls.from_list(data)

    @classmethod
    def from_json_str(cls, text: str) -> FragmentPathwayNode:
        return cls.parse(text)

    def to_compound(self) -> Compound:
        return Compound.from_smiles(self.smiles)
    
    def copy(self) -> FragmentPathwayNode:
        return FragmentPathwayNode(
            smiles=self.smiles,
            precursor_adduct_type=self.precursor_adduct_type.copy() if self.precursor_adduct_type is not None else None,
        )