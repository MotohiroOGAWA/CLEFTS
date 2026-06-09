from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple
import json

from ....libs.mmkit.mmkit import Formula

from .FragmentPathwayNode import FragmentPathwayNode
from .FragmentPathway import FragmentPathway


@dataclass(frozen=True)
class FragmentPathwayGroup:
    pathways: Tuple[FragmentPathway, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pathways, tuple):
            raise TypeError("pathways must be a tuple.")

        if not all(isinstance(pathway, FragmentPathway) for pathway in self.pathways):
            raise TypeError(
                "pathways must contain only FragmentPathway instances."
            )

    def __repr__(self) -> str:
        return (
            "FragmentPathwayGroup("
            f"pathways={self.pathways}"
            ")"
        )

    def __str__(self) -> str:
        return self.to_json_str()

    def __len__(self) -> int:
        return len(self.pathways)

    def __iter__(self):
        return iter(self.pathways)

    def __getitem__(self, index: int) -> FragmentPathway:
        return self.pathways[index]

    @classmethod
    def empty(cls) -> FragmentPathwayGroup:
        return cls(pathways=())

    @classmethod
    def from_list_of_pathways(
        cls,
        pathways: Iterable[FragmentPathway],
    ) -> FragmentPathwayGroup:
        return cls(pathways=tuple(pathways))

    @property
    def is_empty(self) -> bool:
        return len(self.pathways) == 0

    @property
    def formulas(self) -> Tuple[Formula, ...]:
        return tuple(
            pathway.formula
            for pathway in self.pathways
        )

    @property
    def precursor_nodes(self) -> Tuple[FragmentPathwayNode, ...]:
        return tuple(
            pathway.precursor_node
            for pathway in self.pathways
            if pathway.precursor_node is not None
        )

    @property
    def terminal_nodes(self) -> Tuple[FragmentPathwayNode, ...]:
        return tuple(
            pathway.terminal_node
            for pathway in self.pathways
        )

    def to_list(self) -> list[Any]:
        return [
            pathway.to_list()
            for pathway in self.pathways
        ]

    @classmethod
    def from_list(cls, data: list[Any]) -> FragmentPathwayGroup:
        if not isinstance(data, list):
            raise TypeError("FragmentPathwayGroup data must be a list.")

        return cls(
            pathways=tuple(
                FragmentPathway.from_list(pathway_data)
                for pathway_data in data
            )
        )

    def to_json_str(self) -> str:
        return json.dumps(
            self.to_list(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def parse(cls, text: str) -> FragmentPathwayGroup:
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                f"FragmentPathwayGroup JSON must be a list: {text}"
            )

        return cls.from_list(data)

    @classmethod
    def from_json_str(cls, text: str) -> FragmentPathwayGroup:
        return cls.parse(text)