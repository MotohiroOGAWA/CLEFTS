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
    def shortest(self) -> FragmentPathwayGroup:
        """Return a group containing only the shortest pathways."""

        if self.is_empty:
            return self.empty()

        min_depth = min(
            len(pathway)
            for pathway in self.pathways
        )

        return self.__class__(
            pathways=tuple(
                pathway
                for pathway in self.pathways
                if len(pathway) == min_depth
            )
        )


    @property
    def with_precursor(self) -> FragmentPathwayGroup:
        """Return a group containing only pathways that have a precursor node."""

        return self.__class__(
            pathways=tuple(
                pathway
                for pathway in self.pathways
                if pathway.has_precursor_node
            )
        )

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
    def from_list(cls, data: Iterable[Any]) -> FragmentPathwayGroup:
        return cls(
            pathways=tuple(
                item if isinstance(item, FragmentPathway)
                else FragmentPathway.from_list(item)
                for item in data
            )
        )

    def to_pathways(self) -> Tuple[FragmentPathway, ...]:
        return tuple(p.copy() for p in self.pathways)

    @classmethod
    def from_pathways(cls, fragment_pathways: Iterable[FragmentPathway]) -> FragmentPathwayGroup:
        return cls(
            pathways=tuple(fragment_pathways)
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