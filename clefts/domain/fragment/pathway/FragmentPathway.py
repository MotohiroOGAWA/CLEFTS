from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union
import json

from ....libs.mmkit.mmkit import Adduct, Formula

from .FragmentPathwayNode import FragmentPathwayNode
from .FragmentPathwayEdge import FragmentPathwayEdge


FragmentPathwayElement = Union[
    FragmentPathwayNode,
    FragmentPathwayEdge,
]


@dataclass(frozen=True)
class FragmentPathway:
    elements: Tuple[FragmentPathwayElement, ...]
    adduct: Adduct
    _formula: Optional[Formula] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.elements, tuple):
            raise TypeError("elements must be a tuple.")

        if len(self.elements) == 0:
            raise ValueError("elements must not be empty.")

        if len(self.elements) % 2 != 1:
            raise ValueError(
                "FragmentPathway elements must have an odd number of items: "
                "node, edge, node, edge, ..., node."
            )

        for i, element in enumerate(self.elements):
            if i % 2 == 0:
                if not isinstance(element, FragmentPathwayNode):
                    raise TypeError(
                        f"Expected FragmentPathwayNode at position {i}, "
                        f"got {type(element)}."
                    )
            else:
                if not isinstance(element, FragmentPathwayEdge):
                    raise TypeError(
                        f"Expected FragmentPathwayEdge at position {i}, "
                        f"got {type(element)}."
                    )

        if not isinstance(self.adduct, Adduct):
            raise TypeError("adduct must be an Adduct instance.")

    def __repr__(self) -> str:
        return (
            "FragmentPathway("
            f"elements={self.elements}, "
            f"adduct={self.adduct}"
            ")"
        )

    def __str__(self) -> str:
        return self.to_json_str()

    def __len__(self) -> int:
        """Return the number of nodes in the pathway."""
        return(len(self.elements) + 1) // 2  # Number of nodes in the pathway

    @property
    def nodes(self) -> Tuple[FragmentPathwayNode, ...]:
        return tuple(
            element
            for i, element in enumerate(self.elements)
            if i % 2 == 0
        )

    @property
    def edges(self) -> Tuple[FragmentPathwayEdge, ...]:
        return tuple(
            element
            for i, element in enumerate(self.elements)
            if i % 2 == 1
        )
    
    @property
    def root_node(self) -> FragmentPathwayNode:
        return self.elements[0]

    @property
    def terminal_node(self) -> FragmentPathwayNode:
        return self.elements[-1]

    @property
    def precursor_node(self) -> Optional[FragmentPathwayNode]:
        for node in self.nodes:
            if node.is_precursor:
                return node
        return None

    def get_node(self, index: int) -> FragmentPathwayNode:
        return self.nodes[index]

    def get_edge(self, index: int) -> FragmentPathwayEdge:
        return self.edges[index]

    @property
    def formula(self) -> Formula:
        """Calculate the adducted formula of the terminal node.

        This assumes FragmentPathwayNode has smiles.
        """
        if self._formula is None:
            compound = self.terminal_node.to_compound()
            formula = self.adduct.apply_to_formula(compound.formula).normalized
            object.__setattr__(self, "_formula", formula)

        return self._formula

    def to_list(self) -> list[Any]:
        return [
            [
                element.to_list()
                for element in self.elements
            ],
            str(self.adduct),
        ]

    @classmethod
    def from_list(cls, data: list[Any]) -> FragmentPathway:
        if not isinstance(data, list):
            raise TypeError("FragmentPathway data must be a list.")

        if len(data) != 2:
            raise ValueError(
                "FragmentPathway list must have 2 elements: "
                "[elements, adduct]."
            )

        elements_data, adduct_text = data

        if not isinstance(elements_data, list):
            raise TypeError("elements must be a list.")

        if len(elements_data) == 0:
            raise ValueError("elements must not be empty.")

        if len(elements_data) % 2 != 1:
            raise ValueError(
                "elements must have an odd number of items: "
                "node, edge, node, edge, ..., node."
            )

        elements: list[FragmentPathwayElement] = []

        for i, element_data in enumerate(elements_data):
            if not isinstance(element_data, list):
                raise TypeError(
                    f"Element data at position {i} must be a list."
                )

            if i % 2 == 0:
                elements.append(
                    FragmentPathwayNode.from_list(element_data)
                )
            else:
                elements.append(
                    FragmentPathwayEdge.from_list(element_data)
                )

        return cls(
            elements=tuple(elements),
            adduct=Adduct.parse(str(adduct_text)),
        )

    def to_json_str(self) -> str:
        return json.dumps(
            self.to_list(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def parse(cls, text: str) -> FragmentPathway:
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                f"FragmentPathway JSON must be a list: {text}"
            )

        return cls.from_list(data)

    @classmethod
    def from_json_str(cls, text: str) -> FragmentPathway:
        return cls.parse(text)