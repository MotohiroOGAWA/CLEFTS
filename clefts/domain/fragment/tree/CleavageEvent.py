from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class CleavageEvent:
    """One cleavage event associated with a FragmentEdge.

    index:
        Local event index in the FragmentTree.

    cleavage_pattern_id:
        ID of the cleavage pattern used for this event.

    reaction_id:
        ID of the reaction execution.
        Events generated from the same reaction share the same reaction_id.

    product_molecule_id:
        Product molecule index within the reaction result.

    event_id:
        Event ID within the database event ID.

    reactant_indices:
        Atom indices in the reactant/source fragment.

    product_indices:
        Atom indices in the product/target fragment.
    """

    index: int
    cleavage_pattern_id: int
    reaction_id: int
    product_molecule_id: int
    event_id: int
    reactant_indices: Tuple[int, ...]
    product_indices: Tuple[int, ...]

    def __init__(
        self,
        index: int,
        cleavage_pattern_id: int,
        reaction_id: int,
        product_molecule_id: int,
        event_id: int,
        reactant_indices: Iterable[int],
        product_indices: Iterable[int],
    ) -> None:
        object.__setattr__(self, "index", int(index))
        object.__setattr__(self, "cleavage_pattern_id", int(cleavage_pattern_id))
        object.__setattr__(self, "reaction_id", int(reaction_id))
        object.__setattr__(self, "product_molecule_id", int(product_molecule_id))
        object.__setattr__(self, "event_id", int(event_id))
        object.__setattr__(
            self,
            "reactant_indices",
            tuple(int(i) for i in reactant_indices),
        )
        object.__setattr__(
            self,
            "product_indices",
            tuple(int(i) for i in product_indices),
        )

        self._validate()

    def _validate(self) -> None:

        if self.cleavage_pattern_id < 0:
            raise ValueError("cleavage_pattern_id must be non-negative.")

        if self.reaction_id < 0:
            raise ValueError("reaction_id must be non-negative.")

        if self.product_molecule_id < 0:
            raise ValueError("product_molecule_id must be non-negative.")

    @property
    def reactant_indices_str(self) -> str:
        """JSON string representation of reactant_indices."""
        return json.dumps(self.reactant_indices, separators=(",", ":"))

    @property
    def product_indices_str(self) -> str:
        """JSON string representation of product_indices."""
        return json.dumps(self.product_indices, separators=(",", ":"))

    def to_record(self) -> tuple[int, int, int, int, int, str, str]:
        """Return a database-friendly record tuple.

        Returns
        -------
        tuple
            (
                index,
                cleavage_pattern_id,
                reaction_id,
                product_molecule_id,
                event_id,
                reactant_indices_str,
                product_indices_str,
            )
        """
        return (
            self.index,
            self.cleavage_pattern_id,
            self.reaction_id,
            self.product_molecule_id,
            self.event_id,
            self.reactant_indices_str,
            self.product_indices_str,
        )

    def __repr__(self) -> str:
        return "CleavageEvent" + self.__str__()

    def __str__(self) -> str:
        return (
            f"(index={self.index}; "
            f"cleavage_pattern_id={self.cleavage_pattern_id}; "
            f"reaction_id={self.reaction_id}; "
            f"product_molecule_id={self.product_molecule_id}; "
            f"event_id={self.event_id}; "
            f"reactant_indices={self.reactant_indices}; "
            f"product_indices={self.product_indices})"
        )

    def copy(self) -> "CleavageEvent":
        """Return a copy of this cleavage event."""
        return CleavageEvent(
            index=self.index,
            cleavage_pattern_id=self.cleavage_pattern_id,
            reaction_id=self.reaction_id,
            product_molecule_id=self.product_molecule_id,
            event_id=self.event_id,
            reactant_indices=self.reactant_indices,
            product_indices=self.product_indices,
        )