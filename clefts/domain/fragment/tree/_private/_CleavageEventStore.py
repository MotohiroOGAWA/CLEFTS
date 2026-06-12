from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..CleavageEvent import CleavageEvent
from ..FragmentEdge import FragmentEdge


@dataclass(frozen=True)
class _CleavageEventStore:
    """Columnar storage for cleavage events.

    The array position is the local event index in FragmentTree.
    """

    cleavage_pattern_ids: np.ndarray
    reaction_ids: np.ndarray
    product_molecule_ids: np.ndarray
    event_ids: np.ndarray
    reactant_indices_strs: np.ndarray
    product_indices_strs: np.ndarray
    edge_event_indptr: np.ndarray

    def __post_init__(self) -> None:
        cleavage_pattern_ids = np.asarray(
            self.cleavage_pattern_ids,
            dtype=np.int64,
        )
        reaction_ids = np.asarray(self.reaction_ids, dtype=np.int64)
        product_molecule_ids = np.asarray(
            self.product_molecule_ids,
            dtype=np.int64,
        )
        reactant_indices_strs = np.asarray(
            self.reactant_indices_strs,
            dtype=object,
        )
        product_indices_strs = np.asarray(
            self.product_indices_strs,
            dtype=object,
        )
        event_ids = np.asarray(self.event_ids, dtype=np.int64)
        edge_event_indptr = np.asarray(
            self.edge_event_indptr,
            dtype=np.int64,
        )

        n_events = len(event_ids)

        arrays = {
            "cleavage_pattern_ids": cleavage_pattern_ids,
            "reaction_ids": reaction_ids,
            "product_molecule_ids": product_molecule_ids,
            "event_ids": event_ids,
            "reactant_indices_strs": reactant_indices_strs,
            "product_indices_strs": product_indices_strs,
        }

        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"{name} must be a 1D array.")

            if len(array) != n_events:
                raise ValueError(
                    f"{name} must have the same length as event_ids."
                )

        if edge_event_indptr.ndim != 1:
            raise ValueError("edge_event_indptr must be a 1D array.")

        if len(edge_event_indptr) == 0:
            raise ValueError("edge_event_indptr must not be empty.")

        if edge_event_indptr[0] != 0:
            raise ValueError("edge_event_indptr must start at 0.")

        if edge_event_indptr[-1] != n_events:
            raise ValueError(
                "edge_event_indptr last value must equal the number of events."
            )

        if not np.all(edge_event_indptr[1:] >= edge_event_indptr[:-1]):
            raise ValueError("edge_event_indptr must be non-decreasing.")

        object.__setattr__(
            self,
            "cleavage_pattern_ids",
            cleavage_pattern_ids,
        )
        object.__setattr__(self, "reaction_ids", reaction_ids)
        object.__setattr__(
            self,
            "product_molecule_ids",
            product_molecule_ids,
        )
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(
            self,
            "reactant_indices_strs",
            reactant_indices_strs,
        )
        object.__setattr__(
            self,
            "product_indices_strs",
            product_indices_strs,
        )
        object.__setattr__(self, "edge_event_indptr", edge_event_indptr)

    @property
    def num_events(self) -> int:
        return len(self.event_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_event_indptr) - 1

    @staticmethod
    def empty(num_edges: int) -> "_CleavageEventStore":
        return _CleavageEventStore(
            event_ids=np.asarray([], dtype=np.int64),
            cleavage_pattern_ids=np.asarray([], dtype=np.int64),
            reaction_ids=np.asarray([], dtype=np.int64),
            product_molecule_ids=np.asarray([], dtype=np.int64),
            reactant_indices_strs=np.asarray([], dtype=object),
            product_indices_strs=np.asarray([], dtype=object),
            edge_event_indptr=np.zeros(int(num_edges) + 1, dtype=np.int64),
        )

    @staticmethod
    def from_edges(
        edges: Tuple[FragmentEdge, ...],
    ) -> "_CleavageEventStore":
        edges = tuple(sorted(edges, key=lambda edge: edge.index))

        event_ids: list[int] = []
        cleavage_pattern_ids: list[int] = []
        reaction_ids: list[int] = []
        product_molecule_ids: list[int] = []
        reactant_indices_strs: list[str] = []
        product_indices_strs: list[str] = []

        edge_event_indptr = np.zeros(len(edges) + 1, dtype=np.int64)

        for edge_index, edge in enumerate(edges):
            events = tuple(sorted(edge.events, key=lambda event: event.index))

            for event in events:
                cleavage_pattern_ids.append(event.cleavage_pattern_id)
                reaction_ids.append(event.reaction_id)
                product_molecule_ids.append(event.product_molecule_id)
                event_ids.append(event.event_id)
                reactant_indices_strs.append(event.reactant_indices_str)
                product_indices_strs.append(event.product_indices_str)

            edge_event_indptr[edge_index + 1] = len(event_ids)

        return _CleavageEventStore(
            cleavage_pattern_ids=np.asarray(
                cleavage_pattern_ids,
                dtype=np.int64,
            ),
            reaction_ids=np.asarray(reaction_ids, dtype=np.int64),
            product_molecule_ids=np.asarray(
                product_molecule_ids,
                dtype=np.int64,
            ),
            event_ids=np.asarray(event_ids, dtype=np.int64),
            reactant_indices_strs=np.asarray(
                reactant_indices_strs,
                dtype=object,
            ),
            product_indices_strs=np.asarray(
                product_indices_strs,
                dtype=object,
            ),
            edge_event_indptr=edge_event_indptr,
        )

    def get_events(self, edge_index: int) -> Tuple[CleavageEvent, ...]:
        if not 0 <= edge_index < self.num_edges:
            raise IndexError(f"Invalid edge index: {edge_index}")

        start = int(self.edge_event_indptr[edge_index])
        end = int(self.edge_event_indptr[edge_index + 1])

        events: list[CleavageEvent] = []

        for local_event_index, event_index in enumerate(range(start, end)):
            events.append(
                CleavageEvent(
                    index=int(local_event_index),
                    event_id=int(self.event_ids[event_index]),
                    cleavage_pattern_id=int(
                        self.cleavage_pattern_ids[event_index]
                    ),
                    reaction_id=int(self.reaction_ids[event_index]),
                    product_molecule_id=int(
                        self.product_molecule_ids[event_index]
                    ),
                    reactant_indices=json.loads(
                        str(self.reactant_indices_strs[event_index])
                    ),
                    product_indices=json.loads(
                        str(self.product_indices_strs[event_index])
                    ),
                )
            )

        return tuple(events)

    def copy(self) -> "_CleavageEventStore":
        return _CleavageEventStore(
            cleavage_pattern_ids=self.cleavage_pattern_ids.copy(),
            reaction_ids=self.reaction_ids.copy(),
            product_molecule_ids=self.product_molecule_ids.copy(),
            event_ids=self.event_ids.copy(),
            reactant_indices_strs=self.reactant_indices_strs.copy(),
            product_indices_strs=self.product_indices_strs.copy(),
            edge_event_indptr=self.edge_event_indptr.copy(),
        )