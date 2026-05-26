from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..CleavageEvent import CleavageEvent
from ..FragmentEdge import FragmentEdge


@dataclass(frozen=True, init=False)
class _CleavageEventStore:
    """Columnar storage for cleavage events attached to fragment edges."""

    _cleavage_pattern_ids: np.ndarray
    _react_indices_strs: np.ndarray
    _prod_indices_strs: np.ndarray
    _edge_event_indptr: np.ndarray

    def __init__(
        self,
        cleavage_pattern_ids: np.ndarray,
        react_indices_strs: np.ndarray,
        prod_indices_strs: np.ndarray,
        edge_event_indptr: np.ndarray,
        num_edges: int,
    ):
        """Create event storage from column arrays and an edge pointer array."""
        cleavage_pattern_ids = np.asarray(cleavage_pattern_ids, dtype=np.int32)
        react_indices_strs = np.asarray(react_indices_strs, dtype=object)
        prod_indices_strs = np.asarray(prod_indices_strs, dtype=object)
        edge_event_indptr = np.asarray(edge_event_indptr, dtype=np.int64)

        n_events = len(cleavage_pattern_ids)
        assert cleavage_pattern_ids.ndim == 1, "cleavage_pattern_ids must be a 1D array."
        assert react_indices_strs.ndim == 1, "react_indices_strs must be a 1D array."
        assert prod_indices_strs.ndim == 1, "prod_indices_strs must be a 1D array."
        assert len(react_indices_strs) == n_events, "react_indices_strs must match cleavage_pattern_ids."
        assert len(prod_indices_strs) == n_events, "prod_indices_strs must match cleavage_pattern_ids."
        assert edge_event_indptr.ndim == 1, "edge_event_indptr must be a 1D array."
        assert edge_event_indptr.shape[0] == int(num_edges) + 1, "edge_event_indptr must be length E+1."
        assert edge_event_indptr[0] == 0, "edge_event_indptr must start at 0."
        assert edge_event_indptr[-1] == n_events, "edge_event_indptr last element must equal the number of events."
        assert np.all(edge_event_indptr[1:] >= edge_event_indptr[:-1]), "edge_event_indptr must be non-decreasing."

        object.__setattr__(self, "_cleavage_pattern_ids", cleavage_pattern_ids)
        object.__setattr__(self, "_react_indices_strs", react_indices_strs)
        object.__setattr__(self, "_prod_indices_strs", prod_indices_strs)
        object.__setattr__(self, "_edge_event_indptr", edge_event_indptr)

    @staticmethod
    def empty(num_edges: int) -> "_CleavageEventStore":
        """Create empty event storage for ``num_edges`` edges."""
        return _CleavageEventStore(
            cleavage_pattern_ids=np.asarray([], dtype=np.int32),
            react_indices_strs=np.asarray([], dtype=object),
            prod_indices_strs=np.asarray([], dtype=object),
            edge_event_indptr=np.zeros(int(num_edges) + 1, dtype=np.int64),
            num_edges=int(num_edges),
        )

    @staticmethod
    def from_edges(edges: Tuple[FragmentEdge, ...]) -> "_CleavageEventStore":
        """Create event storage from edge objects.

        Event IDs are local to each edge when events are reconstructed, so the
        stored columns keep only the cleavage pattern and atom-index mappings.
        """
        cleavage_pattern_ids = []
        react_indices_strs = []
        prod_indices_strs = []
        edge_event_indptr = np.zeros(len(edges) + 1, dtype=np.int64)

        for edge_index, edge in enumerate(edges):
            for event in edge.events:
                cleavage_pattern_ids.append(event.cleavage_pattern_id)
                react_indices_strs.append(event.react_indices_str)
                prod_indices_strs.append(event.prod_indices_str)
            edge_event_indptr[edge_index + 1] = len(cleavage_pattern_ids)

        return _CleavageEventStore(
            cleavage_pattern_ids=np.asarray(cleavage_pattern_ids, dtype=np.int32),
            react_indices_strs=np.asarray(react_indices_strs, dtype=object),
            prod_indices_strs=np.asarray(prod_indices_strs, dtype=object),
            edge_event_indptr=edge_event_indptr,
            num_edges=len(edges),
        )

    @property
    def cleavage_pattern_ids(self) -> np.ndarray:
        """Cleavage pattern ID for each stored event."""
        return self._cleavage_pattern_ids

    @property
    def react_indices_strs(self) -> np.ndarray:
        """JSON string of source atom indices for each stored event."""
        return self._react_indices_strs

    @property
    def prod_indices_strs(self) -> np.ndarray:
        """JSON string of product atom indices for each stored event."""
        return self._prod_indices_strs

    @property
    def edge_event_indptr(self) -> np.ndarray:
        """Pointer array into event columns for each edge."""
        return self._edge_event_indptr

    def get_events(self, edge_id: int) -> Tuple[CleavageEvent, ...]:
        """Return events associated with an edge.

        ``event_id`` is assigned from zero within the requested edge.
        """
        assert 0 <= edge_id < len(self._edge_event_indptr) - 1, "Invalid edge ID."
        start = int(self._edge_event_indptr[edge_id])
        end = int(self._edge_event_indptr[edge_id + 1])
        events = []
        for local_event_id, flat_id in enumerate(range(start, end)):
            events.append(
                CleavageEvent(
                    event_id=local_event_id,
                    cleavage_pattern_id=int(self._cleavage_pattern_ids[flat_id]),
                    react_indices=json.loads(str(self._react_indices_strs[flat_id])),
                    prod_indices=json.loads(str(self._prod_indices_strs[flat_id])),
                )
            )
        return tuple(events)

    def copy(self) -> "_CleavageEventStore":
        """Return copied event storage."""
        return _CleavageEventStore(
            cleavage_pattern_ids=self.cleavage_pattern_ids.copy(),
            react_indices_strs=self.react_indices_strs.copy(),
            prod_indices_strs=self.prod_indices_strs.copy(),
            edge_event_indptr=self.edge_event_indptr.copy(),
            num_edges=len(self.edge_event_indptr) - 1,
        )
