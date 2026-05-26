from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..FragmentEdge import FragmentEdge
from ._CleavageEventStore import _CleavageEventStore


@dataclass(frozen=True, init=False)
class _FragmentEdgeStore:
    """Array-backed storage for directed fragment edges."""

    _edge_index: np.ndarray
    _event_store: _CleavageEventStore

    def __init__(self, edge_index: np.ndarray, event_store: _CleavageEventStore):
        """Create edge storage from edge-index and cleavage-event storage."""
        edge_index = np.asarray(edge_index, dtype=np.int32)
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2, "edge_index must have shape [E, 2]."
        assert isinstance(event_store, _CleavageEventStore), "event_store must be a _CleavageEventStore."
        assert len(event_store.edge_event_indptr) == edge_index.shape[0] + 1, "event_store must match edge_index."
        object.__setattr__(self, "_edge_index", edge_index)
        object.__setattr__(self, "_event_store", event_store)

    @property
    def edge_index(self) -> np.ndarray:
        """Directed edge array with shape ``[E, 2]``."""
        return self._edge_index

    @property
    def event_store(self) -> _CleavageEventStore:
        """Cleavage-event storage for edges."""
        return self._event_store

    @property
    def num_edges(self) -> int:
        """Number of directed edges."""
        return self._edge_index.shape[0]

    def get_edge(self, edge_id: int) -> FragmentEdge:
        """Return an edge object by ID."""
        assert 0 <= edge_id < self.num_edges, "Invalid edge ID."
        return FragmentEdge(
            id=int(edge_id),
            source_id=int(self._edge_index[edge_id, 0]),
            target_id=int(self._edge_index[edge_id, 1]),
            events=self._event_store.get_events(edge_id),
        )

    def copy(self) -> "_FragmentEdgeStore":
        """Return copied edge storage."""
        return _FragmentEdgeStore(
            edge_index=self.edge_index.copy(),
            event_store=self.event_store.copy(),
        )
