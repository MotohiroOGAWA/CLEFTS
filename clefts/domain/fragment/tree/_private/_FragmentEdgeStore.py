from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..FragmentEdge import FragmentEdge
from ._CleavageEventStore import _CleavageEventStore


@dataclass(frozen=True)
class _FragmentEdgeStore:
    """Array-backed storage for fragment edges.

    The local edge index is the array position.

    edge_ids:
        Database edge IDs.

    source_indices:
        Local source node indices in FragmentTree.

    target_indices:
        Local target node indices in FragmentTree.

    source_ids:
        Database source fragment IDs.

    target_ids:
        Database target fragment IDs.
    """

    edge_ids: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    source_ids: np.ndarray
    target_ids: np.ndarray
    event_store: _CleavageEventStore

    def __post_init__(self) -> None:
        edge_ids = np.asarray(self.edge_ids, dtype=np.int64)
        source_indices = np.asarray(self.source_indices, dtype=np.int64)
        target_indices = np.asarray(self.target_indices, dtype=np.int64)
        source_ids = np.asarray(self.source_ids, dtype=np.int64)
        target_ids = np.asarray(self.target_ids, dtype=np.int64)

        if edge_ids.ndim != 1:
            raise ValueError("edge_ids must be a 1D array.")

        if source_indices.ndim != 1:
            raise ValueError("source_indices must be a 1D array.")

        if target_indices.ndim != 1:
            raise ValueError("target_indices must be a 1D array.")

        if source_ids.ndim != 1:
            raise ValueError("source_ids must be a 1D array.")

        if target_ids.ndim != 1:
            raise ValueError("target_ids must be a 1D array.")

        num_edges = len(edge_ids)

        if len(source_indices) != num_edges:
            raise ValueError("source_indices must match edge_ids.")

        if len(target_indices) != num_edges:
            raise ValueError("target_indices must match edge_ids.")

        if len(source_ids) != num_edges:
            raise ValueError("source_ids must match edge_ids.")

        if len(target_ids) != num_edges:
            raise ValueError("target_ids must match edge_ids.")

        if not isinstance(self.event_store, _CleavageEventStore):
            raise TypeError("event_store must be a _CleavageEventStore.")

        if self.event_store.num_edges != num_edges:
            raise ValueError("event_store must match edge_ids.")

        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "target_indices", target_indices)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "target_ids", target_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_ids)

    @property
    def edge_index(self) -> np.ndarray:
        """Return source/target node indices as shape [E, 2].

        This is computed from source_indices and target_indices.
        """
        return np.stack(
            [self.source_indices, self.target_indices],
            axis=1,
        ).astype(np.int64, copy=False)

    def get_edge(self, index: int) -> FragmentEdge:
        if not 0 <= index < self.num_edges:
            raise IndexError(f"Invalid edge index: {index}")

        return FragmentEdge(
            index=int(index),
            id=int(self.edge_ids[index]),
            source_index=int(self.source_indices[index]),
            target_index=int(self.target_indices[index]),
            source_id=int(self.source_ids[index]),
            target_id=int(self.target_ids[index]),
            events=self.event_store.get_events(index),
        )

    def copy(self) -> "_FragmentEdgeStore":
        return _FragmentEdgeStore(
            edge_ids=self.edge_ids.copy(),
            source_indices=self.source_indices.copy(),
            target_indices=self.target_indices.copy(),
            source_ids=self.source_ids.copy(),
            target_ids=self.target_ids.copy(),
            event_store=self.event_store.copy(),
        )