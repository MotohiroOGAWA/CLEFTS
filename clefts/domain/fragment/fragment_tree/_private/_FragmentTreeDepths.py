from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ._FragmentEdgeStore import _FragmentEdgeStore
from ._FragmentTreeAdjacency import _FragmentTreeAdjacency


@dataclass(frozen=True, init=False)
class _FragmentTreeDepths:
    """Node-depth values for a fragment tree."""

    _node_depths: np.ndarray

    def __init__(self, node_depths: np.ndarray):
        """Create validated node-depth storage."""
        node_depths = np.asarray(node_depths, dtype=np.int32)
        assert node_depths.ndim == 1, "node_depths must be a 1D array."
        object.__setattr__(self, "_node_depths", node_depths)

    @property
    def node_depths(self) -> np.ndarray:
        """Node-depth array."""
        return self._node_depths

    @staticmethod
    def build(
        num_nodes: int,
        edge_store: _FragmentEdgeStore,
        adjacency: _FragmentTreeAdjacency,
    ) -> "_FragmentTreeDepths":
        """Compute minimal edge distance from root nodes to every node."""
        depth = np.full(int(num_nodes), -1, dtype=np.int32)
        if num_nodes == 0:
            return _FragmentTreeDepths(depth)

        _, in_indptr = adjacency.in_edge_csr
        roots = np.flatnonzero((in_indptr[1:] - in_indptr[:-1]) == 0).astype(np.int32, copy=False)
        if roots.size == 0:
            roots = np.asarray([0], dtype=np.int32)

        out_edge_ids, out_indptr = adjacency.out_edge_csr
        queue = deque()
        for root_id in roots:
            root_id = int(root_id)
            depth[root_id] = 0
            queue.append(root_id)

        edge_index = edge_store.edge_index
        while queue:
            source_id = queue.popleft()
            next_depth = int(depth[source_id]) + 1
            start = int(out_indptr[source_id])
            end = int(out_indptr[source_id + 1])
            for edge_id in out_edge_ids[start:end]:
                target_id = int(edge_index[int(edge_id), 1])
                if depth[target_id] == -1:
                    depth[target_id] = next_depth
                    queue.append(target_id)

        return _FragmentTreeDepths(depth)

    def get_nodes_by_depth(self) -> Dict[int, np.ndarray]:
        """Group node IDs by minimal depth from root nodes."""
        out: Dict[int, np.ndarray] = {}
        for depth in sorted(set(int(d) for d in self._node_depths if d >= 0)):
            out[depth] = np.flatnonzero(self._node_depths == depth).astype(np.int32, copy=False)
        return out

    def copy(self) -> "_FragmentTreeDepths":
        """Return copied depth storage."""
        return _FragmentTreeDepths(self._node_depths.copy())
