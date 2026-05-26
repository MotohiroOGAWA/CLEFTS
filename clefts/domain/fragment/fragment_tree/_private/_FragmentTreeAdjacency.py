from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True, init=False)
class _FragmentTreeAdjacency:
    """Incoming and outgoing edge adjacency in CSR form."""

    _in_edge_ids: np.ndarray
    _in_edge_indptr: np.ndarray
    _out_edge_ids: np.ndarray
    _out_edge_indptr: np.ndarray

    def __init__(
        self,
        in_edge_ids: np.ndarray,
        in_edge_indptr: np.ndarray,
        out_edge_ids: np.ndarray,
        out_edge_indptr: np.ndarray,
    ):
        """Create validated CSR adjacency arrays."""
        in_edge_ids = np.asarray(in_edge_ids, dtype=np.int32)
        in_edge_indptr = np.asarray(in_edge_indptr, dtype=np.int64)
        out_edge_ids = np.asarray(out_edge_ids, dtype=np.int32)
        out_edge_indptr = np.asarray(out_edge_indptr, dtype=np.int64)

        assert in_edge_indptr.shape == out_edge_indptr.shape, "Incoming and outgoing indptr arrays must have the same length."
        assert self._valid_csr_arrays(in_edge_ids, in_edge_indptr), "Invalid incoming-edge CSR arrays."
        assert self._valid_csr_arrays(out_edge_ids, out_edge_indptr), "Invalid outgoing-edge CSR arrays."

        object.__setattr__(self, "_in_edge_ids", in_edge_ids)
        object.__setattr__(self, "_in_edge_indptr", in_edge_indptr)
        object.__setattr__(self, "_out_edge_ids", out_edge_ids)
        object.__setattr__(self, "_out_edge_indptr", out_edge_indptr)

    @staticmethod
    def _valid_csr_arrays(edge_ids: np.ndarray, indptr: np.ndarray) -> bool:
        return (
            edge_ids.ndim == 1
            and indptr.ndim == 1
            and indptr.shape[0] >= 1
            and indptr[0] == 0
            and indptr[-1] == len(edge_ids)
            and np.all(indptr[1:] >= indptr[:-1])
        )

    @staticmethod
    def from_edge_index(edge_index: np.ndarray, num_nodes: int) -> "_FragmentTreeAdjacency":
        """Build incoming and outgoing edge adjacency from an edge-index array."""
        edge_index = np.asarray(edge_index, dtype=np.int32)
        num_nodes = int(num_nodes)
        num_edges = int(edge_index.shape[0])

        if num_edges == 0:
            empty_ids = np.asarray([], dtype=np.int32)
            empty_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
            return _FragmentTreeAdjacency(
                in_edge_ids=empty_ids,
                in_edge_indptr=empty_indptr,
                out_edge_ids=empty_ids,
                out_edge_indptr=empty_indptr,
            )

        src = edge_index[:, 0].astype(np.int32, copy=False)
        dst = edge_index[:, 1].astype(np.int32, copy=False)
        edge_ids = np.arange(num_edges, dtype=np.int32)

        order_in = np.argsort(dst, kind="mergesort")
        in_edge_ids = edge_ids[order_in]
        in_counts = np.bincount(dst[order_in], minlength=num_nodes).astype(np.int64)
        in_indptr = np.empty(num_nodes + 1, dtype=np.int64)
        in_indptr[0] = 0
        np.cumsum(in_counts, out=in_indptr[1:])

        order_out = np.argsort(src, kind="mergesort")
        out_edge_ids = edge_ids[order_out]
        out_counts = np.bincount(src[order_out], minlength=num_nodes).astype(np.int64)
        out_indptr = np.empty(num_nodes + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(out_counts, out=out_indptr[1:])

        return _FragmentTreeAdjacency(
            in_edge_ids=in_edge_ids,
            in_edge_indptr=in_indptr,
            out_edge_ids=out_edge_ids,
            out_edge_indptr=out_indptr,
        )


    @property
    def num_nodes(self) -> int:
        """Number of nodes inferred from CSR pointer arrays."""
        return len(self._in_edge_indptr) - 1

    @property
    def in_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Incoming edge adjacency as ``(edge_ids, indptr)``."""
        return self._in_edge_ids, self._in_edge_indptr

    @property
    def out_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Outgoing edge adjacency as ``(edge_ids, indptr)``."""
        return self._out_edge_ids, self._out_edge_indptr

    def get_in_edge_ids(self, node_id: int) -> np.ndarray:
        """Return incoming edge IDs for a node."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID."
        start = int(self._in_edge_indptr[node_id])
        end = int(self._in_edge_indptr[node_id + 1])
        return self._in_edge_ids[start:end]

    def get_out_edge_ids(self, node_id: int) -> np.ndarray:
        """Return outgoing edge IDs for a node."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID."
        start = int(self._out_edge_indptr[node_id])
        end = int(self._out_edge_indptr[node_id + 1])
        return self._out_edge_ids[start:end]

    @property
    def in_edge_ids(self) -> np.ndarray:
        """Incoming edge IDs array."""
        return self._in_edge_ids

    @property
    def in_edge_indptr(self) -> np.ndarray:
        """Incoming edge pointer array."""
        return self._in_edge_indptr

    @property
    def out_edge_ids(self) -> np.ndarray:
        """Outgoing edge IDs array."""
        return self._out_edge_ids

    @property
    def out_edge_indptr(self) -> np.ndarray:
        """Outgoing edge pointer array."""
        return self._out_edge_indptr

    def copy(self) -> "_FragmentTreeAdjacency":
        """Return copied adjacency arrays."""
        return _FragmentTreeAdjacency(
            in_edge_ids=self._in_edge_ids.copy(),
            in_edge_indptr=self._in_edge_indptr.copy(),
            out_edge_ids=self._out_edge_ids.copy(),
            out_edge_indptr=self._out_edge_indptr.copy(),
        )
