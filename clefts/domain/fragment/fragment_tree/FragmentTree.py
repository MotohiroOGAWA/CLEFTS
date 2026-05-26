from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ....libs.mmkit.mmkit import Compound
from .FragmentEdge import FragmentEdge
from .FragmentNode import FragmentNode


@dataclass(frozen=True, init=False)
class FragmentTree:
    """Array-backed data container for a generated fragmentation tree.

    ``FragmentTree`` stores the root molecule, fragment nodes, directed edges,
    and serialized fragmentation steps in compact NumPy arrays. It also provides
    lightweight convenience methods for reconstructing nodes, edges, adjacency,
    and node depths from those arrays. Persistence and path caches are kept
    outside this class.
    """

    _smiles: str
    _node_smiles: np.ndarray
    _edge_index: np.ndarray
    _edge_step_flat: np.ndarray
    _edge_step_indptr: np.ndarray

    _in_edge_ids: Optional[np.ndarray] = field(default=None, repr=False)
    _in_edge_indptr: Optional[np.ndarray] = field(default=None, repr=False)
    _out_edge_ids: Optional[np.ndarray] = field(default=None, repr=False)
    _out_edge_indptr: Optional[np.ndarray] = field(default=None, repr=False)
    _node_depth: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        smiles: str,
        node_smiles: np.ndarray,
        edge_index: np.ndarray,
        edge_step_flat: np.ndarray,
        edge_step_indptr: np.ndarray,
        in_edge_ids: Optional[np.ndarray] = None,
        in_edge_indptr: Optional[np.ndarray] = None,
        out_edge_ids: Optional[np.ndarray] = None,
        out_edge_indptr: Optional[np.ndarray] = None,
        node_depth: Optional[np.ndarray] = None,
    ):
        """Create a fragment tree from array data.

        Parameters
        ----------
        smiles : str
            Canonical SMILES of the root compound.
        node_smiles : numpy.ndarray
            One-dimensional object array. The array index is the node ID.
        edge_index : numpy.ndarray
            Integer array of shape ``[E, 2]`` storing source and target node IDs.
        edge_step_flat : numpy.ndarray
            Flat object array of serialized fragmentation steps.
        edge_step_indptr : numpy.ndarray
            Pointer array of shape ``[E + 1]`` into ``edge_step_flat``.
        in_edge_ids, in_edge_indptr, out_edge_ids, out_edge_indptr : numpy.ndarray, optional
            Optional precomputed adjacency arrays. Invalid arrays are ignored.
        node_depth : numpy.ndarray, optional
            Optional precomputed node-depth array.
        """
        node_smiles = np.asarray(node_smiles, dtype=object)
        edge_index = np.asarray(edge_index, dtype=np.int32)
        edge_step_flat = np.asarray(edge_step_flat, dtype=object)
        edge_step_indptr = np.asarray(edge_step_indptr, dtype=np.int64)

        assert node_smiles.ndim == 1, "node_smiles must be a 1D array."
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2, "edge_index must have shape [E, 2]."
        assert edge_step_indptr.ndim == 1 and edge_step_indptr.shape[0] == edge_index.shape[0] + 1, "edge_step_indptr must be length E+1."
        assert edge_step_flat.ndim == 1, "edge_step_flat must be a 1D array."
        assert edge_step_indptr[0] == 0, "edge_step_indptr must start at 0."
        assert edge_step_indptr[-1] == len(edge_step_flat), "edge_step_indptr last element must equal len(edge_step_flat)."

        object.__setattr__(self, "_smiles", str(smiles))
        object.__setattr__(self, "_node_smiles", node_smiles)
        object.__setattr__(self, "_edge_index", edge_index)
        object.__setattr__(self, "_edge_step_flat", edge_step_flat)
        object.__setattr__(self, "_edge_step_indptr", edge_step_indptr)
        object.__setattr__(self, "_in_edge_ids", None)
        object.__setattr__(self, "_in_edge_indptr", None)
        object.__setattr__(self, "_out_edge_ids", None)
        object.__setattr__(self, "_out_edge_indptr", None)
        object.__setattr__(self, "_node_depth", None if node_depth is None else np.asarray(node_depth, dtype=np.int32))

        if in_edge_ids is not None and in_edge_indptr is not None:
            in_edge_ids = np.asarray(in_edge_ids, dtype=np.int32)
            in_edge_indptr = np.asarray(in_edge_indptr, dtype=np.int64)
            if self._valid_csr(in_edge_ids, in_edge_indptr):
                object.__setattr__(self, "_in_edge_ids", in_edge_ids)
                object.__setattr__(self, "_in_edge_indptr", in_edge_indptr)

        if out_edge_ids is not None and out_edge_indptr is not None:
            out_edge_ids = np.asarray(out_edge_ids, dtype=np.int32)
            out_edge_indptr = np.asarray(out_edge_indptr, dtype=np.int64)
            if self._valid_csr(out_edge_ids, out_edge_indptr):
                object.__setattr__(self, "_out_edge_ids", out_edge_ids)
                object.__setattr__(self, "_out_edge_indptr", out_edge_indptr)

    def __repr__(self) -> str:
        return f"FragmentTree(smiles='{self.smiles}', nodes={self.num_nodes}, edges={self.num_edges})"

    def _valid_csr(self, edge_ids: np.ndarray, indptr: np.ndarray) -> bool:
        return (
            edge_ids.ndim == 1
            and indptr.ndim == 1
            and indptr.shape[0] == self.num_nodes + 1
            and indptr[0] == 0
            and indptr[-1] == len(edge_ids)
        )

    def _build_edge_adjacency(self) -> None:
        """Build incoming and outgoing edge adjacency arrays."""
        n = self.num_nodes
        e = self.num_edges
        if e == 0:
            object.__setattr__(self, "_in_edge_ids", np.asarray([], dtype=np.int32))
            object.__setattr__(self, "_in_edge_indptr", np.zeros(n + 1, dtype=np.int64))
            object.__setattr__(self, "_out_edge_ids", np.asarray([], dtype=np.int32))
            object.__setattr__(self, "_out_edge_indptr", np.zeros(n + 1, dtype=np.int64))
            return

        src = self._edge_index[:, 0].astype(np.int32, copy=False)
        dst = self._edge_index[:, 1].astype(np.int32, copy=False)
        edge_ids = np.arange(e, dtype=np.int32)

        order_in = np.argsort(dst, kind="mergesort")
        dst_sorted = dst[order_in]
        in_edge_ids = edge_ids[order_in]
        in_counts = np.bincount(dst_sorted, minlength=n).astype(np.int64)
        in_indptr = np.empty(n + 1, dtype=np.int64)
        in_indptr[0] = 0
        np.cumsum(in_counts, out=in_indptr[1:])

        order_out = np.argsort(src, kind="mergesort")
        src_sorted = src[order_out]
        out_edge_ids = edge_ids[order_out]
        out_counts = np.bincount(src_sorted, minlength=n).astype(np.int64)
        out_indptr = np.empty(n + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(out_counts, out=out_indptr[1:])

        object.__setattr__(self, "_in_edge_ids", in_edge_ids)
        object.__setattr__(self, "_in_edge_indptr", in_indptr)
        object.__setattr__(self, "_out_edge_ids", out_edge_ids)
        object.__setattr__(self, "_out_edge_indptr", out_indptr)

    @staticmethod
    def empty(compound: Compound) -> "FragmentTree":
        """Create a tree containing only the root compound."""
        return FragmentTree(
            smiles=compound.smiles,
            node_smiles=np.asarray([compound.smiles], dtype=object),
            edge_index=np.zeros((0, 2), dtype=np.int32),
            edge_step_flat=np.asarray([], dtype=object),
            edge_step_indptr=np.zeros(1, dtype=np.int64),
        )

    @staticmethod
    def from_nodes_and_edges(
        smiles: str,
        nodes: Tuple[FragmentNode, ...],
        edges: Tuple[FragmentEdge, ...],
    ) -> "FragmentTree":
        """Create a tree from node and edge objects."""
        node_smiles = np.asarray([node.smiles for node in nodes], dtype=object)
        edge_index = np.zeros((len(edges), 2), dtype=np.int32)
        edge_step_flat_list = []
        edge_step_indptr = np.zeros(len(edges) + 1, dtype=np.int64)

        for i, edge in enumerate(edges):
            edge_index[i, 0] = edge.source_id
            edge_index[i, 1] = edge.target_id
            edge_step_flat_list.extend(edge.fragment_step_strs)
            edge_step_indptr[i + 1] = len(edge_step_flat_list)

        return FragmentTree(
            smiles=smiles,
            node_smiles=node_smiles,
            edge_index=edge_index,
            edge_step_flat=np.asarray(edge_step_flat_list, dtype=object),
            edge_step_indptr=edge_step_indptr,
        )

    @property
    def smiles(self) -> str:
        """Canonical SMILES of the root compound."""
        return self._smiles

    @property
    def node_smiles(self) -> np.ndarray:
        """Node SMILES array. The array index is the node ID."""
        return self._node_smiles.copy()

    @property
    def edge_index(self) -> np.ndarray:
        """Directed edge array with shape ``[E, 2]``."""
        return self._edge_index.copy()

    @property
    def edge_step_flat(self) -> np.ndarray:
        """Flat array of serialized fragmentation steps."""
        return self._edge_step_flat.copy()

    @property
    def edge_step_indptr(self) -> np.ndarray:
        """Pointer array into :attr:`edge_step_flat` for each edge."""
        return self._edge_step_indptr.copy()

    @property
    def _in_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Incoming edge adjacency as ``(edge_ids, indptr)``."""
        if self._in_edge_ids is None or self._in_edge_indptr is None:
            self._build_edge_adjacency()
        return self._in_edge_ids, self._in_edge_indptr

    @property
    def _out_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Outgoing edge adjacency as ``(edge_ids, indptr)``."""
        if self._out_edge_ids is None or self._out_edge_indptr is None:
            self._build_edge_adjacency()
        return self._out_edge_ids, self._out_edge_indptr

    @property
    def in_edge_ids(self) -> Optional[np.ndarray]:
        """Optional incoming-edge IDs array, if already available."""
        return None if self._in_edge_ids is None else self._in_edge_ids.copy()

    @property
    def in_edge_indptr(self) -> Optional[np.ndarray]:
        """Optional incoming-edge pointer array, if already available."""
        return None if self._in_edge_indptr is None else self._in_edge_indptr.copy()

    @property
    def out_edge_ids(self) -> Optional[np.ndarray]:
        """Optional outgoing-edge IDs array, if already available."""
        return None if self._out_edge_ids is None else self._out_edge_ids.copy()

    @property
    def out_edge_indptr(self) -> Optional[np.ndarray]:
        """Optional outgoing-edge pointer array, if already available."""
        return None if self._out_edge_indptr is None else self._out_edge_indptr.copy()

    @property
    def node_depth(self) -> Optional[np.ndarray]:
        """Optional node-depth array, if already available."""
        return None if self._node_depth is None else self._node_depth.copy()

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the tree."""
        return len(self._node_smiles)

    @property
    def num_edges(self) -> int:
        """Number of directed edges in the tree."""
        return self._edge_index.shape[0]

    def get_node(self, node_id: int) -> FragmentNode:
        """Return a node by ID."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID."
        return FragmentNode(id=int(node_id), smiles=str(self._node_smiles[node_id]))

    def get_edges(self, edge_id: int) -> FragmentEdge:
        """Return an edge by ID."""
        assert 0 <= edge_id < self.num_edges, "Invalid edge ID."
        source_id = int(self._edge_index[edge_id, 0])
        target_id = int(self._edge_index[edge_id, 1])
        start_idx = int(self._edge_step_indptr[edge_id])
        end_idx = int(self._edge_step_indptr[edge_id + 1])
        return FragmentEdge(
            id=int(edge_id),
            source_id=source_id,
            target_id=target_id,
            fragment_step_strs=tuple(self._edge_step_flat[start_idx:end_idx]),
        )

    def get_in_edges(self, node_id: int) -> List[FragmentEdge]:
        """Return all incoming edges of a node."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID."
        edge_ids, indptr = self._in_edge_csr
        start = int(indptr[node_id])
        end = int(indptr[node_id + 1])
        return [self.get_edges(int(edge_id)) for edge_id in edge_ids[start:end]]

    def get_out_edges(self, node_id: int) -> List[FragmentEdge]:
        """Return all outgoing edges of a node."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID."
        edge_ids, indptr = self._out_edge_csr
        start = int(indptr[node_id])
        end = int(indptr[node_id + 1])
        return [self.get_edges(int(edge_id)) for edge_id in edge_ids[start:end]]

    def get_parent_nodes(self, child_node_id: int) -> List[FragmentNode]:
        """Return parent nodes connected to a child node."""
        return [self.get_node(edge.source_id) for edge in self.get_in_edges(child_node_id)]

    def get_child_nodes(self, parent_node_id: int) -> List[FragmentNode]:
        """Return child nodes connected from a parent node."""
        return [self.get_node(edge.target_id) for edge in self.get_out_edges(parent_node_id)]

    def get_root_node_ids(self) -> np.ndarray:
        """Return node IDs with no incoming edges."""
        _, indptr = self._in_edge_csr
        indegrees = indptr[1:] - indptr[:-1]
        return np.flatnonzero(indegrees == 0).astype(np.int32, copy=False)

    def _build_node_depths(self) -> np.ndarray:
        """Compute minimal edge distance from root nodes to every node."""
        n = self.num_nodes
        depth = np.full(n, -1, dtype=np.int32)
        if n == 0:
            object.__setattr__(self, "_node_depth", depth)
            return depth

        roots = self.get_root_node_ids()
        if roots.size == 0:
            roots = np.asarray([0], dtype=np.int32)

        edge_ids, indptr = self._out_edge_csr
        queue = deque()
        for root_id in roots:
            root_id = int(root_id)
            depth[root_id] = 0
            queue.append(root_id)

        while queue:
            source_id = queue.popleft()
            next_depth = int(depth[source_id]) + 1
            start = int(indptr[source_id])
            end = int(indptr[source_id + 1])
            for edge_id in edge_ids[start:end]:
                target_id = int(self._edge_index[int(edge_id), 1])
                if depth[target_id] == -1:
                    depth[target_id] = next_depth
                    queue.append(target_id)

        object.__setattr__(self, "_node_depth", depth)
        return depth

    @property
    def node_depths(self) -> np.ndarray:
        """Minimal depth from root nodes for each node."""
        if self._node_depth is None:
            return self._build_node_depths().copy()
        return self._node_depth.copy()

    def get_nodes_by_depth(self) -> Dict[int, np.ndarray]:
        """Group node IDs by minimal depth from root nodes."""
        depths = self.node_depths
        out: Dict[int, np.ndarray] = {}
        for depth in sorted(set(int(d) for d in depths if d >= 0)):
            out[depth] = np.flatnonzero(depths == depth).astype(np.int32, copy=False)
        return out

    def copy(self) -> "FragmentTree":
        """Return a copy of this fragment tree."""
        return FragmentTree(
            smiles=self.smiles,
            node_smiles=self._node_smiles.copy(),
            edge_index=self._edge_index.copy(),
            edge_step_flat=self._edge_step_flat.copy(),
            edge_step_indptr=self._edge_step_indptr.copy(),
            in_edge_ids=self.in_edge_ids,
            in_edge_indptr=self.in_edge_indptr,
            out_edge_ids=self.out_edge_ids,
            out_edge_indptr=self.out_edge_indptr,
            node_depth=self.node_depth,
        )
