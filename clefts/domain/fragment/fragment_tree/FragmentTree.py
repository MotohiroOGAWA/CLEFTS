from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ....libs.mmkit.mmkit import Compound
from .FragmentEdge import FragmentEdge
from .FragmentNode import FragmentNode
from ._private._CleavageEventStore import _CleavageEventStore
from ._private._FragmentEdgeStore import _FragmentEdgeStore
from ._private._FragmentNodeStore import _FragmentNodeStore
from ._private._FragmentTreeAdjacency import _FragmentTreeAdjacency
from ._private._FragmentTreeDepths import _FragmentTreeDepths


@dataclass(frozen=True, init=False)
class FragmentTree:
    """Read-oriented container for a generated fragmentation tree.

    ``FragmentTree`` exposes graph lookup methods such as ``get_node``,
    ``get_edge``, ``get_in_edges``, and ``get_nodes_by_depth``. Low-level
    storage is delegated to private helper classes under
    ``fragment_tree._private``.
    """

    _smiles: str
    _node_store: _FragmentNodeStore
    _edge_store: _FragmentEdgeStore
    _adjacency: Optional[_FragmentTreeAdjacency] = field(default=None, repr=False, compare=False)
    _depths: Optional[_FragmentTreeDepths] = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        smiles: str,
        node_store: _FragmentNodeStore,
        edge_store: _FragmentEdgeStore,
        adjacency: Optional[_FragmentTreeAdjacency] = None,
        depths: Optional[_FragmentTreeDepths] = None,
    ):
        """Create a tree from private storage objects."""
        assert isinstance(node_store, _FragmentNodeStore), "node_store must be a _FragmentNodeStore."
        assert isinstance(edge_store, _FragmentEdgeStore), "edge_store must be a _FragmentEdgeStore."
        assert adjacency is None or isinstance(adjacency, _FragmentTreeAdjacency), "adjacency must be a _FragmentTreeAdjacency."
        assert depths is None or isinstance(depths, _FragmentTreeDepths), "depths must be a _FragmentTreeDepths."

        object.__setattr__(self, "_smiles", str(smiles))
        object.__setattr__(self, "_node_store", node_store)
        object.__setattr__(self, "_edge_store", edge_store)
        object.__setattr__(self, "_adjacency", adjacency)
        object.__setattr__(self, "_depths", depths)

    def __repr__(self) -> str:
        return f"FragmentTree(smiles='{self.smiles}', nodes={self.num_nodes}, edges={self.num_edges})"

    @staticmethod
    def from_arrays(
        smiles: str,
        node_smiles: np.ndarray,
        edge_index: np.ndarray,
        cleavage_pattern_ids: np.ndarray,
        react_indices_strs: np.ndarray,
        prod_indices_strs: np.ndarray,
        edge_event_indptr: np.ndarray,
        adjacency: Optional[_FragmentTreeAdjacency] = None,
        depths: Optional[_FragmentTreeDepths] = None,
    ) -> "FragmentTree":
        """Create a tree from public array inputs."""
        edge_index = np.asarray(edge_index, dtype=np.int32)
        event_store = _CleavageEventStore(
            cleavage_pattern_ids=cleavage_pattern_ids,
            react_indices_strs=react_indices_strs,
            prod_indices_strs=prod_indices_strs,
            edge_event_indptr=edge_event_indptr,
            num_edges=edge_index.shape[0],
        )
        return FragmentTree(
            smiles=smiles,
            node_store=_FragmentNodeStore(node_smiles=node_smiles),
            edge_store=_FragmentEdgeStore(edge_index=edge_index, event_store=event_store),
            adjacency=adjacency,
            depths=depths,
        )

    @staticmethod
    def empty(compound: Compound) -> "FragmentTree":
        """Create a tree containing only the root compound."""
        return FragmentTree.from_arrays(
            smiles=compound.smiles,
            node_smiles=np.asarray([compound.smiles], dtype=object),
            edge_index=np.zeros((0, 2), dtype=np.int32),
            cleavage_pattern_ids=np.asarray([], dtype=np.int32),
            react_indices_strs=np.asarray([], dtype=object),
            prod_indices_strs=np.asarray([], dtype=object),
            edge_event_indptr=np.zeros(1, dtype=np.int64),
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
        for i, edge in enumerate(edges):
            edge_index[i, 0] = edge.source_id
            edge_index[i, 1] = edge.target_id

        event_store = _CleavageEventStore.from_edges(edges)
        return FragmentTree(
            smiles=smiles,
            node_store=_FragmentNodeStore(node_smiles=node_smiles),
            edge_store=_FragmentEdgeStore(edge_index=edge_index, event_store=event_store),
        )

    @property
    def smiles(self) -> str:
        """Canonical SMILES of the root compound."""
        return self._smiles

    @property
    def node_smiles(self) -> np.ndarray:
        """Node SMILES array. The array index is the node ID."""
        return self._node_store.node_smiles

    @property
    def edge_index(self) -> np.ndarray:
        """Directed edge array with shape ``[E, 2]``."""
        return self._edge_store.edge_index

    @property
    def cleavage_pattern_ids(self) -> np.ndarray:
        """Cleavage pattern ID for each stored event."""
        return self._edge_store.event_store.cleavage_pattern_ids

    @property
    def react_indices_strs(self) -> np.ndarray:
        """JSON string of source atom indices for each stored event."""
        return self._edge_store.event_store.react_indices_strs

    @property
    def prod_indices_strs(self) -> np.ndarray:
        """JSON string of product atom indices for each stored event."""
        return self._edge_store.event_store.prod_indices_strs

    @property
    def edge_event_indptr(self) -> np.ndarray:
        """Pointer array into event columns for each edge."""
        return self._edge_store.event_store.edge_event_indptr

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the tree."""
        return self._node_store.num_nodes

    @property
    def num_edges(self) -> int:
        """Number of directed edges in the tree."""
        return self._edge_store.num_edges

    @property
    def in_edge_ids(self) -> Optional[np.ndarray]:
        """Incoming-edge IDs array, if adjacency has already been built."""
        return None if self._adjacency is None else self._adjacency.in_edge_ids

    @property
    def in_edge_indptr(self) -> Optional[np.ndarray]:
        """Incoming-edge pointer array, if adjacency has already been built."""
        return None if self._adjacency is None else self._adjacency.in_edge_indptr

    @property
    def out_edge_ids(self) -> Optional[np.ndarray]:
        """Outgoing-edge IDs array, if adjacency has already been built."""
        return None if self._adjacency is None else self._adjacency.out_edge_ids

    @property
    def out_edge_indptr(self) -> Optional[np.ndarray]:
        """Outgoing-edge pointer array, if adjacency has already been built."""
        return None if self._adjacency is None else self._adjacency.out_edge_indptr

    @property
    def node_depth(self) -> Optional[np.ndarray]:
        """Node-depth array, if already available."""
        return None if self._depths is None else self._depths.node_depths

    def _ensure_adjacency(self) -> _FragmentTreeAdjacency:
        if self._adjacency is None:
            object.__setattr__(
                self,
                "_adjacency",
                _FragmentTreeAdjacency.from_edge_index(
                    edge_index=self._edge_store.edge_index,
                    num_nodes=self._node_store.num_nodes,
                ),
            )
        return self._adjacency

    def _ensure_depths(self) -> _FragmentTreeDepths:
        if self._depths is None:
            object.__setattr__(
                self,
                "_depths",
                _FragmentTreeDepths.build(
                    num_nodes=self.num_nodes,
                    edge_store=self._edge_store,
                    adjacency=self._ensure_adjacency(),
                ),
            )
        return self._depths

    def get_node(self, node_id: int) -> FragmentNode:
        """Return a node by ID."""
        return self._node_store.get_node(node_id)

    def get_edge(self, edge_id: int) -> FragmentEdge:
        """Return an edge by ID."""
        return self._edge_store.get_edge(edge_id)

    def get_in_edges(self, node_id: int) -> List[FragmentEdge]:
        """Return all incoming edges of a node."""
        edge_ids = self._ensure_adjacency().get_in_edge_ids(node_id)
        return [self.get_edge(int(edge_id)) for edge_id in edge_ids]

    def get_out_edges(self, node_id: int) -> List[FragmentEdge]:
        """Return all outgoing edges of a node."""
        edge_ids = self._ensure_adjacency().get_out_edge_ids(node_id)
        return [self.get_edge(int(edge_id)) for edge_id in edge_ids]

    def get_parent_nodes(self, child_node_id: int) -> List[FragmentNode]:
        """Return parent nodes connected to a child node."""
        return [self.get_node(edge.source_id) for edge in self.get_in_edges(child_node_id)]

    def get_child_nodes(self, parent_node_id: int) -> List[FragmentNode]:
        """Return child nodes connected from a parent node."""
        return [self.get_node(edge.target_id) for edge in self.get_out_edges(parent_node_id)]

    def get_root_node_ids(self) -> np.ndarray:
        """Return node IDs with no incoming edges."""
        _, indptr = self._ensure_adjacency().in_edge_csr
        indegrees = indptr[1:] - indptr[:-1]
        return np.flatnonzero(indegrees == 0).astype(np.int32, copy=False)

    @property
    def node_depths(self) -> np.ndarray:
        """Minimal depth from root nodes for each node."""
        return self._ensure_depths().node_depths

    def get_nodes_by_depth(self) -> Dict[int, np.ndarray]:
        """Group node IDs by minimal depth from root nodes."""
        return self._ensure_depths().get_nodes_by_depth()

    def copy(self) -> "FragmentTree":
        """Return a copy of this fragment tree."""
        return FragmentTree(
            smiles=self.smiles,
            node_store=self._node_store.copy(),
            edge_store=self._edge_store.copy(),
            adjacency=None if self._adjacency is None else self._adjacency.copy(),
            depths=None if self._depths is None else self._depths.copy(),
        )
