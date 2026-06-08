from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ....libs.mmkit.mmkit import Compound
from .FragmentEdge import FragmentEdge
from .FragmentNode import FragmentNode
from ._private._CleavageEventStore import _CleavageEventStore
from ._private._FragmentEdgeStore import _FragmentEdgeStore
from ._private._FragmentNodeStore import _FragmentNodeStore
from ._private._FragmentTreeAdjacency import _FragmentTreeAdjacency
from ._private._FragmentTreeDepths import _FragmentTreeDepths


@dataclass(frozen=True)
class FragmentTree:
    """Read-oriented container for a generated fragmentation tree.

    Notes
    -----
    Local indices are used for tree traversal.

    Database IDs are stored separately.

    FragmentNode:
        index = local node index
        id    = database fragment_id

    FragmentEdge:
        index        = local edge index
        id           = database edge_id
        source_index = local source node index
        target_index = local target node index
        source_id    = database source fragment_id
        target_id    = database target fragment_id
    """

    smiles: str
    node_store: _FragmentNodeStore
    edge_store: _FragmentEdgeStore
    adjacency: _FragmentTreeAdjacency | None = None
    depths: _FragmentTreeDepths | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.smiles, str):
            raise TypeError("smiles must be a string.")

        if not isinstance(self.node_store, _FragmentNodeStore):
            raise TypeError("node_store must be a _FragmentNodeStore.")

        if not isinstance(self.edge_store, _FragmentEdgeStore):
            raise TypeError("edge_store must be a _FragmentEdgeStore.")

        if self.adjacency is not None and not isinstance(
            self.adjacency,
            _FragmentTreeAdjacency,
        ):
            raise TypeError("adjacency must be a _FragmentTreeAdjacency.")

        if self.depths is not None and not isinstance(
            self.depths,
            _FragmentTreeDepths,
        ):
            raise TypeError("depths must be a _FragmentTreeDepths.")

    def __repr__(self) -> str:
        return (
            f"FragmentTree("
            f"smiles='{self.smiles}', "
            f"nodes={self.num_nodes}, "
            f"edges={self.num_edges})"
        )

    @staticmethod
    def empty(compound: Compound) -> "FragmentTree":
        """Create a tree containing only the root compound.

        In memory-only mode, node id is temporarily set to 0.
        When using DB-backed building, use the DB fragment_id instead.
        """
        if not isinstance(compound, Compound):
            raise TypeError("compound must be a Compound.")

        return FragmentTree(
            smiles=compound.smiles,
            node_store=_FragmentNodeStore(
                node_ids=np.asarray([0], dtype=np.int64),
                node_smiles=np.asarray([compound.smiles], dtype=object),
            ),
            edge_store=_FragmentEdgeStore(
                edge_ids=np.asarray([], dtype=np.int64),
                source_indices=np.asarray([], dtype=np.int64),
                target_indices=np.asarray([], dtype=np.int64),
                source_ids=np.asarray([], dtype=np.int64),
                target_ids=np.asarray([], dtype=np.int64),
                event_store=_CleavageEventStore.empty(num_edges=0),
            ),
        )

    @staticmethod
    def from_arrays(
        *,
        smiles: str,
        node_ids: np.ndarray,
        node_smiles: np.ndarray,
        edge_ids: np.ndarray,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        source_ids: np.ndarray,
        target_ids: np.ndarray,
        event_ids: np.ndarray,
        cleavage_pattern_ids: np.ndarray,
        reaction_ids: np.ndarray,
        product_molecule_ids: np.ndarray,
        reactant_indices_strs: np.ndarray,
        product_indices_strs: np.ndarray,
        edge_event_indptr: np.ndarray,
        adjacency: _FragmentTreeAdjacency | None = None,
        depths: _FragmentTreeDepths | None = None,
    ) -> "FragmentTree":
        """Create a FragmentTree from array inputs."""
        event_store = _CleavageEventStore(
            event_ids=event_ids,
            cleavage_pattern_ids=cleavage_pattern_ids,
            reaction_ids=reaction_ids,
            product_molecule_ids=product_molecule_ids,
            reactant_indices_strs=reactant_indices_strs,
            product_indices_strs=product_indices_strs,
            edge_event_indptr=edge_event_indptr,
        )

        return FragmentTree(
            smiles=smiles,
            node_store=_FragmentNodeStore(
                node_ids=node_ids,
                node_smiles=node_smiles,
            ),
            edge_store=_FragmentEdgeStore(
                edge_ids=edge_ids,
                source_indices=source_indices,
                target_indices=target_indices,
                source_ids=source_ids,
                target_ids=target_ids,
                event_store=event_store,
            ),
            adjacency=adjacency,
            depths=depths,
        )

    @staticmethod
    def from_nodes_and_edges(
        *,
        smiles: str,
        nodes: tuple[FragmentNode, ...],
        edges: tuple[FragmentEdge, ...],
    ) -> "FragmentTree":
        """Create a FragmentTree from node and edge objects."""
        nodes = tuple(sorted(nodes, key=lambda node: node.index))
        edges = tuple(sorted(edges, key=lambda edge: edge.index))

        FragmentTree._validate_node_indices(nodes)
        FragmentTree._validate_edge_indices(edges)
        FragmentTree._validate_edge_node_indices(
            edges=edges,
            num_nodes=len(nodes),
        )

        node_ids = np.asarray(
            [node.id for node in nodes],
            dtype=np.int64,
        )
        node_smiles = np.asarray(
            [node.smiles for node in nodes],
            dtype=object,
        )

        edge_ids = np.asarray(
            [edge.id for edge in edges],
            dtype=np.int64,
        )
        source_indices = np.asarray(
            [edge.source_index for edge in edges],
            dtype=np.int64,
        )
        target_indices = np.asarray(
            [edge.target_index for edge in edges],
            dtype=np.int64,
        )
        source_ids = np.asarray(
            [edge.source_id for edge in edges],
            dtype=np.int64,
        )
        target_ids = np.asarray(
            [edge.target_id for edge in edges],
            dtype=np.int64,
        )

        event_store = _CleavageEventStore.from_edges(edges)

        return FragmentTree(
            smiles=smiles,
            node_store=_FragmentNodeStore(
                node_ids=node_ids,
                node_smiles=node_smiles,
            ),
            edge_store=_FragmentEdgeStore(
                edge_ids=edge_ids,
                source_indices=source_indices,
                target_indices=target_indices,
                source_ids=source_ids,
                target_ids=target_ids,
                event_store=event_store,
            ),
        )

    @staticmethod
    def _validate_node_indices(nodes: tuple[FragmentNode, ...]) -> None:
        for expected_index, node in enumerate(nodes):
            if node.index != expected_index:
                raise ValueError(
                    "Node indices must be continuous from 0 to num_nodes - 1. "
                    f"Expected {expected_index}, got {node.index}."
                )

    @staticmethod
    def _validate_edge_indices(edges: tuple[FragmentEdge, ...]) -> None:
        for expected_index, edge in enumerate(edges):
            if edge.index != expected_index:
                raise ValueError(
                    "Edge indices must be continuous from 0 to num_edges - 1. "
                    f"Expected {expected_index}, got {edge.index}."
                )

    @staticmethod
    def _validate_edge_node_indices(
        *,
        edges: tuple[FragmentEdge, ...],
        num_nodes: int,
    ) -> None:
        for edge in edges:
            if not 0 <= edge.source_index < num_nodes:
                raise ValueError(
                    f"Invalid source_index: {edge.source_index}"
                )

            if not 0 <= edge.target_index < num_nodes:
                raise ValueError(
                    f"Invalid target_index: {edge.target_index}"
                )

    @property
    def node_ids(self) -> np.ndarray:
        """Database fragment IDs for each local node index."""
        return self.node_store.node_ids

    @property
    def node_smiles(self) -> np.ndarray:
        """SMILES for each local node index."""
        return self.node_store.node_smiles

    @property
    def edge_ids(self) -> np.ndarray:
        """Database edge IDs for each local edge index."""
        return self.edge_store.edge_ids

    @property
    def source_indices(self) -> np.ndarray:
        """Local source node indices for each edge."""
        return self.edge_store.source_indices

    @property
    def target_indices(self) -> np.ndarray:
        """Local target node indices for each edge."""
        return self.edge_store.target_indices

    @property
    def source_ids(self) -> np.ndarray:
        """Database source fragment IDs for each edge."""
        return self.edge_store.source_ids

    @property
    def target_ids(self) -> np.ndarray:
        """Database target fragment IDs for each edge."""
        return self.edge_store.target_ids

    @property
    def event_ids(self) -> np.ndarray:
        """Database event IDs."""
        return self.edge_store.event_store.event_ids

    @property
    def cleavage_pattern_ids(self) -> np.ndarray:
        """Cleavage pattern IDs for each stored event."""
        return self.edge_store.event_store.cleavage_pattern_ids

    @property
    def reaction_ids(self) -> np.ndarray:
        """Reaction IDs for each stored event."""
        return self.edge_store.event_store.reaction_ids

    @property
    def product_molecule_ids(self) -> np.ndarray:
        """Product molecule IDs for each stored event."""
        return self.edge_store.event_store.product_molecule_ids

    @property
    def reactant_indices_strs(self) -> np.ndarray:
        """Reactant atom index strings for each stored event."""
        return self.edge_store.event_store.reactant_indices_strs

    @property
    def product_indices_strs(self) -> np.ndarray:
        """Product atom index strings for each stored event."""
        return self.edge_store.event_store.product_indices_strs

    @property
    def edge_event_indptr(self) -> np.ndarray:
        """Pointer array from edge index to event range."""
        return self.edge_store.event_store.edge_event_indptr

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return self.node_store.num_nodes

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return self.edge_store.num_edges

    @property
    def num_events(self) -> int:
        """Number of cleavage events."""
        return self.edge_store.event_store.num_events

    @property
    def in_edge_indices(self) -> np.ndarray | None:
        """Incoming local edge indices, if adjacency is already built."""
        if self.adjacency is None:
            return None
        return self.adjacency.in_edge_indices

    @property
    def in_edge_indptr(self) -> np.ndarray | None:
        """Incoming edge pointer array, if adjacency is already built."""
        if self.adjacency is None:
            return None
        return self.adjacency.in_edge_indptr

    @property
    def out_edge_indices(self) -> np.ndarray | None:
        """Outgoing local edge indices, if adjacency is already built."""
        if self.adjacency is None:
            return None
        return self.adjacency.out_edge_indices

    @property
    def out_edge_indptr(self) -> np.ndarray | None:
        """Outgoing edge pointer array, if adjacency is already built."""
        if self.adjacency is None:
            return None
        return self.adjacency.out_edge_indptr

    def _ensure_adjacency(self) -> _FragmentTreeAdjacency:
        if self.adjacency is not None:
            return self.adjacency

        adjacency = _FragmentTreeAdjacency.from_source_target_indices(
            source_indices=self.source_indices,
            target_indices=self.target_indices,
            num_nodes=self.num_nodes,
        )

        object.__setattr__(self, "adjacency", adjacency)

        return adjacency

    def _ensure_depths(self) -> _FragmentTreeDepths:
        if self.depths is not None:
            return self.depths

        depths = _FragmentTreeDepths.build(
            num_nodes=self.num_nodes,
            edge_store=self.edge_store,
            adjacency=self._ensure_adjacency(),
        )

        object.__setattr__(self, "depths", depths)

        return depths

    def get_node(self, index: int) -> FragmentNode:
        """Return a node by local node index."""
        return self.node_store.get_node(index)

    def get_edge(self, index: int) -> FragmentEdge:
        """Return an edge by local edge index."""
        return self.edge_store.get_edge(index)

    def get_in_edges(self, node_index: int) -> list[FragmentEdge]:
        """Return incoming edges of a node."""
        edge_indices = self._ensure_adjacency().get_in_edge_indices(
            node_index,
        )

        return [
            self.get_edge(int(edge_index))
            for edge_index in edge_indices
        ]

    def get_out_edges(self, node_index: int) -> list[FragmentEdge]:
        """Return outgoing edges of a node."""
        edge_indices = self._ensure_adjacency().get_out_edge_indices(
            node_index,
        )

        return [
            self.get_edge(int(edge_index))
            for edge_index in edge_indices
        ]

    def get_parent_nodes(self, child_index: int) -> list[FragmentNode]:
        """Return parent nodes connected to a child node."""
        return [
            self.get_node(edge.source_index)
            for edge in self.get_in_edges(child_index)
        ]

    def get_child_nodes(self, parent_index: int) -> list[FragmentNode]:
        """Return child nodes connected from a parent node."""
        return [
            self.get_node(edge.target_index)
            for edge in self.get_out_edges(parent_index)
        ]

    def get_root_node_indices(self) -> np.ndarray:
        """Return local node indices with no incoming edges."""
        _, in_edge_indptr = self._ensure_adjacency().in_edge_csr
        indegrees = in_edge_indptr[1:] - in_edge_indptr[:-1]

        return np.flatnonzero(indegrees == 0).astype(
            np.int64,
            copy=False,
        )

    def get_root_node_ids(self) -> np.ndarray:
        """Return database fragment IDs of root nodes."""
        root_indices = self.get_root_node_indices()
        return self.node_ids[root_indices]

    @property
    def node_depths(self) -> np.ndarray:
        """Minimal depth from root nodes for each local node index."""
        return self._ensure_depths().node_depths

    def get_nodes_by_depth(self) -> Dict[int, np.ndarray]:
        """Group local node indices by minimal depth."""
        return self._ensure_depths().get_nodes_by_depth()

    def copy(self) -> "FragmentTree":
        """Return a copy of this FragmentTree."""
        return FragmentTree(
            smiles=self.smiles,
            node_store=self.node_store.copy(),
            edge_store=self.edge_store.copy(),
            adjacency=(
                None
                if self.adjacency is None
                else self.adjacency.copy()
            ),
            depths=(
                None
                if self.depths is None
                else self.depths.copy()
            ),
        )