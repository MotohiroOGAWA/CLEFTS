from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Iterable, Set, Mapping
from collections import deque, defaultdict

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

    def get_node_by_smiles(self, smiles: str) -> FragmentNode | None:
        """Return FragmentNode with the given SMILES, or None if not found."""
        return self.node_store.get_node_by_smiles(smiles)

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

    def collect_shortest_node_paths_to_child(
        self,
        parent_depths: Mapping[int, int],
        child_id: int,
        max_depth: int,
        *,
        include_unreachable: bool = False,
    ) -> Dict[int, Tuple[Tuple[int, ...], ...]]:
        """Collect shortest node-edge paths from parent nodes to one child node.

        Parameters
        ----------
        parent_depths:
            Mapping from parent node ID to its current depth from the root.
            The keys are treated as parent node IDs.
        child_id:
            Target child node ID.
        max_depth:
            Maximum allowed total depth from the root.
            A route is valid only when:

                parent_depth + path_depth <= max_depth

            where path_depth is the number of edges from the parent node to child_id.
        include_unreachable:
            If True, unreachable or depth-exceeded parent nodes are included with
            an empty tuple.

        Returns
        -------
        Dict[int, Tuple[Tuple[int, ...], ...]]
            Mapping from parent node ID to shortest node-edge paths.

            Each path is represented as an alternating sequence of
            node_index and edge_index:

                node_index, edge_index, node_index, edge_index, ..., node_index
        """

        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative: {max_depth}")

        parent_depths = dict(parent_depths)
        parent_ids_tuple = tuple(dict.fromkeys(parent_depths.keys()))

        result: Dict[int, Tuple[Tuple[int, ...], ...]] = {}

        if not parent_ids_tuple:
            return result

        for parent_id, parent_depth in parent_depths.items():
            if parent_depth < 0:
                raise ValueError(
                    f"parent depth must be non-negative: "
                    f"parent_id={parent_id}, depth={parent_depth}"
                )

        parent_id_set = set(parent_ids_tuple)

        # For each parent, the allowed additional depth from that parent to child.
        max_path_depths: Dict[int, int] = {
            parent_id: max_depth - parent_depth
            for parent_id, parent_depth in parent_depths.items()
        }

        # Parents already deeper than max_depth cannot have valid paths.
        reachable_parent_ids = {
            parent_id
            for parent_id, max_path_depth in max_path_depths.items()
            if max_path_depth >= 0
        }

        if not reachable_parent_ids:
            if include_unreachable:
                return {
                    parent_id: tuple()
                    for parent_id in parent_ids_tuple
                }
            return result

        # We never need to search deeper than the largest allowed path depth.
        global_max_path_depth = max(
            max_path_depths[parent_id]
            for parent_id in reachable_parent_ids
        )

        # child itself is reachable from child with path depth 0.
        distances: Dict[int, int] = {
            child_id: 0,
        }

        # paths_to_child[node_id] stores shortest node-edge paths from node_id to child_id.
        paths_to_child: Dict[int, Tuple[Tuple[int, ...], ...]] = {
            child_id: ((child_id,),),
        }

        queue: deque[int] = deque([child_id])

        while queue:
            current_id = queue.popleft()
            current_path_depth = distances[current_id]

            if current_path_depth >= global_max_path_depth:
                continue

            # Traverse incoming edges:
            # source_index --edge_index--> current_id
            for edge_index in self.adjacency.get_in_edge_indices(current_id):
                source_index = int(self.edge_store.source_indices[edge_index])
                next_path_depth = current_path_depth + 1

                if next_path_depth > global_max_path_depth:
                    continue

                new_paths = tuple(
                    (source_index, edge_index, *path)
                    for path in paths_to_child[current_id]
                )

                old_path_depth = distances.get(source_index)

                if old_path_depth is None:
                    distances[source_index] = next_path_depth
                    paths_to_child[source_index] = new_paths
                    queue.append(source_index)

                elif next_path_depth == old_path_depth:
                    # Another shortest route to the same node.
                    paths_to_child[source_index] = (
                        *paths_to_child[source_index],
                        *new_paths,
                    )

        for parent_id in parent_ids_tuple:
            max_path_depth = max_path_depths[parent_id]

            if max_path_depth < 0:
                paths = tuple()
            else:
                paths = tuple(
                    path
                    for path in paths_to_child.get(parent_id, ())
                    if (len(path) - 1) // 2 <= max_path_depth
                )

            if paths or include_unreachable:
                result[parent_id] = paths

        return result
    
    def collect_global_shortest_node_paths_from_root_via(
        self,
        target_node_index: int,
        via_node_indices: Iterable[int],
        max_depth: int,
        max_via_depth: int,
    ) -> Tuple[Tuple[int, ...], ...]:
        """Return globally shortest paths from root to target via any via node.

        The via condition is OR condition:

            root(0) -> ... -> via -> ... -> target_node_index

        A path is valid only when:

            root_to_target_depth <= max_depth
            root_to_via_depth <= max_via_depth

        Parameters
        ----------
        target_node_index:
            Target node index.
        via_node_indices:
            Candidate via node indices.
            A returned path must pass through at least one of them.
        max_depth:
            Maximum allowed total path depth in edges from root to target.
        max_via_depth:
            Maximum allowed path depth in edges from root to via node.

        Returns
        -------
        Tuple[Tuple[int, ...], ...]
            All globally shortest valid node-edge paths.

            Each path is represented as an alternating sequence of
            node_index and edge_index:

                node_index, edge_index, node_index, edge_index, ..., node_index
        """

        if max_depth < 0:
            return tuple()

        if max_via_depth < 0:
            return tuple()

        root_index = 0
        via_node_set = set(via_node_indices)

        if not via_node_set:
            return tuple()

        # Special case:
        # root == target.
        # The only path is (0,), and it is valid only if root is a via node.
        if target_node_index == root_index:
            if root_index in via_node_set:
                return ((root_index,),)
            return tuple()

        # Collect shortest paths:
        #
        #     root -> ... -> via
        #
        # These prefix paths are also used to compute the current depth of each
        # via node from the root.
        prefix_paths_by_via: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
        parent_depths: Dict[int, int] = {}

        for via_node_index in via_node_set:
            if via_node_index == root_index:
                prefix_paths = ((root_index,),)
            else:
                root_to_via_paths_by_root = self.collect_shortest_node_paths_to_child(
                    parent_depths={root_index: 0},
                    child_id=via_node_index,
                    max_depth=max_via_depth,
                    include_unreachable=False,
                )

                prefix_paths = root_to_via_paths_by_root.get(root_index, tuple())

            if not prefix_paths:
                continue

            # All prefix paths for the same via node are shortest paths,
            # so they should have the same depth.
            prefix_depth = (len(prefix_paths[0]) - 1) // 2

            if prefix_depth > max_via_depth:
                continue

            if prefix_depth > max_depth:
                continue

            prefix_paths_by_via[via_node_index] = prefix_paths
            parent_depths[via_node_index] = prefix_depth

        if not parent_depths:
            return tuple()

        # Collect shortest paths:
        #
        #     via -> ... -> target
        #
        # The parent_depths mapping lets collect_shortest_node_paths_to_child()
        # prune routes where:
        #
        #     root_to_via_depth + via_to_target_depth > max_depth
        via_to_target_paths = self.collect_shortest_node_paths_to_child(
            parent_depths=parent_depths,
            child_id=target_node_index,
            max_depth=max_depth,
            include_unreachable=False,
        )

        if not via_to_target_paths:
            return tuple()

        candidate_paths: List[Tuple[int, ...]] = []

        for via_node_index, suffix_paths in via_to_target_paths.items():
            if not suffix_paths:
                continue

            prefix_paths = prefix_paths_by_via.get(via_node_index, tuple())

            if not prefix_paths:
                continue

            for prefix_path in prefix_paths:
                for suffix_path in suffix_paths:
                    # Avoid duplicating the via node.
                    full_path = prefix_path + suffix_path[1:]
                    full_depth = (len(full_path) - 1) // 2

                    if full_depth > max_depth:
                        continue

                    candidate_paths.append(full_path)

        if not candidate_paths:
            return tuple()

        min_depth = min(
            (len(path) - 1) // 2
            for path in candidate_paths
        )

        globally_shortest_paths = (
            path
            for path in candidate_paths
            if (len(path) - 1) // 2 == min_depth
        )

        # Remove duplicates while preserving order.
        return tuple(dict.fromkeys(globally_shortest_paths))