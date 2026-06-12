from __future__ import annotations

import unittest

import numpy as np

from clefts.domain.fragment.tree.CleavageEvent import CleavageEvent
from clefts.domain.fragment.tree.FragmentEdge import FragmentEdge
from clefts.domain.fragment.tree.FragmentNode import FragmentNode
from clefts.domain.fragment.tree.FragmentTree import FragmentTree
from clefts.domain.fragment.tree._private._CleavageEventStore import (
    _CleavageEventStore,
)
from clefts.domain.fragment.tree._private._FragmentEdgeStore import (
    _FragmentEdgeStore,
)
from clefts.domain.fragment.tree._private._FragmentNodeStore import (
    _FragmentNodeStore,
)
from clefts.domain.fragment.tree._private._FragmentTreeAdjacency import (
    _FragmentTreeAdjacency,
)
from clefts.domain.fragment.tree._private._FragmentTreeDepths import (
    _FragmentTreeDepths,
)


def make_event(
    *,
    index: int,
    event_id: int,
    cleavage_pattern_id: int = 0,
    reaction_id: int = 0,
    product_molecule_id: int = 0,
    reactant_indices: tuple[int, ...] = (0, 1),
    product_indices: tuple[int, ...] = (0,),
) -> CleavageEvent:
    return CleavageEvent(
        index=index,
        event_id=event_id,
        cleavage_pattern_id=cleavage_pattern_id,
        reaction_id=reaction_id,
        product_molecule_id=product_molecule_id,
        reactant_indices=reactant_indices,
        product_indices=product_indices,
    )


def make_nodes() -> tuple[FragmentNode, ...]:
    return (
        FragmentNode(index=0, id=101, smiles="CCO"),
        FragmentNode(index=1, id=102, smiles="CC"),
        FragmentNode(index=2, id=103, smiles="O"),
        FragmentNode(index=3, id=104, smiles="C"),
    )


def make_edges() -> tuple[FragmentEdge, ...]:
    event_0_0 = make_event(
        index=0,
        event_id=0,
        cleavage_pattern_id=10,
        reaction_id=100,
        product_molecule_id=0,
        reactant_indices=(0, 1),
        product_indices=(0,),
    )
    event_0_1 = make_event(
        index=1,
        event_id=1,
        cleavage_pattern_id=11,
        reaction_id=100,
        product_molecule_id=1,
        reactant_indices=(1, 2),
        product_indices=(0,),
    )
    event_1_0 = make_event(
        index=0,
        event_id=0,
        cleavage_pattern_id=12,
        reaction_id=101,
        product_molecule_id=0,
        reactant_indices=(0,),
        product_indices=(0,),
    )
    event_2_0 = make_event(
        index=0,
        event_id=0,
        cleavage_pattern_id=13,
        reaction_id=102,
        product_molecule_id=0,
        reactant_indices=(0,),
        product_indices=(0,),
    )

    return (
        FragmentEdge(
            index=0,
            id=201,
            source_index=0,
            target_index=1,
            source_id=101,
            target_id=102,
            events=(event_0_0, event_0_1),
        ),
        FragmentEdge(
            index=1,
            id=202,
            source_index=0,
            target_index=2,
            source_id=101,
            target_id=103,
            events=(event_1_0,),
        ),
        FragmentEdge(
            index=2,
            id=203,
            source_index=1,
            target_index=3,
            source_id=102,
            target_id=104,
            events=(event_2_0,),
        ),
    )


class TestFragmentNode(unittest.TestCase):

    def test_fragment_node_fields(self) -> None:
        node = FragmentNode(index=0, id=101, smiles="CCO")

        self.assertEqual(node.index, 0)
        self.assertEqual(node.id, 101)
        self.assertEqual(node.smiles, "CCO")

    def test_fragment_node_copy(self) -> None:
        node = FragmentNode(index=0, id=101, smiles="CCO")
        copied = node.copy()

        self.assertEqual(copied, node)
        self.assertIsNot(copied, node)


class TestCleavageEvent(unittest.TestCase):

    def test_cleavage_event_fields(self) -> None:
        event = make_event(
            index=0,
            event_id=5,
            cleavage_pattern_id=10,
            reaction_id=20,
            product_molecule_id=1,
            reactant_indices=(0, 1, 2),
            product_indices=(0, 2),
        )

        self.assertEqual(event.index, 0)
        self.assertEqual(event.event_id, 5)
        self.assertEqual(event.cleavage_pattern_id, 10)
        self.assertEqual(event.reaction_id, 20)
        self.assertEqual(event.product_molecule_id, 1)
        self.assertEqual(event.reactant_indices, (0, 1, 2))
        self.assertEqual(event.product_indices, (0, 2))

    def test_cleavage_event_indices_are_json_serialized(self) -> None:
        event = make_event(
            index=0,
            event_id=0,
            reactant_indices=(0, 1),
            product_indices=(2, 3),
        )

        self.assertEqual(event.reactant_indices_str, "[0,1]")
        self.assertEqual(event.product_indices_str, "[2,3]")

    def test_cleavage_event_copy(self) -> None:
        event = make_event(index=0, event_id=0)
        copied = event.copy()

        self.assertEqual(copied, event)
        self.assertIsNot(copied, event)


class TestFragmentEdge(unittest.TestCase):

    def test_fragment_edge_fields(self) -> None:
        event = make_event(index=0, event_id=0)

        edge = FragmentEdge(
            index=0,
            id=201,
            source_index=0,
            target_index=1,
            source_id=101,
            target_id=102,
            events=(event,),
        )

        self.assertEqual(edge.index, 0)
        self.assertEqual(edge.id, 201)
        self.assertEqual(edge.source_index, 0)
        self.assertEqual(edge.target_index, 1)
        self.assertEqual(edge.source_id, 101)
        self.assertEqual(edge.target_id, 102)
        self.assertEqual(edge.events, (event,))

    def test_fragment_edge_with_event(self) -> None:
        event_0 = make_event(index=0, event_id=0)
        event_1 = make_event(index=1, event_id=1)

        edge = FragmentEdge(
            index=0,
            id=201,
            source_index=0,
            target_index=1,
            source_id=101,
            target_id=102,
            events=(event_0,),
        )

        updated = edge.with_event(event_1)

        self.assertEqual(len(edge.events), 1)
        self.assertEqual(len(updated.events), 2)
        self.assertEqual(updated.events[0], event_0)
        self.assertEqual(updated.events[1], event_1)

    def test_fragment_edge_copy(self) -> None:
        edge = make_edges()[0]
        copied = edge.copy()

        self.assertEqual(copied, edge)
        self.assertIsNot(copied, edge)


class TestFragmentNodeStore(unittest.TestCase):

    def test_get_node(self) -> None:
        store = _FragmentNodeStore(
            node_ids=np.asarray([101, 102], dtype=np.int64),
            node_smiles=np.asarray(["CCO", "CC"], dtype=object),
        )

        node = store.get_node(1)

        self.assertEqual(node.index, 1)
        self.assertEqual(node.id, 102)
        self.assertEqual(node.smiles, "CC")

    def test_num_nodes(self) -> None:
        store = _FragmentNodeStore(
            node_ids=np.asarray([101, 102], dtype=np.int64),
            node_smiles=np.asarray(["CCO", "CC"], dtype=object),
        )

        self.assertEqual(store.num_nodes, 2)

    def test_invalid_node_store_lengths_raise_error(self) -> None:
        with self.assertRaises(ValueError):
            _FragmentNodeStore(
                node_ids=np.asarray([101], dtype=np.int64),
                node_smiles=np.asarray(["CCO", "CC"], dtype=object),
            )


class TestCleavageEventStore(unittest.TestCase):

    def test_from_edges(self) -> None:
        edges = make_edges()

        store = _CleavageEventStore.from_edges(edges)

        np.testing.assert_array_equal(
            store.event_ids,
            np.asarray([0, 1, 0, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            store.cleavage_pattern_ids,
            np.asarray([10, 11, 12, 13], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            store.edge_event_indptr,
            np.asarray([0, 2, 3, 4], dtype=np.int64),
        )

    def test_get_events_uses_local_index_within_edge(self) -> None:
        edges = make_edges()

        store = _CleavageEventStore.from_edges(edges)

        edge_0_events = store.get_events(0)
        edge_1_events = store.get_events(1)

        self.assertEqual(edge_0_events[0].index, 0)
        self.assertEqual(edge_0_events[1].index, 1)

        self.assertEqual(edge_1_events[0].index, 0)
        self.assertEqual(edge_1_events[0].event_id, 0)
        self.assertEqual(edge_1_events[0].cleavage_pattern_id, 12)

    def test_empty_event_store(self) -> None:
        store = _CleavageEventStore.empty(num_edges=2)

        self.assertEqual(store.num_events, 0)
        self.assertEqual(store.num_edges, 2)

        np.testing.assert_array_equal(
            store.edge_event_indptr,
            np.asarray([0, 0, 0], dtype=np.int64),
        )


class TestFragmentEdgeStore(unittest.TestCase):

    def test_get_edge(self) -> None:
        event_store = _CleavageEventStore.from_edges(make_edges())

        store = _FragmentEdgeStore(
            edge_ids=np.asarray([201, 202, 203], dtype=np.int64),
            source_indices=np.asarray([0, 0, 1], dtype=np.int64),
            target_indices=np.asarray([1, 2, 3], dtype=np.int64),
            source_ids=np.asarray([101, 101, 102], dtype=np.int64),
            target_ids=np.asarray([102, 103, 104], dtype=np.int64),
            event_store=event_store,
        )

        edge = store.get_edge(2)

        self.assertEqual(edge.index, 2)
        self.assertEqual(edge.id, 203)
        self.assertEqual(edge.source_index, 1)
        self.assertEqual(edge.target_index, 3)
        self.assertEqual(edge.source_id, 102)
        self.assertEqual(edge.target_id, 104)
        self.assertEqual(len(edge.events), 1)

    def test_num_edges(self) -> None:
        event_store = _CleavageEventStore.empty(num_edges=0)

        store = _FragmentEdgeStore(
            edge_ids=np.asarray([], dtype=np.int64),
            source_indices=np.asarray([], dtype=np.int64),
            target_indices=np.asarray([], dtype=np.int64),
            source_ids=np.asarray([], dtype=np.int64),
            target_ids=np.asarray([], dtype=np.int64),
            event_store=event_store,
        )

        self.assertEqual(store.num_edges, 0)


class TestFragmentTreeAdjacency(unittest.TestCase):

    def test_from_source_target_indices(self) -> None:
        adjacency = _FragmentTreeAdjacency.from_source_target_indices(
            source_indices=np.asarray([0, 0, 1], dtype=np.int64),
            target_indices=np.asarray([1, 2, 3], dtype=np.int64),
            num_nodes=4,
        )

        np.testing.assert_array_equal(
            adjacency.get_out_edge_indices(0),
            np.asarray([0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            adjacency.get_out_edge_indices(1),
            np.asarray([2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            adjacency.get_in_edge_indices(3),
            np.asarray([2], dtype=np.int64),
        )

    def test_empty_adjacency(self) -> None:
        adjacency = _FragmentTreeAdjacency.from_source_target_indices(
            source_indices=np.asarray([], dtype=np.int64),
            target_indices=np.asarray([], dtype=np.int64),
            num_nodes=2,
        )

        np.testing.assert_array_equal(
            adjacency.in_edge_indptr,
            np.asarray([0, 0, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            adjacency.out_edge_indptr,
            np.asarray([0, 0, 0], dtype=np.int64),
        )


class TestFragmentTreeDepths(unittest.TestCase):

    def test_build_depths(self) -> None:
        edges = make_edges()
        event_store = _CleavageEventStore.from_edges(edges)

        edge_store = _FragmentEdgeStore(
            edge_ids=np.asarray([201, 202, 203], dtype=np.int64),
            source_indices=np.asarray([0, 0, 1], dtype=np.int64),
            target_indices=np.asarray([1, 2, 3], dtype=np.int64),
            source_ids=np.asarray([101, 101, 102], dtype=np.int64),
            target_ids=np.asarray([102, 103, 104], dtype=np.int64),
            event_store=event_store,
        )

        adjacency = _FragmentTreeAdjacency.from_source_target_indices(
            source_indices=edge_store.source_indices,
            target_indices=edge_store.target_indices,
            num_nodes=4,
        )

        depths = _FragmentTreeDepths.build(
            num_nodes=4,
            edge_store=edge_store,
            adjacency=adjacency,
        )

        np.testing.assert_array_equal(
            depths.node_depths,
            np.asarray([0, 1, 1, 2], dtype=np.int64),
        )

    def test_get_nodes_by_depth(self) -> None:
        depths = _FragmentTreeDepths(
            node_depths=np.asarray([0, 1, 1, 2], dtype=np.int64),
        )

        nodes_by_depth = depths.get_nodes_by_depth()

        np.testing.assert_array_equal(
            nodes_by_depth[0],
            np.asarray([0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[1],
            np.asarray([1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[2],
            np.asarray([3], dtype=np.int64),
        )


class TestFragmentTree(unittest.TestCase):

    def test_from_nodes_and_edges(self) -> None:
        nodes = make_nodes()
        edges = make_edges()

        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=nodes,
            edges=edges,
        )

        self.assertEqual(tree.smiles, "CCO")
        self.assertEqual(tree.num_nodes, 4)
        self.assertEqual(tree.num_edges, 3)
        self.assertEqual(tree.num_events, 4)

        np.testing.assert_array_equal(
            tree.node_ids,
            np.asarray([101, 102, 103, 104], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            tree.edge_ids,
            np.asarray([201, 202, 203], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            tree.source_indices,
            np.asarray([0, 0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            tree.target_indices,
            np.asarray([1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            tree.source_ids,
            np.asarray([101, 101, 102], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            tree.target_ids,
            np.asarray([102, 103, 104], dtype=np.int64),
        )

    def test_get_node(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        node = tree.get_node(2)

        self.assertEqual(node.index, 2)
        self.assertEqual(node.id, 103)
        self.assertEqual(node.smiles, "O")

    def test_get_edge(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        edge = tree.get_edge(0)

        self.assertEqual(edge.index, 0)
        self.assertEqual(edge.id, 201)
        self.assertEqual(edge.source_index, 0)
        self.assertEqual(edge.target_index, 1)
        self.assertEqual(edge.source_id, 101)
        self.assertEqual(edge.target_id, 102)
        self.assertEqual(len(edge.events), 2)

    def test_get_in_edges(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        in_edges = tree.get_in_edges(3)

        self.assertEqual(len(in_edges), 1)
        self.assertEqual(in_edges[0].index, 2)
        self.assertEqual(in_edges[0].source_index, 1)
        self.assertEqual(in_edges[0].target_index, 3)

    def test_get_out_edges(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        out_edges = tree.get_out_edges(0)

        self.assertEqual(len(out_edges), 2)
        self.assertEqual(out_edges[0].index, 0)
        self.assertEqual(out_edges[1].index, 1)

    def test_get_parent_nodes(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        parent_nodes = tree.get_parent_nodes(3)

        self.assertEqual(len(parent_nodes), 1)
        self.assertEqual(parent_nodes[0].index, 1)
        self.assertEqual(parent_nodes[0].id, 102)
        self.assertEqual(parent_nodes[0].smiles, "CC")

    def test_get_child_nodes(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        child_nodes = tree.get_child_nodes(0)

        self.assertEqual(len(child_nodes), 2)
        self.assertEqual(child_nodes[0].index, 1)
        self.assertEqual(child_nodes[1].index, 2)

    def test_get_root_node_indices(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        np.testing.assert_array_equal(
            tree.get_root_node_indices(),
            np.asarray([0], dtype=np.int64),
        )

    def test_get_root_node_ids(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        np.testing.assert_array_equal(
            tree.get_root_node_ids(),
            np.asarray([101], dtype=np.int64),
        )

    def test_node_depths(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        np.testing.assert_array_equal(
            tree.node_depths,
            np.asarray([0, 1, 1, 2], dtype=np.int64),
        )

    def test_get_nodes_by_depth(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        nodes_by_depth = tree.get_nodes_by_depth()

        np.testing.assert_array_equal(
            nodes_by_depth[0],
            np.asarray([0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[1],
            np.asarray([1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[2],
            np.asarray([3], dtype=np.int64),
        )

    def test_from_nodes_and_edges_rejects_non_continuous_node_indices(self) -> None:
        nodes = (
            FragmentNode(index=0, id=101, smiles="CCO"),
            FragmentNode(index=2, id=102, smiles="CC"),
        )

        with self.assertRaises(ValueError):
            FragmentTree.from_nodes_and_edges(
                smiles="CCO",
                nodes=nodes,
                edges=(),
            )

    def test_from_nodes_and_edges_rejects_invalid_edge_node_index(self) -> None:
        nodes = make_nodes()

        edge = FragmentEdge(
            index=0,
            id=201,
            source_index=0,
            target_index=99,
            source_id=101,
            target_id=999,
            events=(),
        )

        with self.assertRaises(ValueError):
            FragmentTree.from_nodes_and_edges(
                smiles="CCO",
                nodes=nodes,
                edges=(edge,),
            )

    def test_copy(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        copied = tree.copy()

        self.assertEqual(copied.smiles, tree.smiles)
        self.assertEqual(copied.num_nodes, tree.num_nodes)
        self.assertEqual(copied.num_edges, tree.num_edges)
        self.assertIsNot(copied, tree)

        np.testing.assert_array_equal(copied.node_ids, tree.node_ids)
        np.testing.assert_array_equal(copied.edge_ids, tree.edge_ids)


if __name__ == "__main__":
    unittest.main()