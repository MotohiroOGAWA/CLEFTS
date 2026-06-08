from __future__ import annotations

import unittest

from clefts.domain.fragment.fragment_tree.FragmentTree import FragmentTree
from clefts.domain.fragment.fragment_tree.FragmentTreeBuilder import (
    FragmentTreeBuilder,
)
from clefts.libs.mmkit.mmkit import Compound

from tests.domain.fragment.fragment_tree_builder_cases import (
    FragmentTreeBuilderCase,
    FragmentTreeBuilderQuestion,
    make_fragment_tree_builder_questions,
)


def get_node_smiles_by_index(tree: FragmentTree) -> dict[int, str]:
    return {
        node_index: tree.get_node(node_index).smiles
        for node_index in range(tree.num_nodes)
    }


def get_node_smiles_set_at_depth(
    tree: FragmentTree,
    depth: int,
) -> set[str]:
    nodes_by_depth = tree.get_nodes_by_depth()

    if depth not in nodes_by_depth:
        return set()

    return {
        tree.get_node(int(node_index)).smiles
        for node_index in nodes_by_depth[depth]
    }


def get_edge_by_smiles_pair(
    tree: FragmentTree,
    source_smiles: str,
    target_smiles: str,
):
    node_smiles_by_index = get_node_smiles_by_index(tree)

    matched_edges = []

    for edge_index in range(tree.num_edges):
        edge = tree.get_edge(edge_index)

        actual_source_smiles = node_smiles_by_index[edge.source_index]
        actual_target_smiles = node_smiles_by_index[edge.target_index]

        if (
            actual_source_smiles == source_smiles
            and actual_target_smiles == target_smiles
        ):
            matched_edges.append(edge)

    return matched_edges


def get_edge_smiles_pairs(tree: FragmentTree) -> set[tuple[str, str]]:
    node_smiles_by_index = get_node_smiles_by_index(tree)

    return {
        (
            node_smiles_by_index[tree.get_edge(edge_index).source_index],
            node_smiles_by_index[tree.get_edge(edge_index).target_index],
        )
        for edge_index in range(tree.num_edges)
    }


class TestFragmentTreeBuilder(unittest.TestCase):
    """Tests for FragmentTreeBuilder.

    Notes
    -----
    Test data is defined outside this file.

    Each question consists of:
    - FragmentTreeBuilder
    - FragmentTreeBuilderCase list
    """

    def test_questions_have_unique_names(self) -> None:
        questions = make_fragment_tree_builder_questions()

        question_names = [
            question.name
            for question in questions
        ]

        self.assertEqual(len(question_names), len(set(question_names)))

    def test_cases_have_unique_names_in_each_question(self) -> None:
        for question in make_fragment_tree_builder_questions():
            with self.subTest(question=question.name):
                case_names = [
                    case.name
                    for case in question.cases
                ]

                self.assertEqual(len(case_names), len(set(case_names)))

    def test_builder_fields_for_questions(self) -> None:
        for question in make_fragment_tree_builder_questions():
            with self.subTest(question=question.name):
                builder = question.builder

                self.assertIsInstance(builder, FragmentTreeBuilder)
                self.assertEqual(
                    builder.cleavage_patterns,
                    builder.cleavage_pattern_set.patterns,
                )
                self.assertEqual(
                    builder.name,
                    builder.cleavage_pattern_set.name,
                )
                self.assertGreaterEqual(builder.max_depth, 1)

    def test_invalid_max_depth_raises_error(self) -> None:
        question = make_fragment_tree_builder_questions()[0]

        with self.assertRaises(ValueError):
            FragmentTreeBuilder(
                max_depth=0,
                cleavage_pattern_set=question.builder.cleavage_pattern_set,
            )

    def test_invalid_cleavage_pattern_set_raises_error(self) -> None:
        with self.assertRaises(TypeError):
            FragmentTreeBuilder(
                max_depth=1,
                cleavage_pattern_set="invalid",  # type: ignore[arg-type]
            )

    def test_build_returns_fragment_tree_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    tree = self._build_tree(
                        question=question,
                        case=case,
                    )

                    compound = Compound.from_smiles(case.smiles)

                    self.assertIsInstance(tree, FragmentTree)
                    self.assertEqual(tree.smiles, compound.smiles)

    def test_build_has_expected_counts_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    tree = self._build_tree(
                        question=question,
                        case=case,
                    )

                    self.assertEqual(tree.num_nodes, case.expected_num_nodes)
                    self.assertEqual(tree.num_edges, case.expected_num_edges)
                    self.assertGreaterEqual(
                        tree.num_events,
                        case.expected_min_events,
                    )

    def test_build_has_expected_nodes_by_depth_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    tree = self._build_tree(
                        question=question,
                        case=case,
                    )

                    for depth, expected_smiles in (
                        case.expected_node_smiles_by_depth.items()
                    ):
                        with self.subTest(
                            question=question.name,
                            case=case.name,
                            depth=depth,
                        ):
                            actual_smiles = get_node_smiles_set_at_depth(
                                tree=tree,
                                depth=depth,
                            )

                            self.assertEqual(
                                actual_smiles,
                                set(expected_smiles),
                            )

    def test_build_has_expected_edges_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    tree = self._build_tree(
                        question=question,
                        case=case,
                    )

                    for expected_edge in case.expected_edges:
                        with self.subTest(
                            question=question.name,
                            case=case.name,
                            source=expected_edge.source_smiles,
                            target=expected_edge.target_smiles,
                        ):
                            matched_edges = get_edge_by_smiles_pair(
                                tree=tree,
                                source_smiles=expected_edge.source_smiles,
                                target_smiles=expected_edge.target_smiles,
                            )

                            self.assertEqual(len(matched_edges), 1)

                            edge = matched_edges[0]

                            self.assertGreaterEqual(
                                len(edge.events),
                                expected_edge.min_event_count,
                            )

    def test_expected_edges_are_exactly_present_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    tree = self._build_tree(
                        question=question,
                        case=case,
                    )

                    actual_edge_pairs = get_edge_smiles_pairs(tree)

                    expected_edge_pairs = {
                        (
                            expected_edge.source_smiles,
                            expected_edge.target_smiles,
                        )
                        for expected_edge in case.expected_edges
                    }

                    self.assertEqual(actual_edge_pairs, expected_edge_pairs)

    def test_all_edges_have_consistent_source_and_target_ids_for_all_cases(
        self,
    ) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    tree = self._build_tree(
                        question=question,
                        case=case,
                    )

                    for edge_index in range(tree.num_edges):
                        edge = tree.get_edge(edge_index)

                        source_node = tree.get_node(edge.source_index)
                        target_node = tree.get_node(edge.target_index)

                        self.assertEqual(edge.source_id, source_node.id)
                        self.assertEqual(edge.target_id, target_node.id)
                        self.assertGreaterEqual(len(edge.events), 1)

    def test_all_events_have_valid_indices_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    tree = self._build_tree(
                        question=question,
                        case=case,
                    )

                    for edge_index in range(tree.num_edges):
                        edge = tree.get_edge(edge_index)

                        for event_index, event in enumerate(edge.events):
                            self.assertEqual(event.index, event_index)
                            self.assertGreaterEqual(
                                event.cleavage_pattern_id,
                                0,
                            )
                            self.assertGreaterEqual(event.reaction_id, 0)
                            self.assertGreaterEqual(
                                event.product_molecule_id,
                                0,
                            )
                            self.assertGreaterEqual(event.event_id, 0)
                            self.assertGreater(
                                len(event.reactant_indices),
                                0,
                            )
                            self.assertGreater(
                                len(event.product_indices),
                                0,
                            )

    def test_from_compound_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)
                    builder = question.builder

                    tree = builder.build(
                        compound=compound,
                        max_node=-1,
                        max_edge=-1,
                        print_info=False,
                    )

                    self.assertIsInstance(tree, FragmentTree)
                    self.assertEqual(tree.smiles, compound.smiles)
                    self.assertEqual(tree.num_nodes, case.expected_num_nodes)
                    self.assertEqual(tree.num_edges, case.expected_num_edges)
                    self.assertGreaterEqual(
                        tree.num_events,
                        case.expected_min_events,
                    )

    def test_build_respects_max_node_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    tree = question.builder.build(
                        compound,
                        max_node=1,
                        max_edge=-1,
                        print_info=False,
                    )

                    self.assertEqual(tree.num_nodes, 1)
                    self.assertEqual(tree.num_edges, 0)
                    self.assertEqual(tree.num_events, 0)

                    root_node = tree.get_node(0)
                    self.assertEqual(root_node.smiles, compound.smiles)

    def test_build_respects_max_edge_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    tree = question.builder.build(
                        compound,
                        max_node=-1,
                        max_edge=0,
                        print_info=False,
                    )

                    self.assertEqual(tree.num_nodes, 1)
                    self.assertEqual(tree.num_edges, 0)
                    self.assertEqual(tree.num_events, 0)

                    root_node = tree.get_node(0)
                    self.assertEqual(root_node.smiles, compound.smiles)

    def test_copy(self) -> None:
        question = make_fragment_tree_builder_questions()[0]
        builder = question.builder

        copied = builder.copy()

        self.assertEqual(copied, builder)
        self.assertIsNot(copied, builder)

    def _build_tree(
        self,
        *,
        question: FragmentTreeBuilderQuestion,
        case: FragmentTreeBuilderCase,
    ) -> FragmentTree:
        compound = Compound.from_smiles(case.smiles)

        return question.builder.build(
            compound,
            max_node=-1,
            max_edge=-1,
            print_info=False,
        )


if __name__ == "__main__":
    unittest.main()