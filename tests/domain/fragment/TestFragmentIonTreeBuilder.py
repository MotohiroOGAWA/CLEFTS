from __future__ import annotations

import unittest

from clefts.libs.mmkit.mmkit import Compound

from clefts.domain.fragment.ion_tree.FragmentIonTree import (
    FragmentIonTree,
)
from clefts.domain.fragment.ion_tree.FragmentIonTreeBuilder import (
    FragmentIonTreeBuilder,
)

from .fragment_tree_builder_cases import (
    ExpectedFragmentTreeEdge,
    FragmentTreeBuilderQuestion,
    make_fragment_tree_builder_questions,
)


class TestFragmentIonTreeBuilder(unittest.TestCase):
    """Tests for FragmentIonTreeBuilder."""

    def test_build_returns_fragment_ion_tree(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    fragment_ion_tree = builder.build(compound)

                    self.assertIsInstance(fragment_ion_tree, FragmentIonTree)
                    self.assertEqual(fragment_ion_tree.smiles, compound.smiles)

    def test_build_keeps_expected_tree_size(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    self.assertEqual(
                        fragment_ion_tree.num_nodes,
                        case.expected_num_nodes,
                    )
                    self.assertEqual(
                        fragment_ion_tree.num_edges,
                        case.expected_num_edges,
                    )

    def test_build_keeps_expected_nodes_by_depth(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    actual_node_smiles_by_depth = (
                        self._collect_node_smiles_by_depth(fragment_ion_tree)
                    )

                    self.assertEqual(
                        actual_node_smiles_by_depth,
                        case.expected_node_smiles_by_depth,
                    )

    def test_build_keeps_expected_edges(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    for expected_edge in case.expected_edges:
                        edge = self._find_edge_by_smiles(
                            fragment_ion_tree,
                            expected_edge,
                        )

                        self.assertIsNotNone(edge)

                        event_count = len(edge.events)
                        self.assertGreaterEqual(
                            event_count,
                            expected_edge.min_event_count,
                        )

    def test_build_has_expected_min_events(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    actual_event_count = sum(
                        len(fragment_ion_tree.get_edge(edge_index).events)
                        for edge_index in range(fragment_ion_tree.num_edges)
                    )

                    self.assertGreaterEqual(
                        actual_event_count,
                        case.expected_min_events,
                    )

    def test_build_has_expected_hydrogen_state_candidates(self) -> None:
        """Hydrogen state candidates are grouped by adduct type, not by node."""

        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    actual_candidates_by_adduct_type = (
                        self._collect_hydrogen_state_candidates_by_adduct_type(
                            fragment_ion_tree
                        )
                    )

                    expected_candidates_by_adduct_type = (
                        self._make_expected_hydrogen_state_candidates_by_adduct_type(
                            question
                        )
                    )

                    self.assertEqual(
                        actual_candidates_by_adduct_type,
                        expected_candidates_by_adduct_type,
                    )

    def test_build_has_expected_declared_hydrogen_states(self) -> None:
        """Declared hydrogen states should match radical/unsaturation settings."""

        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    store = fragment_ion_tree.hydrogen_state_candidate_store

                    actual_declared_states = tuple(
                        tuple(int(value) for value in row)
                        for row in store.declared_states
                    )

                    expected_declared_states = tuple(
                        (
                            int(rule.unsaturation),
                            int(bool(rule.radical)),
                        )
                        for rule in (
                            question.fragment_ion_adduct_rule_set.adduct_rules
                        )
                    )

                    self.assertEqual(
                        actual_declared_states,
                        expected_declared_states,
                    )

    def test_build_has_expected_shift_rule_mask_by_smiles(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                if case.expected_shift_rule_mask_by_smiles is None:
                    continue

                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    actual_shift_rule_mask_by_smiles = (
                        self._collect_shift_rule_mask_by_smiles(
                            fragment_ion_tree
                        )
                    )

                    self.assertEqual(
                        actual_shift_rule_mask_by_smiles,
                        case.expected_shift_rule_mask_by_smiles,
                    )

    def test_build_has_valid_shift_rule_adduct_types(self) -> None:
        """adduct_types are stored per adduct rule, not per shift rule."""

        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    store = fragment_ion_tree.ion_shift_candidate_store
                    rule_set = question.fragment_ion_adduct_rule_set

                    self.assertEqual(
                        len(store.adduct_types),
                        len(rule_set.adduct_rules),
                    )

                    self.assertEqual(
                        tuple(str(adduct_type) for adduct_type in store.adduct_types),
                        tuple(
                            str(rule.adduct_type)
                            for rule in rule_set.adduct_rules
                        ),
                    )

                    self.assertEqual(
                        len(store.adduct_shift_rule_indptr),
                        len(rule_set.adduct_rules) + 1,
                    )

                    self.assertEqual(
                        int(store.adduct_shift_rule_indptr[0]),
                        0,
                    )

                    self.assertEqual(
                        int(store.adduct_shift_rule_indptr[-1]),
                        store.num_shift_rules,
                    )

                    for adduct_rule_index, adduct_rule in enumerate(
                        rule_set.adduct_rules
                    ):
                        expected_start = int(
                            store.adduct_shift_rule_indptr[
                                adduct_rule_index
                            ]
                        )
                        expected_end = int(
                            store.adduct_shift_rule_indptr[
                                adduct_rule_index + 1
                            ]
                        )

                        self.assertEqual(
                            expected_end - expected_start,
                            len(adduct_rule.ion_shifts),
                        )

                    for shift_rule_index in range(store.num_shift_rules):
                        adduct_rule_index, _ion_shift_index = (
                            store.get_shift_rule_position(shift_rule_index)
                        )

                        expected_adduct_type = str(
                            rule_set
                            .adduct_rules[adduct_rule_index]
                            .adduct_type
                        )

                        actual_adduct_type = str(
                            store.get_adduct_type_for_shift_rule(
                                shift_rule_index
                            )
                        )

                        self.assertEqual(
                            actual_adduct_type,
                            expected_adduct_type,
                        )

    def test_copy_returns_fragment_ion_tree_builder(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            copied = builder.copy()

            self.assertIsInstance(copied, FragmentIonTreeBuilder)
            self.assertEqual(copied.max_depth, builder.max_depth)
            self.assertEqual(
                copied.only_add_min_depth,
                builder.only_add_min_depth,
            )
            self.assertEqual(
                copied.min_depth_only_from,
                builder.min_depth_only_from,
            )
            self.assertIsNot(
                copied.cleavage_pattern_set,
                builder.cleavage_pattern_set,
            )
            self.assertIsNot(
                copied.fragment_ion_adduct_rule_set,
                builder.fragment_ion_adduct_rule_set,
            )

    def _make_fragment_ion_tree_builder(
        self,
        question: FragmentTreeBuilderQuestion,
    ) -> FragmentIonTreeBuilder:
        if question.fragment_ion_adduct_rule_set is None:
            raise ValueError(
                "question.fragment_ion_adduct_rule_set must not be None."
            )

        return FragmentIonTreeBuilder(
            max_depth=question.builder.max_depth,
            cleavage_pattern_set=question.builder.cleavage_pattern_set.copy(),
            only_add_min_depth=question.builder.only_add_min_depth,
            min_depth_only_from=question.builder.min_depth_only_from,
            fragment_ion_adduct_rule_set=(
                question.fragment_ion_adduct_rule_set.copy()
            ),
        )

    def _collect_node_smiles_by_depth(
        self,
        fragment_ion_tree: FragmentIonTree,
    ) -> dict[int, tuple[str, ...]]:
        node_smiles_by_depth: dict[int, list[str]] = {}

        for node_index in range(fragment_ion_tree.num_nodes):
            node = fragment_ion_tree.get_node(node_index)
            depth = int(fragment_ion_tree.node_depths[node_index])

            node_smiles_by_depth.setdefault(depth, []).append(node.smiles)

        return {
            depth: tuple(smiles_list)
            for depth, smiles_list in sorted(node_smiles_by_depth.items())
        }

    def _find_edge_by_smiles(
        self,
        fragment_ion_tree: FragmentIonTree,
        expected_edge: ExpectedFragmentTreeEdge,
    ):
        for edge_index in range(fragment_ion_tree.num_edges):
            edge = fragment_ion_tree.get_edge(edge_index)

            source_node = fragment_ion_tree.get_node(edge.source_index)
            target_node = fragment_ion_tree.get_node(edge.target_index)

            if (
                source_node.smiles == expected_edge.source_smiles
                and target_node.smiles == expected_edge.target_smiles
            ):
                return edge

        return None

    def _collect_hydrogen_state_candidates_by_adduct_type(
        self,
        fragment_ion_tree: FragmentIonTree,
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        store = fragment_ion_tree.hydrogen_state_candidate_store

        candidates_by_adduct_type: dict[str, tuple[tuple[int, int], ...]] = {}

        for adduct_index, adduct_type in enumerate(store.adduct_types):
            start = int(store.adduct_state_indptr[adduct_index])
            end = int(store.adduct_state_indptr[adduct_index + 1])

            candidates_by_adduct_type[str(adduct_type)] = tuple(
                tuple(int(value) for value in row)
                for row in store.candidate_states[start:end]
            )

        return candidates_by_adduct_type

    def _make_expected_hydrogen_state_candidates_by_adduct_type(
        self,
        question: FragmentTreeBuilderQuestion,
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        if question.fragment_ion_adduct_rule_set is None:
            raise ValueError(
                "question.fragment_ion_adduct_rule_set must not be None."
            )

        expected: dict[str, tuple[tuple[int, int], ...]] = {}

        for rule in question.fragment_ion_adduct_rule_set.adduct_rules:
            unsaturation = int(rule.unsaturation)
            radical = int(bool(rule.radical))

            states: list[tuple[int, int]] = []

            for current_unsaturation in range(unsaturation + 1):
                states.append((current_unsaturation, 0))

            if radical:
                states.append((unsaturation, 1))

            expected[str(rule.adduct_type)] = tuple(states)

        return expected

    def _collect_shift_rule_mask_by_smiles(
        self,
        fragment_ion_tree: FragmentIonTree,
    ) -> dict[str, tuple[bool, ...]]:
        shift_rule_mask_by_smiles: dict[str, tuple[bool, ...]] = {}

        store = fragment_ion_tree.ion_shift_candidate_store

        for node_index in range(fragment_ion_tree.num_nodes):
            node = fragment_ion_tree.get_node(node_index)

            shift_rule_mask_by_smiles[node.smiles] = tuple(
                bool(value)
                for value in store.node_shift_rule_mask[node_index]
            )

        return shift_rule_mask_by_smiles


if __name__ == "__main__":
    unittest.main()