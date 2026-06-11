from __future__ import annotations

import unittest

import numpy as np

from clefts.libs.mmkit.mmkit import Compound

from clefts.domain.fragment.tree.FragmentNode import FragmentNode
from clefts.domain.fragment.tree.FragmentTree import FragmentTree
from clefts.domain.fragment.ion_tree.FragmentIonTree import FragmentIonTree
from clefts.domain.fragment.ion_tree.FragmentIonAdductRuleSet import (
    FragmentIonAdductRuleSet,
)
from clefts.domain.fragment.ion_tree._private._FragmentHydrogenStateCandidateStore import (
    _FragmentHydrogenStateCandidateStore,
)
from clefts.domain.fragment.ion_tree._private._FragmentIonShiftCandidateStore import (
    _FragmentIonShiftCandidateStore,
)


class TestFragmentIonTree(unittest.TestCase):
    def setUp(self) -> None:
        self.fragment_tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=(
                FragmentNode(index=0, id=100, smiles="CCO"),
                FragmentNode(index=1, id=101, smiles="CC"),
                FragmentNode(index=2, id=102, smiles="[NH4+]"),
                FragmentNode(index=3, id=103, smiles="Cl"),
            ),
            edges=(),
        )

        self.rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "adduct_rules": [
                    {
                        "name": "pos_H_r1_u2",
                        "adduct_type": "[M+H]+",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "atoms": ["C", "P", "S"],
                                "ion_shift": "[M-H]+",
                            },
                            {
                                "atoms": ["P", "S", "N", "O"],
                                "ion_shift": "[M+H]+",
                            },
                        ],
                    },
                    {
                        "name": "pos_Na_r1_u2",
                        "adduct_type": "[M+Na]+",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "ion_shift": "[M+Na]+",
                            },
                            {
                                "atoms": ["C", "P", "S"],
                                "ion_shift": "[M-H]+",
                            },
                            {
                                "atoms": ["P", "S", "N", "O"],
                                "ion_shift": "[M+H]+",
                            },
                        ],
                    },
                    {
                        "name": "pos_NH4_r1_u2",
                        "adduct_type": "[M+NH4]+",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "ion_shift": "[M+NH4]+",
                            },
                            {
                                "atoms": ["P", "S", "N", "O"],
                                "ion_shift": "[M+H]+",
                            },
                            {
                                "atoms": ["C", "P", "S"],
                                "ion_shift": "[M-H]+",
                            },
                        ],
                    },
                    {
                        "name": "pos_2M_H_to_M_r1_u2",
                        "adduct_type": "[2M+H]+",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "atoms": ["P", "S", "N", "O"],
                                "ion_shift": "[M+H]+",
                            },
                            {
                                "atoms": ["C", "P", "S"],
                                "ion_shift": "[M-H]+",
                            },
                        ],
                    },
                ],
            },
        )

    def test_from_fragment_tree_keeps_original_topology(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        self.assertEqual(fragment_ion_tree.smiles, self.fragment_tree.smiles)
        self.assertEqual(fragment_ion_tree.num_nodes, self.fragment_tree.num_nodes)
        self.assertEqual(fragment_ion_tree.num_edges, self.fragment_tree.num_edges)

        np.testing.assert_array_equal(
            fragment_ion_tree.node_ids,
            self.fragment_tree.node_ids,
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.node_smiles,
            self.fragment_tree.node_smiles,
        )

        self.assertEqual(fragment_ion_tree.get_node(0).smiles, "CCO")
        self.assertEqual(fragment_ion_tree.get_node(1).smiles, "CC")
        self.assertEqual(fragment_ion_tree.get_node(2).smiles, "[NH4+]")
        self.assertEqual(fragment_ion_tree.get_node(3).smiles, "Cl")

    def test_from_fragment_tree_builds_hydrogen_state_candidate_store(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        store = fragment_ion_tree.hydrogen_state_candidate_store

        self.assertEqual(store.num_adduct_types, 4)
        self.assertEqual(store.num_candidate_states, 24)

        self.assertEqual(
            tuple(str(adduct_type) for adduct_type in store.adduct_types),
            (
                "[M+H]+",
                "[M+Na]+",
                "[M+H4N]+",
                "[2M+H]+",
            ),
        )

        expected_declared_states = np.asarray(
            [
                [2, 1],
                [2, 1],
                [2, 1],
                [2, 1],
            ],
            dtype=np.int16,
        )

        np.testing.assert_array_equal(
            store.declared_states,
            expected_declared_states,
        )

        expected_candidates_for_one_adduct = np.asarray(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 0],
                [2, 1],
            ],
            dtype=np.int16,
        )

        for adduct_index in range(store.num_adduct_types):
            np.testing.assert_array_equal(
                store.get_candidate_states(adduct_index),
                expected_candidates_for_one_adduct,
            )

            np.testing.assert_array_equal(
                store.get_candidate_delta_h(adduct_index),
                np.asarray([0, -1, -2, -3, -4, -5], dtype=np.int16),
            )

    def test_from_fragment_tree_builds_shift_candidate_store(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        store = fragment_ion_tree.ion_shift_candidate_store

        self.assertEqual(store.num_nodes, self.fragment_tree.num_nodes)
        self.assertEqual(store.num_adduct_types, 4)
        self.assertEqual(store.num_shift_rules, 10)

        self.assertEqual(
            tuple(str(adduct_type) for adduct_type in store.adduct_types),
            (
                "[M+H]+",
                "[M+Na]+",
                "[M+H4N]+",
                "[2M+H]+",
            ),
        )

        np.testing.assert_array_equal(
            store.adduct_shift_rule_indptr,
            np.asarray([0, 2, 5, 8, 10], dtype=np.int64),
        )

        expected_shift_rule_indices = np.asarray(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
                [2, 2],
                [3, 0],
                [3, 1],
            ],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.shift_rule_indices,
            expected_shift_rule_indices,
        )

        self.assertEqual(
            fragment_ion_tree.node_shift_rule_mask.shape,
            (self.fragment_tree.num_nodes, 10),
        )

    def test_node_adduct_type_mask_respects_shift_ranges(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_adduct_type_mask(0),
            np.asarray([True, True, True, True], dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_adduct_type_mask(1),
            np.asarray([True, True, True, True], dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_adduct_type_mask(2),
            np.asarray([False, False, False, False], dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_adduct_type_mask(3),
            np.asarray([False, True, True, False], dtype=bool),
        )

    def test_node_shift_rule_mask_respects_atoms_and_charge(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(0),
            np.asarray(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                dtype=bool,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(1),
            np.asarray(
                [
                    True,
                    False,
                    True,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ],
                dtype=bool,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(2),
            np.zeros(10, dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(3),
            np.asarray(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                dtype=bool,
            ),
        )

    def test_get_applicable_shift_rule_indices_for_adduct_rule(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        shift_rule_indices = (
            fragment_ion_tree
            .get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=1,
                adduct_rule_index=1,
            )
        )

        np.testing.assert_array_equal(
            shift_rule_indices,
            np.asarray([2, 3], dtype=np.int64),
        )

        ion_shift_rules = (
            fragment_ion_tree
            .get_applicable_ion_shift_rules_for_adduct_rule(
                node_index=1,
                adduct_rule_index=1,
            )
        )

        self.assertEqual(len(ion_shift_rules), 2)
        self.assertEqual(str(ion_shift_rules[0].ion_shift), "[M+Na]+")
        self.assertEqual(str(ion_shift_rules[1].ion_shift), "[M-H]+")

    def test_get_applicable_shift_rule_indices_for_adduct_type(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        shift_rule_indices = (
            fragment_ion_tree
            .get_applicable_shift_rule_indices_for_adduct_type(
                node_index=1,
                adduct_type="[M+Na]+",
            )
        )

        np.testing.assert_array_equal(
            shift_rule_indices,
            np.asarray([2, 3], dtype=np.int64),
        )

    def test_get_ion_shift_rule_by_shift_rule_index(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        adduct_rule_index, ion_shift_index = (
            fragment_ion_tree.get_shift_rule_position(4)
        )

        self.assertEqual(adduct_rule_index, 1)
        self.assertEqual(ion_shift_index, 2)
        self.assertEqual(
            str(fragment_ion_tree.get_shift_rule_adduct_type(4)),
            "[M+Na]+",
        )

        ion_shift_rule = fragment_ion_tree.get_ion_shift_rule(4)

        self.assertEqual(str(ion_shift_rule.ion_shift), "[M+H]+")
        self.assertEqual(ion_shift_rule.atoms, ("P", "S", "N", "O"))

    def test_empty_rule_set_builds_empty_stores(self) -> None:
        empty_rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "adduct_rules": [],
            },
        )

        fragment_ion_tree = self._make_fragment_ion_tree(
            rule_set=empty_rule_set,
        )

        self.assertEqual(fragment_ion_tree.num_adduct_rules, 0)
        self.assertEqual(fragment_ion_tree.num_shift_rules, 0)
        self.assertEqual(fragment_ion_tree.num_hydrogen_state_candidates, 0)

        self.assertEqual(
            fragment_ion_tree.hydrogen_state_candidates.shape,
            (0, 2),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.adduct_state_indptr,
            np.zeros(1, dtype=np.int64),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.ion_shift_candidate_store.adduct_shift_rule_indptr,
            np.zeros(1, dtype=np.int64),
        )

        self.assertEqual(
            fragment_ion_tree.node_shift_rule_mask.shape,
            (self.fragment_tree.num_nodes, 0),
        )

    def test_rule_with_no_ion_shifts_builds_state_but_no_shift(self) -> None:
        rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "adduct_rules": [
                    {
                        "name": "pos_empty_shift_r0_u1",
                        "adduct_type": "[M+H]+",
                        "radical": False,
                        "unsaturation": 1,
                        "ion_shifts": [],
                    }
                ],
            },
        )

        fragment_ion_tree = self._make_fragment_ion_tree(rule_set=rule_set)

        self.assertEqual(fragment_ion_tree.num_adduct_rules, 1)
        self.assertEqual(fragment_ion_tree.num_shift_rules, 0)

        np.testing.assert_array_equal(
            fragment_ion_tree.ion_shift_candidate_store.adduct_shift_rule_indptr,
            np.asarray([0, 0], dtype=np.int64),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_hydrogen_state_candidates_for_adduct_rule(0),
            np.asarray(
                [
                    [0, 0],
                    [1, 0],
                ],
                dtype=np.int16,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree
            .get_hydrogen_state_candidate_delta_h_for_adduct_rule(0),
            np.asarray([0, -2], dtype=np.int16),
        )

        self.assertEqual(
            fragment_ion_tree.node_shift_rule_mask.shape,
            (self.fragment_tree.num_nodes, 0),
        )

    def test_multiple_hydrogen_state_candidates_are_grouped_by_adduct_type(
        self,
    ) -> None:
        rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "adduct_rules": [
                    {
                        "name": "pos_H_r1_u2",
                        "adduct_type": "[M+H]+",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "ion_shift": "[M+H]+",
                            }
                        ],
                    },
                    {
                        "name": "pos_Na_r0_u1",
                        "adduct_type": "[M+Na]+",
                        "radical": False,
                        "unsaturation": 1,
                        "ion_shifts": [
                            {
                                "ion_shift": "[M+Na]+",
                            }
                        ],
                    },
                    {
                        "name": "pos_NH4_r1_u2",
                        "adduct_type": "[M+NH4]+",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "ion_shift": "[M+NH4]+",
                            }
                        ],
                    },
                ],
            },
        )

        fragment_ion_tree = self._make_fragment_ion_tree(rule_set=rule_set)

        np.testing.assert_array_equal(
            fragment_ion_tree.get_hydrogen_state_candidates_for_adduct_type(
                "[M+H]+"
            ),
            np.asarray(
                [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [2, 0],
                    [2, 1],
                ],
                dtype=np.int16,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_hydrogen_state_candidates_for_adduct_type(
                "[M+Na]+"
            ),
            np.asarray(
                [
                    [0, 0],
                    [1, 0],
                ],
                dtype=np.int16,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_hydrogen_state_candidates_for_adduct_type(
                "[M+NH4]+"
            ),
            np.asarray(
                [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [2, 0],
                    [2, 1],
                ],
                dtype=np.int16,
            ),
        )

    def test_generic_ion_shift_is_false_for_charged_fragment(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(2),
            np.zeros(fragment_ion_tree.num_shift_rules, dtype=bool),
        )

    def test_negative_charged_fragment_disables_all_ion_shifts(self) -> None:
        fragment_tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=(
                FragmentNode(index=0, id=200, smiles="[O-]"),
                FragmentNode(index=1, id=201, smiles="O"),
            ),
            edges=(),
        )

        fragment_ion_tree = self._make_fragment_ion_tree(
            fragment_tree=fragment_tree,
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(0),
            np.zeros(fragment_ion_tree.num_shift_rules, dtype=bool),
        )

        self.assertTrue(
            np.any(fragment_ion_tree.get_node_shift_rule_mask(1))
        )

    def test_atom_specific_rules_for_single_element_fragments(self) -> None:
        fragment_tree = FragmentTree.from_nodes_and_edges(
            smiles="COPSCl",
            nodes=(
                FragmentNode(index=0, id=300, smiles="C"),
                FragmentNode(index=1, id=301, smiles="O"),
                FragmentNode(index=2, id=302, smiles="N"),
                FragmentNode(index=3, id=303, smiles="S"),
                FragmentNode(index=4, id=304, smiles="P"),
                FragmentNode(index=5, id=305, smiles="Cl"),
            ),
            edges=(),
        )

        fragment_ion_tree = self._make_fragment_ion_tree(
            fragment_tree=fragment_tree,
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(0),
            np.asarray(
                [
                    True,
                    False,
                    True,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ],
                dtype=bool,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(1),
            np.asarray(
                [
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                    True,
                    False,
                ],
                dtype=bool,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(2),
            np.asarray(
                [
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                    True,
                    False,
                ],
                dtype=bool,
            ),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(3),
            np.ones(10, dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(4),
            np.ones(10, dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(5),
            np.asarray(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                dtype=bool,
            ),
        )

    def test_get_applicable_adduct_rule_indices_for_each_node(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(0),
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(1),
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(2),
            np.asarray([], dtype=np.int64),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(3),
            np.asarray([1, 2], dtype=np.int64),
        )

    def test_has_applicable_shift_for_adduct_rule(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        self.assertTrue(
            fragment_ion_tree.has_applicable_shift_for_adduct_rule(
                node_index=0,
                adduct_rule_index=0,
            )
        )

        self.assertFalse(
            fragment_ion_tree.has_applicable_shift_for_adduct_rule(
                node_index=2,
                adduct_rule_index=0,
            )
        )

        self.assertFalse(
            fragment_ion_tree.has_applicable_shift_for_adduct_rule(
                node_index=3,
                adduct_rule_index=0,
            )
        )

        self.assertTrue(
            fragment_ion_tree.has_applicable_shift_for_adduct_rule(
                node_index=3,
                adduct_rule_index=1,
            )
        )

    def test_invalid_adduct_rule_index_raises(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_adduct_rule(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_adduct_rule(100)

    def test_invalid_shift_rule_index_raises(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_shift_rule_position(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_shift_rule_position(
                fragment_ion_tree.num_shift_rules
            )

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_ion_shift_rule(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_ion_shift_rule(
                fragment_ion_tree.num_shift_rules
            )

    def test_invalid_node_index_for_shift_raises(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_node_shift_rule_mask(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_node_shift_rule_mask(
                fragment_ion_tree.num_nodes
            )

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices(
                fragment_ion_tree.num_nodes
            )

    def test_invalid_adduct_rule_index_for_applicable_shift_raises(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=0,
                adduct_rule_index=-1,
            )

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=0,
                adduct_rule_index=fragment_ion_tree.num_adduct_rules,
            )

    def test_invalid_node_index_for_applicable_shift_raises(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=-1,
                adduct_rule_index=0,
            )

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices_for_adduct_rule(
                node_index=fragment_ion_tree.num_nodes,
                adduct_rule_index=0,
            )

    def test_invalid_fragment_tree_argument_raises(self) -> None:
        with self.assertRaises(TypeError):
            FragmentIonTree.from_fragment_tree(
                object(),
                fragment_ion_adduct_rule_set=self.rule_set,
                hydrogen_state_candidate_store=(
                    self._make_hydrogen_state_candidate_store(self.rule_set)
                ),
                ion_shift_candidate_store=(
                    self._make_ion_shift_candidate_store(
                        fragment_tree=self.fragment_tree,
                        rule_set=self.rule_set,
                    )
                ),
            )

    def test_invalid_rule_set_argument_raises(self) -> None:
        with self.assertRaises(TypeError):
            FragmentIonTree.from_fragment_tree(
                self.fragment_tree,
                fragment_ion_adduct_rule_set=object(),
                hydrogen_state_candidate_store=(
                    self._make_hydrogen_state_candidate_store(self.rule_set)
                ),
                ion_shift_candidate_store=(
                    self._make_ion_shift_candidate_store(
                        fragment_tree=self.fragment_tree,
                        rule_set=self.rule_set,
                    )
                ),
            )

    def test_invalid_smiles_raises(self) -> None:
        fragment_tree = FragmentTree.from_nodes_and_edges(
            smiles="invalid",
            nodes=(
                FragmentNode(index=0, id=400, smiles="this_is_not_smiles"),
            ),
            edges=(),
        )

        with self.assertRaises(ValueError):
            self._make_fragment_ion_tree(fragment_tree=fragment_tree)

    def test_negative_rule_set(self) -> None:
        negative_rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "adduct_rules": [
                    {
                        "name": "neg_H_r1_u2",
                        "adduct_type": "[M-H]-",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "atoms": ["C", "P", "S", "N", "O"],
                                "ion_shift": "[M-H]-",
                            }
                        ],
                    },
                    {
                        "name": "neg_2M_H_to_M_r1_u2",
                        "adduct_type": "[2M-H]-",
                        "radical": True,
                        "unsaturation": 2,
                        "ion_shifts": [
                            {
                                "atoms": ["P", "S", "N", "O"],
                                "ion_shift": "[M-H]-",
                            },
                            {
                                "atoms": ["C", "P", "S"],
                                "ion_shift": "[M-H]-",
                            },
                        ],
                    },
                ],
            },
        )

        fragment_ion_tree = self._make_fragment_ion_tree(
            rule_set=negative_rule_set,
        )

        self.assertEqual(fragment_ion_tree.num_adduct_rules, 2)
        self.assertEqual(fragment_ion_tree.num_shift_rules, 3)

        expected_shift_rule_indices = np.asarray(
            [
                [0, 0],
                [1, 0],
                [1, 1],
            ],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.shift_rule_indices,
            expected_shift_rule_indices,
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(0),
            np.asarray([True, True, True], dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(1),
            np.asarray([True, False, True], dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(2),
            np.asarray([False, False, False], dtype=bool),
        )

        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(3),
            np.asarray([False, False, False], dtype=bool),
        )

    def _make_fragment_ion_tree(
        self,
        *,
        fragment_tree: FragmentTree | None = None,
        rule_set: FragmentIonAdductRuleSet | None = None,
    ) -> FragmentIonTree:
        fragment_tree = self.fragment_tree if fragment_tree is None else fragment_tree
        rule_set = self.rule_set if rule_set is None else rule_set

        hydrogen_state_candidate_store = (
            self._make_hydrogen_state_candidate_store(rule_set)
        )

        ion_shift_candidate_store = self._make_ion_shift_candidate_store(
            fragment_tree=fragment_tree,
            rule_set=rule_set,
        )

        return FragmentIonTree.from_fragment_tree(
            fragment_tree,
            fragment_ion_adduct_rule_set=rule_set,
            hydrogen_state_candidate_store=hydrogen_state_candidate_store,
            ion_shift_candidate_store=ion_shift_candidate_store,
        )

    def _make_hydrogen_state_candidate_store(
        self,
        rule_set: FragmentIonAdductRuleSet,
    ) -> _FragmentHydrogenStateCandidateStore:
        return _FragmentHydrogenStateCandidateStore.from_adduct_rule_set(
            rule_set
        )

    def _make_ion_shift_candidate_store(
        self,
        *,
        fragment_tree: FragmentTree,
        rule_set: FragmentIonAdductRuleSet,
    ) -> _FragmentIonShiftCandidateStore:
        node_atom_symbols: list[frozenset[str]] = []
        node_charges: list[int] = []

        for node_index in range(fragment_tree.num_nodes):
            node = fragment_tree.get_node(node_index)
            compound = Compound.from_smiles(node.smiles)
            mol = self._get_mol_from_compound(compound)

            node_atom_symbols.append(
                frozenset(
                    atom.GetSymbol()
                    for atom in mol.GetAtoms()
                )
            )

            node_charges.append(
                int(
                    sum(
                        atom.GetFormalCharge()
                        for atom in mol.GetAtoms()
                    )
                )
            )

        return _FragmentIonShiftCandidateStore.from_adduct_rule_set(
            adduct_rule_set=rule_set,
            node_atom_symbols=tuple(node_atom_symbols),
            node_charges=tuple(node_charges),
        )

    @staticmethod
    def _get_mol_from_compound(
        compound: Compound,
    ):
        if hasattr(compound, "mol"):
            return compound.mol

        if hasattr(compound, "rdmol"):
            return compound.rdmol

        if hasattr(compound, "to_mol"):
            return compound.to_mol()

        raise TypeError(
            "Compound must expose mol, rdmol, or to_mol()."
        )


if __name__ == "__main__":
    unittest.main()