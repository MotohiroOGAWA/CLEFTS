from __future__ import annotations

import unittest

import numpy as np

from clefts.domain.fragment.tree.FragmentNode import FragmentNode
from clefts.domain.fragment.tree.FragmentTree import FragmentTree
from clefts.domain.fragment.ion_tree.FragmentIonTree import FragmentIonTree
from clefts.domain.fragment.ion_tree.FragmentIonAdductRuleSet import (
    FragmentIonAdductRuleSet,
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
            name="positive",
        )

    def test_from_fragment_tree_keeps_original_topology(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

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

    def test_from_fragment_tree_builds_ion_state_store(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        self.assertEqual(
            fragment_ion_tree.ion_state_store.num_nodes,
            self.fragment_tree.num_nodes,
        )

        self.assertEqual(
            len(fragment_ion_tree.node_state_indptr),
            self.fragment_tree.num_nodes + 1,
        )

        # All rules have the same ion state:
        # unsaturation = 2, radical = True.
        # Therefore each node should have one deduplicated state: [2, 1].
        for node_index in range(fragment_ion_tree.num_nodes):
            node_states = fragment_ion_tree.get_node_ion_states(node_index)

            self.assertEqual(node_states.shape, (1, 2))
            np.testing.assert_array_equal(
                node_states,
                np.asarray([[2, 1]], dtype=np.int16),
            )

            node_delta_h = fragment_ion_tree.get_node_ion_state_delta_h(
                node_index
            )

            # delta_h = -2 * unsaturation - radical
            #         = -2 * 2 - 1
            #         = -5
            np.testing.assert_array_equal(
                node_delta_h,
                np.asarray([-5], dtype=np.int16),
            )

    def test_from_fragment_tree_builds_shift_store(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        self.assertEqual(
            fragment_ion_tree.ion_shift_store.num_nodes,
            self.fragment_tree.num_nodes,
        )

        # The JSON has 10 IonShiftRule entries in total.
        self.assertEqual(fragment_ion_tree.num_shift_rules, 10)

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

    def test_node_shift_rule_mask_respects_atoms_and_charge(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        # node 0: CCO
        # Contains C and O.
        # Therefore C/P/S rules and P/S/N/O rules are applicable.
        # Generic [M+Na]+ and [M+NH4]+ are also applicable.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(0),
            np.asarray(
                [
                    True,   # rule 0, shift 0: C/P/S -> [M-H]+
                    True,   # rule 0, shift 1: P/S/N/O -> [M+H]+
                    True,   # rule 1, shift 0: generic [M+Na]+
                    True,   # rule 1, shift 1: C/P/S -> [M-H]+
                    True,   # rule 1, shift 2: P/S/N/O -> [M+H]+
                    True,   # rule 2, shift 0: generic [M+NH4]+
                    True,   # rule 2, shift 1: P/S/N/O -> [M+H]+
                    True,   # rule 2, shift 2: C/P/S -> [M-H]+
                    True,   # rule 3, shift 0: P/S/N/O -> [M+H]+
                    True,   # rule 3, shift 1: C/P/S -> [M-H]+
                ],
                dtype=bool,
            ),
        )

        # node 1: CC
        # Contains C only.
        # C/P/S rules are applicable.
        # P/S/N/O rules are not applicable.
        # Generic rules are applicable.
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

        # node 2: [NH4+]
        # Already charged.
        # No ion shift should be applicable.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(2),
            np.asarray(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                dtype=bool,
            ),
        )

        # node 3: Cl
        # Neutral, but Cl is not in any atom-specific rule.
        # Only generic [M+Na]+ and [M+NH4]+ rules are applicable.
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
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        # node 1: CC
        # adduct_rule_index 1: [M+Na]+
        #
        # [M+Na]+ rule has shift_rule_index:
        #   2 -> generic [M+Na]+
        #   3 -> C/P/S [M-H]+
        #   4 -> P/S/N/O [M+H]+
        #
        # CC allows 2 and 3, but not 4.
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

    def test_get_ion_shift_rule_by_shift_rule_index(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        adduct_rule_index, ion_shift_index = (
            fragment_ion_tree.get_shift_rule_position(4)
        )

        self.assertEqual(adduct_rule_index, 1)
        self.assertEqual(ion_shift_index, 2)

        ion_shift_rule = fragment_ion_tree.get_ion_shift_rule(4)

        self.assertEqual(str(ion_shift_rule.ion_shift), "[M+H]+")
        self.assertEqual(ion_shift_rule.atoms, ("P", "S", "N", "O"))

    def test_invalid_adduct_rule_index_raises(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_adduct_rule(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_adduct_rule(100)

    def test_invalid_node_index_raises_from_store(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_node_ion_states(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_node_shift_rule_mask(100)

    def test_empty_rule_set_builds_empty_stores(self) -> None:
        empty_rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "adduct_rules": [],
            },
            name="empty",
        )

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=empty_rule_set,
        )

        self.assertEqual(fragment_ion_tree.num_adduct_rules, 0)
        self.assertEqual(fragment_ion_tree.num_shift_rules, 0)
        self.assertEqual(fragment_ion_tree.num_ion_states, 0)

        np.testing.assert_array_equal(
            fragment_ion_tree.node_state_indptr,
            np.zeros(self.fragment_tree.num_nodes + 1, dtype=np.int64),
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
            name="empty_shift",
        )

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=rule_set,
        )

        self.assertEqual(fragment_ion_tree.num_adduct_rules, 1)
        self.assertEqual(fragment_ion_tree.num_shift_rules, 0)

        for node_index in range(fragment_ion_tree.num_nodes):
            np.testing.assert_array_equal(
                fragment_ion_tree.get_node_ion_states(node_index),
                np.asarray([[1, 0]], dtype=np.int16),
            )

            np.testing.assert_array_equal(
                fragment_ion_tree.get_node_ion_state_delta_h(node_index),
                np.asarray([-2], dtype=np.int16),
            )

        self.assertEqual(
            fragment_ion_tree.node_shift_rule_mask.shape,
            (self.fragment_tree.num_nodes, 0),
        )

    def test_multiple_ion_states_are_deduplicated_and_sorted(self) -> None:
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
                        "name": "pos_NH4_r1_u2_duplicate_state",
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
            name="multiple_states",
        )

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=rule_set,
        )

        expected_states = np.asarray(
            [
                [1, 0],
                [2, 1],
            ],
            dtype=np.int16,
        )

        expected_delta_h = np.asarray(
            [
                -2,
                -5,
            ],
            dtype=np.int16,
        )

        for node_index in range(fragment_ion_tree.num_nodes):
            np.testing.assert_array_equal(
                fragment_ion_tree.get_node_ion_states(node_index),
                expected_states,
            )

            np.testing.assert_array_equal(
                fragment_ion_tree.get_node_ion_state_delta_h(node_index),
                expected_delta_h,
            )

    def test_generic_ion_shift_is_false_for_charged_fragment(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        # node 2 is [NH4+].
        # Even generic ion shifts should be False because the fragment is charged.
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

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        # node 0: [O-], charged, all shifts disabled.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(0),
            np.zeros(fragment_ion_tree.num_shift_rules, dtype=bool),
        )

        # node 1: O, neutral, P/S/N/O rules and generic rules enabled.
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

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        # C:
        # C/P/S rules are True.
        # P/S/N/O rules are False.
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

        # O:
        # C/P/S rules are False.
        # P/S/N/O rules are True.
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

        # N:
        # Same as O for this rule set.
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

        # S:
        # S is included in both C/P/S and P/S/N/O.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(3),
            np.ones(10, dtype=bool),
        )

        # P:
        # P is included in both C/P/S and P/S/N/O.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(4),
            np.ones(10, dtype=bool),
        )

        # Cl:
        # Only generic [M+Na]+ and [M+NH4]+ rules are True.
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
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        # node 0: CCO, all adduct rules have at least one applicable shift.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(0),
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )

        # node 1: CC, all adduct rules still have at least one applicable shift.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(1),
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )

        # node 2: [NH4+], no shifts are applicable.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(2),
            np.asarray([], dtype=np.int64),
        )

        # node 3: Cl, only adduct rules with generic shifts are applicable:
        # [M+Na]+ and [M+NH4]+.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_applicable_adduct_rule_indices(3),
            np.asarray([1, 2], dtype=np.int64),
        )

    def test_has_applicable_shift_for_adduct_rule(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

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

    def test_invalid_shift_rule_index_raises(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

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

    def test_invalid_adduct_rule_index_for_applicable_shift_raises(self) -> None:
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

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
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
        )

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices(-1)

        with self.assertRaises(IndexError):
            fragment_ion_tree.get_applicable_shift_rule_indices(
                fragment_ion_tree.num_nodes
            )

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
            )

    def test_invalid_rule_set_argument_raises(self) -> None:
        with self.assertRaises(TypeError):
            FragmentIonTree.from_fragment_tree(
                self.fragment_tree,
                fragment_ion_adduct_rule_set=object(),
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
            FragmentIonTree.from_fragment_tree(
                fragment_tree,
                fragment_ion_adduct_rule_set=self.rule_set,
            )

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
            name="negative",
        )

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=negative_rule_set,
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

        # node 0: CCO
        # shift 0: C/P/S/N/O -> True
        # shift 1: P/S/N/O -> True by O
        # shift 2: C/P/S -> True by C
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(0),
            np.asarray([True, True, True], dtype=bool),
        )

        # node 1: CC
        # shift 0 True by C, shift 1 False, shift 2 True by C.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(1),
            np.asarray([True, False, True], dtype=bool),
        )

        # node 2: [NH4+], charged, all False.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(2),
            np.asarray([False, False, False], dtype=bool),
        )

        # node 3: Cl, no matching atoms.
        np.testing.assert_array_equal(
            fragment_ion_tree.get_node_shift_rule_mask(3),
            np.asarray([False, False, False], dtype=bool),
        )

if __name__ == "__main__":
    unittest.main()