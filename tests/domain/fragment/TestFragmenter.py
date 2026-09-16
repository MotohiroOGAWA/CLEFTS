from __future__ import annotations

from typing import Any, Tuple, List, Dict, Iterable
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import tempfile
import unittest

from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.domain.fragment.cleavage import CleavageActionSequence
from clefts.domain.fragment.tree.FragmentTree import FragmentTree
from clefts.domain.fragment.ion_tree.FragmentIonTree import FragmentIonTree
from clefts.domain.fragment.ion_tree.FragmentIonTreeBuilder import FragmentIonTreeBuilder
from clefts.domain.fragment.pathway.FragmentPathwayGroup import FragmentPathwayGroup
from clefts.domain.mass.tolerance import (
    parse_mass_tolerance,
    format_mass_tolerance,
)
from clefts.libs.mmkit.mmkit import Compound, Adduct, Formula

from .fragment_tree_builder_cases import (
    FragmentTreeBuilderCase,
    FragmentTreeBuilderQuestion,
    make_fragment_tree_builder_questions,
)


class TestFragmenter(unittest.TestCase):
    """Tests for Fragmenter.

    Notes
    -----
    build_fragment_pathways_by_peak is not tested here because it is not
    implemented yet.
    """

    def test_fragment_all_uses_source_action_histories_and_limits(self) -> None:
        from clefts.domain.fragment.cleavage import CleavageActionResult
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        fragmenter = self._make_fragmenter(builder)
        source = Compound.from_smiles("CCO")
        results = fragmenter.fragment_all(source, max_action_count=1)
        self.assertTrue(results)
        self.assertTrue(all(isinstance(result, CleavageActionResult) for result in results))
        self.assertTrue(all(len(result.action_sequence.actions) == 1 for result in results))
        seed = results[0].action_sequence
        seeded = fragmenter.fragment_all(source, seed_action_sequences=(seed,), max_action_count=1)
        self.assertEqual(tuple(result.action_sequence for result in seeded), (seed,))
        self.assertEqual(fragmenter.fragment_all(source, max_action_count=0), ())

    def test_fragmenter_fields_for_questions(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                self.assertIsInstance(fragmenter, Fragmenter)
                self.assertIs(fragmenter.fragment_ion_tree_builder, builder)
                self.assertIs(fragmenter.adduct_rule_set, builder.fragment_ion_adduct_rule_set)
                self.assertEqual(
                    fragmenter.adduct_types,
                    builder.fragment_ion_adduct_rule_set.adduct_types,
                )
                self.assertEqual(fragmenter.tree_max_action_count, builder.max_action_count)
                self.assertIs(
                    fragmenter.cleavage_pattern_set,
                    builder.cleavage_pattern_set,
                )
                self.assertEqual(fragmenter.name, builder.name)

    def test_build_fragment_tree_returns_fragment_tree_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    fragment_tree = fragmenter.build_fragment_tree(
                        compound,
                        max_node=-1,
                        max_edge=-1,
                        print_info=False,
                    )

                    self.assertIsInstance(fragment_tree, FragmentTree)
                    self.assertEqual(fragment_tree.smiles, compound.smiles)
                    self.assertEqual(fragment_tree.num_nodes, case.expected_num_nodes)
                    self.assertEqual(fragment_tree.num_edges, case.expected_num_edges)
                    self.assertGreaterEqual(
                        fragment_tree.num_transitions,
                        case.expected_min_transitions,
                    )

    def test_build_fragment_ion_tree_returns_fragment_ion_tree_for_all_cases(
        self,
    ) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    fragment_ion_tree = fragmenter.build_fragment_ion_tree(
                        compound,
                        max_node=-1,
                        max_edge=-1,
                        print_info=False,
                    )

                    self.assertIsInstance(fragment_ion_tree, FragmentIonTree)
                    self.assertEqual(fragment_ion_tree.smiles, compound.smiles)
                    self.assertEqual(
                        fragment_ion_tree.num_nodes,
                        case.expected_num_nodes,
                    )
                    self.assertEqual(
                        fragment_ion_tree.num_edges,
                        case.expected_num_edges,
                    )

    def test_build_fragment_tree_raises_when_max_node_is_exceeded_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    with self.assertRaisesRegex(
                        ValueError,
                        "Fragment tree node limit exceeded",
                    ):
                        fragmenter.build_fragment_tree(
                            compound,
                            max_node=1,
                            max_edge=-1,
                            print_info=False,
                        )

    def test_build_fragment_tree_raises_when_max_edge_is_exceeded_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    with self.assertRaisesRegex(
                        ValueError,
                        "Fragment tree edge limit exceeded",
                    ):
                        fragmenter.build_fragment_tree(
                            compound,
                            max_node=-1,
                            max_edge=0,
                            print_info=False,
                        )

    def test_build_fragment_ion_tree_raises_when_max_node_is_exceeded_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    with self.assertRaisesRegex(
                        ValueError,
                        "Fragment tree node limit exceeded",
                    ):
                        fragmenter.build_fragment_ion_tree(
                            compound,
                            max_node=1,
                            max_edge=-1,
                            print_info=False,
                        )

    def test_build_fragment_ion_tree_raises_when_max_edge_is_exceeded_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    with self.assertRaisesRegex(
                        ValueError,
                        "Fragment tree edge limit exceeded",
                    ):
                        fragmenter.build_fragment_ion_tree(
                            compound,
                            max_node=-1,
                            max_edge=0,
                            print_info=False,
                        )

    def test_to_dict(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                data = fragmenter.to_dict()

                self.assertIn("fragment_ion_tree_builder", data)
                self.assertIn("mass_tolerance", data)
                self.assertEqual(
                    data["mass_tolerance"],
                    format_mass_tolerance(fragmenter.mass_tolerance),
                )

    def test_from_dict(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                restored = Fragmenter.from_dict(fragmenter.to_dict())

                self.assertIsInstance(restored, Fragmenter)
                self.assertIsInstance(
                    restored.fragment_ion_tree_builder,
                    FragmentIonTreeBuilder,
                )
                self.assertEqual(
                    format_mass_tolerance(restored.mass_tolerance),
                    format_mass_tolerance(fragmenter.mass_tolerance),
                )
                self.assertEqual(restored.name, fragmenter.name)
                self.assertEqual(restored.tree_max_action_count, fragmenter.tree_max_action_count)

    def test_to_json_and_from_json(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / "fragmenter.json"

                    fragmenter.to_json(path)

                    self.assertTrue(path.exists())

                    restored = Fragmenter.from_json(path)

                    self.assertIsInstance(restored, Fragmenter)
                    self.assertEqual(
                        format_mass_tolerance(restored.mass_tolerance),
                        format_mass_tolerance(fragmenter.mass_tolerance),
                    )
                    self.assertEqual(restored.name, fragmenter.name)
                    self.assertEqual(restored.tree_max_action_count, fragmenter.tree_max_action_count)

    def test_copy(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                copied = fragmenter.copy()

                self.assertIsInstance(copied, Fragmenter)
                self.assertIsNot(copied, fragmenter)
                self.assertIsNot(
                    copied.fragment_ion_tree_builder,
                    fragmenter.fragment_ion_tree_builder,
                )
                self.assertEqual(
                    format_mass_tolerance(copied.mass_tolerance),
                    format_mass_tolerance(fragmenter.mass_tolerance),
                )
                self.assertEqual(copied.name, fragmenter.name)
                self.assertEqual(copied.tree_max_action_count, fragmenter.tree_max_action_count)

    def test_assign_fragment_pathways_to_peaks_finds_pathways_for_all_peaks(
        self,
    ) -> None:
        """Assign fragment pathways for each peak m/z list."""

        for group in self._make_fragment_pathway_assignment_test_case_groups():
            with self.subTest(group=group["name"]):
                fragmenter = Fragmenter.from_json(group["fragmenter_json"])
                compound = Compound.from_smiles(group.get("smiles", "CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC"))
                fragmenter = replace(fragmenter, precursor_candidate_max_action_count=group.get(
                    "precursor_candidate_max_action_count", fragmenter.precursor_candidate_max_action_count))

                seeds = self._water_loss_seeds(fragmenter, compound, group.get("seed_water_loss_count", 0))
                fragment_ion_tree = fragmenter.build_fragment_ion_tree(
                    compound,
                    seed_action_sequences=seeds,
                    max_node=-1,
                    max_edge=-1,
                    print_info=False,
                    _include_fragment_compound_cache=True,
                )

                for case in group["test_cases"]:
                    with self.subTest(group=group["name"], case=case["name"]):
                        assigned_result = self._assert_fragment_pathways_by_peak(
                            fragmenter=fragmenter,
                            fragment_ion_tree=fragment_ion_tree,
                            compound=compound,
                            precursor_type=case["precursor_type"],
                            peak_mz_list=case["peak_mz_list"],
                            expected_precursor_formula=case[
                                "expected_precursor_formula"
                            ],
                            expected_precursor_pathway_length=case.get(
                                "precursor_pathway_length",
                                1,
                            ),
                            expected_precursor_main_adduct_type=case.get(
                                "precursor_main_adduct_type",
                                case["precursor_type"],
                            ),
                            expected_unassigned_peaks=case.get("expected_unassigned_peaks", ()),
                            expected_precursor_pathway_count=case.get("expected_precursor_pathway_count", 1),
                        )
                        self._assert_water_loss_case(fragmenter, fragment_ion_tree, compound, case, assigned_result)

    def test_assign_fragment_pathways_to_peak_sets_finds_pathways_for_all_peaks(
        self,
    ) -> None:
        """Assign fragment pathways to multiple peak sets at once."""

        for group in self._make_fragment_pathway_assignment_test_case_groups():
            with self.subTest(group=group["name"]):
                fragmenter = Fragmenter.from_json(group["fragmenter_json"])
                compound = Compound.from_smiles(group.get("smiles", "CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC"))
                fragmenter = replace(fragmenter, precursor_candidate_max_action_count=group.get(
                    "precursor_candidate_max_action_count", fragmenter.precursor_candidate_max_action_count))

                seeds = self._water_loss_seeds(fragmenter, compound, group.get("seed_water_loss_count", 0))
                fragment_ion_tree = fragmenter.build_fragment_ion_tree(
                    compound,
                    seed_action_sequences=seeds,
                    max_node=-1,
                    max_edge=-1,
                    print_info=False,
                    _include_fragment_compound_cache=True,
                )

                peak_sets = tuple(
                    (
                        case["precursor_type"],
                        case["peak_mz_list"],
                    )
                    for case in group["test_cases"]
                )

                assigned_results = fragmenter.assign_fragment_pathways_to_peak_sets(
                    fragment_ion_tree=fragment_ion_tree,
                    peak_sets=peak_sets,
                )

                self.assertEqual(
                    len(assigned_results),
                    len(group["test_cases"]),
                )

                for case, assigned_result in zip(
                    group["test_cases"],
                    assigned_results,
                ):
                    with self.subTest(group=group["name"], case=case["name"]):
                        (
                            precursor_fragment_pathways,
                            fragment_pathways_by_peak,
                        ) = assigned_result

                        self._assert_assigned_fragment_pathways_by_peak(
                            fragmenter=fragmenter,
                            compound=compound,
                            precursor_type=case["precursor_type"],
                            peak_mz_list=case["peak_mz_list"],
                            precursor_fragment_pathways=precursor_fragment_pathways,
                            fragment_pathways_by_peak=fragment_pathways_by_peak,
                            expected_precursor_formula=case[
                                "expected_precursor_formula"
                            ],
                            expected_precursor_pathway_length=case.get(
                                "precursor_pathway_length",
                                1,
                            ),
                            expected_precursor_main_adduct_type=case.get(
                                "precursor_main_adduct_type",
                                case["precursor_type"],
                            ),
                            expected_unassigned_peaks=case.get("expected_unassigned_peaks", ()),
                            expected_precursor_pathway_count=case.get("expected_precursor_pathway_count", 1),
                        )
                        self._assert_water_loss_case(fragmenter, fragment_ion_tree, compound, case, assigned_result)

    def test_multi_action_seed_precursor_limit_counts_actions_not_edges(self) -> None:
        from dataclasses import replace
        from unittest.mock import patch
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        builder = replace(builder, max_action_count=2)
        source = Compound.from_smiles("CCCCO")
        seed_result = next(result for result in builder.cleave_all(source, max_action_count=2)
                           if len(result.action_sequence.actions) == 2)
        fragmenter = Fragmenter(builder, parse_mass_tolerance("0.01Da"),
                                precursor_candidate_max_action_count=1)
        tree = fragmenter.build_fragment_ion_tree(
            source, seed_action_sequences=(seed_result.action_sequence,),
            max_action_count=2, _include_fragment_compound_cache=True)
        seed_edge = tree.get_out_edges(0)[0]
        index = seed_edge.target_index
        self.assertEqual(tree.node_depths[index], 1)
        self.assertEqual(tree.get_min_action_counts()[index], 2)
        main_adduct = fragmenter.adduct_types[0]
        formula = main_adduct.apply_to_formula(seed_result.compound.formula).normalized
        cache = fragmenter._make_fragment_compound_cache(tree)
        with patch.object(FragmentIonTree, "get_nodes_by_depth", side_effect=AssertionError("graph depth used")):
            candidates = fragmenter._find_precursor_node_candidates(tree, main_adduct, formula, cache)
            self.assertFalse(any(node == index for node, _ in candidates))
            allowed = replace(fragmenter, precursor_candidate_max_action_count=2)
            candidates = allowed._find_precursor_node_candidates(tree, main_adduct, formula, cache)
            self.assertTrue(any(node == index for node, _ in candidates))
        self.assertEqual(len(fragmenter._build_precursor_fragment_pathways(tree, {index: [main_adduct]})), 0)
        self.assertGreater(len(allowed._build_precursor_fragment_pathways(tree, {index: [main_adduct]})), 0)
        copied = tree.copy()
        self.assertEqual(copied.get_min_action_counts(), tree.get_min_action_counts())

    def test_pathway_does_not_mix_histories_of_merged_chemical_nodes(self) -> None:
        from clefts.domain.fragment.cleavage import CleavageActionSequence
        from clefts.domain.fragment.tree.FragmentNode import FragmentNode
        from clefts.domain.fragment.tree.FragmentEdge import FragmentEdge
        from clefts.domain.fragment.tree.CleavageActionTransition import CleavageActionTransition
        from clefts.domain.fragment.pathway.build_pathway import build_pathway_items_for_node
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        source = Compound.from_smiles("CCCCC")
        result = next(result for result in builder.cleave_all(source, max_action_count=2)
                      if len(result.action_sequence.actions) == 2
                      and all(len(action.retained_atom_maps) == 4 for action in result.action_sequence.actions))
        a, b = result.action_sequence.actions
        seq_a, seq_b = CleavageActionSequence((a,)), CleavageActionSequence((b,))
        # Both primitive histories produce CCCC, but AB has a specific parent history.
        nodes = (FragmentNode(0, -1, source.smiles), FragmentNode(1, -1, "CCCC"),
                 FragmentNode(2, -1, result.compound.smiles))
        extension = FragmentEdge(1, -1, 1, 2, -1, -1, transitions=(
            CleavageActionTransition(b, result.action_sequence, seq_a),))
        adduct = Adduct.parse("[M+H]+")
        for seed, expected in ((seq_b, 0), (seq_a, 1)):
            with self.subTest(seed=seed):
                root_edge = FragmentEdge(0, -1, 0, 1, -1, -1, transitions=(
                    CleavageActionTransition(None, seed, is_seed=True),))
                tree = FragmentTree.from_nodes_and_edges(smiles=source.smiles, nodes=nodes,
                                                         edges=(root_edge, extension))
                paths = build_pathway_items_for_node(tree, 2, {0: [adduct]},
                    max_action_count=2, precursor_candidate_max_action_count=0)
                self.assertEqual(len(paths), expected)

    def test_precursor_selection_setting_roundtrip_and_copy(self) -> None:
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        fragmenter = Fragmenter(builder, parse_mass_tolerance('0.01Da'), precursor_candidate_max_action_count=1)
        data = fragmenter.to_dict()
        self.assertEqual(data['precursor_candidate_max_action_count'], 1)
        self.assertEqual(data['fragment_ion_tree_builder']['max_action_count'], builder.max_action_count)
        self.assertNotIn('min_depth_only_from', data['fragment_ion_tree_builder'])
        self.assertEqual(Fragmenter.from_dict(data).precursor_candidate_max_action_count, 1)
        self.assertEqual(fragmenter.copy().precursor_candidate_max_action_count, 1)
        with self.assertRaises(ValueError):
            Fragmenter.from_dict({**data, "precursor_candidate_max_depth": 1})
        for value in (True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Fragmenter.from_dict({**data, "precursor_candidate_max_action_count": value})
        self.assertEqual(fragmenter.tree_max_action_count, builder.max_action_count)
        self.assertTrue(builder.only_add_min_action_count)
        self.assertFalse(fragmenter._builder_for_pathway_selection().only_add_min_action_count)
        with self.assertRaises(ValueError):
            Fragmenter(builder, fragmenter.mass_tolerance, precursor_candidate_max_action_count=-1)

    def test_batched_and_individual_assignment_keep_precursor_contexts_separate(self) -> None:
        source = Compound.from_smiles('CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC')
        fragmenter = Fragmenter.from_json(
            Path(__file__).resolve().parents[3] / 'clefts/domain/fragment/presets/fragmenter_single_bond_pos.json')
        tree = fragmenter.build_fragment_ion_tree(source, _include_fragment_compound_cache=True)
        records = ((Adduct.parse('[M+H]+'), (125.0233, 154.0499)),
                   (Adduct.parse('[M+H-C6H6]+'), (125.0233, 154.0499)))
        batch = fragmenter.assign_fragment_pathways_to_peak_sets(tree, records)
        individual = tuple(fragmenter.assign_fragment_pathways_to_peaks(tree, adduct, peaks)
                           for adduct, peaks in records)
        for batched, single in zip(batch, individual):
            for batch_group, single_group in zip((batched[0], *batched[1]), (single[0], *single[1])):
                self.assertEqual({p.to_json_str() for p in batch_group},
                                 {p.to_json_str() for p in single_group})
        self.assertGreater(len(batch[0][1][0]), 0)
        self.assertEqual(len(batch[1][1][0]), 0)
        self.assertGreater(len(batch[1][1][1]), 0)
        # The missing aromatic-only history is prohibited by the new hard
        # conflict before redundancy removal, rather than lost by the cache.
        precursor_type = records[1][0]
        context = fragmenter._build_precursor_assignment_context(
            tree, precursor_type, fragmenter._make_fragment_compound_cache(tree))
        precursor_index = next(iter(context.precursor_adduct_types))
        precursor = tree.get_in_edges(precursor_index)[0].transitions[0].action_sequence
        target_index = tree.get_node_by_smiles('COc1ccoc(=O)c1').index
        target = tree.get_in_edges(target_index)[0].transitions[0].action_sequence
        self.assertTrue(precursor.actions[0].changed_bond_maps & target.actions[0].changed_bond_maps)
        from clefts.domain.fragment.cleavage import CleavageActionSequence
        with self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
            CleavageActionSequence((*precursor.actions, *target.actions))

    def _make_fragment_pathway_assignment_test_case_groups(self) -> list[dict[str, Any]]:
        """Create common test cases for fragment pathway assignment tests."""

        return [
            {
                "name": "positive",
                "fragmenter_json": Path(__file__).resolve().parents[3] / "clefts/domain/fragment/presets/fragmenter_single_bond_pos.json",
                "test_cases": [
                    {
                        "name": "protonated",
                        "precursor_type": Adduct.parse("[M+H]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            288.1158,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H18NO4+"),
                    },
                    {
                        "name": "sodiated",
                        "precursor_type": Adduct.parse("[M+Na]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            310.0977,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H17NNaO4+"),
                    },
                    {
                        "name": "ammonium_adduct",
                        "precursor_type": Adduct.parse("[M+H4N]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            305.1503,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H21N2O4+"),
                    },
                    {
                        "name": "protonated_dimer",
                        "precursor_type": Adduct.parse("[2M+H]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            575.2392,
                        ],
                        "expected_precursor_formula": Formula.parse("C32H35N2O8+"),
                    },
                    {
                        "name": "benzene_loss",
                        # Source-anchored changed-bond conflicts prohibit the
                        # aromatic-only product after this precursor history.
                        "expected_unassigned_peaks": (125.0233,),
                        "precursor_type": Adduct.parse("[M+H-C6H6]+"),
                        "peak_mz_list": [
                            125.0233,
                            154.0499,
                            155.0577,
                            210.0761,
                        ],
                        "expected_precursor_formula": Formula.parse("C10H12NO4+"),
                        "precursor_pathway_length": 2,
                        "precursor_main_adduct_type": Adduct.parse("[M-H]+"),
                    },
                ],
            },
            {
                "name": "negative",
                "fragmenter_json": Path(__file__).resolve().parents[3] / "clefts/domain/fragment/presets/fragmenter_single_bond_neg.json",
                "test_cases": [
                    {
                        "name": "deprotonated",
                        "precursor_type": Adduct.parse("[M-H]-"),
                        "peak_mz_list": [
                            89.0386,
                            123.0077,
                            152.0342,
                            153.0421,
                            227.0703,
                            244.0968,
                            286.1002,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H16NO4-"),
                    },
                    {
                        "name": "deprotonated_dimer",
                        "precursor_type": Adduct.parse("[2M-H]-"),
                        "peak_mz_list": [
                            89.0386,
                            123.0077,
                            152.0342,
                            153.0421,
                            227.0703,
                            244.0968,
                            573.2235,
                        ],
                        "expected_precursor_formula": Formula.parse("C32H33N2O8-"),
                    },
                ],
            },
        ] + self._make_water_loss_assignment_groups()

    def _make_water_loss_assignment_groups(self) -> list[dict[str, Any]]:
        preset_dir = Path(__file__).resolve().parents[3] / "clefts/domain/fragment/presets"
        groups = []
        scenarios = (
            ("primary_alcohol", "CCCO", "positive", 1, 1, "C3H7+", ("CCC",), 1,
             ("C3H7+", "C2H5+", "CH3+"), ("C2H5O+",)),
            ("secondary_alcohol", "CC(O)C", "positive", 1, 1, "C3H7+", ("CCC",), 1,
             ("C3H7+", "C2H5+", "CH3+"), ("C2H5O+",)),
            ("equivalent_hydroxyl_sites", "OCCO", "positive", 1, 1, "C2H5O+", ("CCO",), 2,
             ("C2H5O+", "CH3+", "C2H3+"), ()),
            ("different_hydroxyl_sites", "OCC(O)CO", "positive", 1, 1, "C3H7O2+",
             ("CC(O)CO", "OCCCO"), 3, ("C3H7O2+", "C2H5O+", "CH3+"), ()),
            ("double_water_loss", "OCCO", "positive", 2, 2, "C2H3+", ("CC",), 2,
             ("C2H3+",), ("CH3+", "C2H5O+")),
            ("seeded_double_water_loss", "OCCO", "positive", 2, 2, "C2H3+", ("CC",), 1,
             ("C2H3+",), ("C2H5O+",)),
            ("seeded_double_loss_over_budget", "OCCO", "positive", 2, 1, "C2H3+", (), 0,
             (), ("C2H3+", "C2H5O+")),
            ("double_loss_over_budget", "OCCO", "positive", 2, 1, "C2H3+", (), 0,
             (), ("C2H3+", "C2H5O+")),
            ("zero_precursor_action_budget", "CCCO", "positive", 1, 0, "C3H7+", (), 0,
             (), ("C3H7+", "C2H5+")),
            ("ether_without_hydroxyl", "CCOCC", "positive", 1, 1, "C4H9+", (), 0,
             (), ("C4H9+", "C2H5+")),
            ("negative_water_loss", "CCCO", "negative", 1, 1, "C3H5-", ("CCC",), 1,
             ("C3H5-", "C2H3-", "CH-"), ("C2H3O-",)),
        )
        for name, smiles, polarity, loss_count, limit, formula, structures, count, assigned, unassigned in scenarios:
            suffix = "+" if polarity == "positive" else "-"
            base = "[M+H]+" if polarity == "positive" else "[M-H]-"
            loss = "H2O" if loss_count == 1 else "2H2O"
            precursor = Adduct.parse(base[:-2] + "-" + loss + "]" + suffix)
            main = Adduct.parse("[M" + ("-H" if loss_count == 1 and polarity == "positive" else "-3H") + "]" + suffix)
            peaks = [Formula.parse(text).exact_mass for text in (*assigned, *unassigned)]
            case = dict(name=name, precursor_type=precursor, peak_mz_list=peaks,
                        expected_precursor_formula=Formula.parse(formula),
                        precursor_pathway_length=2 if name.startswith("seeded_") else loss_count + 1,
                        precursor_main_adduct_type=main,
                        expected_precursor_pathway_count=count,
                        expected_unassigned_peaks=tuple(Formula.parse(text).exact_mass for text in unassigned),
                        expected_precursor_smiles=structures, water_loss_count=loss_count)
            cases = [case]
            # The same tree must retain oxygen-containing assignments for the
            # normal precursor while excluding them after its OH has been lost.
            if name == "primary_alcohol":
                cases.append(dict(name="normal_precursor_same_tree", precursor_type=Adduct.parse(base),
                    peak_mz_list=[Formula.parse("C3H9O+").exact_mass, Formula.parse("C2H5O+").exact_mass],
                    expected_precursor_formula=Formula.parse("C3H9O+")))
            groups.append(dict(name=name, smiles=smiles,
                fragmenter_json=preset_dir / ("fragmenter_single_bond_pos.json" if polarity == "positive" else "fragmenter_single_bond_neg.json"),
                precursor_candidate_max_action_count=limit, seed_water_loss_count=loss_count if name.startswith("seeded_") else 0, test_cases=cases))
        return groups

    def _water_loss_seeds(self, fragmenter: Fragmenter, source: Compound, count: int) -> Tuple[CleavageActionSequence, ...]:
        if not count:
            return ()
        hydroxyl_maps = sorted(atom.GetAtomMapNum() for atom in source.mapped_mol.GetAtoms()
                               if atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1)
        actions = fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source)
        selected = tuple(next(action for action in actions if action.discarded_atom_maps == frozenset((atom_map,)))
                         for atom_map in hydroxyl_maps[:count])
        self.assertEqual(len(selected), count)
        sequence = CleavageActionSequence(selected)
        self.assertEqual(len(sequence.actions), count)
        return (sequence,)

    def _assert_water_loss_case(
        self, fragmenter: Fragmenter, tree: FragmentIonTree, source: Compound,
        case: dict[str, Any], assigned_result: Tuple[FragmentPathwayGroup, Tuple[FragmentPathwayGroup, ...]],
    ) -> None:
        precursors, peaks = assigned_result
        # All cases must refer to this record's precursor, including batched
        # records which share a main adduct but have different neutral losses.
        for group in peaks:
            for path in group:
                self.assertEqual(path.root_node.smiles, source.smiles)
                self.assertTrue(path.has_precursor_node)
                node = path.precursor_node
                self.assertEqual(node.precursor_adduct_type.apply_to_formula(
                    Compound.from_smiles(node.smiles).formula).normalized,
                    case["expected_precursor_formula"])
        if "water_loss_count" not in case:
            return
        expected_smiles = set(case["expected_precursor_smiles"])
        self.assertEqual({path.terminal_node.smiles for path in precursors}, expected_smiles)
        # Independently check the ion formula against literal H2O subtraction.
        base = Adduct.parse("[M+H]+" if case["precursor_type"].charge > 0 else "[M-H]-")
        expected_formula = (base.apply_to_formula(source.formula)
                            - Formula.parse("H2O") * case["water_loss_count"]).normalized
        self.assertEqual(case["expected_precursor_formula"], expected_formula)
        for path in precursors:
            self.assertEqual(path.adduct, case["precursor_main_adduct_type"])
            self.assertEqual(path.formula, expected_formula)
            self.assertEqual(path.terminal_node.smiles, path.precursor_node.smiles)
        for group in peaks:
            for path in group:
                precursor_node = path.precursor_node
                self.assertIn(precursor_node.smiles, expected_smiles)
                self.assertEqual(precursor_node.precursor_adduct_type.apply_to_formula(
                    Compound.from_smiles(precursor_node.smiles).formula).normalized, expected_formula)
        individual = fragmenter.assign_fragment_pathways_to_peaks(tree, case["precursor_type"], case["peak_mz_list"])
        for batch_group, single_group in zip((precursors, *peaks), (individual[0], *individual[1])):
            self.assertEqual({path.to_json_str() for path in batch_group},
                             {path.to_json_str() for path in single_group})
        context = fragmenter._build_precursor_assignment_context(
            tree, case["precursor_type"], fragmenter._make_fragment_compound_cache(tree))
        self.assertEqual({tree.get_node(index).smiles for index in context.precursor_adduct_types}, expected_smiles)
        hydroxyl_maps = {atom.GetAtomMapNum() for atom in source.mapped_mol.GetAtoms()
                         if atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1}
        source_maps = frozenset(atom.GetAtomMapNum() for atom in source.mapped_mol.GetAtoms())
        discarded_sites = set()
        for index in context.precursor_adduct_types:
            sequences = {transition.action_sequence for edge in tree.get_in_edges(index)
                         for transition in edge.transitions
                         if len(transition.action_sequence.actions) == case["water_loss_count"]}
            self.assertTrue(sequences)
            for sequence in sequences:
                discarded = source_maps - sequence.retained_atom_maps
                self.assertEqual(len(discarded), case["water_loss_count"])
                self.assertTrue(discarded <= hydroxyl_maps)
                expected_cuts = frozenset(tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
                    for bond in source.mapped_mol.GetBonds()
                    if bond.GetBeginAtom().GetAtomMapNum() in discarded or bond.GetEndAtom().GetAtomMapNum() in discarded)
                self.assertEqual(sequence.cut_bond_maps, expected_cuts)
                self.assertEqual(sequence.bond_updates, frozenset())
                # Every history is executed against the same original Source.
                products = sequence.compile(source).run(source)
                self.assertEqual(len(products), 1)
                self.assertEqual(frozenset(atom.GetAtomMapNum() for atom in products[0].GetAtoms()), sequence.retained_atom_maps)
                product = Chem.Mol(products[0])
                for atom in product.GetAtoms():
                    atom.SetAtomMapNum(0)
                self.assertEqual(Chem.MolToSmiles(product), tree.get_node(index).smiles)
                discarded_sites.add(discarded)
        expected_site_count = (1 if case["water_loss_count"] == 2 else case["expected_precursor_pathway_count"]) if expected_smiles else 0
        self.assertEqual(len(discarded_sites), expected_site_count)

    def _make_fragmenter(
        self,
        builder: FragmentIonTreeBuilder,
    ) -> Fragmenter:
        return Fragmenter(
            fragment_ion_tree_builder=builder,
            mass_tolerance=parse_mass_tolerance("0.01Da"),
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
            max_action_count=question.builder.max_action_count,
            cleavage_pattern_set=question.builder.cleavage_pattern_set.copy(),
            only_add_min_action_count=question.builder.only_add_min_action_count,
            fragment_ion_adduct_rule_set=(
                question.fragment_ion_adduct_rule_set.copy()
            ),
        )

    def _assert_fragment_pathways_by_peak(
        self,
        *,
        fragmenter: Fragmenter,
        fragment_ion_tree: FragmentIonTree,
        compound: Compound,
        precursor_type: Adduct,
        peak_mz_list: list[float],
        expected_precursor_formula: Formula,
        expected_precursor_pathway_count: int = 1,
        expected_precursor_pathway_length: int = 1,
        expected_precursor_main_adduct_type: Adduct | None = None,
        expected_unassigned_peaks: Tuple[float, ...] = (),
    ) -> Tuple[FragmentPathwayGroup, Tuple[FragmentPathwayGroup, ...]]:
        """Assert that fragment pathways are found for all given peaks."""
        precursor_fragment_pathways, fragment_pathways_by_peak = (
            fragmenter.assign_fragment_pathways_to_peaks(
                fragment_ion_tree=fragment_ion_tree,
                precursor_type=precursor_type,
                peaks_mz=peak_mz_list,
            )
        )

        self._assert_precursor_fragment_pathways(
            precursor_fragment_pathways=precursor_fragment_pathways,
            compound=compound,
            precursor_type=precursor_type,
            expected_precursor_formula=expected_precursor_formula,
            expected_precursor_pathway_count=expected_precursor_pathway_count,
            expected_precursor_pathway_length=expected_precursor_pathway_length,
            expected_precursor_main_adduct_type=expected_precursor_main_adduct_type,
        )

        self._assert_fragment_pathways_for_all_peaks(
            fragmenter=fragmenter,
            fragment_pathways_by_peak=fragment_pathways_by_peak,
            peak_mz_list=peak_mz_list,
            expected_unassigned_peaks=expected_unassigned_peaks,
        )
        return precursor_fragment_pathways, fragment_pathways_by_peak

    def _assert_precursor_fragment_pathways(
        self,
        *,
        precursor_fragment_pathways: FragmentPathwayGroup,
        compound: Compound,
        precursor_type: Adduct,
        expected_precursor_formula: Formula,
        expected_precursor_pathway_count: int = 1,
        expected_precursor_pathway_length: int = 1,
        expected_precursor_main_adduct_type: Adduct | None = None,
    ) -> None:
        """Assert precursor fragment pathways."""

        self.assertEqual(len(precursor_fragment_pathways), expected_precursor_pathway_count)
        if expected_precursor_pathway_count == 0:
            return

        expected_adduct = expected_precursor_main_adduct_type or precursor_type
        for precursor_pathway in precursor_fragment_pathways:
            self.assertEqual(len(precursor_pathway), expected_precursor_pathway_length)
            self.assertEqual(precursor_pathway.get_node(0).smiles, compound.smiles)
            self.assertEqual(precursor_pathway.adduct, expected_adduct)
            self.assertEqual(precursor_pathway.formula, expected_precursor_formula)

    def _assert_fragment_pathways_for_all_peaks(
        self,
        *,
        fragmenter: Fragmenter,
        fragment_pathways_by_peak: list[FragmentPathwayGroup],
        peak_mz_list: list[float],
        expected_unassigned_peaks: Tuple[float, ...] = (),
    ) -> None:
        """Assert that each peak has valid fragment pathways."""

        self.assertEqual(len(fragment_pathways_by_peak), len(peak_mz_list))

        for peak_index, (peak_mz, fragment_pathways) in enumerate(
            zip(peak_mz_list, fragment_pathways_by_peak)
        ):
            with self.subTest(peak_index=peak_index, peak_mz=peak_mz):
                if peak_mz in expected_unassigned_peaks:
                    self.assertEqual(len(fragment_pathways), 0)
                    continue
                self.assertGreaterEqual(
                    len(fragment_pathways),
                    1,
                    msg=f"No fragment pathways found for peak m/z {peak_mz}",
                )

                pathway_lengths = [
                    len(pathway)
                    for pathway in fragment_pathways
                ]

                self.assertTrue(
                    all(length <= fragmenter.tree_max_action_count + 1 for length in pathway_lengths),
                    msg=(
                        f"Some fragment pathways exceed max depth "
                        f"{fragmenter.tree_max_action_count}: {pathway_lengths}"
                    ),
                )

                self.assertEqual(
                    len(set(pathway_lengths)),
                    1,
                    msg=(
                        "Fragment pathway lengths are not all equal: "
                        f"{pathway_lengths}"
                    ),
                )

                for pathway_index, fragment_pathway in enumerate(fragment_pathways):
                    theoretical_mz = fragment_pathway.formula.exact_mass

                    with self.subTest(
                        peak_index=peak_index,
                        pathway_index=pathway_index,
                        peak_mz=peak_mz,
                        theoretical_mz=theoretical_mz,
                    ):
                        self.assertTrue(fragment_pathway.has_precursor_node)

                        self.assertTrue(
                            fragmenter.mass_tolerance.within(
                                observed=peak_mz,
                                theoretical=theoretical_mz,
                            ),
                            msg=(
                                "Fragment pathway mass is outside tolerance: "
                                f"peak_index={peak_index}, "
                                f"pathway_index={pathway_index}, "
                                f"peak_mz={peak_mz}, "
                                f"theoretical_mz={theoretical_mz}, "
                                f"mass_error={fragmenter.mass_tolerance.error(peak_mz, theoretical_mz)}, "
                                f"tolerance={fragmenter.mass_tolerance}"
                            ),
                        )

    def _assert_assigned_fragment_pathways_by_peak(
        self,
        *,
        fragmenter: Fragmenter,
        compound: Compound,
        precursor_type: Adduct,
        peak_mz_list: Iterable[float],
        precursor_fragment_pathways: FragmentPathwayGroup,
        fragment_pathways_by_peak: Tuple[FragmentPathwayGroup, ...],
        expected_precursor_formula: Formula,
        expected_precursor_pathway_count: int = 1,
        expected_precursor_pathway_length: int = 1,
        expected_precursor_main_adduct_type: Adduct | None = None,
        expected_unassigned_peaks: Tuple[float, ...] = (),
    ) -> None:
        peak_mz_list = tuple(peak_mz_list)

        if expected_precursor_main_adduct_type is None:
            expected_precursor_main_adduct_type = precursor_type

        self.assertEqual(
            len(fragment_pathways_by_peak),
            len(peak_mz_list),
        )

        self.assertEqual(len(precursor_fragment_pathways.pathways), expected_precursor_pathway_count)

        for precursor_pathway in precursor_fragment_pathways.pathways:
            self.assertEqual(
                precursor_pathway.formula,
                expected_precursor_formula,
            )
            self.assertEqual(
                precursor_pathway.adduct,
                expected_precursor_main_adduct_type,
            )
            self.assertEqual(
                len(precursor_pathway),
                expected_precursor_pathway_length,
            )

        for peak_mz, fragment_pathway_group in zip(
            peak_mz_list,
            fragment_pathways_by_peak,
        ):
            with self.subTest(peak_mz=peak_mz):
                if peak_mz in expected_unassigned_peaks:
                    self.assertEqual(len(fragment_pathway_group.pathways), 0)
                    continue
                self.assertGreater(
                    len(fragment_pathway_group.pathways),
                    0,
                )

                for fragment_pathway in fragment_pathway_group.pathways:
                    mz_error = fragmenter.mass_tolerance.error(
                        observed=peak_mz,
                        theoretical=fragment_pathway.formula.exact_mass,
                    )

                    self.assertTrue(
                        fragmenter.mass_tolerance.within(
                            observed=peak_mz,
                            theoretical=fragment_pathway.formula.exact_mass,
                        ),
                        msg=(
                            f"Peak m/z {peak_mz} was assigned to "
                            f"{fragment_pathway.formula} "
                            f"with exact mass "
                            f"{fragment_pathway.formula.exact_mass}. "
                            f"m/z error: {mz_error} "
                            f"{fragmenter.mass_tolerance.unit}"
                        ),
                    )

    def test_resolve_precursor_actions_reuses_same_tree_without_rdkit(self) -> None:
        """Precursor resolution walks the already-built tree; it never reacts again."""
        fragmenter = Fragmenter.from_json("clefts/domain/fragment/presets/fragmenter_single_bond_pos.json")
        source = Compound.from_smiles("CCCO")
        tree = fragmenter.build_fragment_ion_tree(source, _include_fragment_compound_cache=True)

        root_only = fragmenter.resolve_precursor_actions(tree, Adduct.parse("[M+H]+"))
        self.assertEqual({(pa.node_index, pa.action_sequence) for pa in root_only}, {(0, None)})

        with patch.object(rdChemReactions.ChemicalReaction, "RunReactants",
                          side_effect=AssertionError("RDKit re-invoked")):
            dehydrated = fragmenter.resolve_precursor_actions(tree, Adduct.parse("[M+H-H2O]+"))

        self.assertEqual(len(dehydrated), 1)
        precursor_action = next(iter(dehydrated))
        self.assertIsNotNone(precursor_action.action_sequence)
        self.assertNotEqual(precursor_action.node_index, 0)
        product = precursor_action.action_sequence.compile(source).run(source)[0]
        self.assertEqual(Compound(product).smiles, "CCC")


if __name__ == "__main__":
    unittest.main()