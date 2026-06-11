from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.domain.fragment.tree.FragmentTree import FragmentTree
from clefts.domain.fragment.ion_tree.FragmentIonTree import (
    FragmentIonTree,
)
from clefts.domain.fragment.ion_tree.FragmentIonTreeBuilder import (
    FragmentIonTreeBuilder,
)
from clefts.domain.mass.tolerance import (
    parse_mass_tolerance,
    format_mass_tolerance,
)
from clefts.libs.mmkit.mmkit import Compound

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
                self.assertEqual(fragmenter.tree_max_depth, builder.max_depth)
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
                        fragment_tree.num_events,
                        case.expected_min_events,
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

    def test_build_fragment_tree_respects_max_node_for_all_cases(self) -> None:
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
                        max_node=1,
                        max_edge=-1,
                        print_info=False,
                    )

                    self.assertEqual(fragment_tree.num_nodes, 1)
                    self.assertEqual(fragment_tree.num_edges, 0)
                    self.assertEqual(fragment_tree.num_events, 0)
                    self.assertEqual(fragment_tree.get_node(0).smiles, compound.smiles)

    def test_build_fragment_tree_respects_max_edge_for_all_cases(self) -> None:
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
                        max_edge=0,
                        print_info=False,
                    )

                    self.assertEqual(fragment_tree.num_nodes, 1)
                    self.assertEqual(fragment_tree.num_edges, 0)
                    self.assertEqual(fragment_tree.num_events, 0)
                    self.assertEqual(fragment_tree.get_node(0).smiles, compound.smiles)

    def test_build_fragment_ion_tree_respects_max_node_for_all_cases(self) -> None:
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
                        max_node=1,
                        max_edge=-1,
                        print_info=False,
                    )

                    self.assertEqual(fragment_ion_tree.num_nodes, 1)
                    self.assertEqual(fragment_ion_tree.num_edges, 0)
                    self.assertEqual(fragment_ion_tree.get_node(0).smiles, compound.smiles)

    def test_build_fragment_ion_tree_respects_max_edge_for_all_cases(self) -> None:
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
                        max_edge=0,
                        print_info=False,
                    )

                    self.assertEqual(fragment_ion_tree.num_nodes, 1)
                    self.assertEqual(fragment_ion_tree.num_edges, 0)
                    self.assertEqual(fragment_ion_tree.get_node(0).smiles, compound.smiles)

    # def test_build_fragment_pathways_by_peak_can_be_called(self) -> None:
    #     for question in make_fragment_tree_builder_questions():
    #         if question.fragment_ion_adduct_rule_set is None:
    #             continue

    #         builder = self._make_fragment_ion_tree_builder(question)
    #         fragmenter = self._make_fragmenter(builder)

    #         for case in question.cases:
    #             with self.subTest(question=question.name, case=case.name):
    #                 compound = Compound.from_smiles(case.smiles)

    #                 fragment_ion_tree = fragmenter.build_fragment_ion_tree(
    #                     compound,
    #                     max_node=-1,
    #                     max_edge=-1,
    #                     print_info=False,
    #                     _include_fragment_compound_cache=True,
    #                 )

    #                 adduct_type = fragmenter.adduct_types[0]
    #                 peak_mz_list = [
    #                     100.0,
    #                     150.0,
    #                     200.0,
    #                 ]

    #                 try:
    #                     fragmenter.build_fragment_pathways_by_peak(
    #                         fragment_ion_tree=fragment_ion_tree,
    #                         precursor_type=adduct_type,
    #                         peak_mz_list=peak_mz_list,
    #                     )
    #                 except ValueError:
    #                     pass
    def test_build_fragment_pathways_by_peak_can_be_called(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles("CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC")

                    fragment_ion_tree = fragmenter.build_fragment_ion_tree(
                        compound,
                        max_node=-1,
                        max_edge=-1,
                        print_info=False,
                        _include_fragment_compound_cache=True,
                    )
                    
                    
                    from clefts.libs.mmkit.mmkit import Adduct
                    adduct_type = Adduct.parse("[M+H]+")
                    peak_mz_list = [
                        91.0542,
                        125.0233,
                        154.0499,
                        155.0577,
                        185.0961,
                        200.107,
                        229.0859,
                        246.1125,
                    ]

                    try:
                        fragmenter.build_fragment_pathways_by_peak(
                            fragment_ion_tree=fragment_ion_tree,
                            precursor_type=adduct_type,
                            peak_mz_list=peak_mz_list,
                        )
                    except ValueError:
                        pass

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
                self.assertEqual(restored.tree_max_depth, fragmenter.tree_max_depth)

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
                    self.assertEqual(restored.tree_max_depth, fragmenter.tree_max_depth)

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
                self.assertEqual(copied.tree_max_depth, fragmenter.tree_max_depth)

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
            max_depth=question.builder.max_depth,
            cleavage_pattern_set=question.builder.cleavage_pattern_set.copy(),
            only_add_min_depth=question.builder.only_add_min_depth,
            min_depth_only_from=question.builder.min_depth_only_from,
            fragment_ion_adduct_rule_set=(
                question.fragment_ion_adduct_rule_set.copy()
            ),
        )


if __name__ == "__main__":
    unittest.main()