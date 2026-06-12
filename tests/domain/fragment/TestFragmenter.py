from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from clefts.domain.fragment.fragmenter import Fragmenter
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

    def test_build_fragment_pathways_by_peak_finds_pathways_for_all_peaks(self) -> None:
        """Build fragment pathways for multiple adduct types and peak m/z lists."""

        compound = Compound.from_smiles(
            "CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC"
        )

        test_case_groups = [
            {
                "name": "positive",
                "fragmenter_json": "clefts/domain/fragment/presets/fragmenter_pos.json",
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
                        "precursor_type": Adduct.parse("[M-C6H5]+"),
                        "peak_mz_list": [
                            125.0233,
                            154.0499,
                            155.0577,
                            210.0761,
                        ],
                        "expected_precursor_formula": Formula.parse("C10H12NO4+"),
                        "precursor_pathway_length": 2,
                        "precusor_main_adduct_type": Adduct.parse("[M+H]+"),
                    },
                ],
            },
            {
                "name": "negative",
                "fragmenter_json": "clefts/domain/fragment/presets/fragmenter_neg.json",
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
        ]

        for group in test_case_groups:
            with self.subTest(group=group["name"]):
                fragmenter = Fragmenter.from_json(group["fragmenter_json"])

                fragment_ion_tree = fragmenter.build_fragment_ion_tree(
                    compound,
                    max_node=-1,
                    max_edge=-1,
                    print_info=False,
                    _include_fragment_compound_cache=True,
                )

                for case in group["test_cases"]:
                    with self.subTest(group=group["name"], case=case["name"]):
                        self._assert_fragment_pathways_by_peak(
                            fragmenter=fragmenter,
                            fragment_ion_tree=fragment_ion_tree,
                            compound=compound,
                            precursor_type=case["precursor_type"],
                            peak_mz_list=case["peak_mz_list"],
                            expected_precursor_formula=case["expected_precursor_formula"],
                            expected_precursor_pathway_length=case.get(
                                "precursor_pathway_length",
                                1,
                            ),
                            expected_precursor_main_adduct_type=case.get(
                                "precusor_main_adduct_type",
                                case["precursor_type"],
                            ),
                        )

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

    def _assert_fragment_pathways_by_peak(
        self,
        *,
        fragmenter: Fragmenter,
        fragment_ion_tree,
        compound: Compound,
        precursor_type: Adduct,
        peak_mz_list: list[float],
        expected_precursor_formula: Formula,
        expected_precursor_pathway_length: int = 1,
        expected_precursor_main_adduct_type: Adduct | None = None,
    ) -> None:
        """Assert that fragment pathways are found for all given peaks."""
        pass
        precursor_fragment_pathways, fragment_pathways_by_peak = (
            fragmenter.build_fragment_pathways_by_peak(
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
            expected_precursor_pathway_length=expected_precursor_pathway_length,
            expected_precursor_main_adduct_type=expected_precursor_main_adduct_type,
        )

        self._assert_fragment_pathways_for_all_peaks(
            fragmenter=fragmenter,
            fragment_pathways_by_peak=fragment_pathways_by_peak,
            peak_mz_list=peak_mz_list,
        )

    def _assert_precursor_fragment_pathways(
        self,
        *,
        precursor_fragment_pathways: FragmentPathwayGroup,
        compound: Compound,
        precursor_type: Adduct,
        expected_precursor_formula: Formula,
        expected_precursor_pathway_length: int = 1,
        expected_precursor_main_adduct_type: Adduct | None = None,
    ) -> None:
        """Assert precursor fragment pathways."""

        self.assertEqual(len(precursor_fragment_pathways), 1)

        precursor_pathway = precursor_fragment_pathways[0]

        self.assertEqual(len(precursor_pathway), expected_precursor_pathway_length)
        self.assertEqual(precursor_pathway.get_node(0).smiles, compound.smiles)
        if expected_precursor_main_adduct_type is None:
            self.assertEqual(precursor_pathway.adduct, precursor_type)
        self.assertEqual(precursor_pathway.formula, expected_precursor_formula)

    def _assert_fragment_pathways_for_all_peaks(
        self,
        *,
        fragmenter: Fragmenter,
        fragment_pathways_by_peak: list[FragmentPathwayGroup],
        peak_mz_list: list[float],
    ) -> None:
        """Assert that each peak has valid fragment pathways."""

        self.assertEqual(len(fragment_pathways_by_peak), len(peak_mz_list))

        for peak_index, (peak_mz, fragment_pathways) in enumerate(
            zip(peak_mz_list, fragment_pathways_by_peak)
        ):
            with self.subTest(peak_index=peak_index, peak_mz=peak_mz):
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
                    all(length <= fragmenter.tree_max_depth + 1 for length in pathway_lengths),
                    msg=(
                        f"Some fragment pathways exceed max depth "
                        f"{fragmenter.tree_max_depth}: {pathway_lengths}"
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

if __name__ == "__main__":
    unittest.main()