from __future__ import annotations

import unittest

from clefts.domain.fragment.pathway.CleavageStep import CleavageStep
from clefts.domain.fragment.pathway.FragmentPathwayEdge import (
    FragmentPathwayEdge,
)
from clefts.domain.fragment.pathway.FragmentPathwayGroup import (
    FragmentPathwayGroup,
)
from clefts.domain.fragment.pathway.FragmentPathwayNode import (
    FragmentPathwayNode,
)
from clefts.domain.fragment.pathway.FragmentPathway import (
    FragmentPathway,
)
from clefts.libs.mmkit.mmkit import Adduct


class TestFragmentPathwayGroup(unittest.TestCase):
    def setUp(self) -> None:
        self.precursor_adduct = Adduct.parse("[M+H]+")
        self.fragment_adduct = Adduct.parse("[M+H]+")

        self.precursor_node = FragmentPathwayNode(
            smiles="CCO",
            precursor_adduct_type=self.precursor_adduct,
        )
        self.terminal_node_1 = FragmentPathwayNode(
            smiles="CC",
        )
        self.terminal_node_2 = FragmentPathwayNode(
            smiles="CO",
        )

        self.step_1 = CleavageStep(
            cleavage_pattern_id=1,
            reaction_id=10,
            product_molecule_id=100,
            reactant_indices=(0, 1, 2),
            product_indices=(0, 1),
        )
        self.step_2 = CleavageStep(
            cleavage_pattern_id=2,
            reaction_id=20,
            product_molecule_id=200,
            reactant_indices=(0, 1, 2),
            product_indices=(0, 2),
        )

        self.edge_1 = FragmentPathwayEdge(
            steps=(self.step_1,),
        )
        self.edge_2 = FragmentPathwayEdge(
            steps=(self.step_2,),
        )

        self.pathway_1 = FragmentPathway(
            elements=(
                self.precursor_node,
                self.edge_1,
                self.terminal_node_1,
            ),
            adduct=self.fragment_adduct,
        )
        self.pathway_2 = FragmentPathway(
            elements=(
                self.precursor_node,
                self.edge_2,
                self.terminal_node_2,
            ),
            adduct=self.fragment_adduct,
        )

    def test_empty_creates_empty_group(self) -> None:
        group = FragmentPathwayGroup.empty()

        self.assertEqual(len(group), 0)
        self.assertTrue(group.is_empty)
        self.assertEqual(group.pathways, ())

    def test_init_accepts_tuple_of_pathways(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1, self.pathway_2),
        )

        self.assertEqual(len(group), 2)
        self.assertFalse(group.is_empty)
        self.assertIs(group[0], self.pathway_1)
        self.assertIs(group[1], self.pathway_2)

    def test_init_rejects_non_tuple_pathways(self) -> None:
        with self.assertRaises(TypeError):
            FragmentPathwayGroup(
                pathways=[self.pathway_1],  # type: ignore[arg-type]
            )

    def test_init_rejects_non_pathway_element(self) -> None:
        with self.assertRaises(TypeError):
            FragmentPathwayGroup(
                pathways=(self.pathway_1, "invalid"),  # type: ignore[arg-type]
            )

    def test_from_list_of_pathways(self) -> None:
        group = FragmentPathwayGroup.from_list_of_pathways(
            [self.pathway_1, self.pathway_2]
        )

        self.assertEqual(group.pathways, (self.pathway_1, self.pathway_2))

    def test_iter(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1, self.pathway_2),
        )

        self.assertEqual(
            list(group),
            [self.pathway_1, self.pathway_2],
        )

    def test_terminal_nodes(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1, self.pathway_2),
        )

        self.assertEqual(
            group.terminal_nodes,
            (self.terminal_node_1, self.terminal_node_2),
        )

    def test_precursor_nodes(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1, self.pathway_2),
        )

        self.assertEqual(
            group.precursor_nodes,
            (self.precursor_node, self.precursor_node),
        )

    def test_to_list(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1,),
        )

        expected = [
            self.pathway_1.to_list(),
        ]

        self.assertEqual(group.to_list(), expected)

    def test_from_list(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1, self.pathway_2),
        )

        restored = FragmentPathwayGroup.from_list(group.to_list())

        self.assertEqual(restored, group)

    def test_to_json_str_and_parse(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1, self.pathway_2),
        )

        text = group.to_json_str()
        restored = FragmentPathwayGroup.parse(text)

        self.assertEqual(restored, group)

    def test_from_json_str(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1,),
        )

        restored = FragmentPathwayGroup.from_json_str(
            group.to_json_str()
        )

        self.assertEqual(restored, group)

    def test_str_returns_json_string(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1,),
        )

        self.assertEqual(str(group), group.to_json_str())

    def test_parse_rejects_non_list_json(self) -> None:
        with self.assertRaises(ValueError):
            FragmentPathwayGroup.parse('{"pathways": []}')

    def test_from_list_rejects_non_list(self) -> None:
        with self.assertRaises(TypeError):
            FragmentPathwayGroup.from_list(  # type: ignore[arg-type]
                {"pathways": []}
            )

    def test_formulas(self) -> None:
        group = FragmentPathwayGroup(
            pathways=(self.pathway_1, self.pathway_2),
        )

        formulas = group.formulas

        self.assertEqual(len(formulas), 2)
        self.assertEqual(
            formulas[0],
            self.pathway_1.formula,
        )
        self.assertEqual(
            formulas[1],
            self.pathway_2.formula,
        )


if __name__ == "__main__":
    unittest.main()