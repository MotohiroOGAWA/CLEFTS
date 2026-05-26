import unittest

from clefts.domain.fragment.cleavage.CleavagePattern import CleavagePattern
from clefts.domain.fragment.cleavage.CleavagePatternSet import CleavagePatternSet
from clefts.domain.fragment.fragment_tree.FragmentTree import FragmentTree
from clefts.domain.fragment.fragment_tree.FragmentTreeBuilder import FragmentTreeBuilder
from clefts.libs.mmkit.mmkit import Compound


class TestFragmentTreeBuilder(unittest.TestCase):
    def make_pattern_set(self):
        patterns = (
            CleavagePattern("[C:1]-[C:2]>>[C:1]", name="c-c cleavage", charge_mode="any"),
            CleavagePattern("[C:1]-[O:2]>>[C:1]", name="c-o cleavage", charge_mode="any"),
            CleavagePattern("[C:1]-[N:2]>>[C:1]", name="c-n cleavage", charge_mode="any"),
            CleavagePattern("[C:1]-[S:2]>>[C:1]", name="c-s cleavage", charge_mode="any"),
            CleavagePattern("[C:1]-[F:2]>>[C:1]", name="c-f cleavage", charge_mode="any"),
            CleavagePattern("[C:1]-[Cl:2]>>[C:1]", name="c-cl cleavage", charge_mode="any"),
            CleavagePattern("[C:1]-[Br:2]>>[C:1]", name="c-br cleavage", charge_mode="any"),
        )
        return CleavagePatternSet(patterns)

    def edge_pairs(self, tree):
        return {
            (tree.get_node(edge.source_id).smiles, tree.get_node(edge.target_id).smiles)
            for edge in (tree.get_edges(i) for i in range(tree.num_edges))
        }

    def test_builds_fragment_tree_and_exposes_convenience_methods(self):
        pattern_set = self.make_pattern_set()
        compound = Compound.from_smiles("CCCOCNCSCF")
        builder = FragmentTreeBuilder(max_depth=2, cleavage_pattern_set=pattern_set)

        tree = builder.build(compound)
        alias_tree = builder.create_fragment_tree(compound)
        static_tree = FragmentTreeBuilder.from_compound(
            compound=compound,
            cleavage_pattern_set=pattern_set,
            max_depth=2,
        )

        self.assertIsInstance(tree, FragmentTree)
        self.assertEqual(tree.smiles, compound.smiles)
        self.assertEqual(alias_tree.num_nodes, tree.num_nodes)
        self.assertEqual(static_tree.num_edges, tree.num_edges)

        node_smiles = set(tree.node_smiles)
        self.assertIn("CCCOCNCSCF", node_smiles)
        self.assertIn("CCOCNCSCF", node_smiles)
        self.assertIn("CCOCNCSC", node_smiles)
        self.assertIn("C", node_smiles)
        self.assertGreater(tree.num_edges, 10)

        pairs = self.edge_pairs(tree)
        self.assertIn(("CCCOCNCSCF", "CCOCNCSCF"), pairs)
        self.assertIn(("CCOCNCSCF", "CCOCNCSC"), pairs)

        self.assertEqual(tuple(tree.get_root_node_ids()), (0,))
        root_out_edges = tree.get_out_edges(0)
        self.assertGreater(len(root_out_edges), 0)
        self.assertEqual(root_out_edges[0].source_id, 0)

        second_depth_edge = next(
            tree.get_edges(edge_id)
            for edge_id in range(tree.num_edges)
            if tree.get_edges(edge_id).source_id != 0
        )
        self.assertGreaterEqual(len(tree.get_in_edges(second_depth_edge.target_id)), 1)
        self.assertIn(
            tree.get_node(second_depth_edge.source_id),
            tree.get_parent_nodes(second_depth_edge.target_id),
        )
        self.assertIn(
            tree.get_node(second_depth_edge.target_id),
            tree.get_child_nodes(second_depth_edge.source_id),
        )

        for step in second_depth_edge.fragment_step_strs:
            self.assertIn("cleavage_id=", step)
            self.assertIn("react_indices=", step)
            self.assertIn("prod_indices=", step)

        depths = tree.node_depths
        self.assertEqual(int(depths[0]), 0)
        self.assertEqual(int(depths[second_depth_edge.source_id]), 1)
        self.assertEqual(int(depths[second_depth_edge.target_id]), 2)
        nodes_by_depth = tree.get_nodes_by_depth()
        self.assertIn(0, nodes_by_depth)
        self.assertIn(1, nodes_by_depth)
        self.assertIn(2, nodes_by_depth)

    def test_all_supported_single_bond_patterns_cleave_first_depth(self):
        cases = (
            ("c-c cleavage", "CCCC", {"C", "CC", "CCC"}),
            ("c-o cleavage", "CCCOCC", {"CCC", "CC"}),
            ("c-n cleavage", "CCCNCC", {"CCC", "CC"}),
            ("c-s cleavage", "CCCSCC", {"CCC", "CC"}),
            ("c-f cleavage", "CCCF", {"CCC"}),
            ("c-cl cleavage", "CCCCl", {"CCC"}),
            ("c-br cleavage", "CCCBr", {"CCC"}),
        )
        pattern_set = self.make_pattern_set()

        for pattern_name, smiles, expected_products in cases:
            with self.subTest(pattern=pattern_name, smiles=smiles):
                compound = Compound.from_smiles(smiles)
                tree = FragmentTreeBuilder(max_depth=1, cleavage_pattern_set=pattern_set).build(compound)
                node_smiles = set(tree.node_smiles)

                self.assertEqual(tree.smiles, compound.smiles)
                self.assertTrue(expected_products.issubset(node_smiles))

                pattern = next(p for p in pattern_set.patterns if p.name == pattern_name)
                pattern_id = pattern_set.get_id(pattern)
                root_steps = (
                    step
                    for edge_id in range(tree.num_edges)
                    for step in tree.get_edges(edge_id).fragment_step_strs
                    if tree.get_edges(edge_id).source_id == 0
                )
                self.assertTrue(
                    any(f"cleavage_id={pattern_id}" in step for step in root_steps),
                    f"{pattern_name} was not applied at the root node.",
                )

    def test_build_returns_root_only_when_no_patterns_match(self):
        pattern_set = CleavagePatternSet([
            CleavagePattern("[C:1]-[O:2]>>[C:1]", name="c-o cleavage", charge_mode="any")
        ])
        builder = FragmentTreeBuilder(max_depth=1, cleavage_pattern_set=pattern_set)
        compound = Compound.from_smiles("CCCC")

        tree = builder.build(compound)

        self.assertEqual(tree.smiles, compound.smiles)
        self.assertEqual(tree.num_nodes, 1)
        self.assertEqual(tree.num_edges, 0)
        self.assertEqual(tuple(tree.get_root_node_ids()), (0,))
        self.assertEqual(tree.get_in_edges(0), [])
        self.assertEqual(tree.get_out_edges(0), [])


if __name__ == "__main__":
    unittest.main()
