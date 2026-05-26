import unittest

from clefts.domain.fragment.cleavage.CleavagePattern import CleavagePattern
from clefts.libs.mmkit.mmkit import Compound


class TestCleavagePattern(unittest.TestCase):
    def assert_fragment_product(
        self,
        smirks,
        reactant_smiles,
        expected_product_smiles,
    ):
        pattern = CleavagePattern(
            smirks=smirks,
            name="test cleavage",
            charge_mode="any",
        )
        compound = Compound.from_smiles(reactant_smiles)

        result = pattern.fragment(compound)

        self.assertIsNotNone(result)
        self.assertEqual(result.reactant_smiles, compound.smiles)
        self.assertGreater(len(result.products), 0)

        product = result.products[0]
        self.assertEqual(product.smiles, expected_product_smiles)
        self.assertEqual(len(product.reactant_indices), pattern.num_reactant_atoms)
        self.assertEqual(len(product.product_indices), pattern.num_product_atoms)

    def test_parse_from_string(self):
        smirks = "[C:1]-[O:2]>>[C:1]"
        pattern = CleavagePattern(
            smirks=smirks,
            name="c-o cleavage",
            charge_mode="any",
        )

        pattern_str = str(pattern)
        parsed = CleavagePattern.parse(pattern_str)

        self.assertEqual(parsed.smirks, smirks)
        self.assertEqual(parsed.name, "c-o cleavage")
        self.assertEqual(parsed.charge_mode, "any")

    def test_fragment_various_bond_cleavage_patterns(self):
        cases = (
            {
                "smirks": "[C:1]-[O:2]>>[C:1]",
                "reactant_smiles": "CCCOCC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 2,
            },
            {
                "smirks": "[C:1]-[N:2]>>[C:1]",
                "reactant_smiles": "CCCN(CC)CC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 3,
            },
            {
                "smirks": "[C:1]-[S:2]>>[C:1]",
                "reactant_smiles": "CCCSCC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 2,
            },
            {
                "smirks": "[C:1]-[Cl:2]>>[C:1]",
                "reactant_smiles": "CCCCCl",
                "expected_first_product_smiles": "CCCC",
                "expected_product_count": 1,
            },
            {
                "smirks": "[C:1]-[Br:2]>>[C:1]",
                "reactant_smiles": "CCCCBr",
                "expected_first_product_smiles": "CCCC",
                "expected_product_count": 1,
            },
            {
                "smirks": "[C:1]-[C:2](=[O:3])>>[C:1]",
                "reactant_smiles": "CCCC(=O)OCC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 1,
            },
        )

        for case in cases:
            with self.subTest(smirks=case["smirks"], reactant=case["reactant_smiles"]):
                pattern = CleavagePattern(
                    smirks=case["smirks"],
                    name="test cleavage",
                    charge_mode="any",
                )
                compound = Compound.from_smiles(case["reactant_smiles"])

                result = pattern.fragment(compound)

                self.assertIsNotNone(result)
                self.assertEqual(result.reactant_smiles, compound.smiles)
                self.assertEqual(len(result.products), case["expected_product_count"])

                product = result.products[0]
                self.assertEqual(product.smiles, case["expected_first_product_smiles"])
                self.assertEqual(len(product.reactant_indices), pattern.num_reactant_atoms)
                self.assertEqual(len(product.product_indices), pattern.num_product_atoms)

    def test_fragment_returns_none_when_pattern_does_not_match(self):
        pattern = CleavagePattern(
            smirks="[C:1]-[O:2]>>[C:1]",
            name="c-o cleavage",
            charge_mode="any",
        )
        compound = Compound.from_smiles("CCCC")

        result = pattern.fragment(compound)

        self.assertIsNone(result)

    def test_fragment_keeps_all_matching_products(self):
        pattern = CleavagePattern(
            smirks="[C:1]-[O:2]>>[C:1]",
            name="c-o cleavage",
            charge_mode="any",
        )
        compound = Compound.from_smiles("CCCOCCC")

        result = pattern.fragment(compound)

        self.assertIsNotNone(result)
        self.assertEqual(result.reactant_smiles, compound.smiles)
        self.assertGreaterEqual(len(result.products), 2)


if __name__ == "__main__":
    unittest.main()
