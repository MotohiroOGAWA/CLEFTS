import unittest

from clefts.domain.fragment.cleavage.CleavagePattern import CleavagePattern
from clefts.domain.fragment.cleavage.CleavagePatternSet import CleavagePatternSet
from clefts.libs.mmkit.mmkit import Compound


class TestCleavagePatternSet(unittest.TestCase):
    def setUp(self):
        self.c_o = CleavagePattern(
            smirks="[C:1]-[O:2]>>[C:1]",
            name="c-o cleavage",
            charge_mode="any",
        )
        self.c_n = CleavagePattern(
            smirks="[C:1]-[N:2]>>[C:1]",
            name="c-n cleavage",
            charge_mode="any",
        )
        self.c_s = CleavagePattern(
            smirks="[C:1]-[S:2]>>[C:1]",
            name="c-s cleavage",
            charge_mode="any",
        )

    def test_assigns_same_ids_for_same_patterns_in_different_order(self):
        first = CleavagePatternSet([self.c_s, self.c_o, self.c_n], name="first")
        second = CleavagePatternSet([self.c_n, self.c_s, self.c_o], name="second")

        self.assertEqual(first, second)
        self.assertTrue(first.equals(second))
        self.assertFalse(first.equals(second, include_name=True))
        self.assertEqual(first.get_id(self.c_o), second.get_id(self.c_o))
        self.assertEqual(first.get_id(self.c_n), second.get_id(self.c_n))
        self.assertEqual(first.get_id(self.c_s), second.get_id(self.c_s))

    def test_ignores_duplicate_cleavage_patterns(self):
        pattern_set = CleavagePatternSet([self.c_o, self.c_n, self.c_o])

        self.assertEqual(len(pattern_set), 2)
        self.assertEqual(pattern_set.ids, (0, 1))

    def test_fragment_by_id_applies_the_selected_cleavage(self):
        pattern_set = CleavagePatternSet([self.c_o, self.c_n])
        compound = Compound.from_smiles("CCCN(CC)CC")

        result = pattern_set.fragment_by_id(pattern_set.get_id(self.c_n), compound)

        self.assertIsNotNone(result)
        self.assertEqual(result.cleavage, self.c_n)
        self.assertEqual(len(result.products), 3)
        self.assertEqual(result.products[0].smiles, "CCC")

    def test_fragment_all_applies_all_matching_cleavages(self):
        pattern_set = CleavagePatternSet([self.c_o, self.c_n, self.c_s])
        compound = Compound.from_smiles("CCCOCC")

        results = pattern_set.fragment_all(compound)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].cleavage, self.c_o)
        self.assertEqual(len(results[0].products), 2)

    def test_round_trip_dict_keeps_ids_stable(self):
        pattern_set = CleavagePatternSet([self.c_s, self.c_o, self.c_n], name="test set")

        restored = CleavagePatternSet.from_dict(pattern_set.to_dict())

        self.assertEqual(restored, pattern_set)
        self.assertEqual(restored.name, "test set")
        self.assertEqual(restored.get_id(self.c_o), pattern_set.get_id(self.c_o))
        self.assertEqual(restored.get_id(self.c_n), pattern_set.get_id(self.c_n))
        self.assertEqual(restored.get_id(self.c_s), pattern_set.get_id(self.c_s))


if __name__ == "__main__":
    unittest.main()
