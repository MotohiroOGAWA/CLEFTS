import unittest

from clefts.domain.fragment.fragment_tree.CleavageEvent import CleavageEvent


class TestCleavageEvent(unittest.TestCase):
    def test_stores_event_fields_and_index_strings(self):
        event = CleavageEvent(
            event_id=10,
            cleavage_pattern_id=5,
            react_indices=(3, 4),
            prod_indices=(1,),
        )

        self.assertEqual(event.event_id, 10)
        self.assertEqual(event.cleavage_pattern_id, 5)
        self.assertEqual(event.react_indices, (3, 4))
        self.assertEqual(event.prod_indices, (1,))
        self.assertEqual(event.react_indices_str, "[3,4]")
        self.assertEqual(event.prod_indices_str, "[1]")
        self.assertEqual(event.to_record(), (10, 5, "[3,4]", "[1]"))

    def test_casts_index_values_to_ints(self):
        event = CleavageEvent(
            event_id="1",
            cleavage_pattern_id="3",
            react_indices=["4", 5],
            prod_indices=["6"],
        )

        self.assertEqual(event.event_id, 1)
        self.assertEqual(event.cleavage_pattern_id, 3)
        self.assertEqual(event.react_indices, (4, 5))
        self.assertEqual(event.prod_indices, (6,))

    def test_copy_returns_equal_event(self):
        event = CleavageEvent(
            event_id=1,
            cleavage_pattern_id=3,
            react_indices=(4, 5),
            prod_indices=(6,),
        )

        copied = event.copy()

        self.assertEqual(copied, event)
        self.assertIsNot(copied, event)


if __name__ == "__main__":
    unittest.main()
