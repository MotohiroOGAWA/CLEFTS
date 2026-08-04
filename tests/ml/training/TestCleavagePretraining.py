from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from clefts.ml.training.cleavage_training.dataset import (
    first_stage_edge_mask,
    sample_first_stage_event_rows,
)
from clefts.ml.training.cleavage_training.pretraining_model import (
    atom_neighborhood_indices,
    canonical_fragment_smiles,
)
from clefts.ml.training.cleavage_training.preprocessing import (
    BalancedCleavageBatchSampler,
)


class TestCanonicalFragmentSmiles(unittest.TestCase):
    def test_distinguishes_atom_types_for_the_same_broad_match(self) -> None:
        self.assertEqual(canonical_fragment_smiles("CCO", (0, 1)), "CC")
        self.assertEqual(canonical_fragment_smiles("CCO", (1, 2)), "CO")

    def test_ignores_atom_map_numbers(self) -> None:
        plain = canonical_fragment_smiles("CCO", (0, 1, 2))
        mapped = canonical_fragment_smiles("[CH3:7][CH2:2][OH:9]", (0, 1, 2))
        self.assertEqual(plain, mapped)

    def test_keeps_bond_type_in_the_identity(self) -> None:
        self.assertNotEqual(
            canonical_fragment_smiles("CC", (0, 1)),
            canonical_fragment_smiles("C=C", (0, 1)),
        )


class TestAtomNeighborhoodIndices(unittest.TestCase):
    def test_radius_one_contains_only_directly_attached_atoms(self) -> None:
        self.assertEqual(
            atom_neighborhood_indices("CCCO", center_atom=1, radius=1),
            (0, 2),
        )

    def test_radius_two_contains_atoms_up_to_two_bonds_away(self) -> None:
        self.assertEqual(
            atom_neighborhood_indices("CCCO", center_atom=1, radius=2),
            (0, 2, 3),
        )

    def test_center_atom_is_not_its_own_surrounding_structure(self) -> None:
        self.assertNotIn(
            1,
            atom_neighborhood_indices("CCCO", center_atom=1, radius=2),
        )


class TestFirstStageEdgeMask(unittest.TestCase):
    def test_selects_only_edges_leaving_each_root(self) -> None:
        structure = SimpleNamespace(
            num_nodes=6,
            num_edges=4,
            edge_index=torch.tensor(
                [
                    [0, 1, 3, 4],
                    [1, 2, 4, 5],
                ],
                dtype=torch.long,
            ),
        )

        self.assertEqual(
            first_stage_edge_mask(structure).tolist(),
            [True, False, True, False],
        )

    def test_event_limit_samples_only_first_stage_events(self) -> None:
        structure = SimpleNamespace(
            num_nodes=4,
            num_edges=3,
            num_cleavage_events=4,
            edge_index=torch.tensor([[0, 1, 0], [1, 2, 3]], dtype=torch.long),
            cleavage_event_edge_index=torch.tensor([0, 1, 2, 2]),
        )
        for _ in range(10):
            selected = sample_first_stage_event_rows(structure, 2)
            self.assertEqual(len(selected), 2)
            self.assertTrue(set(selected).issubset({0, 2, 3}))

    def test_event_limit_rejects_non_positive_values(self) -> None:
        structure = SimpleNamespace(num_cleavage_events=0)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            sample_first_stage_event_rows(structure, 0)

    def test_validation_selection_is_deterministic(self) -> None:
        structure = SimpleNamespace(
            num_nodes=4,
            num_edges=3,
            num_cleavage_events=3,
            edge_index=torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long),
            cleavage_event_edge_index=torch.tensor([0, 1, 2]),
        )
        self.assertEqual(
            sample_first_stage_event_rows(structure, 2, randomize=False),
            (0, 1),
        )


class TestBalancedCleavageBatchSampler(unittest.TestCase):
    def test_every_batch_contains_positive_and_negative_reactant_identities(self) -> None:
        index = {
            "counts": {
                "pattern:0": 2,
                "reactant_structure:A": 1,
                "reactant_structure:B": 1,
            },
            "files_by_target": {
                "pattern:0": [0, 1],
                "reactant_structure:A": [0],
                "reactant_structure:B": [1],
            },
            "events_by_target": {
                "pattern:0": [(0, 0), (1, 0)],
                "reactant_structure:A": [(0, 0)],
                "reactant_structure:B": [(1, 0)],
            },
            "identity_files": {"A": [0], "B": [1]},
            "identity_events": {"A": [(0, 0)], "B": [(1, 0)]},
        }
        sampler = BalancedCleavageBatchSampler(
            dataset_size=2,
            batch_size=1,
            index=index,
            min_data_count=1000,
            patience=20,
            max_forced_per_batch=8,
            shuffle=False,
        )
        identity_by_file = {0: "A", 1: "B"}
        for batch in sampler:
            identities = [
                identity_by_file[index[0] if isinstance(index, tuple) else index]
                for index in batch
            ]
            self.assertLess(len(set(identities)), len(identities))
            self.assertGreaterEqual(len(set(identities)), 2)


if __name__ == "__main__":
    unittest.main()
