from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from clefts.ml.training.cleavage_training.dataset import first_stage_edge_mask
from clefts.ml.training.cleavage_training.pretraining_model import (
    atom_neighborhood_indices,
    canonical_fragment_smiles,
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


if __name__ == "__main__":
    unittest.main()
