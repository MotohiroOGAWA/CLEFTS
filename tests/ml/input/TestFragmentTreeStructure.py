from __future__ import annotations

import unittest
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from clefts.ml.input.fragment_tree_structure import FragmentTreeStructure


class TestFragmentTreeStructure(unittest.TestCase):
    """Tests for FragmentTreeStructure."""

    REACTANT_TUPLE_LENGTH_BY_EVENT_TYPE: Dict[Tuple[int, int], int] = {
        (100, 200): 2,
        (101, 201): 1,
    }

    PRODUCT_TUPLE_LENGTH_BY_EVENT_TYPE: Dict[Tuple[int, int, int], int] = {
        (100, 200, 0): 1,
        (101, 201, 0): 2,
    }

    def test_from_structures_combines_basic_counts(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        self.assertEqual(batched.num_nodes, 4)
        self.assertEqual(batched.num_edges, 4)
        self.assertEqual(batched.num_cleavage_events, 4)
        self.assertEqual(batched.num_samples, 2)

    def test_from_structures_offsets_node_graph_offset(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        expected = torch.tensor(
            [0, 3, 5, 7, 9],
            dtype=torch.long,
        )

        self.assertTrue(torch.equal(batched.node_graph_offset, expected))

    def test_from_structures_offsets_edge_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        expected = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ],
            dtype=torch.long,
        )

        self.assertTrue(torch.equal(batched.edge_index, expected))

    def test_from_structures_offsets_cleavage_event_edge_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        expected = torch.tensor(
            [0, 1, 2, 3],
            dtype=torch.long,
        )

        self.assertTrue(
            torch.equal(
                batched.cleavage_event_edge_index,
                expected,
            )
        )

    def test_from_structures_merges_cleavage_atom_idxs_uniquely(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        self.assertEqual(
            self._tensor_rows_to_tuples(batched.cleavage_atom_idxs[1]),
            [
                (3,),
                (2,),
            ],
        )

        self.assertEqual(
            self._tensor_rows_to_tuples(batched.cleavage_atom_idxs[2]),
            [
                (0, 1),
            ],
        )

    def test_from_structures_restores_cleavage_event_atom_tuples(self) -> None:
        """cleavage_event row indices should restore correct atom tuples."""

        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        restored = self._restore_cleavage_event_info(batched)

        expected = [
            (
                0,
                (100, 200, 0),
                (0, 1),
                (3,),
            ),
            (
                1,
                (101, 201, 0),
                (3,),
                (0, 1),
            ),
            (
                2,
                (100, 200, 0),
                (0, 1),
                (2,),
            ),
            (
                3,
                (101, 201, 0),
                (2,),
                (0, 1),
            ),
        ]

        self.assertEqual(restored, expected)

    def test_from_structures_concatenates_sample_values(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        self.assertTrue(
            torch.equal(
                batched.sample_adduct_type_index,
                torch.tensor([0, 1], dtype=torch.long),
            )
        )

        self.assertTrue(
            torch.equal(
                batched.sample_ce_value,
                torch.tensor([10.0, 20.0], dtype=torch.float32),
            )
        )

    def test_from_structures_offsets_sample_edge_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        expected = torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 1, 2, 3],
            ],
            dtype=torch.long,
        )

        self.assertTrue(torch.equal(batched.sample_edge_index, expected))

    def test_from_structures_offsets_sample_precursor_edge_index_path(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        expected = torch.tensor(
            [
                [0, -1],
                [3, -1],
            ],
            dtype=torch.long,
        )

        self.assertTrue(
            torch.equal(
                batched.sample_precursor_edge_index_path,
                expected,
            )
        )

    def test_from_structures_offsets_sample_precursor_path_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = self._batch([structure_a, structure_b])

        expected = torch.tensor(
            [0, 1],
            dtype=torch.long,
        )

        self.assertTrue(
            torch.equal(
                batched.sample_precursor_path_index,
                expected,
            )
        )

    def test_from_structures_requires_tuple_length_lookup(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        with self.assertRaises(ValueError):
            FragmentTreeStructure.from_structures(
                [structure_a, structure_b],
            )

    def test_from_structures_moves_to_device(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        device = torch.device("cpu")

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
            device=device,
            reactant_tuple_length_by_event_type=(
                self.REACTANT_TUPLE_LENGTH_BY_EVENT_TYPE
            ),
            product_tuple_length_by_event_type=(
                self.PRODUCT_TUPLE_LENGTH_BY_EVENT_TYPE
            ),
        )

        self.assertEqual(batched.edge_index.device, device)
        self.assertEqual(batched.cleavage_event.device, device)
        self.assertEqual(batched.cleavage_event_edge_index.device, device)
        self.assertEqual(batched.node_graph_offset.device, device)

        for atom_idxs in batched.cleavage_atom_idxs.values():
            self.assertEqual(atom_idxs.device, device)

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    def _make_structure_a(self) -> FragmentTreeStructure:
        return self._make_structure(
            node_smiles=[
                "a_node_0",
                "a_node_1",
            ],
            atom_counts=[
                3,
                2,
            ],
            edge_index=[
                [0, 1],
                [1, 0],
            ],
            cleavage_event_edge_index=[
                0,
                1,
            ],
            cleavage_event=[
                # Event type: (100, 200, 0)
                # reactant tuple length = 2
                # product tuple length = 1
                #
                # reactant row 0 -> cleavage_atom_idxs[2][0] = (0, 1)
                # product row 0  -> cleavage_atom_idxs[1][0] = (3,)
                [100, 200, 0, 0, 0],

                # Event type: (101, 201, 0)
                # reactant tuple length = 1
                # product tuple length = 2
                #
                # reactant row 1 -> cleavage_atom_idxs[1][1] = (3,)
                # product row 1  -> cleavage_atom_idxs[2][1] = (0, 1)
                #
                # These rows intentionally duplicate row 0.
                # from_structures should unique them and remap row indices.
                [101, 201, 0, 1, 1],
            ],
            cleavage_atom_idxs={
                1: [
                    [3],
                    [3],
                ],
                2: [
                    [0, 1],
                    [0, 1],
                ],
            },
            sample_adduct_type_index=[
                0,
            ],
            sample_ce_value=[
                10.0,
            ],
            sample_edge_index=[
                [0, 0],
                [0, 1],
            ],
            sample_precursor_edge_index_path=[
                [0, -1],
            ],
            sample_precursor_path_index=[
                0,
            ],
        )

    def _make_structure_b(self) -> FragmentTreeStructure:
        return self._make_structure(
            node_smiles=[
                "b_node_0",
                "b_node_1",
            ],
            atom_counts=[
                2,
                2,
            ],
            edge_index=[
                [0, 1],
                [1, 0],
            ],
            cleavage_event_edge_index=[
                0,
                1,
            ],
            cleavage_event=[
                # After batching, atom offset is 5.
                #
                # reactant row 0 -> (0, 1) + 5 = (5, 6)
                # product row 0  -> (2,) + 5 = (7,)
                [100, 200, 0, 0, 0],

                # reactant row 0 -> (2,) + 5 = (7,)
                # product row 0  -> (0, 1) + 5 = (5, 6)
                [101, 201, 0, 0, 0],
            ],
            cleavage_atom_idxs={
                1: [
                    [2],
                ],
                2: [
                    [0, 1],
                ],
            },
            sample_adduct_type_index=[
                1,
            ],
            sample_ce_value=[
                20.0,
            ],
            sample_edge_index=[
                [0, 0],
                [0, 1],
            ],
            sample_precursor_edge_index_path=[
                [1, -1],
            ],
            sample_precursor_path_index=[
                0,
            ],
        )

    def _make_structure(
        self,
        *,
        node_smiles: List[str],
        atom_counts: List[int],
        edge_index: List[List[int]],
        cleavage_event_edge_index: List[int],
        cleavage_event: List[List[int]],
        cleavage_atom_idxs: Dict[int, List[List[int]]],
        sample_adduct_type_index: List[int],
        sample_ce_value: List[float],
        sample_edge_index: List[List[int]],
        sample_precursor_edge_index_path: List[List[int]],
        sample_precursor_path_index: List[int],
    ) -> FragmentTreeStructure:
        return FragmentTreeStructure(
            node_smiles=np.asarray(
                node_smiles,
                dtype=object,
            ),
            node_graph=self._make_node_graph_batch(atom_counts),
            node_graph_offset=self._make_node_graph_offset(atom_counts),
            edge_index=torch.tensor(
                edge_index,
                dtype=torch.long,
            ),
            cleavage_event_edge_index=torch.tensor(
                cleavage_event_edge_index,
                dtype=torch.long,
            ),
            cleavage_event=torch.tensor(
                cleavage_event,
                dtype=torch.long,
            )
            if len(cleavage_event) > 0
            else torch.empty(
                (0, 5),
                dtype=torch.long,
            ),
            cleavage_atom_idxs={
                int(tuple_length): torch.tensor(
                    rows,
                    dtype=torch.long,
                )
                for tuple_length, rows in cleavage_atom_idxs.items()
            },
            sample_adduct_type_index=torch.tensor(
                sample_adduct_type_index,
                dtype=torch.long,
            ),
            sample_ce_value=torch.tensor(
                sample_ce_value,
                dtype=torch.float32,
            ),
            sample_edge_index=torch.tensor(
                sample_edge_index,
                dtype=torch.long,
            )
            if len(sample_edge_index) > 0
            else torch.empty(
                (2, 0),
                dtype=torch.long,
            ),
            sample_precursor_edge_index_path=torch.tensor(
                sample_precursor_edge_index_path,
                dtype=torch.long,
            )
            if len(sample_precursor_edge_index_path) > 0
            else torch.empty(
                (0, 0),
                dtype=torch.long,
            ),
            sample_precursor_path_index=torch.tensor(
                sample_precursor_path_index,
                dtype=torch.long,
            )
            if len(sample_precursor_path_index) > 0
            else torch.empty(
                (0,),
                dtype=torch.long,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _batch(
        self,
        structures: List[FragmentTreeStructure],
    ) -> FragmentTreeStructure:
        return FragmentTreeStructure.from_structures(
            structures,
            reactant_tuple_length_by_event_type=(
                self.REACTANT_TUPLE_LENGTH_BY_EVENT_TYPE
            ),
            product_tuple_length_by_event_type=(
                self.PRODUCT_TUPLE_LENGTH_BY_EVENT_TYPE
            ),
        )

    @staticmethod
    def _make_node_graph_batch(atom_counts: List[int]) -> Batch:
        data_list: List[Data] = []

        value_offset = 0

        for num_atoms in atom_counts:
            x = torch.arange(
                value_offset,
                value_offset + num_atoms,
                dtype=torch.float32,
            ).view(num_atoms, 1)

            data_list.append(Data(x=x))
            value_offset += num_atoms

        return Batch.from_data_list(data_list)

    @staticmethod
    def _make_node_graph_offset(atom_counts: List[int]) -> Tensor:
        offsets = [0]
        total = 0

        for num_atoms in atom_counts:
            total += int(num_atoms)
            offsets.append(total)

        return torch.tensor(
            offsets,
            dtype=torch.long,
        )

    def _restore_cleavage_event_info(
        self,
        structure: FragmentTreeStructure,
    ) -> List[Tuple[int, Tuple[int, int, int], Tuple[int, ...], Tuple[int, ...]]]:
        restored = []

        for event_index in range(structure.num_cleavage_events):
            event_edge_index = int(
                structure.cleavage_event_edge_index[event_index].item()
            )

            cleavage_id = int(structure.cleavage_event[event_index, 0].item())
            reaction_id = int(structure.cleavage_event[event_index, 1].item())
            product_molecule_id = int(
                structure.cleavage_event[event_index, 2].item()
            )

            reactant_row_index = int(
                structure.cleavage_event[event_index, 3].item()
            )
            product_row_index = int(
                structure.cleavage_event[event_index, 4].item()
            )

            reactant_tuple_length = self.REACTANT_TUPLE_LENGTH_BY_EVENT_TYPE[
                (
                    cleavage_id,
                    reaction_id,
                )
            ]

            product_tuple_length = self.PRODUCT_TUPLE_LENGTH_BY_EVENT_TYPE[
                (
                    cleavage_id,
                    reaction_id,
                    product_molecule_id,
                )
            ]

            reactant_atom_tuple = tuple(
                int(value)
                for value in structure.cleavage_atom_idxs[
                    reactant_tuple_length
                ][reactant_row_index].tolist()
            )

            product_atom_tuple = tuple(
                int(value)
                for value in structure.cleavage_atom_idxs[
                    product_tuple_length
                ][product_row_index].tolist()
            )

            restored.append(
                (
                    event_edge_index,
                    (
                        cleavage_id,
                        reaction_id,
                        product_molecule_id,
                    ),
                    reactant_atom_tuple,
                    product_atom_tuple,
                )
            )

        return restored

    @staticmethod
    def _tensor_rows_to_tuples(tensor: Tensor) -> List[Tuple[int, ...]]:
        return [
            tuple(int(value) for value in row.tolist())
            for row in tensor
        ]


if __name__ == "__main__":
    unittest.main()