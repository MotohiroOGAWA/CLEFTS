from __future__ import annotations

import unittest
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from clefts.ml.input.fragment_tree_structure import FragmentTreeStructure


class TestFragmentTreeStructure(unittest.TestCase):
    """Tests for FragmentTreeStructure.from_structures."""

    def test_from_structures_combines_all_fields_as_expected(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b]
        )

        # -------------------------
        # Basic counts
        # -------------------------
        self.assertEqual(batched.num_nodes, 5)
        self.assertEqual(batched.num_edges, 5)
        self.assertEqual(batched.num_cleavage_events, 5)
        self.assertEqual(batched.num_samples, 3)

        # -------------------------
        # node_smiles
        # -------------------------
        np.testing.assert_array_equal(
            batched.node_smiles,
            np.asarray(
                [
                    "A_node_0",
                    "A_node_1",
                    "B_node_0",
                    "B_node_1",
                    "B_node_2",
                ],
                dtype=object,
            ),
        )

        # -------------------------
        # node_graph
        # -------------------------
        self.assertTensorEqual(
            batched.node_graph.x,
            torch.tensor(
                [
                    [0.0],
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [100.0],
                    [101.0],
                    [102.0],
                    [103.0],
                    [104.0],
                    [105.0],
                    [106.0],
                    [107.0],
                ],
                dtype=torch.float32,
            ),
        )

        self.assertTensorEqual(
            batched.node_graph.batch,
            torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                ],
                dtype=torch.long,
            ),
        )

        self.assertTensorEqual(
            batched.node_graph.ptr,
            torch.tensor(
                [0, 4, 7, 9, 14, 15],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # node_graph_offset
        # -------------------------
        self.assertTensorEqual(
            batched.node_graph_offset,
            torch.tensor(
                [0, 4, 7, 9, 14, 15],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # edge_index
        # -------------------------
        self.assertTensorEqual(
            batched.edge_index,
            torch.tensor(
                [
                    [0, 1, 0, 2, 3],
                    [1, 0, 0, 3, 4],
                ],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # cleavage_event_edge_index
        # -------------------------
        self.assertTensorEqual(
            batched.cleavage_event_edge_index,
            torch.tensor(
                [0, 1, 2, 3, 4],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # tuple-length tables
        # -------------------------
        self.assertTensorEqual(
            batched.reactant_tuple_length_table,
            torch.tensor(
                [
                    [10, 20, 2],
                    [11, 21, 1],
                ],
                dtype=torch.long,
            ),
        )

        self.assertTensorEqual(
            batched.product_tuple_length_table,
            torch.tensor(
                [
                    [10, 20, 0, 1],
                    [11, 21, 0, 2],
                ],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # cleavage_atom_idxs
        # -------------------------
        # Important:
        # Values are local atom indices.
        # Therefore, they are NOT shifted by node_graph_offset.
        self.assertEqual(
            self._tensor_rows_to_tuples(batched.cleavage_atom_idxs[1]),
            [
                (2,),
                (0,),
                (3,),
            ],
        )

        self.assertEqual(
            self._tensor_rows_to_tuples(batched.cleavage_atom_idxs[2]),
            [
                (0, 1),
                (1, 2),
            ],
        )

        # -------------------------
        # cleavage_event
        # -------------------------
        # cleavage_event[:, 3] and cleavage_event[:, 4] should point to
        # the merged unique cleavage_atom_idxs rows.
        self.assertTensorEqual(
            batched.cleavage_event,
            torch.tensor(
                [
                    # structure_a event 0
                    # reactant: len2 old row 0 -> (0, 1) -> new row 0
                    # product:  len1 old row 0 -> (2,)   -> new row 0
                    [10, 20, 0, 0, 0],

                    # structure_a event 1
                    # reactant: len1 old row 1 -> (2,)   -> new row 0
                    # product:  len2 old row 1 -> (1, 2) -> new row 1
                    [11, 21, 0, 0, 1],

                    # structure_a event 2
                    # reactant: len2 old row 2 -> (0, 1) -> new row 0
                    # product:  len1 old row 2 -> (0,)   -> new row 1
                    [10, 20, 0, 0, 1],

                    # structure_b event 0
                    # reactant: len2 old row 0 -> (0, 1) -> new row 0
                    # product:  len1 old row 0 -> (2,)   -> new row 0
                    [10, 20, 0, 0, 0],

                    # structure_b event 1
                    # reactant: len1 old row 1 -> (3,)   -> new row 2
                    # product:  len2 old row 1 -> (1, 2) -> new row 1
                    [11, 21, 0, 2, 1],
                ],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # Restored cleavage event information
        # -------------------------
        restored = self._restore_cleavage_event_info(batched)

        expected_restored = [
            (
                0,
                (10, 20, 0),
                (0, 1),
                (2,),
            ),
            (
                1,
                (11, 21, 0),
                (2,),
                (1, 2),
            ),
            (
                2,
                (10, 20, 0),
                (0, 1),
                (0,),
            ),
            (
                3,
                (10, 20, 0),
                (0, 1),
                (2,),
            ),
            (
                4,
                (11, 21, 0),
                (3,),
                (1, 2),
            ),
        ]

        self.assertEqual(restored, expected_restored)

        # -------------------------
        # sample_adduct_type_index
        # -------------------------
        self.assertTensorEqual(
            batched.sample_adduct_type_index,
            torch.tensor(
                [0, 2, 1],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # sample_ce_value
        # -------------------------
        self.assertTensorEqual(
            batched.sample_ce_value,
            torch.tensor(
                [10.0, 20.0, 30.0],
                dtype=torch.float32,
            ),
        )

        # -------------------------
        # sample_edge_index
        # -------------------------
        self.assertTensorEqual(
            batched.sample_edge_index,
            torch.tensor(
                [
                    [0, 0, 1, 1, 2, 2],
                    [0, 2, 1, 2, 3, 4],
                ],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # sample_precursor_edge_index_path
        # -------------------------
        # structure_a has width 2.
        # structure_b has width 3.
        # Therefore, structure_a rows are padded to width 3.
        self.assertTensorEqual(
            batched.sample_precursor_edge_index_path,
            torch.tensor(
                [
                    [0, -1, -1],
                    [2, 1, -1],
                    [4, 3, -1],
                ],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # sample_precursor_path_index
        # -------------------------
        self.assertTensorEqual(
            batched.sample_precursor_path_index,
            torch.tensor(
                [0, 1, 2],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # device
        # -------------------------
        self.assertEqual(batched.device, batched.edge_index.device)

    def test_from_structures_single_structure_still_uniques_atom_rows(
        self) -> None:
        """Even one structure should have cleavage_atom_idxs uniqued and remapped."""

        structure_a = self._make_structure_a()

        batched = FragmentTreeStructure.from_structures([structure_a])

        self.assertEqual(
            self._tensor_rows_to_tuples(batched.cleavage_atom_idxs[1]),
            [
                (2,),
                (0,),
            ],
        )

        self.assertEqual(
            self._tensor_rows_to_tuples(batched.cleavage_atom_idxs[2]),
            [
                (0, 1),
                (1, 2),
            ],
        )

        self.assertTensorEqual(
            batched.cleavage_event,
            torch.tensor(
                [
                    [10, 20, 0, 0, 0],
                    [11, 21, 0, 0, 1],
                    [10, 20, 0, 0, 1],
                ],
                dtype=torch.long,
            ),
        )

        restored = self._restore_cleavage_event_info(batched)

        self.assertEqual(
            restored,
            [
                (
                    0,
                    (10, 20, 0),
                    (0, 1),
                    (2,),
                ),
                (
                    1,
                    (11, 21, 0),
                    (2,),
                    (1, 2),
                ),
                (
                    2,
                    (10, 20, 0),
                    (0, 1),
                    (0,),
                ),
            ],
        )

    def test_from_structures_moves_all_tensor_fields_to_device(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        device = torch.device("cpu")

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
            device=device,
        )

        self.assertEqual(batched.node_graph_offset.device, device)
        self.assertEqual(batched.edge_index.device, device)
        self.assertEqual(batched.cleavage_event_edge_index.device, device)
        self.assertEqual(batched.cleavage_event.device, device)
        self.assertEqual(batched.reactant_tuple_length_table.device, device)
        self.assertEqual(batched.product_tuple_length_table.device, device)
        self.assertEqual(batched.sample_adduct_type_index.device, device)
        self.assertEqual(batched.sample_ce_value.device, device)
        self.assertEqual(batched.sample_edge_index.device, device)
        self.assertEqual(batched.sample_precursor_edge_index_path.device, device)
        self.assertEqual(batched.sample_precursor_path_index.device, device)
        self.assertEqual(batched.node_graph.x.device, device)

        for atom_idxs in batched.cleavage_atom_idxs.values():
            self.assertEqual(atom_idxs.device, device)

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    def _make_structure_a(self) -> FragmentTreeStructure:
        return self._make_structure(
            node_smiles=[
                "A_node_0",
                "A_node_1",
            ],
            atom_counts=[
                4,
                3,
            ],
            x_start=0,
            edge_index=[
                [0, 1, 0],
                [1, 0, 0],
            ],
            cleavage_event_edge_index=[
                0,
                1,
                2,
            ],
            cleavage_event=[
                [10, 20, 0, 0, 0],
                [11, 21, 0, 1, 1],
                [10, 20, 0, 2, 2],
            ],
            cleavage_atom_idxs={
                1: [
                    [2],
                    [2],
                    [0],
                ],
                2: [
                    [0, 1],
                    [1, 2],
                    [0, 1],
                ],
            },
            reactant_tuple_length_table=[
                [11, 21, 1],
                [10, 20, 2],
            ],
            product_tuple_length_table=[
                [11, 21, 0, 2],
                [10, 20, 0, 1],
            ],
            sample_adduct_type_index=[
                0,
                2,
            ],
            sample_ce_value=[
                10.0,
                20.0,
            ],
            sample_edge_index=[
                [0, 0, 1, 1],
                [0, 2, 1, 2],
            ],
            sample_precursor_edge_index_path=[
                [0, -1],
                [2, 1],
            ],
            sample_precursor_path_index=[
                0,
                1,
            ],
        )

    def _make_structure_b(self) -> FragmentTreeStructure:
        return self._make_structure(
            node_smiles=[
                "B_node_0",
                "B_node_1",
                "B_node_2",
            ],
            atom_counts=[
                2,
                5,
                1,
            ],
            x_start=100,
            edge_index=[
                [0, 1],
                [1, 2],
            ],
            cleavage_event_edge_index=[
                0,
                1,
            ],
            cleavage_event=[
                [10, 20, 0, 0, 0],
                [11, 21, 0, 1, 1],
            ],
            cleavage_atom_idxs={
                1: [
                    [2],
                    [3],
                ],
                2: [
                    [0, 1],
                    [1, 2],
                ],
            },
            reactant_tuple_length_table=[
                [10, 20, 2],
                [11, 21, 1],
            ],
            product_tuple_length_table=[
                [10, 20, 0, 1],
                [11, 21, 0, 2],
            ],
            sample_adduct_type_index=[
                1,
            ],
            sample_ce_value=[
                30.0,
            ],
            sample_edge_index=[
                [0, 0],
                [0, 1],
            ],
            sample_precursor_edge_index_path=[
                [1, 0, -1],
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
        x_start: int,
        edge_index: List[List[int]],
        cleavage_event_edge_index: List[int],
        cleavage_event: List[List[int]],
        cleavage_atom_idxs: Dict[int, List[List[int]]],
        reactant_tuple_length_table: List[List[int]],
        product_tuple_length_table: List[List[int]],
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
            node_graph=self._make_node_graph_batch(
                atom_counts=atom_counts,
                x_start=x_start,
            ),
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
            reactant_tuple_length_table=torch.tensor(
                reactant_tuple_length_table,
                dtype=torch.long,
            )
            if len(reactant_tuple_length_table) > 0
            else torch.empty(
                (0, 3),
                dtype=torch.long,
            ),
            product_tuple_length_table=torch.tensor(
                product_tuple_length_table,
                dtype=torch.long,
            )
            if len(product_tuple_length_table) > 0
            else torch.empty(
                (0, 4),
                dtype=torch.long,
            ),
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
    def _restore_cleavage_event_info(
        self,
        structure: FragmentTreeStructure,
    ) -> List[Tuple[int, Tuple[int, int, int], Tuple[int, ...], Tuple[int, ...]]]:
        restored: List[
            Tuple[int, Tuple[int, int, int], Tuple[int, ...], Tuple[int, ...]]
        ] = []

        reactant_tuple_length_by_key = self._table_to_lookup(
            structure.reactant_tuple_length_table,
            key_width=2,
        )

        product_tuple_length_by_key = self._table_to_lookup(
            structure.product_tuple_length_table,
            key_width=3,
        )

        for event_index in range(structure.num_cleavage_events):
            event_edge_index = int(
                structure.cleavage_event_edge_index[event_index].item()
            )

            cleavage_id = int(
                structure.cleavage_event[event_index, 0].item()
            )
            reaction_id = int(
                structure.cleavage_event[event_index, 1].item()
            )
            product_molecule_id = int(
                structure.cleavage_event[event_index, 2].item()
            )

            reactant_row_index = int(
                structure.cleavage_event[event_index, 3].item()
            )
            product_row_index = int(
                structure.cleavage_event[event_index, 4].item()
            )

            reactant_tuple_length = reactant_tuple_length_by_key[
                (
                    cleavage_id,
                    reaction_id,
                )
            ]

            product_tuple_length = product_tuple_length_by_key[
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
    def _table_to_lookup(
        table: Tensor,
        *,
        key_width: int,
    ) -> Dict[Tuple[int, ...], int]:
        lookup: Dict[Tuple[int, ...], int] = {}

        for row in table.tolist():
            row_tuple = tuple(int(value) for value in row)
            key = row_tuple[:key_width]
            tuple_length = row_tuple[-1]
            lookup[key] = int(tuple_length)

        return lookup

    @staticmethod
    def _make_node_graph_batch(
        *,
        atom_counts: List[int],
        x_start: int,
    ) -> Batch:
        data_list: List[Data] = []

        value = int(x_start)

        for atom_count in atom_counts:
            x = torch.arange(
                value,
                value + int(atom_count),
                dtype=torch.float32,
            ).view(int(atom_count), 1)

            data_list.append(Data(x=x))
            value += int(atom_count)

        return Batch.from_data_list(data_list)

    @staticmethod
    def _make_node_graph_offset(atom_counts: List[int]) -> Tensor:
        offsets = [0]
        total = 0

        for atom_count in atom_counts:
            total += int(atom_count)
            offsets.append(total)

        return torch.tensor(
            offsets,
            dtype=torch.long,
        )

    @staticmethod
    def _tensor_rows_to_tuples(tensor: Tensor) -> List[Tuple[int, ...]]:
        return [
            tuple(int(value) for value in row.tolist())
            for row in tensor
        ]

    def assertTensorEqual(
        self,
        actual: Tensor,
        expected: Tensor,
    ) -> None:
        self.assertTrue(
            torch.equal(
                actual.cpu(),
                expected.cpu(),
            ),
            msg=(
                "\nActual:\n"
                f"{actual}\n"
                "Expected:\n"
                f"{expected}"
            ),
        )


if __name__ == "__main__":
    unittest.main()