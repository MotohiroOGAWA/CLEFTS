from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from clefts.ml.input.fragment_tree_structure import FragmentTreeStructure


class TestFragmentTreeStructure(unittest.TestCase):
    def _make_node_graph(self, atom_counts: list[int]) -> tuple[Batch, torch.Tensor]:
        """Create node_graph and node_graph_offset from atom counts."""
        data_list = []

        offset_values = [0]
        total_atoms = 0

        for node_id, num_atoms in enumerate(atom_counts):
            x = torch.full(
                (num_atoms, 3),
                fill_value=float(node_id + 1),
                dtype=torch.float32,
            )

            if num_atoms >= 2:
                edge_index = torch.tensor(
                    [
                        list(range(num_atoms - 1)),
                        list(range(1, num_atoms)),
                    ],
                    dtype=torch.long,
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            data_list.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                )
            )

            total_atoms += num_atoms
            offset_values.append(total_atoms)

        node_graph = Batch.from_data_list(data_list)
        node_graph_offset = torch.tensor(offset_values, dtype=torch.long)

        return node_graph, node_graph_offset

    def _make_structure_a(self) -> FragmentTreeStructure:
        """
        Structure A.

        Nodes:
            0, 1

        Atom counts:
            node 0: 2 atoms
            node 1: 3 atoms

        Total atoms:
            5
        """
        node_graph, node_graph_offset = self._make_node_graph([2, 3])

        return FragmentTreeStructure(
            node_smiles=np.array(["A0", "A1"], dtype=object),
            node_graph=node_graph,
            node_graph_offset=node_graph_offset,

            # One edge: 0 -> 1
            edge_index=torch.tensor(
                [
                    [0],
                    [1],
                ],
                dtype=torch.long,
            ),

            # One cleavage event attached to edge 0
            cleavage_event_edge_index=torch.tensor(
                [0],
                dtype=torch.long,
            ),
            cleavage_event=torch.tensor(
                [
                    [10, 100, 1000, 0, 0],
                ],
                dtype=torch.long,
            ),

            # Atom indices are assumed to be global atom indices in this structure.
            cleavage_reactant_atom_idxs={
                1: torch.tensor(
                    [
                        [0],
                        [2],
                    ],
                    dtype=torch.long,
                ),
                2: torch.tensor(
                    [
                        [0, 1],
                    ],
                    dtype=torch.long,
                ),
            },
            cleavage_product_atom_idxs={
                1: torch.tensor(
                    [
                        [3],
                    ],
                    dtype=torch.long,
                ),
            },

            # Two samples
            sample_adduct_type_index=torch.tensor(
                [1, 2],
                dtype=torch.long,
            ),
            sample_ce_value=torch.tensor(
                [10.0, 20.0],
                dtype=torch.float32,
            ),

            # sample 0 -> edge 0, sample 1 -> edge 0
            sample_edge_index=torch.tensor(
                [
                    [0, 1],
                    [0, 0],
                ],
                dtype=torch.long,
            ),

            # Path edge indices.
            # -1 is padding and should not be shifted.
            sample_precursor_edge_index_path=torch.tensor(
                [
                    [0, -1],
                    [0, 0],
                ],
                dtype=torch.long,
            ),

            # Path belongs to sample 0 and 1.
            sample_precursor_path_index=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )

    def _make_structure_b(self) -> FragmentTreeStructure:
        """
        Structure B.

        Nodes:
            0, 1, 2

        Atom counts:
            node 0: 1 atom
            node 1: 2 atoms
            node 2: 1 atom

        Total atoms:
            4
        """
        node_graph, node_graph_offset = self._make_node_graph([1, 2, 1])

        return FragmentTreeStructure(
            node_smiles=np.array(["B0", "B1", "B2"], dtype=object),
            node_graph=node_graph,
            node_graph_offset=node_graph_offset,

            # Two edges: 0 -> 1, 1 -> 2
            edge_index=torch.tensor(
                [
                    [0, 1],
                    [1, 2],
                ],
                dtype=torch.long,
            ),

            # Two cleavage events attached to local edge 0 and 1
            cleavage_event_edge_index=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            cleavage_event=torch.tensor(
                [
                    [20, 200, 2000, 0, 0],
                    [21, 201, 2001, 1, 1],
                ],
                dtype=torch.long,
            ),

            # These should be shifted by atom offset of structure A, which is 5.
            cleavage_reactant_atom_idxs={
                1: torch.tensor(
                    [
                        [0],
                        [3],
                    ],
                    dtype=torch.long,
                ),
                2: torch.tensor(
                    [
                        [1, 2],
                    ],
                    dtype=torch.long,
                ),
            },
            cleavage_product_atom_idxs={
                1: torch.tensor(
                    [
                        [2],
                    ],
                    dtype=torch.long,
                ),
            },

            # One sample
            sample_adduct_type_index=torch.tensor(
                [3],
                dtype=torch.long,
            ),
            sample_ce_value=torch.tensor(
                [30.0],
                dtype=torch.float32,
            ),

            # sample 0 -> edge 0 and edge 1
            sample_edge_index=torch.tensor(
                [
                    [0, 0],
                    [0, 1],
                ],
                dtype=torch.long,
            ),

            sample_precursor_edge_index_path=torch.tensor(
                [
                    [0, 1],
                ],
                dtype=torch.long,
            ),

            # Path belongs to sample 0.
            sample_precursor_path_index=torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )

    def test_from_structures_raises_for_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            FragmentTreeStructure.from_structures([])

    def test_from_structures_combines_basic_counts(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        self.assertEqual(batched.num_nodes, 5)
        self.assertEqual(batched.num_edges, 3)
        self.assertEqual(batched.num_cleavage_events, 3)
        self.assertEqual(batched.num_samples, 3)

        np.testing.assert_array_equal(
            batched.node_smiles,
            np.array(["A0", "A1", "B0", "B1", "B2"], dtype=object),
        )

        self.assertEqual(batched.node_graph.num_graphs, 5)
        self.assertEqual(batched.node_graph.x.size(0), 9)

    def test_from_structures_offsets_node_graph_offset(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        # A offset: [0, 2, 5]
        # B offset: [0, 1, 3, 4] shifted by 5 -> [5, 6, 8, 9]
        # Combined: [0, 2, 5, 6, 8, 9]
        expected = torch.tensor(
            [0, 2, 5, 6, 8, 9],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.node_graph_offset,
            expected,
        )

    def test_from_structures_offsets_edge_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        # A edge: 0 -> 1
        # B edges are shifted by node offset 2:
        #   0 -> 1 becomes 2 -> 3
        #   1 -> 2 becomes 3 -> 4
        expected = torch.tensor(
            [
                [0, 2, 3],
                [1, 3, 4],
            ],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.edge_index,
            expected,
        )

    def test_from_structures_offsets_cleavage_event_edge_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        # A has 1 edge, so B's cleavage_event_edge_index [0, 1]
        # becomes [1, 2].
        expected = torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.cleavage_event_edge_index,
            expected,
        )

    def test_from_structures_concatenates_cleavage_event_without_row_offset(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        expected = torch.tensor(
            [
                [10, 100, 1000, 0, 0],
                [20, 200, 2000, 0, 0],
                [21, 201, 2001, 1, 1],
            ],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.cleavage_event,
            expected,
        )

    def test_from_structures_offsets_cleavage_atom_index_dicts(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        # Structure B atom indices are shifted by total atoms of structure A = 5.
        expected_reactant_len_1 = torch.tensor(
            [
                [0],
                [2],
                [5],
                [8],
            ],
            dtype=torch.long,
        )
        expected_reactant_len_2 = torch.tensor(
            [
                [0, 1],
                [6, 7],
            ],
            dtype=torch.long,
        )
        expected_product_len_1 = torch.tensor(
            [
                [3],
                [7],
            ],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.cleavage_reactant_atom_idxs[1],
            expected_reactant_len_1,
        )
        torch.testing.assert_close(
            batched.cleavage_reactant_atom_idxs[2],
            expected_reactant_len_2,
        )
        torch.testing.assert_close(
            batched.cleavage_product_atom_idxs[1],
            expected_product_len_1,
        )

    def test_from_structures_concatenates_sample_values(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        torch.testing.assert_close(
            batched.sample_adduct_type_index,
            torch.tensor([1, 2, 3], dtype=torch.long),
        )

        torch.testing.assert_close(
            batched.sample_ce_value,
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
        )

    def test_from_structures_offsets_sample_edge_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        # A sample_edge_index:
        #   sample [0, 1], edge [0, 0]
        #
        # B sample_edge_index:
        #   sample [0, 0], edge [0, 1]
        #
        # B is shifted by:
        #   sample_offset = 2
        #   edge_offset = 1
        #
        # B becomes:
        #   sample [2, 2], edge [1, 2]
        expected = torch.tensor(
            [
                [0, 1, 2, 2],
                [0, 0, 1, 2],
            ],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.sample_edge_index,
            expected,
        )

    def test_from_structures_offsets_sample_precursor_edge_index_path(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        # B paths are shifted by edge_offset=1.
        # Padding -1 is kept unchanged.
        expected = torch.tensor(
            [
                [0, -1],
                [0, 0],
                [1, 2],
            ],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.sample_precursor_edge_index_path,
            expected,
        )

    def test_from_structures_offsets_sample_precursor_path_index(self) -> None:
        structure_a = self._make_structure_a()
        structure_b = self._make_structure_b()

        batched = FragmentTreeStructure.from_structures(
            [structure_a, structure_b],
        )

        # B path index [0] is shifted by sample_offset=2.
        expected = torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        )

        torch.testing.assert_close(
            batched.sample_precursor_path_index,
            expected,
        )

    def test_from_structures_moves_to_device(self) -> None:
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
        self.assertEqual(batched.sample_adduct_type_index.device, device)
        self.assertEqual(batched.sample_ce_value.device, device)
        self.assertEqual(batched.sample_edge_index.device, device)
        self.assertEqual(batched.sample_precursor_edge_index_path.device, device)
        self.assertEqual(batched.sample_precursor_path_index.device, device)

        for tensor in batched.cleavage_reactant_atom_idxs.values():
            self.assertEqual(tensor.device, device)

        for tensor in batched.cleavage_product_atom_idxs.values():
            self.assertEqual(tensor.device, device)


if __name__ == "__main__":
    unittest.main()