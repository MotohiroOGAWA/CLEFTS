from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data

from clefts.ml.input.training_fragment_tree_structure import TrainingFragmentTreeStructure
from clefts.ml.input.training_fragment_tree_structure_builder import (
    TrainingFormulaTarget,
    TrainingFragmentTreeSample,
    TrainingFragmentTreeStructureBuilder,
)


class _FakeFormulaTensorizer:
    element_order = ("C", "H")

    def adducts_to_delta_tensor(self, candidates, *, dtype):
        return torch.zeros((len(candidates), 3), dtype=dtype)


class _FakeCleavageEdgeFNet:
    reactant_tuple_length_by_event_type = {}
    product_tuple_length_by_event_type = {}


class _FakeModel:
    fragmenter = object()
    mol_encoder = object()
    formula_tensorizer = _FakeFormulaTensorizer()
    ion_flat_candidates = [(0, object())]
    unsaturation_flat_candidates = [(0, object())]
    radical_flat_candidates = [(0, object())]
    cleavage_edge_fnet = _FakeCleavageEdgeFNet()

    @staticmethod
    def get_index_by_adduct_type(_adduct):
        return 0


class TestTrainingFragmentTreeStructureBuilder(unittest.TestCase):
    def test_fragmentation_depth_is_relative_to_precursor(self) -> None:
        class Node:
            def __init__(self, is_precursor=False):
                self.is_precursor = is_precursor

        pathway = type("Pathway", (), {"nodes": (Node(), Node(True), Node(), Node())})()
        self.assertEqual(
            TrainingFragmentTreeStructureBuilder._fragmentation_depth(pathway), 2
        )

    def test_to_structure_builds_peak_formula_terminal_expand_pointers(self) -> None:
        builder = TrainingFragmentTreeStructureBuilder(_FakeModel())
        self._populate_builder(builder)

        structure = builder.to_structure()

        self.assertIsInstance(structure, TrainingFragmentTreeStructure)
        self.assertTensorEqual(
            structure.sample_peak_mz,
            torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32),
        )
        self.assertTensorEqual(
            structure.sample_peak_intensity,
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
        )
        self.assertTensorEqual(
            structure.sample_peak_ptr,
            torch.tensor([0, 2, 3], dtype=torch.long),
        )

        # Peak 0 has one formula, peak 1 has no formula, peak 2 has one formula.
        self.assertTensorEqual(
            structure.peak_formula_ptr,
            torch.tensor([0, 1, 1, 2], dtype=torch.long),
        )
        self.assertTensorEqual(
            structure.target_formula,
            torch.tensor(
                [
                    [1.0, 2.0, 0.0],
                    [2.0, 4.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )

        self.assertTensorEqual(
            structure.formula_assignment_ptr,
            torch.tensor([0, 1, 2], dtype=torch.long),
        )
        self.assertTensorEqual(
            structure.target_terminal_node_index,
            torch.tensor([2, 1], dtype=torch.long),
        )
        self.assertTensorEqual(structure.target_ion_index, torch.tensor([3, 4], dtype=torch.long))
        self.assertTensorEqual(structure.target_unsaturation_index, torch.tensor([5, 6], dtype=torch.long))
        self.assertTensorEqual(structure.target_radical_index, torch.tensor([7, 8], dtype=torch.long))

        self.assertTensorEqual(
            structure.target_expand_node_index,
            torch.tensor([0, 1, 0], dtype=torch.long),
        )
        self.assertTensorEqual(
            structure.terminal_expand_ptr,
            torch.tensor([0, 2, 3], dtype=torch.long),
        )
        self.assertTensorEqual(structure.target_peak_depth, torch.tensor([-1, -1, -1]))
        self.assertTensorEqual(structure.terminal_path_ptr, torch.tensor([0, 0, 0]))

        # Backward-compatible flattened assignment views.
        self.assertTensorEqual(structure.target_node_index, torch.tensor([2, 1], dtype=torch.long))
        self.assertTensorEqual(structure.target_sample_index, torch.tensor([0, 1], dtype=torch.long))
        self.assertTensorEqual(structure.target_peak_index, torch.tensor([0, 0], dtype=torch.long))
        self.assertTensorEqual(structure.target_intensity, torch.tensor([10.0, 30.0]))
        self.assertTensorEqual(
            structure.target_assignment_formula,
            torch.tensor(
                [
                    [1.0, 2.0, 0.0],
                    [2.0, 4.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )
        self.assertTensorEqual(structure.target_node_keep, torch.tensor([0.0, 1.0, 1.0]))
        self.assertTensorEqual(structure.target_node_expand, torch.tensor([1.0, 1.0, 0.0]))
        self.assertTensorEqual(structure.target_edge_index, torch.tensor([[0, 1], [1, 0]]))
        self.assertTensorEqual(structure.target_edge_group_index, torch.tensor([0, 1]))

    def test_from_structures_offsets_nodes_and_pointers(self) -> None:
        builder_a = TrainingFragmentTreeStructureBuilder(_FakeModel())
        self._populate_builder(builder_a)
        structure_a = builder_a.to_structure()

        builder_b = TrainingFragmentTreeStructureBuilder(_FakeModel())
        self._populate_builder(builder_b)
        structure_b = builder_b.to_structure()

        batched = TrainingFragmentTreeStructure.from_structures([structure_a, structure_b])

        self.assertTensorEqual(
            batched.sample_peak_ptr,
            torch.tensor([0, 2, 3, 5, 6], dtype=torch.long),
        )
        self.assertTensorEqual(
            batched.peak_formula_ptr,
            torch.tensor([0, 1, 1, 2, 3, 3, 4], dtype=torch.long),
        )
        self.assertTensorEqual(
            batched.formula_assignment_ptr,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        )
        self.assertTensorEqual(
            batched.target_terminal_node_index,
            torch.tensor([2, 1, 5, 4], dtype=torch.long),
        )
        self.assertTensorEqual(
            batched.target_expand_node_index,
            torch.tensor([0, 1, 0, 3, 4, 3], dtype=torch.long),
        )
        self.assertTensorEqual(
            batched.terminal_expand_ptr,
            torch.tensor([0, 2, 3, 5, 6], dtype=torch.long),
        )
        self.assertTensorEqual(batched.target_peak_depth, torch.full((6,), -1, dtype=torch.long))

    def _populate_builder(self, builder: TrainingFragmentTreeStructureBuilder) -> None:
        builder.node_smiles = ["root", "mid", "terminal"]
        builder.node_graph = [
            Data(x=torch.tensor([[0.0]], dtype=torch.float32), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 1))),
            Data(x=torch.tensor([[1.0]], dtype=torch.float32), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 1))),
            Data(x=torch.tensor([[2.0]], dtype=torch.float32), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 1))),
        ]
        builder._node_graph_offset = [0, 1, 2, 3]
        builder.node_formula = [
            torch.tensor([3.0, 6.0, 0.0], dtype=torch.float32),
            torch.tensor([2.0, 4.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, 2.0, 0.0], dtype=torch.float32),
        ]
        builder.edge_src = [0, 1]
        builder.edge_dst = [1, 2]
        builder.samples = {
            0: TrainingFragmentTreeSample(
                adduct_type_index=0,
                ce_value=10.0,
                peak_mz=[100.0, 200.0],
                peak_intensity=[10.0, 20.0],
                edge_indexes={0, 1},
                target_formula_rows=[
                    TrainingFormulaTarget(
                        node_index=2,
                        peak_index=0,
                        group_index=0,
                        ion_index=3,
                        unsaturation_index=5,
                        radical_index=7,
                        formula_tensor=torch.tensor([1.0, 2.0, 0.0], dtype=torch.float32),
                        intensity=10.0,
                        expand_node_indexes=(0, 1),
                    )
                ],
            ),
            1: TrainingFragmentTreeSample(
                adduct_type_index=0,
                ce_value=20.0,
                peak_mz=[300.0],
                peak_intensity=[30.0],
                edge_indexes={0},
                target_formula_rows=[
                    TrainingFormulaTarget(
                        node_index=1,
                        peak_index=0,
                        group_index=0,
                        ion_index=4,
                        unsaturation_index=6,
                        radical_index=8,
                        formula_tensor=torch.tensor([2.0, 4.0, 0.0], dtype=torch.float32),
                        intensity=30.0,
                        expand_node_indexes=(0,),
                    )
                ],
            ),
        }

    def assertTensorEqual(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertTrue(torch.equal(actual.cpu(), expected.cpu()), f"\nactual={actual}\nexpected={expected}")


if __name__ == "__main__":
    unittest.main()
