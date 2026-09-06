from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data

from clefts.libs.mmkit.mmkit import Adduct
from clefts.ml.input.fragment_tree_structure import FragmentTreeStructure
from clefts.ml.specgen.fragment_tree_feature_model import FragmentTreeFeatureModel


class DummyConditionEncoder(nn.Module):
    """Small deterministic MS2 condition encoder for testing."""

    feature_dim = 2

    def forward(
        self,
        sample_adduct_type_index: Tensor,
        sample_ce_value: Tensor,
    ) -> Tensor:
        return torch.stack(
            [
                sample_adduct_type_index.to(torch.float32),
                sample_ce_value.to(torch.float32),
            ],
            dim=1,
        )


class DummyConditionToTreeProjection(nn.Module):
    """Small deterministic projection to tree graph representation."""

    def forward(
        self,
        condition_features: Tensor,
    ) -> Tensor:
        adduct = condition_features[:, 0]
        ce = condition_features[:, 1]

        return torch.stack(
            [
                adduct,
                ce,
                adduct + ce,
                torch.ones_like(adduct),
            ],
            dim=1,
        )


class TestBuildSampleTreePygBatch(unittest.TestCase):
    def test_build_sample_tree_pyg_batch_returns_expected_outputs(self) -> None:
        model = self._make_model()
        ft_features = self._make_fragment_tree_features()

        (
            sample_tree_batch,
            condition_tree_repr,
            kept_sample_ids,
        ) = model._build_sample_tree_pyg_batch(ft_features)

        # -------------------------
        # kept_sample_ids
        # -------------------------
        self.assertTensorEqual(
            kept_sample_ids,
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )

        self.assertTensorEqual(
            sample_tree_batch.kept_sample_ids,
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # condition_tree_repr
        # -------------------------
        self.assertTensorClose(
            condition_tree_repr,
            torch.tensor(
                [
                    [2.0, 10.0],
                    [5.0, 20.0],
                ],
                dtype=torch.float32,
            ),
        )

        # -------------------------
        # PyG Batch basic shape
        # -------------------------
        self.assertEqual(sample_tree_batch.num_graphs, 2)
        self.assertEqual(sample_tree_batch.num_nodes, 7)
        self.assertEqual(sample_tree_batch.num_edges, 5)

        self.assertTensorEqual(
            sample_tree_batch.ptr,
            torch.tensor(
                [0, 4, 7],
                dtype=torch.long,
            ),
        )

        self.assertTensorEqual(
            sample_tree_batch.batch,
            torch.tensor(
                [0, 0, 0, 0, 1, 1, 1],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # node_id_global
        # -------------------------
        # sample 0 uses global nodes: [0, 1, 2, 3]
        # sample 1 uses global nodes: [2, 3, 4]
        self.assertTensorEqual(
            sample_tree_batch.node_id_global,
            torch.tensor(
                [0, 1, 2, 3, 2, 3, 4],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # edge_id_global
        # -------------------------
        # sample 0 uses global edges: [0, 1, 2]
        # sample 1 uses global edges: [3, 4]
        self.assertTensorEqual(
            sample_tree_batch.edge_id_global,
            torch.tensor(
                [0, 1, 2, 3, 4],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # edge_index
        # -------------------------
        # sample 0 local edges:
        #   edge 0: 0 -> 1
        #   edge 1: 1 -> 2
        #   edge 2: 0 -> 3
        #
        # sample 1 local edges before batching:
        #   edge 3: global 3 -> 2 = local 1 -> 0
        #   edge 4: global 2 -> 4 = local 0 -> 2
        #
        # After PyG batching, sample 1 node offset is +4.
        self.assertTensorEqual(
            sample_tree_batch.edge_index,
            torch.tensor(
                [
                    [0, 1, 0, 5, 4],
                    [1, 2, 3, 4, 6],
                ],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # edge_attr
        # -------------------------
        self.assertTensorClose(
            sample_tree_batch.edge_attr,
            torch.tensor(
                [
                    [0.0, 100.0],
                    [1.0, 101.0],
                    [2.0, 102.0],
                    [3.0, 103.0],
                    [4.0, 104.0],
                ],
                dtype=torch.float32,
            ),
        )

        # -------------------------
        # node features
        # -------------------------
        # The function should only add node_role_one_hot.
        # It should NOT add adduct one-hot or neutral-HS one-hot here.
        #
        # node feature dim:
        #   mol_x dim 2 + node_role_one_hot dim 3 = 5
        self.assertEqual(sample_tree_batch.x.size(1), 5)

        expected_x = torch.tensor(
            [
                # sample 0
                # global node 0: precursor root
                [0.0, 10.0, 0.0, 1.0, 0.0],
                # global node 1: precursor path node
                [1.0, 11.0, 1.0, 0.0, 0.0],
                # global node 2: precursor path node
                [2.0, 12.0, 1.0, 0.0, 0.0],
                # global node 3: normal node
                [3.0, 13.0, 0.0, 0.0, 1.0],

                # sample 1
                # global node 2: precursor path node
                [2.0, 12.0, 1.0, 0.0, 0.0],
                # global node 3: precursor root
                [3.0, 13.0, 0.0, 1.0, 0.0],
                # global node 4: precursor path node
                [4.0, 14.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        self.assertTensorClose(
            sample_tree_batch.x,
            expected_x,
        )

        # -------------------------
        # precursor_pathway_ptr
        # -------------------------
        self.assertTensorEqual(
            sample_tree_batch.precursor_pathway_ptr,
            torch.tensor(
                [0, 1, 2],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # edge_ptr
        # -------------------------
        self.assertTensorEqual(
            sample_tree_batch.edge_ptr,
            torch.tensor(
                [0, 3, 5],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # precursor_pathway_seq
        # -------------------------
        # Before Batch-global conversion:
        #
        # sample 0:
        #   global path edges [0, 1]
        #   local node-edge-node sequence:
        #       [0, 0, 1, 1, 2]
        #
        # sample 1:
        #   global path edges [3, 4]
        #   local node-edge-node sequence:
        #       [1, 0, 0, 1, 2]
        #
        # After Batch-global conversion:
        #   sample 0 node offset = 0, edge offset = 0
        #   sample 1 node offset = 4, edge offset = 3
        #
        # Therefore:
        #   sample 0 -> [0, 0, 1, 1, 2]
        #   sample 1 -> [5, 3, 4, 4, 6]
        self.assertTensorEqual(
            sample_tree_batch.precursor_pathway_seq,
            torch.tensor(
                [
                    [0, 0, 1, 1, 2],
                    [5, 3, 4, 4, 6],
                ],
                dtype=torch.long,
            ),
        )

        # -------------------------
        # sample_id attribute
        # -------------------------
        self.assertTensorEqual(
            sample_tree_batch.sample_id,
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )

    def test_precursor_targets_are_assigned_to_terminal_nodes(self) -> None:
        model = self._make_model()
        ft_features = self._make_fragment_tree_features()
        ft_features.structure = replace(
            ft_features.structure,
            precursor_edge_index_path=torch.tensor(
                [
                    [0, 1],
                    [2, -1],
                    [3, 4],
                ],
                dtype=torch.long,
            ),
            precursor_unsaturation_index=torch.tensor(
                [0, 1, 1],
                dtype=torch.long,
            ),
            precursor_radical_index=torch.tensor(
                [0, 1, 1],
                dtype=torch.long,
            ),
            precursor_sample_index=torch.tensor(
                [0, 0, 1],
                dtype=torch.long,
            ),
        )

        sample_tree_batch, _, _ = model._build_sample_tree_pyg_batch(
            ft_features
        )

        self.assertTensorEqual(
            sample_tree_batch.node_precursor_ion_flat_index,
            torch.tensor(
                [-1, -1, 0, 0, -1, -1, 0],
                dtype=torch.long,
            ),
        )
        self.assertTensorEqual(
            sample_tree_batch.node_precursor_unsaturation_flat_index,
            torch.tensor(
                [-1, -1, 0, 1, -1, -1, 1],
                dtype=torch.long,
            ),
        )
        self.assertTensorEqual(
            sample_tree_batch.node_precursor_radical_flat_index,
            torch.tensor(
                [-1, -1, 0, 1, -1, -1, 1],
                dtype=torch.long,
            ),
        )

    def test_build_sample_tree_pyg_batch_does_not_require_optional_hs_features(
        self) -> None:
        model = self._make_model()
        ft_features = self._make_fragment_tree_features()

        self.assertFalse(hasattr(model, "_adduct_type_idx_to_one_hot_vec"))
        self.assertFalse(hasattr(model, "_neutral_hs_adduct_idx_to_one_hot_vec"))

        sample_tree_batch, _, _ = model._build_sample_tree_pyg_batch(
            ft_features
        )

        # mol_x dim 2 + node role dim 3
        self.assertEqual(sample_tree_batch.x.size(1), 5)

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    def _make_model(self) -> FragmentTreeFeatureModel:
        model = FragmentTreeFeatureModel.__new__(FragmentTreeFeatureModel)
        nn.Module.__init__(model)

        model._condition_encoder = DummyConditionEncoder()
        model.ms2_condition_to_tree_proj = DummyConditionToTreeProjection()

        # _build_sample_tree_pyg_batch only needs these Graphormer-like attrs.
        model.tree_encoder = SimpleNamespace(
            dim=4,
            condition_dim=2,
            graph_repr_dim=4,
        )
        model.main_adduct_types = {
            2: Adduct.parse("[M+H]+"),
            5: Adduct.parse("[M+Na]+"),
        }
        model._build_precursor_ion_flat_index = (
            lambda *, main_adduct_type, count, device: torch.zeros(
                (count,), dtype=torch.long, device=device
            )
        )
        model._build_precursor_unsaturation_flat_index = (
            lambda *, main_adduct_type, unsaturation_index, device: unsaturation_index.to(device).long()
        )
        model._build_precursor_radical_flat_index = (
            lambda *, main_adduct_type, radical_index, device: radical_index.to(device).long()
        )

        return model

    def _make_fragment_tree_features(self) -> SimpleNamespace:
        structure = self._make_structure()

        mol_x = torch.tensor(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
                [4.0, 14.0],
            ],
            dtype=torch.float32,
        )
        # [N, mol_graph_dim]

        edge_attr = torch.tensor(
            [
                [0.0, 100.0],
                [1.0, 101.0],
                [2.0, 102.0],
                [3.0, 103.0],
                [4.0, 104.0],
            ],
            dtype=torch.float32,
        )
        # [E, edge_dim]

        return SimpleNamespace(
            structure=structure,
            mol_x=mol_x,
            edge_attr=edge_attr,
        )

    def _make_structure(self) -> FragmentTreeStructure:
        node_smiles = np.asarray(
            [
                "node_0",
                "node_1",
                "node_2",
                "node_3",
                "node_4",
            ],
            dtype=object,
        )

        node_graph = self._make_node_graph(num_nodes=5)

        # Global fragment-tree edges:
        #
        # edge 0: 0 -> 1
        # edge 1: 1 -> 2
        # edge 2: 0 -> 3
        # edge 3: 3 -> 2
        # edge 4: 2 -> 4
        edge_index = torch.tensor(
            [
                [0, 1, 0, 3, 2],
                [1, 2, 3, 2, 4],
            ],
            dtype=torch.long,
        )

        return FragmentTreeStructure(
            node_smiles=node_smiles,
            node_graph=node_graph,
            node_graph_offset=torch.tensor(
                [0, 1, 2, 3, 4, 5],
                dtype=torch.long,
            ),
            node_formula=torch.tensor(
                [
                    [1.0, 2.0, 0.0],
                    [2.0, 4.0, 0.0],
                    [3.0, 6.0, 0.0],
                    [4.0, 8.0, 0.0],
                    [5.0, 10.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            formula_element_order=("C", "H"),
            ion_formula_delta=torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
            unsaturation_formula_delta=torch.tensor([[0.0, -2.0, 0.0]], dtype=torch.float32),
            radical_formula_delta=torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32),
            edge_index=edge_index,
            tree_sample_ptr=torch.tensor([0, 2], dtype=torch.long),

            # These fields are not used by _build_sample_tree_pyg_batch,
            # but they are required by FragmentTreeStructure.
            cleavage_event_edge_index=torch.empty(
                (0,),
                dtype=torch.long,
            ),
            cleavage_event=torch.empty(
                (0, 5),
                dtype=torch.long,
            ),
            cleavage_atom_idxs={},
            reactant_tuple_length_table=torch.empty(
                (0, 3),
                dtype=torch.long,
            ),
            product_tuple_length_table=torch.empty(
                (0, 4),
                dtype=torch.long,
            ),

            # sample 0:
            #   sample_edge_index edges: [0, 2]
            #   precursor path edges: [0, 1]
            #   final graph edges: [0, 1, 2]
            #
            # sample 1:
            #   sample_edge_index edges: [3, 4]
            #   precursor path edges: [3, 4]
            #   final graph edges: [3, 4]
            sample_adduct_type_index=torch.tensor(
                [2, 5],
                dtype=torch.long,
            ),
            sample_ce_value=torch.tensor(
                [10.0, 20.0],
                dtype=torch.float32,
            ),
            sample_edge_index=torch.tensor(
                [
                    [0, 0, 1, 1],
                    [0, 2, 3, 4],
                ],
                dtype=torch.long,
            ),
            precursor_edge_index_path=torch.tensor(
                [
                    [0, 1],
                    [3, 4],
                ],
                dtype=torch.long,
            ),
            precursor_unsaturation_index=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            precursor_radical_index=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            precursor_sample_index=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )

    @staticmethod
    def _make_node_graph(num_nodes: int) -> Batch:
        data_list: List[Data] = []

        for node_index in range(num_nodes):
            data_list.append(
                Data(
                    x=torch.tensor(
                        [[float(node_index)]],
                        dtype=torch.float32,
                    )
                )
            )

        return Batch.from_data_list(data_list)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
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

    def assertTensorClose(
        self,
        actual: Tensor,
        expected: Tensor,
        *,
        atol: float = 1e-6,
    ) -> None:
        self.assertTrue(
            torch.allclose(
                actual.cpu(),
                expected.cpu(),
                atol=atol,
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