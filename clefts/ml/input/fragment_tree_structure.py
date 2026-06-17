from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch


@dataclass(frozen=True)
class FragmentTreeStructure:
    """Torch-ready fragment tree structure."""

    # -------------------------
    # Node-level information
    # -------------------------
    node_smiles: np.ndarray
    # [N] Fragment node SMILES.
    # dtype=object or str.

    node_graph: Batch
    # Batched molecular graphs of all fragment nodes.

    node_graph_offset: Tensor
    # [N + 1] Prefix-sum atom offsets.

    # -------------------------
    # Fragment tree edges
    # -------------------------
    edge_index: Tensor
    # [2, E]
    #
    # edge_index[0, e] = source fragment node index
    # edge_index[1, e] = target fragment node index

    # -------------------------
    # Cleavage events
    # -------------------------
    cleavage_event_edge_index: Tensor
    # [M]

    cleavage_event: Tensor
    # [M, 5]
    #
    # columns:
    #   0: cleavage_id
    #   1: reaction_id
    #   2: product_molecule_id
    #   3: reactant_row_index
    #   4: product_row_index

    cleavage_reactant_atom_idxs: Dict[int, Tensor]
    # tuple_length -> [R_len, tuple_length]

    cleavage_product_atom_idxs: Dict[int, Tensor]
    # tuple_length -> [P_len, tuple_length]

    # -------------------------
    # Sample-level information
    # -------------------------
    sample_adduct_type_index: Tensor
    # [S]

    sample_ce_value: Tensor
    # [S]

    sample_edge_index: Tensor
    # [2, L]
    #
    # sample_edge_index[0, l] = sample index
    # sample_edge_index[1, l] = edge index

    sample_precursor_edge_index_path: Tensor
    # [P, D]

    sample_precursor_path_index: Tensor
    # [P]

    @property
    def num_nodes(self) -> int:
        return int(self.node_smiles.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_cleavage_events(self) -> int:
        return int(self.cleavage_event_edge_index.numel())

    @property
    def num_samples(self) -> int:
        return int(self.sample_adduct_type_index.numel())

    def to(self, device: torch.device | str) -> "FragmentTreeStructure":
        """Move tensor fields to device.

        Notes
        -----
        node_smiles is a numpy array and remains on CPU.
        """

        return FragmentTreeStructure(
            node_smiles=self.node_smiles,
            node_graph=self.node_graph.to(device),
            node_graph_offset=self.node_graph_offset.to(device),
            edge_index=self.edge_index.to(device),
            cleavage_event_edge_index=self.cleavage_event_edge_index.to(device),
            cleavage_event=self.cleavage_event.to(device),
            cleavage_reactant_atom_idxs={
                k: v.to(device)
                for k, v in self.cleavage_reactant_atom_idxs.items()
            },
            cleavage_product_atom_idxs={
                k: v.to(device)
                for k, v in self.cleavage_product_atom_idxs.items()
            },
            sample_adduct_type_index=self.sample_adduct_type_index.to(device),
            sample_ce_value=self.sample_ce_value.to(device),
            sample_edge_index=self.sample_edge_index.to(device),
            sample_precursor_edge_index_path=(
                self.sample_precursor_edge_index_path.to(device)
            ),
            sample_precursor_path_index=(
                self.sample_precursor_path_index.to(device)
            ),
        )