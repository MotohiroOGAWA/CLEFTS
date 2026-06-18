from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional

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
    
    @property
    def cleavage_pattern_ids(self) -> Tensor:
        return self.cleavage_event[:, 0] # [M]

    @property
    def cleavage_reaction_ids(self) -> Tensor:
        return self.cleavage_event[:, 1] # [M]
    
    @property
    def cleavage_product_molecule_ids(self) -> Tensor:
        return self.cleavage_event[:, 2] # [M]

    @property
    def cleavage_reactant_row_indices(self) -> Tensor:
        return self.cleavage_event[:, 3] # [M]
    
    @property
    def cleavage_product_row_indices(self) -> Tensor:
        return self.cleavage_event[:, 4] # [M]
    
    @property
    def device(self) -> torch.device:
        return self.edge_index.device

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

    @classmethod
    def from_structures(
        cls,
        structures: Sequence["FragmentTreeStructure"],
        device: Optional[torch.device] = None,
    ) -> FragmentTreeStructure:
        """
        Combine multiple FragmentTreeStructure objects into one batched structure.

        Parameters
        ----------
        structures:
            FragmentTreeStructure objects to combine.

        device:
            Optional target device. If provided, all tensor fields and node_graph
            are moved to this device.

        Returns
        -------
        batched:
            Combined FragmentTreeStructure.

        Notes
        -----
        Offset handling:
            - node indices are shifted by cumulative num_nodes.
            - edge indices are shifted by cumulative num_edges.
            - sample indices are shifted by cumulative num_samples.
            - atom indices in cleavage atom index dictionaries are shifted by
            cumulative atom count.

        Important
        ---------
        cleavage_event[:, 3] and cleavage_event[:, 4] are kept unchanged.
        This is because the tuple_length key needed to offset these row indices
        is not stored in cleavage_event itself.
        """
        if len(structures) == 0:
            raise ValueError("structures must not be empty")

        if device is not None:
            structures = [structure.to(device) for structure in structures]

        first = structures[0]
        tensor_device = first.edge_index.device

        # -------------------------
        # Node SMILES
        # -------------------------
        node_smiles = np.concatenate(
            [structure.node_smiles for structure in structures],
            axis=0,
        )

        # -------------------------
        # Node molecular graphs
        # -------------------------
        node_graph_data_list = []
        for structure in structures:
            node_graph_data_list.extend(structure.node_graph.to_data_list())

        node_graph = Batch.from_data_list(node_graph_data_list)
        if device is not None:
            node_graph = node_graph.to(device)

        # -------------------------
        # node_graph_offset
        # -------------------------
        node_graph_offset_parts = []
        atom_offset = 0

        for structure in structures:
            offset = structure.node_graph_offset.to(tensor_device)

            # Use all offsets except the final total.
            node_graph_offset_parts.append(offset[:-1] + atom_offset)
            atom_offset += int(offset[-1].item())

        node_graph_offset = torch.cat(
            node_graph_offset_parts
            + [
                torch.tensor(
                    [atom_offset],
                    dtype=first.node_graph_offset.dtype,
                    device=tensor_device,
                )
            ],
            dim=0,
        )

        # -------------------------
        # Fragment tree edge_index
        # -------------------------
        edge_index_parts = []
        node_offset = 0

        for structure in structures:
            edge_index_parts.append(
                structure.edge_index.to(tensor_device) + node_offset
            )
            node_offset += structure.num_nodes

        edge_index = torch.cat(edge_index_parts, dim=1)

        # -------------------------
        # cleavage_event_edge_index
        # -------------------------
        cleavage_event_edge_index_parts = []
        edge_offset = 0

        for structure in structures:
            cleavage_event_edge_index_parts.append(
                structure.cleavage_event_edge_index.to(tensor_device) + edge_offset
            )
            edge_offset += structure.num_edges

        cleavage_event_edge_index = torch.cat(
            cleavage_event_edge_index_parts,
            dim=0,
        )

        # -------------------------
        # cleavage_event
        # -------------------------
        cleavage_event = torch.cat(
            [
                structure.cleavage_event.to(tensor_device)
                for structure in structures
            ],
            dim=0,
        )

        # -------------------------
        # cleavage atom index dictionaries
        # -------------------------
        cleavage_reactant_atom_idxs = cls._concat_atom_index_dicts_with_atom_offset(
            structures=structures,
            field_name="cleavage_reactant_atom_idxs",
            device=tensor_device,
        )

        cleavage_product_atom_idxs = cls._concat_atom_index_dicts_with_atom_offset(
            structures=structures,
            field_name="cleavage_product_atom_idxs",
            device=tensor_device,
        )

        # -------------------------
        # sample-level information
        # -------------------------
        sample_adduct_type_index = torch.cat(
            [
                structure.sample_adduct_type_index.to(tensor_device)
                for structure in structures
            ],
            dim=0,
        )

        sample_ce_value = torch.cat(
            [
                structure.sample_ce_value.to(tensor_device)
                for structure in structures
            ],
            dim=0,
        )

        # -------------------------
        # sample_edge_index
        # -------------------------
        sample_edge_index_parts = []

        sample_offset = 0
        edge_offset = 0

        for structure in structures:
            sample_edge_index = structure.sample_edge_index.to(tensor_device).clone()

            if sample_edge_index.numel() > 0:
                sample_edge_index[0, :] += sample_offset
                sample_edge_index[1, :] += edge_offset

            sample_edge_index_parts.append(sample_edge_index)

            sample_offset += structure.num_samples
            edge_offset += structure.num_edges

        sample_edge_index = torch.cat(sample_edge_index_parts, dim=1)

        # -------------------------
        # sample_precursor_edge_index_path
        # -------------------------
        sample_precursor_edge_index_path_parts = []

        edge_offset = 0

        for structure in structures:
            path = structure.sample_precursor_edge_index_path.to(tensor_device)

            # Keep negative padding values such as -1 unchanged.
            shifted_path = torch.where(
                path >= 0,
                path + edge_offset,
                path,
            )

            sample_precursor_edge_index_path_parts.append(shifted_path)
            edge_offset += structure.num_edges

        sample_precursor_edge_index_path = torch.cat(
            sample_precursor_edge_index_path_parts,
            dim=0,
        )

        # -------------------------
        # sample_precursor_path_index
        # -------------------------
        sample_precursor_path_index_parts = []

        sample_offset = 0

        for structure in structures:
            path_index = structure.sample_precursor_path_index.to(tensor_device)

            shifted_path_index = torch.where(
                path_index >= 0,
                path_index + sample_offset,
                path_index,
            )

            sample_precursor_path_index_parts.append(shifted_path_index)
            sample_offset += structure.num_samples

        sample_precursor_path_index = torch.cat(
            sample_precursor_path_index_parts,
            dim=0,
        )

        return cls(
            node_smiles=node_smiles,
            node_graph=node_graph,
            node_graph_offset=node_graph_offset,
            edge_index=edge_index,
            cleavage_event_edge_index=cleavage_event_edge_index,
            cleavage_event=cleavage_event,
            cleavage_reactant_atom_idxs=cleavage_reactant_atom_idxs,
            cleavage_product_atom_idxs=cleavage_product_atom_idxs,
            sample_adduct_type_index=sample_adduct_type_index,
            sample_ce_value=sample_ce_value,
            sample_edge_index=sample_edge_index,
            sample_precursor_edge_index_path=sample_precursor_edge_index_path,
            sample_precursor_path_index=sample_precursor_path_index,
        )
    
    @staticmethod
    def _concat_atom_index_dicts_with_atom_offset(
        *,
        structures: Sequence["FragmentTreeStructure"],
        field_name: str,
        device: torch.device,
    ) -> Dict[int, Tensor]:
        """
        Concatenate atom index dictionaries with cumulative atom offsets.

        Each dictionary has:
            tuple_length -> [num_rows, tuple_length]

        Atom indices are assumed to be global atom indices inside each structure's
        node_graph. They are shifted by the cumulative atom count of previous
        structures.
        """
        output: Dict[int, list[Tensor]] = {}

        atom_offset = 0

        for structure in structures:
            atom_index_dict = getattr(structure, field_name)

            for tuple_length, atom_idxs in atom_index_dict.items():
                if tuple_length not in output:
                    output[tuple_length] = []

                atom_idxs = atom_idxs.to(device)

                shifted_atom_idxs = torch.where(
                    atom_idxs >= 0,
                    atom_idxs + atom_offset,
                    atom_idxs,
                )

                output[tuple_length].append(shifted_atom_idxs)

            atom_offset += int(structure.node_graph_offset[-1].item())

        return {
            tuple_length: torch.cat(parts, dim=0)
            for tuple_length, parts in output.items()
        }