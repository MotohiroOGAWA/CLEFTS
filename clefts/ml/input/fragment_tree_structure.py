from __future__ import annotations

from dataclasses import dataclass, field
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
    #
    # For a fragment node with node_index:
    #     global_atom_index = local_atom_index + node_graph_offset[node_index]
    #
    # The atom index range of node_index is:
    #     [node_graph_offset[node_index], node_graph_offset[node_index + 1])

    node_formula: Tensor
    # [N, F] Neutral fragment formula tensor.

    formula_element_order: Tuple[str, ...]
    # Element order used by node_formula and adduct delta tensors.

    # -------------------------
    # Fragment tree edges
    # -------------------------
    edge_index: Tensor
    # [2, E]
    #
    # edge_index[0, e] = source fragment node index
    # edge_index[1, e] = target fragment node index

    # -------------------------
    # Tree-batch boundaries
    # -------------------------
    tree_sample_ptr: Tensor
    # [T + 1] Prefix-sum sample offsets identifying which contiguous sample
    # range came from which constituent structure ("tree") before
    # from_structures() concatenated them.
    #
    # Samples of tree t are range(tree_sample_ptr[t], tree_sample_ptr[t + 1]).
    # A single (non-collated) structure has tree_sample_ptr == [0, num_samples].

    # -------------------------
    # Cleavage events
    # -------------------------
    cleavage_event_edge_index: Tensor
    # [M]
    #
    # cleavage_event_edge_index[m] = edge index associated with
    # cleavage_event[m].

    cleavage_event: Tensor
    # [M, 5]
    #
    # columns:
    #   0: cleavage_id
    #   1: reaction_id
    #   2: product_molecule_id
    #   3: reactant_atom_row_index
    #   4: product_atom_row_index
    #
    # Important:
    #   cleavage_event[:, 3] and cleavage_event[:, 4] both refer to
    #   cleavage_atom_idxs[tuple_length].
    #
    #   The tuple_length is not stored in cleavage_event.
    #   It must be inferred from:
    #       reactant: (cleavage_id, reaction_id)
    #       product:  (cleavage_id, reaction_id, product_molecule_id)

    cleavage_atom_idxs: Dict[int, Tensor]
    # tuple_length -> [A_len, tuple_length]
    #
    # Shared atom-index tuple table.
    #
    # Both reactant-side and product-side atom tuples are stored here.
    #
    # cleavage_event[:, 3]:
    #     row index for the reactant atom tuple.
    #
    # cleavage_event[:, 4]:
    #     row index for the product atom tuple.
    #
    # The corresponding tuple_length is determined outside this class from
    # cleavage_id / reaction_id / product_molecule_id.

    reactant_tuple_length_table: Tensor
    # [R, 3]
    #
    # columns:
    #   0: cleavage_id
    #   1: reaction_id
    #   2: reactant_tuple_length

    product_tuple_length_table: Tensor
    # [P, 4]
    #
    # columns:
    #   0: cleavage_id
    #   1: reaction_id
    #   2: product_molecule_id
    #   3: product_tuple_length

    ion_formula_delta: Tensor
    # [C_ion, F] Delta tensor aligned with model.ion_flat_candidates.

    unsaturation_formula_delta: Tensor
    # [C_unsaturation, F] Delta tensor aligned with
    # model.unsaturation_flat_candidates.

    radical_formula_delta: Tensor
    # [C_radical, F] Delta tensor aligned with model.radical_flat_candidates.

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

    precursor_edge_index_path: Tensor
    # [P, D]
    #
    # P:
    #     Number of precursor paths.
    #
    # D:
    #     Padded precursor path length.
    #
    # precursor_edge_index_path[p, d] is the edge index at position d
    # in precursor path p.
    #
    # Each edge index refers to:
    #     edge_index[:, edge_index_value]
    #
    # Negative values such as -1 are treated as padding.

    precursor_unsaturation_index: Tensor
    # [P]
    #
    # precursor_unsaturation_index[p] is the unsaturation state index
    # associated with precursor path p.
    #
    # This tensor is aligned with:
    #     precursor_edge_index_path[p]
    #     precursor_radical_index[p]
    #     precursor_sample_index[p]

    precursor_radical_index: Tensor
    # [P]
    #
    # precursor_radical_index[p] is the radical state index
    # associated with precursor path p.
    #
    # This tensor is aligned with:
    #     precursor_edge_index_path[p]
    #     precursor_unsaturation_index[p]
    #     precursor_sample_index[p]

    precursor_sample_index: Tensor
    # [P]
    #
    # precursor_sample_index[p] is the sample index associated with
    # precursor path p.
    #
    # This tensor is aligned with:
    #     precursor_edge_index_path[p]
    #     precursor_unsaturation_index[p]
    #     precursor_radical_index[p]

    def __post_init__(self) -> None:
        num_precursor_paths = int(self.precursor_edge_index_path.size(0))

        for ptr_name, ptr, expected_end in (
            ("tree_sample_ptr", self.tree_sample_ptr, self.num_samples),
        ):
            if ptr.dim() != 1 or ptr.numel() < 2:
                raise ValueError(
                    f"{ptr_name} must be 1D with at least 2 entries, "
                    f"got shape {tuple(ptr.shape)}."
                )
            if int(ptr[0].item()) != 0 or int(ptr[-1].item()) != int(expected_end):
                raise ValueError(
                    f"{ptr_name} must span [0, {expected_end}], "
                    f"got [{int(ptr[0].item())}, {int(ptr[-1].item())}]."
                )
            if bool((ptr[1:] < ptr[:-1]).any().item()):
                raise ValueError(f"{ptr_name} must be non-decreasing.")

        if self.node_formula.dim() != 2:
            raise ValueError(
                "node_formula must be 2D, "
                f"got shape {tuple(self.node_formula.shape)}."
            )

        if self.node_formula.size(0) != self.num_nodes:
            raise ValueError(
                "node_formula must have one row per node. "
                f"Got {self.node_formula.size(0)} rows and "
                f"{self.num_nodes} nodes."
            )

        if self.node_formula.size(1) != len(self.formula_element_order) + 1:
            raise ValueError(
                "node_formula width must equal len(formula_element_order) + 1 "
                "for the charge column. "
                f"Got width={self.node_formula.size(1)} and "
                f"element_order={self.formula_element_order}."
            )

        for field_name in (
            "ion_formula_delta",
            "unsaturation_formula_delta",
            "radical_formula_delta",
        ):
            value = getattr(self, field_name)
            if value.dim() != 2:
                raise ValueError(
                    f"{field_name} must be 2D, "
                    f"got shape {tuple(value.shape)}."
                )
            if value.size(1) != self.node_formula.size(1):
                raise ValueError(
                    f"{field_name} must have formula width "
                    f"{self.node_formula.size(1)}, got {value.size(1)}."
                )

        if self.precursor_edge_index_path.dim() != 2:
            raise ValueError(
                "precursor_edge_index_path must be 2D, "
                f"got shape {tuple(self.precursor_edge_index_path.shape)}."
            )

        if self.precursor_unsaturation_index.dim() != 1:
            raise ValueError(
                "precursor_unsaturation_index must be 1D, "
                f"got shape {tuple(self.precursor_unsaturation_index.shape)}."
            )

        if self.precursor_radical_index.dim() != 1:
            raise ValueError(
                "precursor_radical_index must be 1D, "
                f"got shape {tuple(self.precursor_radical_index.shape)}."
            )

        if self.precursor_sample_index.dim() != 1:
            raise ValueError(
                "precursor_sample_index must be 1D, "
                f"got shape {tuple(self.precursor_sample_index.shape)}."
            )

        if self.precursor_unsaturation_index.numel() != num_precursor_paths:
            raise ValueError(
                "precursor_unsaturation_index must have the same length as "
                "precursor_edge_index_path rows. "
                f"Got {self.precursor_unsaturation_index.numel()} and "
                f"{num_precursor_paths}."
            )

        if self.precursor_radical_index.numel() != num_precursor_paths:
            raise ValueError(
                "precursor_radical_index must have the same length as "
                "precursor_edge_index_path rows. "
                f"Got {self.precursor_radical_index.numel()} and "
                f"{num_precursor_paths}."
            )

        if self.precursor_sample_index.numel() != num_precursor_paths:
            raise ValueError(
                "precursor_sample_index must have the same length as "
                "precursor_edge_index_path rows. "
                f"Got {self.precursor_sample_index.numel()} and "
                f"{num_precursor_paths}."
            )

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
    def num_trees(self) -> int:
        return int(self.tree_sample_ptr.numel()) - 1

    @property
    def cleavage_pattern_ids(self) -> Tensor:
        return self.cleavage_event[:, 0]  # [M]

    @property
    def cleavage_reaction_ids(self) -> Tensor:
        return self.cleavage_event[:, 1]  # [M]

    @property
    def cleavage_product_molecule_ids(self) -> Tensor:
        return self.cleavage_event[:, 2]  # [M]

    @property
    def cleavage_reactant_row_indices(self) -> Tensor:
        return self.cleavage_event[:, 3]  # [M]

    @property
    def cleavage_product_row_indices(self) -> Tensor:
        return self.cleavage_event[:, 4]  # [M]

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
            node_formula=self.node_formula.to(device),
            formula_element_order=self.formula_element_order,
            edge_index=self.edge_index.to(device),
            tree_sample_ptr=self.tree_sample_ptr.to(device),
            cleavage_event_edge_index=self.cleavage_event_edge_index.to(device),
            cleavage_event=self.cleavage_event.to(device),
            cleavage_atom_idxs={
                int(tuple_length): atom_idxs.to(device)
                for tuple_length, atom_idxs in self.cleavage_atom_idxs.items()
            },
            sample_adduct_type_index=self.sample_adduct_type_index.to(device),
            sample_ce_value=self.sample_ce_value.to(device),
            sample_edge_index=self.sample_edge_index.to(device),
            precursor_edge_index_path=(
                self.precursor_edge_index_path.to(device)
            ),
            precursor_unsaturation_index=(
                self.precursor_unsaturation_index.to(device)
            ),
            precursor_radical_index=(
                self.precursor_radical_index.to(device)
            ),
            precursor_sample_index=(
                self.precursor_sample_index.to(device)
            ),
            reactant_tuple_length_table=(
                self.reactant_tuple_length_table.to(device)
            ),
            product_tuple_length_table=(
                self.product_tuple_length_table.to(device)
            ),
            ion_formula_delta=self.ion_formula_delta.to(device),
            unsaturation_formula_delta=self.unsaturation_formula_delta.to(device),
            radical_formula_delta=self.radical_formula_delta.to(device),
        )
        
    @classmethod
    def from_structures(
        cls,
        structures: Sequence["FragmentTreeStructure"],
        device: Optional[torch.device] = None,
    ) -> "FragmentTreeStructure":
        """
        Combine multiple FragmentTreeStructure objects into one batched structure.

        Notes
        -----
        This method keeps cleavage_atom_idxs as local atom indices.

        Therefore:
            - cleavage_atom_idxs values are NOT shifted by atom offsets.
            - cleavage_atom_idxs are merged uniquely by their values.
            - cleavage_event[:, 3] and cleavage_event[:, 4] are remapped to
              the merged cleavage_atom_idxs rows.

        Offset handling:
            - node indices are shifted by cumulative num_nodes.
            - edge indices are shifted by cumulative num_edges.
            - sample indices are shifted by cumulative num_samples.
            - precursor edge paths are shifted by cumulative num_edges.
            - cleavage atom tuples are not shifted.
        """
        if len(structures) == 0:
            raise ValueError("structures must not be empty.")

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
            node_graph_data_list.extend(
                structure.node_graph.to_data_list()
            )

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

            if offset.dim() != 1:
                raise ValueError(
                    "node_graph_offset must be 1D, "
                    f"got shape {tuple(offset.shape)}."
                )

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
        # Formula tensors
        # -------------------------
        formula_element_order = tuple(first.formula_element_order)

        for structure in structures:
            if tuple(structure.formula_element_order) != formula_element_order:
                raise ValueError(
                    "All FragmentTreeStructure objects must use the same "
                    "formula_element_order. "
                    f"Got {formula_element_order} and "
                    f"{tuple(structure.formula_element_order)}."
                )

        node_formula = torch.cat(
            [
                structure.node_formula.to(tensor_device)
                for structure in structures
            ],
            dim=0,
        )

        ion_formula_delta = first.ion_formula_delta.to(tensor_device)
        unsaturation_formula_delta = first.unsaturation_formula_delta.to(
            tensor_device
        )
        radical_formula_delta = first.radical_formula_delta.to(tensor_device)

        for structure in structures[1:]:
            for field_name, first_value in (
                ("ion_formula_delta", ion_formula_delta),
                ("unsaturation_formula_delta", unsaturation_formula_delta),
                ("radical_formula_delta", radical_formula_delta),
            ):
                value = getattr(structure, field_name).to(tensor_device)
                if tuple(value.shape) != tuple(first_value.shape) or not torch.equal(
                    value,
                    first_value,
                ):
                    raise ValueError(
                        f"All structures must share identical {field_name}."
                    )

        # -------------------------
        # Fragment tree edge_index
        # -------------------------
        edge_index_parts = []
        node_offset = 0

        for structure in structures:
            edge_index = structure.edge_index.to(tensor_device)

            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(
                    "edge_index must have shape [2, E], "
                    f"got shape {tuple(edge_index.shape)}."
                )

            shifted_edge_index = edge_index.clone()

            if shifted_edge_index.numel() > 0:
                shifted_edge_index = shifted_edge_index + node_offset

            edge_index_parts.append(shifted_edge_index)
            node_offset += structure.num_nodes

        edge_index = torch.cat(edge_index_parts, dim=1)

        # -------------------------
        # cleavage_event_edge_index
        # -------------------------
        cleavage_event_edge_index_parts = []
        edge_offset = 0

        for structure in structures:
            event_edge_index = structure.cleavage_event_edge_index.to(
                tensor_device
            ).clone()

            if event_edge_index.dim() != 1:
                raise ValueError(
                    "cleavage_event_edge_index must be 1D, "
                    f"got shape {tuple(event_edge_index.shape)}."
                )

            if event_edge_index.numel() > 0:
                event_edge_index = event_edge_index + edge_offset

            cleavage_event_edge_index_parts.append(event_edge_index)
            edge_offset += structure.num_edges

        cleavage_event_edge_index = torch.cat(
            cleavage_event_edge_index_parts,
            dim=0,
        )

        # -------------------------
        # tuple-length tables
        # -------------------------
        reactant_tuple_length_table = cls._merge_tuple_length_tables(
            structures=structures,
            field_name="reactant_tuple_length_table",
            width=3,
            key_width=2,
            device=tensor_device,
        )

        product_tuple_length_table = cls._merge_tuple_length_tables(
            structures=structures,
            field_name="product_tuple_length_table",
            width=4,
            key_width=3,
            device=tensor_device,
        )

        # -------------------------
        # cleavage atom index dictionary
        # -------------------------
        (
            cleavage_atom_idxs,
            atom_row_index_remaps,
        ) = cls._merge_unique_atom_index_dicts(
            structures=structures,
            field_name="cleavage_atom_idxs",
            device=tensor_device,
        )

        # -------------------------
        # cleavage_event
        # -------------------------
        cleavage_event = cls._concat_cleavage_events_with_atom_row_remap(
            structures=structures,
            device=tensor_device,
            atom_row_index_remaps=atom_row_index_remaps,
            reactant_tuple_length_table=reactant_tuple_length_table,
            product_tuple_length_table=product_tuple_length_table,
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
            sample_edge_index = structure.sample_edge_index.to(
                tensor_device
            ).clone()

            if sample_edge_index.dim() != 2 or sample_edge_index.size(0) != 2:
                raise ValueError(
                    "sample_edge_index must have shape [2, L], "
                    f"got shape {tuple(sample_edge_index.shape)}."
                )

            if sample_edge_index.numel() > 0:
                sample_edge_index[0, :] += sample_offset
                sample_edge_index[1, :] += edge_offset

            sample_edge_index_parts.append(sample_edge_index)

            sample_offset += structure.num_samples
            edge_offset += structure.num_edges

        sample_edge_index = torch.cat(
            sample_edge_index_parts,
            dim=1,
        )

        # -------------------------
        # precursor_edge_index_path
        # -------------------------
        precursor_edge_index_path = (
            cls._concat_precursor_edge_index_paths(
                structures=structures,
                device=tensor_device,
            )
        )

        # -------------------------
        # precursor_unsaturation_index
        # -------------------------
        precursor_unsaturation_index = torch.cat(
            [
                structure.precursor_unsaturation_index.to(
                    tensor_device
                ).long()
                for structure in structures
            ],
            dim=0,
        )

        # -------------------------
        # precursor_radical_index
        # -------------------------
        precursor_radical_index = torch.cat(
            [
                structure.precursor_radical_index.to(
                    tensor_device
                ).long()
                for structure in structures
            ],
            dim=0,
        )

        # -------------------------
        # precursor_sample_index
        # -------------------------
        precursor_sample_index_parts = []

        sample_offset = 0

        for structure in structures:
            precursor_sample_index_part = structure.precursor_sample_index.to(
                tensor_device
            ).clone()

            if precursor_sample_index_part.dim() != 1:
                raise ValueError(
                    "precursor_sample_index must be 1D, "
                    f"got shape {tuple(precursor_sample_index_part.shape)}."
                )

            shifted_precursor_sample_index = torch.where(
                precursor_sample_index_part >= 0,
                precursor_sample_index_part + sample_offset,
                precursor_sample_index_part,
            )

            precursor_sample_index_parts.append(shifted_precursor_sample_index)
            sample_offset += structure.num_samples

        precursor_sample_index = torch.cat(
            precursor_sample_index_parts,
            dim=0,
        )

        # -------------------------
        # tree_sample_ptr
        # -------------------------
        tree_sample_ptr_values = [0]
        tree_sample_offset = 0
        for structure in structures:
            tree_sample_offset += structure.num_samples
            tree_sample_ptr_values.append(tree_sample_offset)
        tree_sample_ptr = torch.tensor(
            tree_sample_ptr_values, dtype=torch.long, device=tensor_device
        )

        return cls(
            node_smiles=node_smiles,
            node_graph=node_graph,
            node_graph_offset=node_graph_offset,
            node_formula=node_formula,
            formula_element_order=formula_element_order,
            edge_index=edge_index,
            tree_sample_ptr=tree_sample_ptr,
            cleavage_event_edge_index=cleavage_event_edge_index,
            cleavage_event=cleavage_event,
            cleavage_atom_idxs=cleavage_atom_idxs,
            sample_adduct_type_index=sample_adduct_type_index,
            sample_ce_value=sample_ce_value,
            sample_edge_index=sample_edge_index,
            precursor_edge_index_path=precursor_edge_index_path,
            precursor_unsaturation_index=precursor_unsaturation_index,
            precursor_radical_index=precursor_radical_index,
            precursor_sample_index=precursor_sample_index,
            reactant_tuple_length_table=reactant_tuple_length_table,
            product_tuple_length_table=product_tuple_length_table,
            ion_formula_delta=ion_formula_delta,
            unsaturation_formula_delta=unsaturation_formula_delta,
            radical_formula_delta=radical_formula_delta,
        )

    @staticmethod
    def _merge_tuple_length_tables(
        *,
        structures: Sequence["FragmentTreeStructure"],
        field_name: str,
        width: int,
        key_width: int,
        device: torch.device,
    ) -> Tensor:
        """
        Merge tuple-length tables and keep unique keys.

        Parameters
        ----------
        structures:
            FragmentTreeStructure objects.

        field_name:
            Name of the tuple-length table field.

            Expected values:
                - "reactant_tuple_length_table"
                - "product_tuple_length_table"

        width:
            Number of columns.

            reactant_tuple_length_table:
                3 columns:
                    cleavage_id,
                    reaction_id,
                    reactant_tuple_length

            product_tuple_length_table:
                4 columns:
                    cleavage_id,
                    reaction_id,
                    product_molecule_id,
                    product_tuple_length

        key_width:
            Number of columns used as key.

            reactant:
                2 columns:
                    cleavage_id,
                    reaction_id

            product:
                3 columns:
                    cleavage_id,
                    reaction_id,
                    product_molecule_id

        device:
            Target device.

        Returns
        -------
        Tensor
            Merged tuple-length table.
        """
        row_by_key: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

        for structure in structures:
            table = getattr(structure, field_name).to(device).long()

            if table.numel() == 0:
                continue

            if table.dim() != 2 or table.size(1) != width:
                raise ValueError(
                    f"{field_name} must have shape [N, {width}], "
                    f"got shape {tuple(table.shape)}."
                )

            for row in table.tolist():
                row_tuple = tuple(int(value) for value in row)
                key = row_tuple[:key_width]

                if key in row_by_key:
                    previous = row_by_key[key]

                    if previous != row_tuple:
                        raise ValueError(
                            f"Inconsistent {field_name} for key={key}: "
                            f"{previous} vs {row_tuple}."
                        )
                else:
                    row_by_key[key] = row_tuple

        rows = list(row_by_key.values())
        rows.sort()

        if len(rows) == 0:
            return torch.empty(
                (0, width),
                dtype=torch.long,
                device=device,
            )

        return torch.tensor(
            rows,
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _merge_unique_atom_index_dicts(
        *,
        structures: Sequence["FragmentTreeStructure"],
        field_name: str,
        device: torch.device,
    ) -> Tuple[Dict[int, Tensor], list[Dict[int, Tensor]]]:
        """
        Merge atom-index dictionaries and make rows unique.

        Important
        ---------
        Atom indices in cleavage_atom_idxs are local atom indices within each
        fragment SMILES. Therefore, atom offsets must NOT be added here.

        Parameters
        ----------
        structures:
            FragmentTreeStructure objects.

        field_name:
            Usually "cleavage_atom_idxs".

        device:
            Target device.

        Returns
        -------
        merged_atom_idxs:
            Dict[tuple_length, Tensor]

            tuple_length -> [A_unique, tuple_length]

        atom_row_index_remaps:
            list[Dict[int, Tensor]]

            atom_row_index_remaps[structure_index][tuple_length][old_row_index]
            gives new_row_index in merged_atom_idxs[tuple_length].
        """
        rows_by_tuple_length: Dict[int, list[Tuple[int, ...]]] = {}
        row_index_by_tuple_length_and_atom_idxs: Dict[
            int,
            Dict[Tuple[int, ...], int],
        ] = {}

        atom_row_index_remaps: list[Dict[int, Tensor]] = []

        for structure in structures:
            atom_index_dict = getattr(structure, field_name)

            structure_remap: Dict[int, Tensor] = {}

            for tuple_length, atom_idxs in atom_index_dict.items():
                tuple_length = int(tuple_length)
                atom_idxs = atom_idxs.to(device).long()

                if atom_idxs.numel() == 0:
                    structure_remap[tuple_length] = torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=device,
                    )
                    continue

                if atom_idxs.dim() != 2:
                    raise ValueError(
                        f"{field_name}[{tuple_length}] must be 2D, "
                        f"got shape {tuple(atom_idxs.shape)}."
                    )

                if atom_idxs.size(1) != tuple_length:
                    raise ValueError(
                        f"{field_name}[{tuple_length}] has invalid width: "
                        f"got {atom_idxs.size(1)}, expected {tuple_length}."
                    )

                if tuple_length not in rows_by_tuple_length:
                    rows_by_tuple_length[tuple_length] = []

                if tuple_length not in row_index_by_tuple_length_and_atom_idxs:
                    row_index_by_tuple_length_and_atom_idxs[tuple_length] = {}

                row_index_by_atom_idxs = (
                    row_index_by_tuple_length_and_atom_idxs[tuple_length]
                )

                old_to_new = torch.empty(
                    (atom_idxs.size(0),),
                    dtype=torch.long,
                    device=device,
                )

                for old_row_index in range(atom_idxs.size(0)):
                    row = atom_idxs[old_row_index]

                    atom_idx_tuple = tuple(
                        int(value)
                        for value in row.tolist()
                    )

                    if atom_idx_tuple in row_index_by_atom_idxs:
                        new_row_index = row_index_by_atom_idxs[atom_idx_tuple]
                    else:
                        new_row_index = len(rows_by_tuple_length[tuple_length])
                        rows_by_tuple_length[tuple_length].append(atom_idx_tuple)
                        row_index_by_atom_idxs[atom_idx_tuple] = new_row_index

                    old_to_new[old_row_index] = int(new_row_index)

                structure_remap[tuple_length] = old_to_new

            atom_row_index_remaps.append(structure_remap)

        merged_atom_idxs: Dict[int, Tensor] = {}

        for tuple_length, rows in rows_by_tuple_length.items():
            tuple_length = int(tuple_length)

            if len(rows) == 0:
                merged_atom_idxs[tuple_length] = torch.empty(
                    (0, tuple_length),
                    dtype=torch.long,
                    device=device,
                )
            else:
                merged_atom_idxs[tuple_length] = torch.tensor(
                    rows,
                    dtype=torch.long,
                    device=device,
                )

        return merged_atom_idxs, atom_row_index_remaps

    @staticmethod
    def _concat_cleavage_events_with_atom_row_remap(
        *,
        structures: Sequence["FragmentTreeStructure"],
        device: torch.device,
        atom_row_index_remaps: Sequence[Dict[int, Tensor]],
        reactant_tuple_length_table: Tensor,
        product_tuple_length_table: Tensor,
    ) -> Tensor:
        """
        Concatenate cleavage_event tensors and remap atom tuple row indices.

        cleavage_event keeps shape [M, 5].

        columns:
            0: cleavage_id
            1: reaction_id
            2: product_molecule_id
            3: reactant_atom_row_index
            4: product_atom_row_index
        """
        if len(structures) != len(atom_row_index_remaps):
            raise ValueError(
                "structures and atom_row_index_remaps must have the same length. "
                f"Got {len(structures)} and {len(atom_row_index_remaps)}."
            )

        has_cleavage_events = any(
            structure.cleavage_event.numel() > 0
            for structure in structures
        )

        if not has_cleavage_events:
            return torch.empty(
                (0, 5),
                dtype=torch.long,
                device=device,
            )

        reactant_tuple_length_by_key = (
            FragmentTreeStructure._tuple_length_table_to_dict(
                table=reactant_tuple_length_table,
                width=3,
                key_width=2,
                name="reactant_tuple_length_table",
            )
        )

        product_tuple_length_by_key = (
            FragmentTreeStructure._tuple_length_table_to_dict(
                table=product_tuple_length_table,
                width=4,
                key_width=3,
                name="product_tuple_length_table",
            )
        )

        cleavage_event_parts: list[Tensor] = []

        for structure_index, structure in enumerate(structures):
            event = structure.cleavage_event.to(device).clone()

            if event.numel() == 0:
                cleavage_event_parts.append(event)
                continue

            if event.dim() != 2 or event.size(1) != 5:
                raise ValueError(
                    "cleavage_event must have shape [M, 5], "
                    f"got shape {tuple(event.shape)}."
                )

            atom_row_index_remap = atom_row_index_remaps[structure_index]

            for event_index in range(event.size(0)):
                cleavage_id = int(event[event_index, 0].item())
                reaction_id = int(event[event_index, 1].item())
                product_molecule_id = int(event[event_index, 2].item())

                reactant_key = (
                    cleavage_id,
                    reaction_id,
                )

                product_key = (
                    cleavage_id,
                    reaction_id,
                    product_molecule_id,
                )

                if reactant_key not in reactant_tuple_length_by_key:
                    raise KeyError(
                        "reactant_tuple_length_table does not contain "
                        f"key={reactant_key}."
                    )

                if product_key not in product_tuple_length_by_key:
                    raise KeyError(
                        "product_tuple_length_table does not contain "
                        f"key={product_key}."
                    )

                reactant_tuple_length = int(
                    reactant_tuple_length_by_key[reactant_key]
                )

                product_tuple_length = int(
                    product_tuple_length_by_key[product_key]
                )

                old_reactant_row_index = int(event[event_index, 3].item())
                old_product_row_index = int(event[event_index, 4].item())

                new_reactant_row_index = (
                    FragmentTreeStructure._map_atom_row_index(
                        atom_row_index_remap=atom_row_index_remap,
                        tuple_length=reactant_tuple_length,
                        old_row_index=old_reactant_row_index,
                        name="reactant_atom_row_index",
                    )
                )

                new_product_row_index = (
                    FragmentTreeStructure._map_atom_row_index(
                        atom_row_index_remap=atom_row_index_remap,
                        tuple_length=product_tuple_length,
                        old_row_index=old_product_row_index,
                        name="product_atom_row_index",
                    )
                )

                event[event_index, 3] = new_reactant_row_index
                event[event_index, 4] = new_product_row_index

            cleavage_event_parts.append(event)

        return torch.cat(
            cleavage_event_parts,
            dim=0,
        )

    @staticmethod
    def _tuple_length_table_to_dict(
        *,
        table: Tensor,
        width: int,
        key_width: int,
        name: str,
    ) -> Dict[Tuple[int, ...], int]:
        """
        Convert a tuple-length table tensor to a Python lookup dict.

        The last column is treated as tuple_length.
        """
        table = table.long()

        if table.numel() == 0:
            return {}

        if table.dim() != 2 or table.size(1) != width:
            raise ValueError(
                f"{name} must have shape [N, {width}], "
                f"got shape {tuple(table.shape)}."
            )

        output: Dict[Tuple[int, ...], int] = {}

        for row in table.tolist():
            row_tuple = tuple(int(value) for value in row)

            key = row_tuple[:key_width]
            tuple_length = int(row_tuple[-1])

            if key in output:
                previous = output[key]

                if previous != tuple_length:
                    raise ValueError(
                        f"Inconsistent tuple length in {name} for key={key}: "
                        f"{previous} vs {tuple_length}."
                    )
            else:
                output[key] = tuple_length

        return output

    @staticmethod
    def _map_atom_row_index(
        *,
        atom_row_index_remap: Dict[int, Tensor],
        tuple_length: int,
        old_row_index: int,
        name: str,
    ) -> int:
        """Map an old atom tuple row index to a new merged row index."""
        tuple_length = int(tuple_length)
        old_row_index = int(old_row_index)

        if tuple_length not in atom_row_index_remap:
            raise KeyError(
                f"{name} requires tuple_length={tuple_length}, "
                "but atom_row_index_remap does not contain that key. "
                f"Available tuple lengths: "
                f"{sorted(atom_row_index_remap.keys())}."
            )

        remap = atom_row_index_remap[tuple_length]

        if old_row_index < 0:
            raise IndexError(
                f"{name} contains negative row index: {old_row_index}."
            )

        if old_row_index >= int(remap.numel()):
            raise IndexError(
                f"{name} is out of range for tuple_length={tuple_length}: "
                f"old_row_index={old_row_index}, remap_size={remap.numel()}."
            )

        return int(remap[old_row_index].item())

    @staticmethod
    def _concat_precursor_edge_index_paths(
        *,
        structures: Sequence["FragmentTreeStructure"],
        device: torch.device,
    ) -> Tensor:
        """
        Concatenate precursor_edge_index_path with edge offsets.

        Negative padding values such as -1 are kept unchanged.

        This method also handles the case where some structures have no
        precursor paths.
        """
        max_path_width = 0

        for structure in structures:
            path = structure.precursor_edge_index_path

            if path.dim() != 2:
                raise ValueError(
                    "precursor_edge_index_path must be 2D, "
                    f"got shape {tuple(path.shape)}."
                )

            max_path_width = max(
                max_path_width,
                int(path.size(1)),
            )

        path_parts = []

        edge_offset = 0

        for structure in structures:
            path = structure.precursor_edge_index_path.to(device).clone()

            if path.size(1) < max_path_width:
                pad_width = max_path_width - int(path.size(1))

                padding = torch.full(
                    (path.size(0), pad_width),
                    -1,
                    dtype=path.dtype,
                    device=device,
                )

                path = torch.cat(
                    [path, padding],
                    dim=1,
                )

            shifted_path = torch.where(
                path >= 0,
                path + edge_offset,
                path,
            )

            path_parts.append(shifted_path)
            edge_offset += structure.num_edges

        if len(path_parts) == 0:
            return torch.empty(
                (0, 0),
                dtype=torch.long,
                device=device,
            )

        return torch.cat(
            path_parts,
            dim=0,
        )