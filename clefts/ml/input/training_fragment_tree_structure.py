from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from .fragment_tree_structure import FragmentTreeStructure


@dataclass(frozen=True)
class TrainingFragmentTreeStructure(FragmentTreeStructure):
    """FragmentTreeStructure with supervised targets for generation training."""

    target_node_keep: Tensor
    # [N] 1 when the fragment node should remain as an emitted candidate.

    target_node_expand: Tensor
    # [N] 1 when the node should be expanded in the next cleavage step.

    target_ion_index: Tensor
    # [T] Flat ion target indexes aligned with target_node_index.

    target_unsaturation_index: Tensor
    # [T] Flat unsaturation target indexes aligned with target_node_index.

    target_radical_index: Tensor
    # [T] Flat radical target indexes aligned with target_node_index.

    target_node_index: Tensor
    # [T] Node indexes for ion/unsaturation/radical targets.

    target_formula: Tensor
    # [T, F] Target formula tensors used for formula-group coverage losses.

    target_intensity: Tensor
    # [T] Target intensity per target formula row.

    target_formula_group_index: Tensor
    # [T] Formula group id. Rows with the same sample, peak, and group are alternatives.

    target_sample_index: Tensor
    # [T] Sample index aligned with target_formula.

    target_peak_index: Tensor
    # [T] Peak index aligned with target_formula.

    target_edge_index: Tensor
    # [2, U] Target traversal edges: row 0 is sample index, row 1 is edge index.

    target_edge_group_index: Tensor
    # [U] Formula group id aligned with target_edge_index columns.

    def to(self, device: torch.device | str) -> "TrainingFragmentTreeStructure":
        base = super().to(device)
        return TrainingFragmentTreeStructure(
            node_smiles=base.node_smiles,
            node_graph=base.node_graph,
            node_graph_offset=base.node_graph_offset,
            node_formula=base.node_formula,
            formula_element_order=base.formula_element_order,
            edge_index=base.edge_index,
            cleavage_event_edge_index=base.cleavage_event_edge_index,
            cleavage_event=base.cleavage_event,
            cleavage_atom_idxs=base.cleavage_atom_idxs,
            reactant_tuple_length_table=base.reactant_tuple_length_table,
            product_tuple_length_table=base.product_tuple_length_table,
            ion_formula_delta=base.ion_formula_delta,
            unsaturation_formula_delta=base.unsaturation_formula_delta,
            radical_formula_delta=base.radical_formula_delta,
            sample_adduct_type_index=base.sample_adduct_type_index,
            sample_ce_value=base.sample_ce_value,
            sample_edge_index=base.sample_edge_index,
            precursor_edge_index_path=base.precursor_edge_index_path,
            precursor_unsaturation_index=base.precursor_unsaturation_index,
            precursor_radical_index=base.precursor_radical_index,
            precursor_sample_index=base.precursor_sample_index,
            target_node_keep=self.target_node_keep.to(device),
            target_node_expand=self.target_node_expand.to(device),
            target_ion_index=self.target_ion_index.to(device),
            target_unsaturation_index=self.target_unsaturation_index.to(device),
            target_radical_index=self.target_radical_index.to(device),
            target_node_index=self.target_node_index.to(device),
            target_formula=self.target_formula.to(device),
            target_intensity=self.target_intensity.to(device),
            target_formula_group_index=self.target_formula_group_index.to(device),
            target_sample_index=self.target_sample_index.to(device),
            target_peak_index=self.target_peak_index.to(device),
            target_edge_index=self.target_edge_index.to(device),
            target_edge_group_index=self.target_edge_group_index.to(device),
        )
    @classmethod
    def from_structures(
        cls,
        structures: Sequence[FragmentTreeStructure],
        device: Optional[torch.device] = None,
    ) -> FragmentTreeStructure:
        if not all(isinstance(structure, TrainingFragmentTreeStructure) for structure in structures):
            return FragmentTreeStructure.from_structures(structures, device=device)

        training_structures = [
            structure for structure in structures if isinstance(structure, TrainingFragmentTreeStructure)
        ]
        base = FragmentTreeStructure.from_structures(training_structures, device=device)
        target_device = base.device

        node_offsets = []
        edge_offsets = []
        sample_offsets = []
        node_offset = 0
        edge_offset = 0
        sample_offset = 0
        for structure in training_structures:
            node_offsets.append(node_offset)
            edge_offsets.append(edge_offset)
            sample_offsets.append(sample_offset)
            node_offset += int(structure.num_nodes)
            edge_offset += int(structure.num_edges)
            sample_offset += int(structure.num_samples)

        target_node_keep = torch.cat(
            [structure.target_node_keep.to(target_device) for structure in training_structures],
            dim=0,
        )
        target_node_expand = torch.cat(
            [structure.target_node_expand.to(target_device) for structure in training_structures],
            dim=0,
        )

        target_node_index_parts = []
        target_edge_index_parts = []
        target_sample_index_parts = []
        for structure, node_off, edge_off, sample_off in zip(
            training_structures, node_offsets, edge_offsets, sample_offsets
        ):
            target_node_index_parts.append(
                structure.target_node_index.to(target_device) + int(node_off)
            )
            edge_index = structure.target_edge_index.to(target_device).clone()
            if edge_index.numel() > 0:
                edge_index[0, :] += int(sample_off)
                edge_index[1, :] += int(edge_off)
            target_edge_index_parts.append(edge_index)
            target_sample_index_parts.append(
                structure.target_sample_index.to(target_device) + int(sample_off)
            )

        target_node_index = torch.cat(target_node_index_parts, dim=0)
        target_sample_index = torch.cat(target_sample_index_parts, dim=0)
        target_edge_index = torch.cat(target_edge_index_parts, dim=1)

        return cls(
            node_smiles=base.node_smiles,
            node_graph=base.node_graph,
            node_graph_offset=base.node_graph_offset,
            node_formula=base.node_formula,
            formula_element_order=base.formula_element_order,
            edge_index=base.edge_index,
            cleavage_event_edge_index=base.cleavage_event_edge_index,
            cleavage_event=base.cleavage_event,
            cleavage_atom_idxs=base.cleavage_atom_idxs,
            reactant_tuple_length_table=base.reactant_tuple_length_table,
            product_tuple_length_table=base.product_tuple_length_table,
            ion_formula_delta=base.ion_formula_delta,
            unsaturation_formula_delta=base.unsaturation_formula_delta,
            radical_formula_delta=base.radical_formula_delta,
            sample_adduct_type_index=base.sample_adduct_type_index,
            sample_ce_value=base.sample_ce_value,
            sample_edge_index=base.sample_edge_index,
            precursor_edge_index_path=base.precursor_edge_index_path,
            precursor_unsaturation_index=base.precursor_unsaturation_index,
            precursor_radical_index=base.precursor_radical_index,
            precursor_sample_index=base.precursor_sample_index,
            target_node_keep=target_node_keep,
            target_node_expand=target_node_expand,
            target_ion_index=torch.cat(
                [structure.target_ion_index.to(target_device) for structure in training_structures],
                dim=0,
            ),
            target_unsaturation_index=torch.cat(
                [structure.target_unsaturation_index.to(target_device) for structure in training_structures],
                dim=0,
            ),
            target_radical_index=torch.cat(
                [structure.target_radical_index.to(target_device) for structure in training_structures],
                dim=0,
            ),
            target_node_index=target_node_index,
            target_formula=torch.cat(
                [structure.target_formula.to(target_device) for structure in training_structures],
                dim=0,
            ),
            target_intensity=torch.cat(
                [structure.target_intensity.to(target_device) for structure in training_structures],
                dim=0,
            ),
            target_formula_group_index=torch.cat(
                [structure.target_formula_group_index.to(target_device) for structure in training_structures],
                dim=0,
            ),
            target_sample_index=target_sample_index,
            target_peak_index=torch.cat(
                [structure.target_peak_index.to(target_device) for structure in training_structures],
                dim=0,
            ),
            target_edge_index=target_edge_index,
            target_edge_group_index=torch.cat(
                [structure.target_edge_group_index.to(target_device) for structure in training_structures],
                dim=0,
            ),
        )

