from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .single_fragment_tree_structure_builder import SingleFragmentTreeStructureBuilder
from .training_fragment_tree_structure import TrainingFragmentTreeStructure


@dataclass
class SupervisedFragmentTreeStructureBuilder(SingleFragmentTreeStructureBuilder):
    """SingleFragmentTreeStructureBuilder that emits target-aware structures."""

    def to_training_structure(
        self,
        *,
        target_node_keep: Optional[Tensor] = None,
        target_node_expand: Optional[Tensor] = None,
        target_node_index: Optional[Tensor] = None,
        target_ion_index: Optional[Tensor] = None,
        target_unsaturation_index: Optional[Tensor] = None,
        target_radical_index: Optional[Tensor] = None,
        target_formula: Optional[Tensor] = None,
        target_intensity: Optional[Tensor] = None,
    ) -> TrainingFragmentTreeStructure:
        structure = self.to_structure()
        device = structure.device
        formula_dim = int(structure.node_formula.size(1))

        if target_node_keep is None:
            target_node_keep = torch.zeros(
                (structure.num_nodes,),
                dtype=torch.float32,
                device=device,
            )
        if target_node_expand is None:
            target_node_expand = torch.zeros(
                (structure.num_nodes,),
                dtype=torch.float32,
                device=device,
            )
        if target_node_index is None:
            target_node_index = torch.empty((0,), dtype=torch.long, device=device)
        if target_ion_index is None:
            target_ion_index = torch.empty((0,), dtype=torch.long, device=device)
        if target_unsaturation_index is None:
            target_unsaturation_index = torch.empty((0,), dtype=torch.long, device=device)
        if target_radical_index is None:
            target_radical_index = torch.empty((0,), dtype=torch.long, device=device)
        if target_formula is None:
            target_formula = torch.empty((0, formula_dim), dtype=torch.float32, device=device)
        if target_intensity is None:
            target_intensity = torch.empty((0,), dtype=torch.float32, device=device)
        target_formula_group_index = torch.empty((0,), dtype=torch.long, device=device)
        target_sample_index = torch.empty((0,), dtype=torch.long, device=device)
        target_peak_index = torch.empty((0,), dtype=torch.long, device=device)
        target_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        target_edge_group_index = torch.empty((0,), dtype=torch.long, device=device)

        return TrainingFragmentTreeStructure(
            node_smiles=structure.node_smiles,
            node_graph=structure.node_graph,
            node_graph_offset=structure.node_graph_offset,
            node_formula=structure.node_formula,
            formula_element_order=structure.formula_element_order,
            edge_index=structure.edge_index,
            cleavage_event_edge_index=structure.cleavage_event_edge_index,
            cleavage_event=structure.cleavage_event,
            cleavage_atom_idxs=structure.cleavage_atom_idxs,
            reactant_tuple_length_table=structure.reactant_tuple_length_table,
            product_tuple_length_table=structure.product_tuple_length_table,
            ion_formula_delta=structure.ion_formula_delta,
            unsaturation_formula_delta=structure.unsaturation_formula_delta,
            radical_formula_delta=structure.radical_formula_delta,
            sample_adduct_type_index=structure.sample_adduct_type_index,
            sample_ce_value=structure.sample_ce_value,
            sample_edge_index=structure.sample_edge_index,
            precursor_edge_index_path=structure.precursor_edge_index_path,
            precursor_unsaturation_index=structure.precursor_unsaturation_index,
            precursor_radical_index=structure.precursor_radical_index,
            precursor_sample_index=structure.precursor_sample_index,
            target_node_keep=target_node_keep.to(device),
            target_node_expand=target_node_expand.to(device),
            target_ion_index=target_ion_index.to(device),
            target_unsaturation_index=target_unsaturation_index.to(device),
            target_radical_index=target_radical_index.to(device),
            target_node_index=target_node_index.to(device),
            target_formula=target_formula.to(device),
            target_intensity=target_intensity.to(device),
            target_formula_group_index=target_formula_group_index,
            target_sample_index=target_sample_index,
            target_peak_index=target_peak_index,
            target_edge_index=target_edge_index,
            target_edge_group_index=target_edge_group_index,
        )
