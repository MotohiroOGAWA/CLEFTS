from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from .fragment_tree_structure import FragmentTreeStructure


@dataclass(frozen=True)
class TrainingFragmentTreeStructure(FragmentTreeStructure):
    """FragmentTreeStructure with supervised spectrum targets.

    The target hierarchy is:

        sample -> peak -> formula -> terminal node/state -> expand nodes

    Pointer tensors follow the usual CSR convention: values for item ``i`` are
    in ``data[ptr[i]:ptr[i + 1]]``. Empty ranges are valid and represent an
    observed peak/formula with no assigned downstream target.
    """

    sample_peak_mz: Tensor
    # [K] Observed peak m/z values flattened over all samples.

    sample_peak_intensity: Tensor
    # [K] Observed peak intensities aligned with sample_peak_mz.

    sample_peak_ptr: Tensor
    # [S + 1] Peak pointer per sample. Peaks for sample s are
    # sample_peak_mz[sample_peak_ptr[s]:sample_peak_ptr[s + 1]].

    target_formula: Tensor
    # [G, F] Target formula tensors assigned to observed peaks.

    peak_formula_ptr: Tensor
    # [K + 1] Formula pointer per flattened observed peak. Formulas for peak k
    # are target_formula[peak_formula_ptr[k]:peak_formula_ptr[k + 1]].

    target_terminal_node_index: Tensor
    # [A] Terminal fragment node indexes aligned with target ion/state tensors.

    target_ion_index: Tensor
    # [A] Flat ion target indexes aligned with target_terminal_node_index.

    target_unsaturation_index: Tensor
    # [A] Flat unsaturation target indexes aligned with target_terminal_node_index.

    target_radical_index: Tensor
    # [A] Flat radical target indexes aligned with target_terminal_node_index.

    formula_assignment_ptr: Tensor
    # [G + 1] Terminal node/state assignment pointer per target formula.

    target_expand_node_index: Tensor
    # [X] Intermediate/source fragment nodes that should be expanded to reach a
    # terminal assignment.

    terminal_expand_ptr: Tensor
    # [A + 1] Expand-node pointer per terminal node/state assignment.

    target_peak_depth: Tensor
    # [K] Selected minimum post-precursor cleavage depth, or -1 for no target.

    target_path_edge_index: Tensor
    # [P] Ordered global edge IDs for every terminal assignment path.

    terminal_path_ptr: Tensor
    # [A + 1] CSR pointer into target_path_edge_index.

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sample_peak_mz.dim() != 1:
            raise ValueError("sample_peak_mz must be 1D.")
        if self.sample_peak_intensity.dim() != 1:
            raise ValueError("sample_peak_intensity must be 1D.")
        if self.sample_peak_mz.numel() != self.sample_peak_intensity.numel():
            raise ValueError("sample_peak_mz and sample_peak_intensity must have the same length.")
        if self.sample_peak_ptr.dim() != 1 or self.sample_peak_ptr.numel() != self.num_samples + 1:
            raise ValueError("sample_peak_ptr must have shape [num_samples + 1].")
        if self.target_formula.dim() != 2:
            raise ValueError("target_formula must be 2D.")
        if self.target_formula.size(1) != self.node_formula.size(1):
            raise ValueError("target_formula width must match node_formula width.")
        if self.peak_formula_ptr.dim() != 1 or self.peak_formula_ptr.numel() != self.sample_peak_mz.numel() + 1:
            raise ValueError("peak_formula_ptr must have shape [num_peaks + 1].")
        assignment_count = int(self.target_terminal_node_index.numel())
        for name in ("target_ion_index", "target_unsaturation_index", "target_radical_index"):
            value = getattr(self, name)
            if value.dim() != 1 or value.numel() != assignment_count:
                raise ValueError(f"{name} must be 1D and aligned with target_terminal_node_index.")
        if self.formula_assignment_ptr.dim() != 1 or self.formula_assignment_ptr.numel() != self.target_formula.size(0) + 1:
            raise ValueError("formula_assignment_ptr must have shape [num_target_formulas + 1].")
        if self.terminal_expand_ptr.dim() != 1 or self.terminal_expand_ptr.numel() != assignment_count + 1:
            raise ValueError("terminal_expand_ptr must have shape [num_terminal_assignments + 1].")
        if self.target_peak_depth.dim() != 1 or self.target_peak_depth.numel() != self.sample_peak_mz.numel():
            raise ValueError("target_peak_depth must be 1D and aligned with sample peaks.")
        if self.terminal_path_ptr.dim() != 1 or self.terminal_path_ptr.numel() != assignment_count + 1:
            raise ValueError("terminal_path_ptr must have shape [num_terminal_assignments + 1].")
        if self.target_path_edge_index.dim() != 1:
            raise ValueError("target_path_edge_index must be 1D.")
        for name, ptr, expected_end in (
            ("sample_peak_ptr", self.sample_peak_ptr, self.sample_peak_mz.numel()),
            ("peak_formula_ptr", self.peak_formula_ptr, self.target_formula.size(0)),
            ("formula_assignment_ptr", self.formula_assignment_ptr, assignment_count),
            ("terminal_expand_ptr", self.terminal_expand_ptr, self.target_expand_node_index.numel()),
            ("terminal_path_ptr", self.terminal_path_ptr, self.target_path_edge_index.numel()),
        ):
            if int(ptr[0]) != 0 or int(ptr[-1]) != int(expected_end) or bool((ptr[1:] < ptr[:-1]).any()):
                raise ValueError(f"{name} must be a monotone CSR pointer spanning its data tensor.")
        if bool((self.target_peak_depth < -1).any()):
            raise ValueError("target_peak_depth values must be -1 or a non-negative depth.")

    def to(self, device: torch.device | str) -> "TrainingFragmentTreeStructure":
        base = super().to(device)
        return TrainingFragmentTreeStructure(
            node_smiles=base.node_smiles,
            node_graph=base.node_graph,
            node_graph_offset=base.node_graph_offset,
            node_formula=base.node_formula,
            formula_element_order=base.formula_element_order,
            edge_index=base.edge_index,
            tree_sample_ptr=base.tree_sample_ptr,
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
            sample_peak_mz=self.sample_peak_mz.to(device),
            sample_peak_intensity=self.sample_peak_intensity.to(device),
            sample_peak_ptr=self.sample_peak_ptr.to(device),
            target_formula=self.target_formula.to(device),
            peak_formula_ptr=self.peak_formula_ptr.to(device),
            target_terminal_node_index=self.target_terminal_node_index.to(device),
            target_ion_index=self.target_ion_index.to(device),
            target_unsaturation_index=self.target_unsaturation_index.to(device),
            target_radical_index=self.target_radical_index.to(device),
            formula_assignment_ptr=self.formula_assignment_ptr.to(device),
            target_expand_node_index=self.target_expand_node_index.to(device),
            terminal_expand_ptr=self.terminal_expand_ptr.to(device),
            target_peak_depth=self.target_peak_depth.to(device),
            target_path_edge_index=self.target_path_edge_index.to(device),
            terminal_path_ptr=self.terminal_path_ptr.to(device),
        )

    @property
    def target_node_index(self) -> Tensor:
        """Backward-compatible alias for terminal node assignments."""
        return self.target_terminal_node_index

    @property
    def target_assignment_formula(self) -> Tensor:
        """[A, F] Formula tensor expanded to terminal assignment rows."""
        formula_index = self.assignment_formula_index
        if formula_index.numel() == 0:
            return self.target_formula.new_empty((0, self.target_formula.size(1)))
        return self.target_formula[formula_index]

    @property
    def target_sample_index(self) -> Tensor:
        """[A] Sample index expanded to terminal assignment rows."""
        peak_index = self.assignment_peak_index
        peak_sample_index = self.peak_sample_index
        if peak_index.numel() == 0:
            return peak_index
        return peak_sample_index[peak_index]

    @property
    def target_peak_index(self) -> Tensor:
        """[A] Local peak index within each sample for terminal assignments."""
        peak_index = self.assignment_peak_index
        peak_sample_index = self.peak_sample_index
        if peak_index.numel() == 0:
            return peak_index
        sample_starts = self.sample_peak_ptr[peak_sample_index[peak_index]]
        return peak_index - sample_starts

    @property
    def target_intensity(self) -> Tensor:
        """[A] Peak intensity expanded to terminal assignment rows."""
        peak_index = self.assignment_peak_index
        if peak_index.numel() == 0:
            return self.sample_peak_intensity.new_empty((0,))
        return self.sample_peak_intensity[peak_index]

    @property
    def target_formula_group_index(self) -> Tensor:
        """[A] Formula row index used as a compatibility group id."""
        return self.assignment_formula_index

    @property
    def target_node_expand(self) -> Tensor:
        """[N] Compatibility binary mask of nodes that appear in expand paths."""
        out = torch.zeros((self.num_nodes,), dtype=torch.float32, device=self.edge_index.device)
        if self.target_expand_node_index.numel() > 0:
            valid = self.target_expand_node_index.long()
            valid = valid[(valid >= 0) & (valid < self.num_nodes)]
            if valid.numel() > 0:
                out[valid.unique()] = 1.0
        return out

    @property
    def target_node_keep(self) -> Tensor:
        """[N] Compatibility binary mask of terminal target nodes."""
        out = torch.zeros((self.num_nodes,), dtype=torch.float32, device=self.edge_index.device)
        if self.target_terminal_node_index.numel() > 0:
            valid = self.target_terminal_node_index.long()
            valid = valid[(valid >= 0) & (valid < self.num_nodes)]
            if valid.numel() > 0:
                out[valid.unique()] = 1.0
        return out

    @property
    def target_edge_index(self) -> Tensor:
        """[2, E] Candidate edges that can produce a target formula assignment.

        Row 0 is the sample index and row 1 is the global fragmentation-edge
        index.  More than one molecular node (and therefore edge) may be a
        valid explanation for the same formula target.
        """
        if self.target_terminal_node_index.numel() == 0 or self.edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.edge_index.device)
        rows = []
        edge_dst = self.edge_index[1].long()
        path_ptr = self.terminal_path_ptr.long()
        for assignment_id, (sample_id, node_id) in enumerate(zip(
            self.target_sample_index.detach().cpu().tolist(),
            self.target_terminal_node_index.detach().cpu().tolist(),
        )):
            start, end = int(path_ptr[assignment_id]), int(path_ptr[assignment_id + 1])
            if end > start:
                rows.append((int(sample_id), int(self.target_path_edge_index[end - 1])))
            else:  # compatibility for programmatically constructed legacy targets
                for edge_id in (edge_dst == int(node_id)).nonzero(as_tuple=False).view(-1).tolist():
                    rows.append((int(sample_id), int(edge_id)))
        if not rows:
            return torch.empty((2, 0), dtype=torch.long, device=self.edge_index.device)
        return torch.tensor(rows, dtype=torch.long, device=self.edge_index.device).t().contiguous()

    @property
    def target_edge_group_index(self) -> Tensor:
        """[E] Target-formula group aligned with :attr:`target_edge_index`."""
        if self.target_terminal_node_index.numel() == 0 or self.edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.edge_index.device)
        groups = []
        edge_dst = self.edge_index[1].long()
        path_ptr = self.terminal_path_ptr.long()
        for assignment_id, (formula_id, node_id) in enumerate(zip(
            self.assignment_formula_index.detach().cpu().tolist(),
            self.target_terminal_node_index.detach().cpu().tolist(),
        )):
            start, end = int(path_ptr[assignment_id]), int(path_ptr[assignment_id + 1])
            count = 1 if end > start else int((edge_dst == int(node_id)).sum().item())
            groups.extend([int(formula_id)] * count)
        return torch.tensor(groups, dtype=torch.long, device=self.edge_index.device)

    @property
    def peak_sample_index(self) -> Tensor:
        """[K] Sample index for each flattened observed peak."""
        counts = self.sample_peak_ptr[1:] - self.sample_peak_ptr[:-1]
        if counts.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.sample_peak_ptr.device)
        return torch.repeat_interleave(
            torch.arange(counts.numel(), dtype=torch.long, device=self.sample_peak_ptr.device),
            counts.long(),
        )

    @property
    def formula_peak_index(self) -> Tensor:
        """[G] Flattened observed peak index for each target formula."""
        counts = self.peak_formula_ptr[1:] - self.peak_formula_ptr[:-1]
        if counts.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.peak_formula_ptr.device)
        return torch.repeat_interleave(
            torch.arange(counts.numel(), dtype=torch.long, device=self.peak_formula_ptr.device),
            counts.long(),
        )

    @property
    def assignment_formula_index(self) -> Tensor:
        """[A] Target formula row index for each terminal assignment."""
        counts = self.formula_assignment_ptr[1:] - self.formula_assignment_ptr[:-1]
        if counts.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.formula_assignment_ptr.device)
        return torch.repeat_interleave(
            torch.arange(counts.numel(), dtype=torch.long, device=self.formula_assignment_ptr.device),
            counts.long(),
        )

    @property
    def assignment_peak_index(self) -> Tensor:
        """[A] Flattened observed peak index for each terminal assignment."""
        formula_index = self.assignment_formula_index
        formula_peak_index = self.formula_peak_index
        if formula_index.numel() == 0:
            return formula_index
        return formula_peak_index[formula_index]

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
        node_offset = 0
        for structure in training_structures:
            node_offsets.append(node_offset)
            node_offset += int(structure.num_nodes)

        sample_peak_mz = torch.cat(
            [structure.sample_peak_mz.to(target_device) for structure in training_structures],
            dim=0,
        )
        sample_peak_intensity = torch.cat(
            [structure.sample_peak_intensity.to(target_device) for structure in training_structures],
            dim=0,
        )

        sample_peak_ptr_parts = []
        peak_offset = 0
        for structure in training_structures:
            ptr = structure.sample_peak_ptr.to(target_device).long()
            sample_peak_ptr_parts.append(ptr[:-1] + peak_offset)
            peak_offset += int(ptr[-1].item())
        sample_peak_ptr = torch.cat(
            sample_peak_ptr_parts
            + [torch.tensor([peak_offset], dtype=torch.long, device=target_device)],
            dim=0,
        )

        target_formula = torch.cat(
            [structure.target_formula.to(target_device) for structure in training_structures],
            dim=0,
        )

        peak_formula_ptr_parts = []
        formula_offset = 0
        for structure in training_structures:
            ptr = structure.peak_formula_ptr.to(target_device).long()
            peak_formula_ptr_parts.append(ptr[:-1] + formula_offset)
            formula_offset += int(ptr[-1].item())
        peak_formula_ptr = torch.cat(
            peak_formula_ptr_parts
            + [torch.tensor([formula_offset], dtype=torch.long, device=target_device)],
            dim=0,
        )

        target_terminal_node_index = torch.cat(
            [
                structure.target_terminal_node_index.to(target_device).long() + int(node_off)
                for structure, node_off in zip(training_structures, node_offsets)
            ],
            dim=0,
        )
        target_ion_index = torch.cat(
            [structure.target_ion_index.to(target_device) for structure in training_structures],
            dim=0,
        )
        target_unsaturation_index = torch.cat(
            [structure.target_unsaturation_index.to(target_device) for structure in training_structures],
            dim=0,
        )
        target_radical_index = torch.cat(
            [structure.target_radical_index.to(target_device) for structure in training_structures],
            dim=0,
        )

        formula_assignment_ptr_parts = []
        assignment_offset = 0
        for structure in training_structures:
            ptr = structure.formula_assignment_ptr.to(target_device).long()
            formula_assignment_ptr_parts.append(ptr[:-1] + assignment_offset)
            assignment_offset += int(ptr[-1].item())
        formula_assignment_ptr = torch.cat(
            formula_assignment_ptr_parts
            + [torch.tensor([assignment_offset], dtype=torch.long, device=target_device)],
            dim=0,
        )

        target_expand_node_index = torch.cat(
            [
                structure.target_expand_node_index.to(target_device).long() + int(node_off)
                for structure, node_off in zip(training_structures, node_offsets)
            ],
            dim=0,
        )

        terminal_expand_ptr_parts = []
        expand_offset = 0
        for structure in training_structures:
            ptr = structure.terminal_expand_ptr.to(target_device).long()
            terminal_expand_ptr_parts.append(ptr[:-1] + expand_offset)
            expand_offset += int(ptr[-1].item())
        terminal_expand_ptr = torch.cat(
            terminal_expand_ptr_parts
            + [torch.tensor([expand_offset], dtype=torch.long, device=target_device)],
            dim=0,
        )

        target_peak_depth = torch.cat(
            [structure.target_peak_depth.to(target_device) for structure in training_structures]
        )
        edge_offsets = []
        edge_offset = 0
        for structure in training_structures:
            edge_offsets.append(edge_offset)
            edge_offset += structure.num_edges
        target_path_edge_index = torch.cat([
            structure.target_path_edge_index.to(target_device).long() + offset
            for structure, offset in zip(training_structures, edge_offsets)
        ])
        terminal_path_ptr_parts = []
        path_offset = 0
        for structure in training_structures:
            ptr = structure.terminal_path_ptr.to(target_device).long()
            terminal_path_ptr_parts.append(ptr[:-1] + path_offset)
            path_offset += int(ptr[-1])
        terminal_path_ptr = torch.cat(terminal_path_ptr_parts + [torch.tensor([path_offset], device=target_device)])

        return cls(
            node_smiles=base.node_smiles,
            node_graph=base.node_graph,
            node_graph_offset=base.node_graph_offset,
            node_formula=base.node_formula,
            formula_element_order=base.formula_element_order,
            edge_index=base.edge_index,
            tree_sample_ptr=base.tree_sample_ptr,
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
            sample_peak_mz=sample_peak_mz,
            sample_peak_intensity=sample_peak_intensity,
            sample_peak_ptr=sample_peak_ptr,
            target_formula=target_formula,
            peak_formula_ptr=peak_formula_ptr,
            target_terminal_node_index=target_terminal_node_index,
            target_ion_index=target_ion_index,
            target_unsaturation_index=target_unsaturation_index,
            target_radical_index=target_radical_index,
            formula_assignment_ptr=formula_assignment_ptr,
            target_expand_node_index=target_expand_node_index,
            terminal_expand_ptr=terminal_expand_ptr,
            target_peak_depth=target_peak_depth,
            target_path_edge_index=target_path_edge_index,
            terminal_path_ptr=terminal_path_ptr,
        )
