from __future__ import annotations

import sys

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from ...domain.fragment.ion_tree import FragmentIonTree
from ...domain.fragment.pathway import FragmentPathwayEdge, FragmentPathwayGroup
from ...libs.mmkit.mmkit import Adduct, Compound
from ...libs.msentity.msentity import MSDataset
from .single_fragment_tree_structure_builder import (
    FragmentTreeSample,
    SingleFragmentTreeStructureBuilder,
)
from .training_fragment_tree_structure import TrainingFragmentTreeStructure


@dataclass
class TrainingFormulaTarget:
    node_index: int
    peak_index: int
    group_index: int
    ion_index: int
    unsaturation_index: int
    radical_index: int
    formula_tensor: Tensor
    intensity: float
    expand_node_indexes: Tuple[int, ...] = ()


@dataclass
class TrainingEdgeTarget:
    edge_index: int
    group_index: int


@dataclass
class TrainingFragmentTreeSample(FragmentTreeSample):
    """One training sample with target spectrum information."""

    peak_mz: List[float] = field(default_factory=list)
    # [K] Peak m/z values.

    peak_intensity: List[float] = field(default_factory=list)
    # [K] Peak intensity values.

    target_edge_indexes: Set[int] = field(default_factory=set)
    # Correct traversal edges for assigned peak pathways.

    target_edge_rows: List[TrainingEdgeTarget] = field(default_factory=list)
    # Correct traversal edges aligned with formula group ids.

    target_node_keep_indexes: Set[int] = field(default_factory=set)
    # Terminal fragment nodes that should be kept as emitted candidates.

    target_node_expand_indexes: Set[int] = field(default_factory=set)
    # Intermediate fragment nodes that should be expanded further.

    target_formula_rows: List[TrainingFormulaTarget] = field(default_factory=list)
    # Peak/formula/intensity targets aligned with terminal fragments.


@dataclass(frozen=True)
class TrainingSampleInputRow:
    adduct_type_index: int
    ce_value: float
    peak_mz: Tuple[float, ...]
    peak_intensity: Tuple[float, ...]
    source_label: str


@dataclass
class TrainingFragmentTreeStructureBuilder(SingleFragmentTreeStructureBuilder):
    def add_training_sample(
        self,
        dataset: MSDataset,
        *,
        precursor_mz_column: str,
        adduct_type_column: str,
        collision_energy_column: str,
        smiles_column: str = "SMILES",
        instrument_column: Optional[str] = None,
        max_node: int = -1,
        max_edge: int = -1,
        max_depth: Optional[int] = None,
    ) -> np.ndarray:
        """Add same-SMILES MSDataset records as training samples."""
        if len(dataset) == 0:
            raise ValueError("dataset must not be empty.")

        smiles_values = dataset[smiles_column].unique()
        if len(smiles_values) != 1:
            raise ValueError(
                "All records in the dataset must have the same SMILES. "
                f"Got unique SMILES: {smiles_values}"
            )

        smiles = str(smiles_values[0])
        fragment_ion_tree, fragment_compound_by_smiles = (
            self._build_and_register_fragment_ion_tree(
                smiles=smiles,
                max_node=max_node,
                max_edge=max_edge,
                max_depth=max_depth,
            )
        )

        sample_rows: List[Optional[TrainingSampleInputRow]] = []
        peak_sets: List[Tuple[Adduct, Tuple[float, ...]]] = []

        for record_index in range(len(dataset)):
            record = dataset[record_index]
            try:
                parsed = self._parse_record_info(
                    record,
                    precursor_mz_column=precursor_mz_column,
                    adduct_type_column=adduct_type_column,
                    collision_energy_column=collision_energy_column,
                    smiles_column=smiles_column,
                    instrument_column=instrument_column,
                )
                record_smiles, _, precursor_type, adduct_type_index, ce_value, _ = parsed
                if record_smiles != smiles:
                    raise ValueError(
                        f"Record SMILES {record_smiles!r} does not match "
                        f"dataset SMILES {smiles!r}."
                    )
            except Exception:
                sample_rows.append(None)
                continue

            peak_mz = tuple(float(peak.mz) for peak in record.peaks)
            sample_rows.append(
                TrainingSampleInputRow(
                    adduct_type_index=int(adduct_type_index),
                    ce_value=float(ce_value),
                    peak_mz=peak_mz,
                    peak_intensity=tuple(float(peak.intensity) for peak in record.peaks),
                    source_label=f"record_index={record_index}",
                )
            )
            peak_sets.append((precursor_type, peak_mz))

        assignments = self._model.fragmenter.assign_fragment_pathways_to_peak_sets(
            fragment_ion_tree=fragment_ion_tree,
            peak_sets=peak_sets,
        )

        return self._add_assigned_training_sample_rows(
            sample_rows=sample_rows,
            assignments=assignments,
            smiles=smiles,
            fragment_ion_tree=fragment_ion_tree,
            fragment_compound_by_smiles=fragment_compound_by_smiles,
            desc="Adding training samples",
        )

    def add_training_samples_from_structure(
        self,
        structure: TrainingFragmentTreeStructure,
        *,
        smiles: str,
        max_node: int = -1,
        max_edge: int = -1,
        max_depth: Optional[int] = None,
    ) -> np.ndarray:
        """Rebuild training samples from a saved structure and current model."""
        fragment_ion_tree, fragment_compound_by_smiles = (
            self._build_and_register_fragment_ion_tree(
                smiles=smiles,
                max_node=max_node,
                max_edge=max_edge,
                max_depth=max_depth,
            )
        )
        sample_rows = self._sample_rows_from_structure(structure)
        peak_sets = [
            (self._model.fragmenter.adduct_types[row.adduct_type_index], row.peak_mz)
            for row in sample_rows
            if row is not None
        ]
        assignments = self._model.fragmenter.assign_fragment_pathways_to_peak_sets(
            fragment_ion_tree=fragment_ion_tree,
            peak_sets=peak_sets,
        )

        return self._add_assigned_training_sample_rows(
            sample_rows=sample_rows,
            assignments=assignments,
            smiles=smiles,
            fragment_ion_tree=fragment_ion_tree,
            fragment_compound_by_smiles=fragment_compound_by_smiles,
            desc="Rebuilding training samples",
        )

    def _build_and_register_fragment_ion_tree(
        self,
        *,
        smiles: str,
        max_node: int,
        max_edge: int,
        max_depth: Optional[int],
    ) -> Tuple[FragmentIonTree, Dict[str, Compound]]:
        compound = Compound.from_smiles(smiles)
        if max_depth is None:
            max_depth = self._model.tree_max_depth

        fragment_ion_tree = self._model.fragmenter.build_fragment_ion_tree(
            compound=compound,
            max_node=max_node,
            max_edge=max_edge,
            max_depth=max_depth,
            _include_fragment_compound_cache=True,
        )
        fragment_compound_by_smiles = self._make_fragment_compound_by_smiles(
            fragment_ion_tree
        )
        self._register_fragment_ion_tree(
            fragment_ion_tree=fragment_ion_tree,
            fragment_compound_by_smiles=fragment_compound_by_smiles,
        )
        return fragment_ion_tree, fragment_compound_by_smiles

    def _sample_rows_from_structure(
        self,
        structure: TrainingFragmentTreeStructure,
    ) -> List[TrainingSampleInputRow]:
        sample_rows: List[TrainingSampleInputRow] = []
        for sample_index in range(structure.num_samples):
            adduct_type_index = int(structure.sample_adduct_type_index[sample_index].item())
            if adduct_type_index < 0 or adduct_type_index >= len(self._model.fragmenter.adduct_types):
                raise ValueError(
                    "Saved sample adduct index is not supported by the current model: "
                    f"{adduct_type_index}"
                )
            start = int(structure.sample_peak_ptr[sample_index].item())
            end = int(structure.sample_peak_ptr[sample_index + 1].item())
            sample_rows.append(
                TrainingSampleInputRow(
                    adduct_type_index=adduct_type_index,
                    ce_value=float(structure.sample_ce_value[sample_index].item()),
                    peak_mz=tuple(
                        float(value)
                        for value in structure.sample_peak_mz[start:end].detach().cpu().tolist()
                    ),
                    peak_intensity=tuple(
                        float(value)
                        for value in structure.sample_peak_intensity[start:end].detach().cpu().tolist()
                    ),
                    source_label=f"sample_index={sample_index}",
                )
            )
        return sample_rows

    def _add_assigned_training_sample_rows(
        self,
        *,
        sample_rows: Sequence[Optional[TrainingSampleInputRow]],
        assignments: Sequence[Tuple[FragmentPathwayGroup, Sequence[FragmentPathwayGroup]]],
        smiles: str,
        fragment_ion_tree: FragmentIonTree,
        fragment_compound_by_smiles: Dict[str, Compound],
        desc: str,
    ) -> np.ndarray:
        sample_indexes: List[int] = []
        assignment_index = 0

        for row in tqdm(sample_rows, desc=desc, mininterval=1.0):
            if row is None:
                sample_indexes.append(-1)
                continue

            (
                precursor_fragment_pathways,
                fragment_pathways_by_peaks,
            ) = assignments[assignment_index]
            assignment_index += 1

            sample_index = self._add_assigned_training_sample(
                row=row,
                precursor_fragment_pathways=precursor_fragment_pathways,
                fragment_pathways_by_peaks=fragment_pathways_by_peaks,
                smiles=smiles,
                fragment_ion_tree=fragment_ion_tree,
                fragment_compound_by_smiles=fragment_compound_by_smiles,
            )
            sample_indexes.append(sample_index)

        return np.asarray(sample_indexes, dtype=int)

    def _add_assigned_training_sample(
        self,
        *,
        row: TrainingSampleInputRow,
        precursor_fragment_pathways: FragmentPathwayGroup,
        fragment_pathways_by_peaks: Sequence[FragmentPathwayGroup],
        smiles: str,
        fragment_ion_tree: FragmentIonTree,
        fragment_compound_by_smiles: Dict[str, Compound],
    ) -> int:
        if len(precursor_fragment_pathways) == 0:
            return -1

        sample_index = len(self.samples)
        sample = TrainingFragmentTreeSample(
            adduct_type_index=int(row.adduct_type_index),
            ce_value=float(row.ce_value),
            peak_mz=list(row.peak_mz),
            peak_intensity=list(row.peak_intensity),
        )

        try:
            self._add_precursor_pathways_to_sample(
                sample=sample,
                precursor_fragment_pathways=precursor_fragment_pathways,
                adduct_type_index=int(row.adduct_type_index),
                fragment_compound_by_smiles=fragment_compound_by_smiles,
            )
            self._add_peak_pathways_to_sample(
                sample=sample,
                fragment_ion_tree=fragment_ion_tree,
                fragment_pathways_by_peaks=fragment_pathways_by_peaks,
                fragment_compound_by_smiles=fragment_compound_by_smiles,
            )
        except Exception as exc:
            print(
                "[WARN] Failed to add training sample "
                f"{row.source_label}, smiles={smiles!r}, "
                f"adduct_type={self._model.fragmenter.adduct_types[int(row.adduct_type_index)]}: {exc}",
                file=sys.stderr,
            )
            return -1

        if not (
            len(sample.precursor_edge_indexes)
            == len(sample.precursor_unsaturation_indexes)
            == len(sample.precursor_radical_indexes)
        ):
            raise ValueError(
                "Mismatch in lengths of precursor_edge_indexes, "
                "precursor_unsaturation_indexes, and "
                "precursor_radical_indexes."
            )

        if len(sample.precursor_edge_indexes) == 0:
            return -1

        self.samples[int(sample_index)] = sample
        return int(sample_index)

    def to_structure(self) -> TrainingFragmentTreeStructure:
        structure = super().to_structure()
        sample_indexes = sorted(self.samples.keys())
        sample_index_remap = {
            old_sample_index: new_sample_index
            for new_sample_index, old_sample_index in enumerate(sample_indexes)
        }

        sample_peak_mz_values: List[float] = []
        sample_peak_intensity_values: List[float] = []
        sample_peak_ptr_values: List[int] = [0]

        formula_rows: List[Tensor] = []
        peak_formula_ptr_counts = [0 for _ in range(1)]
        formula_key_to_index: Dict[Tuple[int, Tuple[float, ...]], int] = {}
        assignments_by_formula: List[List[TrainingFormulaTarget]] = []

        for old_sample_index in sample_indexes:
            sample = self.samples[old_sample_index]
            if not isinstance(sample, TrainingFragmentTreeSample):
                sample_peak_ptr_values.append(len(sample_peak_mz_values))
                continue

            peak_start = len(sample_peak_mz_values)
            sample_peak_mz_values.extend(float(value) for value in sample.peak_mz)
            sample_peak_intensity_values.extend(float(value) for value in sample.peak_intensity)
            sample_peak_ptr_values.append(len(sample_peak_mz_values))

            while len(peak_formula_ptr_counts) < len(sample_peak_mz_values) + 1:
                peak_formula_ptr_counts.append(0)

            for target in sample.target_formula_rows:
                peak_index = int(target.peak_index)
                if peak_index < 0 or peak_index >= len(sample.peak_mz):
                    continue
                global_peak_index = peak_start + peak_index
                formula_tuple = tuple(float(v) for v in target.formula_tensor.detach().cpu().view(-1).tolist())
                formula_key = (int(global_peak_index), formula_tuple)
                formula_index = formula_key_to_index.get(formula_key)
                if formula_index is None:
                    formula_index = len(formula_rows)
                    formula_key_to_index[formula_key] = formula_index
                    formula_rows.append(target.formula_tensor.to(structure.device))
                    assignments_by_formula.append([])
                    peak_formula_ptr_counts[global_peak_index + 1] += 1
                assignments_by_formula[formula_index].append(target)

        for index in range(1, len(peak_formula_ptr_counts)):
            peak_formula_ptr_counts[index] += peak_formula_ptr_counts[index - 1]

        formula_dim = int(structure.node_formula.size(1))
        if formula_rows:
            target_formula = torch.stack(formula_rows, dim=0).to(
                dtype=torch.float32,
                device=structure.device,
            )
        else:
            target_formula = torch.empty(
                (0, formula_dim),
                dtype=torch.float32,
                device=structure.device,
            )

        target_terminal_node_indexes: List[int] = []
        target_ion_indexes: List[int] = []
        target_unsaturation_indexes: List[int] = []
        target_radical_indexes: List[int] = []
        formula_assignment_ptr_values: List[int] = [0]
        target_expand_node_indexes: List[int] = []
        terminal_expand_ptr_values: List[int] = [0]

        for assignments in assignments_by_formula:
            for target in assignments:
                target_terminal_node_indexes.append(int(target.node_index))
                target_ion_indexes.append(int(target.ion_index))
                target_unsaturation_indexes.append(int(target.unsaturation_index))
                target_radical_indexes.append(int(target.radical_index))
                seen_expand_nodes: Set[int] = set()
                for node_index in target.expand_node_indexes:
                    node_index = int(node_index)
                    if node_index in seen_expand_nodes:
                        continue
                    seen_expand_nodes.add(node_index)
                    target_expand_node_indexes.append(node_index)
                terminal_expand_ptr_values.append(len(target_expand_node_indexes))
            formula_assignment_ptr_values.append(len(target_terminal_node_indexes))

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
            sample_peak_mz=torch.tensor(
                sample_peak_mz_values,
                dtype=torch.float32,
                device=structure.device,
            ),
            sample_peak_intensity=torch.tensor(
                sample_peak_intensity_values,
                dtype=torch.float32,
                device=structure.device,
            ),
            sample_peak_ptr=torch.tensor(
                sample_peak_ptr_values,
                dtype=torch.long,
                device=structure.device,
            ),
            target_formula=target_formula,
            peak_formula_ptr=torch.tensor(
                peak_formula_ptr_counts,
                dtype=torch.long,
                device=structure.device,
            ),
            target_terminal_node_index=torch.tensor(
                target_terminal_node_indexes,
                dtype=torch.long,
                device=structure.device,
            ),
            target_ion_index=torch.tensor(
                target_ion_indexes,
                dtype=torch.long,
                device=structure.device,
            ),
            target_unsaturation_index=torch.tensor(
                target_unsaturation_indexes,
                dtype=torch.long,
                device=structure.device,
            ),
            target_radical_index=torch.tensor(
                target_radical_indexes,
                dtype=torch.long,
                device=structure.device,
            ),
            formula_assignment_ptr=torch.tensor(
                formula_assignment_ptr_values,
                dtype=torch.long,
                device=structure.device,
            ),
            target_expand_node_index=torch.tensor(
                target_expand_node_indexes,
                dtype=torch.long,
                device=structure.device,
            ),
            terminal_expand_ptr=torch.tensor(
                terminal_expand_ptr_values,
                dtype=torch.long,
                device=structure.device,
            ),
        )

    def _make_fragment_compound_by_smiles(
        self,
        fragment_ion_tree: FragmentIonTree,
    ) -> Dict[str, Compound]:
        fragment_compound_by_smiles: Dict[str, Compound] = {}
        fragment_compound_by_index = (
            fragment_ion_tree._fragment_compound_by_index
            if hasattr(fragment_ion_tree, "_fragment_compound_by_index")
            else None
        )

        if fragment_compound_by_index is not None:
            for compound in fragment_compound_by_index.values():
                fragment_compound_by_smiles[compound.smiles] = compound

        return fragment_compound_by_smiles

    def _register_fragment_ion_tree(
        self,
        *,
        fragment_ion_tree: FragmentIonTree,
        fragment_compound_by_smiles: Dict[str, Compound],
    ) -> None:
        for smiles in fragment_ion_tree.node_smiles:
            smiles = str(smiles)
            self._ensure_get_node_index(
                smiles,
                compound=fragment_compound_by_smiles.get(smiles),
            )

        for edge_index in range(fragment_ion_tree.num_edges):
            edge = fragment_ion_tree.get_edge(edge_index)
            src_smiles = str(fragment_ion_tree.node_smiles[edge.source_index])
            dst_smiles = str(fragment_ion_tree.node_smiles[edge.target_index])
            self._ensure_get_edge_index(
                src_smiles,
                dst_smiles,
                FragmentPathwayEdge.from_fragment_edge(edge),
            )

    def _add_precursor_pathways_to_sample(
        self,
        *,
        sample: TrainingFragmentTreeSample,
        precursor_fragment_pathways: FragmentPathwayGroup,
        adduct_type_index: int,
        fragment_compound_by_smiles: Dict[str, Compound],
    ) -> None:
        main_adduct_type = self._model.fragmenter.adduct_types[adduct_type_index]

        for precursor_pathway in precursor_fragment_pathways:
            precursor_edge_index_path = self._fragment_pathway_to_edge_index_path(
                fragment_pathway=precursor_pathway,
                padding_length=self._model.fragmenter.precursor_candidate_max_depth,
                fragment_compound_by_smiles=fragment_compound_by_smiles,
            )
            precursor_node = precursor_pathway.precursor_node
            if precursor_node is None:
                continue

            precursor_adduct = precursor_node.precursor_adduct_type
            precursor_delta_h_state = (
                self._model.fragmenter.get_precursor_delta_h_state_by_adduct_type(
                    main_adduct_type=main_adduct_type,
                    adduct_type=precursor_adduct,
                )
            )

            sample.precursor_edge_indexes.append(tuple(precursor_edge_index_path))
            sample.precursor_unsaturation_indexes.append(int(precursor_delta_h_state[0]))
            sample.precursor_radical_indexes.append(int(precursor_delta_h_state[1]))
            sample.edge_indexes.update(
                int(edge_index)
                for edge_index in precursor_edge_index_path
                if edge_index >= 0
            )

    def _add_peak_pathways_to_sample(
        self,
        *,
        sample: TrainingFragmentTreeSample,
        fragment_ion_tree: FragmentIonTree,
        fragment_pathways_by_peaks: Sequence[FragmentPathwayGroup],
        fragment_compound_by_smiles: Dict[str, Compound],
    ) -> None:
        formula_tensorizer = self._get_formula_tensorizer()
        group_index_by_peak_formula: Dict[Tuple[int, str], int] = {}

        for peak_index, fragment_pathways in enumerate(fragment_pathways_by_peaks):
            if len(fragment_pathways) == 0:
                continue

            peak_intensity = (
                float(sample.peak_intensity[peak_index])
                if peak_index < len(sample.peak_intensity)
                else 0.0
            )

            for fragment_pathway in fragment_pathways:
                formula = fragment_pathway.formula.normalized
                expand_node_indexes_for_terminal: List[int] = []
                group_key = (int(peak_index), str(formula))
                if group_key not in group_index_by_peak_formula:
                    group_index_by_peak_formula[group_key] = len(group_index_by_peak_formula)
                group_index = group_index_by_peak_formula[group_key]

                edge_index_path = self._fragment_pathway_to_edge_index_path(
                    fragment_pathway=fragment_pathway,
                    padding_length=len(fragment_pathway) - 1,
                    fragment_compound_by_smiles=fragment_compound_by_smiles,
                )
                valid_edge_indexes = [
                    int(edge_index)
                    for edge_index in edge_index_path
                    if edge_index >= 0
                ]
                sample.target_edge_indexes.update(valid_edge_indexes)
                sample.edge_indexes.update(valid_edge_indexes)
                for node_index in range(len(fragment_pathway) - 1):
                    src_smiles = fragment_pathway.get_node(node_index).smiles
                    dst_smiles = fragment_pathway.get_node(node_index + 1).smiles
                    src_node_index = self._get_node_index(src_smiles)
                    expand_node_indexes_for_terminal.append(int(src_node_index))
                    sample.edge_indexes.update(
                        self._get_outgoing_edge_indexes_from_fragment_ion_tree(
                            fragment_ion_tree=fragment_ion_tree,
                            src_smiles=src_smiles,
                        )
                    )
                    true_edge_index = self._get_edge_index(src_smiles, dst_smiles)
                    sample.target_edge_indexes.add(int(true_edge_index))
                    sample.target_edge_rows.append(
                        TrainingEdgeTarget(
                            edge_index=int(true_edge_index),
                            group_index=int(group_index),
                        )
                    )

                    src_node = fragment_pathway.get_node(node_index)
                    if node_index > 0 and not src_node.is_precursor:
                        sample.target_node_expand_indexes.add(int(src_node_index))

                terminal_node = fragment_pathway.terminal_node
                terminal_node_index = self._get_node_index(terminal_node.smiles)
                ion_index, unsaturation_index, radical_index = (
                    self._resolve_fragment_state_target(
                        fragment_pathway=fragment_pathway,
                        main_adduct_type=self._model.fragmenter.adduct_types[
                            int(sample.adduct_type_index)
                        ],
                    )
                )
                sample.target_node_keep_indexes.add(int(terminal_node_index))
                self._add_or_replace_formula_target(
                    sample=sample,
                    target=TrainingFormulaTarget(
                        node_index=int(terminal_node_index),
                        peak_index=int(peak_index),
                        group_index=int(group_index),
                        ion_index=int(ion_index),
                        unsaturation_index=int(unsaturation_index),
                        radical_index=int(radical_index),
                        formula_tensor=formula_tensorizer.formula_to_tensor(
                            formula,
                            dtype=torch.float32,
                        ),
                        intensity=peak_intensity,
                        expand_node_indexes=tuple(expand_node_indexes_for_terminal),
                    ),
                    main_adduct_type=self._model.fragmenter.adduct_types[
                        int(sample.adduct_type_index)
                    ],
                )

    def _add_or_replace_formula_target(
        self,
        *,
        sample: TrainingFragmentTreeSample,
        target: TrainingFormulaTarget,
        main_adduct_type: Adduct,
    ) -> None:
        target_priority = self._state_target_priority(
            main_adduct_type=main_adduct_type,
            ion_index=target.ion_index,
            unsaturation_index=target.unsaturation_index,
            radical_index=target.radical_index,
        )
        for index, current in enumerate(sample.target_formula_rows):
            if (
                int(current.peak_index) != int(target.peak_index)
                or int(current.node_index) != int(target.node_index)
            ):
                continue
            current_priority = self._state_target_priority(
                main_adduct_type=main_adduct_type,
                ion_index=current.ion_index,
                unsaturation_index=current.unsaturation_index,
                radical_index=current.radical_index,
            )
            if target_priority < current_priority:
                sample.target_formula_rows[index] = target
            return

        sample.target_formula_rows.append(target)

    def _resolve_fragment_state_target(
        self,
        *,
        fragment_pathway,
        main_adduct_type: Adduct,
    ) -> Tuple[int, int, int]:
        target_formula = fragment_pathway.formula.normalized
        terminal_formula = fragment_pathway.terminal_node.to_compound().formula

        adduct_index = self._model.main_adduct_types.inverse[main_adduct_type]
        role_index = 0 if fragment_pathway.terminal_node.is_precursor else 1
        ion_mask = self._model.ion_candidate_valid_mask_by_role_adduct[
            role_index, int(adduct_index)
        ]
        unsaturation_mask = self._model.unsaturation_candidate_valid_mask_by_role_adduct[
            role_index, int(adduct_index)
        ]
        radical_mask = self._model.radical_candidate_valid_mask_by_role_adduct[
            role_index, int(adduct_index)
        ]

        matches: List[Tuple[Tuple[int, int, int, int, int], Tuple[int, int, int]]] = []
        for ion_index in ion_mask.nonzero(as_tuple=False).view(-1).tolist():
            ion_adduct = self._model.ion_flat_candidates[int(ion_index), 1]
            for unsaturation_index in unsaturation_mask.nonzero(as_tuple=False).view(-1).tolist():
                unsaturation_adduct = self._model.unsaturation_flat_candidates[
                    int(unsaturation_index), 1
                ]
                for radical_index in radical_mask.nonzero(as_tuple=False).view(-1).tolist():
                    radical_adduct = self._model.radical_flat_candidates[
                        int(radical_index), 1
                    ]
                    adduct = ion_adduct.add_prefer_self(unsaturation_adduct)
                    adduct = adduct.add_prefer_self(radical_adduct)
                    formula = adduct.apply_to_formula(terminal_formula).normalized
                    if formula != target_formula:
                        continue

                    priority = (
                        self._adduct_distance_from_zero(unsaturation_adduct),
                        self._adduct_distance_from_zero(radical_adduct),
                        abs(int(unsaturation_index)),
                        abs(int(radical_index)),
                        abs(int(ion_index)),
                    )
                    matches.append(
                        (
                            priority,
                            (int(ion_index), int(unsaturation_index), int(radical_index)),
                        )
                    )

        if not matches:
            raise ValueError(
                "Could not resolve state target for fragment pathway: "
                f"formula={target_formula}, main_adduct_type={main_adduct_type}, "
                f"terminal_smiles={fragment_pathway.terminal_node.smiles}."
            )

        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    def _state_target_priority(
        self,
        *,
        main_adduct_type: Adduct,
        ion_index: int,
        unsaturation_index: int,
        radical_index: int,
    ) -> Tuple[int, int, int, int, int]:
        unsaturation_adduct = self._model.unsaturation_flat_candidates[
            int(unsaturation_index), 1
        ]
        radical_adduct = self._model.radical_flat_candidates[int(radical_index), 1]
        return (
            self._adduct_distance_from_zero(unsaturation_adduct),
            self._adduct_distance_from_zero(radical_adduct),
            abs(int(unsaturation_index)),
            abs(int(radical_index)),
            abs(int(ion_index)),
        )

    @staticmethod
    def _adduct_distance_from_zero(adduct: Adduct) -> int:
        return int(abs(adduct.charge)) + sum(
            abs(int(count)) for count in adduct.element_diff.values()
        )

    def _get_outgoing_edge_indexes_from_fragment_ion_tree(
        self,
        *,
        fragment_ion_tree: FragmentIonTree,
        src_smiles: str,
    ) -> Set[int]:
        src_node = fragment_ion_tree.get_node_by_smiles(src_smiles)
        if src_node is None:
            raise ValueError(
                f"Source SMILES {src_smiles!r} not found in fragment ion tree."
            )

        edge_indexes: Set[int] = set()
        for out_edge in fragment_ion_tree.get_out_edges(int(src_node.index)):
            dst_smiles = str(fragment_ion_tree.node_smiles[out_edge.target_index])
            edge_index = self._ensure_get_edge_index(
                src_smiles,
                dst_smiles,
                FragmentPathwayEdge.from_fragment_edge(out_edge),
            )
            if edge_index is not None:
                edge_indexes.add(int(edge_index))

        return edge_indexes
