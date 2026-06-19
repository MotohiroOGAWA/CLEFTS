from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ...libs.mmkit.mmkit import Adduct, Compound
from ..input.fragment_tree_structure import FragmentTreeStructure
from ..input.fragment_tree_features import FragmentTreeFeatures
from ..common.torch_utils.model_base import ModelBase  # Adjust this import path.
from .fragment_tree_probability_model import (
    FragmentTreeProbabilityModel,
    FragmentTreeProbabilityOutput,
    NodeGenerationFlowProbabilities,
)


@dataclass
class FragmentIonPeakAnnotation:
    """One fragment ion annotation attached to a generated peak."""

    peak_type: str
    # "precursor" or "fragment"

    sample_id: int
    batch_node_index: int
    global_node_id: int
    smiles: str
    adduct: Adduct

    formula: str
    mz: float

    intensity: float
    # Contribution intensity after p_emit / max_p_emit normalization.

    probability: float
    # Original graph-wise probability before max normalization.

    pathway_index: Optional[int] = None
    ion_choice_index: Optional[int] = None
    unsaturation_choice_index: Optional[int] = None
    radical_choice_index: Optional[int] = None
    unsaturation_adduct: Optional[Adduct] = None
    radical_adduct: Optional[Adduct] = None
    adduct_probability: Optional[float] = None
    
@dataclass
class GeneratedSpectrumPeak:
    """One formula-grouped generated peak."""

    mz: float
    intensity: float
    sample_id: int

    formula: Optional[str] = None
    fragment_ion_annotations: Optional[List[FragmentIonPeakAnnotation]] = None


@dataclass
class GeneratedMassSpectrum:
    """Generated mass spectrum for one sample."""

    sample_id: int
    peaks: List[GeneratedSpectrumPeak]


@dataclass
class FragmentSpectrumGeneratorOutput:
    """Output of FragmentSpectrumGenerator."""

    probability_output: FragmentTreeProbabilityOutput
    flow_probabilities: NodeGenerationFlowProbabilities
    spectra: List[GeneratedMassSpectrum]


FormulaMzResolverFn = Callable[
    [FragmentTreeProbabilityOutput, FragmentIonPeakAnnotation],
    Tuple[str, float],
]


class FragmentSpectrumGenerator(ModelBase):
    """
    Generate mass spectra using FragmentTreeProbabilityModel.

    Peaks are grouped by formula.
    Intensities are computed from graph-wise emission probabilities:
        intensity = p_emit / max_p_emit

    Precursor peaks are included in p_emit. There is no separate precursor
    emission probability.
    """

    def __init__(
        self,
        probability_model_params: Dict,
        *,
        max_generation_steps: int = 3,
        min_peak_intensity: float = 1e-3,
        include_precursor_peaks: bool = True,
        include_fragment_peaks: bool = True,
        max_fragment_ion_annotations_per_node: Optional[int] = None,
        min_fragment_ion_probability: float = 0.0,
        formula_mz_resolver: Optional[FormulaMzResolverFn] = None,
        normalize_intensity: bool = True,
        max_peaks_per_sample: Optional[int] = None,
    ) -> None:
        super(FragmentSpectrumGenerator, self).__init__(
            ignore_config_keys=["formula_mz_resolver"],
            **{k: v for k, v in locals().items() if k != "self"},
        )

        if max_generation_steps < 0:
            raise ValueError("max_generation_steps must be non-negative.")

        if min_peak_intensity < 0:
            raise ValueError("min_peak_intensity must be non-negative.")

        if min_fragment_ion_probability < 0:
            raise ValueError("min_fragment_ion_probability must be non-negative.")

        if max_fragment_ion_annotations_per_node is not None and max_fragment_ion_annotations_per_node <= 0:
            raise ValueError("max_fragment_ion_annotations_per_node must be positive or None.")

        if max_peaks_per_sample is not None and max_peaks_per_sample <= 0:
            raise ValueError("max_peaks_per_sample must be positive or None.")

        self.probability_model = FragmentTreeProbabilityModel(**probability_model_params)
        self.max_generation_steps = max_generation_steps
        self.min_peak_intensity = min_peak_intensity
        self.include_precursor_peaks = include_precursor_peaks
        self.include_fragment_peaks = include_fragment_peaks
        self.max_fragment_ion_annotations_per_node = max_fragment_ion_annotations_per_node
        self.min_fragment_ion_probability = min_fragment_ion_probability
        self.formula_mz_resolver = formula_mz_resolver
        self.normalize_intensity = normalize_intensity
        self.max_peaks_per_sample = max_peaks_per_sample

    def forward(
        self,
        data: Union[FragmentTreeStructure, FragmentTreeFeatures],
        *,
        include_formula_annotation: bool = False,
        include_fragment_ion_annotation: bool = False,
    ) -> FragmentSpectrumGeneratorOutput:
        """Generate formula-grouped spectra."""

        probability_output = self.probability_model(data)
        flow_probabilities = getattr(probability_output, "flow_probabilities", None)

        if flow_probabilities is None:
            flow_probabilities = self.probability_model.compute_node_generation_flow_probabilities(
                sample_tree_batch=probability_output.sample_tree_batch,
                probabilities=probability_output.probabilities,
                max_generation_steps=self.max_generation_steps,
            )

        max_emit = self._compute_max_p_emit(probability_output, flow_probabilities)
        annotations: List[FragmentIonPeakAnnotation] = []

        if self.include_precursor_peaks:
            annotations.extend(
                self._build_precursor_annotations(
                    probability_output=probability_output,
                    flow_probabilities=flow_probabilities,
                    max_emit=max_emit,
                )
            )

        if self.include_fragment_peaks:
            annotations.extend(
                self._build_fragment_annotations(
                    probability_output=probability_output,
                    flow_probabilities=flow_probabilities,
                    max_emit=max_emit,
                )
            )

        spectra = self._build_formula_grouped_spectra(
            annotations=annotations,
            include_formula_annotation=include_formula_annotation,
            include_fragment_ion_annotation=include_fragment_ion_annotation,
        )

        return FragmentSpectrumGeneratorOutput(
            probability_output=probability_output,
            flow_probabilities=flow_probabilities,
            spectra=spectra,
        )


    @staticmethod
    def _compute_max_p_emit(
        probability_output: FragmentTreeProbabilityOutput,
        flow_probabilities: NodeGenerationFlowProbabilities,
    ) -> Tensor:
        """
        Compute sample-wise max emission probability.

        Returns
        -------
        Tensor
            [N_tree]
            max_p_emit for each node.
            Each value is the maximum p_emit within the same sample graph.

        Notes
        -----
        Precursor peaks are already included in ``flow_probabilities.p_emit``.
        There is no separate precursor emission probability.
        """

        sample_tree_batch = probability_output.sample_tree_batch

        p_emit = flow_probabilities.p_emit
        device = p_emit.device
        dtype = p_emit.dtype

        graph_index_by_node = sample_tree_batch.batch.to(device).long()
        # [N_tree]

        num_graphs = int(sample_tree_batch.kept_sample_ids.numel())

        if graph_index_by_node.numel() != p_emit.numel():
            raise ValueError(
                "sample_tree_batch.batch and p_emit must have the same "
                "node dimension. "
                f"Got {graph_index_by_node.numel()} and {p_emit.numel()}."
            )

        max_p_emit_by_graph = torch.full(
            (num_graphs,),
            -float("inf"),
            dtype=dtype,
            device=device,
        )
        # [G]

        if p_emit.numel() > 0:
            max_p_emit_by_graph.scatter_reduce_(
                dim=0,
                index=graph_index_by_node,
                src=p_emit,
                reduce="amax",
                include_self=True,
            )

        max_p_emit_by_graph = torch.where(
            torch.isfinite(max_p_emit_by_graph) & (max_p_emit_by_graph > 0),
            max_p_emit_by_graph,
            torch.ones_like(max_p_emit_by_graph),
        )

        max_p_emit = max_p_emit_by_graph[graph_index_by_node]
        # [N_tree]

        return max_p_emit


    def _build_precursor_annotations(
        self,
        *,
        probability_output: FragmentTreeProbabilityOutput,
        flow_probabilities: NodeGenerationFlowProbabilities,
        max_emit: Tensor,
    ) -> List[FragmentIonPeakAnnotation]:
        """Build precursor ion annotations.

        Precursor roots are treated as ordinary emitted nodes.
        The precursor ion / unsaturation / radical state is fixed by
        path-level precursor flat indexes generated by the probability model.
        """

        sample_tree_batch = probability_output.sample_tree_batch
        structure = probability_output.ft_features.structure
        probabilities = probability_output.probabilities

        precursor_pathway_seq = sample_tree_batch.precursor_pathway_seq.long()
        precursor_pathway_ptr = sample_tree_batch.precursor_pathway_ptr.long()
        row_counts = precursor_pathway_ptr[1:] - precursor_pathway_ptr[:-1]

        graph_index_by_precursor = torch.repeat_interleave(
            torch.arange(
                row_counts.numel(),
                dtype=torch.long,
                device=precursor_pathway_seq.device,
            ),
            row_counts,
        )
        # [P]

        kept_sample_ids = sample_tree_batch.kept_sample_ids.long()
        annotations: List[FragmentIonPeakAnnotation] = []
        seen_precursor_keys = set()

        for pathway_index in range(int(precursor_pathway_seq.size(0))):
            batch_node_index = int(
                precursor_pathway_seq[pathway_index, 0].detach().cpu().item()
            )

            if batch_node_index < 0:
                raise ValueError(
                    "Each precursor pathway must start with a valid root node. "
                    f"pathway_index={pathway_index}."
                )

            ion_index = int(
                sample_tree_batch.precursor_ion_flat_index[
                    pathway_index
                ].detach().cpu().item()
            )
            unsaturation_index = int(
                sample_tree_batch.precursor_unsaturation_flat_index[
                    pathway_index
                ].detach().cpu().item()
            )
            radical_index = int(
                sample_tree_batch.precursor_radical_flat_index[
                    pathway_index
                ].detach().cpu().item()
            )

            # p_emit is node-level. If multiple precursor pathways point to
            # the same root node and the same fixed precursor state, emitting
            # all rows would duplicate the same peak. Keep one annotation for
            # each unique node/state combination.
            precursor_key = (
                batch_node_index,
                ion_index,
                unsaturation_index,
                radical_index,
            )
            if precursor_key in seen_precursor_keys:
                continue
            seen_precursor_keys.add(precursor_key)

            node_probability = float(
                flow_probabilities.p_emit[batch_node_index].detach().cpu().item()
            )

            ion_probability = float(
                probabilities.ion.prob[
                    batch_node_index,
                    ion_index,
                ].detach().cpu().item()
            )
            unsaturation_probability = float(
                probabilities.unsaturation.prob[
                    batch_node_index,
                    unsaturation_index,
                ].detach().cpu().item()
            )
            radical_probability = float(
                probabilities.radical.prob[
                    batch_node_index,
                    radical_index,
                ].detach().cpu().item()
            )

            adduct_probability = (
                ion_probability
                * unsaturation_probability
                * radical_probability
            )

            probability = node_probability * adduct_probability

            node_max_emit = float(max_emit[batch_node_index].detach().cpu().item())
            intensity = probability / node_max_emit

            if intensity < self.min_peak_intensity:
                continue

            graph_index = int(
                graph_index_by_precursor[pathway_index].detach().cpu().item()
            )
            sample_id = int(kept_sample_ids[graph_index].detach().cpu().item())
            global_node_id = int(
                sample_tree_batch.node_id_global[
                    batch_node_index
                ].detach().cpu().item()
            )
            smiles = self._get_node_smiles(structure, global_node_id)

            ion_adduct = self.probability_model.ion_flat_candidates[
                ion_index,
                1,
            ]
            unsaturation_adduct = (
                self.probability_model.unsaturation_flat_candidates[
                    unsaturation_index,
                    1,
                ]
            )
            radical_adduct = self.probability_model.radical_flat_candidates[
                radical_index,
                1,
            ]

            adduct = self._combine_adducts(
                ion_adduct=ion_adduct,
                unsaturation_adduct=unsaturation_adduct,
                radical_adduct=radical_adduct,
            )

            annotation = FragmentIonPeakAnnotation(
                peak_type="precursor",
                sample_id=sample_id,
                batch_node_index=batch_node_index,
                global_node_id=global_node_id,
                smiles=smiles,
                adduct=adduct,
                formula="",
                mz=0.0,
                intensity=intensity,
                probability=probability,
                pathway_index=pathway_index,
                ion_choice_index=ion_index,
                unsaturation_choice_index=unsaturation_index,
                radical_choice_index=radical_index,
                unsaturation_adduct=unsaturation_adduct,
                radical_adduct=radical_adduct,
                adduct_probability=adduct_probability,
            )

            formula, mz = self._resolve_formula_mz(probability_output, annotation)
            annotation.formula = formula
            annotation.mz = mz
            annotations.append(annotation)

        return annotations


    def _build_fragment_annotations(
        self,
        *,
        probability_output: FragmentTreeProbabilityOutput,
        flow_probabilities: NodeGenerationFlowProbabilities,
        max_emit: Tensor,
    ) -> List[FragmentIonPeakAnnotation]:
        """Build fragment ion annotations."""

        sample_tree_batch = probability_output.sample_tree_batch
        structure = probability_output.ft_features.structure
        probabilities = probability_output.probabilities

        graph_index_by_node = sample_tree_batch.batch.long()
        kept_sample_ids = sample_tree_batch.kept_sample_ids.long()
        annotations: List[FragmentIonPeakAnnotation] = []

        node_is_precursor_root = getattr(
            sample_tree_batch,
            "node_is_precursor_root",
            None,
        )

        for batch_node_index in range(int(flow_probabilities.p_emit.size(0))):
            if node_is_precursor_root is not None:
                if bool(node_is_precursor_root[batch_node_index].detach().cpu().item()):
                    # Precursor roots are handled by _build_precursor_annotations.
                    continue

            node_probability = float(
                flow_probabilities.p_emit[batch_node_index].detach().cpu().item()
            )

            node_max_emit = float(max_emit[batch_node_index].detach().cpu().item())
            node_intensity = node_probability / node_max_emit

            if node_intensity < self.min_peak_intensity:
                continue

            graph_index = int(graph_index_by_node[batch_node_index].detach().cpu().item())
            sample_id = int(kept_sample_ids[graph_index].detach().cpu().item())
            global_node_id = int(
                sample_tree_batch.node_id_global[
                    batch_node_index
                ].detach().cpu().item()
            )
            smiles = self._get_node_smiles(structure, global_node_id)

            adduct_combinations = self._get_fragment_adduct_combinations(
                probabilities=probabilities,
                batch_node_index=batch_node_index,
            )

            for (
                ion_index,
                unsaturation_index,
                radical_index,
                adduct_probability,
            ) in adduct_combinations:
                if adduct_probability < self.min_fragment_ion_probability:
                    continue

                ion_adduct = self.probability_model.ion_flat_candidates[
                    ion_index,
                    1,
                ]
                unsaturation_adduct = (
                    self.probability_model.unsaturation_flat_candidates[
                        unsaturation_index,
                        1,
                    ]
                )
                radical_adduct = self.probability_model.radical_flat_candidates[
                    radical_index,
                    1,
                ]

                adduct = self._combine_adducts(
                    ion_adduct=ion_adduct,
                    unsaturation_adduct=unsaturation_adduct,
                    radical_adduct=radical_adduct,
                )

                contribution_intensity = node_intensity * adduct_probability
                contribution_probability = node_probability * adduct_probability

                annotation = FragmentIonPeakAnnotation(
                    peak_type="fragment",
                    sample_id=sample_id,
                    batch_node_index=batch_node_index,
                    global_node_id=global_node_id,
                    smiles=smiles,
                    adduct=adduct,
                    formula="",
                    mz=0.0,
                    intensity=contribution_intensity,
                    probability=contribution_probability,
                    ion_choice_index=ion_index,
                    unsaturation_choice_index=unsaturation_index,
                    radical_choice_index=radical_index,
                    unsaturation_adduct=unsaturation_adduct,
                    radical_adduct=radical_adduct,
                    adduct_probability=adduct_probability,
                )

                formula, mz = self._resolve_formula_mz(probability_output, annotation)
                annotation.formula = formula
                annotation.mz = mz
                annotations.append(annotation)

        return annotations

    def _get_fragment_adduct_combinations(
        self,
        *,
        probabilities,
        batch_node_index: int,
    ) -> List[Tuple[int, int, int, float]]:
        """Get fragment ion adduct combinations for one node."""

        ion_indices = probabilities.ion.valid_mask[batch_node_index].nonzero(as_tuple=False).view(-1)
        unsaturation_indices = probabilities.unsaturation.valid_mask[batch_node_index].nonzero(as_tuple=False).view(-1)
        radical_indices = probabilities.radical.valid_mask[batch_node_index].nonzero(as_tuple=False).view(-1)

        combinations: List[Tuple[int, int, int, float]] = []

        for ion_index_tensor in ion_indices:
            ion_index = int(ion_index_tensor.detach().cpu().item())
            p_ion = float(probabilities.ion.prob[batch_node_index, ion_index].detach().cpu().item())

            for unsaturation_index_tensor in unsaturation_indices:
                unsaturation_index = int(unsaturation_index_tensor.detach().cpu().item())
                p_unsaturation = float(probabilities.unsaturation.prob[batch_node_index, unsaturation_index].detach().cpu().item())

                for radical_index_tensor in radical_indices:
                    radical_index = int(radical_index_tensor.detach().cpu().item())
                    p_radical = float(probabilities.radical.prob[batch_node_index, radical_index].detach().cpu().item())
                    combinations.append((ion_index, unsaturation_index, radical_index, p_ion * p_unsaturation * p_radical))

        combinations.sort(key=lambda x: x[3], reverse=True)

        if self.max_fragment_ion_annotations_per_node is not None:
            combinations = combinations[: self.max_fragment_ion_annotations_per_node]

        return combinations

    def _build_formula_grouped_spectra(
        self,
        *,
        annotations: List[FragmentIonPeakAnnotation],
        include_formula_annotation: bool,
        include_fragment_ion_annotation: bool,
    ) -> List[GeneratedMassSpectrum]:
        """Group fragment ion annotations by formula and sum intensities."""

        grouped: Dict[Tuple[int, str], List[FragmentIonPeakAnnotation]] = {}

        for annotation in annotations:
            grouped.setdefault((annotation.sample_id, annotation.formula), []).append(annotation)

        peaks_by_sample: Dict[int, List[GeneratedSpectrumPeak]] = {}

        for (sample_id, formula), group_annotations in grouped.items():
            intensity = sum(annotation.intensity for annotation in group_annotations)
            mz = group_annotations[0].mz

            peak = GeneratedSpectrumPeak(
                mz=mz,
                intensity=intensity,
                sample_id=sample_id,
                formula=formula if include_formula_annotation else None,
                fragment_ion_annotations=group_annotations if include_fragment_ion_annotation else None,
            )

            peaks_by_sample.setdefault(sample_id, []).append(peak)

        spectra: List[GeneratedMassSpectrum] = []

        for sample_id in sorted(peaks_by_sample.keys()):
            peaks = peaks_by_sample[sample_id]
            peaks = sorted(peaks, key=lambda peak: peak.intensity, reverse=True)

            if self.max_peaks_per_sample is not None:
                peaks = peaks[: self.max_peaks_per_sample]

            if self.normalize_intensity:
                peaks = self._normalize_intensity(peaks)

            peaks = sorted(peaks, key=lambda peak: peak.mz)
            spectra.append(GeneratedMassSpectrum(sample_id=sample_id, peaks=peaks))

        return spectra


    def _resolve_formula_mz(
        self,
        probability_output: FragmentTreeProbabilityOutput,
        annotation: FragmentIonPeakAnnotation,
    ) -> Tuple[str, float]:
        """Resolve formula and m/z from smiles + final adduct."""

        if self.formula_mz_resolver is not None:
            return self.formula_mz_resolver(probability_output, annotation)

        compound = Compound.from_smiles(annotation.smiles)
        formula = annotation.adduct.apply_to_formula(compound.formula).normalized
        mz = float(formula.exact_mass)

        return str(formula), mz

    @staticmethod
    def _combine_adducts(
        *,
        ion_adduct: Adduct,
        unsaturation_adduct: Optional[Adduct],
        radical_adduct: Optional[Adduct],
    ) -> Adduct:
        """Combine ion, unsaturation, and radical adduct components."""

        adduct = ion_adduct

        if unsaturation_adduct is not None:
            adduct = adduct.add_prefer_self(unsaturation_adduct)

        if radical_adduct is not None:
            adduct = adduct.add_prefer_self(radical_adduct)

        return adduct

    @staticmethod
    def _get_node_smiles(
        structure,
        global_node_id: int,
    ) -> str:
        """Get node SMILES from FragmentTreeStructure."""

        if hasattr(structure, "node_smiles"):
            return str(structure.node_smiles[global_node_id])

        if hasattr(structure, "tree") and hasattr(structure.tree, "get_node"):
            node = structure.tree.get_node(global_node_id)

            if hasattr(node, "smiles"):
                return str(node.smiles)

            if hasattr(node, "compound") and hasattr(node.compound, "smiles"):
                return str(node.compound.smiles)

        raise AttributeError("Cannot get node SMILES from structure.")

    @staticmethod
    def _normalize_intensity(
        peaks: List[GeneratedSpectrumPeak],
        max_intensity: float = 1.0,
    ) -> List[GeneratedSpectrumPeak]:
        """Normalize peak intensity to max = 100."""

        if len(peaks) == 0:
            return peaks

        max_intensity = max(peak.intensity for peak in peaks)

        if max_intensity <= 0:
            return peaks

        return [
            GeneratedSpectrumPeak(
                mz=peak.mz,
                intensity=max_intensity * peak.intensity / max_intensity,
                sample_id=peak.sample_id,
                formula=peak.formula,
                fragment_ion_annotations=peak.fragment_ion_annotations,
            )
            for peak in peaks
        ]