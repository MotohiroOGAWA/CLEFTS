from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch.nn as nn

from ...libs.mmkit.mmkit import Adduct
from ...libs.msentity.msentity import MSDataset
from ...libs.msentity.msentity.core.PeakSeries import PeakSeries
from ..common.torch_utils.model_base import ModelBase
from ..input.fragment_tree_features import FragmentTreeFeatures
from ..input.fragment_tree_structure import FragmentTreeStructure
from .fragment_tree_candidate_selector import FragmentTreeCandidateSelector
from .fragment_tree_feature_model import FragmentTreeFeatureModel
from .fragment_tree_formula_intensity_model import FragmentTreeFormulaIntensityPredictor


@dataclass
class FragmentIonPeakAnnotation:
    """One fragment ion annotation attached to a generated peak."""

    peak_type: str
    sample_id: int
    batch_node_index: int
    global_node_id: int
    smiles: str
    adduct: Adduct
    formula: str
    mz: float
    intensity: float
    probability: float
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
    """Output of FragmentTreeSpectrumPredictor / FragmentSpectrumGenerator."""

    spectra: List[GeneratedMassSpectrum]
    candidate_output: Optional[object] = None
    formula_intensity_output: Optional[object] = None


FormulaMzResolverFn = Callable[
    [FragmentIonPeakAnnotation],
    Tuple[str, float],
]


class FragmentTreeSpectrumPredictor(nn.Module):
    """Predict in-silico spectra from FragmentTreeStructure/Features."""

    def __init__(
        self,
        candidate_selector: FragmentTreeCandidateSelector,
        formula_intensity_predictor: FragmentTreeFormulaIntensityPredictor,
        *,
        expand_cleavages: bool = True,
        normalize_intensity: bool = True,
        min_peak_intensity: float = 1e-3,
        max_peaks_per_sample: Optional[int] = None,
        max_edges_per_step: Optional[int] = 128,
        max_retained_edges: Optional[int] = 30,
        max_next_cleavage_candidates: int = 3,
        mol_encoder_checkpoint: Optional[str] = None,
        freeze_mol_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.candidate_selector = candidate_selector
        self.formula_intensity_predictor = formula_intensity_predictor
        self.expand_cleavages = bool(expand_cleavages)
        self.normalize_intensity = bool(normalize_intensity)
        if min_peak_intensity < 0:
            raise ValueError("min_peak_intensity must be non-negative.")
        self.min_peak_intensity = float(min_peak_intensity)
        self.max_peaks_per_sample = max_peaks_per_sample
        self.max_edges_per_step = max_edges_per_step
        self.max_retained_edges = max_retained_edges
        self.max_next_cleavage_candidates = int(max_next_cleavage_candidates)
        self.mol_encoder_checkpoint = mol_encoder_checkpoint
        self.freeze_mol_encoder = freeze_mol_encoder

    def forward(
        self,
        data: Union[FragmentTreeStructure, FragmentTreeFeatures],
        *,
        include_formula_annotation: bool = False,
        include_fragment_ion_annotation: bool = False,
    ) -> FragmentSpectrumGeneratorOutput:
        additional_depth = (
            min(3, max(0, int(self.candidate_selector.fragmenter.tree_max_depth) - 1))
            if self.expand_cleavages
            else 0
        )
        candidate_output = self.candidate_selector.generate_depth_limited_candidates(
            data,
            max_depth=additional_depth,
        )
        formula_intensity_output = self.formula_intensity_predictor.predict_from_candidate_output(candidate_output)
        peaks_by_sample: Dict[int, List[GeneratedSpectrumPeak]] = {}
        feature_model = self.candidate_selector.feature_model
        for prediction in formula_intensity_output.formula_predictions:
            annotations = None
            if include_fragment_ion_annotation:
                annotations = [
                    FragmentIonPeakAnnotation(
                        peak_type="fragment",
                        sample_id=candidate.sample_id,
                        batch_node_index=candidate.batch_node_index,
                        global_node_id=candidate.global_node_id,
                        smiles=str(candidate_output.features.structure.node_smiles[candidate.global_node_id]),
                        adduct=self._combine_adducts(
                            ion_adduct=feature_model.ion_flat_candidates[candidate.ion_index, 1],
                            unsaturation_adduct=feature_model.unsaturation_flat_candidates[candidate.unsaturation_index, 1],
                            radical_adduct=feature_model.radical_flat_candidates[candidate.radical_index, 1],
                        ),
                        formula=str(candidate.formula),
                        mz=prediction.mz,
                        intensity=prediction.intensity,
                        probability=candidate.probability,
                        ion_choice_index=candidate.ion_index,
                        unsaturation_choice_index=candidate.unsaturation_index,
                        radical_choice_index=candidate.radical_index,
                        adduct_probability=None,
                    )
                    for candidate in prediction.candidates
                ]
            peaks_by_sample.setdefault(prediction.sample_id, []).append(
                GeneratedSpectrumPeak(
                    mz=prediction.mz,
                    intensity=prediction.intensity,
                    sample_id=prediction.sample_id,
                    formula=str(prediction.formula) if include_formula_annotation else None,
                    fragment_ion_annotations=annotations,
                )
            )
        spectra: List[GeneratedMassSpectrum] = []
        sample_ids = sorted(set(peaks_by_sample.keys()) | set(range(candidate_output.features.structure.num_samples)))
        for sample_id in sample_ids:
            peaks = sorted(peaks_by_sample.get(sample_id, []), key=lambda peak: peak.intensity, reverse=True)
            if self.max_peaks_per_sample is not None:
                peaks = peaks[: self.max_peaks_per_sample]
            if self.normalize_intensity:
                peaks = self._normalize_intensity(peaks)
            peaks = [peak for peak in peaks if peak.intensity >= self.min_peak_intensity]
            spectra.append(GeneratedMassSpectrum(sample_id=sample_id, peaks=sorted(peaks, key=lambda peak: peak.mz)))
        return FragmentSpectrumGeneratorOutput(
            spectra=spectra,
            candidate_output=candidate_output,
            formula_intensity_output=formula_intensity_output,
        )

    @staticmethod
    def _combine_adducts(*, ion_adduct: Adduct, unsaturation_adduct: Optional[Adduct], radical_adduct: Optional[Adduct]) -> Adduct:
        adduct = ion_adduct
        if unsaturation_adduct is not None:
            adduct = adduct.add_prefer_self(unsaturation_adduct)
        if radical_adduct is not None:
            adduct = adduct.add_prefer_self(radical_adduct)
        return adduct

    @staticmethod
    def _normalize_intensity(peaks: List[GeneratedSpectrumPeak], max_normalized_intensity: float = 1.0) -> List[GeneratedSpectrumPeak]:
        if len(peaks) == 0:
            return peaks
        max_intensity = max(peak.intensity for peak in peaks)
        if max_intensity <= 0:
            return peaks
        return [
            GeneratedSpectrumPeak(
                mz=peak.mz,
                intensity=max_normalized_intensity * peak.intensity / max_intensity,
                sample_id=peak.sample_id,
                formula=peak.formula,
                fragment_ion_annotations=peak.fragment_ion_annotations,
            )
            for peak in peaks
        ]


def fragment_spectrum_output_to_msdataset(output: FragmentSpectrumGeneratorOutput, *, metadata: Optional[pd.DataFrame] = None) -> MSDataset:
    spectra = output.spectra
    if metadata is None:
        metadata = pd.DataFrame({"sample_id": [s.sample_id for s in spectra]})
    else:
        metadata = metadata.reset_index(drop=True).copy()
    if len(metadata) != len(spectra):
        raise ValueError("metadata rows must match generated spectra count.")
    peak_rows: List[Tuple[float, float]] = []
    peak_metadata_rows: List[Dict[str, object]] = []
    offsets = [0]
    for spectrum in spectra:
        for peak in spectrum.peaks:
            peak_rows.append((float(peak.mz), float(peak.intensity)))
            peak_metadata_rows.append({"sample_id": int(peak.sample_id), "formula": peak.formula})
        offsets.append(len(peak_rows))
    data = np.asarray(peak_rows, dtype=np.float64)
    if data.size == 0:
        data = np.empty((0, 2), dtype=np.float64)
    peak_series = PeakSeries(data=data, offsets=np.asarray(offsets, dtype=np.int64), metadata=pd.DataFrame(peak_metadata_rows), metadata_columns=["sample_id", "formula"], sort_by_mz=True)
    return MSDataset(spectrum_metadata=metadata, peak_series=peak_series, columns=metadata.columns.tolist(), description="In silico spectra generated by CLEFTS")



class FragmentSpectrumGenerator(ModelBase):
    """High-level in-silico spectrum generator backed by FragmentTreeFeatureModel."""

    def __init__(
        self,
        probability_model_params: Dict,
        *,
        min_peak_intensity: float = 1e-3,
        include_precursor_peaks: bool = True,
        include_fragment_peaks: bool = True,
        max_fragment_ion_annotations_per_node: Optional[int] = None,
        min_fragment_ion_probability: float = 0.0,
        formula_mz_resolver: Optional[FormulaMzResolverFn] = None,
        normalize_intensity: bool = True,
        max_peaks_per_sample: Optional[int] = None,
        max_edges_per_step: Optional[int] = 128,
        max_retained_edges: Optional[int] = 30,
        max_next_cleavage_candidates: int = 3,
        edge_condition_interaction_dim: int = 64,
        ranking_loss_weight: float = 1.0,
        ranking_pairs_per_edge: int = 4,
        ranking_intensity_threshold: float = 0.05,
        mol_encoder_checkpoint: Optional[str] = None,
        freeze_mol_encoder: bool = True,
    ) -> None:
        super(FragmentSpectrumGenerator, self).__init__(
            ignore_config_keys=["formula_mz_resolver"],
            **{k: v for k, v in locals().items() if k != "self"},
        )

        if min_peak_intensity < 0:
            raise ValueError("min_peak_intensity must be non-negative.")
        if min_fragment_ion_probability < 0:
            raise ValueError("min_fragment_ion_probability must be non-negative.")
        if max_fragment_ion_annotations_per_node is not None and max_fragment_ion_annotations_per_node <= 0:
            raise ValueError("max_fragment_ion_annotations_per_node must be positive or None.")
        if max_peaks_per_sample is not None and max_peaks_per_sample <= 0:
            raise ValueError("max_peaks_per_sample must be positive or None.")
        if max_edges_per_step is not None and max_edges_per_step <= 0:
            raise ValueError("max_edges_per_step must be positive or None.")
        if max_retained_edges is not None and max_retained_edges <= 0:
            raise ValueError("max_retained_edges must be positive or None.")
        if max_next_cleavage_candidates <= 0:
            raise ValueError("max_next_cleavage_candidates must be positive.")

        self.feature_model = FragmentTreeFeatureModel(**probability_model_params)
        if mol_encoder_checkpoint:
            self.feature_model.load_mol_encoder_checkpoint(mol_encoder_checkpoint)
        if freeze_mol_encoder:
            self.feature_model.freeze_mol_encoder()
        self.candidate_selector = FragmentTreeCandidateSelector(
            self.feature_model,
            max_fragment_ion_candidates=(max_fragment_ion_annotations_per_node or max_retained_edges or 30),
            max_next_cleavage_candidates=max_next_cleavage_candidates,
            max_edges_per_step=max_edges_per_step,
            max_retained_edges=max_retained_edges,
            edge_condition_interaction_dim=edge_condition_interaction_dim,
            ranking_loss_weight=ranking_loss_weight,
            ranking_pairs_per_edge=ranking_pairs_per_edge,
            ranking_intensity_threshold=ranking_intensity_threshold,
        )
        self.formula_intensity_predictor = FragmentTreeFormulaIntensityPredictor(
            self.feature_model,
        )
        self.spectrum_predictor = FragmentTreeSpectrumPredictor(
            self.candidate_selector,
            self.formula_intensity_predictor,
            normalize_intensity=normalize_intensity,
            min_peak_intensity=min_peak_intensity,
            max_peaks_per_sample=max_peaks_per_sample,
            max_edges_per_step=max_edges_per_step,
            max_retained_edges=max_retained_edges,
            max_next_cleavage_candidates=max_next_cleavage_candidates,
        )

        self.min_peak_intensity = min_peak_intensity
        self.include_precursor_peaks = include_precursor_peaks
        self.include_fragment_peaks = include_fragment_peaks
        self.max_fragment_ion_annotations_per_node = max_fragment_ion_annotations_per_node
        self.min_fragment_ion_probability = min_fragment_ion_probability
        self.formula_mz_resolver = formula_mz_resolver
        self.normalize_intensity = normalize_intensity
        self.max_peaks_per_sample = max_peaks_per_sample
        self.max_edges_per_step = max_edges_per_step
        self.max_retained_edges = max_retained_edges
        self.max_next_cleavage_candidates = max_next_cleavage_candidates
        self.mol_encoder_checkpoint = mol_encoder_checkpoint
        self.freeze_mol_encoder = freeze_mol_encoder

    @property
    def probability_model(self) -> FragmentTreeFeatureModel:
        """Backward-compatible access for builders that need the feature model."""
        return self.feature_model

    def forward(
        self,
        data: Union[FragmentTreeStructure, FragmentTreeFeatures],
        *,
        include_formula_annotation: bool = False,
        include_fragment_ion_annotation: bool = False,
    ) -> FragmentSpectrumGeneratorOutput:
        return self.spectrum_predictor(
            data,
            include_formula_annotation=include_formula_annotation,
            include_fragment_ion_annotation=include_fragment_ion_annotation,
        )
