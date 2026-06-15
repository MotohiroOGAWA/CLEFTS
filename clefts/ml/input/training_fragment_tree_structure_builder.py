from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Set, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from ...libs.mmkit.mmkit import Compound, Adduct
from ...libs.msentity.msentity import SpectrumRecord
from ...domain.fragment.ion_tree import FragmentIonTree
from ...domain.fragment.pathway import FragmentPathway, FragmentPathwayGroup, FragmentPathwayNode, FragmentPathwayEdge
from ..specgen import CleftsSpecGen
from .fragment_tree_structure_builder import FragmentTreeSample, FragmentTreeStructureBuilder

@dataclass
class TrainingFragmentTreeSample(FragmentTreeSample):
    """One training sample with target information."""

    peak_mz: List[float] = field(default_factory=list)
    # [K] Peak m/z values.

    peak_intensity: List[float] = field(default_factory=list)
    # [K] Peak intensity values.

@dataclass
class TrainingFragmentTreeStructureBuilder(FragmentTreeStructureBuilder):

    def add_training_sample(
        self,
        record: SpectrumRecord,
        *,
        precursor_mz_column: str,
        adduct_type_column: str,
        collision_energy_column: str,
        smiles_column: str = "smiles",
        instrument_column: Optional[str] = None,
        fragment_ion_tree: Optional[FragmentIonTree] = None,
        precursor_fragment_pathways: Optional[FragmentPathwayGroup] = None,
        fragment_pathways_by_peaks: Optional[Sequence[FragmentPathwayGroup]] = None,
        max_node: int = -1,
        max_edge: int = -1,
    ) -> int:
        """Add one spectrum record as a sample.

        This method registers:
            - sample-specific adduct_type_index
            - sample-specific collision energy value
            - fragment tree edges used by this sample

        Returns
        -------
        int
            sample_index.
        """

        sample_index = len(self.samples)

        precursor_mz = float(record[precursor_mz_column])

        adduct_type_str = str(record[adduct_type_column])
        adduct_type = Adduct.parse(adduct_type_str)

        ce_value_raw = record[collision_energy_column]

        instrument = None
        if instrument_column is not None:
            instrument = record[instrument_column]

        adduct_type_index = self._model.get_index_by_adduct_type(adduct_type)

        ce_value = CleftsSpecGen.parse_ce_to_ev(
            ce_value_raw,
            precursor_mz=precursor_mz,
            instrument=instrument,
        )

        if ce_value is None:
            raise ValueError(
                f"Failed to parse collision energy: {ce_value_raw!r} "
                f"for sample_index={sample_index}."
            )

        smiles = str(record[smiles_column])
        if fragment_ion_tree is None:
            compound = Compound.from_smiles(smiles)
            fragment_ion_tree = self._model.fragmenter.build_fragment_ion_tree(
                compound=compound,
                max_node=max_node,
                max_edge=max_edge,
                _include_fragment_compound_cache=True,
            )
        if precursor_fragment_pathways is None or fragment_pathways_by_peaks is None:
            peaks_mz = [peak.mz for peak in record.peaks]
            precursor_fragment_pathways, fragment_pathways_by_peaks = \
                self._model.fragmenter.assign_fragment_pathways_to_peaks(
                    fragment_ion_tree=fragment_ion_tree,
                    precursor_type=adduct_type,
                    peaks_mz=peaks_mz,
                )
        fragment_compound_by_index = fragment_ion_tree._fragment_compound_by_index if hasattr(fragment_ion_tree, "_fragment_compound_by_index") else None
        fragment_compound_by_smiles = {}
        if fragment_compound_by_index is not None:
            for node_index, compound in fragment_compound_by_index.items():
                fragment_compound_by_smiles[compound.smiles] = compound
        if len(precursor_fragment_pathways) == 0:
            raise ValueError(
                f"No precursor fragment pathways assigned."
            )
        if len(fragment_pathways_by_peaks) != len(record.peaks):
            raise ValueError(
                f"Length of fragment_pathways_by_peaks does not match number of peaks."
            )

        precursor_edge_indexes = self._fragment_pathway_group_to_edge_index_paths(
            fragment_pathway_group=precursor_fragment_pathways,
            fragment_compound_by_smiles=fragment_compound_by_smiles,
            padding_length=self._model.fragmenter.precursor_candidate_max_depth,
        )

        self.samples[sample_index] = TrainingFragmentTreeSample(
            adduct_type_index=adduct_type_index,
            ce_value=ce_value,
            precursor_edge_indexes=precursor_edge_indexes,
            peak_mz=[peak.mz for peak in record.peaks],
            peak_intensity=[peak.intensity for peak in record.peaks],
        )