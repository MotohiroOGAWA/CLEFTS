from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from bidict import bidict

from clefts.domain.fragment import Fragmenter
from clefts.ml.mol import FormulaTensorizer
from clefts.ml.mol.graph_builder import MolGraphBuilder


class FragmentTreePreprocessingContext:
    """Non-neural definitions required to build FragmentTreeStructure data."""

    def __init__(
        self,
        *,
        symbols: Sequence[str],
        fragmenter_params: dict,
        max_node: int = -1,
        max_edge: int = -1,
    ) -> None:
        self.symbols = tuple(str(symbol) for symbol in symbols)
        self.fragmenter_params = dict(fragmenter_params)
        self.max_node = int(max_node)
        self.max_edge = int(max_edge)
        self.fragmenter = Fragmenter.from_dict(self.fragmenter_params)
        self.mol_graph_builder = MolGraphBuilder(symbols=self.symbols)
        self.main_adduct_types = bidict(
            {index: adduct for index, adduct in enumerate(self.fragmenter.adduct_types)}
        )

        candidate_groups = (
            self.fragmenter.get_ion_shift_adducts_by_adduct_type,
            self.fragmenter.get_unsaturation_adduct_candidates_by_adduct_type,
            self.fragmenter.get_radical_adduct_candidates_by_adduct_type,
        )
        tables = []
        all_formula_adducts = []
        for getter in candidate_groups:
            candidates = {
                adduct: np.asarray(getter(adduct), dtype=object)
                for adduct in self.fragmenter.adduct_types
            }
            all_formula_adducts.extend(
                candidate for values in candidates.values() for candidate in values
            )
            tables.append(self._flatten_candidates(candidates))

        self.ion_flat_candidates, ion_slices = tables[0]
        self.unsaturation_flat_candidates, unsaturation_slices = tables[1]
        self.radical_flat_candidates, radical_slices = tables[2]
        self.ion_candidate_valid_mask_by_role_adduct = self._candidate_mask(
            self.ion_flat_candidates, ion_slices, ion_candidates=True
        )
        self.unsaturation_candidate_valid_mask_by_role_adduct = self._candidate_mask(
            self.unsaturation_flat_candidates, unsaturation_slices, ion_candidates=False
        )
        self.radical_candidate_valid_mask_by_role_adduct = self._candidate_mask(
            self.radical_flat_candidates, radical_slices, ion_candidates=False
        )
        self.formula_tensorizer = FormulaTensorizer.from_symbols_and_adducts(
            symbols=self.symbols, adducts=all_formula_adducts
        )

        reactant_lengths = {}
        product_lengths = {}
        for pattern in self.fragmenter.cleavage_pattern_set.patterns:
            for reaction in pattern.cleavage_reactions:
                reactant_lengths[(int(pattern.pattern_id), int(reaction.id))] = len(
                    reaction.react_idx_to_map
                )
                for product_id, product_map in enumerate(reaction.prod_idx_to_maps):
                    product_lengths[
                        (int(pattern.pattern_id), int(reaction.id), int(product_id))
                    ] = len(product_map)
        self.cleavage_definitions = SimpleNamespace(
            reactant_tuple_length_by_event_type=reactant_lengths,
            product_tuple_length_by_event_type=product_lengths,
        )

    @property
    def tree_max_depth(self) -> int:
        return self.fragmenter.tree_max_depth

    @property
    def precursor_candidate_max_depth(self) -> int:
        return self.fragmenter.precursor_candidate_max_depth

    def get_index_by_adduct_type(self, adduct_type) -> int:
        return self.fragmenter.get_index_by_adduct_type(adduct_type)

    def to_dict(self) -> dict:
        return {
            "symbols": list(self.symbols),
            "fragmenter_params": self.fragmenter.to_dict(),
            "max_node": self.max_node,
            "max_edge": self.max_edge,
        }

    def _flatten_candidates(self, candidates_by_adduct):
        rows = []
        slices = {}
        for adduct in self.fragmenter.adduct_types:
            start = len(rows)
            rows.extend((adduct, candidate) for candidate in candidates_by_adduct[adduct])
            slices[adduct] = slice(start, len(rows))
        return np.asarray(rows, dtype=object).reshape(-1, 2), slices

    def _candidate_mask(self, flat_candidates, slices, *, ion_candidates: bool):
        mask = torch.zeros(
            (2, len(self.fragmenter.adduct_types), len(flat_candidates)), dtype=torch.bool
        )
        for adduct_index, adduct in self.main_adduct_types.items():
            candidate_slice = slices[adduct]
            mask[1, int(adduct_index), candidate_slice] = True
            if not ion_candidates:
                mask[0, int(adduct_index), candidate_slice] = True
                continue
            for index in range(candidate_slice.start, candidate_slice.stop):
                if flat_candidates[index, 1] == adduct:
                    mask[0, int(adduct_index), index] = True
        return mask
