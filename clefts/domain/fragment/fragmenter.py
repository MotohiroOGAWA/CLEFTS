from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict, Set, Optional
import json
from pathlib import Path
from collections import defaultdict

from ...libs.mmkit.mmkit import Adduct, Compound

from ..mass.tolerance import MassTolerance, parse_mass_tolerance, format_mass_tolerance
from ..formula import utils as formula_utils
from .tree import *
from .ion_tree import *
from .pathway import *
from .pathway.build_pathway import build_pathway_items_for_node, PathwayItem
from .cleavage import CleavagePatternSet, CleavagePattern, CleavageResult


@dataclass(frozen=True)
class Fragmenter:
    fragment_ion_tree_builder: FragmentIonTreeBuilder
    mass_tolerance: MassTolerance

    @property
    def adduct_types(self) -> Tuple[Adduct, ...]:
        return self.adduct_rule_set.adduct_types
    
    @property
    def adduct_rule_set(self) -> FragmentIonAdductRuleSet:
        return self.fragment_ion_tree_builder.fragment_ion_adduct_rule_set
    
    @property
    def tree_max_depth(self) -> int:
        return self.fragment_ion_tree_builder.max_depth

    @property
    def precursor_candidate_max_depth(self) -> int:
        return self.fragment_ion_tree_builder.min_depth_only_from
    
    @property
    def cleavage_pattern_set(self) -> CleavagePatternSet:
        return self.fragment_ion_tree_builder.cleavage_pattern_set
    
    @property
    def name(self) -> str:
        return self.fragment_ion_tree_builder.name

    def get_index_by_adduct_type(self, adduct_type: Adduct) -> int:
        main_adduct_type = self._resolve_main_adduct_type(adduct_type)
        mapper = self.adduct_rule_set.adduct_type_to_index
        if main_adduct_type not in mapper:
            raise ValueError(
                f"Adduct type not supported by this fragmenter: {adduct_type}"
            )
        return mapper[main_adduct_type]

    def build_fragment_tree(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        max_depth: Optional[int] = None,
        print_info: bool = False,
    ) -> FragmentTree:
        return self.fragment_ion_tree_builder.build_fragment_tree(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            max_depth=max_depth,
            print_info=print_info,
        )
    
    def build_fragment_ion_tree(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        max_depth: Optional[int] = None,
        print_info: bool = False,
        _include_fragment_compound_cache: bool = False,
    ) -> FragmentIonTree:
        return self.fragment_ion_tree_builder.build(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            max_depth=max_depth,
            print_info=print_info,
            _include_fragment_compound_cache=_include_fragment_compound_cache,
        )

    def assign_fragment_pathways_to_peaks(
        self,
        fragment_ion_tree: FragmentIonTree,
        precursor_type: Adduct,
        peaks_mz: Iterable[float],
    ) -> Tuple[FragmentPathwayGroup, Tuple[FragmentPathwayGroup, ...]]:
        """Assign fragment pathways to peaks for one spectrum / record."""

        return self.assign_fragment_pathways_to_peak_sets(
            fragment_ion_tree=fragment_ion_tree,
            peak_sets=((precursor_type, peaks_mz),),
        )[0]

    def assign_fragment_pathways_to_peak_sets(
        self,
        fragment_ion_tree: FragmentIonTree,
        peak_sets: Iterable[Tuple[Adduct, Iterable[float]]],
    ) -> Tuple[Tuple[FragmentPathwayGroup, Tuple[FragmentPathwayGroup, ...]], ...]:
        """Assign fragment pathways to multiple peak sets.

        This method is useful when multiple spectra / records share the same
        FragmentIonTree. Formula candidates and pathway items are reused as much
        as possible before distributing the result back to each peak set.
        """

        peak_sets_tuple: Tuple[Tuple[Adduct, Tuple[float, ...]], ...] = tuple(
            (precursor_type, tuple(peaks_mz))
            for precursor_type, peaks_mz in peak_sets
        )

        fragment_compound_by_index = self._make_fragment_compound_cache(
            fragment_ion_tree
        )

        context_by_precursor_type: Dict[Adduct, _PrecursorAssignmentContext] = {}
        context_by_record_index: Dict[int, _PrecursorAssignmentContext] = {}

        record_indices_by_main_adduct_type: Dict[Adduct, List[int]] = defaultdict(list)

        for record_index, (precursor_type, _) in enumerate(peak_sets_tuple):
            context = context_by_precursor_type.get(precursor_type)

            if context is None:
                context = self._build_precursor_assignment_context(
                    fragment_ion_tree=fragment_ion_tree,
                    precursor_type=precursor_type,
                    fragment_compound_by_index=fragment_compound_by_index,
                )
                context_by_precursor_type[precursor_type] = context

            context_by_record_index[record_index] = context
            record_indices_by_main_adduct_type[context.main_adduct_type].append(
                record_index
            )

        fragment_pathway_lists_by_record_and_peak: List[List[List[FragmentPathway]]] = [
            [
                []
                for _ in peaks_mz
            ]
            for _, peaks_mz in peak_sets_tuple
        ]

        pathway_items_cache: Dict[
            Tuple[Adduct, int],
            Tuple[Tuple[Any, ...], ...],
        ] = {}

        for main_adduct_type, record_indices in record_indices_by_main_adduct_type.items():
            formula_candidates = fragment_ion_tree.get_formula_candidate_group(
                main_adduct_type
            )

            flat_peaks_mz, flat_peak_refs = self._flatten_peak_sets(
                peak_sets_tuple=peak_sets_tuple,
                record_indices=record_indices,
            )

            assigned_peaks = formula_utils.assign_formulas_to_peaks(
                peaks_mz=flat_peaks_mz,
                formula_candidates=formula_candidates.formulas,
                mass_tolerance=self.mass_tolerance,
            )

            peak_indices_by_record_node_and_adduct = (
                self._assign_formula_matches_to_peak_indices(
                    assigned_peaks=assigned_peaks,
                    flat_peak_refs=flat_peak_refs,
                    formula_candidates=formula_candidates,
                )
            )

            self._append_assigned_fragment_pathways(
                fragment_ion_tree=fragment_ion_tree,
                peak_indices_by_record_node_and_adduct=(
                    peak_indices_by_record_node_and_adduct
                ),
                context_by_record_index=context_by_record_index,
                pathway_items_cache=pathway_items_cache,
                fragment_pathway_lists_by_record_and_peak=(
                    fragment_pathway_lists_by_record_and_peak
                ),
            )

        results: List[
            Tuple[FragmentPathwayGroup, Tuple[FragmentPathwayGroup, ...]]
        ] = []

        for record_index, fragment_pathway_lists_by_peak in enumerate(
            fragment_pathway_lists_by_record_and_peak
        ):
            context = context_by_record_index[record_index]

            fragment_pathways_by_peak = tuple(
                FragmentPathwayGroup.from_pathways(
                    fragment_pathways
                ).with_precursor.shortest
                for fragment_pathways in fragment_pathway_lists_by_peak
            )

            results.append(
                (
                    context.precursor_fragment_pathways,
                    fragment_pathways_by_peak,
                )
            )

        return tuple(results)

    def fragment_all(
        self,
        compound: Compound,
    ) -> Tuple[CleavageResult, ...]:
        return self.cleavage_pattern_set.fragment_all(compound)

    def _make_fragment_compound_cache(
        self,
        fragment_ion_tree: FragmentIonTree,
    ) -> Dict[int, Compound]:
        fragment_compound_by_index: Dict[int, Compound] = {}

        cached = getattr(fragment_ion_tree, "_fragment_compound_by_index", None)
        if cached is not None:
            fragment_compound_by_index.update(cached)

        return fragment_compound_by_index

    def _get_fragment_compound(
        self,
        fragment_ion_tree: FragmentIonTree,
        fragment_compound_by_index: Dict[int, Compound],
        node_index: int,
    ) -> Compound:
        compound = fragment_compound_by_index.get(node_index)

        if compound is not None:
            return compound

        node = fragment_ion_tree.get_node(node_index)
        compound = Compound.from_smiles(node.smiles)
        fragment_compound_by_index[node_index] = compound
        return compound

    def _resolve_main_adduct_type(
        self,
        precursor_type: Adduct,
    ) -> Adduct:
        if precursor_type.charge != 1 and precursor_type.charge != -1:
            raise ValueError(
                f"Only singly charged adducts are supported: {precursor_type}"
            )

        matched_adduct_flags, residual_component_adduct = (
            Adduct.split_by_reference_adducts(
                precursor_type,
                reference_adducts=self.adduct_types,
            )
        )

        if sum(matched_adduct_flags) == 0:
            raise ValueError(
                "Adduct does not contain any of the supported adduct types"
                f"({', '.join(str(adduct) for adduct in self.adduct_types)}): "
                f"{precursor_type}"
            )

        if sum(matched_adduct_flags) > 1:
            raise ValueError(
                "Adduct contains multiple supported adduct types"
                f"({', '.join(str(adduct) for adduct in self.adduct_types)}): "
                f"{precursor_type}"
            )

        return self.adduct_types[matched_adduct_flags.index(True)]

    def _build_precursor_assignment_context(
        self,
        fragment_ion_tree: FragmentIonTree,
        precursor_type: Adduct,
        fragment_compound_by_index: Dict[int, Compound],
    ) -> _PrecursorAssignmentContext:
        main_adduct_type = self._resolve_main_adduct_type(precursor_type)

        precursor_compound = self._get_fragment_compound(
            fragment_ion_tree=fragment_ion_tree,
            fragment_compound_by_index=fragment_compound_by_index,
            node_index=0,
        )
        precursor_formula = precursor_type.apply_to_formula(
            precursor_compound.formula
        ).normalized

        precursor_node_candidates = self._find_precursor_node_candidates(
            fragment_ion_tree=fragment_ion_tree,
            main_adduct_type=main_adduct_type,
            precursor_formula=precursor_formula,
            fragment_compound_by_index=fragment_compound_by_index,
        )

        precursor_adduct_types = self._build_precursor_adduct_types(
            main_adduct_type=main_adduct_type,
            precursor_node_candidates=precursor_node_candidates,
        )

        precursor_fragment_pathways = self._build_precursor_fragment_pathways(
            fragment_ion_tree=fragment_ion_tree,
            precursor_adduct_types=precursor_adduct_types,
        )

        return _PrecursorAssignmentContext(
            main_adduct_type=main_adduct_type,
            precursor_adduct_types=precursor_adduct_types,
            precursor_fragment_pathways=precursor_fragment_pathways,
        )

    def _find_precursor_node_candidates(
        self,
        fragment_ion_tree: FragmentIonTree,
        main_adduct_type: Adduct,
        precursor_formula,
        fragment_compound_by_index: Dict[int, Compound],
    ) -> Set[Tuple[int, Adduct]]:
        precursor_node_candidates: Set[Tuple[int, Adduct]] = set()

        nodes_by_depth = fragment_ion_tree.get_nodes_by_depth()
        neutral_delta_h_adducts = (
            fragment_ion_tree
            .get_hydrogen_state_candidate_delta_h_adduct_for_adduct_type(
                main_adduct_type
            )
        )

        for precursor_depth in range(self.precursor_candidate_max_depth + 1):
            for node_index in nodes_by_depth.get(precursor_depth, []):
                node_compound = self._get_fragment_compound(
                    fragment_ion_tree=fragment_ion_tree,
                    fragment_compound_by_index=fragment_compound_by_index,
                    node_index=node_index,
                )

                for neutral_delta_h_adduct in neutral_delta_h_adducts:
                    shifted_formula = neutral_delta_h_adduct.apply_to_formula(
                        node_compound.formula
                    ).normalized
                    shifted_formula = main_adduct_type.apply_to_formula(
                        shifted_formula
                    ).normalized

                    if shifted_formula == precursor_formula:
                        precursor_node_candidates.add(
                            (node_index, neutral_delta_h_adduct)
                        )

        return precursor_node_candidates

    def _build_precursor_adduct_types(
        self,
        main_adduct_type: Adduct,
        precursor_node_candidates: Set[Tuple[int, Adduct]],
    ) -> Dict[int, List[Adduct]]:
        precursor_adduct_types: Dict[int, List[Adduct]] = defaultdict(list)

        for precursor_node_index, neutral_delta_h_adduct in precursor_node_candidates:
            adduct_type = main_adduct_type.add_prefer_self(neutral_delta_h_adduct)
            precursor_adduct_types[precursor_node_index].append(adduct_type)

        return precursor_adduct_types

    def _build_precursor_fragment_pathways(
        self,
        fragment_ion_tree: FragmentIonTree,
        precursor_adduct_types: Dict[int, List[Adduct]],
    ) -> FragmentPathwayGroup:
        precursor_fragment_pathway_list: List[FragmentPathway] = []

        for precursor_node_index, adduct_types in precursor_adduct_types.items():
            pathway_items_list = build_pathway_items_for_node(
                fragment_tree=fragment_ion_tree,
                target_node_index=precursor_node_index,
                precursor_adduct_types=precursor_adduct_types,
                max_depth=self.tree_max_depth,
                precursor_candidate_max_depth=self.precursor_candidate_max_depth,
            )

            if len(pathway_items_list) == 0:
                continue

            for adduct_type in set(adduct_types):
                precursor_fragment_pathway_list.extend(
                    FragmentPathway(
                        tuple(pathway_items),
                        adduct=adduct_type,
                    )
                    for pathway_items in pathway_items_list
                )

        return FragmentPathwayGroup.from_pathways(
            precursor_fragment_pathway_list
        ).with_precursor

    def _flatten_peak_sets(
        self,
        peak_sets_tuple: Tuple[Tuple[Adduct, Tuple[float, ...]], ...],
        record_indices: Iterable[int],
    ) -> Tuple[List[float], List[Tuple[int, int]]]:
        flat_peaks_mz: List[float] = []
        flat_peak_refs: List[Tuple[int, int]] = []

        for record_index in record_indices:
            _, peaks_mz = peak_sets_tuple[record_index]

            for peak_index, peak_mz in enumerate(peaks_mz):
                flat_peaks_mz.append(peak_mz)
                flat_peak_refs.append((record_index, peak_index))

        return flat_peaks_mz, flat_peak_refs

    def _assign_formula_matches_to_peak_indices(
        self,
        assigned_peaks,
        flat_peak_refs: List[Tuple[int, int]],
        formula_candidates,
    ) -> Dict[int, Dict[int, Dict[Adduct, Set[int]]]]:
        peak_indices_by_record_node_and_adduct: Dict[
            int,
            Dict[int, Dict[Adduct, Set[int]]],
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

        for flat_peak_index, info in enumerate(assigned_peaks):
            if info["n_matches"] <= 0:
                continue

            record_index, peak_index = flat_peak_refs[flat_peak_index]

            for formula_index in info["matched_formula_indices"]:
                for node_index, adduct_type in set(
                    formula_candidates.get_candidates_by_formula_index(
                        formula_index
                    )
                ):
                    peak_indices_by_record_node_and_adduct[
                        record_index
                    ][node_index][adduct_type].add(peak_index)

        return peak_indices_by_record_node_and_adduct

    def _append_assigned_fragment_pathways(
        self,
        fragment_ion_tree: FragmentIonTree,
        peak_indices_by_record_node_and_adduct: Dict[
            int,
            Dict[int, Dict[Adduct, Set[int]]],
        ],
        context_by_record_index: Dict[int, _PrecursorAssignmentContext],
        pathway_items_cache: Dict[Tuple[Adduct, int], Tuple[Tuple[Any, ...], ...]],
        fragment_pathway_lists_by_record_and_peak: List[List[List[FragmentPathway]]],
    ) -> None:
        for record_index, peak_indices_by_node_and_adduct in (
            peak_indices_by_record_node_and_adduct.items()
        ):
            context = context_by_record_index[record_index]

            for node_index, peak_indices_by_adduct in (
                peak_indices_by_node_and_adduct.items()
            ):
                pathway_items_list = self._get_pathway_items_for_node(
                    fragment_ion_tree=fragment_ion_tree,
                    context=context,
                    node_index=node_index,
                    pathway_items_cache=pathway_items_cache,
                )

                if len(pathway_items_list) == 0:
                    continue

                for adduct_type, peak_indices in peak_indices_by_adduct.items():
                    fragment_pathways = tuple(
                        FragmentPathway(
                            tuple(pathway_items),
                            adduct=adduct_type,
                        )
                        for pathway_items in pathway_items_list
                    )

                    for peak_index in peak_indices:
                        fragment_pathway_lists_by_record_and_peak[
                            record_index
                        ][peak_index].extend(fragment_pathways)

    def _get_pathway_items_for_node(
        self,
        fragment_ion_tree: FragmentIonTree,
        context: _PrecursorAssignmentContext,
        node_index: int,
        pathway_items_cache: Dict[Tuple[Adduct, int], Tuple[Tuple[Any, ...], ...]],
    ) -> Tuple[Tuple[Any, ...], ...]:
        cache_key = (
            context.main_adduct_type,
            node_index,
        )

        pathway_items_list = pathway_items_cache.get(cache_key)

        if pathway_items_list is not None:
            return pathway_items_list

        pathway_items_list = tuple(
            tuple(pathway_items)
            for pathway_items in build_pathway_items_for_node(
                fragment_tree=fragment_ion_tree,
                target_node_index=node_index,
                precursor_adduct_types=context.precursor_adduct_types,
                max_depth=self.tree_max_depth,
                precursor_candidate_max_depth=self.precursor_candidate_max_depth,
            )
        )

        pathway_items_cache[cache_key] = pathway_items_list
        return pathway_items_list

    def to_dict(self) -> dict[str, Any]:
        builder_dict = self.fragment_ion_tree_builder.to_dict()

        # Hide builder-internal options from the external Fragmenter save format.
        builder_dict.pop("only_add_min_depth", None)
        builder_dict.pop("min_depth_only_from", None)

        return {
            "fragment_ion_tree_builder": builder_dict,
            "precursor_candidate_max_depth": self.precursor_candidate_max_depth,
            "mass_tolerance": format_mass_tolerance(self.mass_tolerance),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> Fragmenter:
        builder_dict = dict(data["fragment_ion_tree_builder"])

        # Restore FragmentTreeBuilder-internal options from the Fragmenter-level option.
        builder_dict["only_add_min_depth"] = True
        builder_dict["min_depth_only_from"] = int(
            data.get("precursor_candidate_max_depth", 0)
        )

        return cls(
            fragment_ion_tree_builder=FragmentIonTreeBuilder.from_dict(
                builder_dict
            ),
            mass_tolerance=parse_mass_tolerance(data["mass_tolerance"]),
        )

    def to_json(self, path: str | Path) -> None:
        path = Path(path)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> Fragmenter:
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def copy(self) -> Fragmenter:
        """Return a copy of this fragmenter."""
        return Fragmenter(
            fragment_ion_tree_builder=self.fragment_ion_tree_builder.copy(),
            mass_tolerance=self.mass_tolerance,
        )

@dataclass(frozen=True)
class _PrecursorAssignmentContext:
    main_adduct_type: Adduct
    precursor_adduct_types: Dict[int, List[Adduct]]
    precursor_fragment_pathways: FragmentPathwayGroup