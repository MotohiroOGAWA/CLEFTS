from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict, Set
import json
from pathlib import Path

from ...libs.mmkit.mmkit import Adduct, Compound

from ..mass.tolerance import MassTolerance, parse_mass_tolerance, format_mass_tolerance
from .tree import *
from .ion_tree import *
from .pathway import *


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
    def cleavage_pattern_set(self) -> Any:
        return self.fragment_ion_tree_builder.cleavage_pattern_set
    
    @property
    def name(self) -> str:
        return self.fragment_ion_tree_builder.name

    def build_fragment_tree(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentTree:
        return self.fragment_ion_tree_builder.build_fragment_tree(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info,
        )
    
    def build_fragment_ion_tree(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
        _include_fragment_compound_cache: bool = False,
    ) -> FragmentIonTree:
        return self.fragment_ion_tree_builder.build(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info,
            _include_fragment_compound_cache=_include_fragment_compound_cache,
        )

    def build_fragment_pathways_by_peak(
        self,
        fragment_ion_tree: FragmentIonTree,
        precursor_type: Adduct,
        peak_mz_list: Iterable[float],
    ) -> Tuple[FragmentPathwayGroup, Tuple[FragmentPathwayGroup, ...]]:

        if precursor_type.charge != 1 and precursor_type.charge != -1:
            raise ValueError(f"Only singly charged adducts are supported: {precursor_type}")
        
        empty_adduct = Adduct.parse("[M]")

        matched_adduct_flags, residual_component_adduct = Adduct.split_by_reference_adducts(
            precursor_type,
            reference_adducts=self.adduct_types,
        )

        if sum(matched_adduct_flags) == 0:
            raise ValueError(f"Adduct does not contain any of the supported adduct types({', '.join(str(adduct) for adduct in self.adduct_types)}): {precursor_type}")
        if sum(matched_adduct_flags) > 1:
            raise ValueError(f"Adduct contains multiple supported adduct types({', '.join(str(adduct) for adduct in self.adduct_types)}): {precursor_type}")
        main_adduct_type = self.adduct_types[matched_adduct_flags.index(True)]

        adduct_rule = self.adduct_rule_set.get_rule_by_adduct_type(main_adduct_type)


        fragment_compound_by_index: Dict[int, Compound] = {}
        if getattr(fragment_ion_tree, "_fragment_compound_by_index", None) is not None:
            fragment_compound_by_index = fragment_ion_tree._fragment_compound_by_index
        def _get_fragment_compound(idx: int) -> Compound:
            c = fragment_compound_by_index.get(idx)

            if c is not None:
                return c

            n = fragment_ion_tree.get_node(idx)
            c = Compound.from_smiles(n.smiles)
            fragment_compound_by_index[idx] = c
            return c


        precusor_compound = _get_fragment_compound(0)
        precursor_formula = precursor_type.apply_to_formula(precusor_compound.formula).normalized

        precursor_node_candidates: Set[Tuple[int, Adduct]] = set()
        nodes_by_depth = fragment_ion_tree.get_nodes_by_depth()
        neutral_delta_h_adducts = fragment_ion_tree.get_hydrogen_state_candidate_delta_h_adduct_for_adduct_type(main_adduct_type)

        for precursor_depth in range(self.precursor_candidate_max_depth + 1):
            for node_index in nodes_by_depth.get(precursor_depth, []):
                node = fragment_ion_tree.get_node(node_index)
                node_compound = _get_fragment_compound(node_index)

                for neutral_delta_h_adduct in neutral_delta_h_adducts:
                    shifted_formula = neutral_delta_h_adduct.apply_to_formula(node_compound.formula).normalized
                    shifted_formula = main_adduct_type.apply_to_formula(shifted_formula).normalized

                    if shifted_formula == precursor_formula:
                        precursor_node_candidates.add((node_index, neutral_delta_h_adduct))
        
        precursor_node_indices = set({node_index for node_index, _ in precursor_node_candidates})
        pass





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