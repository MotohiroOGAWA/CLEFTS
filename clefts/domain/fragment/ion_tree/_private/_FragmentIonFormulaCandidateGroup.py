from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from .....libs.mmkit.mmkit import Adduct, Compound, Formula

if TYPE_CHECKING:
    from ..FragmentIonTree import FragmentIonTree


@dataclass(frozen=True)
class _FragmentIonFormulaCandidateGroup:
    """Formula-indexed node/adduct candidates for one main adduct type.

    This class stores candidates for one main adduct type.

    For a formula_index:

        start = indptr[formula_index]
        end = indptr[formula_index + 1]

    Then candidates are:

        node_indices[start:end]
        candidate_adduct_indices[start:end]

    and the candidate adduct is:

        candidate_adducts[candidate_adduct_indices[i]]
    """

    adduct_type: Adduct
    formulas: np.ndarray

    indptr: np.ndarray
    node_indices: np.ndarray
    candidate_adduct_indices: np.ndarray

    candidate_adducts: Tuple[Adduct, ...]

    hydrogen_candidate_indices: np.ndarray
    shift_rule_indices: np.ndarray

    def __post_init__(self) -> None:
        adduct_type = self.adduct_type

        formulas = np.asarray(self.formulas, dtype=object)
        indptr = np.asarray(self.indptr, dtype=np.int64)
        node_indices = np.asarray(self.node_indices, dtype=np.int64)
        candidate_adduct_indices = np.asarray(
            self.candidate_adduct_indices,
            dtype=np.int64,
        )
        hydrogen_candidate_indices = np.asarray(
            self.hydrogen_candidate_indices,
            dtype=np.int64,
        )
        shift_rule_indices = np.asarray(
            self.shift_rule_indices,
            dtype=np.int64,
        )

        candidate_adducts = tuple(adduct for adduct in self.candidate_adducts)

        if formulas.ndim != 1:
            raise ValueError("formulas must be a 1D array.")

        if indptr.ndim != 1:
            raise ValueError("indptr must be a 1D array.")

        if len(indptr) != len(formulas) + 1:
            raise ValueError("indptr length must be len(formulas) + 1.")

        if indptr[0] != 0:
            raise ValueError("indptr[0] must be 0.")

        if np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("indptr must be non-decreasing.")

        num_candidates = int(indptr[-1])

        if len(node_indices) != num_candidates:
            raise ValueError("node_indices length must match indptr[-1].")

        if len(candidate_adduct_indices) != num_candidates:
            raise ValueError(
                "candidate_adduct_indices length must match indptr[-1]."
            )

        if len(hydrogen_candidate_indices) != num_candidates:
            raise ValueError(
                "hydrogen_candidate_indices length must match indptr[-1]."
            )

        if len(shift_rule_indices) != num_candidates:
            raise ValueError(
                "shift_rule_indices length must match indptr[-1]."
            )

        if np.any(node_indices < 0):
            raise ValueError("node_indices must be non-negative.")

        if np.any(candidate_adduct_indices < 0):
            raise ValueError(
                "candidate_adduct_indices must be non-negative."
            )

        if (
            len(candidate_adduct_indices) > 0
            and np.any(candidate_adduct_indices >= len(candidate_adducts))
        ):
            raise ValueError(
                "candidate_adduct_indices contains out-of-range values."
            )

        object.__setattr__(self, "adduct_type", adduct_type)
        object.__setattr__(self, "formulas", formulas)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "node_indices", node_indices)
        object.__setattr__(
            self,
            "candidate_adduct_indices",
            candidate_adduct_indices,
        )
        object.__setattr__(self, "candidate_adducts", candidate_adducts)
        object.__setattr__(
            self,
            "hydrogen_candidate_indices",
            hydrogen_candidate_indices,
        )
        object.__setattr__(self, "shift_rule_indices", shift_rule_indices)

    @classmethod
    def from_fragment_ion_tree(
        cls,
        *,
        fragment_ion_tree: FragmentIonTree,
        adduct_rule_index: int,
    ) -> _FragmentIonFormulaCandidateGroup:
        """Build a formula candidate group for one main adduct rule."""

        fragment_ion_tree._validate_adduct_rule_index(adduct_rule_index)

        adduct_rule = (
            fragment_ion_tree
            .fragment_ion_adduct_rule_set
            .adduct_rules[adduct_rule_index]
        )

        delta_h_values = (
            fragment_ion_tree
            .get_hydrogen_state_candidate_delta_h_for_adduct_rule(
                adduct_rule_index
            )
        )

        shift_rule_indices = (
            fragment_ion_tree
            .ion_shift_candidate_store
            .get_shift_rule_indices_for_adduct_rule(adduct_rule_index)
        )

        formula_to_candidates: dict[
            Formula,
            list[tuple[int, int, int, int]],
        ] = {}

        candidate_adducts: list[Adduct] = []
        candidate_adduct_key_to_index: dict[str, int] = {}


        fragment_compound_by_index: dict[int, Compound] = {}
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

        for node_index in range(fragment_ion_tree.num_nodes):
            compound = _get_fragment_compound(node_index)
            base_formula = compound.formula.normalized

            for hydrogen_candidate_index, delta_h in enumerate(delta_h_values):
                hydrogen_delta_adduct = Adduct.from_dict(
                    {"H": int(delta_h)}
                )

                for shift_rule_index in shift_rule_indices:
                    shift_rule_index = int(shift_rule_index)

                    ion_shift_adduct = (
                        fragment_ion_tree
                        .ion_shift_candidate_store
                        .get_ion_shift_adduct_type(shift_rule_index)
                    )

                    formula = cls._calc_formula_candidate(
                        base_formula=base_formula,
                        hydrogen_delta_adduct=hydrogen_delta_adduct,
                        ion_shift_adduct=ion_shift_adduct,
                    )

                    if not formula.is_nonnegative:
                        continue

                    candidate_adduct = cls._make_candidate_adduct(
                        hydrogen_delta_adduct=hydrogen_delta_adduct,
                        ion_shift_adduct=ion_shift_adduct,
                    )

                    candidate_adduct_index = cls._register_adduct(
                        adduct=candidate_adduct,
                        adducts=candidate_adducts,
                        adduct_key_to_index=candidate_adduct_key_to_index,
                    )

                    formula_key = cls._formula_key(formula)

                    formula_to_candidates.setdefault(
                        formula_key,
                        [],
                    ).append(
                        (
                            int(node_index),
                            int(candidate_adduct_index),
                            int(hydrogen_candidate_index),
                            int(shift_rule_index),
                        )
                    )

        formulas: list[Formula] = []
        indptr: list[int] = [0]

        node_indices: list[int] = []
        candidate_adduct_indices: list[int] = []
        hydrogen_candidate_indices: list[int] = []
        collected_shift_rule_indices: list[int] = []

        for formula_key in sorted(formula_to_candidates, key=lambda formula: formula.exact_mass,):
            candidates = formula_to_candidates[formula_key]

            formula = formula_key
            formulas.append(formula)

            for (
                node_index,
                candidate_adduct_index,
                hydrogen_candidate_index,
                shift_rule_index,
            ) in candidates:
                node_indices.append(node_index)
                candidate_adduct_indices.append(candidate_adduct_index)
                hydrogen_candidate_indices.append(hydrogen_candidate_index)
                collected_shift_rule_indices.append(shift_rule_index)

            indptr.append(len(node_indices))

        return cls(
            adduct_type=adduct_rule.adduct_type,
            formulas=np.asarray(formulas, dtype=object),
            indptr=np.asarray(indptr, dtype=np.int64),
            node_indices=np.asarray(node_indices, dtype=np.int64),
            candidate_adduct_indices=np.asarray(
                candidate_adduct_indices,
                dtype=np.int64,
            ),
            candidate_adducts=tuple(candidate_adducts),
            hydrogen_candidate_indices=np.asarray(
                hydrogen_candidate_indices,
                dtype=np.int64,
            ),
            shift_rule_indices=np.asarray(
                collected_shift_rule_indices,
                dtype=np.int64,
            ),
        )

    @classmethod
    def _calc_formula_candidate(
        cls,
        *,
        base_formula: Formula,
        hydrogen_delta_adduct: Adduct,
        ion_shift_adduct: Adduct,
    ) -> Formula:
        formula = hydrogen_delta_adduct.apply_to_formula(base_formula)
        formula = ion_shift_adduct.apply_to_formula(formula)

        return formula.normalized

    @staticmethod
    def _make_candidate_adduct(
        *,
        hydrogen_delta_adduct: Adduct,
        ion_shift_adduct: Adduct,
    ) -> Adduct:
        """Return candidate adduct stored in this group."""
        return ion_shift_adduct.add_prefer_self(hydrogen_delta_adduct)

    @staticmethod
    def _formula_key(formula: Formula) -> Formula:
        return formula.normalized

    @staticmethod
    def _register_adduct(
        *,
        adduct: Adduct,
        adducts: list[Adduct],
        adduct_key_to_index: dict[str, int],
    ) -> int:
        key = str(adduct)

        existing_index = adduct_key_to_index.get(key)

        if existing_index is not None:
            return existing_index

        adduct_index = len(adducts)
        adducts.append(adduct)
        adduct_key_to_index[key] = adduct_index

        return adduct_index

    @property
    def num_formulas(self) -> int:
        return int(self.formulas.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.node_indices.shape[0])

    def get_formula_index(
        self,
        formula: Formula,
    ) -> int:
        formula_key = self._formula_key(formula)

        for formula_index, current_formula in enumerate(self.formulas):
            if self._formula_key(current_formula) == formula_key:
                return formula_index

        raise KeyError(f"Unknown formula: {formula_key}")

    def get_candidate_range(
        self,
        formula_index: int,
    ) -> slice:
        self._validate_formula_index(formula_index)

        start = int(self.indptr[formula_index])
        end = int(self.indptr[formula_index + 1])

        return slice(start, end)

    def get_node_indices_by_formula_index(
        self,
        formula_index: int,
    ) -> np.ndarray:
        candidate_range = self.get_candidate_range(formula_index)
        return self.node_indices[candidate_range]

    def get_candidate_adducts_by_formula_index(
        self,
        formula_index: int,
    ) -> Tuple[Adduct, ...]:
        candidate_range = self.get_candidate_range(formula_index)

        return tuple(
            self.candidate_adducts[int(candidate_adduct_index)]
            for candidate_adduct_index in (
                self.candidate_adduct_indices[candidate_range]
            )
        )

    def get_candidates_by_formula_index(
        self,
        formula_index: int,
    ) -> tuple[tuple[int, Adduct], ...]:
        candidate_range = self.get_candidate_range(formula_index)

        return tuple(
            (
                int(node_index),
                self.candidate_adducts[int(candidate_adduct_index)],
            )
            for node_index, candidate_adduct_index in zip(
                self.node_indices[candidate_range],
                self.candidate_adduct_indices[candidate_range],
            )
        )

    def get_candidates_by_formula(
        self,
        formula: object,
    ) -> tuple[tuple[int, Adduct], ...]:
        formula_index = self.get_formula_index(formula)

        return self.get_candidates_by_formula_index(formula_index)

    def copy(self) -> _FragmentIonFormulaCandidateGroup:
        return _FragmentIonFormulaCandidateGroup(
            adduct_type=self.adduct_type,
            formulas=self.formulas.copy(),
            indptr=self.indptr.copy(),
            node_indices=self.node_indices.copy(),
            candidate_adduct_indices=self.candidate_adduct_indices.copy(),
            candidate_adducts=tuple(self.candidate_adducts),
            hydrogen_candidate_indices=self.hydrogen_candidate_indices.copy(),
            shift_rule_indices=self.shift_rule_indices.copy(),
        )

    def _validate_formula_index(
        self,
        formula_index: int,
    ) -> None:
        if not isinstance(formula_index, int):
            raise TypeError("formula_index must be an int.")

        if not 0 <= formula_index < self.num_formulas:
            raise IndexError(
                "formula_index must be in "
                f"[0, {self.num_formulas}). "
                f"Got {formula_index}."
            )