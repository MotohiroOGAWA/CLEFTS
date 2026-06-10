from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .....libs.mmkit.mmkit import Adduct
from ..FragmentIonAdductRuleSet import FragmentIonAdductRuleSet


@dataclass(frozen=True)
class _FragmentHydrogenStateCandidateStore:
    """Hydrogen state candidates that can be applied to fragment nodes.

    The candidates are generated from each adduct rule's radical and
    unsaturation settings.

    This class does not store states for each concrete fragment node.
    Instead, it stores candidates per adduct type using an indptr layout.

    States are stored as [unsaturation, radical].

    Hydrogen delta is calculated as:

        delta_h = -2 * unsaturation - radical
    """

    adduct_types: Tuple[Adduct, ...]
    declared_states: np.ndarray
    candidate_states: np.ndarray
    adduct_state_indptr: np.ndarray

    def __post_init__(self) -> None:
        adduct_types = tuple(
            self._normalize_adduct_type(adduct_type)
            for adduct_type in self.adduct_types
        )

        declared_states = np.asarray(
            self.declared_states,
            dtype=np.int16,
        )

        candidate_states = np.asarray(
            self.candidate_states,
            dtype=np.int16,
        )

        adduct_state_indptr = np.asarray(
            self.adduct_state_indptr,
            dtype=np.int64,
        )

        if len(set(str(adduct_type) for adduct_type in adduct_types)) != len(
            adduct_types
        ):
            raise ValueError("adduct_types must be unique.")

        self._validate_states(
            declared_states,
            name="declared_states",
        )

        self._validate_states(
            candidate_states,
            name="candidate_states",
        )

        if len(declared_states) != len(adduct_types):
            raise ValueError(
                "declared_states must have one row per adduct type."
            )

        if adduct_state_indptr.ndim != 1:
            raise ValueError("adduct_state_indptr must be a 1D array.")

        if len(adduct_state_indptr) != len(adduct_types) + 1:
            raise ValueError(
                "adduct_state_indptr must have length "
                "num_adduct_types + 1."
            )

        if len(adduct_state_indptr) == 0:
            raise ValueError("adduct_state_indptr must not be empty.")

        if adduct_state_indptr[0] != 0:
            raise ValueError("adduct_state_indptr[0] must be 0.")

        if np.any(adduct_state_indptr[1:] < adduct_state_indptr[:-1]):
            raise ValueError("adduct_state_indptr must be non-decreasing.")

        if adduct_state_indptr[-1] != len(candidate_states):
            raise ValueError(
                "adduct_state_indptr[-1] must equal number of "
                "candidate states."
            )

        self._validate_candidate_state_ranges(
            declared_states=declared_states,
            candidate_states=candidate_states,
            adduct_state_indptr=adduct_state_indptr,
        )

        object.__setattr__(self, "adduct_types", adduct_types)
        object.__setattr__(self, "declared_states", declared_states)
        object.__setattr__(self, "candidate_states", candidate_states)
        object.__setattr__(self, "adduct_state_indptr", adduct_state_indptr)

    @staticmethod
    def _normalize_adduct_type(
        adduct_type: Adduct | str,
    ) -> Adduct:
        if isinstance(adduct_type, Adduct):
            return adduct_type

        if isinstance(adduct_type, str):
            return Adduct.parse(adduct_type)

        raise TypeError(
            "adduct_type must be an Adduct or str. "
            f"Got {type(adduct_type).__name__}."
        )

    @staticmethod
    def _validate_states(
        states: np.ndarray,
        *,
        name: str,
    ) -> None:
        if states.ndim != 2:
            raise ValueError(f"{name} must be a 2D array.")

        if states.shape[1] != 2:
            raise ValueError(f"{name} must have shape (num_states, 2).")

        unsaturations = states[:, 0]
        radicals = states[:, 1]

        if np.any(unsaturations < 0):
            raise ValueError(
                f"{name} unsaturation values must be non-negative."
            )

        if not np.all((radicals == 0) | (radicals == 1)):
            raise ValueError(f"{name} radical values must be 0 or 1.")

    @staticmethod
    def _validate_candidate_state_ranges(
        *,
        declared_states: np.ndarray,
        candidate_states: np.ndarray,
        adduct_state_indptr: np.ndarray,
    ) -> None:
        for adduct_index, declared_state in enumerate(declared_states):
            start = int(adduct_state_indptr[adduct_index])
            end = int(adduct_state_indptr[adduct_index + 1])

            if start == end:
                raise ValueError(
                    "Each adduct type must have at least one "
                    "candidate state."
                )

            expected_states = (
                _FragmentHydrogenStateCandidateStore
                ._build_candidate_states(
                    radical=int(declared_state[1]),
                    unsaturation=int(declared_state[0]),
                )
            )

            actual_states = candidate_states[start:end]

            if not np.array_equal(actual_states, expected_states):
                raise ValueError(
                    "candidate_states range does not match declared_states. "
                    f"adduct_index={adduct_index}"
                )

    @staticmethod
    def _normalize_radical(radical: bool | int) -> int:
        if isinstance(radical, bool):
            return int(radical)

        if radical in (0, 1):
            return int(radical)

        raise ValueError("radical must be bool, 0, or 1.")

    @staticmethod
    def _normalize_unsaturation(unsaturation: int) -> int:
        unsaturation = int(unsaturation)

        if unsaturation < 0:
            raise ValueError("unsaturation must be non-negative.")

        return unsaturation

    @classmethod
    def _build_candidate_states(
        cls,
        *,
        radical: bool | int,
        unsaturation: int,
    ) -> np.ndarray:
        radical_value = cls._normalize_radical(radical)
        unsaturation_value = cls._normalize_unsaturation(unsaturation)

        states: list[tuple[int, int]] = []

        for current_unsaturation in range(unsaturation_value + 1):
            states.append((current_unsaturation, 0))

        if radical_value == 1:
            states.append((unsaturation_value, 1))

        return np.asarray(states, dtype=np.int16)

    @classmethod
    def from_adduct_rule_set(
        cls,
        adduct_rule_set: FragmentIonAdductRuleSet,
    ) -> _FragmentHydrogenStateCandidateStore:
        """Create candidate states from a fragment ion adduct rule set."""

        adduct_types: list[Adduct] = []
        declared_states: list[tuple[int, int]] = []
        candidate_state_blocks: list[np.ndarray] = []
        indptr: list[int] = [0]

        for adduct_rule in adduct_rule_set.adduct_rules:
            radical_value = cls._normalize_radical(adduct_rule.radical)
            unsaturation_value = cls._normalize_unsaturation(
                adduct_rule.unsaturation
            )

            adduct_types.append(adduct_rule.adduct_type)
            declared_states.append((unsaturation_value, radical_value))

            candidate_states = cls._build_candidate_states(
                radical=radical_value,
                unsaturation=unsaturation_value,
            )

            candidate_state_blocks.append(candidate_states)
            indptr.append(indptr[-1] + len(candidate_states))

        if declared_states:
            declared_states_array = np.asarray(
                declared_states,
                dtype=np.int16,
            )
        else:
            declared_states_array = np.empty((0, 2), dtype=np.int16)

        if candidate_state_blocks:
            candidate_states_array = np.concatenate(
                candidate_state_blocks,
                axis=0,
            )
        else:
            candidate_states_array = np.empty((0, 2), dtype=np.int16)

        return cls(
            adduct_types=tuple(adduct_types),
            declared_states=declared_states_array,
            candidate_states=candidate_states_array,
            adduct_state_indptr=np.asarray(indptr, dtype=np.int64),
        )

    @classmethod
    def empty(cls) -> _FragmentHydrogenStateCandidateStore:
        """Create an empty store with no adduct types."""

        return cls(
            adduct_types=tuple(),
            declared_states=np.empty((0, 2), dtype=np.int16),
            candidate_states=np.empty((0, 2), dtype=np.int16),
            adduct_state_indptr=np.zeros(1, dtype=np.int64),
        )

    @property
    def num_adduct_types(self) -> int:
        return len(self.adduct_types)

    @property
    def num_candidate_states(self) -> int:
        return len(self.candidate_states)

    @property
    def declared_unsaturations(self) -> np.ndarray:
        return self.declared_states[:, 0]

    @property
    def declared_radicals(self) -> np.ndarray:
        return self.declared_states[:, 1].astype(bool)

    @property
    def candidate_unsaturations(self) -> np.ndarray:
        return self.candidate_states[:, 0]

    @property
    def candidate_radicals(self) -> np.ndarray:
        return self.candidate_states[:, 1].astype(bool)

    @property
    def declared_delta_h(self) -> np.ndarray:
        return -2 * self.declared_states[:, 0] - self.declared_states[:, 1]

    @property
    def candidate_delta_h(self) -> np.ndarray:
        return -2 * self.candidate_states[:, 0] - self.candidate_states[:, 1]

    def get_adduct_index(
        self,
        adduct_type: Adduct | str,
    ) -> int:
        key = str(self._normalize_adduct_type(adduct_type))

        for index, current_adduct_type in enumerate(self.adduct_types):
            if str(current_adduct_type) == key:
                return index

        raise KeyError(f"Unknown adduct type: {key}")

    def get_adduct_type(
        self,
        adduct_index: int,
    ) -> Adduct:
        self._validate_adduct_index(adduct_index)
        return self.adduct_types[adduct_index]

    def get_candidate_state_range(
        self,
        adduct_index: int,
    ) -> slice:
        self._validate_adduct_index(adduct_index)

        start = int(self.adduct_state_indptr[adduct_index])
        end = int(self.adduct_state_indptr[adduct_index + 1])

        return slice(start, end)

    def get_declared_state(
        self,
        adduct_index: int,
    ) -> np.ndarray:
        self._validate_adduct_index(adduct_index)
        return self.declared_states[adduct_index]

    def get_candidate_states(
        self,
        adduct_index: int,
    ) -> np.ndarray:
        state_range = self.get_candidate_state_range(adduct_index)
        return self.candidate_states[state_range]

    def get_candidate_delta_h(
        self,
        adduct_index: int,
    ) -> np.ndarray:
        state_range = self.get_candidate_state_range(adduct_index)
        return self.candidate_delta_h[state_range]

    def get_candidate_states_by_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> np.ndarray:
        adduct_index = self.get_adduct_index(adduct_type)
        return self.get_candidate_states(adduct_index)

    def get_candidate_delta_h_by_adduct_type(
        self,
        adduct_type: Adduct | str,
    ) -> np.ndarray:
        adduct_index = self.get_adduct_index(adduct_type)
        return self.get_candidate_delta_h(adduct_index)

    def copy(self) -> _FragmentHydrogenStateCandidateStore:
        return _FragmentHydrogenStateCandidateStore(
            adduct_types=tuple(self.adduct_types),
            declared_states=self.declared_states.copy(),
            candidate_states=self.candidate_states.copy(),
            adduct_state_indptr=self.adduct_state_indptr.copy(),
        )

    def _validate_adduct_index(
        self,
        adduct_index: int,
    ) -> None:
        if not isinstance(adduct_index, int):
            raise TypeError("adduct_index must be an int.")

        if not 0 <= adduct_index < self.num_adduct_types:
            raise IndexError(
                "adduct_index must be in "
                f"[0, {self.num_adduct_types}). "
                f"Got {adduct_index}."
            )