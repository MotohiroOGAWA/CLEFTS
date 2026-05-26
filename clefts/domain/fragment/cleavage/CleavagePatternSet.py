from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple, Union

from ....libs.mmkit.mmkit import Compound
from .CleavagePattern import CleavagePattern
from .CleavageResult import CleavageResult


class CleavagePatternSet:
    """
    Order-independent collection of cleavage patterns.

    Pattern IDs are assigned after sorting by each pattern key, so the same
    combination of patterns receives the same IDs even if constructor order
    differs. Duplicate pattern keys are treated as the same cleavage pattern.
    """

    def __init__(self, patterns: Iterable[CleavagePattern], name: str = ""):
        self._name = name
        unique_patterns = {}
        for pattern in patterns:
            assert isinstance(pattern, CleavagePattern), "patterns must contain only CleavagePattern instances."
            unique_patterns.setdefault(pattern.key(), pattern.copy())

        self._patterns = tuple(
            unique_patterns[key] for key in sorted(unique_patterns)
        )
        self._id_to_pattern = dict(enumerate(self._patterns))
        self._pattern_key_to_id = {
            pattern.key(): pattern_id
            for pattern_id, pattern in self._id_to_pattern.items()
        }

    def __repr__(self) -> str:
        return f"CleavagePatternSet(name='{self.name}', n_patterns={len(self)})"

    def __iter__(self) -> Iterator[CleavagePattern]:
        return iter(self._patterns)

    def __len__(self) -> int:
        return len(self._patterns)

    def __contains__(self, pattern: CleavagePattern) -> bool:
        if not isinstance(pattern, CleavagePattern):
            return False
        return pattern.key() in self._pattern_key_to_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CleavagePatternSet):
            return False
        return self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    @property
    def name(self) -> str:
        return self._name

    @property
    def patterns(self) -> Tuple[CleavagePattern, ...]:
        return self._patterns

    @property
    def ids(self) -> Tuple[int, ...]:
        return tuple(self._id_to_pattern.keys())

    def key(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(pattern.key() for pattern in self._patterns)

    def equals(self, other: CleavagePatternSet, include_name: bool = False) -> bool:
        if not isinstance(other, CleavagePatternSet):
            return False
        if include_name and self.name != other.name:
            return False
        return self == other

    def get_id(self, pattern: CleavagePattern) -> int:
        assert isinstance(pattern, CleavagePattern), "pattern must be a CleavagePattern instance."
        try:
            return self._pattern_key_to_id[pattern.key()]
        except KeyError as exc:
            raise KeyError(f"CleavagePattern is not in this set: {pattern}") from exc

    def get_pattern(self, cleavage_id: int) -> CleavagePattern:
        try:
            return self._id_to_pattern[cleavage_id]
        except KeyError as exc:
            raise KeyError(f"Invalid cleavage_id: {cleavage_id}") from exc

    def fragment_by_id(self, cleavage_id: int, compound: Compound) -> Union[CleavageResult, None]:
        return self.get_pattern(cleavage_id).fragment(compound)

    def fragment_all(self, compound: Compound) -> Tuple[CleavageResult, ...]:
        results = []
        for pattern in self._patterns:
            result = pattern.fragment(compound)
            if result is not None and len(result.products) > 0:
                results.append(result)
        return tuple(results)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "patterns": [pattern.to_dict() for pattern in self._patterns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CleavagePatternSet":
        patterns = [
            CleavagePattern.from_dict(pattern_data)
            for pattern_data in data.get("patterns", [])
        ]
        return cls(patterns=patterns, name=str(data.get("name", "")))

    def copy(self) -> "CleavagePatternSet":
        return self.__class__(patterns=self._patterns, name=self.name)
