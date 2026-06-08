from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

from clefts.libs.mmkit.mmkit import Compound

from .CleavagePattern import _CleavagePattern, _CleavageResult

@dataclass(frozen=True)
class CleavagePattern(_CleavagePattern):
    """Public cleavage pattern with stable ID."""

    pattern_id: int

    @classmethod
    def from_base(
        cls,
        pattern_id: int,
        base: _CleavagePattern,
    ) -> CleavagePattern:
        """Create public CleavagePattern from internal _CleavagePattern."""
        return cls(
            pattern_id=pattern_id,
            name=base.name,
            reactant_smarts=base.reactant_smarts,
            reactant_query=base.reactant_query,
            cleavage_reactions=base.cleavage_reactions,
        )

    def fragment(
        self,
        compound: Compound,
    ) -> CleavageResult | None:
        """Apply this pattern and return result with pattern_id."""
        result = super().fragment(compound)

        if result is None:
            return None

        if len(result.products) == 0:
            return None

        return CleavageResult.from_base(
            pattern_id=self.pattern_id,
            cleavage=self,
            base=result,
        )

@dataclass(frozen=True)
class CleavageResult(_CleavageResult):
    """Public cleavage result with pattern ID."""

    pattern_id: int

    @classmethod
    def from_base(
        cls,
        pattern_id: int,
        cleavage: object,
        base: _CleavageResult,
    ) -> CleavageResult:
        return cls(
            pattern_id=pattern_id,
            cleavage=cleavage,
            reactant_smiles=base.reactant_smiles,
            products=base.products,
        )

@dataclass(frozen=True)
class CleavagePatternSet:
    """A stable set of cleavage patterns.

    This class is intentionally a plain dataclass.
    It does not normalize patterns in __init__.

    Use from_patterns(...) to:
    - remove duplicate patterns
    - sort patterns by identity
    - assign stable pattern IDs
    """

    name: str
    patterns: Tuple[CleavagePattern, ...]

    @classmethod
    def from_patterns(
        cls,
        patterns: Iterable[_CleavagePattern],
        name: str = "",
    ) -> CleavagePatternSet:
        """Create a normalized CleavagePatternSet.

        Parameters
        ----------
        patterns:
            Internal cleavage patterns without IDs.
        name:
            Human-readable name of this pattern set.

        Returns
        -------
        CleavagePatternSet
            Pattern set with stable pattern IDs.
        """
        unique_patterns: dict[
            str,
            _CleavagePattern,
        ] = {}

        for pattern in patterns:
            if not isinstance(pattern, _CleavagePattern):
                raise TypeError(
                    "patterns must contain only _CleavagePattern instances."
                )

            unique_patterns.setdefault(
                pattern.key,
                pattern,
            )

        sorted_patterns = tuple(
            unique_patterns[key]
            for key in sorted(unique_patterns)
        )

        public_patterns = tuple(
            CleavagePattern.from_base(
                pattern_id=pattern_id,
                base=pattern,
            )
            for pattern_id, pattern in enumerate(sorted_patterns)
        )

        return cls(
            name=name,
            patterns=public_patterns,
        )

    def __iter__(self) -> Iterator[CleavagePattern]:
        return iter(self.patterns)

    def __len__(self) -> int:
        return len(self.patterns)

    def __contains__(self, pattern: object) -> bool:
        if not isinstance(pattern, _CleavagePattern):
            return False

        return pattern.key in self._pattern_id_by_identity

    @property
    def pattern_ids(self) -> Tuple[int, ...]:
        """Pattern IDs in this set."""
        return tuple(pattern.pattern_id for pattern in self.patterns)

    @property
    def _pattern_by_id(self) -> dict[int, CleavagePattern]:
        return {
            pattern.pattern_id: pattern
            for pattern in self.patterns
        }

    @property
    def _pattern_id_by_identity(self) -> dict[tuple[str, tuple[str, ...]], int]:
        return {
            pattern.key: pattern.pattern_id
            for pattern in self.patterns
        }

    def identity(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the identity of this pattern set.

        The set name is intentionally not included.
        """
        return tuple(
            pattern.key
            for pattern in self.patterns
        )

    def get_pattern(
        self,
        pattern_id: int,
    ) -> CleavagePattern:
        """Get a cleavage pattern by stable pattern ID."""
        try:
            return self._pattern_by_id[pattern_id]
        except KeyError as exc:
            raise KeyError(f"Invalid cleavage pattern ID: {pattern_id}") from exc

    def get_pattern_id(
        self,
        pattern: _CleavagePattern,
    ) -> int:
        """Get the stable ID of an equivalent cleavage pattern."""
        try:
            return self._pattern_id_by_identity[pattern.key]
        except KeyError as exc:
            raise KeyError(f"CleavagePattern is not in this set: {pattern}") from exc

    def fragment_by_id(
        self,
        pattern_id: int,
        compound: Compound,
    ) -> CleavageResult | None:
        """Apply one cleavage pattern to a compound."""
        pattern = self.get_pattern(pattern_id)
        return pattern.fragment(compound)

    def fragment_all(
        self,
        compound: Compound,
    ) -> Tuple[CleavageResult, ...]:
        """Apply all cleavage patterns to a compound."""
        results: list[CleavageResult] = []

        for pattern in self.patterns:
            result = pattern.fragment(compound)

            if result is None:
                continue

            if len(result.products) == 0:
                continue

            results.append(result)

        return tuple(results)

    def to_dict(self) -> dict[str, object]:
        """Serialize this pattern set.

        RDKit compiled objects are not serialized here.
        Use this only for metadata-level export.
        """
        return {
            "name": self.name,
            "patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "name": pattern.name,
                    "reactant_smarts": pattern.reactant_smarts,
                    "products": [
                        {
                            "name": product.name,
                            "smarts": product.smarts,
                        }
                        for product in pattern.products
                    ],
                }
                for pattern in self.patterns
            ],
        }
    
    def copy(self) -> CleavagePatternSet:
        """Return a copy of this pattern set."""
        return CleavagePatternSet(
            name=self.name,
            patterns=tuple(pattern.copy() for pattern in self.patterns),
        )