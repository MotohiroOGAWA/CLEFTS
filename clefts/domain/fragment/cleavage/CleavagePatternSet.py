from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple, Union

from ....libs.mmkit.mmkit import Compound
from .CleavagePattern import CleavagePattern
from .CleavageResult import CleavageResult


class CleavagePatternSet:
    """Manage a stable collection of cleavage patterns.

    ``CleavagePatternSet`` groups multiple :class:`CleavagePattern` objects and
    assigns each pattern a stable integer ID. IDs are assigned after sorting by
    each pattern's identity key, so the same combination of cleavage patterns
    receives the same IDs even when the constructor order differs.

    Duplicate patterns, as defined by ``CleavagePattern.key()``, are collapsed
    into a single pattern. This makes the class suitable for reproducible
    fragment-tree generation and serialized configuration files.
    """

    def __init__(self, patterns: Iterable[CleavagePattern], name: str = ""):
        """Create a pattern set.

        Parameters
        ----------
        patterns : iterable of CleavagePattern
            Cleavage patterns to include in the set. Input order does not affect
            the assigned IDs.
        name : str, optional
            Human-readable name for the pattern set. The name is not part of the
            default equality comparison.
        """
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
        """Iterate over patterns in stable ID order."""
        return iter(self._patterns)

    def __len__(self) -> int:
        """Return the number of unique cleavage patterns in the set."""
        return len(self._patterns)

    def __contains__(self, pattern: CleavagePattern) -> bool:
        """Return whether an equivalent pattern is included in this set."""
        if not isinstance(pattern, CleavagePattern):
            return False
        return pattern.key() in self._pattern_key_to_id

    def __eq__(self, other: object) -> bool:
        """Return whether two sets contain the same cleavage identities."""
        if not isinstance(other, CleavagePatternSet):
            return False
        return self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    @property
    def name(self) -> str:
        """Human-readable name of the pattern set."""
        return self._name

    @property
    def patterns(self) -> Tuple[CleavagePattern, ...]:
        """Cleavage patterns sorted in stable ID order."""
        return self._patterns

    @property
    def ids(self) -> Tuple[int, ...]:
        """Stable integer IDs assigned to patterns in this set."""
        return tuple(self._id_to_pattern.keys())

    def key(self) -> Tuple[Tuple[str, str], ...]:
        """Return the order-independent identity key for the full set."""
        return tuple(pattern.key() for pattern in self._patterns)

    def equals(self, other: CleavagePatternSet, include_name: bool = False) -> bool:
        """Compare this set with another pattern set.

        Parameters
        ----------
        other : CleavagePatternSet
            Pattern set to compare with this instance.
        include_name : bool, optional
            If ``True``, compare ``name`` in addition to pattern identities.

        Returns
        -------
        bool
            ``True`` when the two sets are equivalent under the selected
            comparison rule.
        """
        if not isinstance(other, CleavagePatternSet):
            return False
        if include_name and self.name != other.name:
            return False
        return self == other

    def get_id(self, pattern: CleavagePattern) -> int:
        """Return the stable ID assigned to a cleavage pattern.

        Parameters
        ----------
        pattern : CleavagePattern
            Pattern whose equivalent entry should be looked up.

        Returns
        -------
        int
            Stable integer ID for the pattern.

        Raises
        ------
        KeyError
            If no equivalent pattern is present in the set.
        """
        assert isinstance(pattern, CleavagePattern), "pattern must be a CleavagePattern instance."
        try:
            return self._pattern_key_to_id[pattern.key()]
        except KeyError as exc:
            raise KeyError(f"CleavagePattern is not in this set: {pattern}") from exc

    def get_pattern(self, cleavage_id: int) -> CleavagePattern:
        """Return the pattern assigned to an ID.

        Parameters
        ----------
        cleavage_id : int
            Stable cleavage pattern ID.

        Returns
        -------
        CleavagePattern
            Pattern associated with ``cleavage_id``.

        Raises
        ------
        KeyError
            If ``cleavage_id`` is not defined in this set.
        """
        try:
            return self._id_to_pattern[cleavage_id]
        except KeyError as exc:
            raise KeyError(f"Invalid cleavage_id: {cleavage_id}") from exc

    def fragment_by_id(self, cleavage_id: int, compound: Compound) -> Union[CleavageResult, None]:
        """Apply one selected cleavage pattern to a compound.

        Parameters
        ----------
        cleavage_id : int
            Stable ID of the cleavage pattern to apply.
        compound : Compound
            Reactant molecule.

        Returns
        -------
        CleavageResult or None
            Result returned by the selected pattern. Returns ``None`` when the
            pattern does not match the compound.
        """
        return self.get_pattern(cleavage_id).fragment(compound)

    def fragment_all(self, compound: Compound) -> Tuple[CleavageResult, ...]:
        """Apply all matching cleavage patterns to a compound.

        Parameters
        ----------
        compound : Compound
            Reactant molecule.

        Returns
        -------
        tuple of CleavageResult
            Non-empty cleavage results produced by patterns in stable ID order.
        """
        results = []
        for pattern in self._patterns:
            result = pattern.fragment(compound)
            if result is not None and len(result.products) > 0:
                results.append(result)
        return tuple(results)

    def to_dict(self) -> Dict[str, object]:
        """Serialize the pattern set to a dictionary."""
        return {
            "name": self.name,
            "patterns": [pattern.to_dict() for pattern in self._patterns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CleavagePatternSet":
        """Create a pattern set from a serialized dictionary."""
        patterns = [
            CleavagePattern.from_dict(pattern_data)
            for pattern_data in data.get("patterns", [])
        ]
        return cls(patterns=patterns, name=str(data.get("name", "")))

    def copy(self) -> "CleavagePatternSet":
        """Return a new pattern set with the same patterns and name."""
        return self.__class__(patterns=self._patterns, name=self.name)
