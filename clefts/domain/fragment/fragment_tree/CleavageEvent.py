from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True, init=False)
class CleavageEvent:
    """Cleavage event associated with one fragment edge.

    ``CleavageEvent`` is the structured replacement for serialized fragment-step
    strings. It records which cleavage pattern was applied, and how atom indices
    map from source fragment to product fragment. The parent edge owns the
    event, so ``edge_id`` belongs to the edge table rather than this value
    object.
    """

    _event_id: int
    _cleavage_pattern_id: int
    _react_indices: Tuple[int, ...]
    _prod_indices: Tuple[int, ...]

    def __init__(
        self,
        event_id: int,
        cleavage_pattern_id: int,
        react_indices: Iterable[int],
        prod_indices: Iterable[int],
    ):
        """Create a cleavage event.

        Parameters
        ----------
        event_id : int
            Event identifier, unique within the generated tree.
        cleavage_pattern_id : int
            Stable cleavage pattern identifier.
        react_indices : iterable of int
            Atom indices in the source fragment.
        prod_indices : iterable of int
            Atom indices in the target fragment.
        """
        object.__setattr__(self, "_event_id", int(event_id))
        object.__setattr__(self, "_cleavage_pattern_id", int(cleavage_pattern_id))
        object.__setattr__(self, "_react_indices", tuple(int(i) for i in react_indices))
        object.__setattr__(self, "_prod_indices", tuple(int(i) for i in prod_indices))

    @property
    def event_id(self) -> int:
        """Event identifier."""
        return self._event_id

    @property
    def cleavage_pattern_id(self) -> int:
        """Stable cleavage pattern identifier."""
        return self._cleavage_pattern_id

    @property
    def react_indices(self) -> Tuple[int, ...]:
        """Atom indices in the source fragment."""
        return self._react_indices

    @property
    def prod_indices(self) -> Tuple[int, ...]:
        """Atom indices in the target fragment."""
        return self._prod_indices

    @property
    def react_indices_str(self) -> str:
        """JSON string representation of :attr:`react_indices`."""
        return json.dumps(self.react_indices, separators=(",", ":"))

    @property
    def prod_indices_str(self) -> str:
        """JSON string representation of :attr:`prod_indices`."""
        return json.dumps(self.prod_indices, separators=(",", ":"))

    def to_record(self) -> Tuple[int, int, str, str]:
        """Return a SQLite-friendly record tuple.

        Returns
        -------
        tuple
            ``(event_id, cleavage_pattern_id, react_indices_str,
            prod_indices_str)``.
        """
        return (
            self.event_id,
            self.cleavage_pattern_id,
            self.react_indices_str,
            self.prod_indices_str,
        )

    def copy(self) -> "CleavageEvent":
        """Return a copy of this cleavage event."""
        return CleavageEvent(
            event_id=self.event_id,
            cleavage_pattern_id=self.cleavage_pattern_id,
            react_indices=self.react_indices,
            prod_indices=self.prod_indices,
        )
