from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .CleavageEvent import CleavageEvent


@dataclass(frozen=True, init=False)
class FragmentEdge:
    """Directed edge between two fragment nodes.

    A ``FragmentEdge`` connects a source fragment to a product fragment in a
    :class:`FragmentTree`. One edge can store multiple :class:`CleavageEvent`
    objects when different cleavage events produce the same source-to-target
    relationship.
    """

    _id: int
    _source_id: int
    _target_id: int
    _events: Tuple[CleavageEvent, ...]

    def __init__(
        self,
        id: int,
        source_id: int,
        target_id: int,
        events: Tuple[CleavageEvent, ...] = (),
    ):
        """Create a fragment edge.

        Parameters
        ----------
        id : int
            Edge identifier in the fragment tree.
        source_id : int
            Source node identifier.
        target_id : int
            Target node identifier.
        events : tuple of CleavageEvent, optional
            Cleavage events represented by this edge.
        """
        assert all(isinstance(event, CleavageEvent) for event in events), "events must contain only CleavageEvent instances."
        object.__setattr__(self, "_id", int(id))
        object.__setattr__(self, "_source_id", int(source_id))
        object.__setattr__(self, "_target_id", int(target_id))
        object.__setattr__(self, "_events", tuple(sorted(set(events), key=lambda event: event.event_id)))

    @property
    def id(self) -> int:
        """Edge identifier."""
        return self._id

    @property
    def source_id(self) -> int:
        """Source node identifier."""
        return self._source_id

    @property
    def target_id(self) -> int:
        """Target node identifier."""
        return self._target_id

    @property
    def events(self) -> Tuple[CleavageEvent, ...]:
        """Cleavage events associated with this edge."""
        return self._events

    def with_event(self, event: CleavageEvent) -> "FragmentEdge":
        """Return a copy with one additional cleavage event."""
        assert isinstance(event, CleavageEvent), "event must be a CleavageEvent instance."
        if event in self.events:
            return self.copy()
        return FragmentEdge(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            events=self.events + (event,),
        )

    def __repr__(self):
        return "FragmentEdge" + self.__str__()

    def __str__(self):
        return f"({self.id};{self.source_id} -> {self.target_id};events={len(self.events)})"

    def copy(self) -> "FragmentEdge":
        """Return a copy of this fragment edge."""
        return FragmentEdge(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            events=self.events,
        )
