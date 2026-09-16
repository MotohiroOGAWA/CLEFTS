from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from .CleavageEvent import CleavageEvent
from .CleavageActionTransition import CleavageActionTransition


@dataclass(frozen=True)
class FragmentEdge:
    """Edge in a FragmentTree.

    Action transitions describe a parent action state plus one primitive action
    (or a complete seed sequence), normalized into a child action state. They
    do not mean that a chemical reaction was applied to the parent fragment;
    every combined reaction is applied to Original Source.
    Legacy events remain available for previously stored trees.

    Parameters
    ----------
    index:
        Local edge index in the FragmentTree.

    id:
        Database edge ID.
        This corresponds to fragment_edge.edge_id.

    source_index:
        Local index of the source FragmentNode in the FragmentTree.

    target_index:
        Local index of the target FragmentNode in the FragmentTree.

    source_id:
        Database fragment ID of the source node.
        This corresponds to fragment_edge.source_id.

    target_id:
        Database fragment ID of the target node.
        This corresponds to fragment_edge.target_id.

    events:
        Cleavage events associated with this edge.
    """

    index: int
    id: int
    source_index: int
    target_index: int
    source_id: int
    target_id: int
    events: Tuple[CleavageEvent, ...] = ()

    transitions: Tuple[CleavageActionTransition, ...] = ()

    def __post_init__(self) -> None:

        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if not all(isinstance(t, CleavageActionTransition) for t in self.transitions):
            raise TypeError("transitions must contain CleavageActionTransition instances")
        if not all(isinstance(event, CleavageEvent) for event in self.events):
            raise TypeError("events must contain only CleavageEvent instances.")

    @property
    def edge_id(self) -> int:
        """Alias for database edge ID."""
        return self.id

    def with_event(self, event: CleavageEvent) -> "FragmentEdge":
        """Return a copy with one additional cleavage event."""
        if not isinstance(event, CleavageEvent):
            raise TypeError("event must be a CleavageEvent instance.")

        if event in self.events:
            return self.copy()

        return FragmentEdge(
            index=self.index,
            id=self.id,
            source_index=self.source_index,
            target_index=self.target_index,
            source_id=self.source_id,
            target_id=self.target_id,
            events=self.events + (event,),
            transitions=self.transitions,
        )

    def with_transition(self, transition: CleavageActionTransition) -> FragmentEdge:
        if not isinstance(transition, CleavageActionTransition):
            raise TypeError("transition must be a CleavageActionTransition")
        if transition in self.transitions:
            return self.copy()
        return replace(self, transitions=self.transitions + (transition,))

    def __repr__(self) -> str:
        return "FragmentEdge" + self.__str__()

    def __str__(self) -> str:
        return (
            f"(index={self.index}; "
            f"id={self.id}; "
            f"source_index={self.source_index}; "
            f"target_index={self.target_index}; "
            f"source_id={self.source_id}; "
            f"target_id={self.target_id}; "
            f"events={len(self.events)}; transitions={len(self.transitions)})"
        )

    def copy(self) -> "FragmentEdge":
        """Return a copy of this fragment edge."""
        return FragmentEdge(
            index=self.index,
            id=self.id,
            source_index=self.source_index,
            target_index=self.target_index,
            source_id=self.source_id,
            target_id=self.target_id,
            events=self.events,
            transitions=self.transitions,
        )