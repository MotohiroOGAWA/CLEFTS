from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, init=False)
class FragmentEdge:
    """Directed edge between two fragment nodes.

    A ``FragmentEdge`` connects a source fragment to a product fragment in a
    :class:`FragmentTree`. One edge can store multiple fragmentation steps when
    different cleavage patterns produce the same source-to-target relationship.
    """

    _id: int
    _source_id: int
    _target_id: int
    _fragment_step_strs: Tuple[str, ...]

    def __init__(
        self,
        id: int,
        source_id: int,
        target_id: int,
        fragment_step_strs: Tuple[str, ...] = (),
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
        fragment_step_strs : tuple of str, optional
            Serialized fragmentation steps represented by this edge.
        """
        object.__setattr__(self, "_id", int(id))
        object.__setattr__(self, "_source_id", int(source_id))
        object.__setattr__(self, "_target_id", int(target_id))
        object.__setattr__(self, "_fragment_step_strs", tuple(sorted(set(fragment_step_strs))))

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
    def fragment_step_strs(self) -> Tuple[str, ...]:
        """Serialized fragmentation steps associated with this edge."""
        return self._fragment_step_strs

    def with_fragment_step(self, fragment_step_str: str) -> "FragmentEdge":
        """Return a copy with one additional fragmentation step.

        Existing steps are preserved and de-duplicated.
        """
        if fragment_step_str in self.fragment_step_strs:
            return self.copy()
        return FragmentEdge(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            fragment_step_strs=self.fragment_step_strs + (fragment_step_str,),
        )

    def __repr__(self):
        return "FragmentEdge" + self.__str__()

    def __str__(self):
        return f"({self.id};{self.source_id} -> {self.target_id};steps={len(self.fragment_step_strs)})"

    def copy(self) -> "FragmentEdge":
        """Return a copy of this fragment edge."""
        return FragmentEdge(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            fragment_step_strs=self.fragment_step_strs,
        )
