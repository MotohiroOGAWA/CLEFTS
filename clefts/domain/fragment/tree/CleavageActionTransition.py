"""In-memory exploration transitions expressed in Original Source atom maps."""
from __future__ import annotations
from dataclasses import dataclass
from ..cleavage.CleavageAction import CleavageAction
from ..cleavage.CleavageActionSequence import CleavageActionSequence


@dataclass(frozen=True)
class CleavageActionTransition:
    """Add an action to a parent history; the reaction runs on Original Source.

    Seed links attach an entire starting sequence to node 0. For those links
    added_action is None and is_seed is True. Local intermediate atom indices
    and database cleavage events are deliberately absent.
    """
    added_action: CleavageAction | None
    action_sequence: CleavageActionSequence
    parent_action_sequence: CleavageActionSequence | None = None
    is_seed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action_sequence, CleavageActionSequence):
            raise TypeError("action_sequence must be a CleavageActionSequence")
        if self.parent_action_sequence is not None and not isinstance(
                self.parent_action_sequence, CleavageActionSequence):
            raise TypeError("parent_action_sequence must be a CleavageActionSequence or None")
        if self.added_action is not None and not isinstance(self.added_action, CleavageAction):
            raise TypeError("added_action must be a CleavageAction or None")
        if self.is_seed:
            if self.added_action is not None or self.parent_action_sequence is not None:
                raise ValueError("Seed transitions attach a complete sequence to Original Source")
        elif self.added_action is None:
            raise ValueError("Exploration transitions require an added_action")
