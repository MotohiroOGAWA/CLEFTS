from __future__ import annotations

from dataclasses import dataclass

from ..cleavage.CleavageActionSequence import CleavageActionSequence


@dataclass(frozen=True)
class PrecursorAction:
    """One way of reaching a given precursor ion from Original Source.

    action_sequence is None exactly when Source itself already is the
    precursor (node_index == 0, the FragmentTree root): no action needs to
    be applied before further fragmentation can begin. Ambiguous chemistry
    (several equally short routes, or several candidate precursor nodes with
    the same mass) is recorded as several PrecursorAction instances rather
    than resolved to a single arbitrary choice.
    """
    node_index: int
    action_sequence: CleavageActionSequence | None

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node_index must be a non-negative integer.")
        if self.action_sequence is not None and not isinstance(self.action_sequence, CleavageActionSequence):
            raise TypeError("action_sequence must be a CleavageActionSequence or None.")
        if (self.node_index == 0) != (self.action_sequence is None):
            raise ValueError("node_index 0 (Source) must pair with action_sequence=None, and vice versa.")
