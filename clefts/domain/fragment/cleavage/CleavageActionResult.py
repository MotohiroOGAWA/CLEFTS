"""A Source-anchored action sequence and its generated target."""
from __future__ import annotations
from dataclasses import dataclass
from clefts.libs.mmkit.mmkit import Compound
from .CleavageActionSequence import CleavageActionSequence


@dataclass(frozen=True)
class CleavageActionResult:
    action_sequence: CleavageActionSequence
    compound: Compound
    smirks: str
