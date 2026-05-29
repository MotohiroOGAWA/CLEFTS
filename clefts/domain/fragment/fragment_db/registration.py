from __future__ import annotations

from typing import List
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CleavagePatternKeyRegistration:
    smirks: str
    charge_mode: str

@dataclass(slots=True)
class FragmentEdgeAttrRegistration:
    cleavage_pattern_key: CleavagePatternKeyRegistration
    react_indices: str
    prod_indices: str


@dataclass(slots=True)
class FragmentEdgeRegistration:
    source_smiles: str
    target_smiles: str
    events: List[FragmentEdgeAttrRegistration]