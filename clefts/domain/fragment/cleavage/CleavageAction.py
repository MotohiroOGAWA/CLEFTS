"""Concrete edits at one manually selected reactant SMARTS match."""
from __future__ import annotations

from dataclasses import dataclass, field
from rdkit import Chem


def mapped_source(source) -> Chem.Mol:
    mol = Chem.Mol(source if isinstance(source, Chem.Mol) else source.mapped_mol)
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if not maps or any(m <= 0 for m in maps) or len(set(maps)) != len(maps):
        raise ValueError("Source atoms must have unique positive atom maps")
    return mol


def bond_key(u: int, v: int) -> tuple[int, int]:
    if u <= 0 or v <= 0 or u == v:
        raise ValueError("Bonds require two distinct positive Source atom maps")
    return min(u, v), max(u, v)


@dataclass(frozen=True)
class CleavageAction:
    product_molecule_id: int = field(kw_only=True)
    cleavage_pattern_id: int
    reaction_id: int
    # Query atom order, never sorted.
    source_atom_maps: tuple[int, ...]
    retained_atom_maps: frozenset[int]
    discarded_atom_maps: frozenset[int]
    matched_bond_maps: frozenset[tuple[int, int]]
    cut_bond_maps: frozenset[tuple[int, int]] = frozenset()

    bond_updates: frozenset[tuple[int, int, Chem.BondType]] = frozenset()
    changed_bond_maps: frozenset[tuple[int, int]] = frozenset()

    def __post_init__(self):
        if self.product_molecule_id < 0:
            raise ValueError("product_molecule_id must be nonnegative")
        object.__setattr__(self, "bond_updates", frozenset((*bond_key(u, v), order)
                           for u, v, order in self.bond_updates))
        object.__setattr__(self, "source_atom_maps", tuple(self.source_atom_maps))
        for name in ("retained_atom_maps", "discarded_atom_maps"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        for name in ("matched_bond_maps", "cut_bond_maps", "changed_bond_maps"):
            object.__setattr__(self, name, frozenset(bond_key(*b) for b in getattr(self, name)))
        universe = self.retained_atom_maps | self.discarded_atom_maps
        if not universe or any(m <= 0 for m in universe):
            raise ValueError("Source atom maps must be positive")
        if self.retained_atom_maps & self.discarded_atom_maps:
            raise ValueError("Retained and discarded atoms must be disjoint")
        if (not self.source_atom_maps or len(set(self.source_atom_maps)) != len(self.source_atom_maps)
                or not set(self.source_atom_maps) <= universe):
            raise ValueError("Match requires distinct atoms belonging to Source")
        if any(not set(b) <= set(self.source_atom_maps) for b in self.matched_bond_maps):
            raise ValueError("Matched bonds must belong to the concrete SMARTS match")
        if not self.cut_bond_maps <= self.matched_bond_maps:
            raise ValueError("Cut bonds must belong to matched SMARTS bonds")

        update_edges = frozenset((u, v) for u, v, _ in self.bond_updates)
        if not update_edges <= self.matched_bond_maps or update_edges & self.cut_bond_maps:
            raise ValueError("Updated bonds must be matched and must not be cut")
        if any(not {u, v} <= self.retained_atom_maps for u, v, _ in self.bond_updates):
            raise ValueError("Updated bonds must retain both endpoints")
        if len(update_edges) != len(self.bond_updates):
            raise ValueError("Each bond must have one target bond order")
        if any(order not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
               for _, _, order in self.bond_updates):
            raise ValueError("Only single, double and triple bond updates are supported")
        changed = self.changed_bond_maps | self.cut_bond_maps | update_edges | frozenset(
            edge for edge in self.matched_bond_maps if not set(edge) <= self.retained_atom_maps)
        if any(not set(edge) <= universe for edge in changed):
            raise ValueError("Changed bonds must belong to Source")
        object.__setattr__(self, "changed_bond_maps", changed)

    @property
    def key(self) -> tuple:
        return (self.cleavage_pattern_id, self.reaction_id, self.product_molecule_id, self.source_atom_maps,
                tuple(sorted(self.retained_atom_maps)), tuple(sorted(self.discarded_atom_maps)),
                tuple(sorted(self.matched_bond_maps)), tuple(sorted(self.cut_bond_maps)),
                tuple(sorted(self.bond_updates)), tuple(sorted(self.changed_bond_maps)))

    @classmethod
    def from_match(cls, *, source, cleavage_pattern, source_atom_maps,
                   retained_atom_maps, product_molecule_id, cleavage_pattern_id=None, reaction_id=0,
                   cut_bond_maps=()):
        """Bind an existing pattern to an explicitly selected match (no enumeration).

        Retention is supplied manually; template bond cuts and order changes are extracted.
        Only bonds explicitly present in the outer query graph are recorded.
        """
        mol = mapped_source(source)
        query = cleavage_pattern.reactant_query
        roles = tuple(source_atom_maps)
        by_map = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
        if len(roles) != query.GetNumAtoms() or len(set(roles)) != len(roles):
            raise ValueError("source_atom_maps must follow reactant query atom order")
        if not set(roles) <= by_map.keys():
            raise ValueError("Match references atoms absent from Source")
        # Pin each query atom to its chosen source index without listing matches.
        from rdkit.Chem import rdqueries
        probe = Chem.Mol(mol)
        pinned = Chem.Mol(query)
        for atom in probe.GetAtoms():
            atom.SetIntProp("_cleavage_source_index", atom.GetIdx())
        for atom, m in zip(pinned.GetAtoms(), roles):
            atom.ExpandQuery(rdqueries.HasIntPropWithValueQueryAtom("_cleavage_source_index", by_map[m]))
        if not probe.HasSubstructMatch(pinned):
            raise ValueError("Selected Source atoms do not match reactant SMARTS")
        reaction = next((r for r in cleavage_pattern.cleavage_reactions if r.id == reaction_id), None)
        if reaction is None:
            raise ValueError("Unknown reaction_id")
        if not 0 <= product_molecule_id < len(reaction.prod_temp):
            raise ValueError("Unknown product_molecule_id")
        pattern_id = cleavage_pattern_id
        if pattern_id is None:
            pattern_id = getattr(cleavage_pattern, "id", None)
        if pattern_id is None:
            raise ValueError("cleavage_pattern_id is required for this pattern")
        retained = frozenset(retained_atom_maps)
        if not retained <= by_map.keys():
            raise ValueError("Retained atoms must belong to Source")
        matched = frozenset(bond_key(roles[b.GetBeginAtomIdx()], roles[b.GetEndAtomIdx()])
                            for b in query.GetBonds())
        template = reaction.prod_temp[product_molecule_id]
        map_to_source = {a.GetAtomMapNum(): roles[a.GetIdx()] for a in query.GetAtoms()}
        product_bonds = {}
        for bond in template.GetBonds():
            u = map_to_source.get(bond.GetBeginAtom().GetAtomMapNum())
            v = map_to_source.get(bond.GetEndAtom().GetAtomMapNum())
            if u is None or v is None:
                raise ValueError("Creation of atoms is unsupported")
            product_bonds[bond_key(u, v)] = bond.GetBondType()
        if not product_bonds.keys() <= matched:
            raise ValueError("Creation of bonds is unsupported")
        cuts = set(cut_bond_maps)
        updates = set()
        for edge in matched:
            if edge not in product_bonds:
                cuts.add(edge)
            elif set(edge) <= retained:
                original = mol.GetBondBetweenAtoms(*(by_map[m] for m in edge))
                order = product_bonds[edge]
                if order != Chem.BondType.UNSPECIFIED and original.GetBondType() != order:
                    updates.add((*edge, order))
        changed = frozenset(bond_key(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())
                            for b in mol.GetBonds()
                            if not {b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()} <= retained)
        return cls(pattern_id, reaction_id, roles, retained, frozenset(by_map) - retained,
                   matched, frozenset(cuts), frozenset(updates), changed,
                   product_molecule_id=product_molecule_id)
