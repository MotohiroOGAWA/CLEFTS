# CleavageAction PoC

Combine manually selected cleavages into one molecule-specific reaction without changing the existing CleavagePattern API.

```python
action = CleavageAction.from_match(
    source=source,
    cleavage_pattern=pattern,
    cleavage_pattern_id=0,
    reaction_id=0,
    product_molecule_id=0,  # Zero-based ProductMolecule ID.
    source_atom_maps=(2, 3),  # Reactant query atom order.
    retained_atom_maps=frozenset({1, 2}),
    cut_bond_maps=(),
)
sequence = CleavageActionSequence.from_actions((action,))
reaction = sequence.compile(source)
products = reaction.run(source)
```

`source` accepts a Compound or an RDKit Mol whose atoms have unique positive atom maps. `from_match` validates the selected match and records only bonds explicitly present in the outer reactant SMARTS query graph. Bonds inside recursive SMARTS are outside this scope.

Supply retained atoms manually. `product_molecule_id` is required and selects a zero-based product template within the selected reaction. `source_atom_maps` follows RDKit reactant query atom order, regardless of template atom map numbers.

`from_match` extracts cuts and single/double/triple bond order changes from the selected product template. Additional cuts may be supplied manually. `bond_updates` records `(source_map_u, source_map_v, target_bond_type)` triples. `changed_bond_maps` includes cuts, order changes, and Source bonds removed through atom deletion, including bonds outside the matched query. Direct dataclass callers must register those additional deleted Source bonds themselves. Charge updates and atom/bond creation remain outside this scope.

Sequences reject overlapping `changed_bond_maps` after duplicate removal and before redundancy removal. Shared unchanged context bonds are allowed. Redundancy removal preserves surviving cuts and bond order updates. Compilation applies the combined edits and recalculates hydrogen counts at affected atoms.

`reaction.run` accepts the original mapped Source, calls RunReactants once, and restores Source atom maps. RDKit atom maps alone do not constrain match locations, so the compiled `rxn` includes additional queries matching Source atom map properties. Reconstructing a reaction from the `smirks` string loses these additional queries. Use `run` to execute the reaction at its concrete locations.

```bash
PYTHONPATH=. python -m unittest tests.domain.fragment.TestCleavageAction tests.domain.fragment.TestCleavagePattern tests.domain.fragment.TestCleavagePatternSet
```
