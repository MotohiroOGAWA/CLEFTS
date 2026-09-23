# Source-anchored fragment-tree model

Primitive cleavage actions are encoded once from the source molecular graph.
Each branch group is identified by `(compound, normalized main adduct)` and
starts from the empty Source action state. `ActionStateEncoder` represents the
selected action set without positional encoding. `BranchingCleavageDecoder`
scores every chemically compatible next action from the complete primitive
action universe; collision energy is not an input to this scorer.

Actions in a state are an unordered, simultaneous cleavage set. There is no
"run A before B" interpretation. For candidate `B` and every selected action
`A`, compatibility requires both:

```text
B.source_atom_maps ⊆ A.retained_atom_maps
A.source_atom_maps ⊆ B.retained_atom_maps
A.changed_bond_maps ∩ B.changed_bond_maps = ∅
```

The tensor candidate mask is therefore:

```text
pool_valid(B)
AND B not already selected
AND mutual_source_retention(A, B) for every selected A
AND no_changed_bond_conflict(A, B) for every selected A
AND intersection(retained_atom_maps of the normalized set) is non-empty
AND normalized set differs from the parent set
AND normalized action count <= max_action_count
```

Preparation stores the directional invalidation relation `A -> B` when
`B.source_atom_maps` is not a subset of `A.retained_atom_maps`. The mask checks
all ordered pairs in the proposed set, so either `A -> B` or `B -> A` rejects
the composite action. A changed bond merely appearing in another action's
matched SMARTS context does not establish ordering; only two actions changing
the same Source bond is a conflict.

Search accumulates `logsigmoid` transition scores, prunes on cumulative path
probability when enabled, deduplicates equal normalized states by maximum score,
and enforces `max_action_count` plus one shared `max_fragment_nodes` budget per
branch group. There is no learned absolute action filter, top-k pool, beam, or
EOS classifier.

After search, selected states cross the explicit chemistry boundary and are
materialized. The post model encodes the resulting fragment tree. Prepared ion
explanations embed normalized ion identity, unsaturation, and radical state;
normalized attention combines explanations of the same physical ion. An
independent intensity is then predicted from fragment, physical ion, main
adduct, and collision-energy features.

Training uses prepared graphs and packed transition/path/candidate tensors only.
It does not parse SMILES, match SMARTS, materialize fragments, calculate
formulas, generate ion candidates, or call RDKit.
