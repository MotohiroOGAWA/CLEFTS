# Source-anchored FragmentTreeBuilder

`FragmentTreeBuilder` matches every pattern once on Original Source, inspects product templates once, and enumerates unordered primitive action combinations in Python. After conflict, precedence-cycle, redundancy, and duplicate checks, each unique final effect is compiled and applied once to Original Source. Intermediate fragments are never matched or reacted.

```python
builder = FragmentTreeBuilder(
    max_action_count=3,
    cleavage_pattern_set=patterns,
)
actions = builder.create_cleavage_actions(source_compound)
seed = CleavageActionSequence((actions[0],))
tree = builder.build(
    source_compound,
    seed_action_sequences=(seed,),
    max_action_count=3,
)
```

Without seeds (including an empty seed collection), exploration starts at Source, node 0. With seeds, Source remains node 0 and exploration starts only at the seed histories. Seed action counts are included in the normalized action limit. Seeds with multiple actions attach their whole sequence in a transition with `is_seed=True` and `added_action=None`.

Nodes merge by canonical chemical SMILES. Expansion states retain each different Source action history even when the chemical node is shared. Equal final effects reuse a generated compound; equal SMILES alone do not identify equal effects because the retained Source atoms can differ.

If a primitive target fails chemical valence but its combined target is valid, the builder can connect that already computed target through a generated alternative predecessor. Canonical combination enumeration remains separate from this tree presentation; the effect is never executed twice.

`FragmentEdge.transitions` stores `CleavageActionTransition` values. Normal transitions represent adding one primitive action and normalizing the child history; they do not describe a reaction on the parent compound. `FragmentTree.num_transitions` counts them. Legacy `CleavageEvent`, `CleavageResult`, and the array APIs for historical events remain available, but the new builder creates no local-index cleavage events. Transition data is retained by in-memory edge access and copying, not serialized to the legacy database schema.

The retained set of a primitive action is the Source connected component containing all selected product anchors after removing the cuts of the entire reaction. Products whose anchors span components are unsupported. New atoms/bonds, element/charge updates, and aromatic bond order changes are omitted. Existing unspecified charges and unchanged aromatic bonds are preserved; single/double/triple bond order changes are supported.

`max_action_count` replaces the builder's `max_depth`; `min_depth_only_from`, `cleave_by_pattern`, and `cleave_by_pattern_id` were removed. Builder dictionaries now require `max_action_count`. `only_add_min_depth` remains a presentation option for minimum action count edges and never suppresses expansion of a different history. Graph traversal depth remains the number of edges, which can differ from action count for multi-action seeds.

`builder._build_result(source_compound)` returns `search_stats`, `expansion_states`, and `processed_expansion_states` alongside the tree and node compounds. Counters include primitive actions, raw combinations, hard conflicts, precedence cycles, normalization reductions, duplicate histories/effects, compiled sequences, RDKit executions, and generated effects. For the generic single-bond pattern on `CC(O)N` at action limit 3, the current search considers 28 raw candidates and runs 10 unique effects.

`FragmentIonTreeBuilder` and `Fragmenter` forward `max_action_count` and `seed_action_sequences`; their presets and shared test fixtures use the new builder schema. Existing ion candidate calculations, `tree/db`, pathway and ML implementations remain unchanged. `Fragmenter.precursor_candidate_max_depth` is a separate pathway selection setting, preserved by serialization and copying. Non-root precursor selection retains all action history transitions. `tree_max_depth` remains a legacy traversal budget accessor for existing callers; `tree_max_action_count` exposes the builder limit explicitly.

Canonical enumeration computes each unordered combination once, but the tree can retain multiple valid one-action predecessors of that already generated history. This supports paths through distinct precursor histories without repeating reactions. Batched pathway caching also distinguishes precursor contexts. An old benzene-loss expectation at m/z 125.0233 is now explicitly unassigned because its Source action overlaps the precursor's changed bonds; the hard conflict rule is preserved.

```bash
python -m unittest tests.domain.fragment.TestFragmentTreeBuilder tests.domain.fragment.TestCleavageAction tests.domain.fragment.TestCleavagePattern tests.domain.fragment.TestCleavagePatternSet tests.domain.fragment.TestFragmentTree
```
