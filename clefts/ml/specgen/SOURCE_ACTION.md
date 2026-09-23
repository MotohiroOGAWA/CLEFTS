# Source-anchored fragment-tree model

Primitive cleavage actions are encoded once from the source molecular graph.
Each branch group is identified by `(compound, normalized main adduct)` and
starts from the empty Source action state. `ActionStateEncoder` represents the
selected action set without positional encoding. `BranchingCleavageDecoder`
scores every chemically compatible next action from the complete primitive
action universe; collision energy is not an input to this scorer.

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
