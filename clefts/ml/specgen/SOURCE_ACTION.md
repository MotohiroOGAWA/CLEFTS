# Source anchored action model (schema v4)

`SourceAnchoredFragmentSpectrumGenerator` selects normalized sets of actions
on the original Source. `ActionEncoder` pools Source atom roles with a
SetTransformer; `ActionStateEncoder` uses attention over a BOS token and the
selected action set without positional encodings. SMARTS role IDs describe
chemical roles and preserve the correspondence with `source_atom_maps`.
Permuting the action list or the selected set does not change its meaning.

The shared Source MolEncoder and static action encoder run once per unique
Source, across all CE/adduct conditions. The absolute condition scorer applies
a logit threshold followed by top K (default 64, maximum 128). Sparse domain
conflict, precedence and dominance tables are converted to dense relations
only within that pool. Retained Source regions use packed 64-bit words.

The Torch decoder checks conflict and precedence cycles before normalization,
supports replacement without increasing the action count, rejects no-ops and
empty retained regions, and groups equal normalized states by maximum score.
EOS is available at BOS and is the only choice at the action count limit.
Its bounded iteration loop operates on tensor batches of states and candidates.

`forward()` performs action selection only. `predict()` then invokes
`materialize_action_states` on terminal states and their ancestor closure,
compiling against the original Source and caching each unique effect across
conditions. The downstream model consumes those molecular graphs, tree edges,
ion states and formula groups. Target graphs, formula and m/z never enter
action selection.

## Prepare and train

Use `clefts/presets/spectrum_generator_params/source_anchored_pos_model_config.json`
as the configuration. Regenerate teachers from the original MSDataset:

```sh
python -m clefts.ml.data_preparation.fragment_tree.create_action_training_data \
  --input train.msds --params model.json --output-dir train_structures
python -m clefts.ml.data_preparation.fragment_tree.create_action_training_data \
  --input validation.msds --params model.json --output-dir validation_structures
python -m clefts.ml.training.fragment_tree_training.action_training \
  --params model.json --train-dir train_structures \
  --val-dir validation_structures --output-dir run --epochs 10
```

The existing training entry point dispatches configurations with this
architecture to the same trainer. Prediction loads its `last.pt` through the
existing prediction CLI; no first-cleavage cache is prepared for this model.
Resume with `--resume run/last.pt` in the action trainer.

Schema v4 stores Source graphs, primitive action features, sparse relation
indices, CSR teacher prefixes and multiple positive next actions/EOS, transition
DAGs, state-to-node correspondence and precomputed downstream molecular graphs.
Teacher preparation retains all valid orders leading to target states,
including replacement histories. Training force-includes their positive
actions; recall is measured before that inclusion. A pool union exceeding the
configured maximum raises an error rather than dropping teacher actions.
Training and validation forward, including intensity loss, do not run RDKit.

The old full-tree edge model and v3 reader remain for legacy configurations and
checkpoints. The action trainer rejects v3 data and legacy checkpoints; it does
not convert them or infer their action representation.

## Verification

`bash tests/run_tests.sh` captures diagnostic output and emits unittest results.
Action tests cover set permutation invariance, shared encoding, conflict,
two-/three-action precedence cycles, replacement, no-op, BOS/EOS,
multi-positive likelihood, forced inclusion, target blindness, RDKit-free
forward/backward, materialization parity, schema offsets and dehydration
precursor training. Randomized parity checks compare 10,368 Torch expansions
against domain normalization. CUDA parity checks run when CUDA is available.
