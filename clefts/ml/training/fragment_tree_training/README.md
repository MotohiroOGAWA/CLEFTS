# Fragment-tree spectrum training

This package trains the spectrum model on saved
`TrainingFragmentTreeStructure` files.  It follows the two pretraining
workflows in the neighbouring directories:

- `mol_training` supplies the molecular-node encoder checkpoint.
- `cleavage_training` supplies the cleavage-edge encoder checkpoint (and may
  also contain the molecular encoder state).

The model predicts which molecular fragments annotate an observed peak, their
ion/unsaturation/radical states, which retained fragments should be cleaved at
the next depth, and one intensity for each formula.  Formula candidates are
grouped by `(sample, formula)`, so one peak has one formula while any number of
molecular fragments may annotate that peak.

## Why training and inference differ

Training uses one teacher-forced forward pass over the saved candidate tree.
The target edge/node indexes in `TrainingFragmentTreeStructure` provide all
depths at once.  Running the expensive sequential beam search for every
optimizer step would multiply training time and would make hard top-k choices
part of the gradient path.

Inference uses staged pruning:

1. Encode molecular nodes and cleavage events once.
2. Evaluate at most `max_edges_per_step` new edges.
3. On later windows, evaluate those new edges together with the survivors from
   the preceding window.
4. Retain at most `max_retained_edges` edges (fewer are kept when fewer exist).
5. Select at most `max_next_cleavage_candidates` retained molecular nodes to
   expand at the next cleavage depth.
6. Predict a non-negative molecular-node-to-formula-node edge score and sum
   all scores in the same `(sample, formula)` group.  That sum is the formula
   intensity.

For example, with `max_edges_per_step=128` and
`max_retained_edges=30`, 300 candidates are processed as `128`, then
`30 + 128`, then `30 + 44`; only the best 30 survive.  The limits are applied
per spectrum sample.  `None` disables an edge limit.

The molecular and cleavage feature tensors are cached inside one prediction,
so windowing bounds the tree-transformer workload without repeatedly running
the pretrained encoders.

## Where model parameters come from

Do not specify `mol_encoder_params`, `cleavage_edge_fnet_params`, or a
cleavage pattern set by hand. The training command builds them from the two
pretrained model files:

- `--mol-encoder-checkpoint`: supplies `mol_encoder_params` and the pretrained
  MolEncoder weights.
- `--cleavage-edge-fnet-checkpoint`: supplies
  `cleavage_edge_fnet_params`, the pretrained edge-network weights, and its
  `cleavage_pattern_set_params`.
- `--fragmenter-params`: supplies all other Fragmenter settings, including ion
  adduct rules, depth, precursor-candidate depth, and mass tolerance.

The cleavage pattern set stored in the cleavage checkpoint is authoritative.
If the file passed to `--fragmenter-params` also contains
`fragment_ion_tree_builder.cleavage_pattern_set`, that value is replaced by
the checkpoint value. This guarantees that the CleavageEdgeFNet is constructed
with the same pattern/reaction/product definitions used during its pretraining.

Both pretrained encoders are frozen. Checkpoint loading is strict and checks
that the MolEncoder dimensions used by the cleavage checkpoint are compatible
with the separately supplied MolEncoder checkpoint. The two paths are not
arbitrary: `graph_dim` must equal the cleavage model's `mol_dim`, and
`node_dim` must equal its `atom_dim`. Use the same MolEncoder checkpoint that
was passed when that cleavage model was pretrained.

The number of cleavage depths is taken only from the fragmenter configuration
(`fragmenter_params.fragment_ion_tree_builder.max_depth`).  The spectrum model
does not define a second depth setting.  `max_next_cleavage_candidates=3`
means that at most three molecular nodes per sample seed each following depth.

## Data layout

```text
<project>/
  config/
    model_config.json
    train_config.json
    preprocessing_config.json
  train_structures/data/*.pt
  validation_structures/data/*.pt
```

The structure files are produced by the existing fragment-tree data creation
workflow.  A target peak may contain several rows in `target_node_index`; they
share the same `(target_sample_index, target_peak_index)` and formula.  This is
how multiple fragment annotations for one formula peak are represented.

## Run

From the application root, provide the two pretrained models and the
Fragmenter JSON. All newly trained model and optimizer settings are ordinary
command-line arguments.

```bash
python -m clefts.ml.training.fragment_tree_training.training_model \
  --train-dir data/train_preprocessing/single_bond_pos/train_structures \
  --val-dir data/train_preprocessing/single_bond_pos/validation_structures \
  --output-dir data/training/fragment_tree_projects/main \
  --mol-encoder-checkpoint data/training/mol_projects/main/runs/node16_gdim64_layers2_heads8_deg4_spd3_edged3_drop0p5/pretraining_last.pt \
  --cleavage-edge-fnet-checkpoint data/training/cleavage_projects/main/best.pt \
  --fragmenter-params clefts/domain/fragment/presets/fragmenter_single_bond_pos.json \
  --condition-adduct-embedding-dim 16 \
  --condition-ce-feature-dim 16 \
  --condition-ce-fc-dims 32 \
  --condition-feature-dim 64 \
  --condition-fc-dims 128,64 \
  --tree-hidden-dim 128 \
  --tree-num-layers 2 \
  --tree-num-heads 8 \
  --tree-max-degree 16 \
  --dropout 0.5 \
  --max-edges-per-step 128 \
  --max-retained-edges 30 \
  --max-next-cleavage-candidates 3 \
  --experiment-name spectrum_v1 \
  --batch-size 2 \
  --device cuda \
  --epochs 100 \
  --validation-interval-steps 1000 \
  --save-interval-epochs 1 \
  --save-interval-steps 1000 \
  --optimizer AdamW \
  --lr 0.00001 \
  --weight-decay 0 \
  --grad-clip-norm 1 \
  --early-stopping-patience 10
```

`--train-dir` and `--val-dir` accept either a split directory containing a
`data/` subdirectory or the `data/` directory itself. `--output-dir` is kept
separate and receives experiments, checkpoints, metrics, and generated
validation spectra. The preprocessing configuration used for compatibility
checks is discovered from the `--train-dir` parent project; it is not required
under `--output-dir`.

The initial `ValStart(0)` pass is disabled by default so training starts
without first generating all validation spectra. Add `--validate-at-start`
when an untrained baseline validation measurement is required. Validation at
configured step intervals and after training is unchanged.

Comma-separated dimension arguments such as `--condition-fc-dims` are written
without spaces. The effective checkpoint-derived and command-line settings are
saved as `model_config.json` and `train_config.json` in the run directory, so
the run remains reproducible without maintaining a handwritten model config.

Training writes managed checkpoints, CSV metrics, TensorBoard events, and
validation spectra below `<project>/experiments/<experiment_name>/`.

## Important limits

- `max_edges_per_step` is an inference memory control, not a training-data
  truncation option.
- `max_retained_edges` is a maximum; the model does not pad to that number.
- Set checkpoint paths before training starts.  Changing encoder definitions
  after structure creation is rejected by preprocessing compatibility checks.
- Deterministic top-k pruning is non-differentiable by design.  The training
  losses supervise peak coverage, negative candidates, ion states, next
  cleavage decisions, edge coverage, and formula intensity in one pass.
