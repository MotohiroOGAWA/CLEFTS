# Fragment-tree spectrum training

This package trains the spectrum model on saved
`TrainingFragmentTreeStructure` files.  It follows the two pretraining
workflows in the neighbouring directories:

- `mol_training` supplies the molecular-node encoder checkpoint.
- `cleavage_training` supplies the cleavage-edge encoder checkpoint (and may
  also contain the molecular encoder state).

The model predicts which molecular fragments annotate an observed peak, their
ion/unsaturation/radical states, which retained fragments should be cleaved at
the next depth, and one intensity contribution for each fragment/ion-state
candidate. Fragment-producing edges compete within a spectrum and ion/adduct
states compete within a node. Candidate intensities are predicted without a
formula-composition input and are only then summed by `(sample, formula)`.

## Why training and inference differ

Training uses one teacher-forced forward pass.  For every sample it includes
all positive edges needed by the target terminal/expand nodes, fills the rest
of a `max_edges_per_step` window with randomly sampled negative edges, and
selects at most `max_retained_edges` molecular candidates.  Selection and
formula intensity losses are computed on every pass.  It does not run the
expensive sequential beam search for every optimizer step.

Inference uses staged pruning:

1. Encode molecular nodes and cleavage events once.
2. Evaluate at most `max_edges_per_step` new edges.
3. On later windows, evaluate those new edges together with the survivors from
   the preceding window.
4. Retain at most `max_retained_edges` edges (fewer are kept when fewer exist).
5. Select at most `max_next_cleavage_candidates` retained molecular nodes to
   expand at the next cleavage depth.
6. Multiply the competing edge and node-state probabilities, predict a
   non-negative intensity for every candidate without using its formula, and
   only then sum candidate intensities in the same `(sample, formula)` group.
7. Normalize peak intensities and discard peaks below `min_peak_intensity`.

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
- `fragmenter.json` in each structure split supplies the Fragmenter settings,
  including ion adduct rules, depth, precursor-candidate depth, and mass tolerance.

The train and validation copies of `fragmenter.json` must be identical, and
their cleavage pattern set must match the cleavage checkpoint. The saved
preprocessing symbols must also match the MolEncoder checkpoint.

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

From the application root, provide the two pretrained models. The Fragmenter
configuration is loaded from the structure directories. Other model and optimizer settings are ordinary
command-line arguments.

```bash
python -m clefts.ml.training.fragment_tree_training.training_model \
  --train-dir data/train_preprocessing/single_bond_pos/train_structures \
  --val-dir data/train_preprocessing/single_bond_pos/validation_structures \
  --output-dir data/training/fragment_tree_projects/main \
  --mol-encoder-checkpoint data/training/mol_projects/main/runs/node16_gdim64_layers2_heads8_deg4_spd3_edged3_drop0p5/pretraining_last.pt \
  --cleavage-edge-fnet-checkpoint data/training/cleavage_projects/main/best.pt \
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
  --train-log-interval-steps 100 \
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

Training metrics are recorded every 100 successful optimizer steps by default.
Use `--train-log-interval-steps` to change that interval. The log contains both
the cumulative epoch averages and averages over the most recent logging window
for total, selection, and intensity loss.

Comma-separated dimension arguments such as `--condition-fc-dims` are written
without spaces. The effective checkpoint-derived and command-line settings are
saved as `model_config.json` and `train_config.json` in the run directory, so
the run remains reproducible without maintaining a handwritten model config.

Training writes managed checkpoints, CSV metrics, TensorBoard events, and
validation spectra below `<project>/experiments/<experiment_name>/`.
The validation cosine is written both to the ordinary metrics and to
`validation/validation_cosine.tsv`. TensorBoard also receives five measured
vs generated mirror plots, sampled at evenly spaced ranks from the highest to
the lowest validation cosine.

Validation also reports peak-selection diagnostics using one-to-one peak
matching within 0.01 Da:

- precision, recall, and F1 of generated peaks, plus predicted/target/matched counts;
- the fraction of total measured intensity covered by selected peaks;
- recall among the measured top 5, 10, and 20 peaks by intensity;
- max-normalized intensity MAE, both unweighted and measured-intensity weighted,
  among correctly selected peaks.

The mean values are columns in `metrics.tsv`. Per-spectrum values are appended
to `validation/validation_peak_selection.tsv`, while mean/min/Q1/median/Q3/max
are appended to `validation/validation_peak_selection_summary.tsv`. TensorBoard
shows each distribution statistic as a line over training steps under
`peak_selection/`.

## Important limits

- `max_edges_per_step` is both the inference new-edge window and the one-shot
  training/validation candidate limit per sample. Required target-path edges
  are never replaced by random negatives.
- `max_retained_edges` is a maximum; the model does not pad to that number.
- Set checkpoint paths before training starts.  Changing encoder definitions
  after structure creation is rejected by preprocessing compatibility checks.
- Deterministic top-k pruning is non-differentiable by design.  The training
  losses supervise peak coverage, negative candidates, ion states, next
  cleavage decisions, edge coverage, and formula intensity in one pass.
