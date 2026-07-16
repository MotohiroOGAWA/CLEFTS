# Cleavage Training

`cleavage_training` pretrains cleavage-event features on fragment-tree edges. It uses a pretrained `mol_training` checkpoint containing `mol_encoder_state_dict` and `mol_encoder_params` to compute atom and molecule features in each batch, and keeps that MolEncoder frozen. Multiple cleavage events on the same fragment-tree edge are kept as separate training rows.

## Data Source

The training data is the saved `TrainingFragmentTreeStructure` produced by `clefts.ml.input`. This keeps edge pretraining aligned with the same fragment structures used by spectrum training.

Only first-stage cleavage events are used for pretraining. In each batched
structure, nodes with no incoming fragment edge are treated as root molecules,
and only events on edges leaving those roots are passed to every prediction
head. Deeper candidate-tree stages remain in the input files but do not
contribute to the loss.

Recommended layout:

```text
<project>/train_structures/data/*.pt
<project>/validation_structures/data/*.pt
```

Pass the split directories (`train_structures` and `validation_structures`) to `--train-dir` and `--val-dir`; the loader reads their `data/` subdirectories internally.

The positive edge target comes from `structure.target_edge_index`, i.e. the spectrum-assigned traversal edges. If a structure does not contain supervised targets, `sample_edge_index` is used as a fallback candidate-edge target.

## Model

```text
FragmentTreeStructure.node_graph
  -> pretrained frozen MolEncoder
  -> FragmentTreeFeatures
  -> CleavageEdgeFeatureNet.encode_events
  -> event embedding
  -> prediction heads
```

The heads predict:

- observed traversal edge: whether the event's edge is used by assigned training pathways
- cleavage pattern id per event
- reaction id per event
- product molecule id per event
- reactant SMARTS match identity: every pair of matched reactant substructures is classified as the same (1) or different (0). The matched atoms and their internal bonds are canonicalized after clearing atom-map numbers, so broad SMARTS patterns still learn concrete differences such as C-C versus C-O.
- reactant/product atom location masks per event inside source and target fragment molecules
- surrounding structure masks: for every reactant SMARTS-matched atom, predict the atoms attached within a configurable bond radius (`--surrounding-structure-radius`, default 2).

Source/target ECFP is intentionally not included here because the fragment-tree nodes already carry molecule-level information from the frozen pretrained MolEncoder.

## CLI

```bash
python -m clefts.ml.training.cleavage_training.training_model \
  --train-dir <project>/train_structures \
  --val-dir <project>/validation_structures \
  --output-dir <project>/cleavage_pretraining \
  --cleavage-pattern-set-json clefts/domain/fragment/presets/fragmenter_pos.json \
  --mol-encoder-checkpoint <mol_training>/pretraining_best.pt \
  --device cuda
```

`--cleavage-pattern-set-json` may point either to a raw `CleavagePatternSet` JSON or to a fragmenter preset containing `fragment_ion_tree_builder.cleavage_pattern_set`.

## TensorBoard

Metrics are written to `<output-dir>/tensorboard`.

Examples:

- `observed_edge_acc/edge`
- `pattern_acc/pattern`
- `reaction_acc/reaction`
- `product_molecule_acc/product_molecule`
- `reactant_structure_loss/reactant_structure`
- `reactant_structure_acc/reactant_structure`
- `surrounding_structure_loss/surrounding_structure`
- `surrounding_structure_acc/surrounding_structure`
- `atom_location_acc/atom_location`

ID classification metrics are also grouped by class, for example
`pattern_acc/pattern_by_class`, `reaction_acc/reaction_by_class`, and
`product_molecule_acc/product_molecule_by_class`. Binary mask tasks expose
separate positive and negative series so that a high accuracy caused only by
the majority class is visible.

## Training reports

Training uses the existing `.pt` structure files directly; no balanced-data
preprocessing is required. The output directory contains:

- `target_class_report.json` and `target_class_report.csv`: train/validation
  counts for every pattern, reaction, product molecule, and joint ID
- `metrics.csv`: epoch 0 baseline and every train/validation metric, including
  per-class accuracy and count
- `training_summary.json`: best/final epoch, early-stopping state, final
  metrics, and the target distribution report
- `best.pt` and `last.pt`: best-validation and final checkpoints
- `tensorboard_command.txt`: command for opening the generated TensorBoard log

Optional early stopping follows the `mol_training` workflow and can be enabled
with `--early-stopping-patience`.
