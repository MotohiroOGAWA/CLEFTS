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

- cleavage pattern id per event
- reaction id per event
- product molecule id per event
- reactant SMARTS match identity: every pair of matched reactant substructures is classified as the same (1) or different (0). The matched atoms and their internal bonds are canonicalized after clearing atom-map numbers, so broad SMARTS patterns still learn concrete differences such as C-C versus C-O.
- surrounding local graph labels: from the cleavage feature and an unlabeled graph skeleton containing the reactant SMARTS atoms plus a configurable bond radius, predict every atom and bond label. Rooted positional features identify each reactant SMARTS slot and its graph distance without exposing source atom labels.

`observed_event_edge` and `atom_location` are intentionally not trained.

Source/target ECFP is intentionally not included here because the fragment-tree nodes already carry molecule-level information from the frozen pretrained MolEncoder.

## CLI

```bash
python -m clefts.ml.training.cleavage_training.training_model \
  --train-dir <project>/train_structures \
  --val-dir <project>/validation_structures \
  --output-dir <project>/cleavage_pretraining \
  --cleavage-pattern-set-json clefts/domain/fragment/presets/fragmenter_pos.json \
  --mol-encoder-checkpoint <mol_training>/pretraining_best.pt \
  --max-cleavage-events 128 \
  --device cuda
```

`--cleavage-pattern-set-json` may point either to a raw `CleavagePatternSet` JSON or to a fragmenter preset containing `fragment_ion_tree_builder.cleavage_pattern_set`.

### Limiting events per SMILES

`--max-cleavage-events` limits how many eligible first-stage cleavage events
from one SMILES structure are encoded in a regular dataset entry. This avoids
running out of accelerator memory for large compounds with many candidate
edges. The limit is applied before `CleavageFNet` constructs event features,
so unselected events do not consume memory in the event encoder.

Training randomly resamples the limited event subset whenever a SMILES
structure is loaded. Different events can therefore be learned across epochs
instead of permanently discarding events beyond the limit. Validation selects
a deterministic subset so its metrics remain comparable between epochs.

Rare-target and reactant positive/negative forced sampling remains active.
An event explicitly added by the balanced sampler is always retained even when
regular entries are limited. Omitting `--max-cleavage-events` preserves the
previous behavior and encodes all eligible first-stage events. The value must
be at least `1` when specified.

## TensorBoard

Metrics are written to `<output-dir>/tensorboard`.

Examples:

- `pattern_acc/pattern`
- `reaction_acc/reaction`
- `product_molecule_acc/product_molecule`
- `reactant_structure_loss/reactant_structure`
- `reactant_structure_acc/reactant_structure`
- `surrounding_structure_loss/surrounding_structure`
- `surrounding_structure_acc/surrounding_structure`

Surrounding-structure evaluation also reports each reconstructed label group
separately:

- atom: `symbol`, `charge`, `ring_type`, `hybridization`,
  `num_hydrogens`, and `valence_electrons`
- bond: `bond_type` and `ring_type`

For example, TensorBoard contains `train/atom_symbol`,
`val/atom_symbol`, `train/bond_type`, and `val/bond_type` series under
the surrounding-structure accuracy/count charts.

ID classification metrics are also grouped by class, for example
`pattern_acc/pattern_by_class`, `reaction_acc/reaction_by_class`, and
`product_molecule_acc/product_molecule_by_class`. Reactant-structure batches
are expanded from the preprocessing index so positive and negative pairs are
both present whenever the split contains at least two identities. The loss
uses the same number of positive and negative pairs, preventing the quadratic
number of pair combinations from letting either side dominate the gradient.
Validation selects this balanced subset deterministically.

## Preprocessing cache and balanced sampling

Before training, every first-stage event is indexed by source file, edge,
event row, ID targets, and canonical reactant identity. The reusable cache
defaults to `<output-dir>/main_preprocessing_cache.pt`. If it exists and its
input-file manifest matches, it is loaded without rescanning. `--overwrite`
keeps this cache and removes only the other generated training outputs.
Human-readable copies are written as
`preprocessing_event_index.csv` and `preprocessing_summary.json`.

- `--preprocessing-cache`: explicit cache path
- `--rebuild-preprocessing-cache`: force a rescan
- `--min-data-count`: targets whose total event count is at or below this value are treated as rare. For example, `1000` makes every target represented by at most 1000 cached events eligible for forced coverage.
- `--mask-balance-patience`: number of consecutive batches in which a rare target may be absent. Once this reaches the configured value, one matching cached cleavage event is forced into the next batch. The default is 20.
- `--mask-balance-max-forced-per-batch`: upper bound on the number of rare cleavage events added to one batch by the preceding rule. The default is 8; it prevents balancing from expanding a batch without limit.

Forced entries are event-level, not file-level. The cache locates the source
file, edge, and cleavage-event row, but only the selected cleavage event
contributes to the next forward pass and loss. Other events in that file are
not trained during that forced entry. Ordinary shuffled dataset entries train
all first-stage events in their file unless `--max-cleavage-events` is set; if
it is set, they use the randomly resampled subset described above.

## Training reports

Training uses the existing `.pt` structure files through the cached event
index. The output directory contains:

- `target_class_report.json` and `target_class_report.csv`: train/validation
  counts for every pattern, reaction, product molecule, and joint ID
- `metrics.csv`: epoch 0 baseline and every train/validation metric, including
  per-class accuracy and count
- `training_summary.json`: best/final epoch, early-stopping state, final
  metrics, and the target distribution report
- `best.pt` and `last.pt`: best-validation and final checkpoints
- `tensorboard_command.txt`: command for opening the generated TensorBoard log
- `preprocessing_event_index.csv`: file/edge/event-to-target mapping
- `preprocessing_summary.json`: cached target totals

Optional early stopping follows the `mol_training` workflow and can be enabled
with `--early-stopping-patience`.
