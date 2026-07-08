# Cleavage Training

`cleavage_training` pretrains cleavage-event features on fragment-tree edges. It is designed as the edge-side companion to `mol_training`, which pretrains molecular graph/node representations. Multiple cleavage events on the same fragment-tree edge are kept as separate training rows.

## Data Source

The training data is the saved `TrainingFragmentTreeStructure` produced by `clefts.ml.input`. This keeps edge pretraining aligned with the same fragment structures used by spectrum training.

Recommended layout:

```text
<project>/train_structures/data/*.pt
<project>/validation_structures/data/*.pt
```

The positive edge target comes from `structure.target_edge_index`, i.e. the spectrum-assigned traversal edges. If a structure does not contain supervised targets, `sample_edge_index` is used as a fallback candidate-edge target.

## Model

```text
FragmentTreeStructure.node_graph
  -> MolEncoder
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
- reactant/product atom location masks per event inside source and target fragment molecules

Source/target ECFP is intentionally not included here because the fragment-tree nodes already carry molecule-level information and can be pretrained by `mol_training`.

## CLI

```bash
python -m clefts.ml.training.cleavage_training.training_model \
  --train-dir <project>/train_structures/data \
  --val-dir <project>/validation_structures/data \
  --output-dir <project>/cleavage_pretraining \
  --cleavage-pattern-set-json clefts/domain/fragment/presets/fragmenter_pos.json \
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
- `atom_location_acc/atom_location`
