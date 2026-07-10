# Cleavage Training

`cleavage_training` pretrains cleavage-event features on fragment-tree edges. It uses a pretrained `mol_training` checkpoint containing `mol_encoder_state_dict` and `mol_encoder_params` to compute atom and molecule features in each batch, and keeps that MolEncoder frozen. Multiple cleavage events on the same fragment-tree edge are kept as separate training rows.

## Data Source

The training data is the saved `TrainingFragmentTreeStructure` produced by `clefts.ml.input`. This keeps edge pretraining aligned with the same fragment structures used by spectrum training.

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
- product-substructure identity contrast: event embeddings are pulled together when the product-side atoms matched by the reaction SMARTS canonicalize to the same atom-map-independent fragment structure, and pushed apart otherwise
- reactant/product atom location masks per event inside source and target fragment molecules

The compound identity target is built from the actual product fragment node and the product atom tuple stored for each cleavage event. The selected atoms and their internal bonds are canonicalized with RDKit after clearing atom-map numbers, so atom-map renumbering does not create a new identity. Different atom symbols or bond types remain different identities, even when the cleavage pattern itself used a broad query such as `[!#1]`.

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
- `compound_identity_loss/compound_identity`
- `compound_identity_nearest_acc/compound_identity`
- `atom_location_acc/atom_location`
