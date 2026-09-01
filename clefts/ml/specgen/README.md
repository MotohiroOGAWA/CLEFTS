# Spectrum inference

## Fragment-tree and spectrum representation

The fragment tree is a bipartite-augmented graph with two kinds of nodes and
two kinds of edges:

- Molecular fragment nodes are encoded by the pretrained `MolEncoder`.
- Molecular-node-to-molecular-node edges represent cleavages. Their structural
  embeddings are computed once per stored tree and shared across MS/MS samples.
- Every selected molecular node is connected to its corresponding formula
  node. Several molecular nodes may connect to the same formula node.
- The model predicts a non-negative score on every
  molecular-node-to-formula-node edge. The intensity of a formula node is the
  sum of all scores on its incident molecular edges; its formula mass and
  charge determine m/z.

Prediction first selects the necessary molecular nodes from the fragment-tree
structure and then predicts these molecular-to-formula relation scores. The
formula connections are present while molecular nodes are selected; formula
coverage is therefore part of the selection objective rather than an
after-the-fact annotation step.

Inference scores depth-1 edges and retains at most `max_edges_per_depth[0]` per
sample. It then ranks fragment nodes by their
continuation logits and selects at most `max_next_cleavage_candidates` nodes per
sample (3 by default). Every outgoing edge already stored for each selected node
is considered. The condition scorer retains at most the corresponding
`max_edges_per_depth[d]` edges per sample. The same node-selection step is
repeated at subsequent depths. `max_edges_per_step` only chunks computation.

Training uses only the candidate DAG and path annotations already stored in the
`.pt` structure. In particular, `target_expand_node_index` and
`terminal_expand_ptr` supervise the node continuation head. The training loop
does not call Fragmenter, RDKit, or any other chemical fragmentation routine.
Selection and intensity losses are evaluated together on each pass. Validation
additionally performs real staged spectrum generation,
records measured-vs-generated cosine similarity in TensorBoard/TSV, and logs
five representative mirror plots ordered from high to low similarity.

`predict_spectrum.py` loads a checkpoint produced by
`fragment_tree_training` and predicts an MS/MS spectrum. A direct prediction
needs four values: the trained model, SMILES, collision energy (CE), and
`AdductType`.

Run commands from the application root (the directory containing
`pyproject.toml`).

## Direct prediction

```bash
python -m clefts.ml.specgen.predict_spectrum \
  --model /path/to/checkpoints/<branch>/<checkpoint>/model.pt \
  --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' \
  --ce 20 \
  --adduct-type '[M+H]+' \
  --output predicted.msds
```

`--adduct` is an alias for `--adduct-type`. The spelling is `AdductType`
(not `AddcutType`). The precursor m/z is calculated from the neutral exact
mass represented by the SMILES and the specified adduct.

CE should normally be an eV number such as `20`, or a string such as `20eV`.
Percentage CE is also accepted, although a numeric eV value is preferable
when no instrument is supplied.

Multiple SMILES can be passed after `--smiles`; the same CE and adduct are
used for each one:

```bash
python -m clefts.ml.specgen.predict_spectrum \
  --model /path/to/model.pt \
  --smiles 'CCO' 'CCN' \
  --ce 20 \
  --adduct-type '[M+H]+' \
  --output predictions.msds
```

The default device is CUDA when available, otherwise CPU. Override it with
`--device cpu` or `--device cuda`. The output is an `MSDataset` (`.msds`)
whose peaks contain m/z, normalized intensity, and formula metadata.

## Model and configuration

Managed checkpoints written by `fragment_tree_training` include both
`model_config` and `model_state_dict`, so only `--model` is required. They are
stored below the training experiment's `checkpoints` directory as `model.pt`.

For a raw `state_dict`, also supply the matching generator configuration:

```bash
python -m clefts.ml.specgen.predict_spectrum \
  --model /path/to/state_dict.pt \
  --params /path/to/model_config.json \
  --smiles 'CCO' \
  --ce 20 \
  --adduct-type '[M+H]+'
```

The configuration must describe `FragmentSpectrumGenerator` and must match
the preprocessing definitions and model dimensions used for training.

## Predict records from an existing MSDataset

The earlier dataset-based mode remains available:

```bash
python -m clefts.ml.specgen.predict_spectrum \
  --input input.msds \
  --model /path/to/model.pt \
  --output predicted.msds
```

By default it uses the `SMILES`, `PrecursorMZ`, `AdductType`, and
`CollisionEnergy` columns and predicts all unique SMILES. Use the corresponding
`--*-column` options when a dataset uses different column names. Passing
`--smiles` in this mode restricts prediction to matching records.

Use `--strict` when you want checkpoint loading to fail on any missing or
unexpected parameter. Without it, counts are printed for incompatible keys.
