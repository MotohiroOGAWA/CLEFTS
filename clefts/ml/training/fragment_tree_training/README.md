# Fragment-tree training

This package trains two-stage fragment-edge ranking, fragment-ion candidate
selection, and direct formula-intensity prediction on stored fragment trees.

The current architecture uses a pretrained `mol_training` checkpoint, but it no
longer uses a pretrained `cleavage_training` checkpoint. The MolEncoder is loaded
and frozen. The fragment-edge encoder is initialized and trained as part of the
fragment-tree model.

## Architecture overview

```text
TrainingFragmentTreeStructure
    |
    | node_graph
    v
frozen MolEncoder
    |-- molecule features [N_molecule, F_molecule]
    `-- atom features     [N_atom, F_atom]

Stage 1: shared structural edge encoding
    every structural edge (once per stored tree)
        -> ordered SMARTS-slot queries
        -> cross-attention over all source/reactant atoms
        -> shared edge embedding
    shared edge embedding + CE/adduct embedding
        -> base score + lightweight dot-product interaction
        -> sample-specific independent scalar score
        -> intensity-order pairwise RankNet loss

Stage 2: competitive fragment-tree model
    depth-1 edges capped by max_edges_per_depth[0]
        -> Graphormer tree encoder
        -> rank at most 3 nodes that should fragment again
        -> evaluate every stored outgoing edge of those nodes
        -> repeat node selection at the next depth
        -> direct intensity head using cached edge features and both scores
```

No edge is removed before its attention-aware stage-1 feature and score are
computed. `max_edges_per_step` bounds attention chunks and activation memory;
it does not alter candidate recall. `max_samples` controls how many experimental
conditions share a batch, so the shape preflight measures their product.

## Responsibilities of the main files

| File | Responsibility |
|---|---|
| `model.py` | Pairwise edge-ranking loss and downstream competitive/intensity losses |
| `training_model.py` | CLI, configuration, datasets, epochs, validation, checkpoints, TensorBoard |
| `workflow.py` | Sample chunking/packing and maximum-shape preflight |
| `../../specgen/components/fragment_edge/conditioned_fragment_edge_encoder.py` | Category vocabulary, all-edge chunking, and atom attention |
| `../../specgen/fragment_tree_feature_model.py` | MolEncoder, condition encoder, edge encoder, and sample-tree integration |
| `../../specgen/fragment_tree_candidate_selector.py` | Node, edge, ion-state, and formula candidate heads |

## Input structure

Training consumes `TrainingFragmentTreeStructure`, which extends
`FragmentTreeStructure` with supervised spectrum targets.

Important fields include:

| Field | Meaning |
|---|---|
| `node_graph` | Batched PyG atom graphs for fragment molecules |
| `node_graph_offset` | Atom-feature boundaries for each fragment molecule |
| `edge_index` | Directed fragment-tree edges, shape `[2, E]` |
| `cleavage_event` | Pattern, reaction, product, and atom-tuple row IDs |
| `cleavage_event_edge_index` | Mapping from cleavage events to tree edges |
| `cleavage_atom_idxs` | Ordered SMARTS-slot atom-index tables |
| `sample_adduct_type_index` | Adduct category for each MS/MS sample |
| `sample_ce_value` | Collision energy for each MS/MS sample |
| `sample_edge_index` | Mapping from samples to available tree edges |
| `target_edge_index` | Required edge IDs for supervised samples |
| `target_edge_group_index` | Groups of alternative edges that can satisfy a target formula |
| `target_expand_node_index` | Stored intermediate nodes that must be expanded along supervised paths |
| `terminal_expand_ptr` | CSR pointer mapping terminal assignments to their stored expand nodes |

Several cleavage events may map to one tree edge. Event representations are
averaged into one condition-independent structural edge embedding. Conditions
are applied only afterward by the lightweight edge-condition scorer.

## Frozen MolEncoder

The model configuration is built from a `mol_training` checkpoint containing:

- `mol_encoder_params`
- `mol_encoder_state_dict`

The MolEncoder remains in evaluation mode during fragment-tree training and its
parameters have `requires_grad=False`.

Two kinds of MolEncoder output are used:

1. Molecule-level features
   - Used for source/target absolute_ranker inputs.
   - Used as fragment-tree node features.
2. Atom-level features
   - Used as Key/Value tensors in selected-edge reactant attention.

The new fragment-edge encoder does not directly consume fixed-shape
`reactant_atom_feats` and `product_atom_feats` tensors. Product atom features are
not used in cross-attention.

## Learned edge-category vocabulary

Each cleavage event contains three learned category embeddings:

```text
pattern_embedding(pattern_id)
reaction_embedding(global_reaction_id)
product_embedding(global_product_id)
```

Reaction IDs and product IDs are local to their parent definitions. The model
therefore stores the following lookup tables as buffers:

```text
reaction_lookup_table:
    [pattern_id, local_reaction_id, global_reaction_id]

product_lookup_table:
    [pattern_id, local_reaction_id, local_product_id, global_product_id]
```

The actual embeddings remain ordinary `[N, F_category]` tensors. Missing category
combinations raise an error rather than silently falling back to row zero.

### Extending the vocabulary for fine-tuning

Call `ConditionedFragmentEdgeEncoder.extend_category_vocabulary()` before creating
the optimizer.

Rules for extension:

- Preserve every existing lookup row and global ID.
- Append new definitions instead of reordering old definitions.
- Existing embedding weights are copied unchanged.
- Only new rows are initialized.
- Do not replace embeddings after the optimizer has been constructed.

The SMARTS center-slot count is not fixed by the embedding layout. Future patterns
may contain more than two ordered center slots.

## Structural edge encoder and conditional scorer

The structural edge embedding is condition-independent and shared by all MS/MS
samples using the same stored tree. Collision energy and adduct type are excluded
from structural encoding and enter only in the later conditional scorer.

For source molecule feature `s` and target molecule feature `t`, the molecular
input is:

```text
[s, t, t - s, abs(t - s), s * t]
```

This explicitly includes target-node information, allowing the absolute_ranker to
consider what the cleavage produces. This target information is a molecule-level
MolEncoder feature, not a product-atom attention input.

The molecular input is concatenated with:

```text
[pattern_embedding, reaction_embedding, product_embedding]
```

An MLP and local atom attention produce the shared event/edge representation.
The final sample-specific edge logit is

```text
base_edge_head(edge_h)
  + dot(edge_projection(edge_h), condition_projection(condition_h))
```

The scorer produces scalar values only for valid sample/tree-edge pairs and does
not materialize a `[num_samples, num_edges, hidden_dim]` tensor.

## Current absolute_ranker loss

`target_edge_group_index` groups alternative edges that can explain the same
target formula/m/z. These alternatives are not all forced to be positive. The
group uses a smooth multiple-instance-learning OR:

```text
group_score = logsumexp(alternative_edge_logits)
positive component = mean(softplus(-group_score))
negative component = BCEWithLogits(negative_logits, 0)
loss = mean(available components)
```

Thus one sufficiently strong explanation can satisfy a measured peak. Pairwise
intensity ranking likewise uses the currently highest-scoring alternative as a
latent representative and excludes the other alternatives from zero-intensity
negative pairs. This avoids training mutually valid paths against each other.

### Training-only influential-edge sampling

Inference (`eval`) computes the attention-aware representation and absolute score
for every stored edge. During training, the inexpensive base logits are first
computed for all edges, then expensive reaction-center atom attention is limited
per sample. The default sampler keeps up to 32 edges per sample using:

1. target groups ordered by measured intensity, selecting one alternative path
   per retained group;
2. high-base-score zero-intensity edges as hard negatives;
3. random zero-intensity edges for exploration.

Alternative paths are sampled from a softmax of their detached base logits, so a
single currently preferred explanation is trained in one step without permanently
discarding other explanations. The selected structural edges are shared across
conditions in the same stored tree.

Important current limitation: the positive component is not yet weighted by
`intensity / max_intensity`. A high-intensity target edge and a low-intensity
target edge currently receive the same positive-edge weight. Capacity-adjusted
recall for trees whose required targets exceed the configured budget is also not
yet implemented. See "Known limitations" below.

## Progressive frontier selection

Selection is performed by cleavage depth rather than by taking one global top-k
over the complete graph.

### Depth 1

Nodes without incoming edges are roots. Every outgoing root edge is structurally
encoded once and receives a condition-dependent score. At most
`max_edges_per_depth[0]` edges are passed to the sample-tree model per sample.

### Depth 2 and later

After depth-1 edges are scored and capped, the node continuation head selects the
fragment nodes that should be cleaved again:

```text
root
  -> score depth-1 edges; retain the configured depth-1 limit
      -> select at most 3 expandable nodes
          -> score stored outgoing edges; retain the depth-2 limit
              -> select at most 3 expandable nodes
                  -> score stored outgoing edges; retain the depth-3 limit
```

At each depth the selector ranks expandable fragment nodes and keeps at most
`max_next_cleavage_candidates`. Their stored outgoing edges are ranked by the
condition-dependent edge score, then capped by `max_edges_per_depth[d]` for each
sample.

During training, `target_expand_node_index` and `terminal_expand_ptr` from the
stored `.pt` structure supervise this continuation score. No new fragmentation
or other cheminformatics processing is performed in the training loop.

Node continuation logits are ranked independently per sample. Nodes without any
stored outgoing edge are excluded. The selected nodes define the eligible child
edges; `max_edges_per_depth[d]` then bounds that union per sample.

Disconnected trees in a collated batch receive independent root budgets.

The current implementation limits expensive computation over an already stored
fragment tree. It does not yet make Fragmenter itself fully lazy. Avoiding the
generation of unselected depth-2 and later fragments requires a frontier-expansion
API in Fragmenter and a corresponding inference integration.

## Phase 0: absolute_ranker-only warm-up

Training starts with `training_phase=0`.

Executed in phase 0:

- Frozen MolEncoder
- Category embeddings
- Source/target molecular MLP
- AbsoluteRanker head
- Progressive frontier selection
- AbsoluteRanker loss and coverage metrics

Skipped in phase 0:

- Reactant atom cross-attention
- Sample-conditioned edge features
- Graphormer tree encoder
- Fragment candidate heads
- Formula-intensity predictor
- Spectrum cosine validation

This prevents downstream computation from consuming memory before the absolute_ranker
can reliably preserve required paths.

## Adaptive transition to phase 1

The transition is metric-driven, not epoch-driven. The following natural-selection
metrics are checked:

- `target_edge_recall`
- `target_group_recall`
- `target_node_recall`

For every metric, the worse value between training and validation is used. Default
requirements are:

| Statistic | Threshold |
|---|---:|
| minimum | 0.80 |
| first quartile | 0.95 |
| median | 0.98 |

All requirements must pass for three consecutive validations. A failed validation
resets the success counter. `training_phase` and `phase_success_count` are model
buffers and are saved in checkpoints.

The current transition is one-way. Automatic rollback from phase 1 to phase 0 is
not implemented.

## Natural selection versus teacher forcing

Coverage and phase-transition metrics are calculated before adding gold edges.
During phase-1 training, gold edges are added to the heavy path so that downstream
losses remain computable.

```text
natural selected edges ---> metrics and phase transition
          |
          `-- union(gold edges) ---> heavy training path only
```

Teacher forcing therefore cannot artificially inflate the reported natural
absolute_ranker recall.

## Collision-energy and adduct conditioning

CE and adduct information enter only after preselection, at the selected
`edge x sample` stage.

### Collision energy

The collision-energy representation is fixed to 16 sinusoidal values: eight sine
and eight cosine values.

```text
frequency_i = exp(-log(10000) * i / 8), i = 0..7
angle_i = normalized_CE * frequency_i
CE_raw = [sin(angle_0..7), cos(angle_0..7)]
```

The raw representation passes through the configured projection MLP and LayerNorm.
The CLI rejects `--condition-ce-feature-dim` values other than 16.

### Adduct type

Adduct type uses a learned embedding. The condition encoder fuses the adduct
embedding with the CE representation. The fused condition is used both by the
selected-edge query and by the tree encoder.

## Reactant atom cross-attention

Cross-attention runs only for selected `edge x sample` pairs.

### Query

The base query includes:

- The cheap edge hidden representation
- Learned pattern/reaction/product information
- Source/target molecule state
- Projected CE/adduct sample condition

Each ordered SMARTS reactant slot then receives its own query:

```text
slot_query = edge_base
           + projected_center_atom
           + ordered_slot_encoding
           + slot_relation_encoding
```

### Preserving SMARTS slot order

Slot order is chemically meaningful. For example, one side of a single-bond
cleavage may be retained while the other side becomes a neutral loss. Center atoms
are therefore not treated as an unordered set.

Slot index uses a 16-dimensional sinusoidal encoding and does not require a fixed
number of slots.

The relation feature additionally includes:

- Distance to the nearest other center
- Number of other centers tied at that distance
- Normalized slot position

### Key and Value

Key/Value tensors contain MolEncoder atom features for every atom in the
source/reactant molecule.

They do not contain:

- Product atom features
- Product atom tuples
- A separate attention pass over the product molecule

The target molecule is still represented at graph level in the absolute_ranker and
edge state.

### Graph-distance bias and hard mask

Undirected shortest-path distance is calculated separately from each SMARTS center
to every source atom:

```text
distance[slot, atom]
```

A learned distance embedding produces a different additive bias for each attention
head. Atoms farther than `attention_max_graph_distance` from every center are hard
masked:

```text
allowed(atom) = min_over_slots(distance[slot, atom]) <= radius
```

The default radius is 4. Large scaffold reactions can use a larger radius, but the
larger atom set increases attention cost. Adding chemically meaningful anchor slots
may be preferable to making the radius arbitrarily large.

### Residual path

Attention and feed-forward blocks use pre-norm gated residual updates:

```text
q = q + attention_gate * CrossAttention(LayerNorm(q), K, V)
q = q + ffn_gate * FFN(LayerNorm(q))
```

Both gates are initialized to 0.1, preserving a short direct path at initialization.
Slot queries are mean-pooled and projected to produce the final edge feature.

## TensorBoard layout

Autograd anomaly detection is disabled by default because it slows training. For
diagnosis, enable it with `--detect-anomaly`; programmatic callers can pass
`detect_anomaly=True` to `main` or `build_train_config`.

The same metric's training and validation values are placed on one card. They are
not separated into different cards.

For every absolute_ranker metric, the card tag is:

```text
absolute_ranker/<metric>
```

Its series include:

```text
train_min
train_q1
train_mean
train_median
train_q3
train_max
validation_min
validation_q1
validation_mean
validation_median
validation_q3
validation_max
```

The requested minimum, first quartile, mean, third quartile, and maximum are thus
visible together. Median is retained as an additional useful series.

Loss cards follow the same grouping rule:

```text
loss/total       -> train, train_window, validation
loss/selection   -> train, train_window, validation
loss/intensity   -> train, train_window, validation
```

In addition, the headline values are written as ordinary scalar tags in the run's
root event file every 50 successful training steps and at every epoch end. This
makes them visible even when TensorBoard is opened directly on one run directory:

```text
train/loss/total
train/loss/selection
train/loss/intensity
train/edge/edge_ranking_loss
train/edge/pairwise_ranking_accuracy
train/edge/edge_retain_precision
train/edge/edge_retain_recall
train/edge/edge_total_loss
```

Equivalent `train_window/...` and `validation/...` tags are emitted when those
aggregates are available. A failed batch is not counted as a successful training
step; its exception is printed and no metric is fabricated for it.

### Overall absolute_ranker metrics

| Metric | Meaning |
|---|---|
| `target_edge_recall` | Fraction of required edges retained naturally |
| `target_group_recall` | Fraction of formula groups with at least one retained edge |
| `target_node_recall` | Fraction of required target nodes retained naturally |
| `original_edge_count` | Candidate edge count before selection |
| `retained_edge_count` | Edge count after selection |
| `pruned_fraction` | Fraction removed by preselection |
| `over_limit` | Whether selection removed candidates |

### Metrics by cleavage depth

Root outgoing edges are depth 1. The implementation computes shortest reachable
node depths from every root and reports edge, group, and node recall for each stage:

```text
absolute_ranker/by_depth/depth_1/target_edge_recall
absolute_ranker/by_depth/depth_1/target_group_recall
absolute_ranker/by_depth/depth_1/target_node_recall
absolute_ranker/by_depth/depth_2/target_edge_recall
...
```

Each depth card contains the same training/validation distribution series. This
makes it possible to detect good overall recall that hides poor depth-2 or depth-3
coverage.

### Metrics by adduct

Adduct labels use their chemical string representation rather than internal names
such as `adduct_0`:

```text
absolute_ranker/by_adduct/[M+H]+/target_edge_recall
```

`/` inside a label is replaced with `∕` to avoid conflicting with TensorBoard's
tag hierarchy. Missing labels are shown as `unknown-<index>`.

### Metrics by collision-energy range

Finite CE values in an evaluation batch are divided using q1, median, and q3. Tags
use readable labels:

```text
by_ce_range/min-to-q1
by_ce_range/q1-to-median
by_ce_range/median-to-q3
by_ce_range/q3-to-max
by_ce_range/non-finite
```

## `max_samples` workflow setting

`max_samples` is a workflow setting, not a model-architecture parameter. A model
checkpoint can be used with different values.

`balanced_sample_chunks(sample_count, max_samples)` evenly splits an oversized
compound, avoiding a very small final chunk:

```text
sample_count=201, max_samples=100
-> 67, 67, 67
```

`pack_compound_chunks(sample_counts, max_samples)` applies first-fit-decreasing
packing and returns `(compound_index, sample_start, sample_stop)` entries.

Molecule features and shared structural edge embeddings should be cached across
chunks belonging to the same compound. Condition embeddings and scalar edge
scores remain sample-specific.

Current status: chunking and packing helpers, configuration persistence, and
preflight integration exist. Automatic sample-pointer slicing and remapping of a
stored `TrainingFragmentTreeStructure` are not yet connected to the DataLoader.

## Performance preflight and reports

Performance profiling is disabled by default. Pass `--profile-performance` to scan
the stored training batches, select the batch with the largest estimated
sample-tree size, and profile one real forward/backward pass with PyTorch Profiler
before any optimizer update.

Reports are stored under `<run_dir>/performance_profile/`:

```text
summary.json         total elapsed time, batch shape, CUDA peak allocated/reserved
modules.json         module type, parameter count, trainable count, parameter bytes
operators.json       operator CPU/CUDA time, calls, shapes, and self memory
operator_table.txt   human-readable PyTorch Profiler table
trace.json           Chrome/Perfetto trace timeline
```

If the dry-run runs out of CUDA memory, `summary.json` is still written with
`status: cuda_oom`, the peak values, batch shape, and exception. Training then
stops instead of skipping every batch.

### Lightweight shape preflight

Before training, shape checks model the shared and sample-specific allocations as

```text
[num_structural_edges, edge_feature_dim]
+ [max_samples, condition_embedding_dim]
+ [num_valid_sample_edge_pairs] scalar scores
```

Results are written to
`preflight.json` and numeric values are logged under `preflight/*`.

Recorded fields:

- `device`
- `max_samples`
- `feature_dim`
- `edge_sample_pairs`
- `elapsed_seconds`
- `cuda_peak_memory_bytes`

This synthetic edge/condition allocation is retained as a lightweight smoke
test. It is not the full-model measurement; use `performance_profile/summary.json`
for the real model peak.
Fragmenter, MolEncoder, shortest-path calculation, attention, and Graphormer.

## Important configuration options

### Fragment edge encoder

| Option | Default | Meaning |
|---|---:|---|
| `--edge-feature-dim` | 256 | Edge hidden/output dimension |
| `--edge-category-dim` | 32 | Dimension of each category embedding |
| `--edge-attention-heads` | 8 | Number of atom-attention heads |
| `--attention-max-graph-distance` | 4 | Hard-mask radius from reaction centers |
| `--max-edges-per-depth` | 128,64,32 | Per-sample edge limit for depth 1, 2, and 3 |
| `--max-next-cleavage-candidates` | 3 | Primary per-sample limit on nodes selected for the next cleavage stage |
| `--training-edges-per-sample` | 32 | Training-only budget for expensive attention-aware edge encoding |
| `--training-zero-edge-fraction` | 0.25 | Fraction reserved for hard/random zero-intensity edges |

`edge_feature_dim` must be divisible by `edge_attention_heads`. The effective
node limit is `max_next_cleavage_candidates`. The number of
`max_edges_per_depth` values must exactly equal `fragmenter.tree_max_depth`; model
construction raises `ValueError` otherwise.

### Condition encoder

| Option | Default | Meaning |
|---|---:|---|
| `--condition-adduct-embedding-dim` | 16 | Adduct embedding dimension |
| `--condition-ce-feature-dim` | 16 | CE sinusoidal dimension; fixed at 16 |
| `--condition-ce-fc-dims` | 32 | CE projection MLP |
| `--condition-feature-dim` | 64 | Fused condition dimension |
| `--condition-fc-dims` | 128,64 | Condition-fusion MLP |

### Workflow

The standalone command's help formatter shows the value used when every optional
argument is omitted, including arguments without a separate help description.
Constrained choices such as `{16}` and their default are shown together:

```text
--condition-ce-feature-dim {16} (default: 16)
```

| Option | Default | Meaning |
|---|---:|---|
| `--max-samples` | 100 | Workflow sample limit and preflight dimension |
| `--batch-size` | 1 | Number of structure files per DataLoader batch |
| `--validation-interval-steps` | 100 | Step-validation interval |
| `--train-log-interval-steps` | 50 | TensorBoard/metrics training-log interval; epoch aggregates are always logged |
| `--profile-performance` / `--no-profile-performance` | disabled | Profile the largest estimated stored batch before training |

`batch-size` counts stored structure files, whereas `max-samples` counts MS/MS
samples. They are different units.

## Dataset, MolEncoder, and resume configuration

These values are configurable. Runtime/dataset options and model-construction
options intentionally live in separate configurations:

- `train_config.json` contains `training_structure_dir`,
  `validation_structure_dir`, `validation_valid_records_file`, and `ckpt_id`.
- `model_config.json` contains `mol_encoder_checkpoint` and
  `freeze_mol_encoder`. The MolEncoder architecture is read from that checkpoint
  when the model configuration is initially generated.
- The CLI options `--training-structure-dir`, `--validation-structure-dir`,
  `--mol-encoder-checkpoint`, and `--ckpt-id` set the same values directly.

Example `PROJECT_DIR/config/train_config.json`:

```json
{
  "experiment_name": "exp_main",
  "ckpt_id": null,
  "batch_size": 1,
  "device": "cuda",
  "epoch": 100,
  "validation_interval_steps": 100,
  "train_log_interval_steps": 50,
  "save_interval": 1,
  "save_interval_steps": 100,
  "optimizer": {
    "name": "AdamW",
    "lr": 1e-5,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0
  },
  "training_structure_dir": "data/train_preprocessing/demo_pos_1pct/train_structures/data",
  "validation_structure_dir": "data/train_preprocessing/demo_pos_1pct/validation_structures/data",
  "validation_valid_records_file": "data/train_preprocessing/demo_pos_1pct/validation_structures/valid_records.msds",
  "shuffle": true,
  "validate_at_start": false,
  "max_samples": 100,
  "early_stopping": {}
}
```

For a new run, use `"ckpt_id": null`. To resume, specify a checkpoint ID such
as `"ckpt_id": "42"`. A branch-node ID resolves to its latest checkpoint;
a checkpoint-node ID loads that exact checkpoint. The checkpoint is searched
under `PROJECT_DIR/experiments/<experiment_name>`.

The corresponding model configuration contains the MolEncoder checkpoint at
its top level:

```json
{
  "probability_model_params": {
    "mol_encoder_params": {},
    "condition_encoder_params": {},
    "fragment_edge_encoder_params": {
      "max_edges_per_step": 128
    },
    "tree_encoder_params": {},
    "fragmenter_params": {},
    "dropout": 0.1
  },
  "mol_encoder_checkpoint": "data/training/mol_projects/main/best.pt",
  "freeze_mol_encoder": true,
  "max_edges_per_step": 128,
  "max_retained_edges": 30,
  "max_next_cleavage_candidates": 3
}
```

The abbreviated `{}` sections above must contain the effective architecture
parameters. Normally, let the CLI build this file from the MolEncoder checkpoint
and the preprocessing metadata instead of writing those sections manually:

```bash
python -m clefts train fragment-tree \
  data/training/fragment_tree_projects/main \
  --training-structure-dir data/train_preprocessing/demo_pos_1pct/train_structures/data \
  --validation-structure-dir data/train_preprocessing/demo_pos_1pct/validation_structures/data \
  --mol-encoder-checkpoint data/training/mol_projects/main/best.pt \
  --ckpt-id 42 \
  --max-samples 100 \
  --max-edges-per-step 128 \
  --device cuda \
  --epochs 100
```

Omit `--ckpt-id` for a new run. Relative paths are interpreted from the process
working directory; absolute paths are also accepted.

## Running with `python -m`

Run commands from the application root containing `pyproject.toml`.

Use a plural parent directory for multiple training projects. The final path below
identifies the `main` project inside `fragment_tree_projects`:

```bash
python -m clefts.ml.training.fragment_tree_training.training_model \
  --train-dir data/fragment_tree_datasets/main/train_structures \
  --val-dir data/fragment_tree_datasets/main/validation_structures \
  --output-dir data/training/fragment_tree_projects/main \
  --mol-encoder-checkpoint data/training/mol_projects/main/best.pt \
  --edge-feature-dim 256 \
  --edge-category-dim 32 \
  --edge-attention-heads 8 \
  --attention-max-graph-distance 4 \
  --max-edges-per-depth 128,64,32 \
  --max-next-cleavage-candidates 3 \
  --max-samples 100 \
  --batch-size 1 \
  --device cuda \
  --epochs 100
```

`--output-dir` identifies one project directory. Keep multiple projects below a
plural parent directory, for example
`data/training/fragment_tree_projects/main` and
`data/training/fragment_tree_projects/ablation_without_distance_bias`.

## Running through the CLEFTS CLI

The positional path is one project directory below the plural parent directory:

```bash
python -m clefts train fragment-tree \
  data/training/fragment_tree_projects/main \
  --mol-encoder-checkpoint data/training/mol_projects/main/best.pt \
  --max-edges-per-depth 128,64,32 \
  --max-next-cleavage-candidates 3 \
  --max-samples 100 \
  --device cuda
```

If the package entry point is installed, `clefts` can replace `python -m clefts`.

## Checkpoints and resume

The effective model configuration stores:

- MolEncoder parameters and checkpoint path
- `freeze_mol_encoder=true`
- Condition-encoder parameters
- Fragment-edge encoder parameters
- Tree-encoder parameters
- Fragmenter parameters
- Candidate and workflow budgets

The model state contains the learned category embeddings, absolute_ranker, atom
attention, tree encoder, downstream heads, `training_phase`, and
`phase_success_count`.

Resume with the same fragmenter and category lookup definitions. To fine-tune with
new categories, preserve old IDs, extend the vocabulary, and explicitly handle the
new embedding rows when loading the checkpoint.

## Known limitations and planned work

1. Intensity weighting is not yet applied to absolute_ranker positives.
   - `intensity / max_intensity` should weight edge and group losses.
   - A small floor should preserve gradients for low-intensity targets.
2. Continuation recall is not yet capacity-adjusted.
   - Raw recall can be below one when more than three path nodes must continue
     fragmenting for one sample and depth, even under an oracle ranking.
   - Raw, weighted, oracle-at-budget, and capacity-adjusted recall should be logged
     separately and per depth.
3. Fragmenter is not yet fully lazy.
   - Expensive model computation is frontier-limited, but stored candidate trees are
     currently materialized before model selection.
4. `max_samples` is not yet connected to automatic structure slicing.
   - Pointer and sample-index remapping are required in the DataLoader path.
5. Phase rollback is not implemented.
   - The current adaptive transition only moves from phase 0 to phase 1.
6. Shortest-path calculation currently uses Python BFS.
   - Distance-matrix caching or a batched implementation may be needed for large
     molecules and many events.
7. CE ranges are batch-relative quartiles.
   - Fixed physical CE boundaries should be configurable when cross-run comparison
     is required.
8. A dedicated worst-tree ID report is not yet implemented.
   - Distribution minima are logged, but the corresponding structure path is not
     yet written as TensorBoard text.

Keep this README synchronized with algorithm, configuration, and workflow changes.
