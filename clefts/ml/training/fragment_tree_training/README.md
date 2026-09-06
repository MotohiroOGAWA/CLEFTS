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
| `target_peak_depth` | One preselected minimum post-precursor cleavage depth per observed peak (`-1` means empty) |
| `target_path_edge_index` | Ordered edge paths for each terminal assignment |
| `terminal_path_ptr` | CSR pointer mapping assignments to complete edge paths |

Targets are frozen during preprocessing.  For each observed `sample_peak_mz`,
only assignments at the smallest cleavage depth after the actual precursor node
are retained.  Thus a peak is represented by exactly one depth-specific node
family (or an empty family), while formulas and alternative molecular paths at
that depth remain grouped together.

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

The absolute_ranker loss actually wired into training is
`PairwiseEdgeIntensityRankingLoss` (`model.py`). `target_edge_group_index`
groups alternative edges that can explain the same target formula/m/z. These
alternatives are not all forced to be positive; instead their combined
evidence is a smooth multiple-instance-learning OR:

```text
group_evidence = logsumexp(alternative_edge_logits)
```

Per sample, groups are sorted by observed peak intensity (descending). For
each group (the "anchor"), up to `top_n` comparison partners are selected in
three tiers, filled in order until `top_n` is reached:

1. **Tier 1** (`nearest_lower_partners`, default 1): the nearest lower-intensity
   groups in sorted order.
2. **Tier 2** (`extended_lower_partners`, default 3): more lower-intensity
   groups, farther away in the sorted order.
3. **Tier 3** (`background_partners`, default 10): unassigned/background edges
   (no observed intensity at all), individually compared against the anchor
   as separate pairwise terms — not pooled into one aggregated comparison.

Tiers 1 and 2 skip (without consuming budget) any candidate whose
`sqrt(intensity)` gap to the anchor is `<= intensity_threshold`, so the scan
continues past near-equal-intensity groups instead of stopping there. Each
selected pair contributes `softplus(-(anchor_evidence - partner_evidence))` to
the loss, weighted by a reciprocal-rank weight `(1/rank_i) / sum(1/rank_n)`
computed over that sample's intensity-sorted groups (rank 1 = most intense
group), regardless of which tier the partner came from. A comparison
involving the most intense group therefore contributes more to the total loss
than an equally-sized comparison further down the ranking. Intensity values
are still used to sort groups and to gate near-equal-intensity pairs in tiers
1-2, but no longer serve as the weight magnitude directly. This avoids
training mutually valid paths against each other.

The diagnostic `pairwise_ranking_accuracy` metric (fed by `build_pairs()`)
mirrors this same tiered selection, scoped per source fragment node instead of
per sample; it therefore now also reflects tier-3 target-vs-background
separation, which it did not before this design was introduced.

A separate, unwired class, `FragmentEdgeAbsoluteRankerTrainingLoss`, implements
an alternative smooth-OR-plus-BCE design (`group_score = logsumexp(...)`,
`positive = mean(softplus(-group_score))`, `negative =
BCEWithLogits(negative_logits, 0)`) but is not instantiated anywhere in
`FragmentTreeTrainingModel`; it exists only for its own unit test.

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

This sampler only runs when `max_edges_per_tree` (below) is `None`; the
per-tree budget, when set, replaces it at both training and inference time.

Pairwise ranking terms are weighted by reciprocal rank rather than by raw
intensity (see "Current absolute_ranker loss" above), so a high-intensity
target edge's comparisons already carry more weight than a low-intensity
edge's. Capacity-adjusted recall for trees whose required targets exceed the
configured budget is not yet implemented. See "Known limitations" below.

### Per-tree expensive-edge budget (`max_edges_per_tree`)

`max_edges_per_tree` (default 256, enabled out of the box) bounds how many
distinct edges of one *stored tree* ever receive expensive attention-aware
encoding, **shared across every MS/MS sample that references that tree** —
unlike every other budget on this page, which is per-sample. A batch holding
several stored trees (`--batch-size` > 1) gets this budget applied
independently per tree, so the effective total scales with the number of
trees in the batch (e.g. 256 x 2 trees ~ 512 edges attended).

Edges are ranked once per forward call by summed cross-sample importance:

```text
importance(edge) = sum over samples s referencing edge e of
    [ large constant, if e is a target/positive edge for s
      else condition_edge_scorer(base_logit_e, condition_h_s) ]
```

independently within each tree, grouped via `FragmentTreeStructure.tree_sample_ptr`
(which sample belongs to which stored tree) joined through `sample_edge_index`
(which edges each sample references) — an edge no sample references has zero
importance and is never worth the attention budget regardless of which tree
it structurally belongs to, so only referenced edges are ever candidates; no
separate edge-level tree boundary is needed. Scoring uses only cheap,
no-attention primitives (`StructuralEdgeEncoder.encode_base` and
`ConditionEdgeScorer`, both already used elsewhere in this pipeline). The
target/positive bonus means a required target edge is never dropped in favor
of a merely high-scoring non-target edge — though a tree whose target edges
alone exceed the budget can still lose some; `tree_edge_budget/target_edge_recall`
(see "TensorBoard layout" below) makes this measurable. Edges shared by many
samples of the same tree accumulate more combined importance and are
naturally favored.

This mechanism applies identically to **both** training
(`FragmentTreeCandidateSelector.forward`, replacing the whole-batch
`_sample_training_edges` cap above whenever `max_edges_per_tree` is set) and
inference/validation (`generate_depth_limited_candidates`, replacing today's
"attend every stored edge" default). Validation-time spectrum generation
(`predict_validation_msdataset`) already goes through the real,
budget-respecting `FragmentTreeSpectrumPredictor`/`generate_depth_limited_candidates`
path rather than a raw training forward pass, so it automatically inherits
this budget too.

Set `--max-edges-per-tree` to `None`/omit it to fully restore the previous
behavior (unbounded attention at eval time, the whole-batch stochastic sampler
at training time).

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
stored `.preft.pt` structure supervise this continuation score. No new fragmentation
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

Cards are grouped by the processing stage:

| Group | Contents |
| --- | --- |
| `loss` | Total, selection, intensity, edge-ranking, and precursor keep losses |
| `edge_absolute_score` | Absolute edge-score/ranking diagnostics |
| `edge_selection` | Edge retention, coverage, and tree edge budgets |
| `peak_selection` | Peak precision, recall, counts, and top-k coverage |
| `precursor_selection` | Precursor selection and detection diagnostics |
| `intensity` | Intensity errors, coverage, and cosine similarities |

Each metric has separate `overall`, `by_adduct`, and `by_ce_range` cards when
those breakdowns are available. Precursor-excluded metrics use their own
`excl_precursor` card. For example:

```text
edge_selection/edge_retain_precision/overall
edge_selection/edge_retain_precision/by_adduct
edge_selection/edge_retain_precision/by_ce_range
loss/precursor/keep_loss/overall
intensity/intensity_cosine_similarity/overall
```

The `train`, `train_window`, and `validation` series share each applicable card.
Comparison cards show **mean only**. Each stage also has a dedicated
`<stage>_statistics` group, where min/q1/mean/median/q3/max each have a separate
card. For example, `intensity/intensity_cosine_similarity/overall` compares means,
while `intensity_statistics/intensity_cosine_similarity/overall/q1` compares only
q1. The mean also has its own card in this statistics group. Train, train-window,
and validation remain series on the corresponding card. Adduct and CE cards
include the condition label in each series name. Chemical adduct labels
are preserved; slashes in labels are replaced by `∕`. CE bins use quartiles of
the current population, with non-finite values in a separate bin. Collision-energy
conversion accepts numeric precursor-mass strings; invalid CE metadata or failed
NCE conversion produces a missing value instead of interrupting metric logging.
Headline loss cards (`loss/total`, `loss/selection`, `loss/intensity`) also overlay
these three splits. Duplicate split-prefixed loss cards are no longer written.
Open TensorBoard on the run's log directory including its child directories.
Existing event files retain their old tags; new events use this layout.

Every validation pass generates spectra from the validation MSDataset and
compares them against observed spectra. Both assignment-score partitions are
processed, even when the above-threshold partition is empty. The valid-record
MSDataset and intensity predictor are required; missing inputs and total
prediction failures are reported as errors rather than silently skipping this
comparison.

Up to five representative spectra, ranked from high to low cosine similarity,
are displayed as mirror plots in TensorBoard Images under
`validation_spectra/validation` and
`validation_spectra/validation_below_threshold`. Measured peaks point upward,
generated peaks downward, and each spectrum is normalized to a maximum intensity
of one for visibility. Titles show the spectrum index and cosine similarity.
Images are also saved under
`validation/{filtered,below_threshold}/spectra/step_XXXXXXXX/level_N.png`.
Per-spectrum scores and aggregate summaries remain in the validation TSV files.

## Depth diagnostics

Training and validation now report diagnostics for each shortest graph depth.
Roots have depth 0; an edge from a depth-d source has depth d+1. For a DAG with
multiple routes to one fragment, this is the shortest reachable depth, not the
number of expansion iterations. Unreachable components without a root are excluded.
Precursor pseudo-edges (negative edge IDs) are excluded from cleavage metrics.

| TensorBoard group | What is evaluated |
| --- | --- |
| `next_cleavage` | Actual top-K next-cleavage node choices against stored supervised intermediate nodes |
| `edge_selection` | Conditioned edge scores and survival of assigned target edges at each depth |
| `fragment_selection` | Actual kept fragment candidates against assigned terminal nodes (node identity, not ion/formula state correctness) |
| `expanded_edge_selection` | Child edges of the parents actually chosen in a progressive validation step |
| `path_coverage` | Whether complete stored supervised paths and their constituent edges survived in the candidate graph |

Selection diagnostics include precision, recall, F1, accuracy, candidate coverage,
conditional recall, TP/FP/FN counts, and target/candidate/selected counts. Undefined
ratios are NaN, not perfect scores; counts make empty or small populations visible.
Sample IDs are part of every comparison, so a correct edge for one spectrum does
not count as correct for a different spectrum of the same compound.

- Edge precision/recall/F1 use `edge_absolute_logit > 0`, matching the existing
  edge-retention diagnostic. `retained_precision` evaluates all surviving candidate
  edges regardless of this threshold. `candidate_coverage` measures how many target
  edges survived; recall includes missing/pruned target edges, whereas
  `conditional_recall` considers only targets still available. Accuracy is measured
  over available candidates, not over a huge implicit set of absent true negatives.
- `target_group_recall` accepts any retained positive edge explaining the same
  sample/formula group. It supplements strict edge recall when several explanations
  are valid alternatives.
- Next-cleavage `hit_at_k` measures the fraction of samples with a positive node
  among their choices at that depth. `top1_accuracy` evaluates the highest-scored
  choice at that depth; a missing choice for a positive sample is a miss.
- Expansion metrics evaluate **all** previously unseen outgoing edges of the
  chosen parents, including correct children subsequently removed by depth budgets.
  Targets under unchosen parents are accounted for by the full rollout recall,
  not by conditional child-edge precision/recall.
- `complete_path_recall` is the fraction of stored assignment paths whose edges all
  remain available; `path_edge_coverage` measures coverage of their union. These
  concern candidate availability, not whether every intermediate is an observed
  peak or passes the final score threshold. Alternative paths are separate stored
  assignments here; group recall above supplies the alternative-aware measure.

Normal cards describe the single forward pass used for supervised training/loss
validation. During validation, an additional inference pass follows the model's
own top-K node choices and edge budgets on the saved structure DAG, without
injecting target edges. `rollout_step_0` is the initial depth-1 pass;
`rollout_step_N` follows N attempted expansion steps. `rollout_final` includes
unreached deeper targets as misses even when expansion stops early. Next-cleavage
rollout targets contain only nodes with supervised path edges still pending,
so selecting a node whose required children are already present is not rewarded.
This evaluates decisions on the saved candidate graph; the independent spectrum
validation still rebuilds and generates spectra from the raw validation records.

Example cards:

```text
next_cleavage/depth_1/top1_accuracy/overall
edge_selection/depth_2/recall/overall
expanded_edge_selection/rollout_step_1/depth_2/conditional_recall/overall
path_coverage/rollout_final/depth_3/complete_path_recall/overall
```

Train, train-window, and validation summaries share the corresponding normal
cards. Rollout cards are validation-only. Per-batch metrics are summarized with
mean and quantiles (they are not corpus-wide confusion-matrix ratios). Mean-only
comparison cards and the separate `<stage>_statistics` cards use the same layout
as the other metrics. All dynamic
metric summaries, including depth diagnostics, are also appended to
`metric_distributions.tsv` in the run directory, with event, epoch, global step,
split, metric, and value columns. Rollout evaluation adds inference work to each
validation batch.

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

### Absolute ranker and edge budgets

| Option | Default | Meaning |
|---|---:|---|
| `--max-edges-per-step` | 128 | Attention compute-chunk size in `StructuralEdgeEncoder.forward`; does not prune candidates |
| `--max-retained-edges` | 30 | Per-sample beam width kept between progressive inference stages |
| `--max-edges-per-tree` | 256 | Per-tree, cross-sample-shared budget on edges receiving expensive attention encoding (see "Per-tree expensive-edge budget" above); set to omit/`None` to disable |
| `--edge-condition-interaction-dim` | 64 | Projection width for the edge x spectrum-condition dot-product score |
| `--ranking-loss-weight` | 1.0 | Multiplier applied to the absolute edge-ranking loss inside the total loss |
| `--top-n` | 10 | Total comparison partners per anchor group, capping tiers 1-3 combined |
| `--nearest-lower-partners` | 1 | Tier 1: nearest lower-intensity partners always compared |
| `--extended-lower-partners` | 3 | Tier 2: additional farther lower-intensity partners compared after tier 1 |
| `--background-partners` | 10 | Tier 3: unassigned/background edges compared after tiers 1-2 |
| `--ranking-intensity-threshold` | 0.05 | Minimum `sqrt(intensity)` gap required for an ordered tier 1/2 comparison |

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
| `--assignment-score-threshold` | 0.8 | Minimum `assignment_score` used for training and the filtered validation view; accepted when `score >= threshold` |
| `--max-samples` | 100 | Workflow sample limit and preflight dimension |
| `--batch-size` | 1 | Number of structure files per DataLoader batch |
| `--validation-interval-steps` | 100 | Step-validation interval |
| `--train-log-interval-steps` | 50 | TensorBoard/metrics training-log interval; epoch aggregates are always logged |
| `--profile-performance` / `--no-profile-performance` | disabled | Profile the largest estimated stored batch before training |

`batch-size` counts stored structure files, whereas `max-samples` counts MS/MS
samples. They are different units.

### Assignment-score filtering and validation scopes

`--assignment-score-threshold FLOAT` controls which spectra provide supervised
training targets. Its valid range is `0.0` through `1.0`, inclusive, and the
default is `0.8`. A sample is selected when its `assignment_score` is greater
than or equal to the threshold. This is a sample-level filter: if one
`.preft.pt` file contains both accepted and rejected spectra, only the accepted
samples and their corresponding peak/formula/assignment targets are loaded for
training.

The trainer requires one `assignment_scores.tsv` for both the training and
validation split. For a structure directory such as
`train_structures/data`, it searches in this order:

1. `train_structures/data/assignment_scores.tsv`
2. `train_structures/assignment_scores.tsv`

The TSV must contain `structure_file` and `assignment_score` columns. Files
written by the fragment-tree data-preparation command already use this schema.
Training stops with a clear error if the TSV is missing, a score is invalid, or
no sample meets the configured threshold.

Validation produces three scopes:

| Scope | Samples | Computation |
|---|---|---|
| `filtered` | `assignment_score >= threshold` | Evaluated once; this is the validation value used by training/checkpoint logic |
| `below_threshold` | `assignment_score < threshold` | Evaluated once as the complementary diagnostic partition |
| `unfiltered` | All validation samples | Assembled from the two disjoint cached partitions; filtered samples are not inferred again |

Distributional metrics, including cosine similarity, selection precision,
selection recall, F1, intensity coverage, and top-k recall, are reported with
`q1`, `median`, and `q3` (and where applicable mean/min/max). Important files in
each run are:

- `assignment_score_report.json`: threshold, selected/excluded sample counts,
  and assignment-score distribution.
- `validation/validation_scope_summary.tsv`: filtered, below-threshold, and
  unfiltered loss summaries.
- `validation/filtered/*_summary.tsv` and
  `validation/below_threshold/*_summary.tsv`: per-partition metric summaries.
- `validation/unfiltered_cosine_summary.tsv` and
  `validation/unfiltered_peak_selection_summary.tsv`: exact unfiltered
  summaries combined from cached per-spectrum results.

The same option is available in the VS Code Workbench as **Assignment score
threshold** under **Training, optimizer, and checkpoints**. Saved Workbench
configurations use the `assignmentScoreThreshold` key and pass it to the CLI as
`--assignment-score-threshold`.

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
  "assignment_score_threshold": 0.8,
  "early_stopping": {}
}
```

Training and the filtered validation view use only samples with
`assignment_score >= assignment_score_threshold` from each split's
`assignment_scores.tsv`. The default threshold is `0.8`; it can be changed with
`--assignment-score-threshold`. Filtering is sample-level, including when one
structure file contains a mixture of high- and low-score spectra.

Each run writes `assignment_score_report.json` with selected/excluded counts and
the assignment-score mean, quartiles, median, and range. Validation evaluates
the filtered and below-threshold partitions once each. The unfiltered result is
then assembled from those cached, disjoint results, so filtered spectra are not
inferred twice. Scope summaries are written below `runs/<timestamp>/validation/`;
cosine and peak-selection summaries include `q1`, `median`, and `q3` in addition
to the mean.

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

1. Continuation recall is not yet capacity-adjusted.
   - Raw recall can be below one when more than three path nodes must continue
     fragmenting for one sample and depth, even under an oracle ranking.
   - Raw, weighted, oracle-at-budget, and capacity-adjusted recall should be logged
     separately and per depth.
2. Fragmenter is not yet fully lazy.
   - Expensive model computation is frontier-limited, but stored candidate trees are
     currently materialized before model selection.
3. `max_samples` is not yet connected to automatic structure slicing.
   - Pointer and sample-index remapping are required in the DataLoader path.
4. Phase rollback is not implemented.
   - The current adaptive transition only moves from phase 0 to phase 1.
5. Shortest-path calculation currently uses Python BFS.
   - Distance-matrix caching or a batched implementation may be needed for large
     molecules and many events.
6. CE ranges are batch-relative (training-batch or validation-split) quartiles.
   - Fixed physical CE boundaries should be configurable when cross-run comparison
     is required.
7. A dedicated worst-tree ID report is not yet implemented.
   - Distribution minima are logged, but the corresponding structure path is not
     yet written as TensorBoard text.
8. `max_edges_per_tree` capacity is not adjusted per tree.
   - A tree whose target/positive edges alone exceed the budget still loses
     some of them; `tree_edge_budget/target_edge_recall` quantifies this but
     nothing yet raises the budget or splits such a tree automatically.
   - Enabling `max_edges_per_tree` (the default) also disables
     `_sample_training_edges`'s stochastic exploration (random negatives,
     softmax-sampled alternative paths) in favor of a deterministic
     importance top-k; this trade-off has not been evaluated end-to-end.

Keep this README synchronized with algorithm, configuration, and workflow changes.
