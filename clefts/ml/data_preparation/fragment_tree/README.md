# Preparing fragment-tree training data

This directory turns experimental MS/MS spectra into `.preft.pt` training
structures for the fragment-tree model. Preparation is the **chemistry
boundary** of the project: it parses molecules, enumerates cleavage actions,
runs RDKit, assigns peaks to fragmentation pathways and stores everything the
model needs as tensors. Training loads these tensors directly and never calls
RDKit.

```bash
python -m clefts.cli train create-fragment-tree-data \
  --input train.msds --validation-input validation.msds \
  --output-dir prepared --params model_config.json \
  --num-workers 4 --chunk-size 4 \
  --max-unique-fragment-smiles 1000 --max-cleavage-combinations 50000 \
  --normalize-intensities 1 --overwrite 1
```

Contents:

1. [Pipeline overview](#pipeline-overview)
2. [Input dataset](#input-dataset)
3. [Processing unit: one unique SMILES](#processing-unit-one-unique-smiles)
4. [Building the fragment tree](#building-the-fragment-tree)
5. [Ions, precursor and peak assignment](#ions-precursor-and-peak-assignment)
6. [Teacher structure](#teacher-structure)
7. [Validation split](#validation-split)
8. [Search limits](#search-limits)
9. [Skip conditions](#skip-conditions)
10. [Multiprocessing](#multiprocessing)
11. [Output files](#output-files)
12. [`manifest.tsv`](#manifesttsv)
13. [`action_statistics.json`](#action_statisticsjson)
14. [Statistics reference](#statistics-reference)
15. [Code map](#code-map)

---

## Pipeline overview

For every unique source SMILES:

```text
records with that SMILES (all adducts and collision energies)
  │
  ├─ cleavage action generation      SMARTS patterns matched once on the source
  ├─ action combination search       lazy, by increasing action count ──┐ max_cleavage_combinations
  │    └─ pruning                     conflicts, invalidation, redundancy │
  ├─ RDKit materialization           one reaction per unique effect      │
  │    └─ canonical SMILES            unique fragment set ────────────────┘ max_unique_fragment_smiles
  ├─ fragment tree                    nodes merged by canonical SMILES
  ├─ fragment ion tree                hydrogen states, ion shifts, formula candidates
  ├─ precursor resolution             records with an unreachable precursor are dropped
  ├─ peak / pathway assignment        peaks → formulas → pathways from the source
  ├─ teacher structure                teacher states, positive / weak-negative actions
  ├─ materialized teacher trees       physical-ion candidates and intensity targets
  └─ data/<stem>.preft.pt + one manifest row
```

Search and materialization are interleaved, not sequential: the search yields one
combination at a time, RDKit materializes it, its canonical SMILES is counted,
and only then is the next combination requested. Both search limits can
therefore stop a source early, without enumerating every combination first.

## Input dataset

`--input` (and optionally `--validation-input`) accepts `.msds`, `.msp`,
`.mgf`, `.tsv`, `.csv` and `.parquet`. Table formats need a `Peak` column in
msentity spectrum-table syntax.

Required metadata columns, renamable with `--smiles-column`,
`--adduct-type-column`, `--collision-energy-column` and
`--precursor-mz-column` (and `--validation-*-column` for a validation file
with different names):

| Column (default) | Check |
|---|---|
| `SMILES` | Parsed by RDKit. Every atom must be in the configured element `symbols` (plus H). |
| `AdductType` | Parsed, charge ±1, and contains exactly one main adduct registered in the fragmenter. |
| `CollisionEnergy` | Converts to a finite, non-negative eV value (`NCE=…%` uses precursor m/z). |
| `PrecursorMZ` | Finite and positive. |

A record that fails any check is **excluded**, not fatal. Excluded records are
listed with their original index, id (`SpecID` if present) and reasons in
`invalid_records.json`. A missing column, or an input with no usable records,
stops the run.

Peaks are then preprocessed per record: peaks below
`--minimum-relative-intensity` × the spectrum's maximum are dropped, and with
`--normalize-intensities 1` (the default) intensities are scaled so that the
maximum is 1.

## Processing unit: one unique SMILES

Records are grouped by their **exact SMILES string**. There is no
canonicalization, so two different strings for the same molecule become two
sources. One group becomes one *source*: one fragment tree, one `.preft.pt`
file and one `manifest.tsv` row. Collision energies and adducts of that
molecule share the source.

Inside a source, records are split into **branch groups** by normalized main
adduct (for example `[M+H]+` and `[M+Na]+`). Collision energy is only a
per-sample condition feature, so an action observed at one energy is never a
negative at another energy of the same branch group.

## Building the fragment tree

### Cleavage action generation

Each cleavage pattern's reactant SMARTS is matched **once** on the atom-mapped
source (`GetSubstructMatches(uniquify=False)`). Every match × product template
becomes a primitive `CleavageAction`. An action records:

- the atoms it retains and discards,
- the bonds it cuts and any bond-order updates,
- the matched atoms and bonds it depends on.

Templates that would create atoms or bonds, or that change aromatic bonds, are
unsupported. No-op actions and actions that retain nothing are discarded. The
number of primitive actions is `num_primitive_actions`.

### Action combination search

`CleavageActionSearch` enumerates unordered combinations of primitive actions
with at most `max_action_count` actions (from the fragmenter's
`fragment_ion_tree_builder.max_action_count`). A priority queue expands parents
in order of increasing normalized action count, and each child adds a primitive
action with a larger index than the parent's cursor.

Every (parent, candidate action) pair examined counts as one raw combination
(`num_raw_combinations`). This count covers all sizes: with
`max_action_count=3`, it includes 1-action, 2-action and 3-action
combinations.

### Pruning

Most raw combinations never reach RDKit:

| Rule | Counter | Meaning |
|---|---|---|
| Hard conflict | `num_hard_conflict_pruned` | Two actions change the same bond. |
| Invalidated action | `num_invalidated_action_pruned` | One action discards atoms that another action's match needs. Actions are simultaneous, so no order can fix this. |
| Redundancy | `num_redundancy_pruned` | Normalization dropped an action dominated by another. The normalized combination **continues**; this only counts the absorption. |
| Duplicate sequence | `num_duplicate_sequence_pruned` | The normalized history equals its parent (skipped), or it was already visited. An already-visited history is still yielded once per new parent, as an alternative route. |
| Duplicate effect | `num_duplicate_effect_pruned` | A different history has the same net effect (retained atoms, cuts, bond updates) as one already materialized, so the cached product is reused without running RDKit. |

Combinations that retain no atoms, or that normalize to more than
`max_action_count` actions, are skipped without a counter.

### RDKit materialization

Each new effect is compiled into a source-specific reaction and run once:
`num_compiled_sequences` and `num_rdkit_run_reactants` count these.

- If the reaction gives exactly one product, it counts toward
  `num_generated_fragments`.
- If sanitization fails (for example a valence violation), the effect is
  cached as invalid and every history with that effect is dropped silently.
- More than one product is an error, and the source is skipped.

### Unique fragments and tree nodes

After each successful materialization, the product's **canonical SMILES**
(without atom maps) is added to a set that **excludes the source SMILES**. The
set's size is `num_unique_fragment_smiles`. The `max_unique_fragment_smiles`
limit is checked right here, immediately after each new SMILES.

Tree nodes are merged by the same canonical SMILES; node 0 is the source.
Edges connect nodes and carry every distinct action transition between them.

- With `only_add_min_action_count` (the default), an edge is stored only for
  the minimum action count that reaches its target node. Longer histories
  remain expandable. The flag is turned off when
  `precursor_candidate_max_action_count > 0`.
- A materialized fragment whose parent history never produced a node (and
  which has no alternative predecessor) is not linked into the tree. It still
  counts in the statistics.

In the common case, tree nodes = 1 + `num_unique_fragment_smiles`.

## Ions, precursor and peak assignment

### Ion and formula candidates

`FragmentIonTreeBuilder` extends the tree with ion chemistry from the
fragmenter's adduct rule set:

- **Hydrogen states:** for each adduct rule, every (unsaturation *u*,
  radical *r*) pair up to the rule's limits, with ΔH = −2u − r.
- **Ion shifts:** one per `ion_shifts` entry. A shift's `atoms` list limits it
  to nodes containing those elements, and charged nodes get no shift.
- **Formula candidates:** per main adduct and for each node × hydrogen state
  × ion shift, the ion formula and its exact mass, sorted by mass.

### Precursor resolution

For each record, the precursor formula is its adduct applied to the source
formula. A tree node can be the precursor if its minimum action count is at most
`precursor_candidate_max_action_count` (default 0, which means only the source
itself) and its formula plus a hydrogen state plus the main adduct equals the
precursor formula.

- Records with no such precursor are dropped. The manifest reports them as
  `rejected_sample_count`, with the reason `N of M records dropped: unresolved
  precursor action sequence`.
- If no record of the source is left, the source is skipped
  (`unresolved_precursor`).

### Peak / pathway assignment

Per main adduct, every peak is matched to every formula candidate within
`mass_tolerance`. Each matched (node, adduct) pair gives pathways from the
source, and per peak only the **shortest pathways that pass through a
precursor node** are kept.

A peak is flagged as the precursor peak if it is within tolerance of the
*theoretical* precursor m/z, not the dataset's `PrecursorMZ`. Per sample:

- `assignmentScore` = matched intensity / total intensity.
- `assignmentScoreWithoutPrecursor` = the same ratio without precursor peaks.

## Teacher structure

Assigned pathways become action histories (teacher chains) from the source.
`prepare_source_actions` builds, for each branch group, a DAG of **teacher
states**: sets of primitive-action indices, starting from the empty state.

- Each state's `ms2 depth` is its number of actions.
- Peak → path lists are kept for multiple-instance learning.
- Positive transitions are the unique (parent, child, action) steps.
- For every state below `max_action_count`, each other action is checked for
  chemical validity (relations, then an actual RDKit run cached per effect):
  - Valid actions that are not positive become **weak negatives**.
  - A positive action that is invalid is an error.

The teacher states are then materialized into one fragment tree per branch
group. `prepare_post_materialization` groups each node's formula entries into
**physical-ion candidates**: (node, formula, net adduct, charge), each with its
(ion shift, *u*, *r*) explanations.

- A candidate is positive for a sample if the sample's matched peaks explain
  that (node, adduct).
- `target_intensity` is the highest observed intensity within tolerance of the
  candidate m/z, or 0.

The saved payload contains source graphs and primitive-action features, teacher
action sets at every depth, valid continuations and next-state indices,
positive paths per experimental peak, weak negatives, materialized fragment
graph tensors, physical-ion candidates with explanations, formula tensors,
exact m/z, charge, targets and peak annotations. The payload has a single
current format with no version field, so regenerate `.preft.pt` files after a
format change.

## Validation split

Molecules never cross splits, and splits compare exact SMILES strings.

| Options | Behaviour |
|---|---|
| neither option | Training only; structures are written to `train_structures/`. |
| `--validation-ratio r` | Unique training SMILES are shuffled with `--validation-seed` (default 0). `min(n−1, max(1, round(n·r)))` of them become validation. Needs at least 2 unique SMILES. |
| `--validation-input` | Validation records whose SMILES appears in training are always removed. |
| `--validation-input` + `--validation-ratio r` | Additionally caps validation at `max(1, round(training unique SMILES × r))` SMILES, chosen with the seed. |

Overlap removal and shortfalls are reported as JSON events on stdout:
`validation_overlap_removed`, `validation_below_target`, `validation_empty`.
If validation ends up empty, the run continues with training only.

## Search limits

Two per-source limits bound the cost of one molecule. Both default to `-1`
(unlimited); otherwise they must be positive integers. A source that exceeds
either one is **skipped as a whole**. Its data is never truncated.

### `max_unique_fragment_smiles`

CLI `--max-unique-fragment-smiles`, Workbench **Max Unique Fragment Smiles**
(`maxUniqueFragmentSmiles` in saved configurations).

- Counts distinct canonical SMILES of materialized fragments. **The source
  molecule itself is not counted**, even if a fragment happens to equal it.
- It is checked every time RDKit generates a fragment. When the count reaches
  `limit + 1`, the source stops at once: no further combination is requested
  and no further reaction runs.
- It does not count duplicates: two histories producing the same SMILES
  count once.
- A value equal to a source's own unlimited `num_unique_fragment_smiles` keeps
  that source's tree identical; one less skips it.

### `max_cleavage_combinations`

CLI `--max-cleavage-combinations`, Workbench **Max Cleavage Combinations**
(`maxCleavageCombinations`).

- Bounds `CleavageActionSearch.stats.num_raw_combinations`: every
  (parent, candidate action) pair examined, over all combination sizes from 1
  to `max_action_count`, including pairs that pruning then rejects.
- It is checked inside `CleavageActionSearch`, as soon as the count reaches
  `limit + 1` and before that combination is yielded. The combination that
  crosses the limit never reaches RDKit.
- This is the cheaper limit: it stops molecules with many primitive actions
  before they spend time in RDKit.

### Choosing values

Run once with unlimited values, or generous ones, and read the maxima in
`action_statistics.json` together with the per-source columns in
`manifest.tsv`. For example, `max_raw_combinations` and
`max_raw_combinations_smiles` show the most expensive prepared molecule.

Values in a parameter file (`max_unique_fragment_smiles`,
`max_cleavage_combinations` at the top level of `--params` /
`--params-json`) are used unless the CLI flag is given.

These limits replace the former `max_node` / `max_edge` (`--max-node`,
`--max-edge`, `maxNode`, `maxEdge`). Those counted tree nodes and edge
transitions, which are different quantities, so they are **not converted**:

- The old CLI flags are rejected.
- A parameter file containing `max_node` or `max_edge` is rejected with an
  error.
- Workbench drops legacy `maxNode` / `maxEdge` values when it loads an old
  configuration.

## Skip conditions

A failure in one source never stops the run. The source is recorded as skipped
and preparation continues, in serial and in parallel mode alike.

| `category` | Cause | `reason` example |
|---|---|---|
| `limit_exceeded` | A search limit was exceeded. `limit` names it. | `max_cleavage_combinations exceeded: limit=50000, observed>50000` |
| `unresolved_precursor` | No record of the source has a reachable precursor. | `No sample in this group has a valid precursor action sequence from Source` |
| `error` | Any other exception while preparing the source, for example an unsupported element or invalid peak values. `error` holds the traceback. | `ValueError: Unsupported atom symbol: 'As' ...` |
| `worker_crash` | The worker process died (native crash, OOM kill) even when the source was run on its own. `error` holds the stderr tail. | `Worker process exited with status -9` |

Limit skips are deliberate filtering, so they are kept separate from errors in
`category` / `skip_category`, in `num_skipped_sources_by_category`, and in
`num_limit_skipped_sources`.

A skipped source keeps the search statistics observed **up to the moment it
stopped**:

- For a limit skip, they are lower bounds, and the limited value is exactly
  `limit + 1`.
- For `unresolved_precursor`, and for errors raised after the tree was built,
  they are the full statistics.
- For `worker_crash`, and for errors raised before the tree search, they are
  empty.

Record-level exclusions (invalid metadata, see [Input dataset](#input-dataset))
happen before grouping and are reported in `invalid_records.json`, not as
skipped sources.

## Multiprocessing

With `--num-workers 1` (default), or fewer than two sources, sources are
prepared serially in the main process.

Otherwise:

- Sources are sent in chunks of `--chunk-size` SMILES groups, capped at
  `groups // workers` so every worker has work.
- Chunks run in subprocesses (`subprocess_worker.py`) through
  `clefts.utils.parallel_subprocess.run_parallel_subprocesses`. Each subprocess
  builds its own RDKit/fragmenter context.
- At most `num_workers` chunk files exist at once, in a temporary
  `clefts-preparation-*` directory inside the split directory. Each is deleted
  as soon as its result is read.
- Results are collected as chunks finish.
  - Output file names come from each source's index, so they match serial
    preparation.
  - Maxima ties resolve to the smallest SMILES, so `action_statistics.json`
    does not depend on completion order.
  - The rows of `manifest.tsv` are sorted by SMILES.
- If a worker process dies, its chunk is re-run one source per process, and
  only a source that still crashes is skipped (`worker_crash`).
- Skip records, including limit skips and their partial statistics, travel
  back from the workers exactly as in serial mode.

## Output files

```text
<output-dir>/
  preparation_config.json      resolved CLI/model configuration and Workbench form state
  invalid_records.json         excluded records per input: {"train": [...], "validation": [...]}
  train_structures/
    fragment-tree.pft          Workbench configuration of this split (status: running → completed)
    data/smiles_<index:06d>_<sha1[:16]>.preft.pt
    manifest.tsv               one row per source (completed or skipped)
    assignment_scores.tsv      one row per kept sample
    skipped_sources.json       one entry per skipped source
    action_statistics.json     split-level counts, search maxima, limits, configuration
    invalid_records.json       re-inspection of this split (normally [])
  validation_structures/       same layout, when validation exists
```

- A `.preft.pt` file holds the structure plus `metadata = {smiles,
  record_indexes, sample_annotations}`.
- `record_indexes` are positions within the split's metadata-valid dataset.
- If the output directory exists, the CLI requires `--overwrite 1` (or
  interactive confirmation). With overwrite, the whole directory is cleared
  first. It refuses to clear a directory that contains the inputs, the
  parameter file or the application.

`skipped_sources.json` entries:

```json
{
  "source_index": 13,
  "smiles": "CCC(C)C1NC(=O)...",
  "record_indexes": [13],
  "reason": "max_unique_fragment_smiles exceeded: limit=60, observed>60",
  "category": "limit_exceeded",
  "limit": "max_unique_fragment_smiles",
  "search_stats": {"num_primitive_actions": 50, "num_raw_combinations": 373,
                   "num_hard_conflict_pruned": 253, "...": "...",
                   "num_unique_fragment_smiles": 61}
}
```

`search_stats` is the full `CleavageActionSearchStats`, including the pruning
counters, or `null` if the tree search never ran. `error` entries add an
`error` field with the traceback.

## `manifest.tsv`

Tab-separated, one row per source, sorted by SMILES.

| Column | Meaning |
|---|---|
| `file` | `.preft.pt` name under `data/` (empty for skipped sources) |
| `smiles` | Source SMILES |
| `record_indexes` | JSON list of kept record indexes (all input records for skipped sources) |
| `num_input_records` | Records of this source |
| `num_valid_samples` | Records kept in the structure |
| `rejected_sample_count` | Records dropped (unresolved precursor), or all records of a skipped source |
| `rejection_log` | `skipped_sources.json` for skipped sources |
| `num_branch_groups` | Distinct main adducts |
| `num_teacher_nodes` | Teacher states, including one empty root per branch group |
| `num_transition_states` | Teacher states plus one-step next states of valid actions |
| `num_positive_transitions` | Unique teacher (parent, child, action) steps |
| `num_physical_ion_candidates` | Physical-ion candidates, replicated per sample |
| `num_ion_explanations` | (ion shift, *u*, *r*) explanations of those candidates |
| `max_ms2_depth` | Largest number of actions in a teacher state |
| `num_nodes`, `num_edges` | Nodes and edges of the **materialized teacher trees**, not the full search tree |
| `assignment_score`, `assignment_score_without_precursor` | Mean per-sample assignment scores |
| `num_primitive_actions` … `num_unique_fragment_smiles` | Action search statistics (see [reference](#statistics-reference)); partial for limit skips, empty when unavailable |
| `status` | `completed` or `skipped` |
| `skip_category` | Empty, or `limit_exceeded` / `unresolved_precursor` / `error` / `worker_crash` |
| `reason` | Skip reason, or the partial-drop note for a completed source |

## `action_statistics.json`

Split-level summary:

```json
{
  "architecture": "fragment-tree-physical-ion",
  "max_primitive_actions": 83,
  "max_primitive_actions_smiles": "…",
  "max_raw_combinations": 91234,
  "max_raw_combinations_smiles": "…",
  "max_compiled_sequences": 15231,
  "max_compiled_sequences_smiles": "…",
  "max_rdkit_run_reactants": 15231,
  "max_rdkit_run_reactants_smiles": "…",
  "max_generated_fragments": 15231,
  "max_generated_fragments_smiles": "…",
  "max_observed_unique_fragment_smiles": 812,
  "max_observed_unique_fragment_smiles_smiles": "…",
  "search_limits": {"max_unique_fragment_smiles": 1000, "max_cleavage_combinations": 100000},
  "num_skipped_sources_by_category": {"error": 2, "limit_exceeded": 41},
  "num_limit_skipped_sources": {"max_unique_fragment_smiles": 30, "max_cleavage_combinations": 11},
  "num_sources": 9000,
  "num_samples": 120000,
  "…": "…"
}
```

- `max_*` values are the largest per-source search statistics among
  **completed** sources, each with the SMILES that reached it (`*_smiles`;
  ties go to the lexicographically smallest SMILES). They describe the data
  that was actually prepared. Partial statistics of skipped sources are in
  `manifest.tsv` and `skipped_sources.json`.
- The largest unique-fragment count is named
  `max_observed_unique_fragment_smiles`, so it cannot be mistaken for the
  `max_unique_fragment_smiles` **limit** in `search_limits`.
- With no completed source, maxima and their SMILES are `null`.
- Other keys:
  - `num_teacher_nodes`, `num_positive_transitions`,
    `mean_teacher_nodes_per_sample`, `mean_positive_edges_per_sample`
  - `num_branch_groups`, `num_transition_states`,
    `num_physical_ion_candidates`, `num_ion_explanations`,
    `max_teacher_ms2_depth`
  - `num_sources`, `num_samples`, `num_metadata_valid_records`,
    `num_skipped_sources`, `num_skipped_records`, `num_rejected_records`,
    `num_excluded_records`
  - `num_workers`, `chunk_size`, `model_config`,
    `minimum_relative_intensity`, `normalize_intensities`

## Statistics reference

Per-source statistics come from `CleavageActionSearchStats`, filled while the
source's fragment tree is built.

| Statistic | In manifest | Meaning |
|---|---|---|
| `num_primitive_actions` | yes | Primitive cleavage actions matched on the source |
| `num_raw_combinations` | yes | (parent, action) pairs examined by the search over all sizes, before pruning; bounded by `max_cleavage_combinations` |
| `num_hard_conflict_pruned` | skipped only | Pairs rejected because two actions change the same bond |
| `num_invalidated_action_pruned` | skipped only | Pairs rejected because one action removes atoms another needs |
| `num_redundancy_pruned` | skipped only | Pairs whose normalization absorbed a dominated action (not discarded) |
| `num_duplicate_sequence_pruned` | skipped only | Pairs that normalized to their parent or to an already visited history |
| `num_duplicate_effect_pruned` | skipped only | Histories that reused a cached product of an identical effect |
| `num_compiled_sequences` | yes | Unique effects compiled into a reaction |
| `num_rdkit_run_reactants` | yes | RDKit `RunReactants` calls (one per compiled effect) |
| `num_generated_fragments` | yes | Successful single-product reactions |
| `num_unique_fragment_smiles` | yes | Distinct canonical fragment SMILES, excluding the source; bounded by `max_unique_fragment_smiles` |

"skipped only" means the value appears in `skipped_sources.json` but has no
manifest column.

Typical relations:

- `num_raw_combinations` ≥ `num_compiled_sequences` = `num_rdkit_run_reactants`
  ≥ `num_generated_fragments` ≥ `num_unique_fragment_smiles`.
- `num_rdkit_run_reactants` − `num_generated_fragments` is the number of
  effects that failed sanitization.

## Code map

| File | Role |
|---|---|
| `create_training_data.py` | CLI, splits, grouping, serial/parallel orchestration, skip handling, all output files |
| `subprocess_worker.py` | Worker process for one chunk of sources |
| `record_validation.py` | Record-level metadata checks (`invalid_records.json`) |
| `datasets.py` | Input loading, `split_by_smiles`, `dedupe_validation` |
| `context.py` | Fragmenter / graph builder / tensorizer context, `validate_limits` |
| `manifest_summary.py` | Structure-level manifest columns |
| `clefts/ml/input/structure_builder.py` | `ActionStructureBuilder`: tree, precursor, assignment, teacher, targets; `last_search_stats` |
| `clefts/ml/input/source_action_structure.py` | Teacher structure tensors and `.preft.pt` saving |
| `clefts/domain/fragment/cleavage/CleavageActionSearch.py` | Combination search, statistics, `max_cleavage_combinations`, `FragmentTreeLimitExceeded` |
| `clefts/domain/fragment/tree/FragmentTreeBuilder.py` | Materialization, unique SMILES, `max_unique_fragment_smiles`, tree construction |
| `clefts/domain/fragment/ion_tree/FragmentIonTreeBuilder.py` | Ion candidates; attaches `_search_stats` to the ion tree |

User-facing guides: [data preparation](../../../../docs/guides/data-preparation.md)
and [Workbench](../../../../docs/guides/workbench.md). Training details are in
[`fragment_tree_training/README.md`](../../training/fragment_tree_training/README.md).
