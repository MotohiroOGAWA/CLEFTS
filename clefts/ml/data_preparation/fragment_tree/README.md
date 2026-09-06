# Creating Fragment Tree Training Data

`create_fragment_tree_training_data.py` groups an `MSDataset` by SMILES and creates one training `FragmentTreeStructure` (`.preft.pt`) file for each SMILES group. Executable dataset-building workflows and reporting helpers live in `clefts.ml.data_preparation`; reusable dataset and structure classes remain in `clefts.ml.input`.

Run the commands below from the repository's `mnt/app` directory.

## Basic example

```bash
python -m clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data \
  --train-input path/to/training_data.msds \
  --params path/to/fragmenter_params.json \
  --symbols C N O P S F Cl Br I \
  --output-dir path/to/output_directory \
  --num-workers 1
```

Add `--overwrite --overwrite-model-config` to rebuild existing output:

```bash
python -m clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data \
  --train-input path/to/training_data.msds \
  --validation-input path/to/validation_candidates.msds \
  --validation-smiles-ratio 0.1 \
  --params path/to/fragmenter_params.json \
  --symbols C N O P S F Cl Br I \
  --output-dir path/to/output_directory \
  --num-workers 4 \
  --chunk-size 32 \
  --save-train-valid-records \
  --overwrite \
  --overwrite-model-config
```

Choose positive- or negative-mode fragmenter parameters explicitly. Neural-network
dimensions, layer counts, and dropout are not preprocessing inputs.

Both `--train-input` and `--output-dir` are required. Reusing the same output directory without `--overwrite` resumes training: existing SMILES structure files are skipped and missing ones are built. Validation sampling starts only after this training pass has completed, and its maximum Tanimoto values are calculated against the SMILES in the completed files under `train_structures/data`.

The immutable preprocessing settings are saved in
`config/preprocessing_config.pftprep.json`. They contain `symbols`, normalized
`fragmenter_params`, `max_node`, and `max_edge`; they do not contain neural-network
dimensions, layer counts, or dropout. Training validates the model's symbols and
fragmenter definitions against this file before constructing the model.

The `.pftprep.json` double extension is CLEFTS's dedicated marker for this file
kind (matching `.pft.json`/`.pfttrain.json`/`.clevageset.json` elsewhere in the
Workbench), not just a plain `.json`. Structure directories created before this
convention was introduced still load correctly: every reader
(`find_previous_preprocessing_config`, `load_and_validate_split_preprocessing`,
`validate_preprocessing_compatibility`, and the Workbench's fragment-tree-result
inspector) falls back to the legacy plain `preprocessing_config.json` name when
the dedicated one is absent. New runs always write the dedicated name.

## Main output files

```text
OUTPUT_DIR/
├── config/preprocessing_config.pftprep.json
├── statistics/
│   ├── train_assigned_cleavage_events.tsv
│   ├── train_assigned_cleavage_events_by_sample.tsv
│   ├── train_assigned_cleavage_events_by_pattern.tsv
│   ├── train_assigned_cleavage_events_by_pattern_reaction.tsv
│   └── train_assigned_cleavage_events_by_pattern_reaction_product.tsv
├── train_structures/
│   ├── fragmenter.json
│   ├── data/*.preft.pt
│   ├── manifest.tsv
│   └── assignment_scores.tsv
└── validation_structures/
    ├── fragmenter.json
    └── ...
```

When `--validation-input` is supplied, the corresponding `statistics/validation_*` files are also generated.
Each generated structure split contains a `fragmenter.json` in the native
`Fragmenter` format. It can be loaded directly with
`Fragmenter.from_json(".../train_structures/fragmenter.json")` (or the validation
equivalent) and used to reproduce fragmentation with the split's settings.

By default, `--validation-smiles-ratio` is `0.1`: the number of selected validation SMILES is 10% of the number of unique training SMILES (rounded up and capped by the available validation SMILES). Candidates are divided into ten bins by their maximum Morgan-fingerprint Tanimoto similarity to training data, then selected round-robin across non-empty bins. Thus similarity 1.0 (also present in training) through structurally dissimilar candidates near 0.0 are represented as evenly as the candidate pool allows. The selection is deterministic by default; use `--validation-sampling-seed` to change it. `validation_structures/max_tanimoto_index.tsv` records the selected SMILES and similarities.

With `--num-workers 2` or greater, each worker writes
`part_*_assigned_cleavage_events.tsv` and
`part_*_assigned_cleavage_events_by_sample.tsv` under the split's parallel
temporary directory immediately after building its chunk. The parent process
only sums the chunk aggregates and concatenates the per-sample rows. It does
not rescan every final `.preft.pt` file after all structure workers finish. The part
files are removed with the other parallel temporary files unless
`--keep-parallel-temp` is specified.

## Assigned cleavage-event statistics

These statistics count cleavage events in `FragmentPathway` objects that were actually assigned to observed spectrum peaks. They do not count every SMARTS match in the root compound or every event available in the shared fragment-tree structure.

A structure can contain many spectrum samples for the same SMILES. Each target assignment is traced separately through:

```text
sample -> observed peak -> assigned FragmentPathway -> pathway edges -> cleavage events
```

This ensures that samples sharing one structure are counted separately. If the same pathway is assigned in multiple samples, its events contribute once for each assignment in each sample. Shared pathway prefixes are therefore counted according to their actual use by assigned pathways.

- `*_assigned_cleavage_events.tsv`: Aggregate counts sorted by `assigned_cleavage_event_count` in descending order. `reactant_smarts` identifies the cleavage pattern, while `matched_substructure` describes the actual matched atoms, such as `C-C`, `C-N`, or `C-O`. `assigned_pathway_count` is the number of assigned pathways containing that type, and `sample_count` is the number of distinct samples in which it occurs.
- `*_assigned_cleavage_events_by_sample.tsv`: Per-sample detail used to produce the aggregate table. It includes the source record index, `SpecID` when the input dataset provides it, structure filename, structure-local sample index, SMARTS, matched substructure, and event count.
- `*_assigned_cleavage_events_by_pattern.tsv`: Counts grouped by `pattern_id`.
- `*_assigned_cleavage_events_by_pattern_reaction.tsv`: Counts grouped by `(pattern_id, reaction_id)`.
- `*_assigned_cleavage_events_by_pattern_reaction_product.tsv`: Counts grouped by `(pattern_id, reaction_id, product_molecule_id)`.

For example:

```text
reactant_smarts       matched_substructure  assigned_cleavage_event_count
[!#1:1]-[!#1:2]      C-O                   1000
```

The compact label uses the actual atom and bond types at the matched reactant atom indexes. Single, double, triple, and aromatic bonds are represented by `-`, `=`, `#`, and `:`, respectively.

## Common options

- `--max-node`, `--max-edge`: Fragment tree size limits. Use `-1` for no limit.
- `--num-workers`: Use two or more subprocess workers to process SMILES groups in parallel. SMARTS statistics are calculated once per split by the parent process.
- `--chunk-size`: Number of SMILES groups assigned to each parallel chunk. When `num_workers * chunk_size` exceeds the split's unique SMILES count, it is automatically reduced to `max(1, unique_smiles_count // num_workers)`. Training and validation are adjusted independently.
- `--smiles-column`: Name of the SMILES metadata column. The default is `SMILES`.
- `--symbols`: Element symbols that define the atom-feature columns. This is required and immutable for generated data.
- `--validation-smiles-ratio`: Target validation count divided by unique training SMILES count. The unit is SMILES, not spectrum records.
- `--tanimoto-num-bins`: Number of intervals used to balance maximum Tanimoto similarity (default: 10).
- `--save-train-valid-records`: Save training records for which structures were successfully generated as an `.msds` file.
- `--no-save-validation-valid-records`: Disable saving valid validation records.
- `--overwrite`: Remove and rebuild existing structure output.

Display all available options with:

```bash
python -m clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data --help
```
