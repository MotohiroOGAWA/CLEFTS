# Creating Fragment Tree Training Data

`create_fragment_tree_training_data.py` groups an `MSDataset` by SMILES and creates one training `FragmentTreeStructure` (`.pt`) file for each SMILES group. It also reports how often each cleavage pattern's reactant SMARTS matches the input compounds.

Run the commands below from the repository's `mnt/app` directory.

## Basic example

```bash
cd /workspaces/CLEFTS/mnt/app

python -m clefts.ml.input.create_fragment_tree_training_data \
  --train-input data/minidata/MSMS-Pos-NIST23_cf_mini.msds \
  --params clefts/domain/fragment/presets/fragmenter_pos.json \
  --output-dir data/test/fragment_tree_training_structures \
  --num-workers 1
```

Add `--overwrite --overwrite-model-config` to rebuild existing output:

```bash
python -m clefts.ml.input.create_fragment_tree_training_data \
  --train-input data/minidata/MSMS-Pos-NIST23_cf_mini.msds \
  --validation-input data/minidata/MSMS-Pos-MoNA_cf_mini.msds \
  --params clefts/domain/fragment/presets/fragmenter_pos.json \
  --output-dir data/test/fragment_tree_training_structures \
  --num-workers 4 \
  --chunk-size 32 \
  --save-train-valid-records \
  --overwrite \
  --overwrite-model-config
```

To update previously generated structures using a configuration containing new cleavage patterns:

```bash
python -m clefts.ml.input.create_fragment_tree_training_data \
  --train-structures-input-dir data/old/fragment_tree_training_structures/train_structures \
  --params path/to/new_model_config.json \
  --output-dir data/new/fragment_tree_training_structures \
  --structure-rebuild-policy all-fragments
```

`--structure-rebuild-policy` accepts:

- `root`: rebuild when an added pattern matches the root compound.
- `all-fragments`: rebuild when an added pattern matches any saved fragment. This is the default.
- `always`: always rebuild the structure.

Assigned cleavage-event statistics are also generated when existing structures are used as input, based on the target assignments saved in the rebuilt or copied structures.

## Main output files

```text
OUTPUT_DIR/
├── config/model_config.json
├── statistics/
│   ├── train_assigned_cleavage_events.tsv
│   ├── train_assigned_cleavage_events_by_sample.tsv
│   ├── train_assigned_cleavage_events_by_pattern.tsv
│   ├── train_assigned_cleavage_events_by_pattern_reaction.tsv
│   └── train_assigned_cleavage_events_by_pattern_reaction_product.tsv
├── train_structures/
│   ├── data/*.pt
│   ├── manifest.tsv
│   └── assignment_scores.tsv
└── validation_structures/ ...
```

When `--validation-input` is supplied, the corresponding `statistics/validation_*` files are also generated.

With `--num-workers 2` or greater, each worker writes
`part_*_assigned_cleavage_events.tsv` and
`part_*_assigned_cleavage_events_by_sample.tsv` under the split's parallel
temporary directory immediately after building its chunk. The parent process
only sums the chunk aggregates and concatenates the per-sample rows. It does
not rescan every final `.pt` file after all structure workers finish. The part
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
- `--chunk-size`: Number of SMILES groups assigned to each parallel chunk.
- `--smiles-column`: Name of the SMILES metadata column. The default is `SMILES`.
- `--save-train-valid-records`: Save training records for which structures were successfully generated as an `.msds` file.
- `--no-save-validation-valid-records`: Disable saving valid validation records.
- `--overwrite`: Remove and rebuild existing structure output.

Display all available options with:

```bash
python -m clefts.ml.input.create_fragment_tree_training_data --help
```
