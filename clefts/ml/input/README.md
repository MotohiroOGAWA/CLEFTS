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

SMARTS coverage statistics cannot be generated when only an existing structure directory is supplied because the original `MSDataset` metadata is unavailable.

## Main output files

```text
OUTPUT_DIR/
├── config/model_config.json
├── statistics/
│   ├── train_summary.json
│   ├── train_cleavage_pattern_coverage.tsv
│   ├── train_cleavage_pattern_by_class.tsv
│   └── train_matched_pattern_count_distribution.tsv
├── train_structures/
│   ├── data/*.pt
│   ├── manifest.tsv
│   └── assignment_scores.tsv
└── validation_structures/ ...
```

When `--validation-input` is supplied, the corresponding `statistics/validation_*` files are also generated.

## SMARTS statistics

A compound is counted by its unique, non-empty SMILES rather than by its number of spectra. If one compound has spectra at several collision energies, it is still counted only once.

- `*_cleavage_pattern_coverage.tsv`: One row per cleavage pattern, sorted by `matched_compound_count` in descending order. This makes common pattern types such as C-C or S-C easy to inspect. `compound_coverage` is a fraction from 0 to 1, and `substructure_match_count` is the total number of matching substructure positions across all compounds. Patterns with no matches are included.
- `*_cleavage_pattern_by_class.tsv`: Match counts and coverage grouped by `Kingdom`, `Superclass`, `Class`, `Subclass`, and `DirectParent`. Only classification columns available in the input are used. Zero-match combinations are included, making it possible to check whether a specialized SMARTS, such as a flavonoid pattern, has high coverage only in the expected class.
- `*_matched_pattern_count_distribution.tsv`: Distribution of the number of distinct reactant SMARTS types matched by each compound. Compounds matching zero types are included.
- `*_summary.json`: Input record count, unique SMILES count, valid compound count, invalid SMILES count, cleavage pattern count, and the classification columns used.

Coverage is calculated as:

```text
compound_coverage = matched_compound_count / total_compound_count
```

Matching uses each cleavage pattern's `reactant_query`. It does not depend on successful product generation or on the fragment tree `--max-node` and `--max-edge` limits.

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
