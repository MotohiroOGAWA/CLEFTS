# Prepare fragment-tree training data

Choose experimental spectra with SMILES, precursor adduct, collision energy, precursor m/z and peak lists. Supported inputs are `.msds`, `.msp`, `.mgf`, `.tsv`, `.csv` and `.parquet`. Tables require a `Peak` column in msentity spectrum-table syntax. Preview and column validation use the same input loader as preparation.

## Train and validation datasets

```bash
python -m clefts.cli train create-fragment-tree-data \
  --input train.msds --validation-input validation.msds \
  --output-dir data/prepared \
  --params model.json
```

With an explicit validation dataset, outputs are written to `train_structures` and `validation_structures`. To create a held-out split by unique SMILES, replace `--validation-input` with `--validation-ratio 0.1 --validation-seed 0`. Molecules are kept within a single split. At least two distinct SMILES are needed for automatic splitting.

Without either validation option, the CLI retains the single-dataset behavior and writes structures directly to the output directory.

## Column mapping

Map actual metadata names with `--smiles-column`, `--adduct-type-column`, `--collision-energy-column` and `--precursor-mz-column`. Workbench shows whether each mapped column exists in both selected datasets and whether the values are valid. Selecting a file loads the preview and checks column presence only. Click **Validate Dataset Values** to check all records. SMILES are parsed by RDKit; adducts must resolve to a main adduct registered in the fragmenter; collision energy uses the CLEFTS parser, including supported strings; precursor m/z must convert to a finite positive number. Input, mapping or fragmentation changes invalidate this check.

## Fragmentation and assignment

Edit the maximum action count and precursor action limit directly. **Cleavage Pattern Set** and **Fragment Ion Adduct Rule Set** are collapsed summaries that expand their shared dedicated editors inline, without leaving Fragmentation. The collapsed Pattern summary shows its set name; the collapsed adduct summary shows its set name and comma-separated main adducts. Adduct-rule sets can also be created under **Data → Adduct Rules**; sets use `*.adductset.json` and individual rules use `*.adduct.json`. Data Preparation exposes element **Symbols** through a periodic table; it requires no neural model dimensions or separate adduct embedding list. The adduct embedding order is resolved automatically from the Fragmenter rules and observed dataset metadata; there is no CLI option for setting it independently. The CLI accepts `--symbols-json '["C","N","O","P","S"]'`.

`--max-node` limits unique fragment nodes after identical fragments are merged, including the source. `--max-edge` limits the total distinct cleavage transitions summed over all node pairs. Multiple routes between the same nodes count separately. Both default to `-1` (unlimited); limits fail explicitly rather than silently truncating prepared data. Mass tolerance controls peak-to-formula assignment, for example `--mass-tolerance 0.01Da,10ppm`. The minimum relative peak intensity option filters low-intensity experimental peaks before assignment. Normalization is enabled by default and scales each spectrum's maximum retained intensity to 1.

## Parallel processing

Use `--num-workers 4 --chunk-size 1` to prepare independent SMILES groups in subprocesses. Workbench exposes these under **Output → Parallel Processing**. One worker is the default and runs serially. Worker outputs retain the same source indices and filenames as serial preparation.

## Configuration precedence

Preset defaults are followed by `--params` file values, inline `--params-json`, section JSON options, named options and repeated `--set PATH=JSON` overrides, in that order.

```bash
python -m clefts.cli train create-fragment-tree-data \
  --input train.msds --output-dir data/prepared \
  --params model.json --max-action-count 2 \
  --mass-tolerance 0.02Da \
  --max-node 10000 --max-edge 50000
```

Workbench passes the edited configuration snapshot, so a later form edit takes priority over an imported file.

## Outputs

Structures are saved as `.preft.pt` files under `train_structures/data` and `validation_structures/data`, with source and record metadata. Single-dataset outputs use `<output>/data`. Each structure directory contains `manifest.tsv` listing SMILES, files, record counts and skip reasons for the result tables. `action_statistics.json` records counts and configuration. Existing structure files are rejected unless overwrite is explicitly enabled. With overwrite enabled, the entire specified Output Directory is cleared before generating fresh outputs. Input and application directories cannot be used as overwrite destinations. JSON Lines progress events let Workbench display the current source and split.

Invalid SMILES, unsupported or malformed adducts, unparseable collision energies and invalid precursor m/z values exclude the affected records. They do not block preparation when usable records remain. **Invalid Records** in Workbench shows the original zero-based record number, ID, column value and exclusion reason, with search and pagination. A record with multiple invalid fields is excluded once. Required columns must still exist, and each input must contain usable records. Automatic splitting requires at least two usable unique SMILES.

The CLI applies the same checks before splitting or collecting adduct conditions. `invalid_records.json` in the output root lists excluded records from each original input for review.

A source tree that exceeds `max_node` or `max_edge` is skipped, and preparation continues with the next source, including with multiple worker processes. `skipped_sources.json` lists skipped SMILES, record indexes and reasons. `action_statistics.json` records successfully prepared records as `num_samples`, metadata-valid records as `num_metadata_valid_records`, and skipped source/record counts separately.

The parent process displays a `tqdm` progress bar on stderr, including prepared/skipped source counts, and also emits JSON Lines for Workbench. Progress advances for each attempted source, including skipped sources.

Resolved preparation settings are saved as `preparation_config.json` directly in Output Directory before computation in both single-dataset and train/validation modes. Preparation configuration files are not written inside `train_structures` or `validation_structures`. Workbench additionally saves `fragment-tree.pft.json` in the output root before starting the CLI; use **Load Configuration** to restore the preparation form from this file.

When `H` is absent from **Symbols**, explicit hydrogen atoms are omitted from encoder graphs; attached explicit hydrogen neighbors are included in the heavy atom's hydrogen-count feature. The original chemical molecule, formula and valence are preserved. This also permits hydrogen-only materialized fragments to have an empty encoder graph. Selecting `H` keeps hydrogen nodes in the graph. Source action indexes are remapped to the represented graph atoms.

Parallel preparation uses `clefts.utils.parallel_subprocess.run_parallel_subprocesses` to launch independent Python commands for chunks of SMILES groups. Each subprocess initializes its own RDKit preparation context. The common utility displays tqdm progress and notifies the parent after a successful chunk; the parent updates source counters and collects saved results. Parallel progress consumes completed chunks immediately, so a slow earlier chunk does not hold back updates from later completed chunks. Output filenames and the returned file list remain deterministic. The counter advances when a chunk completes; a molecule still being computed is not counted as complete.

Both root configuration files can be loaded by Workbench. Loading restores training/validation input paths, output path, column mapping, split settings, intensity options, overwrite, worker/chunk settings, symbols, node/edge limits and all fragmentation parameters. Legacy Python snake_case keys are converted to the Workbench form keys. Dataset value checks must be rerun after loading.

Source action combinations are explored lazily. Each valid combination is materialized and its unique SMILES node and distinct transitions are counted before requesting further combinations. Exceeding a node or edge limit raises `FragmentTreeLimitExceeded` immediately, stopping additional combination exploration and RDKit reactions for that source. Alternative transitions between merged nodes count toward the edge limit as soon as their predecessor and target are available.

The result viewer reads each split manifest to display the structures. For older outputs missing `manifest.tsv`, it reads the saved structure metadata from either the legacy flat directory or its `data` subdirectory. Training recursively discovers `.preft.pt` files, so select `train_structures` and `validation_structures` as before.
