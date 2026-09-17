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

Edit the maximum action count, precursor action limit, cleavage patterns and fragment ion adduct rules. The clickable import area opens Browse and accepts dropped JSON files, including cleavage pattern sets and fragment ion adduct rule sets. Data Preparation exposes element **Symbols** through a periodic table; it requires no neural model dimensions or separate adduct embedding list. The CLI accepts `--symbols-json '["C","N","O","P","S"]'`.

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

Structures are saved as `.preft.pt` files, with source and record metadata. `action_statistics.json` records counts and configuration. Existing structure files are rejected unless overwrite is explicitly enabled. JSON Lines progress events let Workbench display the current source and split.
