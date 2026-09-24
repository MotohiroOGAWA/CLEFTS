# CLEFTS

**Fragment the unknown. From spectra to structures.**

<!-- summary:start -->
CLEFTS is a source-anchored fragmentation framework for learning from experimental MS/MS spectra and predicting spectra from molecular structures. It connects primitive cleavage actions, fragment trees and formula candidates to explain how molecular fragmentation produces a spectrum.
<!-- summary:end -->

## Get started

1. **Prepare data.** Import experimental spectra, map their metadata columns and create source-anchored fragment-tree training structures. Keep training and validation molecules separate.
2. **Train a model.** Fit the action and fragment-intensity models to prepared structures. Start from a compatible checkpoint to fine-tune, or expand a frozen base with additional cleavage patterns in advanced settings.
3. **Predict spectra.** Select a trained model and provide a molecular structure, precursor adduct and collision energy. Run a single prediction or process an MSDataset in batches.
4. **Explore results.** Inspect spectra, molecular structures, fragment trees and cleavage actions. Use recorded metrics and logs to understand model behavior.

## Inputs and conditions

Each training spectrum needs a molecular **SMILES**, precursor **adduct**, **collision energy**, **precursor m/z** and experimental peaks. CLEFTS converts supported collision-energy representations to eV. Values that cannot be parsed must be corrected before processing.

MSDataset (`.msds`), MSP, MGF and spectrum tables are supported for data preparation. CSV, TSV and Parquet tables require a `Peak` column using the msentity spectrum-table format, for example `100.1,0.5;150.2,1.0`. A metadata-only table is not a training spectrum dataset.

## Source-anchored fragmentation

Primitive cleavage actions are defined on the source molecule. Compatible action sets describe retained fragments. A main-adduct-conditioned branch scorer searches normalized action states, then the fragment-tree model predicts collision-energy-conditioned physical-ion intensities after materialization.

Peak assignment uses the configured mass tolerance and precursor-specific ion rules. Action limits, cleavage patterns and adduct rules must be compatible across data preparation, training and prediction.

## Workbench

CLEFTS Workbench provides the same preparation, training, prediction and inspection workflows inside VS Code. It invokes the CLEFTS Python CLI and APIs; scientific processing remains in CLEFTS.

Use **Data → Training Data** to choose train and validation datasets, inspect previews, check column mapping and configure fragmentation, assignment and output. A missing validation dataset can be replaced by a deterministic split of unique SMILES. The Run button explains unmet requirements in red.

Parameter files are optional imports: Browse or drop a JSON file to populate individual fields. Edited values are used for execution. Hover over parameter labels for descriptions; advanced buttons reveal less frequently changed settings.

The Home page provides recent jobs, discovered sample datasets and model checkpoints. Path actions copy real filesystem paths; result actions open the corresponding outputs. Quick forms use the same full training and prediction settings.

## Documentation

The complete Sphinx documentation is in `docs/`. Install `pip install -e '.[docs]'` and build it with `python -m sphinx -W --keep-going -b html docs docs/_build/html`. The Document button opens the built documentation when available and this shared overview otherwise.

## Project

Source and issue reports: [CLEFTS on GitHub](https://github.com/MotohiroOGAWA/CLEFTS).
