# CLEFTS Workbench for VS Code

A VS Code extension for configuring, running, and inspecting CLEFTS applications. It includes Fragment Tree data preparation and model training.

## Development

1. Open this directory in VS Code and press `F5` to launch an Extension Development Host, or run `npm run dev`. The F5 launch configuration automatically opens the parent CLEFTS application directory, so `Open Folder` is not required on each restart.
2. Run `CLEFTS: Open Workbench` from the Command Palette.
3. Select the input, edit or load the Fragmenter parameters, select an output directory, and press `Run CLI`.

In a Development Host, the extension watches JavaScript, CSS, HTML, and JSON files below `src/` and `media/`. Saving a change reloads the Development Host after approximately 250 ms. The watcher is disabled in normally installed VSIX builds.

The extension does not duplicate the data-preparation implementation. It invokes the existing Python CLI as a child process:

```bash
python -m clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data ...
```

The `Training` tab follows the same workflow: select the generated training and validation structure directories, configure the model and optimizer, then run or copy the existing training CLI command:

```bash
python -m clefts.ml.training.fragment_tree_training.training_model ...
```

Training configurations use the dedicated `*.pfttrain.json` suffix. Every CLI run writes `fragment_tree.pfttrain.json` into its output directory, and the same file can be loaded from the Training tab. The Workbench only passes arguments to the CLI and displays its output; training, configuration output, checkpoints, and model artifacts remain owned by the Python implementation.

Set `clefts.pythonPath` when the required Python environment is not available as `python`. The CLEFTS project directory defaults to the parent directory of this extension and can be overridden with `clefts.applicationRoot`.

## Batch spectrum prediction

In **Predict Spectrum**, the single-molecule preview remains available and a
separate **Batch MSDataset prediction** section accepts an input `.msds` and an
output directory. Select the trained checkpoint and device, specify the source column names,
DB label, chemical precompute batch size, and first-cleavage worker count, then run
or copy the CLI command. The
Workbench invokes:

```bash
python -m clefts.ml.specgen.predict_spectrum --input input.msds --output-dir prediction-output --model model.pt --db MoNA
```

The checkpoint is loaded once. CLEFTS creates a temporary directory next to the
output and invokes a separate Python subprocess worker for each compound batch.
Each worker reconstructs only the non-neural chemistry context, not a copy of
the full prediction model. Neural prediction then expands only model-selected fragments and
finishes that compound before moving on. Compact `tqdm` status is shown in the
tab without appending every refresh to the log; the operation can be stopped
from the Workbench. Output
metadata preserves source values and adds interoperable prediction provenance;
the final `.msds` is written below that directory, and run details plus
record-level failures are written under `<output-dir>/run/`.

**Load Configuration** and **Save Configuration** preserve the model/device,
single-spectrum SMILES, adduct and collision energy, plus every batch prediction
path, column mapping, DB label, chemical batch size, precompute worker count,
temporary-cache option, and overwrite option.

## Configuration and results

- `Save Configuration` and `Load Configuration` export and import Workbench settings as JSON.
- Fragmenter parameters use a reusable form component with separate `Load Fragmenter` and `Save Fragmenter` actions. Add/remove AdductType rules, ion shifts, and atoms with the +/− controls. `Edit Cleavage Pattern Set` opens the existing pattern editor; `Apply to Fragmenter` applies its changes.
- Every run also writes `fragment-tree.pft.json` with input/output settings and embedded `fragmenterParams` to its output directory. Load this file to restore the run. Both Run CLI and Copy Command pass the edited values using `--params-json`; loading a Fragmenter file never makes it an output destination.
- Every output directory receives a `fragment-tree.pft` result manifest. Opening it in Explorer displays the run status and structure manifests in the CLEFTS result viewer.
- Select a JSON or TSV entry in the result viewer to open it in the standard VS Code editor.
- Select an individual `.preft.pt` structure to inspect its fragment-tree drawing,
  samples, peaks, selected
  cleavage depth, formula groups, terminal nodes, ordered edge paths, and
  expansion nodes directly inside the result viewer.
- The Workbench header includes `Open Result` for reopening any generated
  `fragment-tree.pft` file.

## Cleavage Pattern Set editor

Select `Cleavage Pattern Set` on the left side of the Workbench navigation. The tab provides its own `Load Configuration` and `Save Configuration` actions for `*.clevageset.json` documents. It allows you to:

- edit the pattern-set name;
- add and remove patterns;
- edit each pattern name and `reactant_smarts` value;
- add and remove products;
- edit each product name and `smarts` value.
- load or save an individual pattern as `*.cleavage.json` (legacy `*.clevage.json` files remain supported);
- build a pattern visually from a SMILES structure using RDKit;
- draw single, double, triple, and aromatic bonds in a PubChem-style structure view;
- select atoms and bonds individually or with a freehand lasso, and clear the selection explicitly;
- show element symbols and source atom indices on every atom;
- open a dedicated VS Code periodic-table panel with the current elements preselected and return the applied selection;
- use the source element initially or choose any non-hydrogen atom;
- combine single, double, triple, and aromatic bond types with OR (initially the source bond type), and optionally require a ring bond;
- edit the reactant graph directly in Product Transformation to retain/delete atoms and preserve, change, or delete bonds.

The editor reads and writes this JSON shape:

```json
{
  "cleavage_pattern_set": {
    "name": "single_bond_cleavage_pattern_set",
    "patterns": [
      {
        "name": "single_bond_cleavage",
        "reactant_smarts": "[!#1:1]-[!#1:2]",
        "products": [
          {
            "name": "single_bond_cleavage",
            "smarts": "[!#1:1]"
          }
        ]
      }
    ]
  }
}
```

`Apply Reactant` and `Add Product` validate their generated SMARTS/SMIRKS with CLEFTS `_CleavagePattern.from_rules()`. Validation failures appear as VS Code error notifications.

Opening a `*.clevageset.json` file directly in Explorer also uses the dedicated structured editor. Directly opened documents support VS Code save, undo, and redo. Use `Reopen Editor With... > Text Editor` when raw JSON editing is preferred.

## Package

```bash
npm install
npm run check
npm run package
```

Packages are written to `artifacts/clefts_workbench-<version>.vsix`, keeping generated files out of the extension root. Install the generated file with `Extensions: Install from VSIX...` in VS Code.

### Assignment score distribution

The result viewer reads all sample rows from `assignment_scores.tsv` in the result
folder and its `train_structures` / `validation_structures` folders. Select a dataset
in **Assignment Score Distribution** and adjust the bin width (0.001–1; default
0.1 gives 10 bins). Bars show sample counts on the left axis; the line shows the
cumulative percentage from low to high scores on the right axis. Bins include
their lower boundary, with score 1 included in the final bin. Missing, nonnumeric,
and out-of-range scores are excluded and their count is displayed.

Bar, line, background, and text colors are editable. **Save PNG** saves the current
chart at 1200 × 640 pixels; **Save TSV** exports bin bounds, counts, cumulative
counts, and cumulative percentages. The same distribution is available in the
expandable table. Use **Refresh** after score files are generated or updated.

### SMARTS Search

Select **SMARTS Search**, browse for an `.msds` file, enter a SMARTS query
(for example `c1ccccc1`), and press **Run CLI**. The SMILES column defaults to
`SMILES` and can be changed. **Cancel** stops an active search.

Results show matching records and matching unique molecules, with counts and
percentages. Record percentages use all valid SMILES records as the denominator;
unique percentages use valid molecules deduplicated by RDKit canonical isomeric
SMILES. Missing and invalid SMILES are excluded and reported separately. A record
is counted once regardless of the number of substructure matches. Matching uses
RDKit's default substructure behavior (chirality is not required). An empty valid
set displays `N/A` for its percentage.

The Workbench only invokes the CLEFTS CLI and displays its JSON output. Search
and counting logic live in the Python application, not the extension. **Copy
Command** copies the same command and arguments used by **Run CLI**, using
`clefts.pythonPath`. Run the copied command from the application directory shown
in the status message (`mnt/app` by default).

The CLI can also be used independently from the application directory:

```bash
python -m clefts.cli molecule smarts-search --input /path/to/data.msds --smarts 'c1ccccc1' --smiles-column SMILES
```

An installed CLEFTS environment also supports `clefts molecule smarts-search`.
Results go to stdout as JSON; errors go to stderr with a nonzero exit code.

Run `python scripts/check-smarts-search.py` with that environment to check the
counting behavior.

Run `node scripts/check-smarts-cli.js` to verify CLI results, failures, and copied
command quoting with the CLEFTS Python environment available as `python`.

Search results include separate **Detected compounds** and **Not detected
compounds** lists. Each row shows a canonical isomeric SMILES and the number of
source records for that molecule. Lists show 50 molecules per page and support
SMILES text filtering. Missing and invalid SMILES remain excluded from both lists.

Workbench runs and copied commands include `--include-compounds`. This CLI flag
adds a `compounds` array to the JSON result; each entry contains `smiles`,
`matched`, and `records`. Without the flag, CLI output remains summary-only.

Use **Add reactant SMARTS** / **Remove** to edit multiple queries (at least one
is required). The summary compares every query with **Any reactant SMARTS (OR)**,
for both record counts/percentages and unique-molecule counts/percentages. The OR
row counts each record or molecule only once even when multiple queries match.
All rows use the same valid-SMILES denominators. Select **Compound lists for** to
inspect the detected/undetected lists for the union or an individual query.

Repeat `--smarts` when using the CLI:

```bash
python -m clefts.cli molecule smarts-search --input data.msds --smarts O --smarts N --include-compounds
```

Top-level JSON counts describe the OR union. `patterns` contains each query's
counts, percentages and zero-based `index`. Compound entries include
`matchedPatterns` (matching query indices); `matched` indicates any query match.
The file is loaded once and each distinct input SMILES is parsed once per run.

### Fragment Tree Fine-tuning

Open **Fine-tuning**, select a base fragment-tree `model.pt`, a complete expanded
Cleavage Pattern Set, and training/validation splits regenerated with that set.
Select a separate output directory. All old parameters and MolEncoder are frozen;
only new category embeddings and small per-projection low-rank expansions train.
**Added nodes per projection** defaults to 8. The original model dimensions stay
the same. Use representative old compounds as well as new-pattern examples when
checking performance.

**Validate only** checks preprocessing/configuration and checkpoint compatibility
and reports trainable/frozen parameter counts. **Run Fine-tuning CLI** invokes
`python -m clefts.cli train fragment-tree-finetune`; **Copy Command** copies the
same arguments. Logs stream into the tab and the CLEFTS output channel. The new
set must include all old definitions, and other Fragmenter settings must match
the base. The validation split must include `valid_records.msds`.
**Load Configuration** and **Save Configuration** preserve all file paths and
fine-tuning hyperparameters in an editable, versioned JSON document.

See `../clefts/ml/training/fragment_tree_training/README.md` for CLI usage,
checkpoint/resume behavior, and the exact expansion architecture.

### Training metrics

Open **Metrics** in Workbench, enter the absolute training run directory, and click
**Load / Refresh**. Select a metric and click **＋ Add chart** to add charts; use
**×** to remove them. Click **Add all charts** to add every available chart at
once. Enter a search in **Filter charts** to use **Add matching charts** instead.
**Clear charts** removes all selected charts. Set **Layout** to **Vertical** to stack all charts in one
column, or **Grid** to arrange them side by side. The layout selection is retained
in the webview state. **Auto refresh (15s)** updates charts while the tab is visible.
The viewer reads `metrics.tsv` and `metric_distributions.tsv` directly. Distribution charts combine min–max and q1–q3 bands with a median line over global
steps. Mean charts group adducts (or CE ranges) for comparison. Each chart combines
train, train_window, and validation, with colors and line styles plus toggles in the
legend. These are recorded summary statistics, rather than raw sample histograms. Non-finite and missing values are omitted.
Hover over a chart to inspect its nearest recorded step and value. The run path and
selected charts are retained in the webview state.
