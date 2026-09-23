# CLEFTS Workbench

Install the packaged VSIX in VS Code, open the CLEFTS workspace and run **CLEFTS: Open Workbench**. Configure `clefts.pythonPath` and, if automatic discovery does not locate the project, `clefts.applicationRoot`.

The sidebar contains **Home**, **Data**, **Training**, **Prediction**, **Visualization** and **Settings**. Documentation, GitHub, theme and help controls remain in the top toolbar.

## Home

The product description comes from the same overview included in this documentation. Quick-start actions lead to data preparation, training, prediction and visualization. The resource tabs show real jobs, sample datasets and discovered checkpoints; use their actions to select a path, copy it or inspect results.

Quick training requires prepared train/validation structure directories, output, epochs, batch size and learning rate. An optional compatible pretrained checkpoint initializes a fine-tuning run. Quick prediction transfers model, SMILES, adduct and collision energy to the full prediction workflow and uses the model's supported conditions.

## Data preparation

The tabs are **Input Dataset**, **Fragmentation**, **Assignment**, **Output** and **Run**. Dataset inspection displays file size, record counts, unique SMILES, adduct distribution and bounded record/spectrum previews. Column validation checks train and validation datasets. When validation input is omitted, select the unique-SMILES split ratio and seed.

The fixed bottom-right **Run** button is always clickable. Clicking it checks dataset validation and required settings; red text explains any unmet requirements and processing does not start. Clicking during an active job also displays a message and prevents duplicate runs. The backend validates settings again before launching processing.

## Cleavage patterns

Open **Data → Cleavage Patterns**, or run **CLEFTS: Edit Cleavage Patterns** from the command palette. Creation controls appear above the pattern list. **Add Pattern** creates an editable reactant/product SMARTS entry. **Add Pattern Visually** opens the existing structure builder: draw a source molecule, select atoms and bonds, configure their queries, then **Apply Reactant**. While dragging with the left mouse button held down, a colored dashed line follows the mouse in both the source and product views. When drawing a selection loop, a bond is selected only when both of its endpoint atoms are enclosed. Draw and map each product, edit bond types, and use **Add Product**. Existing entries also support visual editing.

Import a pattern set or an individual pattern by dropping its JSON file onto the corresponding import control, or click that control to browse. **Save Configuration** exports the complete pattern set; individual entries can also be saved separately. Import errors appear in the editor.

To update a preparation run, open **Training Data → Fragmentation → Cleavage Patterns** and choose **Edit Cleavage Patterns** or **Add Pattern Visually**. The editor starts with that run's current pattern set. After adding, editing or importing patterns, click **Apply Patterns to Parameters** to return to Data Preparation and include the edited set in the preparation configuration. Saving a standalone pattern set does not apply it to a preparation run.

## Mol Training descriptor targets

In **Mol Training → Training Tasks & Loss Weights → Descriptor Regression**, select individual descriptor targets using the checkboxes. The standard 14 training descriptors are all checked by default. Each target includes a short description. **Select All** and **Clear All** update the selection together.

Keep at least one descriptor selected while Descriptor Regression is enabled. Disabling the task also disables its target controls without discarding the selection. Copy Command passes the selected names through `--descriptor-names`; Start Training uses the same selection. Save Configuration records the selected names, and Load Configuration restores the selection and its order, including configurations exported by the training CLI.

## Mol Training startup and progress

A `Validation target warning` reports node or edge classes with too few validation targets. It is informational and does not block training. Sampling-index construction follows this summary and displays its progress.

Mol Training starts optimizer updates without an epoch-0 evaluation by default. Earlier versions silently evaluated the entire training split before the first update, which could take a long time on large CPU datasets. To record initial subset metrics, set **Initial evaluation batches (0 skips)** in Basic Training Settings, or pass `--initial-evaluation-batches N` to the CLI. A positive value limits initial evaluation to at most N batches from each split; 0 skips it. These subset metrics are excluded from best-checkpoint selection and early stopping. Full validation continues after each training epoch.

Logs include training and epoch start, first-batch start, completed-batch progress, and epoch-end metrics. Validation also displays a progress bar. TensorBoard records batch training loss as well as epoch metrics. If an individual first forward pass is slow, its `batch_start` event distinguishes it from preprocessing or initial evaluation. The CLI's `--device cpu` selects CPU execution; `--device cuda` selects CUDA execution.

## Themes and help

Theme preference is saved in the webview's workspace state. The default follows the VS Code theme; the toolbar toggles light/dark overrides. Help explains the current page and links to the product guide.


## Workbench defaults

Open **Settings → Preferences** to edit CLEFTS settings. **Default Output Directory**
is initially empty, so output fields start blank. When configured, new forms use
`<root>/preparation-<timestamp>`, `<root>/training-<timestamp>` and
`<root>/prediction-<timestamp>`. Relative roots resolve against the CLEFTS project.
Existing form edits and restored job/configuration paths remain under your control.

**Save as Workbench Defaults** on Data, Training or Prediction stores the current
parameter values in VS Code workspace settings (user settings if no workspace is
open). Defaults apply to newly opened Workbench forms. Generated output paths are
excluded from these snapshots. You can also edit the three default objects directly:

```json
{
  "clefts.workbench.defaultOutputDirectory": "/data/clefts-runs",
  "clefts.workbench.dataDefaults": {
    "normalizeIntensities": true,
    "numWorkers": 4,
    "chunkSize": 1,
    "maxNode": 10000,
    "maxEdge": 50000,
    "symbols": ["C", "N", "O", "P", "S"],
    "fragmenterParams": {"mass_tolerance": "0.02Da,10ppm"}
  },
  "clefts.workbench.trainingDefaults": {"epochs": 100, "batchSize": 16, "lr": 0.0001},
  "clefts.workbench.predictionDefaults": {"ce": 30, "adductType": "[M+H]+"}
}
```

Nested parameter objects merge with the built-in preset; explicit form edits and
loaded configurations take priority. Normalize Intensities defaults to enabled.

## Parallel data preparation

Open **Data → Output → Parallel Processing**. **Worker Processes** selects the
number of independent subprocesses processing SMILES groups; `1` runs serially.
**Chunk Size** selects how many groups are handled by each subprocess command.
Data preparation uses the shared `clefts/utils/parallel_subprocess.py` runner. Chunks are capped to
keep enough work for the available workers. Training and validation run separately.
The CLI equivalents are `--num-workers 4 --chunk-size 1`. Stopping preparation also
terminates its worker processes.

## Fragment Tree Dataset Results

Use the preparation page's top-right **Load Configuration** drop area to select `fragment-tree.pft.json` by clicking or dragging from the filesystem or Explorer. The preparation heading does not display a fixed preset name.

The result viewer has a toolbar and light/dark toggle, without a sidebar. Select an output file to inspect its stored fragment trees. **Structure Detail** shows total nodes, edges, transitions, spectra, branch groups, physical-ion candidates, explanations, primitive actions and teacher states. The tree and table display one entry per materialized fragment node, with unique node IDs. In-memory trees without assigned persistent IDs use their local node indices, indicated in the legend. Diamonds mark precursors; orange circles mark terminal branch nodes.

Click a node or its table entry to open a separate molecule panel. Later selections update the same panel. Both molecule views and the tree have zoom controls. Expand a node's **Action sets** to inspect alternative normalized histories. Action links select the corresponding entry in **Action Reference**, which provides cleavage pattern, reactant/reaction and product molecule IDs, query-ordered Source atom maps, retained/discarded atom maps and changed/cut bonds. The mapped Original Source drawing and saved SMARTS definitions provide the reference for these IDs. Click an edge to inspect its source and target nodes, added action and parent/target action sets.


Each selected spectrum shows its ion adduct followed by the main adduct in parentheses,
for example `[M+H-H2O]+([M+H]+)`. Collision energy is rounded to one decimal place
using half-up rounding. Detected fragments use an orange outline with no fill;
precursors use a green diamond outline.

The **Peak Assignments** table contains one row per retained experimental peak,
including its m/z and intensity. Annotation cells remain empty for unmatched peaks.
Matches include fragment node IDs, SMILES, formula, theoretical m/z, mass error
in ppm, hydrogen shift relative to the main adduct, and ion adduct. Multiple
annotations stay in the same peak row. Intensities reflect the preparation
normalization setting.

Preparation saves two intensity-coverage scores for each sample: assigned peak
intensity divided by total peak intensity, and the same ratio after removing
peaks matching the precursor m/z within the configured mass tolerance from both
the numerator and denominator. Each peak contributes once even when it has
multiple matches. A zero denominator produces an unavailable score. Scores are
stored in the structure and in `assignment_scores.tsv` directly under each split
directory. Existing datasets without saved peak records must be regenerated
to display the peak table and both scores.


Open `train_structures/fragment-tree.pft.json` or
`validation_structures/fragment-tree.pft.json` to view that split independently.
These files also retain the preparation settings for import. The Python
`preparation_config.json` remains at the output root. A training-only CLI run
uses `train_structures` as well.

**Assignment Score Distribution** shows a histogram for the selected split.
Choose **All peaks** or **Without precursor peaks** and change **Bins** from 1
to 100. Scores of 100% belong to the final bin; unavailable scores are counted
separately. Hover over a bar to see its interval and sample count. Light mode
uses a white page background.


The Output Files table includes rejected sample counts, a rejection-log link
when records were skipped, and total stored node/edge counts across the samples
in each structure. Both assignment-score columns show the mean of available
per-sample scores; select the structure and a spectrum to see individual scores.
A zero count or score is displayed explicitly. Missing values in older
manifests are recovered from saved structures and assignment-score TSV files.
Workbench light mode uses white page, panel, toolbar and navigation surfaces.
