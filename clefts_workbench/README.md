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
- load or save an individual pattern as `*.clevage.json`;
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
