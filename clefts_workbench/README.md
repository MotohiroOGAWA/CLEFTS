# CLEFTS Workbench for VS Code

A VS Code extension for configuring, running, and inspecting CLEFTS applications. The first release includes Fragment Tree training-data preparation.

## Development

1. Open this directory in VS Code and press `F5` to launch an Extension Development Host, or run `npm run dev`.
2. Run `CLEFTS: Open Workbench` from the Command Palette.
3. Select the input, edit or load the Fragmenter parameters, select an output directory, and press `Run CLI`.

In a Development Host, the extension watches JavaScript, CSS, HTML, and JSON files below `src/` and `media/`. Saving a change reloads the Development Host after approximately 250 ms. The watcher is disabled in normally installed VSIX builds.

The extension does not duplicate the data-preparation implementation. It invokes the existing Python CLI as a child process:

```bash
python -m clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data ...
```

Set `clefts.pythonPath` when the required Python environment is not available as `python`. The CLEFTS project directory defaults to the parent directory of this extension and can be overridden with `clefts.applicationRoot`.

## Configuration and results

- `Save Configuration` and `Load Configuration` export and import Workbench settings as JSON.
- Fragmenter parameters have a dedicated JSON editor and separate `Load Fragmenter` and `Save Fragmenter` actions.
- Every run also writes `clefts-run-config.json` to its output directory.
- Every output directory receives a `fragment-tree.clefts-result` manifest. Opening it in Explorer displays the run status, structure and table counts, and output-file list in the CLEFTS result viewer.
- Select a JSON or TSV entry in the result viewer to open it in the standard VS Code editor.

## Cleavage Pattern Set editor

Select `Cleavage Pattern Set` on the left side of the Workbench navigation. The tab provides its own `Load Configuration` and `Save Configuration` actions for `*.clevageset.json` documents. It allows you to:

- edit the pattern-set name;
- add and remove patterns;
- edit each pattern name and `reactant_smarts` value;
- add and remove products;
- edit each product name and `smarts` value.
- load or save an individual pattern as `*.clevage.json`;
- build a pattern visually from a SMILES structure using RDKit;
- select atoms and bonds and assign exact, any-heavy-atom, C/N, or C/N/O atom constraints;
- delete atoms and change or remove bonds when generating a product.

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
