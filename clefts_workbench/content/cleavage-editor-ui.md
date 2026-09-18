# Cleavage Pattern Editor UI update

## Changed files

- `src/features/cleavage-pattern-set/visual-editor.js`: new shared Workbench UI and interaction layer.
- `src/features/cleavage-pattern-set/periodic-table.js`: shared 18-group periodic-table data.
- `src/extension.js`: installs the UI after the existing editor script; copies generated strings through the VS Code clipboard.
- `src/features/cleavage-pattern-set/backend.py`: serializes Non-H and three ring states; supports product bond query constraints.
- `scripts/check-cleavage-pattern-ui.js`: extends the existing jsdom interaction tests.
- `scripts/check-cleavage-visual-products.py`: adds RDKit query matching and mapped reaction serialization tests.
- `package.json`: includes the new UI syntax and interaction checks in `npm run check`.
- `package-lock.json`: synchronizes the previously missing declared jsdom dependency.

## Components and state

The new layer shares the selection toolbar, selected-item chips, atom/bond inspector, periodic-table dialog, output/copy field and canvas interactions between both steps. It reuses the existing SVG atom/bond renderer, hit targets, polygon geometry, RDKit parsers/generators, atom mapping, product graph operations, pattern storage and Webview request protocol.

Each editor keeps its mode (`selection` or `edit`), selection shape (`lasso` or `box`), selected atom/bond sets and viewport. Existing structured constraint objects remain separate from generated strings:

- Atom: `mode`, `elements`, `nonHydrogen`; existing raw queries retain `mode: custom` and `smarts`.
- Bond: `mode`, `types`, `ringStatus: any | inRing | notInRing`.
- Product operations retain the existing added-bond records, bond overrides and deleted-atom set.

Custom atom queries use element OR with `;!#1`; Any emits `*`, or `!#1` when its independent Non-H option is enabled. Bond types use OR and append `;@` or `;!@` for ring matching; Any emits `~` without ring restrictions. Non-H defaults to enabled for new element constraints. Ring status follows explicit SMARTS ring predicates, or the RDKit ring flag for an unconstrained source bond. Raw atom SMARTS from imported patterns is preserved.

## Usage

1. Open Cleavage Patterns and choose Add Pattern Visually (or edit an existing pattern visually).
2. Select input type, enter SMILES/SMARTS and choose Draw Structure.
3. Use the pointer icon to toggle atoms/bonds; Shift removes items. Drag a lasso (default), or choose the rectangle icon. Both selected bond endpoints must be selected.
4. Use the pencil icon, then click an already selected atom/bond to edit only that item without changing the selection. Clicking an unselected item in Edit mode has no effect. In selection mode, Inspector changes can apply to all selected atoms or bonds. The element + button opens the periodic table with persistent blue highlights and pressed states; chips remove individual elements/items.
5. Right-drag pans and the wheel zooms in either mode. View buttons zoom or fit the structure.
6. Enter the pattern name and Apply. RDKit generates and validates the reactant before Step 1 collapses and Step 2 opens. The summary includes SMARTS, counts and Copy. Opening either header collapses the other editor.
7. Edit Product with the same selection and inspector tools. The scissors cuts/restores a bond, + connects two clicked atoms with a single bond, and the atom icon deletes/restores an atom. Undo restores the previous edit, including bonds removed by an atom deletion. Use Product bond type in the right inspector to change the actual bond type; Allowed types are SMARTS alternatives (OR). Cut/Delete Bond share one tool because their existing domain operation is identical. Red previews mark deletion and green marks additions.
8. Review operation counts and Reaction SMARTS / Reaction SMIRKS / Product SMARTS tabs; Copy uses the VS Code clipboard. Add/Update Product validates before saving to the pattern.

Reactant generation and product serialization use the existing asynchronous RDKit backend. Live output is debounced and older responses are discarded. Reaction fields compose the mapped reactant/product queries with `>>`.

## Validation and remaining limitations

Passed: expanded cleavage UI tests, RDKit visual-product and query semantics tests, and `npm run check:workbench` (HTML/CSP, settings, jobs, parameters, defaults). Individual SMARTS compound-view and training-metrics checks also passed.

The full existing `npm run check` stops at a missing `src/features/fragment-tree-result/score-distribution` module. Individually running the remaining checks also exposes existing fine-tune command and step-validation assertion failures. These are outside the cleavage editor changes.

No VS Code/browser screenshot verification was performed. Reaction SMARTS and Reaction SMIRKS currently show the same mapped query reaction; strict SMIRKS conversion of arbitrary SMARTS queries is not implemented. Advanced atom creation and charge editing remain future work. Newly added bonds also support selection and editing through the shared inspector.

## Follow-up interaction changes

The molecule area uses `--vscode-editor-background` and `--vscode-editor-foreground`; other panels retain the dark navy theme. RDKit coordinates use one fixed scale on both axes, preserving angles and relative distances; initial/Fit view adds padding without stretching the molecule.

Step 2 shows only Product. A + icon appends products to the list, which highlights the active product. The first + registers the current initial draft; subsequent + adds another product skeleton. Switching products or adding another product validates and saves the current registered product first, so edits are retained.

Added tests verify persistent periodic-table colors and pressed state, Any + Non-H matching, unchanged selection in Edit mode, isotropic coordinates, removed Reactant/change-tool UI, multiple-product registration/switching, deletion undo, actual product bond type updates and new-bond serialization. Cleavage UI/chemistry checks and Workbench checks pass.

## Periodic-table and generation regressions

The second and third periods previously had 19 cells, shifting the following rows. Both element pickers now use the same 18-column data; the inline picker places cells by row/group and scrolls horizontally on narrow screens. SMILES is the default for new input; opening an existing SMARTS pattern still explicitly selects SMARTS.

Generate SMARTS validates and displays the current reactant query without registering a pattern or opening Step 2. Apply continues to register/validate the pattern and advance. Unconstrained source bonds initialize their ring status from RDKit, so clicking a ring bond shows In ring and clicking a chain bond shows Not in ring; explicit radio changes remain editable.

Regression tests cover the period/group positions, SMILES default on reset, Generate staying in Step 1 without registering a pattern, and the exact `CC=CC1CCCCC1` source with real RDKit parsing, ring selection, radio changes and generated `@` constraints.
