const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const cleavagePatternSetEditor = require('./features/cleavage-pattern-set/editor');

const RESULT_NAME = 'fragment-tree.pft';
let runningProcess;

function activate(context) {
  const output = vscode.window.createOutputChannel('CLEFTS');
  const provider = new ResultEditorProvider(context);
  context.subscriptions.push(
    output,
    vscode.commands.registerCommand('clefts.openWorkbench', () => openWorkbench(context, output)),
    vscode.commands.registerCommand('clefts.openResult', openResultPicker),
    vscode.window.registerCustomEditorProvider('clefts.resultViewer', provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: true
    })
  );
  cleavagePatternSetEditor.register(context);
  enableDevelopmentReload(context, output);
}

function enableDevelopmentReload(context, output) {
  if (context.extensionMode !== vscode.ExtensionMode.Development) return;
  const pattern = new vscode.RelativePattern(context.extensionPath, '{src,media}/**/*');
  const watcher = vscode.workspace.createFileSystemWatcher(pattern);
  let reloadTimer;
  const reload = uri => {
    if (!/\.(js|css|html|json)$/.test(uri.fsPath)) return;
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(() => {
      output.appendLine(`[development] ${path.relative(context.extensionPath, uri.fsPath)} changed; reloading Extension Host.`);
      vscode.commands.executeCommand('workbench.action.reloadWindow');
    }, 250);
  };
  watcher.onDidChange(reload);
  watcher.onDidCreate(reload);
  watcher.onDidDelete(reload);
  context.subscriptions.push(watcher, { dispose: () => clearTimeout(reloadTimer) });
  output.appendLine('[development] Auto reload is enabled.');
}

function projectRoot(context) {
  const configured = vscode.workspace.getConfiguration('clefts').get('applicationRoot', '').trim();
  const workspaceFolders = vscode.workspace.workspaceFolders || [];
  if (configured) {
    const base = workspaceFolders[0] ? workspaceFolders[0].uri.fsPath : process.cwd();
    const candidate = path.resolve(base, configured);
    if (isCleftsRoot(candidate)) return candidate;
    throw new Error(`clefts.applicationRoot does not point to a CLEFTS application: ${configured}`);
  }
  const searchRoots = [
    ...workspaceFolders.map(folder => folder.uri.fsPath),
    context.extensionPath
  ];
  for (const searchRoot of searchRoots) {
    let current = searchRoot;
    while (true) {
      for (const candidate of [current, path.join(current, 'mnt', 'app'), path.join(current, 'app')]) {
        if (isCleftsRoot(candidate)) return candidate;
      }
      const parent = path.dirname(current);
      if (parent === current) break;
      current = parent;
    }
  }
  throw new Error('CLEFTS application root was not found. Open the CLEFTS workspace or set clefts.applicationRoot to a workspace-relative path.');
}

function isCleftsRoot(candidate) {
  return fs.existsSync(path.join(candidate, 'clefts')) && fs.existsSync(path.join(candidate, 'pyproject.toml'));
}

function defaultConfig(context) {
  const root = projectRoot(context);
  const defaultParams = path.join(root, 'clefts', 'domain', 'fragment', 'presets', 'fragmenter_single_bond_pos.json');
  return {
    application: 'fragment-tree-data-preparation',
    trainInput: '', validationInput: '', validationStructuresInputDir: '',
    validationSmilesRatio: 0.1, tanimotoNumBins: 10, tanimotoRadius: 2,
    tanimotoNBits: 2048, validationSamplingSeed: 0,
    outputDir: path.join(root, 'data', 'training', 'fragment_tree_projects', 'gui-run'),
    params: defaultParams, symbols: ['C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'],
    smilesColumn: 'SMILES', precursorMzColumn: 'PrecursorMZ', adductTypeColumn: 'AdductType',
    collisionEnergyColumn: 'CollisionEnergy', instrumentColumn: '', maxNode: -1, maxEdge: -1,
    numWorkers: 1, chunkSize: 32, structureRebuildPolicy: 'all-fragments',
    overwrite: false, overwritePreprocessingConfig: false, saveTrainValidRecords: false,
    saveValidationValidRecords: true, keepParallelTemp: false
  };
}

function openWorkbench(context, output) {
  const panel = vscode.window.createWebviewPanel('clefts.workbench', 'CLEFTS Workbench', vscode.ViewColumn.One, {
    enableScripts: true, retainContextWhenHidden: true
  });
  const initialConfig = defaultConfig(context);
  let initialFragmenter = '{}';
  try { initialFragmenter = fs.readFileSync(initialConfig.params, 'utf8'); } catch (_) {}
  panel.webview.html = workbenchHtml(initialConfig, initialFragmenter);
  panel.webview.onDidReceiveMessage(async message => {
    try {
      if (message.type === 'pick') {
        const folders = message.kind === 'folder';
        const picked = await vscode.window.showOpenDialog({ canSelectFiles: !folders, canSelectFolders: folders, canSelectMany: false });
        if (picked && picked[0]) {
          panel.webview.postMessage({ type: 'picked', field: message.field, value: picked[0].fsPath });
          if (message.field === 'params') {
            const text = await fs.promises.readFile(picked[0].fsPath, 'utf8');
            JSON.parse(text);
            panel.webview.postMessage({ type: 'fragmenter', path: picked[0].fsPath, text });
          }
        }
      } else if (message.type === 'loadFragmenter') {
        const picked = await vscode.window.showOpenDialog({ filters: { 'Fragmenter parameters': ['json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const text = await fs.promises.readFile(picked[0].fsPath, 'utf8');
          JSON.parse(text);
          panel.webview.postMessage({ type: 'fragmenter', path: picked[0].fsPath, text });
        }
      } else if (message.type === 'saveFragmenter') {
        JSON.parse(message.text);
        const target = await vscode.window.showSaveDialog({ filters: { 'Fragmenter parameters': ['json'] }, defaultUri: message.path ? vscode.Uri.file(message.path) : vscode.Uri.file('fragmenter-params.json') });
        if (target) {
          await fs.promises.writeFile(target.fsPath, JSON.stringify(JSON.parse(message.text), null, 2) + '\n');
          panel.webview.postMessage({ type: 'fragmenterSaved', path: target.fsPath });
        }
      } else if (message.type === 'saveConfig') {
        const target = await vscode.window.showSaveDialog({ filters: { 'CLEFTS run configuration': ['pft.json', 'json'] }, defaultUri: vscode.Uri.file('fragment-tree.pft.json') });
        if (target) await fs.promises.writeFile(target.fsPath, JSON.stringify(normalizeConfig(message.config), null, 2) + '\n');
      } else if (message.type === 'loadConfig') {
        const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS run configuration': ['pft.json', 'json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const config = JSON.parse(await fs.promises.readFile(picked[0].fsPath, 'utf8'));
          panel.webview.postMessage({ type: 'config', config: { ...defaultConfig(context), ...config } });
        }
      } else if (message.type === 'openResult') {
        await openResultPicker();
      } else if (message.type === 'copyCommand') {
        const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
        const command = shellDisplay(python, buildArgs(normalizeConfig(message.config)));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'status', status: 'idle', text: 'Command copied.', command });
      } else if (message.type === 'run') {
        if (!message.config.params) throw new Error('Fragmenter parameters JSON path is required.');
        JSON.parse(message.fragmenterText);
        await fs.promises.mkdir(path.dirname(message.config.params), { recursive: true });
        await fs.promises.writeFile(message.config.params, JSON.stringify(JSON.parse(message.fragmenterText), null, 2) + '\n');
        await runFragmentTree(context, output, normalizeConfig(message.config), panel);
      } else if (message.type === 'stop') {
        if (runningProcess) runningProcess.kill('SIGTERM');
      } else if (message.type === 'loadCleavagePatternSet') {
        const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS Cleavage Pattern Set': ['clevageset.json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const value = cleavagePatternSetEditor.normalizeDocument(JSON.parse(await fs.promises.readFile(picked[0].fsPath, 'utf8')));
          panel.webview.postMessage({ type: 'cleavagePatternSet', value, path: picked[0].fsPath });
        }
      } else if (message.type === 'saveCleavagePatternSet') {
        const value = cleavagePatternSetEditor.normalizeDocument(message.value);
        const defaultUri = message.path ? vscode.Uri.file(message.path) : vscode.Uri.file('patterns.clevageset.json');
        const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern Set': ['clevageset.json'] }, defaultUri });
        if (selected) {
          const target = selected.fsPath.endsWith(cleavagePatternSetEditor.FILE_SUFFIX) ? selected : vscode.Uri.file(`${selected.fsPath}${cleavagePatternSetEditor.FILE_SUFFIX}`);
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(value, null, 2)}\n`);
          panel.webview.postMessage({ type: 'cleavagePatternSetSaved', path: target.fsPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'loadCleavagePattern') {
        const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS Cleavage Pattern': ['clevage.json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const pattern = normalizePattern(JSON.parse(await fs.promises.readFile(picked[0].fsPath, 'utf8')));
          panel.webview.postMessage({ type: 'cleavagePatternLoaded', pattern, index: message.index });
        }
      } else if (message.type === 'saveCleavagePattern') {
        const pattern = normalizePattern(message.pattern);
        const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern': ['clevage.json'] }, defaultUri: vscode.Uri.file(`${safeFileStem(pattern.name || 'pattern')}.clevage.json`) });
        if (selected) {
          const target = selected.fsPath.endsWith('.clevage.json') ? selected : vscode.Uri.file(`${selected.fsPath}.clevage.json`);
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(pattern, null, 2)}\n`);
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'selectElements') {
        const elements = await showElementPicker(message.elements);
        if (elements) panel.webview.postMessage({ type: 'elementSelectionResult', requestId: message.requestId, elements });
      } else if (message.type === 'chemistry') {
        const result = await runChemistryBackend(context, message.command, message.payload);
        panel.webview.postMessage({ type: 'chemistryResult', requestId: message.requestId, result });
      }
    } catch (error) {
      if (message.type === 'chemistry') panel.webview.postMessage({ type: 'chemistryError', requestId: message.requestId, message: String(error.message || error) });
      panel.webview.postMessage({ type: 'status', status: 'error', text: String(error.message || error) });
      vscode.window.showErrorMessage(`CLEFTS: ${error.message || error}`);
    }
  });
}

function showElementPicker(initialElements) {
  return new Promise(resolve => {
    const panel = vscode.window.createWebviewPanel('clefts.periodicTable', 'Select Elements', vscode.ViewColumn.Active, { enableScripts: true, retainContextWhenHidden: false });
    const selected = new Set(Array.isArray(initialElements) ? initialElements : []);
    const rows = [
      ['H','','','','','','','','','','','','','','','','','He'],
      ['Li','Be','','','','','','','','','','','','B','C','N','O','F','Ne'],
      ['Na','Mg','','','','','','','','','','','','Al','Si','P','S','Cl','Ar'],
      ['K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr'],
      ['Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe'],
      ['Cs','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn'],
      ['Fr','Ra','Ac','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og'],
      ['','','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',''],
      ['','','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','']
    ];
    panel.webview.html = `<!doctype html><html><head><meta charset="UTF-8"><style>
      :root{color-scheme:light dark}*{box-sizing:border-box}body{margin:0;padding:24px;color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);font-family:var(--vscode-font-family)}h1{font-size:22px;margin:0 0 6px}.muted{opacity:.65;margin:0 0 20px}.table{display:grid;grid-template-columns:repeat(18,minmax(36px,1fr));gap:4px;min-width:760px}.element,.gap{aspect-ratio:1}.element{padding:0;border:1px solid var(--vscode-input-border,#777);border-radius:5px;color:inherit;background:var(--vscode-button-secondaryBackground);font-weight:600;cursor:pointer}.element:hover{border-color:var(--vscode-focusBorder)}.element.selected{color:var(--vscode-button-foreground);background:var(--vscode-button-background);outline:2px solid var(--vscode-focusBorder);outline-offset:1px}.series-gap{margin-top:14px}.actions{position:sticky;bottom:0;display:flex;justify-content:flex-end;gap:8px;padding-top:20px;background:var(--vscode-editor-background)}.actions button{padding:8px 16px;border:1px solid var(--vscode-input-border,#777);border-radius:5px;color:inherit;background:var(--vscode-button-secondaryBackground)}.actions .primary{color:var(--vscode-button-foreground);background:var(--vscode-button-background)}</style></head><body><h1>Periodic Table</h1><p class="muted">Choose one or more elements. The current query is preselected.</p><div class="table">${rows.flatMap((row,ri)=>row.map(symbol=>symbol?`<button class="element${selected.has(symbol)?' selected':''}${ri===7?' series-gap':''}" data-element="${symbol}">${symbol}</button>`:`<span class="gap${ri===7?' series-gap':''}"></span>`)).join('')}</div><div class="actions"><button id="cancel">Cancel</button><button id="apply" class="primary">Apply selection</button></div><script>const vscode=acquireVsCodeApi(),chosen=new Set(${JSON.stringify([...selected])});document.querySelector('.table').onclick=e=>{const b=e.target.closest('[data-element]');if(!b)return;chosen.has(b.dataset.element)?chosen.delete(b.dataset.element):chosen.add(b.dataset.element);b.classList.toggle('selected',chosen.has(b.dataset.element))};document.getElementById('apply').onclick=()=>vscode.postMessage({type:'apply',elements:[...chosen]});document.getElementById('cancel').onclick=()=>vscode.postMessage({type:'cancel'});</script></body></html>`;
    let completed = false;
    panel.webview.onDidReceiveMessage(message => {
      if (message.type === 'apply') { completed = true; resolve(message.elements); panel.dispose(); }
      if (message.type === 'cancel') panel.dispose();
    });
    panel.onDidDispose(() => { if (!completed) resolve(null); });
  });
}

function normalizePattern(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('The cleavage pattern must be a JSON object.');
  return {
    name: String(value.name || ''), reactant_smarts: String(value.reactant_smarts || ''),
    products: Array.isArray(value.products) ? value.products.map(product => ({ name: String(product && product.name || ''), smarts: String(product && product.smarts || '') })) : []
  };
}

function safeFileStem(value) { return String(value).trim().replace(/[^A-Za-z0-9_.-]+/g, '_') || 'pattern'; }

function runChemistryBackend(context, command, payload) {
  return new Promise((resolve, reject) => {
    const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'cleavage-pattern-set', 'backend.py');
    let root;
    try { root = projectRoot(context); }
    catch (error) { reject(error); return; }
    if (!isCleftsRoot(root)) { reject(new Error('The detected working directory is not a CLEFTS application.')); return; }
    const pythonPath = ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const child = spawn(python, [script], { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath } });
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('error', reject);
    child.on('close', () => {
      try { const response = JSON.parse(stdout); response.ok ? resolve(response.result) : reject(new Error(response.error)); }
      catch (_) { reject(new Error(stderr || stdout || 'The RDKit backend did not return a response.')); }
    });
    child.stdin.end(JSON.stringify({ command, payload }));
  });
}

function normalizeConfig(config) {
  return { ...config, symbols: Array.isArray(config.symbols) ? config.symbols : String(config.symbols || '').split(/[ ,]+/).filter(Boolean) };
}

async function runFragmentTree(context, output, config, panel) {
  if (runningProcess) throw new Error('Another CLEFTS process is already running.');
  for (const key of ['trainInput', 'outputDir', 'params']) if (!config[key]) throw new Error(`${key} is required.`);
  if (!config.symbols.length) throw new Error('At least one symbol is required.');
  const root = projectRoot(context);
  const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
  await fs.promises.mkdir(config.outputDir, { recursive: true });
  const configPath = path.join(config.outputDir, 'fragment-tree.pft.json');
  const resultPath = path.join(config.outputDir, RESULT_NAME);
  await fs.promises.writeFile(configPath, JSON.stringify(config, null, 2) + '\n');
  const args = buildArgs(config);
  await writeResult(resultPath, { schemaVersion: 1, application: config.application, status: 'running', outputDirectory: config.outputDir, configFile: configPath, command: [python, ...args], startedAt: new Date().toISOString() });
  output.clear(); output.show(true); output.appendLine(`$ ${shellDisplay(python, args)}`);
  panel.webview.postMessage({ type: 'status', status: 'running', text: 'Running CLI…', command: shellDisplay(python, args) });
  runningProcess = spawn(python, args, { cwd: root, env: process.env });
  runningProcess.stdout.on('data', chunk => output.append(chunk.toString()));
  runningProcess.stderr.on('data', chunk => output.append(chunk.toString()));
  runningProcess.on('error', async error => {
    runningProcess = undefined;
    await writeResult(resultPath, { schemaVersion: 1, application: config.application, status: 'failed', outputDirectory: config.outputDir, configFile: configPath, error: error.message, finishedAt: new Date().toISOString() });
    panel.webview.postMessage({ type: 'status', status: 'error', text: error.message });
  });
  runningProcess.on('close', async code => {
    runningProcess = undefined;
    const status = code === 0 ? 'completed' : 'failed';
    await writeResult(resultPath, { schemaVersion: 1, application: config.application, status, exitCode: code, outputDirectory: config.outputDir, configFile: configPath, command: [python, ...args], finishedAt: new Date().toISOString() });
    panel.webview.postMessage({ type: 'status', status, text: code === 0 ? 'Completed.' : `Failed with exit code ${code}.`, resultPath });
    if (code === 0) vscode.window.showInformationMessage('CLEFTS fragment tree data preparation completed.', 'Open Result').then(choice => { if (choice) openResult(vscode.Uri.file(resultPath)); });
  });
}

function buildArgs(c) {
  const a = ['-m', 'clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data', '--train-input', c.trainInput, '--output-dir', c.outputDir, '--params', c.params, '--symbols', ...c.symbols];
  const values = {
    validationInput: '--validation-input', validationStructuresInputDir: '--validation-structures-input-dir',
    validationSmilesRatio: '--validation-smiles-ratio', tanimotoNumBins: '--tanimoto-num-bins', tanimotoRadius: '--tanimoto-radius', tanimotoNBits: '--tanimoto-n-bits', validationSamplingSeed: '--validation-sampling-seed',
    smilesColumn: '--smiles-column', precursorMzColumn: '--precursor-mz-column', adductTypeColumn: '--adduct-type-column', collisionEnergyColumn: '--collision-energy-column', instrumentColumn: '--instrument-column',
    maxNode: '--max-node', maxEdge: '--max-edge', numWorkers: '--num-workers', chunkSize: '--chunk-size', structureRebuildPolicy: '--structure-rebuild-policy'
  };
  for (const [key, flag] of Object.entries(values)) if (c[key] !== '' && c[key] !== null && c[key] !== undefined) a.push(flag, String(c[key]));
  const flags = { overwrite: '--overwrite', overwritePreprocessingConfig: '--overwrite-preprocessing-config', saveTrainValidRecords: '--save-train-valid-records', keepParallelTemp: '--keep-parallel-temp' };
  for (const [key, flag] of Object.entries(flags)) if (c[key]) a.push(flag);
  if (c.saveValidationValidRecords === false) a.push('--no-save-validation-valid-records');
  return a;
}

function shellDisplay(program, args) { return [program, ...args].map(v => /^[A-Za-z0-9_./:=,-]+$/.test(v) ? v : `'${v.replace(/'/g, "'\\''")}'`).join(' '); }
async function writeResult(file, patch) {
  let old = {}; try { old = JSON.parse(await fs.promises.readFile(file, 'utf8')); } catch (_) {}
  await fs.promises.writeFile(file, JSON.stringify({ ...old, ...patch }, null, 2) + '\n');
}

async function openResultPicker() {
  const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS result': ['pft', 'clefts-result'] }, canSelectMany: false });
  if (picked && picked[0]) openResult(picked[0]);
}
function openResult(uri) { return vscode.commands.executeCommand('vscode.openWith', uri, 'clefts.resultViewer'); }

class ResultEditorProvider {
  constructor(context) { this.context = context; }
  async openCustomDocument(uri) { return { uri, dispose() {} }; }
  async resolveCustomEditor(document, panel) {
    panel.webview.options = { enableScripts: true };
    const refresh = async () => {
      try { panel.webview.html = await resultHtml(document.uri.fsPath); }
      catch (e) { panel.webview.html = errorHtml(e.message); }
    };
    panel.webview.onDidReceiveMessage(async m => {
      if (m.type === 'refresh') refresh();
      if (m.type === 'openFile') {
        const base = path.dirname(document.uri.fsPath), target = path.resolve(base, m.path);
        if (!(target === base || target.startsWith(base + path.sep))) return;
        if (target.endsWith('.preft.pt') || target.endsWith('.preft') || target.endsWith('.pt')) {
          panel.webview.postMessage({ type: 'structureLoading', path: m.path });
          try { panel.webview.postMessage({ type: 'structureDetail', result: await inspectFragmentTreeResult(this.context, target) }); }
          catch (error) { panel.webview.postMessage({ type: 'structureError', message: String(error.message || error) }); }
        } else vscode.commands.executeCommand('vscode.open', vscode.Uri.file(target));
      } else if (m.type === 'inspectMolecule') {
        try { panel.webview.postMessage({ type: 'moleculeDetail', smiles: m.smiles, node: m.node, result: await runChemistryBackend(this.context, 'depict', { smiles: m.smiles }) }); }
        catch (error) { panel.webview.postMessage({ type: 'structureError', message: String(error.message || error) }); }
      }
    });
    await refresh();
  }
}

async function resultHtml(manifestPath) {
  const manifest = JSON.parse(await fs.promises.readFile(manifestPath, 'utf8'));
  const root = path.dirname(manifestPath);
  const files = await scan(root, root, 3, 500);
  const manifests = await readStructureManifests(root);
  const summary = summarize(files);
  summary.preft = manifests.reduce((total, item) => total + item.rows.filter(row => row.exists).length, 0);
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}.tree-scroll{overflow:auto;border:1px solid var(--border);border-radius:8px;background:var(--vscode-editor-background)}.fragment-tree{display:block;width:100%;min-width:700px}.tree-edge{stroke:color-mix(in srgb,var(--vscode-editor-foreground) 35%,transparent);stroke-width:1.5}.tree-edge.target-path{stroke:var(--accent);stroke-width:3}.tree-node circle{fill:var(--vscode-editor-background);stroke:var(--vscode-editor-foreground);stroke-width:1.5}.tree-node text{fill:var(--vscode-editor-foreground);font-size:9px;pointer-events:none}.tree-node{cursor:pointer}.molecule-preview{width:100%;max-height:360px;overflow:hidden;background:#fff;border-radius:6px}.molecule-preview svg{display:block;width:100%;height:auto}.missing-file{display:block;color:var(--vscode-errorForeground)}.tree-controls{display:flex;justify-content:flex-end;margin:8px 0}.tree-controls select{width:auto}.sortable{cursor:pointer;user-select:none}.sortable:hover{color:var(--accent)}.detail-layout{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px;align-items:start}.molecule-side{position:sticky;top:12px;border:1px solid var(--border);border-radius:8px;padding:14px}.molecule-side[hidden]{display:block;visibility:hidden}.molecule-side code{display:block;overflow-wrap:anywhere;margin:8px 0}@media(max-width:900px){.detail-layout{grid-template-columns:1fr}.molecule-side{position:static}.molecule-side[hidden]{display:none}}</style></head><body><main>
    <header><div><span class="eyebrow">CLEFTS RESULT</span><h1>Fragment Tree Dataset</h1><p class="muted">${escapeHtml(root)}</p></div><button onclick="vscode.postMessage({type:'refresh'})">Refresh</button></header>
    <section class="cards"><article><b>${escapeHtml(manifest.status || 'unknown')}</b><span>status</span></article><article><b>${summary.preft}</b><span>structures (.preft.pt)</span></article><article><b>${summary.tsv}</b><span>tables (.tsv)</span></article><article><b>${formatBytes(summary.bytes)}</b><span>listed size</span></article></section>
    <section><h2>Run Information</h2><dl><dt>Application</dt><dd>${escapeHtml(manifest.application || '')}</dd><dt>Finished</dt><dd>${escapeHtml(manifest.finishedAt || '—')}</dd><dt>Exit code</dt><dd>${escapeHtml(String(manifest.exitCode ?? '—'))}</dd><dt>Command</dt><dd><code>${escapeHtml((manifest.command || []).join(' '))}</code></dd></dl></section>
    <section><h2>Structure Manifests</h2><p class="muted">Only manifest rows are shown; structure data directories are not enumerated.</p>${manifests.map(manifestTableHtml).join('') || '<p>No structure manifest was found.</p>'}</section>
    <section><h2>Supporting Files <small>${files.length} files</small></h2><div class="files">${files.map(f => `<button class="file" data-open-file="${encodeURIComponent(f.relative)}"><span>${escapeHtml(f.relative)}</span><em>${formatBytes(f.size)}</em></button>`).join('')}</div></section>
    <section id="detail" hidden></section>
    <script>const vscode=acquireVsCodeApi(),detail=document.getElementById('detail'),sortState=new Map();const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function graphSvg(g,sample){if(!g||!g.nodes.length)return '';const targetEdges=new Set(sample.targetPathEdges),precursors=new Set(sample.precursorNodes),depth=new Map(g.nodes.map(n=>[n.id,0]));for(let pass=0;pass<g.nodes.length;pass++){let changed=false;for(const e of g.edges){const value=(depth.get(e.source)||0)+1;if(value>(depth.get(e.target)||0)){depth.set(e.target,value);changed=true}}if(!changed)break}const levels={};for(const n of g.nodes)(levels[depth.get(n.id)]||(levels[depth.get(n.id)]=[])).push(n);const pos=new Map(),width=1000,maxRows=Math.max(...Object.values(levels).map(x=>x.length)),height=Math.max(240,maxRows*52+40),maxDepth=Math.max(...depth.values(),1);for(const [d,nodes] of Object.entries(levels))nodes.forEach((n,i)=>pos.set(n.id,{x:35+Number(d)/maxDepth*930,y:30+(i+1)*height/(nodes.length+1)}));const lines=g.edges.map(e=>{const a=pos.get(e.source),b=pos.get(e.target);return '<line class="tree-edge '+(targetEdges.has(e.id)?'target-path':'')+'" x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y+'"><title>edge '+e.id+'</title></line>'}).join('');const nodes=g.nodes.map(n=>{const p=pos.get(n.id),label=(precursors.has(n.id)?'p':'')+n.id;return '<g class="tree-node" data-node="'+n.id+'" data-smiles="'+esc(n.smiles)+'" transform="translate('+p.x+' '+p.y+')"><circle r="13"><title>'+esc(n.smiles)+'</title></circle><text text-anchor="middle" dominant-baseline="central">'+label+'</text></g>'}).join('');return '<div class="tree-scroll"><svg class="fragment-tree" viewBox="0 0 '+width+' '+height+'">'+lines+nodes+'</svg></div>'+(g.truncated?'<p class="warning">Tree preview is limited to the first 500 nodes.</p>':'')}
    function render(r){const s=r.summary;detail.hidden=false;detail.innerHTML='<h2>Structure Detail</h2><p class="muted">'+esc(r.file)+'</p><div class="cards detail-cards"><article><b>'+s.samples+'</b><span>samples</span></article><article><b>'+s.nodes+'</b><span>all calculated nodes</span></article><article><b>'+s.edges+'</b><span>all edges</span></article><article><b>'+s.assignedPeaks+' / '+s.peaks+'</b><span>assigned peaks</span></article></div><p><b>Depths:</b> '+esc(Object.entries(s.depthCounts).map(x=>(x[0]==='0'?'0':'depth '+x[0])+': '+x[1]).join(', ')||'none')+'</p><div class="detail-layout"><div class="detail-main">'+(r.truncated?'<p class="warning">Sample preview truncated for responsiveness.</p>':'')+r.samples.map(sample=>'<details><summary>sample '+sample.id+' · CE '+esc(sample.collisionEnergyRaw)+' ('+(Number.isFinite(sample.collisionEnergy)?sample.collisionEnergy:'?')+') · adduct '+esc(sample.adduct)+' ('+sample.adductIndex+') · '+sample.peakCount+' peaks · assignment score '+Math.max(0,Math.min(1,sample.assignmentScore)).toFixed(4)+'</summary><p class="muted">Green edges occur in this sample\\'s target paths. Click a node to inspect its structure.</p>'+'<div class="tree-controls"><label>Tree nodes <select data-tree-mode><option value="active">Active target paths only</option><option value="all">All calculated nodes (preview)</option></select></label></div>'+graphSvg(sample.activeGraph,sample)+'<table><thead><tr><th>#</th><th class="sortable" data-sort="mz">m/z ↕</th><th class="sortable" data-sort="intensity">intensity ↕</th><th>depth</th><th>target formulas and nodes</th></tr></thead><tbody data-sample="'+sample.id+'">'+peakRows(sample.peaks)+'</tbody></table></details>').join('')+'</div><aside id="moleculeDetail" class="molecule-side" hidden><p class="muted">Click a tree node to display its structure.</p></aside></div>';detail.dataset.result=JSON.stringify(r);detail.scrollIntoView({behavior:'smooth'});}
    function peakRows(peaks){return peaks.map(p=>'<tr><td>'+p.index+'</td><td>'+p.mz.toFixed(6)+'</td><td>'+p.intensity.toPrecision(5)+'</td><td>'+(p.depth<0?'empty':p.depth)+'</td><td>'+(p.formulas.map(f=>'<div class="formula"><b>'+esc(f.formula)+'</b><small>nodes '+f.assignments.map(a=>a.terminalNode).sort((a,b)=>a-b).join(', ')+'</small></div>').join('')||'<span class="muted">No assignment</span>')+'</td></tr>').join('')}
    document.addEventListener('click',e=>{const file=e.target.closest('[data-open-file]');if(file)vscode.postMessage({type:'openFile',path:decodeURIComponent(file.dataset.openFile)})});
    detail.onclick=e=>{const node=e.target.closest('[data-node]');if(node){vscode.postMessage({type:'inspectMolecule',node:Number(node.dataset.node),smiles:node.dataset.smiles});return}const header=e.target.closest('[data-sort]');if(header){const result=JSON.parse(detail.dataset.result),body=header.closest('details').querySelector('tbody'),sample=result.samples.find(x=>x.id===Number(body.dataset.sample)),key=sample.id+':'+header.dataset.sort,previous=sortState.get(key)||0,direction=previous===0?(header.dataset.sort==='mz'?1:-1):-previous;sortState.set(key,direction);header.closest('tr').querySelectorAll('[data-sort]').forEach(item=>item.textContent=item.dataset.sort==='mz'?'m/z ↕':'intensity ↕');header.textContent=(header.dataset.sort==='mz'?'m/z ':'intensity ')+(direction>0?'↑':'↓');const peaks=[...sample.peaks].sort((a,b)=>direction*(header.dataset.sort==='mz'?a.mz-b.mz:a.intensity-b.intensity));body.innerHTML=peakRows(peaks)}if(e.target.id==='copySmiles')navigator.clipboard.writeText(e.target.dataset.smiles)};
    detail.onchange=e=>{const select=e.target.closest('[data-tree-mode]');if(!select)return;const result=JSON.parse(detail.dataset.result),body=select.closest('details').querySelector('tbody'),sample=result.samples.find(x=>x.id===Number(body.dataset.sample)),current=select.closest('details').querySelector('.tree-scroll'),wrapper=document.createElement('div');wrapper.innerHTML=graphSvg(select.value==='active'?sample.activeGraph:result.graph,sample);if(current)current.replaceWith(wrapper.querySelector('.tree-scroll'))};
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='structureLoading'){detail.hidden=false;detail.innerHTML='<h2>Loading structure…</h2><p class="muted">'+esc(m.path)+'</p>'}if(m.type==='structureDetail')render(m.result);if(m.type==='moleculeDetail'){const box=document.getElementById('moleculeDetail');box.hidden=false;box.innerHTML='<h3>Node '+m.node+'</h3><div class="molecule-preview">'+m.result.svg+'</div><code>'+esc(m.smiles)+'</code> <button id="copySmiles" data-smiles="'+esc(m.smiles)+'">Copy SMILES</button>'}if(m.type==='structureError')detail.innerHTML='<h2>Unable to inspect structure</h2><pre>'+esc(m.message)+'</pre>'});</script></main></body></html>`;
}

async function readStructureManifests(root) {
  const results = [];
  for (const directory of ['train_structures', 'validation_structures']) {
    const file = path.join(root, directory, 'manifest.tsv');
    if (!fs.existsSync(file)) continue;
    const lines = (await fs.promises.readFile(file, 'utf8')).trim().split(/\r?\n/);
    if (!lines.length || !lines[0]) continue;
    const columns = lines[0].split('\t');
    const rows = lines.slice(1).map(line => {
      const values = line.split('\t');
      const row = Object.fromEntries(columns.map((column, index) => [column, values[index] || '']));
      row.relative = row.file ? path.join(directory, 'data', row.file) : '';
      row.exists = Boolean(row.relative && fs.existsSync(path.join(root, row.relative)));
      return row;
    });
    results.push({ directory, rows });
  }
  return results;
}

function manifestTableHtml(manifest) {
  const columns = ['file', 'status', 'smiles', 'num_input_records', 'num_valid_samples', 'num_nodes', 'num_edges'];
  return `<h3>${escapeHtml(manifest.directory)}</h3><div class="table-scroll"><table><thead><tr>${columns.map(column => `<th>${escapeHtml(column)}</th>`).join('')}</tr></thead><tbody>${manifest.rows.map(row => `<tr>${columns.map(column => { const value = row[column] || ''; if (column === 'file' && value && row.exists) return `<td><button class="manifest-file" data-open-file="${encodeURIComponent(row.relative)}">${escapeHtml(value)}</button></td>`; if (column === 'file' && value) return `<td><span>${escapeHtml(value)}</span><small class="missing-file">not generated (${escapeHtml(row.status || 'missing')})</small></td>`; return `<td>${escapeHtml(value)}</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function inspectFragmentTreeResult(context, target) {
  return new Promise((resolve, reject) => {
    const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'fragment-tree-result', 'backend.py');
    let root; try { root = projectRoot(context); } catch (error) { reject(error); return; }
    const pythonPath = ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const child = spawn(python, [script], { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath } });
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('error', reject);
    child.on('close', () => { try { const response = JSON.parse(stdout); response.ok ? resolve(response.result) : reject(new Error(response.error)); } catch (_) { reject(new Error(stderr || stdout || 'Structure inspector returned no JSON response.')); } });
    child.stdin.end(JSON.stringify({ path: target }));
  });
}

async function scan(root, dir, depth, limit, out = []) {
  if (depth < 0 || out.length >= limit) return out;
  for (const entry of await fs.promises.readdir(dir, { withFileTypes: true })) {
    if (entry.name === RESULT_NAME || entry.name === 'fragment-tree.clefts-result' || entry.name.startsWith('.')) continue;
    if (entry.isDirectory() && entry.name === 'data' && /(?:train|validation)_structures$/.test(path.basename(dir))) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await scan(root, full, depth - 1, limit, out);
    else { const stat = await fs.promises.stat(full); out.push({ relative: path.relative(root, full), size: stat.size }); }
    if (out.length >= limit) break;
  }
  return out.sort((a,b) => a.relative.localeCompare(b.relative));
}
function summarize(files) { return files.reduce((s,f) => { s.bytes += f.size; if (f.relative.endsWith('.preft.pt') || f.relative.endsWith('.preft') || f.relative.endsWith('.pt')) s.preft++; if (f.relative.endsWith('.tsv')) s.tsv++; return s; }, {preft:0,tsv:0,bytes:0}); }
function formatBytes(n) { if (n < 1024) return `${n} B`; if (n < 1048576) return `${(n/1024).toFixed(1)} KB`; return `${(n/1048576).toFixed(1)} MB`; }
function escapeHtml(v) { return String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function errorHtml(message) { return `<!doctype html><style>${commonCss()}</style><main><h1>Unable to display result</h1><pre>${escapeHtml(message)}</pre></main>`; }

const HELP = {
  trainInput: 'Training input MSDataset path. Required.', validationInput: 'Optional validation input MSDataset path.',
  params: 'Fragmenter parameter JSON used for preprocessing. Required.', outputDir: 'Output root directory. Required.',
  symbols: 'Element symbols that define atom-feature columns. Required and immutable for generated data.',
  maxNode: 'Maximum fragment-tree nodes. Use -1 for no limit.', maxEdge: 'Maximum fragment-tree edges. Use -1 for no limit.',
  smilesColumn: 'Metadata column containing SMILES strings. Default: SMILES.', precursorMzColumn: 'Metadata column containing precursor m/z values. Default: PrecursorMZ.',
  adductTypeColumn: 'Metadata column containing adduct types. Default: AdductType.', collisionEnergyColumn: 'Metadata column containing collision energies. Default: CollisionEnergy.',
  instrumentColumn: 'Metadata column containing instrument names. Default: Instrument.', validationSmilesRatio: 'Target validation SMILES count relative to unique training SMILES. Default: 0.1.',
  tanimotoNumBins: 'Number of similarity intervals used for balanced validation sampling. Default: 10.', tanimotoRadius: 'Morgan fingerprint radius used for validation similarity. Default: 2.',
  tanimotoNBits: 'Morgan fingerprint bit count. Default: 2048.', validationSamplingSeed: 'Random seed for deterministic validation sampling. Default: 0.',
  validationStructuresInputDir: 'Optional existing validation structure directory; mutually exclusive with validation input.',
  numWorkers: 'Number of parallel subprocess workers. Use 1 to disable parallel processing.', chunkSize: 'Number of SMILES groups assigned to each parallel chunk.',
  structureRebuildPolicy: 'Rebuild saved structures always, on root matches, or on matches in any saved fragment.',
  overwrite: 'Overwrite existing .preft.pt structure files.', overwritePreprocessingConfig: 'Overwrite an existing preprocessing configuration without prompting.',
  saveTrainValidRecords: 'Save successfully processed training records as an MSDataset.', saveValidationValidRecords: 'Save successfully processed validation records as an MSDataset.',
  keepParallelTemp: 'Keep temporary parallel-processing files after merging.'
};

function workbenchHtml(config, fragmenterText) {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}${formCss()}</style></head><body><main>
  <header><div><span class="eyebrow">CLEFTS PLATFORM</span><h1>Workbench</h1><p id="appSubtitle" class="muted">Fragment Tree Data Preparation</p></div><div id="dataActions" class="actions"><button id="openResult">Open Result</button><button id="load">Load Configuration</button><button id="save">Save Configuration</button></div></header>
  <nav><button class="tab" data-app="cleavage">Cleavage Pattern Set</button><button class="tab active" data-app="data">Data Preparation</button><button class="tab" disabled>Training <small>coming soon</small></button></nav>
  <div id="cleavageApp" hidden><section><div class="section-title"><div><h2>Cleavage Pattern Set Configuration</h2><p id="cleavagePath" class="muted">Not saved</p></div><div class="actions"><button type="button" id="loadCleavage">Load Configuration</button><button type="button" id="saveCleavage">Save Configuration</button></div></div><label>Pattern set name<input id="cleavageSetName" placeholder="single_bond_cleavage_pattern_set"></label></section><section id="visualBuilder" hidden><div class="section-title"><div><h2>Visual Pattern Builder</h2><p class="muted">Click or right-click items individually, or hold either mouse button and draw a loop around atoms and bonds.</p></div><div class="actions"><button type="button" id="clearSelection">Clear all</button><button type="button" id="closeBuilder">Close</button></div></div><div class="row"><label class="grow">Source SMILES<input id="builderSmiles" placeholder="O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"></label><button type="button" id="drawMolecule" class="primary">Draw Structure</button></div><div id="moleculeCanvas" class="molecule-canvas"></div><div class="builder-columns"><div><h3>Atom queries</h3><div id="atomConstraints" class="query-list"></div></div><div><h3>Bond queries</h3><div id="bondConstraints" class="query-list"></div></div></div><label>Pattern name<input id="builderPatternName" placeholder="flavonoid_substructure"></label><div class="actions"><button type="button" id="applyReactant" class="primary">Apply Reactant</button></div><div id="productBuilder" hidden><h3>Product Transformation</h3><p class="muted">Edit the reactant graph directly: click or right-click an atom to retain/delete it; click or right-click a bond to select it, then choose its product bond type.</p><div id="productCanvas" class="molecule-canvas product-canvas"></div><div id="productBondTools" class="bond-tools"><span>No bond selected</span></div><label>Product name<input id="builderProductName" placeholder="fragment_product"></label><button type="button" id="applyProduct" class="primary">Add Product</button></div></section><div id="cleavagePatterns"></div><div class="actions"><button type="button" id="addCleavagePattern" class="primary">Add Pattern</button><button type="button" id="addVisualPattern">Add Pattern Visually</button><button type="button" id="loadSinglePattern">Load Pattern</button></div></div>
  <form id="form" data-app-panel="data">
    <section><h2>Input and Output</h2>${pathField('trainInput','Training MSDataset *','file')}${pathField('validationInput','Validation MSDataset','file')}${pathField('outputDir','Output directory *','folder')}</section>
    <section><div class="section-title"><div><h2>Fragmenter Parameters</h2><p class="muted">Edit the complete Fragmenter JSON used by the CLI.</p></div><div class="actions"><button type="button" id="loadFragmenter">Load Fragmenter</button><button type="button" id="saveFragmenter">Save Fragmenter</button></div></div>${pathField('params','Fragmenter parameters JSON *','file')}<textarea id="fragmenterEditor" spellcheck="false" title="Fragmenter configuration JSON passed through the CLI --params option."></textarea></section>
    <section><h2>Fragment tree</h2><div class="grid">${field('symbols','Symbols (space separated)','text')}${field('maxNode','Max nodes','number')}${field('maxEdge','Max edges','number')}${field('smilesColumn','SMILES column','text')}${field('precursorMzColumn','Precursor m/z column','text')}${field('adductTypeColumn','Adduct column','text')}${field('collisionEnergyColumn','Collision energy column','text')}${field('instrumentColumn','Instrument column','text')}</div></section>
    <section><h2>Validation sampling</h2><div class="grid">${field('validationSmilesRatio','SMILES ratio','number','any')}${field('tanimotoNumBins','Tanimoto bins','number')}${field('tanimotoRadius','Morgan radius','number')}${field('tanimotoNBits','Morgan bits','number')}${field('validationSamplingSeed','Random seed','number')}</div>${pathField('validationStructuresInputDir','Existing validation structures','folder')}</section>
    <section><h2>Performance & output</h2><div class="grid">${field('numWorkers','Workers','number')}${field('chunkSize','Chunk size','number')}<label data-help="${HELP.structureRebuildPolicy}">Rebuild policy<select name="structureRebuildPolicy"><option>all-fragments</option><option>root</option><option>always</option></select></label></div><div class="checks">${check('overwrite','Overwrite structures')}${check('overwritePreprocessingConfig','Overwrite preprocessing config')}${check('saveTrainValidRecords','Save valid training records')}${check('saveValidationValidRecords','Save valid validation records')}${check('keepParallelTemp','Keep parallel temp')}</div></section>
    <footer><div><div id="status" class="status idle">Ready</div><code id="command"></code></div><div class="actions"><button type="button" id="copyCommand">Copy Command</button><button type="button" id="stop" disabled>Stop</button><button type="submit" class="primary">Run</button></div></footer>
  </form><div id="helpTooltip" role="tooltip"></div><script>const vscode=acquireVsCodeApi(); const initial=${safeJson(config)}; const initialFragmenter=${safeJson(fragmenterText)}; ${webviewScript()}</script></main></body></html>`;
}
function pathField(name,label,kind) { return `<label data-help="${HELP[name] || ''}">${label}<div class="path"><input name="${name}"><button type="button" data-pick="${name}" data-kind="${kind}">Browse</button></div></label>`; }
function field(name,label,type,step='1') { return `<label data-help="${HELP[name] || ''}">${label}<input name="${name}" type="${type}"${type==='number'?` step="${step}"`:''}></label>`; }
function check(name,label) { return `<label class="check" data-help="${HELP[name] || ''}"><input name="${name}" type="checkbox">${label}</label>`; }
function safeJson(value) { return JSON.stringify(value).replace(/</g, '\\u003c'); }
function webviewScript() { return `
    const form=document.getElementById('form'), statusEl=document.getElementById('status'), stop=document.getElementById('stop');
    let cleavagePath='',cleavageModel={cleavage_pattern_set:{name:'',patterns:[]}};
    const htmlEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function setConfig(c){ for(const [k,v] of Object.entries(c)){const el=form.elements[k];if(!el)continue;if(el.type==='checkbox')el.checked=!!v;else el.value=Array.isArray(v)?v.join(' '):v??'';} }
    function getConfig(){const c={application:'fragment-tree-data-preparation'};for(const el of form.elements){if(!el.name)continue;if(el.type==='checkbox')c[el.name]=el.checked;else if(el.type==='number')c[el.name]=el.value===''?'':Number(el.value);else c[el.name]=el.value;}c.symbols=String(c.symbols).split(/[ ,]+/).filter(Boolean);return c;}
    const fragmenterEditor=document.getElementById('fragmenterEditor'); fragmenterEditor.value=initialFragmenter;
    const tooltip=document.getElementById('helpTooltip'); let tooltipTimer;
    document.querySelectorAll('[data-help]').forEach(el=>{el.addEventListener('mouseenter',()=>{tooltipTimer=setTimeout(()=>{const r=el.getBoundingClientRect();tooltip.textContent=el.dataset.help;tooltip.style.left=Math.min(r.left,window.innerWidth-390)+'px';tooltip.style.top=(r.bottom+7)+'px';tooltip.classList.add('visible');},500);});el.addEventListener('mouseleave',()=>{clearTimeout(tooltipTimer);tooltip.classList.remove('visible');});});
    setConfig(initial); document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>vscode.postMessage({type:'pick',field:b.dataset.pick,kind:b.dataset.kind}));
    document.querySelectorAll('[data-app]').forEach(button=>button.onclick=()=>{document.querySelectorAll('[data-app]').forEach(x=>x.classList.toggle('active',x===button));const cleavage=button.dataset.app==='cleavage';document.getElementById('cleavageApp').hidden=!cleavage;form.hidden=cleavage;document.getElementById('dataActions').hidden=cleavage;document.getElementById('appSubtitle').textContent=cleavage?'Cleavage Pattern Set Editor':'Fragment Tree Data Preparation';});
    document.getElementById('save').onclick=()=>vscode.postMessage({type:'saveConfig',config:getConfig()}); document.getElementById('load').onclick=()=>vscode.postMessage({type:'loadConfig'}); document.getElementById('openResult').onclick=()=>vscode.postMessage({type:'openResult'});
    function renderCleavage(){document.getElementById('cleavageSetName').value=cleavageModel.cleavage_pattern_set.name;document.getElementById('cleavagePatterns').innerHTML=cleavageModel.cleavage_pattern_set.patterns.map((p,pi)=>\`<section class="pattern-card"><div class="section-title"><h2>Pattern \${pi+1}</h2><div class="actions"><button type="button" data-edit-visual="\${pi}">Edit Visually</button><button type="button" data-load-pattern="\${pi}">Load</button><button type="button" data-save-pattern="\${pi}">Save</button><button type="button" class="danger" data-remove-pattern="\${pi}">Remove Pattern</button></div></div><div class="grid"><label>Pattern name<input data-pattern="\${pi}" data-key="name" value="\${htmlEscape(p.name)}" placeholder="single_bond_cleavage"></label><label>Reactant SMARTS<input data-pattern="\${pi}" data-key="reactant_smarts" value="\${htmlEscape(p.reactant_smarts)}" placeholder="[!#1:1]-[!#1:2]"></label></div><div class="section-title product-title"><h3>Products</h3><button type="button" data-add-product="\${pi}">Add Product</button></div><div class="product-list">\${p.products.map((product,xi)=>\`<div class="product-row"><label>Product name<input data-pattern="\${pi}" data-product="\${xi}" data-key="name" value="\${htmlEscape(product.name)}"></label><label>Product SMARTS<input data-pattern="\${pi}" data-product="\${xi}" data-key="smarts" value="\${htmlEscape(product.smarts)}" placeholder="[!#1:1]"></label><button type="button" class="danger" data-remove-product="\${pi}:\${xi}">Remove</button></div>\`).join('')}</div></section>\`).join('');}
    document.getElementById('cleavageSetName').oninput=e=>cleavageModel.cleavage_pattern_set.name=e.target.value;document.getElementById('addCleavagePattern').onclick=()=>{cleavageModel.cleavage_pattern_set.patterns.push({name:'',reactant_smarts:'',products:[]});renderCleavage()};
    document.getElementById('cleavagePatterns').addEventListener('input',e=>{const pi=Number(e.target.dataset.pattern);if(!Number.isInteger(pi))return;const xi=e.target.dataset.product;if(xi===undefined)cleavageModel.cleavage_pattern_set.patterns[pi][e.target.dataset.key]=e.target.value;else cleavageModel.cleavage_pattern_set.patterns[pi].products[Number(xi)][e.target.dataset.key]=e.target.value;});
    document.getElementById('cleavagePatterns').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.editVisual!==undefined){editVisualPattern(Number(b.dataset.editVisual));return}else if(b.dataset.addProduct!==undefined)cleavageModel.cleavage_pattern_set.patterns[Number(b.dataset.addProduct)].products.push({name:'',smarts:''});else if(b.dataset.removePattern!==undefined)cleavageModel.cleavage_pattern_set.patterns.splice(Number(b.dataset.removePattern),1);else if(b.dataset.removeProduct){const [pi,xi]=b.dataset.removeProduct.split(':').map(Number);cleavageModel.cleavage_pattern_set.patterns[pi].products.splice(xi,1)}else if(b.dataset.savePattern!==undefined){vscode.postMessage({type:'saveCleavagePattern',pattern:cleavageModel.cleavage_pattern_set.patterns[Number(b.dataset.savePattern)]});return}else if(b.dataset.loadPattern!==undefined){vscode.postMessage({type:'loadCleavagePattern',index:Number(b.dataset.loadPattern)});return}else return;renderCleavage()});
    document.getElementById('loadCleavage').onclick=()=>vscode.postMessage({type:'loadCleavagePatternSet'});document.getElementById('saveCleavage').onclick=()=>vscode.postMessage({type:'saveCleavagePatternSet',value:cleavageModel,path:cleavagePath});renderCleavage();
    document.getElementById('loadSinglePattern').onclick=()=>vscode.postMessage({type:'loadCleavagePattern',index:-1});
    let chemistrySequence=0,chemistryWaiters=new Map(),molGraph=null,selectedAtoms=new Set(),selectedBonds=new Set(),atomConstraintState={},bondConstraintState={},atomMapBySource={},visualPatternIndex=-1,visualSourceType='smiles',productKeptAtoms=new Set(),productBondOverrides={},productSelectedBond=null;
    function chemistry(command,payload){return new Promise((resolve,reject)=>{const requestId=++chemistrySequence;chemistryWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'chemistry',requestId,command,payload})})}
    const atomColor=s=>({N:'#2478c5',O:'#e53935',S:'#d6a800',P:'#e67e22',F:'#28a745',Cl:'#28a745',Br:'#9b3b22',I:'#7b4ab5',H:'#57898d'}[s]||'currentColor');
    function layout(){const xs=molGraph.atoms.map(a=>a.x),ys=molGraph.atoms.map(a=>a.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);return {x:a=>55+(a.x-minX)/(maxX-minX||1)*590,y:a=>45+(a.y-minY)/(maxY-minY||1)*350}}
    function bondMarkup(b,l,selected,deleted=false,override='preserve'){const a=molGraph.atoms[b.begin],z=molGraph.atoms[b.end],x1=l.x(a),y1=l.y(a),x2=l.x(z),y2=l.y(z),dx=x2-x1,dy=y2-y1,n=Math.hypot(dx,dy)||1,ox=-dy/n*6,oy=dx/n*6,order=override==='preserve'?b.order:Number(override),kind=order===1.5?'aromatic':order===3?'triple':order===2?'double':'single',cls='bond-group bond-'+kind+' '+(selected?'selected ':'')+(deleted?'deleted':'');let lines='';if(order===2)lines=\`<line x1="\${x1+ox}" y1="\${y1+oy}" x2="\${x2+ox}" y2="\${y2+oy}"/><line x1="\${x1-ox}" y1="\${y1-oy}" x2="\${x2-ox}" y2="\${y2-oy}"/>\`;else if(order===1.5)lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/><line class="aromatic-mark" x1="\${x1+ox}" y1="\${y1+oy}" x2="\${x2+ox}" y2="\${y2+oy}"/>\`;else if(order===3)lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/><line x1="\${x1+ox*1.7}" y1="\${y1+oy*1.7}" x2="\${x2+ox*1.7}" y2="\${y2+oy*1.7}"/><line x1="\${x1-ox*1.7}" y1="\${y1-oy*1.7}" x2="\${x2-ox*1.7}" y2="\${y2-oy*1.7}"/>\`;else lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/>\`;return \`<g data-mol-bond="\${b.index}" class="\${cls}">\${lines}<line class="bond-hit" x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/></g>\`}
    function atomMarkup(a,l,selected,deleted=false){const neighbors=molGraph.bonds.filter(b=>b.begin===a.index||b.end===a.index).map(b=>molGraph.atoms[b.begin===a.index?b.end:b.begin]),vx=neighbors.reduce((sum,n)=>sum+l.x(n)-l.x(a),0),vy=neighbors.reduce((sum,n)=>sum+l.y(n)-l.y(a),0),length=Math.hypot(vx,vy),ix=length>1?-vx/length*25:20,iy=length>1?-vy/length*25:-20;return \`<g data-mol-atom="\${a.index}" class="mol-atom \${selected?'selected ':''}\${deleted?'deleted':''}" transform="translate(\${l.x(a)} \${l.y(a)})"><circle class="atom-hit" r="25"/><circle class="atom-label-bg" r="16"/><text class="atom-symbol" style="fill:\${atomColor(a.symbol)}" text-anchor="middle" dominant-baseline="central">\${a.symbol}</text><circle class="atom-index-bg" cx="\${ix}" cy="\${iy}" r="9" style="fill:var(--vscode-editor-background);stroke:var(--vscode-focusBorder);stroke-width:1"/><text class="atom-index" x="\${ix}" y="\${iy}" text-anchor="middle" dominant-baseline="central">\${a.index}</text></g>\`}
    const sourceBondType=b=>b.order===1.5?'aromatic':b.order===3?'triple':b.order===2?'double':'single';
    function queryPanels(){document.getElementById('atomConstraints').innerHTML=[...selectedAtoms].sort((a,b)=>a-b).map(i=>{const state=atomConstraintState[i]||(atomConstraintState[i]={mode:'elements',elements:[molGraph.atoms[i].symbol]});return \`<div class="query-card"><div class="query-head"><b>Atom \${i}</b><span>\${molGraph.atoms[i].symbol}</span></div><div class="query-modes"><button data-atom-mode="\${i}:elements" class="\${state.mode==='elements'?'active':''}">Elements</button><button data-atom-mode="\${i}:any-heavy" class="\${state.mode==='any-heavy'?'active':''}">Any non-H</button></div><button class="element-picker" data-open-elements="\${i}">Open periodic table</button><div class="element-summary">\${state.mode==='elements'?state.elements.join(', '):'Any non-hydrogen atom'}</div></div>\`}).join('')||'<p class="muted">Select atoms in the structure.</p>';document.getElementById('bondConstraints').innerHTML=[...selectedBonds].sort((a,b)=>a-b).map(i=>{const b=molGraph.bonds[i],state=bondConstraintState[i]||(bondConstraintState[i]={types:[sourceBondType(b)],ring:false});return \`<div class="query-card"><div class="query-head"><b>Bond \${i}</b><span>\${b.begin}–\${b.end}</span></div><div class="bond-type-choices">\${['any','single','double','triple','aromatic'].map(type=>\`<button data-query-bond-type="\${i}:\${type}" class="\${state.types.includes(type)?'active':''}">\${type==='any'?'Any bond':type[0].toUpperCase()+type.slice(1)}</button>\`).join('')}</div><label class="check"><input type="checkbox" data-ring-bond="\${i}" \${state.ring?'checked':''}>Must be in a ring</label></div>\`}).join('')||'<p class="muted">Select bonds in the structure.</p>'}
    function renderMolecule(){if(!molGraph)return;const l=layout();document.getElementById('moleculeCanvas').innerHTML=\`<svg viewBox="0 0 700 440">\${molGraph.bonds.map(b=>bondMarkup(b,l,selectedBonds.has(b.index))).join('')}\${molGraph.atoms.map(a=>atomMarkup(a,l,selectedAtoms.has(a.index))).join('')}<polyline class="selection-lasso" hidden/></svg>\`;queryPanels()}
    function renderProduct(){if(!molGraph)return;const l=layout();document.getElementById('productCanvas').innerHTML=\`<svg viewBox="0 0 700 440">\${molGraph.bonds.filter(b=>selectedBonds.has(b.index)).map(b=>bondMarkup(b,l,productSelectedBond===b.index,!productKeptAtoms.has(b.begin)||!productKeptAtoms.has(b.end)||productBondOverrides[b.index]==='remove',productBondOverrides[b.index]||'preserve')).join('')}\${molGraph.atoms.filter(a=>selectedAtoms.has(a.index)).map(a=>atomMarkup(a,l,productKeptAtoms.has(a.index),!productKeptAtoms.has(a.index))).join('')}</svg>\`;const tools=document.getElementById('productBondTools');tools.innerHTML=productSelectedBond===null?'<span>Click a bond to edit it</span>':\`<b>Bond \${productSelectedBond}</b>\${[['preserve','Keep'],['1','Single'],['2','Double'],['3','Triple'],['1.5','Aromatic'],['remove','Delete']].map(x=>\`<button data-product-type="\${x[0]}" class="\${(productBondOverrides[productSelectedBond]||'preserve')===x[0]?'active':''}">\${x[1]}</button>\`).join('')}\`}
    function sourcePayload(){const value=document.getElementById('builderSmiles').value;return visualSourceType==='smarts'?{sourceType:'smarts',smarts:value}:{sourceType:'smiles',smiles:value}}
    function resetVisualBuilder(){molGraph=null;selectedAtoms=new Set();selectedBonds=new Set();atomConstraintState={};bondConstraintState={};atomMapBySource={};visualPatternIndex=-1;visualSourceType='smiles';productKeptAtoms=new Set();productBondOverrides={};productSelectedBond=null;document.getElementById('builderSmiles').value='';document.getElementById('builderPatternName').value='';document.getElementById('builderProductName').value='';document.getElementById('moleculeCanvas').innerHTML='';document.getElementById('atomConstraints').innerHTML='';document.getElementById('bondConstraints').innerHTML='';document.getElementById('productCanvas').innerHTML='';document.getElementById('productBuilder').hidden=true}
    async function editVisualPattern(index){const pattern=cleavageModel.cleavage_pattern_set.patterns[index];resetVisualBuilder();visualPatternIndex=index;visualSourceType='smarts';document.getElementById('visualBuilder').hidden=false;document.getElementById('builderSmiles').value=pattern.reactant_smarts;document.getElementById('builderPatternName').value=pattern.name;try{molGraph=await chemistry('molecule',{smarts:pattern.reactant_smarts});selectedAtoms=new Set(molGraph.atoms.map(a=>a.index));selectedBonds=new Set(molGraph.bonds.map(b=>b.index));molGraph.atoms.forEach(a=>{atomConstraintState[a.index]=a.symbol==='*'?{mode:'any-heavy',elements:[]}:{mode:'elements',elements:[a.symbol]};atomMapBySource[a.index]=a.mapNumber||a.index+1});renderMolecule();productKeptAtoms=new Set(selectedAtoms);document.getElementById('productBuilder').hidden=false;renderProduct()}catch(e){}}
    document.getElementById('addVisualPattern').onclick=()=>{resetVisualBuilder();document.getElementById('visualBuilder').hidden=false;document.getElementById('builderSmiles').focus()};document.getElementById('closeBuilder').onclick=()=>{resetVisualBuilder();document.getElementById('visualBuilder').hidden=true};
    document.getElementById('drawMolecule').onclick=async()=>{try{molGraph=await chemistry('molecule',sourcePayload());selectedAtoms=new Set();selectedBonds=new Set();atomConstraintState={};bondConstraintState={};renderMolecule()}catch(e){}};
    let elementRequestSequence=0;const elementRequests=new Map();
    document.getElementById('atomConstraints').onclick=e=>{const mode=e.target.dataset.atomMode,open=e.target.dataset.openElements;if(mode){const [i,value]=mode.split(':');atomConstraintState[i].mode=value;queryPanels()}if(open!==undefined){const requestId=++elementRequestSequence;elementRequests.set(requestId,Number(open));vscode.postMessage({type:'selectElements',requestId,elements:atomConstraintState[open].elements})}};
    document.getElementById('bondConstraints').onclick=e=>{const value=e.target.dataset.queryBondType;if(value){const [i,type]=value.split(':'),state=bondConstraintState[i];if(type==='any')state.types=['any'];else{if(state.types.includes('any'))state.types=[];const pos=state.types.indexOf(type);if(pos<0)state.types.push(type);else if(state.types.length>1)state.types.splice(pos,1)}queryPanels()}};
    document.getElementById('bondConstraints').onchange=e=>{if(e.target.dataset.ringBond!==undefined)bondConstraintState[e.target.dataset.ringBond].ring=e.target.checked};
    const canvas=document.getElementById('moleculeCanvas');let drag=null;
    function svgPoint(svg,e){const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse())}
    function pointInPolygon(x,y,points){let inside=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if(((a.y>y)!==(b.y>y))&&(x<(b.x-a.x)*(y-a.y)/(b.y-a.y)+a.x))inside=!inside}return inside}
    function orientation(a,b,c){return Math.sign((b.y-a.y)*(c.x-b.x)-(b.x-a.x)*(c.y-b.y))}
    function segmentsIntersect(a,b,c,d){return orientation(a,b,c)!==orientation(a,b,d)&&orientation(c,d,a)!==orientation(c,d,b)}
    function bondInsideLasso(a,b,points){for(const t of [.15,.3,.5,.7,.85])if(pointInPolygon(a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,points))return true;for(let i=0;i<points.length;i++)if(segmentsIntersect(a,b,points[i],points[(i+1)%points.length]))return true;return false}
    canvas.oncontextmenu=e=>e.preventDefault();
    canvas.onpointerdown=e=>{if(!molGraph||(e.button!==0&&e.button!==2))return;e.preventDefault();const svg=canvas.querySelector('svg'),q=svgPoint(svg,e),atom=e.target.closest('[data-mol-atom]'),bond=e.target.closest('[data-mol-bond]');drag={points:[{x:q.x,y:q.y}],moved:false,atom:atom?Number(atom.dataset.molAtom):null,bond:bond?Number(bond.dataset.molBond):null};svg.setPointerCapture(e.pointerId)};
    canvas.onpointermove=e=>{if(!drag)return;const svg=canvas.querySelector('svg'),q=svgPoint(svg,e),last=drag.points[drag.points.length-1],lasso=svg.querySelector('.selection-lasso');if(Math.hypot(q.x-last.x,q.y-last.y)<2)return;drag.points.push({x:q.x,y:q.y});drag.moved=drag.moved||drag.points.length>2;lasso.hidden=!drag.moved;lasso.setAttribute('points',drag.points.map(p=>p.x+','+p.y).join(' '));lasso.style.display=drag.moved?'block':'none'};
    canvas.onpointerup=e=>{if(!drag)return;const finished=drag;if(!finished.moved){if(finished.atom!==null){const i=finished.atom;selectedAtoms.has(i)?selectedAtoms.delete(i):selectedAtoms.add(i)}else if(finished.bond!==null){const i=finished.bond;selectedBonds.has(i)?selectedBonds.delete(i):selectedBonds.add(i)}}else if(finished.points.length>2){const points=finished.points,l=layout();molGraph.atoms.forEach(a=>{if(pointInPolygon(l.x(a),l.y(a),points))selectedAtoms.add(a.index)});molGraph.bonds.forEach(b=>{const a=molGraph.atoms[b.begin],z=molGraph.atoms[b.end],start={x:l.x(a),y:l.y(a)},end={x:l.x(z),y:l.y(z)};if(bondInsideLasso(start,end,points))selectedBonds.add(b.index)})}drag=null;requestAnimationFrame(renderMolecule)};
    canvas.onpointercancel=()=>{drag=null;renderMolecule()};
    document.getElementById('clearSelection').onclick=()=>{selectedAtoms.clear();selectedBonds.clear();atomConstraintState={};bondConstraintState={};renderMolecule()};
    document.getElementById('applyReactant').onclick=async()=>{try{const name=document.getElementById('builderPatternName').value,result=await chemistry('reactant',{...sourcePayload(),name,atoms:[...selectedAtoms],bonds:[...selectedBonds],constraints:atomConstraintState,bondConstraints:bondConstraintState});atomMapBySource=result.atomMapBySource;if(visualPatternIndex<0){cleavageModel.cleavage_pattern_set.patterns.push({name,reactant_smarts:result.smarts,products:[]});visualPatternIndex=cleavageModel.cleavage_pattern_set.patterns.length-1}else{const pattern=cleavageModel.cleavage_pattern_set.patterns[visualPatternIndex];pattern.name=name;pattern.reactant_smarts=result.smarts}visualSourceType='smarts';document.getElementById('builderSmiles').value=result.smarts;renderCleavage();productKeptAtoms=new Set(selectedAtoms);productBondOverrides={};productSelectedBond=null;document.getElementById('productBuilder').hidden=false;renderProduct()}catch(e){}};
    const productCanvas=document.getElementById('productCanvas');
    function editProductTarget(e){const atom=e.target.closest('[data-mol-atom]'),bond=e.target.closest('[data-mol-bond]');if(atom){const i=Number(atom.dataset.molAtom);productKeptAtoms.has(i)?productKeptAtoms.delete(i):productKeptAtoms.add(i)}else if(bond)productSelectedBond=Number(bond.dataset.molBond);else return;renderProduct()}
    productCanvas.onclick=editProductTarget;
    productCanvas.oncontextmenu=e=>{e.preventDefault();editProductTarget(e)};
    document.getElementById('productBondTools').onclick=e=>{if(e.target.dataset.productType===undefined||productSelectedBond===null)return;const value=e.target.dataset.productType;if(value==='preserve')delete productBondOverrides[productSelectedBond];else productBondOverrides[productSelectedBond]=value;renderProduct()};
    document.getElementById('applyProduct').onclick=async()=>{let pattern,added=false;try{const result=await chemistry('product',{...sourcePayload(),atoms:[...selectedAtoms],keptAtoms:[...productKeptAtoms],atomMapBySource,bondOverrides:productBondOverrides});pattern=cleavageModel.cleavage_pattern_set.patterns[visualPatternIndex];pattern.products.push({name:document.getElementById('builderProductName').value,smarts:result.smarts});added=true;await chemistry('validate',pattern);renderCleavage()}catch(e){if(added)pattern.products.pop()}};
    document.getElementById('loadFragmenter').onclick=()=>vscode.postMessage({type:'loadFragmenter'}); document.getElementById('saveFragmenter').onclick=()=>vscode.postMessage({type:'saveFragmenter',path:form.elements.params.value,text:fragmenterEditor.value});
    form.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'run',config:getConfig(),fragmenterText:fragmenterEditor.value});}; document.getElementById('copyCommand').onclick=()=>vscode.postMessage({type:'copyCommand',config:getConfig()}); stop.onclick=()=>vscode.postMessage({type:'stop'});
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='picked')form.elements[m.field].value=m.value;if(m.type==='config')setConfig(m.config);if(m.type==='fragmenter'){form.elements.params.value=m.path;fragmenterEditor.value=m.text;}if(m.type==='fragmenterSaved')form.elements.params.value=m.path;if(m.type==='cleavagePatternSet'){cleavageModel=m.value;cleavagePath=m.path;document.getElementById('cleavagePath').textContent=m.path;renderCleavage()}if(m.type==='cleavagePatternSetSaved'){cleavagePath=m.path;document.getElementById('cleavagePath').textContent=m.path}if(m.type==='cleavagePatternLoaded'){if(m.index>=0)cleavageModel.cleavage_pattern_set.patterns[m.index]=m.pattern;else cleavageModel.cleavage_pattern_set.patterns.push(m.pattern);renderCleavage()}if(m.type==='elementSelectionResult'){const atom=elementRequests.get(m.requestId);if(atom!==undefined&&m.elements.length){atomConstraintState[atom].elements=m.elements;atomConstraintState[atom].mode='elements';elementRequests.delete(m.requestId);queryPanels()}}if(m.type==='chemistryResult'){const waiter=chemistryWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);chemistryWaiters.delete(m.requestId)}}if(m.type==='chemistryError'){const waiter=chemistryWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));chemistryWaiters.delete(m.requestId)}}if(m.type==='status'){statusEl.textContent=m.text;statusEl.className='status '+m.status;stop.disabled=m.status!=='running';if(m.command)document.getElementById('command').textContent=m.command;}});`;
}
function commonCss() { return `:root{color-scheme:light dark;--accent:#36c5a2;--panel:color-mix(in srgb,var(--vscode-editor-background) 88%,var(--vscode-editor-foreground));--border:color-mix(in srgb,var(--vscode-editor-foreground) 18%,transparent)}*{box-sizing:border-box}body{font-family:var(--vscode-font-family);color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);margin:0}main{max-width:1100px;margin:auto;padding:32px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:28px}h1{font-size:32px;margin:4px 0}h2{font-size:17px;margin:0 0 18px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.14em;font-size:11px}.muted,small{opacity:.65}button{font:inherit;color:inherit;background:var(--vscode-button-secondaryBackground);border:1px solid var(--border);border-radius:6px;padding:8px 13px;cursor:pointer}button:hover{background:var(--vscode-button-secondaryHoverBackground)}section{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:22px;margin:14px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;background:none;border:0;padding:0}.cards article{background:var(--panel);border:1px solid var(--border);padding:18px;border-radius:9px}.cards b{display:block;font-size:23px;color:var(--accent)}.cards span{opacity:.65}.files{display:grid;gap:4px}.file{display:flex;justify-content:space-between;text-align:left;background:transparent;border:0;border-bottom:1px solid var(--border);border-radius:0}.file em{opacity:.55;font-style:normal}dl{display:grid;grid-template-columns:110px 1fr;gap:10px}dt{opacity:.6}dd{margin:0;overflow-wrap:anywhere}code{font-family:var(--vscode-editor-font-family);font-size:12px}details{border-top:1px solid var(--border);padding:10px 0}summary{cursor:pointer;font-weight:600}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--border);padding:7px}.formula{margin-bottom:7px}.formula small{display:block;margin-top:3px;font-family:var(--vscode-editor-font-family)}.warning{color:var(--vscode-editorWarning-foreground)}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}main{padding:18px}table{display:block;overflow:auto}}`; }
function formCss() { return `nav{display:flex;gap:8px;margin-bottom:18px}.tab{border-radius:20px}.tab.active{border-color:var(--accent)}[hidden]{display:none!important}label{display:block;font-size:12px;opacity:.8;margin:12px 0}input,select,textarea{width:100%;display:block;margin-top:6px;padding:9px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,var(--border));border-radius:5px;font:inherit}.path{display:flex;gap:7px}.path input{flex:1}.row{display:flex;align-items:end;gap:12px}.grow{flex:1}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0 16px}.checks{display:flex;flex-wrap:wrap;gap:8px 22px}.check{display:flex;align-items:center;gap:7px}.check input{width:auto;margin:0}.section-title{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.section-title h2{margin-bottom:4px}.section-title p{margin:0}.actions{display:flex;gap:8px}textarea{min-height:280px;resize:vertical;font-family:var(--vscode-editor-font-family);font-size:12px;line-height:1.5}.pattern-card{border-left:3px solid var(--accent)}.product-title{align-items:center;margin-top:18px}.product-title h3{margin:0}.product-list{margin-left:18px}.product-row{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:end;border-top:1px solid var(--border);padding:6px 0}.danger{color:var(--vscode-errorForeground)}.molecule-canvas{min-height:260px;margin:16px 0;border:1px solid var(--border);border-radius:8px;background:var(--vscode-editor-background);color:var(--vscode-editor-foreground);touch-action:none;user-select:none}.molecule-canvas svg{display:block;width:100%;height:440px}.bond-group{stroke:currentColor;stroke-width:3;stroke-linecap:round;cursor:pointer}.bond-aromatic .aromatic-mark{stroke-dasharray:8 6}.bond-group.selected{stroke:var(--accent);stroke-width:5}.bond-group.deleted{opacity:.2;stroke-dasharray:5 4}.bond-hit{stroke:transparent!important;stroke-width:24}.mol-atom{cursor:pointer}.atom-hit{fill:transparent;stroke:transparent}.atom-label-bg{fill:var(--vscode-editor-background);stroke:color-mix(in srgb,var(--vscode-editor-foreground) 45%,transparent);stroke-width:1.5}.mol-atom text{fill:var(--vscode-editor-foreground);font-size:16px;font-family:var(--vscode-font-family);font-weight:700;pointer-events:none}.mol-atom .atom-index{font-size:11px;font-weight:600;fill:var(--vscode-descriptionForeground)}.mol-atom.selected .atom-label-bg{stroke:var(--accent);stroke-width:3;fill:color-mix(in srgb,var(--accent) 20%,var(--vscode-editor-background))}.mol-atom.selected .atom-hit{fill:#36c5a214}.mol-atom.deleted{opacity:.25}.selection-lasso{fill:none;stroke:var(--vscode-editor-foreground);stroke-width:2.5;stroke-dasharray:7 5;stroke-linecap:round;stroke-linejoin:round;pointer-events:none}.builder-columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.query-list{display:grid;gap:8px}.query-card{border:1px solid var(--border);border-radius:7px;padding:10px}.query-head{display:flex;justify-content:space-between}.query-modes,.bond-type-choices,.bond-tools{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.query-card button,.bond-tools button{padding:5px 8px}.query-card button.active,.bond-tools button.active{background:var(--accent);color:#10251f;border-color:var(--accent)}.element-picker{width:100%;margin-top:8px}.element-summary{margin-top:7px;padding:6px 8px;border-radius:4px;background:var(--vscode-textBlockQuote-background);font-size:12px}.product-canvas{max-height:350px}.product-canvas svg{height:350px}.bond-tools{align-items:center;margin-bottom:12px}#productBuilder{margin-top:22px;padding-top:16px;border-top:1px solid var(--border)}#helpTooltip{position:fixed;z-index:50;display:none;max-width:380px;padding:9px 11px;border:1px solid var(--vscode-editorHoverWidget-border,var(--border));border-radius:5px;background:var(--vscode-editorHoverWidget-background);color:var(--vscode-editorHoverWidget-foreground);box-shadow:0 4px 14px #0005;font-size:12px;line-height:1.4}#helpTooltip.visible{display:block}.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground);font-weight:600}.primary:hover{background:var(--vscode-button-hoverBackground)}footer{position:sticky;bottom:0;background:var(--vscode-editor-background);border-top:1px solid var(--border);padding:17px 0;display:flex;justify-content:space-between;align-items:center;gap:20px}.status{font-weight:600}.status.running{color:#e9b949}.status.completed{color:var(--accent)}.status.error,.status.failed{color:#ef6b73}#command{display:block;opacity:.65;max-width:700px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:5px}@media(max-width:700px){.grid,.product-row,.builder-columns{grid-template-columns:1fr}.row{display:block}footer{position:static}}`; }

function deactivate() { if (runningProcess) runningProcess.kill('SIGTERM'); }
module.exports = { activate, deactivate, buildArgs, normalizeConfig, resultHtml };
