const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const RESULT_NAME = 'fragment-tree.clefts-result';
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
  return configured ? path.resolve(configured) : path.resolve(context.extensionPath, '..');
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
    collisionEnergyColumn: 'CollisionEnergy', instrumentColumn: 'Instrument', maxNode: -1, maxEdge: -1,
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
        const target = await vscode.window.showSaveDialog({ filters: { 'CLEFTS configuration': ['json'] }, defaultUri: vscode.Uri.file('fragment-tree-config.json') });
        if (target) await fs.promises.writeFile(target.fsPath, JSON.stringify(normalizeConfig(message.config), null, 2) + '\n');
      } else if (message.type === 'loadConfig') {
        const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS configuration': ['json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const config = JSON.parse(await fs.promises.readFile(picked[0].fsPath, 'utf8'));
          panel.webview.postMessage({ type: 'config', config: { ...defaultConfig(context), ...config } });
        }
      } else if (message.type === 'run') {
        if (!message.config.params) throw new Error('Fragmenter parameters JSON path is required.');
        JSON.parse(message.fragmenterText);
        await fs.promises.mkdir(path.dirname(message.config.params), { recursive: true });
        await fs.promises.writeFile(message.config.params, JSON.stringify(JSON.parse(message.fragmenterText), null, 2) + '\n');
        await runFragmentTree(context, output, normalizeConfig(message.config), panel);
      } else if (message.type === 'stop') {
        if (runningProcess) runningProcess.kill('SIGTERM');
      }
    } catch (error) {
      panel.webview.postMessage({ type: 'status', status: 'error', text: String(error.message || error) });
      vscode.window.showErrorMessage(`CLEFTS: ${error.message || error}`);
    }
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
  const configPath = path.join(config.outputDir, 'clefts-run-config.json');
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
  const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS result': ['clefts-result'] }, canSelectMany: false });
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
        if (target === base || target.startsWith(base + path.sep)) vscode.commands.executeCommand('vscode.open', vscode.Uri.file(target));
      }
    });
    await refresh();
  }
}

async function resultHtml(manifestPath) {
  const manifest = JSON.parse(await fs.promises.readFile(manifestPath, 'utf8'));
  const root = path.dirname(manifestPath);
  const files = await scan(root, root, 3, 500);
  const summary = summarize(files);
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}</style></head><body><main>
    <header><div><span class="eyebrow">CLEFTS RESULT</span><h1>Fragment Tree Dataset</h1><p class="muted">${escapeHtml(root)}</p></div><button onclick="vscode.postMessage({type:'refresh'})">Refresh</button></header>
    <section class="cards"><article><b>${escapeHtml(manifest.status || 'unknown')}</b><span>status</span></article><article><b>${summary.pt}</b><span>structures (.pt)</span></article><article><b>${summary.tsv}</b><span>tables (.tsv)</span></article><article><b>${formatBytes(summary.bytes)}</b><span>listed size</span></article></section>
    <section><h2>Run Information</h2><dl><dt>Application</dt><dd>${escapeHtml(manifest.application || '')}</dd><dt>Finished</dt><dd>${escapeHtml(manifest.finishedAt || '—')}</dd><dt>Exit code</dt><dd>${escapeHtml(String(manifest.exitCode ?? '—'))}</dd><dt>Command</dt><dd><code>${escapeHtml((manifest.command || []).join(' '))}</code></dd></dl></section>
    <section><h2>Output Files <small>${files.length} files</small></h2><div class="files">${files.map(f => `<button class="file" onclick='vscode.postMessage({type:"openFile",path:${JSON.stringify(f.relative)}})'><span>${escapeHtml(f.relative)}</span><em>${formatBytes(f.size)}</em></button>`).join('')}</div></section>
    <script>const vscode=acquireVsCodeApi();</script></main></body></html>`;
}

async function scan(root, dir, depth, limit, out = []) {
  if (depth < 0 || out.length >= limit) return out;
  for (const entry of await fs.promises.readdir(dir, { withFileTypes: true })) {
    if (entry.name === RESULT_NAME || entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await scan(root, full, depth - 1, limit, out);
    else { const stat = await fs.promises.stat(full); out.push({ relative: path.relative(root, full), size: stat.size }); }
    if (out.length >= limit) break;
  }
  return out.sort((a,b) => a.relative.localeCompare(b.relative));
}
function summarize(files) { return files.reduce((s,f) => { s.bytes += f.size; if (f.relative.endsWith('.pt')) s.pt++; if (f.relative.endsWith('.tsv')) s.tsv++; return s; }, {pt:0,tsv:0,bytes:0}); }
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
  overwrite: 'Overwrite existing .pt structure files.', overwritePreprocessingConfig: 'Overwrite an existing preprocessing configuration without prompting.',
  saveTrainValidRecords: 'Save successfully processed training records as an MSDataset.', saveValidationValidRecords: 'Save successfully processed validation records as an MSDataset.',
  keepParallelTemp: 'Keep temporary parallel-processing files after merging.'
};

function workbenchHtml(config, fragmenterText) {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}${formCss()}</style></head><body><main>
  <header><div><span class="eyebrow">CLEFTS PLATFORM</span><h1>Workbench</h1><p class="muted">Fragment Tree Data Preparation</p></div><div class="actions"><button id="load">Load Configuration</button><button id="save">Save Configuration</button></div></header>
  <nav><button class="tab active">Data preparation</button><button class="tab" disabled>Training <small>coming soon</small></button></nav>
  <form id="form">
    <section><h2>Input and Output</h2>${pathField('trainInput','Training MSDataset *','file')}${pathField('validationInput','Validation MSDataset','file')}${pathField('outputDir','Output directory *','folder')}</section>
    <section><div class="section-title"><div><h2>Fragmenter Parameters</h2><p class="muted">Edit the complete Fragmenter JSON used by the CLI.</p></div><div class="actions"><button type="button" id="loadFragmenter">Load Fragmenter</button><button type="button" id="saveFragmenter">Save Fragmenter</button></div></div>${pathField('params','Fragmenter parameters JSON *','file')}<textarea id="fragmenterEditor" spellcheck="false" title="Fragmenter configuration JSON passed through the CLI --params option."></textarea></section>
    <section><h2>Fragment tree</h2><div class="grid">${field('symbols','Symbols (space separated)','text')}${field('maxNode','Max nodes','number')}${field('maxEdge','Max edges','number')}${field('smilesColumn','SMILES column','text')}${field('precursorMzColumn','Precursor m/z column','text')}${field('adductTypeColumn','Adduct column','text')}${field('collisionEnergyColumn','Collision energy column','text')}${field('instrumentColumn','Instrument column','text')}</div></section>
    <section><h2>Validation sampling</h2><div class="grid">${field('validationSmilesRatio','SMILES ratio','number','any')}${field('tanimotoNumBins','Tanimoto bins','number')}${field('tanimotoRadius','Morgan radius','number')}${field('tanimotoNBits','Morgan bits','number')}${field('validationSamplingSeed','Random seed','number')}</div>${pathField('validationStructuresInputDir','Existing validation structures','folder')}</section>
    <section><h2>Performance & output</h2><div class="grid">${field('numWorkers','Workers','number')}${field('chunkSize','Chunk size','number')}<label data-help="${HELP.structureRebuildPolicy}">Rebuild policy<select name="structureRebuildPolicy"><option>all-fragments</option><option>root</option><option>always</option></select></label></div><div class="checks">${check('overwrite','Overwrite structures')}${check('overwritePreprocessingConfig','Overwrite preprocessing config')}${check('saveTrainValidRecords','Save valid training records')}${check('saveValidationValidRecords','Save valid validation records')}${check('keepParallelTemp','Keep parallel temp')}</div></section>
    <footer><div><div id="status" class="status idle">Ready</div><code id="command"></code></div><div class="actions"><button type="button" id="stop" disabled>Stop</button><button type="submit" class="primary">Run CLI</button></div></footer>
  </form><div id="helpTooltip" role="tooltip"></div><script>const vscode=acquireVsCodeApi(); const initial=${safeJson(config)}; const initialFragmenter=${safeJson(fragmenterText)}; ${webviewScript()}</script></main></body></html>`;
}
function pathField(name,label,kind) { return `<label data-help="${HELP[name] || ''}">${label}<div class="path"><input name="${name}"><button type="button" data-pick="${name}" data-kind="${kind}">Browse</button></div></label>`; }
function field(name,label,type,step='1') { return `<label data-help="${HELP[name] || ''}">${label}<input name="${name}" type="${type}"${type==='number'?` step="${step}"`:''}></label>`; }
function check(name,label) { return `<label class="check" data-help="${HELP[name] || ''}"><input name="${name}" type="checkbox">${label}</label>`; }
function safeJson(value) { return JSON.stringify(value).replace(/</g, '\\u003c'); }
function webviewScript() { return `
    const form=document.getElementById('form'), statusEl=document.getElementById('status'), stop=document.getElementById('stop');
    function setConfig(c){ for(const [k,v] of Object.entries(c)){const el=form.elements[k];if(!el)continue;if(el.type==='checkbox')el.checked=!!v;else el.value=Array.isArray(v)?v.join(' '):v??'';} }
    function getConfig(){const c={application:'fragment-tree-data-preparation'};for(const el of form.elements){if(!el.name)continue;if(el.type==='checkbox')c[el.name]=el.checked;else if(el.type==='number')c[el.name]=el.value===''?'':Number(el.value);else c[el.name]=el.value;}c.symbols=String(c.symbols).split(/[ ,]+/).filter(Boolean);return c;}
    const fragmenterEditor=document.getElementById('fragmenterEditor'); fragmenterEditor.value=initialFragmenter;
    const tooltip=document.getElementById('helpTooltip'); let tooltipTimer;
    document.querySelectorAll('[data-help]').forEach(el=>{el.addEventListener('mouseenter',()=>{tooltipTimer=setTimeout(()=>{const r=el.getBoundingClientRect();tooltip.textContent=el.dataset.help;tooltip.style.left=Math.min(r.left,window.innerWidth-390)+'px';tooltip.style.top=(r.bottom+7)+'px';tooltip.classList.add('visible');},500);});el.addEventListener('mouseleave',()=>{clearTimeout(tooltipTimer);tooltip.classList.remove('visible');});});
    setConfig(initial); document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>vscode.postMessage({type:'pick',field:b.dataset.pick,kind:b.dataset.kind}));
    document.getElementById('save').onclick=()=>vscode.postMessage({type:'saveConfig',config:getConfig()}); document.getElementById('load').onclick=()=>vscode.postMessage({type:'loadConfig'});
    document.getElementById('loadFragmenter').onclick=()=>vscode.postMessage({type:'loadFragmenter'}); document.getElementById('saveFragmenter').onclick=()=>vscode.postMessage({type:'saveFragmenter',path:form.elements.params.value,text:fragmenterEditor.value});
    form.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'run',config:getConfig(),fragmenterText:fragmenterEditor.value});}; stop.onclick=()=>vscode.postMessage({type:'stop'});
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='picked')form.elements[m.field].value=m.value;if(m.type==='config')setConfig(m.config);if(m.type==='fragmenter'){form.elements.params.value=m.path;fragmenterEditor.value=m.text;}if(m.type==='fragmenterSaved')form.elements.params.value=m.path;if(m.type==='status'){statusEl.textContent=m.text;statusEl.className='status '+m.status;stop.disabled=m.status!=='running';if(m.command)document.getElementById('command').textContent=m.command;}});`;
}
function commonCss() { return `:root{color-scheme:light dark;--accent:#36c5a2;--panel:color-mix(in srgb,var(--vscode-editor-background) 88%,var(--vscode-editor-foreground));--border:color-mix(in srgb,var(--vscode-editor-foreground) 18%,transparent)}*{box-sizing:border-box}body{font-family:var(--vscode-font-family);color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);margin:0}main{max-width:1100px;margin:auto;padding:32px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:28px}h1{font-size:32px;margin:4px 0}h2{font-size:17px;margin:0 0 18px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.14em;font-size:11px}.muted,small{opacity:.65}button{font:inherit;color:inherit;background:var(--vscode-button-secondaryBackground);border:1px solid var(--border);border-radius:6px;padding:8px 13px;cursor:pointer}button:hover{background:var(--vscode-button-secondaryHoverBackground)}section{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:22px;margin:14px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;background:none;border:0;padding:0}.cards article{background:var(--panel);border:1px solid var(--border);padding:18px;border-radius:9px}.cards b{display:block;font-size:23px;color:var(--accent)}.cards span{opacity:.65}.files{display:grid;gap:4px}.file{display:flex;justify-content:space-between;text-align:left;background:transparent;border:0;border-bottom:1px solid var(--border);border-radius:0}.file em{opacity:.55;font-style:normal}dl{display:grid;grid-template-columns:110px 1fr;gap:10px}dt{opacity:.6}dd{margin:0;overflow-wrap:anywhere}code{font-family:var(--vscode-editor-font-family);font-size:12px}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}main{padding:18px}}`; }
function formCss() { return `nav{display:flex;gap:8px;margin-bottom:18px}.tab{border-radius:20px}.tab.active{border-color:var(--accent)}label{display:block;font-size:12px;opacity:.8;margin:12px 0}input,select,textarea{width:100%;display:block;margin-top:6px;padding:9px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,var(--border));border-radius:5px;font:inherit}.path{display:flex;gap:7px}.path input{flex:1}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0 16px}.checks{display:flex;flex-wrap:wrap;gap:8px 22px}.check{display:flex;align-items:center;gap:7px}.check input{width:auto;margin:0}.section-title{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.section-title h2{margin-bottom:4px}.section-title p{margin:0}.actions{display:flex;gap:8px}textarea{min-height:280px;resize:vertical;font-family:var(--vscode-editor-font-family);font-size:12px;line-height:1.5}#helpTooltip{position:fixed;z-index:50;display:none;max-width:380px;padding:9px 11px;border:1px solid var(--vscode-editorHoverWidget-border,var(--border));border-radius:5px;background:var(--vscode-editorHoverWidget-background);color:var(--vscode-editorHoverWidget-foreground);box-shadow:0 4px 14px #0005;font-size:12px;line-height:1.4}#helpTooltip.visible{display:block}.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground);font-weight:600}.primary:hover{background:var(--vscode-button-hoverBackground)}footer{position:sticky;bottom:0;background:var(--vscode-editor-background);border-top:1px solid var(--border);padding:17px 0;display:flex;justify-content:space-between;align-items:center;gap:20px}.status{font-weight:600}.status.running{color:#e9b949}.status.completed{color:var(--accent)}.status.error,.status.failed{color:#ef6b73}#command{display:block;opacity:.65;max-width:700px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:5px}@media(max-width:700px){.grid{grid-template-columns:1fr}footer{position:static}}`; }

function deactivate() { if (runningProcess) runningProcess.kill('SIGTERM'); }
module.exports = { activate, deactivate, buildArgs, normalizeConfig };
