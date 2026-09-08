const trainingMetrics = require('./features/training-metrics/editor');
const fineTune = require('./features/fragment-tree-finetune/editor');
const smartsSearch = require('./features/smarts-search/editor');
const evaluation = require('./features/evaluation/editor');
const spectrumPrediction = require('./features/spectrum-prediction/editor');
const { createFragmenterEditor } = require('./components/fragmenter-editor');
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const cleavagePatternSetEditor = require('./features/cleavage-pattern-set/editor');

const { readScores, distributionHtml } = require('./features/fragment-tree-result/score-distribution');

const RESULT_NAME = 'fragment-tree.pft';
let runningProcess;

function activate(context) {
  const output = vscode.window.createOutputChannel('CLEFTS');
  const provider = new ResultEditorProvider(context);
  context.subscriptions.push(
    output,
    vscode.commands.registerCommand('clefts.openWorkbench', () => openWorkbench(context, output)),
    vscode.commands.registerCommand('clefts.openResult', openResultPicker),
    vscode.commands.registerCommand('clefts.openEvaluation', () => evaluation.open(context, output, projectRoot)),
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
    requirePrecursorPathTargets: true,
    overwrite: false, overwritePreprocessingConfig: false, saveTrainValidRecords: false,
    saveValidationValidRecords: true, keepParallelTemp: false
  };
}

function defaultTrainingConfig(context) {
  const root = projectRoot(context);
  return {
    application: 'fragment-tree-training', trainDir: '', valDir: '',
    outputDir: path.join(root, 'data', 'training', 'fragment_tree_models', 'gui-run'),
    molEncoderCheckpoint: '', numWorkers: 4,
    conditionAdductEmbeddingDim: 16, conditionCeFeatureDim: 16,
    conditionCeFcDims: '32', conditionFeatureDim: 64, conditionFcDims: '128,64',
    treeHiddenDim: 128, treeNumLayers: 2, treeNumHeads: 8, treeMaxDegree: 16,
    dropout: 0.5, edgeFeatureDim: 256, edgeCategoryDim: 32,
    edgeAttentionHeads: 8, attentionMaxGraphDistance: 4,
    maxEdgesPerDepth: '128,64,32', trainingEdgesPerSample: 32,
    trainingZeroEdgeFraction: 0.25, assignmentScoreThreshold: 0.8,
    maxSamples: 100, maxEdgesPerStep: 128,
    maxRetainedEdges: 30, maxEdgesPerTree: 256, maxNextCleavageCandidates: 3,
    edgeConditionInteractionDim: 64, rankingLossWeight: 1.0,
    topN: 10, nearestLowerPartners: 1, extendedLowerPartners: 3, backgroundPartners: 10, rankingIntensityThreshold: 0.05,
    experimentName: 'exp_main', ckptId: '', batchSize: 1, device: 'cpu',
    epochs: 10, validationIntervalSteps: 100, stepValidationFraction: 1.0, trainLogIntervalSteps: 50,
    validateAtStart: false, detectAnomaly: false, profilePerformance: false,
    saveIntervalEpochs: 1, saveIntervalSteps: 100, optimizer: 'AdamW',
    lr: 0.00001, weightDecay: 0, gradClipNorm: 1.0,
    earlyStoppingPatience: '', shuffle: true
  };
}

function openWorkbench(context, output) {
  const panel = vscode.window.createWebviewPanel('clefts.workbench', 'CLEFTS Workbench', vscode.ViewColumn.One, {
    enableScripts: true, retainContextWhenHidden: true
  });
  smartsSearch.attach(panel, context, projectRoot);
  fineTune.attach(panel, context, projectRoot, output);
  spectrumPrediction.attach(panel, context, projectRoot, output);
  trainingMetrics.attach(panel);
  const initialConfig = defaultConfig(context);
  const initialTrainingConfig = defaultTrainingConfig(context);
  let initialFragmenter = '{}';
  try { initialFragmenter = fs.readFileSync(initialConfig.params, 'utf8'); } catch (_) {}
  panel.webview.html = workbenchHtml(initialConfig, initialFragmenter, initialTrainingConfig);
  panel.webview.onDidReceiveMessage(async message => {
    try {
      if (message.type === 'pick') {
        const folders = message.kind === 'folder';
        const picked = await vscode.window.showOpenDialog({ canSelectFiles: !folders, canSelectFolders: folders, canSelectMany: false });
        if (picked && picked[0]) {
          panel.webview.postMessage({ type: 'picked', form: message.form, field: message.field, value: picked[0].fsPath });
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
          if (!config.fragmenterParams && config.params) config.fragmenterParams = JSON.parse(await fs.promises.readFile(path.resolve(path.dirname(picked[0].fsPath), config.params), 'utf8'));
          panel.webview.postMessage({ type: 'config', config: { ...defaultConfig(context), ...config } });
        }
      } else if (message.type === 'saveTrainingConfig') {
        const target = await vscode.window.showSaveDialog({ filters: { 'CLEFTS training configuration': ['pfttrain.json'] }, defaultUri: vscode.Uri.file('fragment_tree.pfttrain.json') });
        if (target) await fs.promises.writeFile(target.fsPath, JSON.stringify(normalizeTrainingConfig(message.config), null, 2) + '\n');
      } else if (message.type === 'loadTrainingConfig') {
        const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS training configuration': ['pfttrain.json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const config = JSON.parse(await fs.promises.readFile(picked[0].fsPath, 'utf8'));
          panel.webview.postMessage({ type: 'trainingConfig', config: { ...defaultTrainingConfig(context), ...config } });
        }
      } else if (message.type === 'openResult') {
        await openResultPicker();
      } else if (message.type === 'openEvaluation') {
        evaluation.open(context, output, projectRoot);
      } else if (message.type === 'copyCommand') {
        const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
        const command = shellDisplay(python, buildArgs(normalizeConfig(message.config)));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'status', status: 'idle', text: 'Command copied.', command });
      } else if (message.type === 'run') {
        await runFragmentTree(context, output, normalizeConfig(message.config), panel);
      } else if (message.type === 'copyTrainingCommand') {
        const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
        const command = shellDisplay(python, buildTrainingArgs(normalizeTrainingConfig(message.config)));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'trainingStatus', status: 'idle', text: 'Command copied.', command });
      } else if (message.type === 'runTraining') {
        await runTraining(context, output, normalizeTrainingConfig(message.config), panel);
      } else if (message.type === 'pickCheckpoint') {
        const checkpointId = await showCheckpointPicker(
          message.outputDir, message.experimentName
        );
        if (checkpointId !== undefined) {
          panel.webview.postMessage({ type: 'checkpointPicked', checkpointId });
        }
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
        const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern Set': ['json'] }, defaultUri });
        if (selected) {
          const target = vscode.Uri.file(ensureFileSuffix(selected.fsPath, cleavagePatternSetEditor.FILE_SUFFIX));
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(value, null, 2)}\n`);
          panel.webview.postMessage({ type: 'cleavagePatternSetSaved', path: target.fsPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'loadCleavagePattern') {
        const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS Cleavage Pattern': ['json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const pattern = normalizePattern(JSON.parse(await fs.promises.readFile(picked[0].fsPath, 'utf8')));
          panel.webview.postMessage({ type: 'cleavagePatternLoaded', pattern, index: message.index });
        }
      } else if (message.type === 'saveCleavagePattern') {
        const pattern = normalizePattern(message.pattern);
        const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern': ['json'] }, defaultUri: vscode.Uri.file(`${safeFileStem(pattern.name || 'pattern')}.cleavage.json`) });
        if (selected) {
          const target = vscode.Uri.file(ensureFileSuffix(selected.fsPath, '.cleavage.json', ['.clevage.json']));
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(pattern, null, 2)}\n`);
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'selectElements') {
        const elements = await showElementPicker(message.elements);
        if (elements) panel.webview.postMessage({ type: 'elementSelectionResult', requestId: message.requestId, elements });
      } else if (message.type === 'chemistry') {
        const result = await runChemistryBackend(context, message.command, message.payload);
        panel.webview.postMessage({ type: 'chemistryResult', requestId: message.requestId, result });
      } else if (message.type === 'predictSpectrum') {
        try {
          const result = await runSpectrumPredictionBackend(context, { command: 'predict', payload: message.payload });
          panel.webview.postMessage({ type: 'predictSpectrumResult', requestId: message.requestId, result });
        } catch (error) {
          panel.webview.postMessage({ type: 'predictSpectrumError', requestId: message.requestId, message: String(error.message || error) });
        }
      } else if (message.type === 'listPredictAdducts') {
        try {
          const result = await runSpectrumPredictionBackend(context, { command: 'listAdducts', payload: message.payload });
          panel.webview.postMessage({ type: 'listPredictAdductsResult', requestId: message.requestId, result });
        } catch (error) {
          panel.webview.postMessage({ type: 'listPredictAdductsError', requestId: message.requestId, message: String(error.message || error) });
        }
      }
    } catch (error) {
      if (message.type === 'chemistry') panel.webview.postMessage({ type: 'chemistryError', requestId: message.requestId, message: String(error.message || error) });
      const trainingMessage = ['runTraining', 'copyTrainingCommand', 'saveTrainingConfig', 'loadTrainingConfig'].includes(message.type);
      panel.webview.postMessage({ type: trainingMessage ? 'trainingStatus' : 'status', status: 'error', text: String(error.message || error) });
      vscode.window.showErrorMessage(`CLEFTS: ${error.message || error}`);
    }
  });
}

async function showCheckpointPicker(outputDir, experimentName) {
  if (!outputDir) throw new Error('Select the model output directory first.');
  const treePath = path.join(
    outputDir, 'experiments', experimentName || 'exp_main',
    'checkpoints', 'ckpt_tree.tsv'
  );
  if (!fs.existsSync(treePath)) {
    throw new Error(`Checkpoint history was not found: ${treePath}`);
  }
  const lines = (await fs.promises.readFile(treePath, 'utf8')).trim().split(/\r?\n/);
  const headers = lines.shift().split('\t');
  const rows = lines.filter(Boolean).map(line => {
    const values = line.split('\t');
    return Object.fromEntries(headers.map((header, index) => [header, values[index] || '']));
  }).reverse();
  return new Promise(resolve => {
    const picker = vscode.window.createWebviewPanel(
      'clefts.checkpointHistory', 'CLEFTS Checkpoint History',
      vscode.ViewColumn.Active, { enableScripts: true }
    );
    const branches = [...new Set(rows.map(row => row.branch_id))];
    picker.webview.html = `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}.history{max-width:1050px}.toolbar{display:flex;gap:8px;align-items:center}.toolbar input{flex:1;padding:8px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--border)}table{font-size:12px}.graph{display:flex;align-items:center;min-width:${Math.max(branches.length * 20, 30)}px;height:34px}.lane{width:20px;height:34px;position:relative}.lane::before{content:'';position:absolute;left:9px;top:-8px;bottom:-8px;border-left:2px solid color-mix(in srgb,var(--accent) 45%,transparent)}.lane.node::after{content:'';position:absolute;left:4px;top:12px;width:10px;height:10px;border:2px solid var(--accent);border-radius:50%;background:var(--vscode-editor-background)}tr{cursor:pointer}tr:hover td{background:color-mix(in srgb,var(--accent) 10%,transparent)}tr.selected td{background:color-mix(in srgb,var(--accent) 20%,transparent)}code{white-space:nowrap}</style></head><body><main class="history"><header><div><span class="eyebrow">CHECKPOINT GRAPH</span><h1>${escapeHtml(experimentName || 'exp_main')}</h1><p class="muted">${escapeHtml(treePath)}</p></div></header><div class="toolbar"><input id="filter" placeholder="Filter ID, name, epoch, step, or comment"><button id="cancel">Cancel</button><button id="use" class="primary" disabled>Use checkpoint</button></div><table><thead><tr><th>Graph</th><th>ID</th><th>Name</th><th>Epoch</th><th>Step</th><th>Parent</th><th>Comment</th></tr></thead><tbody id="rows"></tbody></table></main><script>const vscode=acquireVsCodeApi(),rows=${safeJson(rows)},branches=${safeJson(branches)},esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let selected='';const body=document.getElementById('rows'),filter=document.getElementById('filter'),use=document.getElementById('use');function render(){const q=filter.value.toLowerCase();body.innerHTML=rows.filter(r=>Object.values(r).join(' ').toLowerCase().includes(q)).map(r=>'<tr data-id="'+esc(r.id)+'" class="'+(selected===r.id?'selected':'')+'"><td><div class="graph">'+branches.map(b=>'<span class="lane '+(b===r.branch_id?'node':'')+'"></span>').join('')+'</div></td><td><code>'+esc(r.id)+'</code></td><td>'+esc(r.name)+'</td><td>'+esc(r.epoch)+'</td><td>'+esc(r.iter)+'</td><td><code>'+esc(r.parent_id)+'</code></td><td>'+esc(r.comment)+'</td></tr>').join('')}body.onclick=e=>{const row=e.target.closest('tr[data-id]');if(!row)return;selected=row.dataset.id;use.disabled=false;render()};body.ondblclick=e=>{const row=e.target.closest('tr[data-id]');if(row)vscode.postMessage({type:'select',id:row.dataset.id})};filter.oninput=render;use.onclick=()=>vscode.postMessage({type:'select',id:selected});document.getElementById('cancel').onclick=()=>vscode.postMessage({type:'cancel'});render()</script></body></html>`;
    let settled = false;
    picker.webview.onDidReceiveMessage(message => {
      if (message.type === 'select') { settled = true; resolve(message.id); picker.dispose(); }
      if (message.type === 'cancel') picker.dispose();
    });
    picker.onDidDispose(() => { if (!settled) resolve(undefined); });
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

function ensureFileSuffix(filePath, suffix, acceptedAliases = []) {
  let result = String(filePath);
  const accepted = [suffix, ...acceptedAliases].map(value => String(value).toLowerCase());
  while (result.toLowerCase().endsWith('.json.json')) result = result.slice(0, -5);
  for (const ending of accepted) {
    while (result.toLowerCase().endsWith(ending + ending)) result = result.slice(0, -ending.length);
  }
  if (accepted.some(ending => result.toLowerCase().endsWith(ending))) return result;
  if (result.toLowerCase().endsWith('.json')) result = result.slice(0, -5);
  return `${result}${suffix}`;
}

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

function runSpectrumPredictionBackend(context, payload) {
  return new Promise((resolve, reject) => {
    const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'spectrum-prediction', 'backend.py');
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
      catch (_) { reject(new Error(stderr || stdout || 'The spectrum prediction backend did not return a response.')); }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

function normalizeConfig(config) {
  const { params, ...values } = config;
  return { ...values, symbols: Array.isArray(config.symbols) ? config.symbols : String(config.symbols || '').split(/[ ,]+/).filter(Boolean) };
}

function normalizeTrainingConfig(config) {
  const fraction = Number(config.stepValidationFraction ?? 1.0);
  if (!Number.isFinite(fraction) || fraction <= 0 || fraction > 1) {
    throw new Error('Step validation fraction must be greater than 0 and at most 1.');
  }
  return { ...config, stepValidationFraction: fraction };
}

async function runTraining(context, output, config, panel) {
  if (runningProcess) throw new Error('Another CLEFTS process is already running.');
  for (const key of ['trainDir', 'valDir', 'outputDir', 'molEncoderCheckpoint']) {
    if (!config[key]) throw new Error(`${key} is required.`);
  }
  const root = projectRoot(context);
  const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
  const args = buildTrainingArgs(config);
  const command = shellDisplay(python, args);
  output.clear(); output.show(true); output.appendLine(`$ ${command}`);
  panel.webview.postMessage({ type: 'trainingStatus', status: 'running', text: 'Running training CLI…', command });
  runningProcess = spawn(python, args, { cwd: root, env: process.env });
  runningProcess.stdout.on('data', chunk => output.append(chunk.toString()));
  runningProcess.stderr.on('data', chunk => output.append(chunk.toString()));
  runningProcess.on('error', error => {
    runningProcess = undefined;
    panel.webview.postMessage({ type: 'trainingStatus', status: 'error', text: error.message });
  });
  runningProcess.on('close', code => {
    runningProcess = undefined;
    panel.webview.postMessage({ type: 'trainingStatus', status: code === 0 ? 'completed' : 'failed', text: code === 0 ? 'Training completed.' : `Training failed with exit code ${code}.` });
    if (code === 0) vscode.window.showInformationMessage('CLEFTS fragment tree training completed.');
  });
}

async function runFragmentTree(context, output, config, panel) {
  if (runningProcess) throw new Error('Another CLEFTS process is already running.');
  for (const key of ['trainInput', 'outputDir', 'fragmenterParams']) if (!config[key]) throw new Error(`${key} is required.`);
  if (!config.symbols.length) throw new Error('At least one symbol is required.');
  const root = projectRoot(context);
  const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
  await fs.promises.mkdir(config.outputDir, { recursive: true });
  const configPath = path.join(config.outputDir, 'fragment-tree.pft.json');
  const resultPath = path.join(config.outputDir, 'train_structures', RESULT_NAME);
  await fs.promises.writeFile(configPath, JSON.stringify(config, null, 2) + '\n');
  const args = buildArgs(config);
  output.clear(); output.show(true); output.appendLine(`$ ${shellDisplay(python, args)}`);
  panel.webview.postMessage({ type: 'status', status: 'running', text: 'Running CLI…', command: shellDisplay(python, args) });
  runningProcess = spawn(python, args, { cwd: root, env: process.env });
  runningProcess.stdout.on('data', chunk => output.append(chunk.toString()));
  runningProcess.stderr.on('data', chunk => output.append(chunk.toString()));
  runningProcess.on('error', async error => {
    runningProcess = undefined;
    panel.webview.postMessage({ type: 'status', status: 'error', text: error.message });
  });
  runningProcess.on('close', async code => {
    runningProcess = undefined;
    const status = code === 0 ? 'completed' : 'failed';
    panel.webview.postMessage({ type: 'status', status, text: code === 0 ? 'Completed.' : `Failed with exit code ${code}.`, resultPath });
    if (code === 0 && fs.existsSync(resultPath)) vscode.window.showInformationMessage('CLEFTS fragment tree data preparation completed.', 'Open Train Result').then(choice => { if (choice) openResult(vscode.Uri.file(resultPath)); });
  });
}

function buildArgs(c) {
  const a = ['-m', 'clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data', '--train-input', c.trainInput, '--output-dir', c.outputDir, '--params-json', JSON.stringify(c.fragmenterParams), '--symbols', ...c.symbols];
  const values = {
    validationInput: '--validation-input', validationStructuresInputDir: '--validation-structures-input-dir',
    validationSmilesRatio: '--validation-smiles-ratio', tanimotoNumBins: '--tanimoto-num-bins', tanimotoRadius: '--tanimoto-radius', tanimotoNBits: '--tanimoto-n-bits', validationSamplingSeed: '--validation-sampling-seed',
    smilesColumn: '--smiles-column', precursorMzColumn: '--precursor-mz-column', adductTypeColumn: '--adduct-type-column', collisionEnergyColumn: '--collision-energy-column', instrumentColumn: '--instrument-column',
    preprocessingConfigOutput: '--preprocessing-config-output', trainValidOutput: '--train-valid-output', validationValidOutput: '--validation-valid-output', trainAssignmentScoreOutput: '--train-assignment-score-output', validationAssignmentScoreOutput: '--validation-assignment-score-output', parallelTempDir: '--parallel-temp-dir',
    maxNode: '--max-node', maxEdge: '--max-edge', numWorkers: '--num-workers', chunkSize: '--chunk-size', structureRebuildPolicy: '--structure-rebuild-policy'
  };
  for (const [key, flag] of Object.entries(values)) if (c[key] !== '' && c[key] !== null && c[key] !== undefined) a.push(flag, String(c[key]));
  const flags = { overwrite: '--overwrite', overwritePreprocessingConfig: '--overwrite-preprocessing-config', saveTrainValidRecords: '--save-train-valid-records', keepParallelTemp: '--keep-parallel-temp' };
  for (const [key, flag] of Object.entries(flags)) if (c[key]) a.push(flag);
  if (c.saveValidationValidRecords === false) a.push('--no-save-validation-valid-records');
  if (c.requirePrecursorPathTargets === false) a.push('--no-require-precursor-path-targets');
  return a;
}

function buildTrainingArgs(c) {
  c = normalizeTrainingConfig(c);
  const a = ['-m', 'clefts.ml.training.fragment_tree_training.training_model',
    '--train-dir', c.trainDir, '--val-dir', c.valDir, '--output-dir', c.outputDir,
    '--mol-encoder-checkpoint', c.molEncoderCheckpoint];
  const values = {
    numWorkers: '--num-workers', conditionAdductEmbeddingDim: '--condition-adduct-embedding-dim',
    conditionCeFeatureDim: '--condition-ce-feature-dim', conditionCeFcDims: '--condition-ce-fc-dims',
    conditionFeatureDim: '--condition-feature-dim', conditionFcDims: '--condition-fc-dims',
    treeHiddenDim: '--tree-hidden-dim', treeNumLayers: '--tree-num-layers',
    treeNumHeads: '--tree-num-heads', treeMaxDegree: '--tree-max-degree', dropout: '--dropout',
    edgeFeatureDim: '--edge-feature-dim', edgeCategoryDim: '--edge-category-dim',
    edgeAttentionHeads: '--edge-attention-heads', attentionMaxGraphDistance: '--attention-max-graph-distance',
    maxEdgesPerDepth: '--max-edges-per-depth', trainingEdgesPerSample: '--training-edges-per-sample',
    trainingZeroEdgeFraction: '--training-zero-edge-fraction',
    assignmentScoreThreshold: '--assignment-score-threshold', maxSamples: '--max-samples',
    maxEdgesPerStep: '--max-edges-per-step', maxRetainedEdges: '--max-retained-edges',
    maxEdgesPerTree: '--max-edges-per-tree',
    maxNextCleavageCandidates: '--max-next-cleavage-candidates',
    edgeConditionInteractionDim: '--edge-condition-interaction-dim', rankingLossWeight: '--ranking-loss-weight',
    topN: '--top-n', nearestLowerPartners: '--nearest-lower-partners',
    extendedLowerPartners: '--extended-lower-partners', backgroundPartners: '--background-partners',
    rankingIntensityThreshold: '--ranking-intensity-threshold',
    experimentName: '--experiment-name', ckptId: '--ckpt-id', batchSize: '--batch-size', device: '--device',
    epochs: '--epochs', validationIntervalSteps: '--validation-interval-steps',
    stepValidationFraction: '--step-validation-fraction',
    trainLogIntervalSteps: '--train-log-interval-steps', saveIntervalEpochs: '--save-interval-epochs',
    saveIntervalSteps: '--save-interval-steps', optimizer: '--optimizer', lr: '--lr',
    weightDecay: '--weight-decay', gradClipNorm: '--grad-clip-norm',
    earlyStoppingPatience: '--early-stopping-patience'
  };
  for (const [key, flag] of Object.entries(values)) {
    if (c[key] !== '' && c[key] !== null && c[key] !== undefined) a.push(flag, String(c[key]));
  }
  if (c.validateAtStart) a.push('--validate-at-start');
  if (c.detectAnomaly) a.push('--detect-anomaly');
  if (c.profilePerformance) a.push('--profile-performance');
  if (c.shuffle === false) a.push('--no-shuffle');
  return a;
}

function shellDisplay(program, args) { return [program, ...args].map(v => /^[A-Za-z0-9_./:=,-]+$/.test(v) ? v : `'${v.replace(/'/g, "'\\''")}'`).join(' '); }
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
      if (m.type === 'exportScoreDistribution') {
        try {
          if (!['png', 'tsv'].includes(m.format) || typeof m.data !== 'string') return;
          const target = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(path.join(path.dirname(document.uri.fsPath), 'assignment-score-distribution.' + m.format)),
            filters: m.format === 'png' ? { 'PNG image': ['png'] } : { 'TSV table': ['tsv'] }
          });
          if (target) await vscode.workspace.fs.writeFile(target, Buffer.from(m.data, m.format === 'png' ? 'base64' : 'utf8'));
        } catch (error) { vscode.window.showErrorMessage('Unable to save distribution: ' + error.message); }
        return;
      }
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
        try { panel.webview.postMessage({ type: 'moleculeDetail', smiles: m.smiles, node: m.node, result: await runChemistryBackend(this.context, 'depict', { smiles: m.smiles, adducts: m.adducts || [], usedAdduct: m.usedAdduct, stateModified: m.stateModified, precursorSmiles: m.precursorSmiles }) }); }
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
  const scoreDatasets = await readScores(root);
  const summary = summarize(files);
  summary.preft = manifests.reduce((total, item) => total + item.rows.filter(row => row.exists).length, 0);
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}:root{--detected:#ff8a3d}.result-main{max-width:none;width:100%}.tree-scroll{overflow:auto;max-height:70vh;overscroll-behavior:contain;border:1px solid var(--border);border-radius:8px;background:var(--vscode-editor-background);scrollbar-gutter:stable}.fragment-tree{display:block;width:100%;min-width:700px}.tree-edge{stroke:color-mix(in srgb,var(--vscode-editor-foreground) 35%,transparent);stroke-width:1.5}.tree-edge.detected{stroke:var(--detected);stroke-width:3}.tree-node circle{fill:var(--vscode-editor-background);stroke:var(--vscode-editor-foreground);stroke-width:1.5}.tree-node.detected circle{fill:color-mix(in srgb,var(--detected) 28%,var(--vscode-editor-background));stroke:var(--detected);stroke-width:3}.tree-node text{fill:var(--vscode-editor-foreground);font-size:9px;pointer-events:none}.tree-node{cursor:pointer}.molecule-preview{width:100%;max-height:360px;overflow:hidden;background:#fff;border-radius:6px}.molecule-preview svg{display:block;width:100%;height:auto}.chem-info{width:100%;font-size:11px}.chem-info th,.chem-info td{padding:5px}.used-adduct{background:color-mix(in srgb,var(--accent) 24%,transparent);outline:1px solid var(--accent)}.used-adduct.modified-state{background:color-mix(in srgb,#b06cff 30%,transparent);outline-color:#b06cff}.used-adduct td:first-child::before{content:'✓ ';color:var(--accent);font-weight:700}.used-adduct.modified-state td:first-child::before{color:#b06cff}.manifest-grid{margin-top:18px}.manifest-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}.manifest-controls input{min-width:260px;flex:1;padding:7px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--border);border-radius:5px}.manifest-controls label{display:flex;align-items:center;gap:6px}.manifest-controls select{width:auto;margin:0}.manifest-scroll{max-height:none;overflow-x:auto;overflow-y:visible}.column-filters input{width:100%;min-width:80px;padding:4px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--border)}.missing-file{display:block;color:var(--vscode-errorForeground)}.tree-controls,.detail-controls{display:flex;justify-content:flex-end;align-items:center;gap:8px;flex-wrap:wrap;margin:8px 0}.tree-controls select{width:auto}.tree-controls button{padding:5px 9px}.detail-controls input{width:180px;margin:0}.sortable{cursor:pointer;user-select:none}.sortable:hover{color:var(--accent)}.detail-layout{display:grid;grid-template-columns:minmax(0,1fr) var(--molecule-width,35%);gap:18px;align-items:start}.molecule-side{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto;border:1px solid var(--border);border-radius:8px;padding:14px}.molecule-side[hidden]{display:block;visibility:hidden}.molecule-side code{display:block;overflow-wrap:anywhere;margin:8px 0}.target-diagnostic{font-size:11px;padding:3px 0;border:0}.target-diagnostic summary{font-weight:400}.target-diagnostic code{display:block;white-space:normal;margin:2px 0}@media(max-width:900px){.detail-layout{grid-template-columns:1fr}.molecule-side{position:static;max-height:none}.molecule-side[hidden]{display:none}}</style></head><body><main class="result-main">
    <header><div><span class="eyebrow">CLEFTS RESULT</span><h1>Fragment Tree Dataset</h1><p class="muted">${escapeHtml(root)}</p></div><button onclick="vscode.postMessage({type:'refresh'})">Refresh</button></header>
    <section class="cards"><article><b>${escapeHtml(manifest.status || 'unknown')}</b><span>status</span></article><article><b>${summary.preft}</b><span>structures (.preft.pt)</span></article><article><b>${summary.tsv}</b><span>tables (.tsv)</span></article><article><b>${formatBytes(summary.bytes)}</b><span>listed size</span></article></section>
    <section><h2>Run Information</h2><dl><dt>Application</dt><dd>${escapeHtml(manifest.application || '')}</dd><dt>Finished</dt><dd>${escapeHtml(manifest.finishedAt || '—')}</dd><dt>Exit code</dt><dd>${escapeHtml(String(manifest.exitCode ?? '—'))}</dd><dt>Command</dt><dd><code>${escapeHtml((manifest.command || []).join(' '))}</code></dd></dl></section>
    <section><h2>Structure Manifests</h2><p class="muted">Only manifest rows are shown; structure data directories are not enumerated.</p>${manifests.map((manifest,index)=>manifestTableHtml(manifest,index)).join('') || '<p>No structure manifest was found.</p>'}</section>
    <section><h2>Supporting Files <small>${files.length} files</small></h2><div class="files">${files.map(f => `<button class="file" data-open-file="${encodeURIComponent(f.relative)}"><span>${escapeHtml(f.relative)}</span><em>${formatBytes(f.size)}</em></button>`).join('')}</div></section>
    <section id="detail" hidden></section>
    <script>const vscode=acquireVsCodeApi(),detail=document.getElementById('detail'),sortState=new Map();const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function graphSvg(g,sample,layout='depth'){if(!g||!g.nodes.length)return '';const detectedEdges=new Set(sample.targetPathEdges),detectedNodes=new Set(sample.targetTerminalNodes),precursors=new Set(sample.precursorNodes),hasStoredDepth=layout==='depth'&&g.nodes.every(n=>Number.isInteger(n.depth)&&n.depth>=0),depth=new Map(g.nodes.map(n=>[n.id,hasStoredDepth?n.depth:0]));if(!hasStoredDepth)for(let pass=0;pass<g.nodes.length;pass++){let changed=false;for(const e of g.edges){const value=(depth.get(e.source)||0)+1;if(value>(depth.get(e.target)||0)){depth.set(e.target,value);changed=true}}if(!changed)break}const levels={};for(const n of g.nodes)(levels[depth.get(n.id)]||(levels[depth.get(n.id)]=[])).push(n);for(const nodes of Object.values(levels))nodes.sort((a,b)=>a.id-b.id);const pos=new Map(),width=Math.max(900,(Math.max(...depth.values(),1)+1)*220),maxRows=Math.max(...Object.values(levels).map(x=>x.length)),height=Math.max(240,maxRows*58+50),maxDepth=Math.max(...depth.values(),1);for(const [d,nodes] of Object.entries(levels))nodes.forEach((n,i)=>pos.set(n.id,{x:45+Number(d)/maxDepth*(width-90),y:30+(i+1)*(height-40)/(nodes.length+1)}));const lines=g.edges.map(e=>{const a=pos.get(e.source),b=pos.get(e.target);return '<line class="tree-edge '+(detectedEdges.has(e.id)?'detected':'')+'" x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y+'"><title>edge '+e.id+'</title></line>'}).join('');const nodes=g.nodes.map(n=>{const p=pos.get(n.id),label=(precursors.has(n.id)?'p':'')+n.id;return '<g class="tree-node '+(detectedNodes.has(n.id)?'detected':'')+'" data-node="'+n.id+'" data-smiles="'+esc(n.smiles)+'" transform="translate('+p.x+' '+p.y+')"><circle r="13"><title>'+esc(n.smiles)+'</title></circle><text text-anchor="middle" dominant-baseline="central">'+label+'</text></g>'}).join('');return '<div class="tree-scroll"><svg class="fragment-tree" viewBox="0 0 '+width+' '+height+'">'+lines+nodes+'</svg></div>'+(g.truncated?'<p class="warning">Tree preview is limited to the first 500 nodes.</p>':'')}
    function render(r){const s=r.summary,t=s.targets;detail.hidden=false;detail.innerHTML='<h2>Structure Detail</h2><p class="muted">'+esc(r.file)+'</p><div class="detail-controls"><label>Molecule pane <input type="range" min="20" max="55" value="35" data-pane-width> <span data-pane-value>35%</span></label></div><div class="cards detail-cards"><article><b>'+s.samples+'</b><span>samples</span></article><article><b>'+s.nodes+'</b><span>all calculated nodes</span></article><article><b>'+s.edges+'</b><span>all edges</span></article><article><b>'+s.assignedPeaks+' / '+s.peaks+'</b><span>assigned peaks</span></article></div><p><b>Depths:</b> '+esc(Object.entries(s.depthCounts).map(x=>(x[0]==='0'?'0':'depth '+x[0])+': '+x[1]).join(', ')||'none')+'</p><p><b>Training targets:</b> schema '+esc(t.schemaVersion)+' · '+t.formulas+' formulas · '+t.assignments+' terminal assignments · '+t.pathEdges+' path-edge references · '+t.expandNodes+' expand-node references<br><small>depth policy: '+esc(t.depthPolicy)+' · precursor path required: '+(t.requirePrecursorPath?'yes':'no')+'</small></p><p><b>State notation:</b> ion / unsaturation / radical candidate index. Target details also show each configured candidate, for example <code>0/1/0 ([M-H]+ / [M-2H] / [M])</code>.</p><div class="detail-layout"><div class="detail-main">'+(r.truncated?'<p class="warning">Sample preview truncated for responsiveness.</p>':'')+r.samples.map(sample=>'<details><summary>sample '+sample.id+' · CE '+esc(sample.collisionEnergyRaw)+' ('+(Number.isFinite(sample.collisionEnergy)?sample.collisionEnergy:'?')+') · adduct '+esc(sample.adduct)+' ('+sample.adductIndex+') · '+sample.peakCount+' peaks · assignment score '+Math.max(0,Math.min(1,sample.assignmentScore)).toFixed(4)+'</summary><p class="muted">Orange edges show the complete detected paths. Only detected terminal nodes are orange; intermediate nodes remain unhighlighted. Click a node to inspect its structure.</p>'+'<div class="tree-controls"><label>Layout <select data-tree-layout><option value="depth">Depth columns (aligned)</option><option value="legacy">Legacy layout</option></select></label><label>Tree nodes <select data-tree-mode><option value="active">Active target paths only</option><option value="all">All calculated nodes (preview)</option></select></label><button data-tree-zoom="out">−</button><button data-tree-zoom="reset">100%</button><button data-tree-zoom="in">＋</button></div>'+graphSvg(sample.activeGraph,sample)+'<table><thead><tr><th>#</th><th class="sortable" data-sort="mz">m/z ↕</th><th class="sortable" data-sort="intensity">intensity ↕</th><th>depth</th><th>target formulas and nodes</th></tr></thead><tbody data-sample="'+sample.id+'">'+peakRows(sample.peaks)+'</tbody></table></details>').join('')+'</div><aside id="moleculeDetail" class="molecule-side" hidden><p class="muted">Click a tree node to display its structure.</p></aside></div>';detail.dataset.result=JSON.stringify(r);detail.scrollIntoView({behavior:'smooth'});}
    function stateText(a){const indexes=a.ion+'/'+a.unsaturation+'/'+a.radical;if(!a.stateLabels)return indexes;return indexes+' ('+a.stateLabels.ion+' / '+a.stateLabels.unsaturation+' / '+a.stateLabels.radical+')'}function pathText(a){if(!a.pathSteps.length)return '['+a.terminalNode+']';return '['+[a.pathSteps[0].source,...a.pathSteps.map(step=>step.target)].join(' - ')+']'}function peakRows(peaks){return peaks.map(p=>'<tr><td>'+p.index+'</td><td>'+p.mz.toFixed(6)+'</td><td>'+p.intensity.toPrecision(5)+'</td><td>'+(p.depth<0?'empty':p.depth)+'</td><td>'+(p.formulas.map(f=>'<div class="formula"><b>'+esc(f.formula)+'</b><small>nodes '+f.assignments.map(a=>a.terminalNode).sort((a,b)=>a-b).join(', ')+'</small><details class="target-diagnostic"><summary>target details</summary>'+f.assignments.map(a=>'<code>node '+a.terminalNode+' · state '+stateText(a)+' · path '+pathText(a)+' · expand nodes ['+[...a.expandNodes].sort((x,y)=>x-y).join(', ')+']</code>').join('')+'</details></div>').join('')||'<span class="muted">No assignment</span>')+'</td></tr>').join('')}
    document.addEventListener('click',e=>{const file=e.target.closest('[data-open-file]');if(file)vscode.postMessage({type:'openFile',path:decodeURIComponent(file.dataset.openFile)})});
    const manifestStates=new WeakMap();
    function updateManifest(grid,resetPage=false){const state=manifestStates.get(grid)||{page:0,sortColumn:-1,sortDirection:1};if(resetPage)state.page=0;const search=grid.querySelector('[data-manifest-search]').value.toLowerCase(),filters=[...grid.querySelectorAll('[data-column-filter]')].map(input=>input.value.toLowerCase()),body=grid.querySelector('tbody'),rows=[...body.rows].filter(row=>{const values=[...row.cells].map(cell=>(cell.dataset.value||'').toLowerCase());return(!search||values.some(value=>value.includes(search)))&&filters.every((filter,index)=>!filter||values[index].includes(filter))});if(state.sortColumn>=0)rows.sort((a,b)=>{const left=a.cells[state.sortColumn].dataset.value||'',right=b.cells[state.sortColumn].dataset.value||'',ln=Number(left),rn=Number(right),result=left!==''&&right!==''&&Number.isFinite(ln)&&Number.isFinite(rn)?ln-rn:left.localeCompare(right,undefined,{numeric:true,sensitivity:'base'});return state.sortDirection*result});for(const row of rows)body.appendChild(row);const pageSize=Number(grid.querySelector('[data-page-size]').value),pages=Math.max(1,Math.ceil(rows.length/pageSize));state.page=Math.min(state.page,pages-1);const pageRows=new Set(rows.slice(state.page*pageSize,(state.page+1)*pageSize));for(const row of body.rows)row.hidden=!pageRows.has(row);grid.querySelector('[data-page-label]').textContent=(rows.length?state.page*pageSize+1:0)+'–'+Math.min((state.page+1)*pageSize,rows.length)+' / '+rows.length+' · page '+(state.page+1)+'/'+pages;grid.querySelector('[data-page-prev]').disabled=state.page===0;grid.querySelector('[data-page-next]').disabled=state.page>=pages-1;manifestStates.set(grid,state)}
    document.querySelectorAll('[data-manifest-grid]').forEach(grid=>updateManifest(grid));
    document.addEventListener('input',e=>{const grid=e.target.closest('[data-manifest-grid]');if(grid&&(e.target.matches('[data-manifest-search]')||e.target.matches('[data-column-filter]')))updateManifest(grid,true)});
    document.addEventListener('change',e=>{const grid=e.target.closest('[data-manifest-grid]');if(grid&&e.target.matches('[data-page-size]'))updateManifest(grid,true)});
    document.addEventListener('click',e=>{const grid=e.target.closest('[data-manifest-grid]');if(!grid)return;const state=manifestStates.get(grid);if(e.target.closest('[data-page-prev]'))state.page--;else if(e.target.closest('[data-page-next]'))state.page++;else{const header=e.target.closest('[data-manifest-sort]');if(!header)return;const column=Number(header.dataset.manifestSort);state.sortDirection=state.sortColumn===column?-state.sortDirection:1;state.sortColumn=column;grid.querySelectorAll('[data-manifest-sort]').forEach(item=>item.textContent=item.dataset.label+' ↕');header.textContent=header.dataset.label+(state.sortDirection>0?' ↑':' ↓');state.page=0}updateManifest(grid)});
    detail.onclick=e=>{const node=e.target.closest('[data-node]');if(node){const result=JSON.parse(detail.dataset.result),body=node.closest('details').querySelector('tbody'),sample=result.samples.find(x=>x.id===Number(body.dataset.sample)),nodeId=Number(node.dataset.node),assignments=sample.peaks.flatMap(p=>p.formulas).flatMap(f=>f.assignments).filter(a=>a.terminalNode===nodeId),usedAdduct=result.registeredAdducts[sample.adductIndex]||'',stateModified=assignments.some(a=>a.stateModified);vscode.postMessage({type:'inspectMolecule',node:nodeId,smiles:node.dataset.smiles,adducts:result.registeredAdducts,usedAdduct,stateModified,precursorSmiles:result.metadata.smiles||result.graph.nodes[0].smiles});return}const zoom=e.target.closest('[data-tree-zoom]');if(zoom){const tree=zoom.closest('details').querySelector('.fragment-tree'),old=Number(tree.dataset.zoom||100),next=zoom.dataset.treeZoom==='reset'?100:Math.max(40,Math.min(300,old+(zoom.dataset.treeZoom==='in'?20:-20)));tree.dataset.zoom=next;tree.style.width=next+'%';tree.style.minWidth=(7*next)+'px';zoom.closest('.tree-controls').querySelector('[data-tree-zoom="reset"]').textContent=next+'%';return}const header=e.target.closest('[data-sort]');if(header){const result=JSON.parse(detail.dataset.result),body=header.closest('details').querySelector('tbody'),sample=result.samples.find(x=>x.id===Number(body.dataset.sample)),key=sample.id+':'+header.dataset.sort,previous=sortState.get(key)||0,direction=previous===0?(header.dataset.sort==='mz'?1:-1):-previous;sortState.set(key,direction);header.closest('tr').querySelectorAll('[data-sort]').forEach(item=>item.textContent=item.dataset.sort==='mz'?'m/z ↕':'intensity ↕');header.textContent=(header.dataset.sort==='mz'?'m/z ':'intensity ')+(direction>0?'↑':'↓');const peaks=[...sample.peaks].sort((a,b)=>direction*(header.dataset.sort==='mz'?a.mz-b.mz:a.intensity-b.intensity));body.innerHTML=peakRows(peaks)}if(e.target.id==='copySmiles')navigator.clipboard.writeText(e.target.dataset.smiles)};
    detail.onchange=e=>{const changed=e.target.closest('[data-tree-mode],[data-tree-layout]');if(!changed)return;const sampleDetail=changed.closest('details'),mode=sampleDetail.querySelector('[data-tree-mode]').value,layout=sampleDetail.querySelector('[data-tree-layout]').value,result=JSON.parse(detail.dataset.result),body=sampleDetail.querySelector('tbody'),sample=result.samples.find(x=>x.id===Number(body.dataset.sample)),current=sampleDetail.querySelector('.tree-scroll'),wrapper=document.createElement('div');wrapper.innerHTML=graphSvg(mode==='active'?sample.activeGraph:result.graph,sample,layout);if(current)current.replaceWith(wrapper.querySelector('.tree-scroll'));sampleDetail.querySelector('[data-tree-zoom="reset"]').textContent='100%'};detail.oninput=e=>{if(e.target.matches('[data-pane-width]')){const layout=detail.querySelector('.detail-layout'),value=e.target.value;layout.style.setProperty('--molecule-width',value+'%');detail.querySelector('[data-pane-value]').textContent=value+'%'}};
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='structureLoading'){detail.hidden=false;detail.innerHTML='<h2>Loading structure…</h2><p class="muted">'+esc(m.path)+'</p>'}if(m.type==='structureDetail')render(m.result);if(m.type==='moleculeDetail'){const box=document.getElementById('moleculeDetail'),rows=m.result.adducts.map(a=>'<tr class="'+(a.used?'used-adduct ':'')+(a.stateModified?'modified-state':'')+'"><td>'+esc(a.adduct)+'</td><td>'+(a.error?'<span class="warning">'+esc(a.error)+'</span>':esc(a.formula))+'</td><td>'+(a.mz===undefined?'—':a.mz.toFixed(6))+'</td><td>'+(a.neutralLoss===null||a.neutralLoss===undefined?'—':a.neutralLoss.toFixed(6))+'</td></tr>').join('');box.hidden=false;box.innerHTML='<h3>Node '+m.node+'</h3><div class="molecule-preview">'+m.result.svg+'</div><code>'+esc(m.smiles)+'</code> <button id="copySmiles" data-smiles="'+esc(m.smiles)+'">Copy SMILES</button><dl><dt>Formula</dt><dd>'+esc(m.result.formula)+'</dd><dt>Exact mass</dt><dd>'+m.result.exactMass.toFixed(6)+'</dd></dl><h4>Registered adducts</h4><p class="muted">Green ✓: assigned main adduct. Purple ✓: assigned main adduct with unsaturation or radical state.</p><div class="table-scroll"><table class="chem-info"><thead><tr><th>Adduct</th><th>Ion formula</th><th>m/z</th><th>NL</th></tr></thead><tbody>'+rows+'</tbody></table></div>'}if(m.type==='structureError')detail.innerHTML='<h2>Unable to inspect structure</h2><pre>'+esc(m.message)+'</pre>'});</script>${distributionHtml(scoreDatasets)}</main></body></html>`;
}

async function readStructureManifests(root) {
  const results = [];
  const localManifest = path.join(root, 'manifest.tsv');
  if (fs.existsSync(localManifest)) {
    const lines = (await fs.promises.readFile(localManifest, 'utf8')).trim().split(/\r?\n/);
    if (lines.length && lines[0]) {
      const columns = lines[0].split('\t');
      const rows = lines.slice(1).map(line => {
        const values = line.split('\t'), row = Object.fromEntries(columns.map((column, index) => [column, values[index] || '']));
        row.relative = row.file ? path.join('data', row.file) : '';
        row.exists = Boolean(row.relative && fs.existsSync(path.join(root, row.relative)));
        return row;
      });
      results.push({ directory: path.basename(root), relativeBase: '', rows });
    }
  }
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
    results.push({ directory, relativeBase: directory, rows });
  }
  return results;
}

function manifestTableHtml(manifest, index) {
  const columns = ['file', 'status', 'smiles', 'num_input_records', 'num_valid_samples', 'rejected_sample_count', 'rejection_log', 'num_nodes', 'num_edges'];
  const rows = manifest.rows.map(row => `<tr>${columns.map(column => { const value = row[column] || ''; if (column === 'file' && value && row.exists) return `<td data-value="${escapeHtml(value)}"><button class="manifest-file" data-open-file="${encodeURIComponent(row.relative)}">${escapeHtml(value)}</button></td>`; if (column === 'file' && value) return `<td data-value="${escapeHtml(value)}"><span>${escapeHtml(value)}</span><small class="missing-file">not generated (${escapeHtml(row.status || 'missing')})</small></td>`; if (column === 'rejection_log' && value) return `<td data-value="${escapeHtml(value)}"><button class="manifest-file" data-open-file="${encodeURIComponent(path.join(manifest.relativeBase || '', value))}">${escapeHtml(value)}</button></td>`; return `<td data-value="${escapeHtml(value)}">${escapeHtml(value)}</td>`; }).join('')}</tr>`).join('');
  return `<div class="manifest-grid" data-manifest-grid="${index}"><h3>${escapeHtml(manifest.directory)}</h3><div class="manifest-controls"><input type="search" data-manifest-search placeholder="Filter all columns…"><label>Rows <select data-page-size><option>20</option><option selected>50</option><option>100</option><option>250</option></select></label><button data-page-prev>Previous</button><span data-page-label></span><button data-page-next>Next</button></div><div class="table-scroll manifest-scroll"><table><thead><tr>${columns.map((column,columnIndex) => `<th class="sortable" data-manifest-sort="${columnIndex}" data-label="${escapeHtml(column)}">${escapeHtml(column)} ↕</th>`).join('')}</tr><tr class="column-filters">${columns.map((column,columnIndex) => `<th><input data-column-filter="${columnIndex}" placeholder="Filter…" aria-label="Filter ${escapeHtml(column)}"></th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
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
  instrumentColumn: 'Optional metadata column containing instrument names. Leave blank when unavailable; for NIST-style data it is commonly InstrumentType.', validationSmilesRatio: 'Target validation SMILES count relative to unique training SMILES. Default: 0.1.',
  tanimotoNumBins: 'Number of similarity intervals used for balanced validation sampling. Default: 10.', tanimotoRadius: 'Morgan fingerprint radius used for validation similarity. Default: 2.',
  tanimotoNBits: 'Morgan fingerprint bit count. Default: 2048.', validationSamplingSeed: 'Random seed for deterministic validation sampling. Default: 0.',
  validationStructuresInputDir: 'Optional existing validation structure directory; mutually exclusive with validation input.',
  numWorkers: 'Number of parallel subprocess workers. Use 1 to disable parallel processing.', chunkSize: 'Number of SMILES groups assigned to each parallel chunk.',
  structureRebuildPolicy: 'When reusing an existing structure directory, decide whether newly added cleavage patterns require rebuilding: match any saved fragment (recommended), match only the root molecule, or always rebuild. An existing output is still skipped unless Overwrite structures is enabled.',
  requirePrecursorPathTargets: 'When enabled, only peak pathways that explicitly pass through a precursor node are saved as supervised assignments.',
  overwrite: 'Overwrite existing .preft.pt structure files.', overwritePreprocessingConfig: 'Overwrite an existing preprocessing configuration without prompting.',
  saveTrainValidRecords: 'Save successfully processed training records as an MSDataset.', saveValidationValidRecords: 'Save successfully processed validation records as an MSDataset.',
  keepParallelTemp: 'Keep temporary parallel-processing files after merging.'
  ,trainDir: 'Directory containing generated training structures and preprocessing_config.pftprep.json.',
  valDir: 'Directory containing generated validation structures with compatible preprocessing settings.',
  molEncoderCheckpoint: 'Pretrained molecular encoder checkpoint used by FragmentTreeFeatureModel.',
  conditionAdductEmbeddingDim: 'Width of the learned main-adduct embedding.',
  conditionCeFeatureDim: 'Fixed feature width produced from collision energy.',
  conditionCeFcDims: 'Comma-separated hidden widths in the collision-energy MLP.',
  conditionFeatureDim: 'Output width of the complete spectrum-condition encoder.',
  conditionFcDims: 'Comma-separated hidden widths used to fuse adduct and collision-energy features.',
  treeHiddenDim: 'Node representation width in the fragment-tree transformer.',
  treeNumLayers: 'Number of fragment-tree transformer layers.',
  treeNumHeads: 'Attention heads used by every fragment-tree transformer layer.',
  treeMaxDegree: 'Largest node degree represented by the tree positional encoding.',
  dropout: 'Dropout shared by the constructed fragment-tree model.',
  edgeFeatureDim: 'Width of each structural cleavage-edge representation.',
  edgeCategoryDim: 'Embedding width for cleavage pattern, reaction, and product IDs.',
  edgeAttentionHeads: 'Attention heads used while an edge attends to nearby atoms.',
  attentionMaxGraphDistance: 'Maximum atom-graph distance visible from a cleavage center.',
  maxEdgesPerDepth: 'Comma-separated inference budgets for successive cleavage depths.',
  trainingEdgesPerSample: 'Target/path and background candidates proposed per sample before the global step cap.',
  trainingZeroEdgeFraction: 'Fraction of the per-sample proposal budget reserved for unassigned/background edges.',
  assignmentScoreThreshold: 'Minimum assignment score accepted for training and the filtered validation view. Reads assignment_scores.tsv from each split. Default: 0.8. Validation also evaluates all samples by combining disjoint above- and below-threshold results without repeating inference.',
  maxSamples: 'Maximum spectra packed into one loaded compound batch.',
  maxEdgesPerStep: 'Hard upper bound on edges receiving expensive atom attention in one optimization step.',
  maxRetainedEdges: 'Highest-scoring edges retained at each progressive inference stage.',
  maxEdgesPerTree: 'Per stored tree (shared across every sample that references it), the highest cross-sample-importance edges kept for expensive attention encoding. Applies identically during training and inference; target/positive edges are always favored so the budget cannot silently drop them.',
  maxNextCleavageCandidates: 'Nodes allowed to produce the next cleavage depth during inference.',
  edgeConditionInteractionDim: 'Projection width for the edge × spectrum-condition score.',
  rankingLossWeight: 'Multiplier applied to the absolute edge-ranking loss.',
  topN: 'Total comparison partners per anchor group/edge, capping tiers 1-3 combined.',
  nearestLowerPartners: 'Tier 1: nearest lower-intensity partners always compared.',
  extendedLowerPartners: 'Tier 2: additional farther lower-intensity partners compared after tier 1.',
  backgroundPartners: 'Tier 3: unassigned/background edges compared after tiers 1-2.',
  rankingIntensityThreshold: 'Minimum intensity difference required for an ordered comparison (gates tiers 1-2 only).',
  experimentName: 'Checkpoint experiment directory name.', ckptId: 'Optional checkpoint identifier to resume.',
  batchSize: 'Number of prepared compound batches per optimizer step.', device: 'PyTorch execution device.',
  epochs: 'Number of complete training epochs.', validationIntervalSteps: 'Run validation after this many optimizer steps; use 0 to disable step validation.',
  stepValidationFraction: 'Fraction of validation compounds (SMILES) used at step intervals: 1 = all, 0.1 = 10%. Uses a fixed subset, rounded up to at least one compound. Epoch-end validation always evaluates all compounds.',
  trainLogIntervalSteps: 'Write training metrics after this many steps.', saveIntervalEpochs: 'Save a checkpoint after this many epochs.',
  saveIntervalSteps: 'Save a checkpoint after this many steps; use 0 to disable.', earlyStoppingPatience: 'Stop after this many non-improving validations; blank disables it.',
  optimizer: 'PyTorch optimizer class name.', lr: 'Optimizer learning rate.', weightDecay: 'Optimizer weight decay.', gradClipNorm: 'Maximum gradient norm; non-positive disables clipping.',
  validateAtStart: 'Run validation before the first training update.', detectAnomaly: 'Enable PyTorch autograd anomaly detection.',
  profilePerformance: 'Write detailed runtime performance profiles.', shuffle: 'Shuffle training structure batches each epoch.'
  ,modelPath: 'Trained fragment-tree model checkpoint (model.pt or a raw state_dict) used for spectrum prediction.'
};

function workbenchHtml(config, fragmenterText, trainingConfig) {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}${formCss()}${trainingCss()}.selection-lasso{stroke:#fff}</style></head><body><main>
  <header><div><span class="eyebrow">CLEFTS PLATFORM</span><h1>Workbench</h1><p id="appSubtitle" class="muted">Fragment Tree Data Preparation</p></div><div class="actions"><button id="openEvaluation" class="primary">Evaluation</button><div id="dataActions" class="actions"><button id="openResult">Open Result</button><button id="load">Load Configuration</button><button id="save">Save Configuration</button></div></div></header>
  <nav><button class="tab" data-app="smarts">SMARTS Search</button><button class="tab" data-app="cleavage">Cleavage Pattern Set</button><button class="tab active" data-app="data">Data Preparation</button><button class="tab" data-app="training">Training</button><button class="tab" data-app="metrics">Metrics</button><button class="tab" data-app="finetune">Fine-tuning</button><button class="tab" data-app="predict">Predict Spectrum</button></nav>
  ${trainingMetrics.html()}
  ${fineTune.html()}
  ${smartsSearch.html()}
  <div id="cleavageApp" hidden><section><div class="section-title"><div><h2>Cleavage Pattern Set Configuration</h2><p id="cleavagePath" class="muted">Not saved</p></div><div class="actions"><button type="button" id="applyFragmenterCleavage" hidden>Apply to Fragmenter</button><button type="button" id="loadCleavage">Load Configuration</button><button type="button" id="saveCleavage">Save Configuration</button></div></div><label>Pattern set name<input id="cleavageSetName" placeholder="single_bond_cleavage_pattern_set"></label></section><section id="visualBuilder" hidden><div class="section-title"><div><h2>Visual Pattern Builder</h2><p class="muted">Click or right-click items individually, or hold either mouse button and draw a loop around atoms and bonds.</p></div><div class="actions"><button type="button" id="clearSelection">Clear all</button><button type="button" id="closeBuilder">Close</button></div></div><div class="row"><label class="grow">Source SMILES<input id="builderSmiles" placeholder="O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"></label><button type="button" id="drawMolecule" class="primary">Draw Structure</button></div><div id="moleculeCanvas" class="molecule-canvas"></div><div class="builder-columns"><div><h3>Atom queries</h3><div id="atomConstraints" class="query-list"></div></div><div><h3>Bond queries</h3><div id="bondConstraints" class="query-list"></div></div></div><label>Pattern name<input id="builderPatternName" placeholder="flavonoid_substructure"></label><div class="actions"><button type="button" id="applyReactant" class="primary">Apply Reactant</button></div><div id="productBuilder" hidden><h3>Product Structure</h3><p class="muted">Draw an independent product skeleton from SMILES or SMARTS. Atoms are shown as mapping numbers; select an atom to change its number, or select a bond to change its type.</p><div class="row"><label class="grow">Product<select id="builderProductSelect"><option value="-1">New product</option></select></label><button type="button" id="newVisualProduct">New Product</button></div><div class="row"><label>Input type<select id="builderProductSourceType"><option value="smiles">SMILES</option><option value="smarts">SMARTS</option></select></label><label class="grow">Product SMILES / SMARTS<input id="builderProductSource" placeholder="O=c1ccccc1"></label><button type="button" id="drawProduct" class="primary">Draw Product</button></div><label class="check"><input type="checkbox" id="allowProductAtomTypes">Allow atom type changes</label><div id="productCanvas" class="molecule-canvas product-canvas"></div><div id="productAtomTools" class="bond-tools"><span>Click an atom to edit its mapping number.</span></div><div id="productBondTools" class="bond-tools"><span>Click a bond to edit its type.</span></div><label>Product name<input id="builderProductName" placeholder="fragment_product"></label><button type="button" id="applyProduct" class="primary">Add Product</button></div></section><div id="cleavagePatterns"></div><div class="actions"><button type="button" id="addCleavagePattern" class="primary">Add Pattern</button><button type="button" id="addVisualPattern">Add Pattern Visually</button><button type="button" id="loadSinglePattern">Load Pattern</button></div></div>
  <form id="form" data-app-panel="data">
    <section><h2>Input and Output</h2>${pathField('trainInput','Training MSDataset *','file')}${pathField('validationInput','Validation MSDataset','file')}${pathField('outputDir','Output directory *','folder')}</section>
    <section><div class="section-title"><div><h2>Fragmenter Parameters</h2><p class="muted">Edit Fragmenter settings. These values are included in the configuration and passed directly to the CLI.</p></div><div class="actions"><button type="button" id="loadFragmenter">Load Fragmenter</button><button type="button" id="saveFragmenter">Save Fragmenter</button></div></div><div id="fragmenterEditor"></div></section>
    <section><h2>Fragment tree</h2><div class="grid">${field('symbols','Symbols (space separated)','text')}${field('maxNode','Max nodes','number')}${field('maxEdge','Max edges','number')}${field('smilesColumn','SMILES column','text')}${field('precursorMzColumn','Precursor m/z column','text')}${field('adductTypeColumn','Adduct column','text')}${field('collisionEnergyColumn','Collision energy column','text')}${field('instrumentColumn','Instrument column','text')}</div><div class="checks">${check('requirePrecursorPathTargets','Require precursor in target path')}</div></section>
    <section><h2>Validation sampling</h2><div class="grid">${field('validationSmilesRatio','SMILES ratio','number','any')}${field('tanimotoNumBins','Tanimoto bins','number')}${field('tanimotoRadius','Morgan radius','number')}${field('tanimotoNBits','Morgan bits','number')}${field('validationSamplingSeed','Random seed','number')}</div>${pathField('validationStructuresInputDir','Existing validation structures','folder')}</section>
    <section><h2>Performance & output</h2><div class="grid">${field('numWorkers','Workers','number')}${field('chunkSize','Chunk size','number')}<label data-help="${HELP.structureRebuildPolicy}">Rebuild policy<select name="structureRebuildPolicy"><option value="all-fragments">all-fragments (recommended)</option><option value="root">root only</option><option value="always">always</option></select></label></div><div class="checks">${check('overwrite','Overwrite structures')}${check('overwritePreprocessingConfig','Overwrite preprocessing config')}${check('saveTrainValidRecords','Save valid training records')}${check('saveValidationValidRecords','Save valid validation records')}${check('keepParallelTemp','Keep parallel temp')}</div></section>
    <footer><div><div id="status" class="status idle">Ready</div><code id="command"></code></div><div class="actions"><button type="button" id="copyCommand">Copy Command</button><button type="button" id="stop" disabled>Stop</button><button type="submit" class="primary">Run</button></div></footer>
  </form>
  <form id="trainingForm" data-app-panel="training" hidden>
    <section class="model-map"><div class="section-title"><div><h2>FragmentTreeTrainingModel architecture</h2><p class="muted">Tensor flow for both learning and progressive prediction. Select any model node to edit the parameters consumed there.</p></div><div class="actions"><button type="button" id="loadTraining">Load Configuration</button><button type="button" id="saveTraining">Save Configuration</button></div></div><div class="architecture"><div class="architecture-input"><button type="button" data-open-model="io"><b>TrainingFragmentTreeStructure</b><code>[node, edge, sample, target, path tensors]</code><span>Prepared tensors + frozen pretrained MolEncoder</span></button></div><div class="architecture-branches"><button type="button" data-open-model="condition"><b>ConditionEncoder</b><code>adduct + collision energy → condition_h</code><span>Adduct embedding and condition MLP</span></button><button type="button" data-open-model="tree"><b>FragmentTreeEncoder</b><code>molecular node graph → node_h</code><span>Tree-aware transformer representation</span></button><button type="button" data-open-model="edge"><b>StructuralEdgeEncoder</b><code>cleavage event + local atoms → edge_h</code><span>Bounded expensive atom attention</span></button></div><div class="architecture-merge"><button type="button" data-open-model="edge"><b>ConditionEdgeScorer + CandidateSelector heads</b><code>edge_h × condition_h + node_h → absolute / retain / cleave / state logits</code><span>Shared learned scores used by both paths below</span></button></div><div class="architecture-modes"><div class="mode training-mode"><strong>TRAINING PATH</strong><button type="button" data-open-model="ranking"><b>Pair and target losses</b><code>intensity-weighted comparisons + path supervision</code><span>ranking + selection + formula intensity → total loss</span></button><button type="button" data-open-model="run"><b>Backward / Optimizer</b><code>total loss → gradients → checkpoint graph</code><span>Scheduled train logs and validation</span></button></div><div class="mode prediction-mode"><strong>PREDICTION PATH</strong><button type="button" data-open-model="edge"><b>Progressive edge selection</b><code>depth 1 → top edges → next cleavage nodes</code><span>Repeat scoring with retained survivors</span></button><button type="button" data-open-model="edge"><b>Spectrum candidates</b><code>absolute scores → formula candidates → intensities</code><span>Inference budgets control compute cost</span></button></div></div></div><p id="modelHint" class="model-hint">Select a model component to locate the parameters consumed by it.</p></section>
    <section data-model-block="io"><div><h2>Training data and output</h2><p class="muted">The Workbench passes these paths to the existing training CLI.</p></div>${pathField('trainDir','Training structures directory *','folder','training')}${pathField('valDir','Validation structures directory *','folder','training')}${pathField('outputDir','Model output directory *','folder','training')}${pathField('molEncoderCheckpoint','Molecular encoder checkpoint *','file','training')}</section>
    <section data-model-block="condition"><h2>Condition encoder</h2><div class="grid">${field('conditionAdductEmbeddingDim','Adduct embedding dim','number')}${field('conditionCeFeatureDim','CE feature dim','number')}${field('conditionCeFcDims','CE FC dims (CSV)','text')}${field('conditionFeatureDim','Condition feature dim','number')}${field('conditionFcDims','Condition FC dims (CSV)','text')}${field('dropout','Shared dropout','number','any')}</div></section>
    <section data-model-block="tree"><h2>Fragment tree encoder</h2><div class="grid">${field('treeHiddenDim','Hidden dimension','number')}${field('treeNumLayers','Transformer layers','number')}${field('treeNumHeads','Attention heads','number')}${field('treeMaxDegree','Maximum node degree','number')}</div></section>
    <section data-model-block="edge"><h2>Edge scorer and bounded candidate selection</h2><div class="model-notes"><p><b>Structural score:</b> embeds cleavage pattern/reaction/product IDs and the source/target molecular representations. Selected edges then attend only to atoms within the graph-distance limit.</p><p><b>Training budget:</b> intensity-weighted target groups and their path edges are proposed first; background edges fill the remaining per-sample budget. The union is capped by <i>Expensive edges / step</i>.</p><p><b>Prediction budget:</b> <i>Inference edges / depth</i> controls new edges at each cleavage depth, <i>Inference retained edges</i> carries survivors forward, and <i>Next-cleavage nodes</i> controls expansion.</p><p><b>Per-tree budget:</b> <i>Expensive edges / tree</i> is a separate, larger-scope cap shared across every sample that references the same stored tree (not per-sample like the budgets above). Edges are ranked by cross-sample importance (condition-scored, with target/positive edges always favored) and the same cap applies identically during training and inference, replacing the old training-only whole-batch cap and the old "attend every edge" inference default.</p></div><div class="grid">${field('edgeFeatureDim','Edge feature dim','number')}${field('edgeCategoryDim','Category embedding dim','number')}${field('edgeAttentionHeads','Attention heads','number')}${field('attentionMaxGraphDistance','Atom graph distance','number')}${field('maxEdgesPerDepth','Inference edges / depth (CSV)','text')}${field('trainingEdgesPerSample','Candidate edges / sample','number')}${field('trainingZeroEdgeFraction','Background fraction','number','any')}${field('maxSamples','Samples / batch structure','number')}${field('maxEdgesPerStep','Expensive edges / step','number')}${field('maxRetainedEdges','Inference retained edges','number')}${field('maxEdgesPerTree','Expensive edges / tree','number')}${field('maxNextCleavageCandidates','Next-cleavage nodes','number')}${field('edgeConditionInteractionDim','Condition interaction dim','number')}</div></section>
    <section data-model-block="ranking"><h2>Losses</h2><div class="model-losses"><span>Absolute edge ranking</span><span>+</span><span>Candidate selection</span><span>+</span><span>Formula intensity</span></div><div class="model-notes"><p><b>Absolute edge ranking:</b> for each intensity-sorted anchor group, up to <i>Comparisons / anchor</i> partners are compared in three tiers, filled in order: tier 1 (<i>Tier 1</i> nearest lower-intensity partners), tier 2 (<i>Tier 2</i> more, farther lower-intensity partners), tier 3 (<i>Tier 3</i> unassigned/background edges). <i>Intensity difference threshold</i> gates tiers 1-2 only. Each comparison is weighted by reciprocal rank within the sample (the most intense group counts most), not by raw intensity.</p><p><b>Candidate selection:</b> supervises terminal fragment nodes, ion/unsaturation/radical states, every edge along a target path, and intermediate nodes that must be expanded. Selected-peak intensity coverage measures how much observed intensity remains reachable.</p><p><b>Formula intensity:</b> learns presence and relative abundance after equal-formula candidates are merged. Cosine similarity reports agreement between predicted and observed intensity vectors.</p></div><div class="grid">${field('rankingLossWeight','Ranking loss weight','number','any')}${field('topN','Comparisons / anchor (total)','number')}${field('nearestLowerPartners','Tier 1 (nearest lower)','number')}${field('extendedLowerPartners','Tier 2 (farther lower)','number')}${field('backgroundPartners','Tier 3 (background)','number')}${field('rankingIntensityThreshold','Intensity difference threshold','number','any')}</div></section>
    <section data-model-block="run"><h2>Training, optimizer, and checkpoints</h2><div class="model-notes"><p><b>Assignment-score filtering:</b> training and filtered validation use spectra at or above the threshold. Unfiltered validation combines disjoint evaluated partitions, so accepted spectra are not evaluated twice.</p></div><div class="grid">${field('experimentName','Experiment name','text')}<label data-help="${HELP.ckptId}">Resume checkpoint ID<div class="path"><input name="ckptId"><button type="button" id="checkpointGraph">Graph…</button></div><small class="field-help">${HELP.ckptId}</small></label>${field('assignmentScoreThreshold','Assignment score threshold','number','any')}${field('batchSize','Batch size','number')}<label>Device<select name="device"><option value="cpu">cpu</option><option value="cuda">cuda</option><option value="mps">mps</option></select><small class="field-help">${HELP.device}</small></label>${field('epochs','Epochs','number')}${field('numWorkers','Data-loader workers','number')}${field('validationIntervalSteps','Validation interval steps','number')}${field('stepValidationFraction','Step validation fraction','number','any')}${field('trainLogIntervalSteps','Log interval steps','number')}${field('saveIntervalEpochs','Save interval epochs','number')}${field('saveIntervalSteps','Save interval steps','number')}${field('earlyStoppingPatience','Early-stopping patience','number')}<label>Optimizer<select name="optimizer"><option value="AdamW">AdamW</option><option value="Adam">Adam</option><option value="SGD">SGD</option></select><small class="field-help">${HELP.optimizer}</small></label>${field('lr','Learning rate','number','any')}${field('weightDecay','Weight decay','number','any')}${field('gradClipNorm','Gradient clipping norm','number','any')}</div><div class="checks">${check('validateAtStart','Validate at start')}${check('detectAnomaly','Detect anomaly')}${check('profilePerformance','Profile performance')}${check('shuffle','Shuffle')}</div></section>
    <footer><div><div id="trainingStatus" class="status idle">Ready</div><code id="trainingCommand"></code></div><div class="actions"><button type="button" id="trainingCopyCommand">Copy Command</button><button type="button" id="trainingStop" disabled>Stop</button><button type="submit" class="primary">Run Training</button></div></footer>
  </form>
  <form id="predictForm" data-app-panel="predict" hidden>
    <section><h2>Model</h2><p class="muted">Points at a checkpoint produced by fragment-tree training (a full <code>model.pt</code>, or a raw generator state_dict). Apply it to load its trained main adduct types before specifying a molecule.</p>${pathField('modelPath','Model checkpoint *','file','predict')}<label>Device<select name="device"><option value="cpu">cpu</option><option value="cuda">cuda</option><option value="mps">mps</option></select></label><div class="actions"><button type="button" id="predictApplyModel" class="primary">Apply Model</button></div><div id="predictModelStatus" class="status idle">No model applied yet.</div></section>
    ${spectrumPrediction.html()}
    <section><h2>Molecule and conditions</h2><label>SMILES *<input name="smiles" placeholder="CC(=O)Oc1ccccc1C(=O)O"></label><div class="grid">${field('ce','Collision energy (eV) *','text')}<label>Adduct type *<select name="adductType"><option value="">Apply a model first…</option></select></label></div><div class="actions"><button type="button" id="predictPreview">Preview Molecule</button></div><div id="predictMoleculePreview" class="molecule-preview" hidden></div></section>
    <section id="predictResultSection" hidden><h2>Predicted Spectrum</h2><div id="predictResult"></div></section>
    <footer><div><div id="predictStatus" class="status idle">Ready</div></div><div class="actions"><button type="submit" class="primary" id="predictSubmit">Predict Spectrum</button></div></footer>
  </form><div id="helpTooltip" role="tooltip"></div><script>const vscode=acquireVsCodeApi(); const initial=${safeJson(config)}; const initialTraining=${safeJson(trainingConfig)}; const initialFragmenter=${safeJson(fragmenterText)}; ${webviewScript()}${spectrumPrediction.script()}</script></main></body></html>`;
}
function pathField(name,label,kind,form='data') { return `<label data-help="${HELP[name] || ''}">${label}<div class="path"><input name="${name}"><button type="button" data-pick="${name}" data-kind="${kind}" data-form="${form}">Browse</button></div>${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</label>`; }
function field(name,label,type,step='1') { return `<label data-help="${HELP[name] || ''}">${label}<input name="${name}" type="${type}"${type==='number'?` step="${step}"`:''}>${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</label>`; }
function check(name,label) { return `<label class="check" data-help="${HELP[name] || ''}"><input name="${name}" type="checkbox"><span>${label}${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</span></label>`; }
function safeJson(value) { return JSON.stringify(value).replace(/</g, '\\u003c'); }
function webviewScript() { return `
    const createFragmenterEditor = ${createFragmenterEditor.toString()};
    const form=document.getElementById('form'), trainingForm=document.getElementById('trainingForm'), predictForm=document.getElementById('predictForm'), statusEl=document.getElementById('status'), stop=document.getElementById('stop'),trainingStatusEl=document.getElementById('trainingStatus'),trainingStop=document.getElementById('trainingStop');
    let cleavagePath='',cleavageModel={cleavage_pattern_set:{name:'',patterns:[]}};
    ${trainingMetrics.script()}
    ${fineTune.script()}
    ${smartsSearch.script()}
    const htmlEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function setFormConfig(target,c){for(const [k,v] of Object.entries(c)){const el=target.elements[k];if(!el)continue;if(el.type==='checkbox')el.checked=!!v;else el.value=Array.isArray(v)?v.join(target===trainingForm?',':' '):v??'';}}
    function readForm(target,application){const c={application};for(const el of target.elements){if(!el.name)continue;if(el.type==='checkbox')c[el.name]=el.checked;else if(el.type==='number')c[el.name]=el.value===''?'':Number(el.value);else c[el.name]=el.value;}return c;}
    let extraConfig={};
    function setConfig(c){extraConfig={...c};setFormConfig(form,c);if(c.fragmenterParams)fragmenterEditor.setValue(c.fragmenterParams)}
    function getConfig(){const c={...extraConfig,...readForm(form,'fragment-tree-data-preparation')};delete c.params;c.fragmenterParams=fragmenterEditor.getValue();c.symbols=String(c.symbols).split(/[ ,]+/).filter(Boolean);return c;}
    function getTrainingConfig(){return readForm(trainingForm,'fragment-tree-training')}
    let applyFragmenterCleavage=null;
    const fragmenterEditor=createFragmenterEditor(document.getElementById('fragmenterEditor'),{value:JSON.parse(initialFragmenter),editCleavagePatternSet:(value,apply)=>{cleavageModel={cleavage_pattern_set:value};cleavagePath='';applyFragmenterCleavage=apply;document.getElementById('cleavagePath').textContent='Editing Fragmenter Cleavage Pattern Set';renderCleavage();document.getElementById('applyFragmenterCleavage').hidden=false;document.querySelector('[data-app="cleavage"]').click();}});
    const tooltip=document.getElementById('helpTooltip'); let tooltipTimer;
    document.querySelectorAll('[data-help]').forEach(el=>{el.addEventListener('mouseenter',()=>{tooltipTimer=setTimeout(()=>{const r=el.getBoundingClientRect();tooltip.textContent=el.dataset.help;tooltip.style.left=Math.min(r.left,window.innerWidth-390)+'px';tooltip.style.top=(r.bottom+7)+'px';tooltip.classList.add('visible');},500);});el.addEventListener('mouseleave',()=>{clearTimeout(tooltipTimer);tooltip.classList.remove('visible');});});
    setConfig(initial);setFormConfig(trainingForm,initialTraining);setPredictEnabled(false); document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>vscode.postMessage({type:'pick',form:b.dataset.form||'data',field:b.dataset.pick,kind:b.dataset.kind}));
    document.querySelectorAll('[data-open-model]').forEach(button=>button.onclick=()=>{const key=button.dataset.openModel,target=trainingForm.querySelector('[data-model-block="'+key+'"]');document.querySelectorAll('[data-open-model]').forEach(x=>x.classList.toggle('selected',x===button));document.querySelectorAll('[data-model-block]').forEach(x=>x.classList.toggle('selected',x===target));document.getElementById('modelHint').textContent=button.querySelector('b').textContent+': '+button.querySelector('span').textContent;target.scrollIntoView({behavior:'smooth',block:'start'});});
    document.querySelectorAll('[data-app]').forEach(button=>button.onclick=()=>{document.querySelectorAll('[data-app]').forEach(x=>x.classList.toggle('active',x===button));const app=button.dataset.app;document.getElementById('metricsApp').hidden=app!=='metrics';document.getElementById('fineTuneApp').hidden=app!=='finetune';document.getElementById('smartsApp').hidden=app!=='smarts';document.getElementById('cleavageApp').hidden=app!=='cleavage';form.hidden=app!=='data';trainingForm.hidden=app!=='training';predictForm.hidden=app!=='predict';document.getElementById('dataActions').hidden=app!=='data';document.getElementById('appSubtitle').textContent=app==='metrics'?'Training Metrics':app==='finetune'?'Fragment Tree Fine-tuning':app==='smarts'?'SMARTS Search':app==='cleavage'?'Cleavage Pattern Set Editor':app==='training'?'Fragment Tree Training':app==='predict'?'Predict Spectrum':'Fragment Tree Data Preparation';});
    document.getElementById('save').onclick=()=>vscode.postMessage({type:'saveConfig',config:getConfig()}); document.getElementById('load').onclick=()=>vscode.postMessage({type:'loadConfig'}); document.getElementById('openResult').onclick=()=>vscode.postMessage({type:'openResult'}); document.getElementById('openEvaluation').onclick=()=>vscode.postMessage({type:'openEvaluation'});
    function renderCleavage(){document.getElementById('cleavageSetName').value=cleavageModel.cleavage_pattern_set.name;document.getElementById('cleavagePatterns').innerHTML=cleavageModel.cleavage_pattern_set.patterns.map((p,pi)=>\`<section class="pattern-card"><div class="section-title"><h2>Pattern \${pi+1}</h2><div class="actions"><button type="button" data-edit-visual="\${pi}">Edit Visually</button><button type="button" data-load-pattern="\${pi}">Load</button><button type="button" data-save-pattern="\${pi}">Save</button><button type="button" class="danger" data-remove-pattern="\${pi}">Remove Pattern</button></div></div><div class="grid"><label>Pattern name<input data-pattern="\${pi}" data-key="name" value="\${htmlEscape(p.name)}" placeholder="single_bond_cleavage"></label><label>Reactant SMARTS<input data-pattern="\${pi}" data-key="reactant_smarts" value="\${htmlEscape(p.reactant_smarts)}" placeholder="[!#1:1]-[!#1:2]"></label></div><div class="section-title product-title"><h3>Products</h3><button type="button" data-add-product="\${pi}">Add Product</button></div><div class="product-list">\${p.products.map((product,xi)=>\`<div class="product-row"><label>Product name<input data-pattern="\${pi}" data-product="\${xi}" data-key="name" value="\${htmlEscape(product.name)}"></label><label>Product SMARTS<input data-pattern="\${pi}" data-product="\${xi}" data-key="smarts" value="\${htmlEscape(product.smarts)}" placeholder="[!#1:1]"></label><button type="button" class="danger" data-remove-product="\${pi}:\${xi}">Remove</button></div>\`).join('')}</div></section>\`).join('');}
    document.getElementById('cleavageSetName').oninput=e=>cleavageModel.cleavage_pattern_set.name=e.target.value;document.getElementById('addCleavagePattern').onclick=()=>{cleavageModel.cleavage_pattern_set.patterns.push({name:'',reactant_smarts:'',products:[]});renderCleavage()};
    document.getElementById('cleavagePatterns').addEventListener('input',e=>{const pi=Number(e.target.dataset.pattern);if(!Number.isInteger(pi))return;const xi=e.target.dataset.product;if(xi===undefined)cleavageModel.cleavage_pattern_set.patterns[pi][e.target.dataset.key]=e.target.value;else cleavageModel.cleavage_pattern_set.patterns[pi].products[Number(xi)][e.target.dataset.key]=e.target.value;});
    document.getElementById('cleavagePatterns').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.editVisual!==undefined){editVisualPattern(Number(b.dataset.editVisual));return}else if(b.dataset.addProduct!==undefined)cleavageModel.cleavage_pattern_set.patterns[Number(b.dataset.addProduct)].products.push({name:'',smarts:''});else if(b.dataset.removePattern!==undefined)cleavageModel.cleavage_pattern_set.patterns.splice(Number(b.dataset.removePattern),1);else if(b.dataset.removeProduct){const [pi,xi]=b.dataset.removeProduct.split(':').map(Number);cleavageModel.cleavage_pattern_set.patterns[pi].products.splice(xi,1)}else if(b.dataset.savePattern!==undefined){vscode.postMessage({type:'saveCleavagePattern',pattern:cleavageModel.cleavage_pattern_set.patterns[Number(b.dataset.savePattern)]});return}else if(b.dataset.loadPattern!==undefined){vscode.postMessage({type:'loadCleavagePattern',index:Number(b.dataset.loadPattern)});return}else return;renderCleavage()});
    document.getElementById('loadCleavage').onclick=()=>vscode.postMessage({type:'loadCleavagePatternSet'});document.getElementById('saveCleavage').onclick=()=>vscode.postMessage({type:'saveCleavagePatternSet',value:cleavageModel,path:cleavagePath});renderCleavage();
    document.getElementById('loadSinglePattern').onclick=()=>vscode.postMessage({type:'loadCleavagePattern',index:-1});
    let chemistrySequence=0,chemistryWaiters=new Map(),molGraph=null,selectedAtoms=new Set(),selectedBonds=new Set(),atomConstraintState={},bondConstraintState={},atomMapBySource={},visualPatternIndex=-1,visualProductIndex=-1,visualSourceType='smiles',productGraph=null,productAtomMaps={},productAtomOverrides={},productBondOverrides={},productDeletedAtoms=new Set(),productSelectedAtoms=new Set(),productAddedBonds=[],nextProductBondId=1,productBondStart=null,productSelectedAtom=null,productSelectedBond=null;
    const builderSourceInput=document.getElementById('builderSmiles'),builderSourceLabel=builderSourceInput.closest('label');builderSourceLabel.childNodes[0].nodeValue='Source SMILES / SMARTS';builderSourceLabel.insertAdjacentHTML('beforebegin','<label>Input type<select id="builderSourceType"><option value="smarts">SMARTS</option><option value="smiles">SMILES</option></select></label>');document.getElementById('builderSourceType').onchange=e=>visualSourceType=e.target.value;
    function chemistry(command,payload){return new Promise((resolve,reject)=>{const requestId=++chemistrySequence;chemistryWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'chemistry',requestId,command,payload})})}
    const atomColor=s=>({N:'#2478c5',O:'#e53935',S:'#d6a800',P:'#e67e22',F:'#28a745',Cl:'#28a745',Br:'#9b3b22',I:'#7b4ab5',H:'#57898d'}[s]||'currentColor');
    function layout(){const xs=molGraph.atoms.map(a=>a.x),ys=molGraph.atoms.map(a=>a.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);return {x:a=>55+(a.x-minX)/(maxX-minX||1)*590,y:a=>45+(a.y-minY)/(maxY-minY||1)*350}}
    function bondMarkup(b,l,selected,deleted=false,override='preserve'){const a=molGraph.atoms[b.begin],z=molGraph.atoms[b.end],x1=l.x(a),y1=l.y(a),x2=l.x(z),y2=l.y(z),dx=x2-x1,dy=y2-y1,n=Math.hypot(dx,dy)||1,ox=-dy/n*6,oy=dx/n*6,order=override==='preserve'?b.order:Number(override),kind=order===1.5?'aromatic':order===3?'triple':order===2?'double':'single',cls='bond-group bond-'+kind+' '+(selected?'selected ':'')+(deleted?'deleted':'');let lines='';if(order===2)lines=\`<line x1="\${x1+ox}" y1="\${y1+oy}" x2="\${x2+ox}" y2="\${y2+oy}"/><line x1="\${x1-ox}" y1="\${y1-oy}" x2="\${x2-ox}" y2="\${y2-oy}"/>\`;else if(order===1.5)lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/><line class="aromatic-mark" x1="\${x1+ox}" y1="\${y1+oy}" x2="\${x2+ox}" y2="\${y2+oy}"/>\`;else if(order===3)lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/><line x1="\${x1+ox*1.7}" y1="\${y1+oy*1.7}" x2="\${x2+ox*1.7}" y2="\${y2+oy*1.7}"/><line x1="\${x1-ox*1.7}" y1="\${y1-oy*1.7}" x2="\${x2-ox*1.7}" y2="\${y2-oy*1.7}"/>\`;else lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/>\`;return \`<g data-mol-bond="\${b.index}" class="\${cls}">\${lines}<line class="bond-hit" x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/></g>\`}
    function atomMarkup(a,l,selected,deleted=false){const neighbors=molGraph.bonds.filter(b=>b.begin===a.index||b.end===a.index).map(b=>molGraph.atoms[b.begin===a.index?b.end:b.begin]),vx=neighbors.reduce((sum,n)=>sum+l.x(n)-l.x(a),0),vy=neighbors.reduce((sum,n)=>sum+l.y(n)-l.y(a),0),length=Math.hypot(vx,vy),ix=length>1?-vx/length*25:20,iy=length>1?-vy/length*25:-20,mapNumber=atomMapBySource[a.index]||a.mapNumber||a.index+1;return \`<g data-mol-atom="\${a.index}" class="mol-atom \${selected?'selected ':''}\${deleted?'deleted':''}" transform="translate(\${l.x(a)} \${l.y(a)})"><circle class="atom-hit" r="25"/><circle class="atom-label-bg" r="16"/><text class="atom-symbol" style="fill:\${atomColor(a.symbol)}" text-anchor="middle" dominant-baseline="central">\${a.symbol}</text><circle class="atom-index-bg" cx="\${ix}" cy="\${iy}" r="9" style="fill:var(--vscode-editor-background);stroke:var(--vscode-focusBorder);stroke-width:1"/><text class="atom-index" x="\${ix}" y="\${iy}" text-anchor="middle" dominant-baseline="central">\${mapNumber}</text></g>\`}
    const sourceBondType=b=>b.order===1.5?'aromatic':b.order===3?'triple':b.order===2?'double':'single';
    function queryPanels(){document.getElementById('atomConstraints').innerHTML=[...selectedAtoms].sort((a,b)=>a-b).map(i=>{const state=atomConstraintState[i]||(atomConstraintState[i]={mode:'elements',elements:[molGraph.atoms[i].symbol]});return \`<div class="query-card"><div class="query-head"><b>Atom map \${atomMapBySource[i]||molGraph.atoms[i].mapNumber||i+1}</b><span>\${molGraph.atoms[i].symbol}</span></div><div class="query-modes"><button data-atom-mode="\${i}:elements" class="\${state.mode==='elements'?'active':''}">Elements</button><button data-atom-mode="\${i}:any-heavy" class="\${state.mode==='any-heavy'?'active':''}">Any non-H</button><button data-atom-mode="\${i}:custom" class="\${state.mode==='custom'?'active':''}">Custom SMARTS</button></div>\${state.mode==='custom'?'<label>Atom SMARTS<input data-custom-atom="'+i+'" value="'+htmlEscape(state.smarts||'')+'"></label>':'<button class="element-picker" data-open-elements="'+i+'">Open periodic table</button><div class="element-summary">'+(state.mode==='elements'?state.elements.join(', '):'Any non-hydrogen atom')+'</div>'}</div>\`}).join('')||'<p class="muted">Select atoms in the structure.</p>';document.getElementById('bondConstraints').innerHTML=[...selectedBonds].sort((a,b)=>a-b).map(i=>{const b=molGraph.bonds[i],state=bondConstraintState[i]||(bondConstraintState[i]={types:[sourceBondType(b)],ring:false});return \`<div class="query-card"><div class="query-head"><b>Bond \${i}</b><span>\${b.begin}–\${b.end}</span></div><div class="bond-type-choices">\${['any','single','double','triple','aromatic'].map(type=>\`<button data-query-bond-type="\${i}:\${type}" class="\${state.types.includes(type)?'active':''}">\${type==='any'?'Any bond':type[0].toUpperCase()+type.slice(1)}</button>\`).join('')}</div><label class="check"><input type="checkbox" data-ring-bond="\${i}" \${state.ring?'checked':''}>Must be in a ring</label></div>\`}).join('')||'<p class="muted">Select bonds in the structure.</p>'}
    function renderMolecule(){if(!molGraph)return;const l=layout();document.getElementById('moleculeCanvas').innerHTML=\`<svg viewBox="0 0 700 440">\${molGraph.bonds.map(b=>bondMarkup(b,l,selectedBonds.has(b.index))).join('')}\${molGraph.atoms.map(a=>atomMarkup(a,l,selectedAtoms.has(a.index))).join('')}<polyline class="selection-lasso" hidden/></svg>\`;queryPanels()}
    function productLayout(){const xs=productGraph.atoms.map(a=>a.x),ys=productGraph.atoms.map(a=>a.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);return {x:a=>55+(a.x-minX)/(maxX-minX||1)*590,y:a=>45+(a.y-minY)/(maxY-minY||1)*350}}
    function productBondMarkup(b,l,token,override='preserve'){const a=productGraph.atoms[b.begin],z=productGraph.atoms[b.end],x1=l.x(a),y1=l.y(a),x2=l.x(z),y2=l.y(z),dx=x2-x1,dy=y2-y1,n=Math.hypot(dx,dy)||1,ox=-dy/n*6,oy=dx/n*6,order=override==='preserve'?b.order:Number(override),kind=order===1.5?'aromatic':order===3?'triple':order===2?'double':'single',deleted=override==='remove'||productDeletedAtoms.has(b.begin)||productDeletedAtoms.has(b.end),cls='bond-group bond-'+kind+' '+(productSelectedBond===token?'selected ':'')+(deleted?'deleted':'');let lines='';if(order===2)lines=\`<line x1="\${x1+ox}" y1="\${y1+oy}" x2="\${x2+ox}" y2="\${y2+oy}"/><line x1="\${x1-ox}" y1="\${y1-oy}" x2="\${x2-ox}" y2="\${y2-oy}"/>\`;else if(order===1.5)lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/><line class="aromatic-mark" x1="\${x1+ox}" y1="\${y1+oy}" x2="\${x2+ox}" y2="\${y2+oy}"/>\`;else if(order===3)lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/><line x1="\${x1+ox*1.7}" y1="\${y1+oy*1.7}" x2="\${x2+ox*1.7}" y2="\${y2+oy*1.7}"/><line x1="\${x1-ox*1.7}" y1="\${y1-oy*1.7}" x2="\${x2-ox*1.7}" y2="\${y2-oy*1.7}"/>\`;else lines=\`<line x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/>\`;return \`<g data-product-bond="\${token}" class="\${cls}">\${lines}<line class="bond-hit" x1="\${x1}" y1="\${y1}" x2="\${x2}" y2="\${y2}"/></g>\`}
    function productAtomMarkup(a,l){const selected=productSelectedAtom===a.index||productBondStart===a.index||productSelectedAtoms.has(a.index);return \`<g data-mol-atom="\${a.index}" class="mol-atom \${selected?'selected ':''}\${productDeletedAtoms.has(a.index)?'deleted':''}" transform="translate(\${l.x(a)} \${l.y(a)})"><circle class="atom-hit" r="25"/><circle class="atom-label-bg" r="17"/><text class="atom-symbol product-map-number" text-anchor="middle" dominant-baseline="central">\${htmlEscape(productAtomMaps[a.index])}</text></g>\`}
    function selectedProductBondValue(){if(!productSelectedBond)return'preserve';if(productSelectedBond.startsWith('base:'))return productBondOverrides[Number(productSelectedBond.slice(5))]||'preserve';const added=productAddedBonds.find(b=>b.id===Number(productSelectedBond.slice(6)));return added?String(added.order):'1'}
    function renderProduct(){const canvas=document.getElementById('productCanvas'),atomTools=document.getElementById('productAtomTools'),bondTools=document.getElementById('productBondTools');if(!productGraph){canvas.innerHTML='<p class="muted">Enter Product SMILES or SMARTS and click Draw Product.</p>';atomTools.innerHTML='<span>Click an atom to edit its mapping number.</span>';bondTools.innerHTML='<span>Click a bond to edit its type.</span>';return}const l=productLayout(),base=productGraph.bonds.map(b=>productBondMarkup(b,l,'base:'+b.index,productBondOverrides[b.index]||'preserve')).join(''),added=productAddedBonds.map(b=>productBondMarkup({...b,index:b.id},l,'added:'+b.id,String(b.order))).join('');canvas.innerHTML=\`<svg viewBox="0 0 700 440">\${base}\${added}\${productGraph.atoms.map(a=>productAtomMarkup(a,l)).join('')}<polyline class="selection-lasso" hidden/></svg>\`;if(productSelectedAtoms.size)atomTools.innerHTML='<b>'+productSelectedAtoms.size+' atoms selected</b><button data-delete-selected-product-atoms>Delete Selected</button><button data-restore-selected-product-atoms>Restore Selected</button><button data-clear-selected-product-atoms>Clear Selection</button>';else if(productSelectedAtom===null)atomTools.innerHTML=productBondStart===null?'<span>Click an atom to edit, or draw a loop around atoms to select them together.</span>':'<b>Choose the destination atom for the new bond.</b>';else{const atom=productGraph.atoms[productSelectedAtom],query=productAtomOverrides[productSelectedAtom]??String(atom.smarts||atom.symbol).replace(/:\\d+(?=\\])/,'');atomTools.innerHTML='<b>Atom '+productSelectedAtom+'</b><label>Reactant map number <input data-product-map type="number" min="1" step="1" value="'+htmlEscape(productAtomMaps[productSelectedAtom])+'"></label><label>Atom SMARTS <input data-product-atom-type value="'+htmlEscape(query)+'" '+(document.getElementById('allowProductAtomTypes').checked?'':'disabled')+'></label><button data-toggle-product-atom>'+(productDeletedAtoms.has(productSelectedAtom)?'Restore Atom':'Delete Atom')+'</button> <button data-start-product-bond '+(productDeletedAtoms.has(productSelectedAtom)?'disabled':'')+'>Start New Bond</button>'}if(productSelectedBond===null)bondTools.innerHTML='<span>Click a bond to edit its type.</span>';else{const value=selectedProductBondValue(),addedBond=productSelectedBond.startsWith('added:');bondTools.innerHTML=\`<b>\${addedBond?'New bond':'Bond'} \${htmlEscape(productSelectedBond.split(':')[1])}</b>\${[[addedBond?'1':'preserve',addedBond?'Single':'Keep'],['1','Single'],['2','Double'],['3','Triple'],['1.5','Aromatic'],['remove','Delete']].filter((x,i,a)=>a.findIndex(y=>y[0]===x[0])===i).map(x=>\`<button data-product-type="\${x[0]}" class="\${value===x[0]?'active':''}">\${x[1]}</button>\`).join('')}\`}}
    function sourcePayload(){const value=document.getElementById('builderSmiles').value,picker=document.getElementById('builderSourceType'),type=visualPatternIndex>=0?visualSourceType:picker.value;return type==='smarts'?{sourceType:'smarts',smarts:value}:{sourceType:'smiles',smiles:value}}
    function productSourcePayload(){const value=document.getElementById('builderProductSource').value,type=document.getElementById('builderProductSourceType').value;return type==='smarts'?{sourceType:'smarts',smarts:value}:{sourceType:'smiles',smiles:value}}
    async function drawProductStructure(){productGraph=await chemistry('molecule',productSourcePayload());productAtomMaps={};productAtomOverrides={};productBondOverrides={};productDeletedAtoms=new Set();productSelectedAtoms=new Set();productAddedBonds=[];nextProductBondId=1;productBondStart=null;productSelectedAtom=null;productSelectedBond=null;productGraph.atoms.forEach((atom,index)=>productAtomMaps[atom.index]=atom.mapNumber||index+1);renderProduct()}
    function resetVisualBuilder(){molGraph=null;selectedAtoms=new Set();selectedBonds=new Set();atomConstraintState={};bondConstraintState={};atomMapBySource={};visualPatternIndex=-1;visualProductIndex=-1;visualSourceType='smiles';productGraph=null;productAtomMaps={};productAtomOverrides={};productBondOverrides={};productDeletedAtoms=new Set();productSelectedAtoms=new Set();productAddedBonds=[];nextProductBondId=1;productBondStart=null;productSelectedAtom=null;productSelectedBond=null;document.getElementById('builderSmiles').value='';document.getElementById('builderPatternName').value='';document.getElementById('builderProductName').value='';document.getElementById('builderProductSource').value='';document.getElementById('builderProductSourceType').value='smiles';document.getElementById('allowProductAtomTypes').checked=false;document.getElementById('builderProductSelect').innerHTML='<option value="-1">New product</option>';document.getElementById('moleculeCanvas').innerHTML='';document.getElementById('atomConstraints').innerHTML='';document.getElementById('bondConstraints').innerHTML='';document.getElementById('productCanvas').innerHTML='';document.getElementById('productAtomTools').innerHTML='';document.getElementById('productBuilder').hidden=true}
    const resetVisualBuilderBase=resetVisualBuilder;resetVisualBuilder=()=>{resetVisualBuilderBase();document.getElementById('builderSourceType').value='smarts';visualSourceType='smarts'};
    function renderProductChoices(){const pattern=cleavageModel.cleavage_pattern_set.patterns[visualPatternIndex],select=document.getElementById('builderProductSelect');select.innerHTML='<option value="-1">New product</option>'+((pattern&&pattern.products)||[]).map((product,index)=>'<option value="'+index+'">'+htmlEscape(product.name||('Product '+(index+1)))+'</option>').join('');select.value=String(visualProductIndex);document.getElementById('applyProduct').textContent=visualProductIndex<0?'Add Product':'Update Product'}
    async function loadVisualProduct(index){visualProductIndex=index;const pattern=cleavageModel.cleavage_pattern_set.patterns[visualPatternIndex],product=index<0?null:pattern.products[index];document.getElementById('builderProductName').value=product?product.name:'';document.getElementById('builderProductSourceType').value='smarts';document.getElementById('builderProductSource').value=product?product.smarts:pattern.reactant_smarts;renderProductChoices();await drawProductStructure()}
    async function editVisualPattern(index){const pattern=cleavageModel.cleavage_pattern_set.patterns[index];resetVisualBuilder();visualPatternIndex=index;visualSourceType='smarts';document.getElementById('visualBuilder').hidden=false;document.getElementById('builderSmiles').value=pattern.reactant_smarts;document.getElementById('builderPatternName').value=pattern.name;try{molGraph=await chemistry('molecule',{smarts:pattern.reactant_smarts});selectedAtoms=new Set(molGraph.atoms.map(a=>a.index));selectedBonds=new Set(molGraph.bonds.map(b=>b.index));molGraph.atoms.forEach(a=>{atomConstraintState[a.index]={mode:'custom',smarts:String(a.smarts||a.symbol).replace(/:\\d+(?=\\]$)/,'')};atomMapBySource[a.index]=a.mapNumber||a.index+1});renderMolecule();document.getElementById('productBuilder').hidden=false;await loadVisualProduct(pattern.products.length?0:-1)}catch(e){}}
    document.getElementById('addVisualPattern').onclick=()=>{resetVisualBuilder();document.getElementById('visualBuilder').hidden=false;document.getElementById('builderSmiles').focus()};document.getElementById('closeBuilder').onclick=()=>{resetVisualBuilder();document.getElementById('visualBuilder').hidden=true};
    document.getElementById('drawMolecule').onclick=async()=>{try{molGraph=await chemistry('molecule',sourcePayload());selectedAtoms=new Set();selectedBonds=new Set();atomConstraintState={};bondConstraintState={};atomMapBySource={};renderMolecule()}catch(e){}};
    let elementRequestSequence=0;const elementRequests=new Map();
    document.getElementById('atomConstraints').onclick=e=>{const mode=e.target.dataset.atomMode,open=e.target.dataset.openElements;if(mode){const [i,value]=mode.split(':');atomConstraintState[i].mode=value;queryPanels()}if(open!==undefined){const requestId=++elementRequestSequence;elementRequests.set(requestId,Number(open));vscode.postMessage({type:'selectElements',requestId,elements:atomConstraintState[open].elements})}};
    document.getElementById('atomConstraints').oninput=e=>{if(e.target.dataset.customAtom!==undefined)atomConstraintState[e.target.dataset.customAtom].smarts=e.target.value};
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
    document.getElementById('applyReactant').onclick=async()=>{try{const name=document.getElementById('builderPatternName').value,result=await chemistry('reactant',{...sourcePayload(),name,atoms:[...selectedAtoms],bonds:[...selectedBonds],constraints:atomConstraintState,bondConstraints:bondConstraintState,atomMapBySource});atomMapBySource=result.atomMapBySource;if(visualPatternIndex<0){cleavageModel.cleavage_pattern_set.patterns.push({name,reactant_smarts:result.smarts,products:[]});visualPatternIndex=cleavageModel.cleavage_pattern_set.patterns.length-1}else{const pattern=cleavageModel.cleavage_pattern_set.patterns[visualPatternIndex];pattern.name=name;pattern.reactant_smarts=result.smarts}visualSourceType='smarts';document.getElementById('builderSmiles').value=result.smarts;renderCleavage();document.getElementById('productBuilder').hidden=false;await loadVisualProduct(-1)}catch(e){}};
    document.getElementById('applyReactant').addEventListener('click',()=>{document.getElementById('builderSourceType').value='smarts';visualSourceType='smarts'});
    const productCanvas=document.getElementById('productCanvas');
    function productBondBetween(begin,end){const base=productGraph.bonds.find(b=>(b.begin===begin&&b.end===end)||(b.begin===end&&b.end===begin));if(base)return'base:'+base.index;const added=productAddedBonds.find(b=>(b.begin===begin&&b.end===end)||(b.begin===end&&b.end===begin));return added?'added:'+added.id:null}
    function editProductTarget(e){const atom=e.target.closest('[data-mol-atom]'),bond=e.target.closest('[data-product-bond]');productSelectedAtoms.clear();if(atom){const index=Number(atom.dataset.molAtom);if(productBondStart!==null){if(index!==productBondStart&&!productDeletedAtoms.has(index)){const existing=productBondBetween(productBondStart,index);if(existing){productSelectedBond=existing;if(existing.startsWith('base:'))delete productBondOverrides[Number(existing.slice(5))]}else{const added={id:nextProductBondId++,begin:productBondStart,end:index,order:'1'};productAddedBonds.push(added);productSelectedBond='added:'+added.id}}productBondStart=null;productSelectedAtom=null}else{productSelectedAtom=index;productSelectedBond=null}}else if(bond){productSelectedBond=bond.dataset.productBond;productSelectedAtom=null;productBondStart=null}else return;renderProduct()}
    let productDrag=null;
    productCanvas.onpointerdown=e=>{if(!productGraph||e.button!==0)return;e.preventDefault();const svg=productCanvas.querySelector('svg'),q=svgPoint(svg,e);productDrag={points:[{x:q.x,y:q.y}],moved:false,target:e.target};svg.setPointerCapture(e.pointerId)};
    productCanvas.onpointermove=e=>{if(!productDrag)return;const svg=productCanvas.querySelector('svg'),q=svgPoint(svg,e),last=productDrag.points[productDrag.points.length-1],lasso=svg.querySelector('.selection-lasso');if(Math.hypot(q.x-last.x,q.y-last.y)<2)return;productDrag.points.push({x:q.x,y:q.y});productDrag.moved=productDrag.moved||productDrag.points.length>2;lasso.hidden=!productDrag.moved;lasso.setAttribute('points',productDrag.points.map(p=>p.x+','+p.y).join(' '));lasso.style.display=productDrag.moved?'block':'none'};
    productCanvas.onpointerup=e=>{if(!productDrag)return;const finished=productDrag;productDrag=null;if(!finished.moved){editProductTarget({target:finished.target});return}const l=productLayout();productSelectedAtoms.clear();productGraph.atoms.forEach(atom=>{if(pointInPolygon(l.x(atom),l.y(atom),finished.points))productSelectedAtoms.add(atom.index)});productSelectedAtom=null;productSelectedBond=null;productBondStart=null;renderProduct()};
    productCanvas.onpointercancel=()=>{productDrag=null;renderProduct()};
    productCanvas.oncontextmenu=e=>{e.preventDefault();editProductTarget(e)};
    document.getElementById('drawProduct').onclick=()=>drawProductStructure().catch(()=>{});
    document.getElementById('allowProductAtomTypes').onchange=renderProduct;
    document.getElementById('productAtomTools').onclick=e=>{if(e.target.dataset.deleteSelectedProductAtoms!==undefined){productSelectedAtoms.forEach(index=>productDeletedAtoms.add(index));productAddedBonds=productAddedBonds.filter(b=>!productSelectedAtoms.has(b.begin)&&!productSelectedAtoms.has(b.end));productSelectedAtoms.clear();renderProduct();return}if(e.target.dataset.restoreSelectedProductAtoms!==undefined){productSelectedAtoms.forEach(index=>productDeletedAtoms.delete(index));productSelectedAtoms.clear();renderProduct();return}if(e.target.dataset.clearSelectedProductAtoms!==undefined){productSelectedAtoms.clear();renderProduct();return}if(productSelectedAtom===null)return;if(e.target.dataset.toggleProductAtom!==undefined){if(productDeletedAtoms.has(productSelectedAtom))productDeletedAtoms.delete(productSelectedAtom);else{productDeletedAtoms.add(productSelectedAtom);productAddedBonds=productAddedBonds.filter(b=>b.begin!==productSelectedAtom&&b.end!==productSelectedAtom);if(productBondStart===productSelectedAtom)productBondStart=null}renderProduct()}else if(e.target.dataset.startProductBond!==undefined&&!productDeletedAtoms.has(productSelectedAtom)){productBondStart=productSelectedAtom;productSelectedAtom=null;productSelectedBond=null;renderProduct()}};
    document.getElementById('productAtomTools').onchange=e=>{if(productSelectedAtom===null||e.target.dataset.productMap===undefined)return;const number=Number(e.target.value);if(Number.isInteger(number)&&number>0){productAtomMaps[productSelectedAtom]=number;renderProduct()}};
    document.getElementById('productAtomTools').oninput=e=>{if(productSelectedAtom!==null&&e.target.dataset.productAtomType!==undefined)productAtomOverrides[productSelectedAtom]=e.target.value};
    document.getElementById('productBondTools').onclick=e=>{if(e.target.dataset.productType===undefined||productSelectedBond===null)return;const value=e.target.dataset.productType;if(productSelectedBond.startsWith('base:')){const index=Number(productSelectedBond.slice(5));if(value==='preserve')delete productBondOverrides[index];else productBondOverrides[index]=value}else{const id=Number(productSelectedBond.slice(6)),added=productAddedBonds.find(b=>b.id===id);if(!added)return;if(value==='remove'){productAddedBonds=productAddedBonds.filter(b=>b.id!==id);productSelectedBond=null}else added.order=value==='preserve'?'1':value}renderProduct()};
    document.getElementById('builderProductSelect').onchange=e=>loadVisualProduct(Number(e.target.value)).catch(()=>{});
    document.getElementById('newVisualProduct').onclick=()=>loadVisualProduct(-1).catch(()=>{});
    document.getElementById('applyProduct').onclick=async()=>{let pattern,previous,added=false;try{if(!productGraph)throw new Error('Draw a product structure first.');pattern=cleavageModel.cleavage_pattern_set.patterns[visualPatternIndex];const result=await chemistry('productFromStructure',{...productSourcePayload(),reactantSmarts:pattern.reactant_smarts,atomMaps:productAtomMaps,atomOverrides:document.getElementById('allowProductAtomTypes').checked?productAtomOverrides:{},bondOverrides:productBondOverrides,deletedAtoms:[...productDeletedAtoms],addedBonds:productAddedBonds}),next={name:document.getElementById('builderProductName').value,smarts:result.smarts};if(visualProductIndex<0){pattern.products.push(next);visualProductIndex=pattern.products.length-1;added=true}else{previous=pattern.products[visualProductIndex];pattern.products[visualProductIndex]=next}await chemistry('validate',pattern);document.getElementById('builderProductSourceType').value='smarts';document.getElementById('builderProductSource').value=result.smarts;renderCleavage();renderProductChoices()}catch(e){if(added)pattern.products.pop();else if(pattern&&previous)pattern.products[visualProductIndex]=previous}};
    document.getElementById('applyFragmenterCleavage').onclick=()=>{if(applyFragmenterCleavage)applyFragmenterCleavage(cleavageModel.cleavage_pattern_set);applyFragmenterCleavage=null;document.getElementById('applyFragmenterCleavage').hidden=true;document.querySelector('[data-app="data"]').click();};
    document.getElementById('loadFragmenter').onclick=()=>vscode.postMessage({type:'loadFragmenter'}); document.getElementById('saveFragmenter').onclick=()=>vscode.postMessage({type:'saveFragmenter',text:JSON.stringify(fragmenterEditor.getValue())});
    form.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'run',config:getConfig()});}; document.getElementById('copyCommand').onclick=()=>vscode.postMessage({type:'copyCommand',config:getConfig()}); stop.onclick=()=>vscode.postMessage({type:'stop'});
    trainingForm.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'runTraining',config:getTrainingConfig()})};document.getElementById('trainingCopyCommand').onclick=()=>vscode.postMessage({type:'copyTrainingCommand',config:getTrainingConfig()});trainingStop.onclick=()=>vscode.postMessage({type:'stop'});document.getElementById('saveTraining').onclick=()=>vscode.postMessage({type:'saveTrainingConfig',config:getTrainingConfig()});document.getElementById('loadTraining').onclick=()=>vscode.postMessage({type:'loadTrainingConfig'});
    let predictSequence=0,predictWaiters=new Map(),predictListSequence=0,predictListWaiters=new Map();
    function predictSpectrumRequest(payload){return new Promise((resolve,reject)=>{const requestId=++predictSequence;predictWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'predictSpectrum',requestId,payload})})}
    function listPredictAdductsRequest(payload){return new Promise((resolve,reject)=>{const requestId=++predictListSequence;predictListWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'listPredictAdducts',requestId,payload})})}
    function setPredictEnabled(enabled){predictForm.elements.smiles.disabled=!enabled;predictForm.elements.ce.disabled=!enabled;predictForm.elements.adductType.disabled=!enabled;document.getElementById('predictPreview').disabled=!enabled;document.getElementById('predictSubmit').disabled=!enabled;}
    function renderAdductOptions(adducts){const select=predictForm.elements.adductType;select.innerHTML=adducts.length?adducts.map(a=>'<option value="'+htmlEscape(a)+'">'+htmlEscape(a)+'</option>').join(''):'<option value="">No adducts available</option>';}
    function resetPredictModelState(){document.getElementById('predictModelStatus').textContent='Model checkpoint changed — apply it again to refresh adduct types.';document.getElementById('predictModelStatus').className='status idle';renderAdductOptions([]);predictForm.elements.adductType.innerHTML='<option value="">Apply a model first…</option>';setPredictEnabled(false);}
    document.getElementById('predictApplyModel').onclick=async()=>{const modelPath=predictForm.elements.modelPath.value.trim(),device=predictForm.elements.device.value,modelStatusEl=document.getElementById('predictModelStatus');if(!modelPath){modelStatusEl.textContent='Model checkpoint path is required.';modelStatusEl.className='status error';return}modelStatusEl.textContent='Loading model…';modelStatusEl.className='status running';setPredictEnabled(false);try{const result=await listPredictAdductsRequest({modelPath,device});renderAdductOptions(result.adducts);modelStatusEl.textContent='Model applied. '+result.adducts.length+' main adduct type(s) available.';modelStatusEl.className='status completed';setPredictEnabled(true);}catch(error){modelStatusEl.textContent=String(error.message||error);modelStatusEl.className='status error';setPredictEnabled(false);}};
    predictForm.elements.modelPath.addEventListener('input',resetPredictModelState);
    function spectrumSvg(peaks){if(!peaks.length)return '<p class="muted">No peaks were predicted.</p>';const maxMz=Math.max(...peaks.map(p=>p.mz))*1.05,maxIntensity=Math.max(...peaks.map(p=>p.intensity))||1,width=900,height=260,padL=55,padB=30,padT=12,padR=20,x=mz=>padL+(mz/maxMz)*(width-padL-padR),y=intensity=>height-padB-(intensity/maxIntensity)*(height-padT-padB);const bars=peaks.map(p=>'<line class="spectrum-bar" x1="'+x(p.mz)+'" x2="'+x(p.mz)+'" y1="'+y(0)+'" y2="'+y(p.intensity)+'"><title>m/z '+p.mz.toFixed(4)+' · intensity '+p.intensity.toPrecision(4)+(p.formula?(' · '+p.formula):'')+'</title></line>').join('');const ticks=[0,.25,.5,.75,1].map(f=>'<text x="'+x(maxMz*f)+'" y="'+(height-8)+'" text-anchor="middle" class="spectrum-tick">'+(maxMz*f).toFixed(0)+'</text>').join('');return '<svg class="spectrum-chart" viewBox="0 0 '+width+' '+height+'"><line x1="'+padL+'" y1="'+padT+'" x2="'+padL+'" y2="'+y(0)+'" class="spectrum-axis"/><line x1="'+padL+'" y1="'+y(0)+'" x2="'+(width-padR)+'" y2="'+y(0)+'" class="spectrum-axis"/>'+bars+ticks+'</svg>'}
    function peakTable(peaks){return '<table><thead><tr><th>#</th><th>m/z</th><th>intensity</th><th>formula</th></tr></thead><tbody>'+peaks.map((p,i)=>'<tr><td>'+(i+1)+'</td><td>'+p.mz.toFixed(6)+'</td><td>'+p.intensity.toPrecision(5)+'</td><td>'+htmlEscape(p.formula||'')+'</td></tr>').join('')+'</tbody></table>'}
    let lastPredictTree=null;
    function predictTreeSvg(tree){if(!tree||!tree.nodes.length)return '<p class="muted">No fragment tree was produced.</p>';const detectedNodes=new Set(tree.nodes.filter(n=>n.annotations&&n.annotations.length).map(n=>n.id)),depth=new Map(tree.nodes.map(n=>[n.id,Math.max(0,n.depth)]));const levels={};for(const n of tree.nodes)(levels[depth.get(n.id)]||(levels[depth.get(n.id)]=[])).push(n);for(const nodes of Object.values(levels))nodes.sort((a,b)=>a.id-b.id);const maxDepth=Math.max(...depth.values(),1),width=Math.max(900,(maxDepth+1)*220),maxRows=Math.max(...Object.values(levels).map(x=>x.length)),height=Math.max(240,maxRows*58+50),pos=new Map();for(const [d,nodes] of Object.entries(levels))nodes.forEach((n,i)=>pos.set(n.id,{x:45+Number(d)/maxDepth*(width-90),y:30+(i+1)*(height-40)/(nodes.length+1)}));const lines=tree.edges.map(e=>{const a=pos.get(e.source),b=pos.get(e.target);return '<line class="tree-edge" x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y+'"><title>edge '+e.id+'</title></line>'}).join('');const nodes=tree.nodes.map(n=>{const p=pos.get(n.id);return '<g class="tree-node '+(detectedNodes.has(n.id)?'detected':'')+'" data-node="'+n.id+'" transform="translate('+p.x+' '+p.y+')"><circle r="13"><title>'+htmlEscape(n.smiles)+'</title></circle><text text-anchor="middle" dominant-baseline="central">'+n.id+'</text></g>'}).join('');return '<div class="tree-scroll"><svg class="fragment-tree" viewBox="0 0 '+width+' '+height+'">'+lines+nodes+'</svg></div>'}
    document.getElementById('predictPreview').onclick=async()=>{const smiles=predictForm.elements.smiles.value.trim(),adductType=predictForm.elements.adductType.value.trim(),box=document.getElementById('predictMoleculePreview');if(!smiles){box.hidden=true;return}try{const result=await chemistry('depict',{smiles,adducts:adductType?[adductType]:[],usedAdduct:adductType});box.hidden=false;const row=result.adducts[0];box.innerHTML='<div>'+result.svg+'</div><dl><dt>Formula</dt><dd>'+htmlEscape(result.formula)+'</dd><dt>Exact mass</dt><dd>'+result.exactMass.toFixed(6)+'</dd>'+(row?'<dt>Precursor m/z ('+htmlEscape(row.adduct)+')</dt><dd>'+(row.error?('<span class="warning">'+htmlEscape(row.error)+'</span>'):row.mz.toFixed(6))+'</dd>':'')+'</dl>'}catch(e){box.hidden=false;box.innerHTML='<p class="warning">'+htmlEscape(e.message||String(e))+'</p>'}};
    predictForm.onsubmit=async e=>{e.preventDefault();const modelPath=predictForm.elements.modelPath.value.trim(),smiles=predictForm.elements.smiles.value.trim(),ce=predictForm.elements.ce.value.trim(),adductType=predictForm.elements.adductType.value.trim(),device=predictForm.elements.device.value,predictStatusEl=document.getElementById('predictStatus'),resultSection=document.getElementById('predictResultSection'),resultBox=document.getElementById('predictResult');if(!modelPath||!smiles||!ce||!adductType){predictStatusEl.textContent='Model checkpoint, SMILES, collision energy, and adduct type are required.';predictStatusEl.className='status error';return}predictStatusEl.textContent='Predicting… this can take a while for large models.';predictStatusEl.className='status running';lastPredictTree=null;try{const result=await predictSpectrumRequest({modelPath,smiles,ce,adductType,device});predictStatusEl.textContent='Predicted '+result.peaks.length+' peaks.';predictStatusEl.className='status completed';resultSection.hidden=false;lastPredictTree=result.tree;resultBox.innerHTML='<div class="predict-layout"><div>'+spectrumSvg(result.peaks)+peakTable(result.peaks)+'<h3>Fragment Tree</h3><p class="muted">Circles are candidate fragments explored while predicting. Accent-colored fragments back at least one predicted peak. Click a fragment to inspect its structure.</p>'+predictTreeSvg(result.tree)+'<div id="predictFragmentDetail" hidden></div></div><aside class="predict-side"><div class="molecule-preview">'+result.svg+'</div><dl><dt>Formula</dt><dd>'+htmlEscape(result.formula)+'</dd><dt>Exact mass</dt><dd>'+result.exactMass.toFixed(6)+'</dd><dt>Adduct</dt><dd>'+htmlEscape(result.adduct)+'</dd><dt>Precursor m/z</dt><dd>'+result.precursorMz.toFixed(6)+'</dd></dl></aside></div>'}catch(error){predictStatusEl.textContent=String(error.message||error);predictStatusEl.className='status error';resultSection.hidden=true}};
    document.getElementById('predictResult').addEventListener('click',async e=>{const nodeEl=e.target.closest('[data-node]');if(!nodeEl||!lastPredictTree)return;const nodeId=Number(nodeEl.dataset.node),node=lastPredictTree.nodes.find(n=>n.id===nodeId);if(!node)return;const detail=document.getElementById('predictFragmentDetail');detail.hidden=false;detail.innerHTML='<p class="muted">Loading…</p>';detail.scrollIntoView({behavior:'smooth',block:'nearest'});try{const adducts=(node.annotations||[]).map(a=>a.adduct);const depicted=await chemistry('depict',{smiles:node.smiles,adducts,usedAdduct:adducts[0]||''});const rows=(node.annotations||[]).map(a=>'<tr><td>'+htmlEscape(a.adduct)+'</td><td>'+htmlEscape(a.ionFormula||'')+'</td><td>'+a.peakMz.toFixed(6)+'</td><td>'+a.peakIntensity.toPrecision(4)+'</td><td>'+(a.probability*100).toFixed(1)+'%</td></tr>').join('');detail.innerHTML='<h4>Fragment node '+nodeId+'</h4><div class="predict-layout"><div class="molecule-preview">'+depicted.svg+'</div><div><dl><dt>SMILES</dt><dd>'+htmlEscape(node.smiles)+'</dd><dt>Formula</dt><dd>'+htmlEscape(node.formula)+'</dd><dt>Exact mass</dt><dd>'+node.exactMass.toFixed(6)+'</dd></dl>'+(rows?('<table><thead><tr><th>Adduct</th><th>Ion formula</th><th>m/z</th><th>intensity</th><th>probability</th></tr></thead><tbody>'+rows+'</tbody></table>'):'<p class="muted">This fragment did not directly back a predicted peak.</p>')+'</div></div>'}catch(error){detail.innerHTML='<p class="warning">'+htmlEscape(error.message||String(error))+'</p>'}});
    document.getElementById('checkpointGraph').onclick=()=>vscode.postMessage({type:'pickCheckpoint',outputDir:trainingForm.elements.outputDir.value,experimentName:trainingForm.elements.experimentName.value});
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='picked'){const target=m.form==='training'?trainingForm:m.form==='predict'?predictForm:form;if(target.elements[m.field])target.elements[m.field].value=m.value;if(m.form==='predict'&&m.field==='modelPath')resetPredictModelState()}if(m.type==='checkpointPicked')trainingForm.elements.ckptId.value=m.checkpointId;if(m.type==='config')setConfig(m.config);if(m.type==='trainingConfig')setFormConfig(trainingForm,m.config);if(m.type==='fragmenter'){fragmenterEditor.setValue(JSON.parse(m.text));}if(m.type==='cleavagePatternSet'){cleavageModel=m.value;cleavagePath=m.path;document.getElementById('cleavagePath').textContent=m.path;renderCleavage()}if(m.type==='cleavagePatternSetSaved'){cleavagePath=m.path;document.getElementById('cleavagePath').textContent=m.path}if(m.type==='cleavagePatternLoaded'){if(m.index>=0)cleavageModel.cleavage_pattern_set.patterns[m.index]=m.pattern;else cleavageModel.cleavage_pattern_set.patterns.push(m.pattern);renderCleavage()}if(m.type==='elementSelectionResult'){const atom=elementRequests.get(m.requestId);if(atom!==undefined&&m.elements.length){atomConstraintState[atom].elements=m.elements;atomConstraintState[atom].mode='elements';elementRequests.delete(m.requestId);queryPanels()}}if(m.type==='chemistryResult'){const waiter=chemistryWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);chemistryWaiters.delete(m.requestId)}}if(m.type==='chemistryError'){const waiter=chemistryWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));chemistryWaiters.delete(m.requestId)}}if(m.type==='predictSpectrumResult'){const waiter=predictWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);predictWaiters.delete(m.requestId)}}if(m.type==='predictSpectrumError'){const waiter=predictWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));predictWaiters.delete(m.requestId)}}if(m.type==='listPredictAdductsResult'){const waiter=predictListWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);predictListWaiters.delete(m.requestId)}}if(m.type==='listPredictAdductsError'){const waiter=predictListWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));predictListWaiters.delete(m.requestId)}}if(m.type==='status'){statusEl.textContent=m.text;statusEl.className='status '+m.status;stop.disabled=m.status!=='running';if(m.command)document.getElementById('command').textContent=m.command;}if(m.type==='trainingStatus'){trainingStatusEl.textContent=m.text;trainingStatusEl.className='status '+m.status;trainingStop.disabled=m.status!=='running';if(m.command)document.getElementById('trainingCommand').textContent=m.command;}});`;
}
function commonCss() { return `:root{color-scheme:light dark;--accent:#36c5a2;--panel:color-mix(in srgb,var(--vscode-editor-background) 88%,var(--vscode-editor-foreground));--border:color-mix(in srgb,var(--vscode-editor-foreground) 18%,transparent)}*{box-sizing:border-box}body{font-family:var(--vscode-font-family);color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);margin:0}main{max-width:1100px;margin:auto;padding:32px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:28px}h1{font-size:32px;margin:4px 0}h2{font-size:17px;margin:0 0 18px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.14em;font-size:11px}.muted,small{opacity:.65}button{font:inherit;color:inherit;background:var(--vscode-button-secondaryBackground);border:1px solid var(--border);border-radius:6px;padding:8px 13px;cursor:pointer}button:hover{background:var(--vscode-button-secondaryHoverBackground)}section{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:22px;margin:14px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;background:none;border:0;padding:0}.cards article{background:var(--panel);border:1px solid var(--border);padding:18px;border-radius:9px}.cards b{display:block;font-size:23px;color:var(--accent)}.cards span{opacity:.65}.files{display:grid;gap:4px}.file{display:flex;justify-content:space-between;text-align:left;background:transparent;border:0;border-bottom:1px solid var(--border);border-radius:0}.file em{opacity:.55;font-style:normal}dl{display:grid;grid-template-columns:110px 1fr;gap:10px}dt{opacity:.6}dd{margin:0;overflow-wrap:anywhere}code{font-family:var(--vscode-editor-font-family);font-size:12px}details{border-top:1px solid var(--border);padding:10px 0}summary{cursor:pointer;font-weight:600}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--border);padding:7px}.formula{margin-bottom:7px}.formula small{display:block;margin-top:3px;font-family:var(--vscode-editor-font-family)}.warning{color:var(--vscode-editorWarning-foreground)}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}main{padding:18px}table{display:block;overflow:auto}}`; }
function trainingCss() { return `.model-map{background:color-mix(in srgb,var(--vscode-editor-background) 94%,var(--accent))}.architecture{display:grid;gap:13px;margin-top:15px}.architecture-input,.architecture-merge,.architecture-branches{display:grid;gap:9px}.architecture-branches{grid-template-columns:1fr 1fr 1.35fr}.architecture-modes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mode{display:grid;grid-template-columns:1fr 1fr;gap:8px;border:1px solid var(--border);border-radius:8px;padding:10px}.mode strong{grid-column:1/-1;font-size:10px;letter-spacing:.14em}.training-mode strong{color:#e9b949}.prediction-mode strong{color:#63a8ff}.architecture button{position:relative;text-align:left;min-height:78px;background:var(--vscode-editor-background)}.architecture button b,.architecture button code,.architecture button span{display:block}.architecture button b{color:var(--accent);margin-bottom:5px}.architecture button code{font-size:11px;margin-bottom:5px;white-space:normal}.architecture button span{font-size:11px;opacity:.68}.architecture button.selected{outline:2px solid var(--accent);background:color-mix(in srgb,var(--accent) 14%,var(--vscode-editor-background))}.architecture-input button::after,.architecture-branches button::after,.architecture-merge button::after{content:'↓';position:absolute;left:50%;bottom:-22px;color:var(--accent);z-index:2;font-size:17px}.model-hint{margin:13px 0 0;font-size:12px;opacity:.75}[data-model-block]{scroll-margin-top:12px;transition:border-color .15s,box-shadow .15s}[data-model-block].selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}.model-losses{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}.model-losses span:nth-child(odd){padding:8px 11px;border:1px solid var(--border);border-radius:18px;background:var(--vscode-editor-background)}.model-notes{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0 16px}.model-notes p{margin:0;padding:10px;border-left:3px solid var(--accent);background:var(--vscode-editor-background);font-size:12px;line-height:1.45}.field-help{display:block;margin-top:5px;line-height:1.35;opacity:.7}.check .field-help{margin-top:2px}.check{align-items:flex-start}@media(max-width:850px){.architecture-branches,.architecture-modes,.mode,.model-notes{grid-template-columns:1fr}.mode strong{grid-column:1}.architecture button::after{display:none}}`; }
function formCss() { return `nav{display:flex;gap:8px;margin-bottom:18px}.molecule-preview{width:100%;max-height:340px;overflow:hidden;background:#fff;border-radius:6px;margin:14px 0}.molecule-preview svg{display:block;width:100%;height:auto}.predict-layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px;align-items:start}.predict-side{border:1px solid var(--border);border-radius:8px;padding:14px}.spectrum-chart{width:100%;height:260px;margin:14px 0}.spectrum-bar{stroke:var(--accent);stroke-width:2}.spectrum-axis{stroke:var(--vscode-editor-foreground);stroke-opacity:.4}.spectrum-tick{font-size:10px;fill:var(--vscode-editor-foreground);opacity:.7}.tree-scroll{overflow:auto;max-height:60vh;border:1px solid var(--border);border-radius:8px;background:var(--vscode-editor-background);margin:10px 0}.fragment-tree{display:block;width:100%;min-width:700px}.tree-edge{stroke:color-mix(in srgb,var(--vscode-editor-foreground) 35%,transparent);stroke-width:1.5}.tree-node{cursor:pointer}.tree-node circle{fill:var(--vscode-editor-background);stroke:var(--vscode-editor-foreground);stroke-width:1.5}.tree-node.detected circle{fill:color-mix(in srgb,#ff8a3d 28%,var(--vscode-editor-background));stroke:#ff8a3d;stroke-width:3}.tree-node text{fill:var(--vscode-editor-foreground);font-size:9px;pointer-events:none}#predictFragmentDetail{border-top:1px solid var(--border);margin-top:14px;padding-top:14px}@media(max-width:850px){.predict-layout{grid-template-columns:1fr}}.tab{border-radius:20px}.tab.active{border-color:var(--accent)}[hidden]{display:none!important}label{display:block;font-size:12px;opacity:.8;margin:12px 0}input,select,textarea{width:100%;display:block;margin-top:6px;padding:9px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,var(--border));border-radius:5px;font:inherit}.path{display:flex;gap:7px}.path input{flex:1}.row{display:flex;align-items:end;gap:12px}.grow{flex:1}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0 16px}.checks{display:flex;flex-wrap:wrap;gap:8px 22px}.check{display:flex;align-items:center;gap:7px}.check input{width:auto;margin:0}.section-title{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.section-title h2{margin-bottom:4px}.section-title p{margin:0}.actions{display:flex;gap:8px}textarea{min-height:280px;resize:vertical;font-family:var(--vscode-editor-font-family);font-size:12px;line-height:1.5}.pattern-card{border-left:3px solid var(--accent)}.product-title{align-items:center;margin-top:18px}.product-title h3{margin:0}.product-list{margin-left:18px}.product-row{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:end;border-top:1px solid var(--border);padding:6px 0}.danger{color:var(--vscode-errorForeground)}.molecule-canvas{min-height:260px;margin:16px 0;border:1px solid var(--border);border-radius:8px;background:var(--vscode-editor-background);color:var(--vscode-editor-foreground);touch-action:none;user-select:none}.molecule-canvas svg{display:block;width:100%;height:440px}.bond-group{stroke:currentColor;stroke-width:3;stroke-linecap:round;cursor:pointer}.bond-aromatic .aromatic-mark{stroke-dasharray:8 6}.bond-group.selected{stroke:var(--accent);stroke-width:5}.bond-group.deleted{opacity:.2;stroke-dasharray:5 4}.bond-hit{stroke:transparent!important;stroke-width:24}.mol-atom{cursor:pointer}.atom-hit{fill:transparent;stroke:transparent}.atom-label-bg{fill:var(--vscode-editor-background);stroke:color-mix(in srgb,var(--vscode-editor-foreground) 45%,transparent);stroke-width:1.5}.mol-atom text{fill:var(--vscode-editor-foreground);font-size:16px;font-family:var(--vscode-font-family);font-weight:700;pointer-events:none}.mol-atom .atom-index{font-size:11px;font-weight:600;fill:var(--vscode-descriptionForeground)}.mol-atom.selected .atom-label-bg{stroke:var(--accent);stroke-width:3;fill:color-mix(in srgb,var(--accent) 20%,var(--vscode-editor-background))}.mol-atom.selected .atom-hit{fill:#36c5a214}.mol-atom.deleted{opacity:.25}.selection-lasso{fill:none;stroke:var(--vscode-editor-foreground);stroke-width:2.5;stroke-dasharray:7 5;stroke-linecap:round;stroke-linejoin:round;pointer-events:none}.builder-columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.query-list{display:grid;gap:8px}.query-card{border:1px solid var(--border);border-radius:7px;padding:10px}.query-head{display:flex;justify-content:space-between}.query-modes,.bond-type-choices,.bond-tools{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.query-card button,.bond-tools button{padding:5px 8px}.query-card button.active,.bond-tools button.active{background:var(--accent);color:#10251f;border-color:var(--accent)}.element-picker{width:100%;margin-top:8px}.element-summary{margin-top:7px;padding:6px 8px;border-radius:4px;background:var(--vscode-textBlockQuote-background);font-size:12px}.product-canvas{max-height:350px}.product-canvas svg{height:350px}.bond-tools{align-items:center;margin-bottom:12px}#productBuilder{margin-top:22px;padding-top:16px;border-top:1px solid var(--border)}#helpTooltip{position:fixed;z-index:50;display:none;max-width:380px;padding:9px 11px;border:1px solid var(--vscode-editorHoverWidget-border,var(--border));border-radius:5px;background:var(--vscode-editorHoverWidget-background);color:var(--vscode-editorHoverWidget-foreground);box-shadow:0 4px 14px #0005;font-size:12px;line-height:1.4}#helpTooltip.visible{display:block}.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground);font-weight:600}.primary:hover{background:var(--vscode-button-hoverBackground)}footer{position:sticky;bottom:0;background:var(--vscode-editor-background);border-top:1px solid var(--border);padding:17px 0;display:flex;justify-content:space-between;align-items:center;gap:20px}.status{font-weight:600}.status.running{color:#e9b949}.status.completed{color:var(--accent)}.status.error,.status.failed{color:#ef6b73}footer code{display:block;opacity:.65;max-width:700px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:5px}@media(max-width:700px){.grid,.product-row,.builder-columns{grid-template-columns:1fr}.row{display:block}footer{position:static}}`; }

function deactivate() { if (runningProcess) runningProcess.kill('SIGTERM'); }
module.exports = { activate, deactivate, buildArgs, normalizeConfig, buildTrainingArgs, normalizeTrainingConfig, resultHtml, workbenchHtml, ensureFileSuffix };
