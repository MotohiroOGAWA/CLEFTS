const projects=require('../../workbench/project');
const jobs=require('../../workbench/panel');
const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');
const { saveConfiguration, loadConfiguration } = require('../../configuration');

const defaults = {
  db: 'unspecified', smilesColumn: 'SMILES',
  precursorMzColumn: 'PrecursorMZ', adductTypeColumn: 'AdductType',
  collisionEnergyColumn: 'CollisionEnergy', instrumentColumn: '',
  specIdColumn: 'SpecID', outputName: 'predicted.msds', overwrite: false, maxSamples:128
};

function buildBatchArgs(config) {
  const args = ['-m', 'clefts.ml.specgen.predict_spectrum',
    '--input', config.input, '--output-dir', config.outputDir,
    '--output-name', config.outputName || 'predicted.msds', '--model', config.modelPath,
    '--device', config.device || 'cpu', '--max-samples',String(config.maxSamples??128), '--db', config.db || 'unspecified',
    '--smiles-column', config.smilesColumn || 'SMILES',
    '--precursor-mz-column', config.precursorMzColumn || 'PrecursorMZ',
    '--adduct-type-column', config.adductTypeColumn || 'AdductType',
    '--collision-energy-column', config.collisionEnergyColumn || 'CollisionEnergy',
    '--spec-id-column', config.specIdColumn || 'SpecID'];
  if (String(config.instrumentColumn || '').trim()) args.push('--instrument-column', config.instrumentColumn.trim());
  if (config.overwrite) args.push('--overwrite');
  return args;
}

function shellDisplay(program, args) {
  return [program, ...args].map(value => /^[A-Za-z0-9_./:=,-]+$/.test(value)
    ? value : "'" + value.replace(/'/g, "'\\''") + "'").join(' ');
}

function attach(panel, context, projectRoot, output) {
  let child;
  const panelContext=context;
  panel.onDidDispose(() => { if (child) child.kill('SIGTERM'); });
  panel.webview.onDidReceiveMessage(async message => {
    const context=projects.capture(panelContext);
    let observed;
    if (!['predictBatchPick', 'predictBatchCopy', 'predictBatchRun', 'predictBatchStop', 'predictSaveConfig', 'predictLoadConfig'].includes(message.type)) return;
    const post = data => panel.webview.postMessage({ type: 'predictBatchStatus', ...data });
    try {
      if (message.type === 'predictSaveConfig') {
        const saved = await saveConfiguration('spectrum-prediction', message.config, 'spectrum-prediction.json');
        if (saved) panel.webview.postMessage({ type: 'predictConfigSaved', path: saved });
        return;
      }
      if (message.type === 'predictLoadConfig') {
        const loaded = await loadConfiguration('spectrum-prediction');
        if (loaded) panel.webview.postMessage({ type: 'predictConfigLoaded', ...loaded });
        return;
      }
      if (message.type === 'predictBatchStop') { if (child) child.kill('SIGTERM'); return; }
      if (message.type === 'predictBatchPick') {
        const selected = message.field === 'outputDir'
          ? (await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, canSelectMany: false }))?.[0]
          : (await vscode.window.showOpenDialog({ filters: { 'MSDataset': ['msds'] }, canSelectMany: false }))?.[0];
        if (selected) panel.webview.postMessage({ type: 'predictBatchPicked', field: message.field, value: selected.fsPath });
        return;
      }
      const config = { ...defaults, ...message.config };
      for (const key of ['input', 'outputDir', 'modelPath']) if (!String(config[key] || '').trim()) throw new Error(`${key} is required.`);
      if(!Number.isInteger(Number(config.maxSamples))||Number(config.maxSamples)<1)throw new Error('Maximum simultaneous samples must be a positive integer.');
      const root = projectRoot(context);
      const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
      const args = buildBatchArgs(config);
      const command = shellDisplay(python, args);
      if (message.type === 'predictBatchCopy') {
        await vscode.env.clipboard.writeText(command);
        post({ text: 'Batch prediction command copied.', command });
        return;
      }
      if (child) throw new Error('Batch prediction is already running.');
      output.show(true); output.appendLine('$ ' + command);
      post({ running: true, clear: true, text: 'Predicting MSDataset…', command });
      child = spawn(python, args, { cwd: root, env: { ...process.env, PYTHONUNBUFFERED: '1',
        PYTHONPATH: [root, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) } });
      observed=jobs.observeJob(context,{type:'batch-prediction',workflow:'prediction',name:config.outputName||'Batch prediction',outputDir:config.outputDir,child,command:[python,...args],config,base:root});
      child.stdin.on('error', () => {}); child.stdin.end();
      const stream = data => {
        observed.log(data.toString());
        const clean = data.toString().replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
        const lines = clean.split(/[\r\n]+/).map(line => line.trim()).filter(Boolean);
        const isProgress = line => /\d+%.*\|/.test(line) || /^\[\d+\/\d+\]/.test(line);
        const progress = lines.filter(isProgress).at(-1);
        const staticLog = lines.filter(line => !isProgress(line)).join('\n');
        if (staticLog) output.appendLine(staticLog);
        post({ ...(staticLog ? { log: staticLog + '\n' } : {}), ...(progress ? { progress } : {}) });
      };
      child.stdout.on('data', stream); child.stderr.on('data', stream);
      const code = await new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', (exitCode, signal) => signal ? reject(new Error('Batch prediction cancelled.')) : resolve(exitCode));
      });
      observed.finish({code,partial:code===3});
      if (code !== 0 && code !== 3) throw new Error(`Prediction CLI failed (exit ${code}). See output for details.`);
      post({ running: false, text: code === 3 ? 'Prediction completed with some failed records.' : 'MSDataset prediction completed.' });
    } catch (error) {
      if(observed)observed.finish({error:error.message});
      post({ running: false, text: error.message });
    } finally {
      if (message.type === 'predictBatchRun') child = undefined;
    }
  });
}

function html() {
  return `<section id="predictConfiguration"><div class="section-title"><div><h2>Prediction Configuration</h2><p class="muted">Save or restore the model, single-spectrum conditions, and batch MSDataset settings.</p></div><div class="actions"><button type="button" id="predictLoadConfig">Load Configuration</button><button type="button" id="predictSaveConfig">Save Configuration</button></div></div></section><section id="predictBatchSection"><div class="section-title"><div><h2>Batch MSDataset prediction</h2><p class="muted">Predict in bounded batches and reuse molecular features across shared conditions.</p></div></div>
  <div class="grid"><label>Output directory *<div class="path"><input name="batchOutputDir" data-path-kind="folder"><button type="button" data-predict-batch-pick="outputDir">Browse</button></div></label><label>Output MSDataset filename<input name="batchOutputName" value="predicted.msds"></label><label>Input MSDataset *<div class="path"><input name="batchInput" data-path-kind="file"><button type="button" data-predict-batch-pick="input">Browse</button></div></label><label>Maximum simultaneous samples<input name="batchMaxSamples" type="number" min="1" step="1" value="128"></label><label>DB label<input name="batchDb" value="unspecified"></label><label>SpecID column<input name="batchSpecIdColumn" value="SpecID"></label><label>SMILES column<input name="batchSmilesColumn" value="SMILES"></label><label>Precursor m/z column<input name="batchPrecursorMzColumn" value="PrecursorMZ"></label><label>Adduct column<input name="batchAdductTypeColumn" value="AdductType"></label><label>Collision energy column<input name="batchCollisionEnergyColumn" value="CollisionEnergy"></label><label>Instrument column (optional)<input name="batchInstrumentColumn"></label></div>
  <label class="check"><input name="batchOverwrite" type="checkbox"><span>Overwrite existing output</span></label>
  <div class="actions"><button type="button" id="predictBatchCopy">Copy CLI Command</button><button type="button" id="predictBatchStop" disabled>Stop</button><button type="button" id="predictBatchRun" class="primary">Predict MSDataset</button></div>
  <p id="predictBatchStatus" class="status idle">Ready</p><pre id="predictBatchCommand" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre><pre id="predictBatchLog" style="max-height:280px;overflow:auto;white-space:pre-wrap"></pre></section>`;
}

function script() { return `(${client.toString()})(${JSON.stringify(defaults)});`; }
function client(defaults) {
  const el = id => document.getElementById(id), predictForm = el('predictForm');
  const value = name => predictForm.elements[name]?.value || '';
  const config = () => ({
    input: value('batchInput'), outputDir: value('batchOutputDir'), outputName: value('batchOutputName'), db: value('batchDb'),
    maxSamples:Number(value('batchMaxSamples')), specIdColumn: value('batchSpecIdColumn'),
    smilesColumn: value('batchSmilesColumn'), precursorMzColumn: value('batchPrecursorMzColumn'),
    adductTypeColumn: value('batchAdductTypeColumn'), collisionEnergyColumn: value('batchCollisionEnergyColumn'),
    instrumentColumn: value('batchInstrumentColumn'), overwrite: !!predictForm.elements.batchOverwrite?.checked,
    modelPath: value('modelPath'), device: value('device')
  });
  const fullConfig = () => ({ ...config(), smiles: value('smiles'), ce: value('ce'), adductType: value('adductType') });
  const send = type => vscode.postMessage({ type, config: config() });
  el('predictBatchRun').onclick = () => send('predictBatchRun');
  el('predictBatchCopy').onclick = () => send('predictBatchCopy');
  el('predictBatchStop').onclick = () => vscode.postMessage({ type: 'predictBatchStop' });
  el('predictSaveConfig').onclick = () => vscode.postMessage({ type: 'predictSaveConfig', config: fullConfig() });
  el('predictLoadConfig').onclick = () => vscode.postMessage({ type: 'predictLoadConfig' });
  document.querySelectorAll('[data-predict-batch-pick]').forEach(button => button.onclick = () =>
    vscode.postMessage({ type: 'predictBatchPick', field: button.dataset.predictBatchPick }));
  window.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'predictBatchPicked') {
      const field = message.field === 'input' ? 'batchInput' : 'batchOutputDir';
      predictForm.elements[field].value = message.value;
    }
    if (message.type === 'predictConfigLoaded') {
      const loaded = { ...defaults, ...(message.config || {}) };
      if (!loaded.outputDir && loaded.output) loaded.outputDir = loaded.output.replace(/[\\/][^\\/]+$/, '');
      const fields = { maxSamples:'batchMaxSamples',input: 'batchInput', outputDir: 'batchOutputDir', outputName: 'batchOutputName', db: 'batchDb',
        specIdColumn: 'batchSpecIdColumn', smilesColumn: 'batchSmilesColumn', precursorMzColumn: 'batchPrecursorMzColumn',
        adductTypeColumn: 'batchAdductTypeColumn', collisionEnergyColumn: 'batchCollisionEnergyColumn', instrumentColumn: 'batchInstrumentColumn' };
      for (const [key, name] of Object.entries(fields)) if (predictForm.elements[name]) predictForm.elements[name].value = loaded[key] ?? '';
      predictForm.elements.batchOverwrite.checked = !!loaded.overwrite;
      for (const name of ['modelPath', 'device', 'smiles', 'ce']) if (predictForm.elements[name] && loaded[name] !== undefined) predictForm.elements[name].value = loaded[name];
      if (loaded.adductType) predictForm.elements.adductType.dataset.restoreValue = loaded.adductType;
      el('predictBatchStatus').textContent = 'Loaded configuration ' + message.path;
      if (loaded.modelPath) el('predictApplyModel').click();
    }
    if (message.type === 'predictConfigSaved') el('predictBatchStatus').textContent = 'Saved configuration ' + message.path;
    if (message.type !== 'predictBatchStatus') return;
    if (message.running !== undefined) {
      el('predictBatchRun').disabled = message.running;
      el('predictBatchStop').disabled = !message.running;
    }
    if (message.text) el('predictBatchStatus').textContent = message.text;
    if (message.progress) el('predictBatchStatus').textContent = message.progress;
    if (message.command) el('predictBatchCommand').textContent = message.command;
    if (message.clear) el('predictBatchLog').textContent = '';
    if (message.log) el('predictBatchLog').textContent = (el('predictBatchLog').textContent + message.log).slice(-30000);
  });
}

module.exports = { attach, html, script, buildBatchArgs, shellDisplay, defaults };
