const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');

const defaults = {
  db: 'unspecified', batchSize: 32, smilesColumn: 'SMILES',
  precursorMzColumn: 'PrecursorMZ', adductTypeColumn: 'AdductType',
  collisionEnergyColumn: 'CollisionEnergy', instrumentColumn: '',
  specIdColumn: 'SpecID', overwrite: false
};

function buildBatchArgs(config) {
  const args = ['-m', 'clefts.ml.specgen.predict_spectrum',
    '--input', config.input, '--output', config.output, '--model', config.modelPath,
    '--device', config.device || 'cpu', '--db', config.db || 'unspecified',
    '--batch-size', String(config.batchSize || 32),
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
  panel.onDidDispose(() => { if (child) child.kill('SIGTERM'); });
  panel.webview.onDidReceiveMessage(async message => {
    if (!['predictBatchPick', 'predictBatchCopy', 'predictBatchRun', 'predictBatchStop'].includes(message.type)) return;
    const post = data => panel.webview.postMessage({ type: 'predictBatchStatus', ...data });
    try {
      if (message.type === 'predictBatchStop') { if (child) child.kill('SIGTERM'); return; }
      if (message.type === 'predictBatchPick') {
        const save = message.field === 'output';
        const selected = save
          ? await vscode.window.showSaveDialog({ filters: { 'MSDataset': ['msds'] }, defaultUri: vscode.Uri.file('predicted.msds') })
          : (await vscode.window.showOpenDialog({ filters: { 'MSDataset': ['msds'] }, canSelectMany: false }))?.[0];
        if (selected) panel.webview.postMessage({ type: 'predictBatchPicked', field: message.field, value: selected.fsPath });
        return;
      }
      const config = { ...defaults, ...message.config };
      for (const key of ['input', 'output', 'modelPath']) if (!String(config[key] || '').trim()) throw new Error(`${key} is required.`);
      const batchSize = Number(config.batchSize);
      if (!Number.isInteger(batchSize) || batchSize <= 0) throw new Error('batchSize must be a positive integer.');
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
      child.stdin.on('error', () => {}); child.stdin.end();
      const stream = data => { const log = data.toString(); output.append(log); post({ log }); };
      child.stdout.on('data', stream); child.stderr.on('data', stream);
      const code = await new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', (exitCode, signal) => signal ? reject(new Error('Batch prediction cancelled.')) : resolve(exitCode));
      });
      if (code !== 0 && code !== 3) throw new Error(`Prediction CLI failed (exit ${code}). See output for details.`);
      post({ running: false, text: code === 3 ? 'Prediction completed with some failed records.' : 'MSDataset prediction completed.' });
    } catch (error) {
      post({ running: false, text: error.message });
    } finally {
      if (message.type === 'predictBatchRun') child = undefined;
    }
  });
}

function html() {
  return `<section id="predictBatchSection"><div class="section-title"><div><h2>Batch MSDataset prediction</h2><p class="muted">Load the model once, predict the database in bounded SMILES chunks, and save compatible predicted spectra with provenance metadata.</p></div></div>
  <div class="grid"><label>Input MSDataset *<div class="path"><input name="batchInput"><button type="button" data-predict-batch-pick="input">Browse</button></div></label><label>Output MSDataset *<div class="path"><input name="batchOutput"><button type="button" data-predict-batch-pick="output">Browse</button></div></label><label>DB label<input name="batchDb" value="unspecified"></label><label>Unique-SMILES chunk size<input name="batchSize" type="number" min="1" value="32"></label><label>SpecID column<input name="batchSpecIdColumn" value="SpecID"></label><label>SMILES column<input name="batchSmilesColumn" value="SMILES"></label><label>Precursor m/z column<input name="batchPrecursorMzColumn" value="PrecursorMZ"></label><label>Adduct column<input name="batchAdductTypeColumn" value="AdductType"></label><label>Collision energy column<input name="batchCollisionEnergyColumn" value="CollisionEnergy"></label><label>Instrument column (optional)<input name="batchInstrumentColumn"></label></div>
  <label class="check"><input name="batchOverwrite" type="checkbox"><span>Overwrite existing output</span></label>
  <div class="actions"><button type="button" id="predictBatchCopy">Copy CLI Command</button><button type="button" id="predictBatchStop" disabled>Stop</button><button type="button" id="predictBatchRun" class="primary">Predict MSDataset</button></div>
  <p id="predictBatchStatus" class="status idle">Ready</p><pre id="predictBatchCommand" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre><pre id="predictBatchLog" style="max-height:280px;overflow:auto;white-space:pre-wrap"></pre></section>`;
}

function script() { return `(${client.toString()})();`; }
function client() {
  const el = id => document.getElementById(id), predictForm = el('predictForm');
  const value = name => predictForm.elements[name]?.value || '';
  const config = () => ({
    input: value('batchInput'), output: value('batchOutput'), db: value('batchDb'),
    batchSize: value('batchSize'), specIdColumn: value('batchSpecIdColumn'),
    smilesColumn: value('batchSmilesColumn'), precursorMzColumn: value('batchPrecursorMzColumn'),
    adductTypeColumn: value('batchAdductTypeColumn'), collisionEnergyColumn: value('batchCollisionEnergyColumn'),
    instrumentColumn: value('batchInstrumentColumn'), overwrite: !!predictForm.elements.batchOverwrite?.checked,
    modelPath: value('modelPath'), device: value('device')
  });
  const send = type => vscode.postMessage({ type, config: config() });
  el('predictBatchRun').onclick = () => send('predictBatchRun');
  el('predictBatchCopy').onclick = () => send('predictBatchCopy');
  el('predictBatchStop').onclick = () => vscode.postMessage({ type: 'predictBatchStop' });
  document.querySelectorAll('[data-predict-batch-pick]').forEach(button => button.onclick = () =>
    vscode.postMessage({ type: 'predictBatchPick', field: button.dataset.predictBatchPick }));
  window.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'predictBatchPicked') {
      const field = message.field === 'input' ? 'batchInput' : 'batchOutput';
      predictForm.elements[field].value = message.value;
    }
    if (message.type !== 'predictBatchStatus') return;
    if (message.running !== undefined) {
      el('predictBatchRun').disabled = message.running;
      el('predictBatchStop').disabled = !message.running;
    }
    if (message.text) el('predictBatchStatus').textContent = message.text;
    if (message.command) el('predictBatchCommand').textContent = message.command;
    if (message.clear) el('predictBatchLog').textContent = '';
    if (message.log) el('predictBatchLog').textContent = (el('predictBatchLog').textContent + message.log).slice(-30000);
  });
}

module.exports = { attach, html, script, buildBatchArgs, shellDisplay, defaults };
