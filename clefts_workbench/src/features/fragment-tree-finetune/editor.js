const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');

const fields = [
  ['checkpoint', 'Base fragment-tree checkpoint', 'file'],
  ['cleavagePatternSet', 'New complete Cleavage Pattern Set', 'file'],
  ['trainDir', 'Regenerated training structures', 'folder'],
  ['valDir', 'Regenerated validation structures', 'folder'],
  ['outputDir', 'Fine-tuning output directory', 'folder'],
  ['adapterWidth', 'Added nodes per projection', 'number', '8'],
  ['lr', 'Learning rate', 'number', '0.0001'],
  ['weightDecay', 'Weight decay', 'number', '0'],
  ['epochs', 'Epochs', 'number', '10'],
  ['batchSize', 'Batch size', 'number', '1'],
  ['maxSamples', 'Max samples', 'number', '100'],
  ['numWorkers', 'Workers', 'number', '0'],
  ['assignmentScoreThreshold', 'Assignment score threshold', 'number', '0.8'],
  ['device', 'Device', 'text', 'cpu'],
  ['experimentName', 'Experiment', 'text', 'exp_finetune'],
  ['ckptId', 'Resume checkpoint ID (optional)', 'text', '']
];
function buildArgs(config, dryRun = false) {
  const args = ['-m', 'clefts.cli', 'train', 'fragment-tree-finetune'];
  for (const [key] of fields) {
    const value = String(config[key] ?? '').trim();
    if (key === 'ckptId' && !value) continue;
    args.push('--' + key.replace(/[A-Z]/g, c => '-' + c.toLowerCase()), value);
  }
  if (dryRun) args.push('--dry-run');
  return args;
}
function shellDisplay(program, args) {
  return [program, ...args].map(v => /^[A-Za-z0-9_./:=,-]+$/.test(v) ? v : "'" + v.replace(/'/g, "'\\''") + "'").join(' ');
}
function attach(panel, context, projectRoot, output) {
  let child;
  panel.onDidDispose(() => { if (child) child.kill('SIGTERM'); });
  panel.webview.onDidReceiveMessage(async message => {
    if (!['fineTunePick', 'fineTuneCopy', 'fineTuneRun', 'fineTuneStop'].includes(message.type)) return;
    if (message.type === 'fineTuneRun' && child) return;
    const post = data => panel.webview.postMessage({ type: 'fineTuneStatus', ...data });
    try {
      if (message.type === 'fineTuneStop') { if (child) child.kill('SIGTERM'); return; }
      if (message.type === 'fineTunePick') {
        const field = fields.find(([key]) => key === message.field);
        if (!field || !['file', 'folder'].includes(field[2])) return;
        const folder = field[2] === 'folder';
        const selected = await vscode.window.showOpenDialog({ canSelectMany: false, canSelectFiles: !folder,
          canSelectFolders: folder, ...(folder ? {} : { filters: message.field === 'checkpoint' ? { 'PyTorch checkpoint': ['pt'] } : { 'Cleavage Pattern Set': ['clevageset.json', 'json'] } }) });
        if (selected?.[0]) panel.webview.postMessage({ type: 'fineTunePicked', field: message.field, value: selected[0].fsPath });
        return;
      }
      const root = projectRoot(context);
      const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
      const args = buildArgs(message.config, !!message.dryRun);
      const command = shellDisplay(python, args);
      if (message.type === 'fineTuneCopy') {
        await vscode.env.clipboard.writeText(command);
        post({ text: 'Command copied. Run from: ' + root, command }); return;
      }
      if (child) return;
      output.show(true); output.appendLine('$ ' + command);
      post({ running: true, clear: true, text: message.dryRun ? 'Validating…' : 'Fine-tuning…', command });
      child = spawn(python, args, { cwd: root, env: { ...process.env, PYTHONUNBUFFERED: '1',
        PYTHONPATH: [root, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) } });
      child.stdin.on('error', () => {}); child.stdin.end();
      const stream = data => { const log = data.toString(); output.append(log); post({ log }); };
      child.stdout.on('data', stream); child.stderr.on('data', stream);
      await new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', (code, signal) => {
          if (signal) reject(new Error('Fine-tuning cancelled.'));
          else if (code !== 0) reject(new Error(`CLI failed (exit ${code}). See output for details.`));
          else resolve();
        });
      });
      post({ running: false, text: message.dryRun ? 'Validation completed.' : 'Fine-tuning completed.' });
    } catch (error) {
      post({ running: false, text: error.message });
    } finally {
      if (message.type === 'fineTuneRun') child = undefined;
    }
  });
}
function html() {
  return `<div id="fineTuneApp" hidden><section><h2>Fragment Tree Fine-tuning</h2>
  <p>MolEncoder and all existing parameters stay frozen. Train only added low-rank nodes and new cleavage-category embeddings.</p>
  <p>Choose a complete set containing the old patterns plus new patterns. First regenerate both data splits with this set in Data Preparation. Keep other Fragmenter settings unchanged.</p>
  <form id="fineTuneForm"><div class="grid">${fields.map(([key, label, kind, value = '']) =>
    `<label>${label}<input name="${key}" type="${kind === 'number' ? 'number' : 'text'}" value="${value}" ${kind === 'number' ? 'step="any"' : ''} ${key === 'ckptId' ? '' : 'required'}>${['file','folder'].includes(kind) ? `<button type="button" data-finetune-pick="${key}">Browse…</button>` : ''}</label>`).join('')}</div>
  <div class="actions"><button type="button" id="fineTuneCopy">Copy Command</button><button type="button" id="fineTuneValidate">Validate only</button><button type="button" id="fineTuneStop" disabled>Stop</button><button type="submit" class="primary">Run Fine-tuning CLI</button></div></form>
  <p id="fineTuneStatus" role="status">Ready</p><pre id="fineTuneCommand" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre>
  <pre id="fineTuneLog" style="max-height:400px;overflow:auto;white-space:pre-wrap"></pre></section></div>`;
}
function script() { return `(${client.toString()})();`; }
function client() {
  const el = id => document.getElementById(id), form = el('fineTuneForm');
  const config = () => Object.fromEntries([...form.elements].filter(e => e.name).map(e => [e.name, e.value]));
  function send(type, dryRun = false) { if (form.reportValidity()) vscode.postMessage({ type, config: config(), dryRun }); }
  form.onsubmit = event => { event.preventDefault(); send('fineTuneRun'); };
  el('fineTuneCopy').onclick = () => send('fineTuneCopy');
  el('fineTuneValidate').onclick = () => send('fineTuneRun', true);
  el('fineTuneStop').onclick = () => vscode.postMessage({ type: 'fineTuneStop' });
  form.querySelectorAll('[data-finetune-pick]').forEach(button => button.onclick = () => vscode.postMessage({ type: 'fineTunePick', field: button.dataset.finetunePick }));
  window.addEventListener('message', event => {
    const m = event.data;
    if (m.type === 'fineTunePicked' && form.elements[m.field]) form.elements[m.field].value = m.value;
    if (m.type !== 'fineTuneStatus') return;
    if (m.running !== undefined) {
      for (const element of form.elements) element.disabled = m.running;
      el('fineTuneStop').disabled = !m.running;
    }
    if (m.text) el('fineTuneStatus').textContent = m.text;
    if (m.command) el('fineTuneCommand').textContent = m.command;
    if (m.clear) el('fineTuneLog').textContent = '';
    if (m.log) el('fineTuneLog').textContent = (el('fineTuneLog').textContent + m.log).slice(-20000);
  });
}
module.exports = { attach, html, script, buildArgs, shellDisplay, fields };
