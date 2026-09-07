const { spawn } = require('child_process');
const path = require('path');
const vscode = require('vscode');

function buildArgs(payload) {
  return ['-m', 'clefts.cli', 'molecule', 'smarts-search',
    '--input', String(payload.file || '').trim(),
    '--smarts', String(payload.smarts || '').trim(),
    '--smiles-column', String(payload.column || 'SMILES').trim(), '--include-compounds'];
}

function shellDisplay(program, args) {
  return [program, ...args].map(value => /^[A-Za-z0-9_./:=,-]+$/.test(value)
    ? value : "'" + value.replace(/'/g, "'\\''") + "'").join(' ');
}

function attach(panel, context, projectRoot) {
  let child;
  panel.onDidDispose(() => { if (child) child.kill(); });
  panel.webview.onDidReceiveMessage(async message => {
    if (message.type === 'pickSmartsDataset') {
      const picked = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { MSDataset: ['msds'] } });
      if (picked?.[0]) panel.webview.postMessage({ type: 'smartsDataset', file: picked[0].fsPath });
    }
    if (message.type === 'cancelSmartsSearch') { if (child) child.kill(); }
    if (message.type === 'copySmartsCommand') {
      try {
        const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
        const command = shellDisplay(python, buildArgs(message.payload));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'smartsCommandCopied', command, root: projectRoot(context) });
      } catch (error) {
        panel.webview.postMessage({ type: 'smartsCommandCopied', error: error.message });
      }
      return;
    }
    if (message.type !== 'smartsSearch' || child) return;
    try {
      const root = projectRoot(context);
      const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
      child = spawn(python, buildArgs(message.payload), {
        cwd: root, env: { ...process.env, PYTHONPATH: [root, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
      });
      let stdout = '', stderr = '';
      child.stdout.on('data', data => { stdout += data; });
      child.stderr.on('data', data => { stderr = (stderr + data).slice(-8000); });
      child.stdin.on('error', () => {});
      const response = await new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', (code, signal) => {
          if (signal) { reject(new Error('Search cancelled.')); return; }
          if (code !== 0) { reject(new Error(stderr || `Search failed (exit ${code}).`)); return; }
          try { resolve(JSON.parse(stdout)); }
          catch (_) { reject(new Error(stderr || `Search failed (exit ${code}).`)); }
        });
        child.stdin.end();
      });
      panel.webview.postMessage({ type: 'smartsSearchResult', ok: true, result: response });
    } catch (error) {
      panel.webview.postMessage({ type: 'smartsSearchResult', ok: false, error: error.message });
    } finally { child = undefined; }
  });
}

function html() {
  return `<div id="smartsApp" hidden><section><h2>SMARTS Search</h2>
  <p>Count SMILES containing a SMARTS substructure. Each record is counted once, even if it contains multiple matches.</p>
  <form id="smartsForm"><label>MSDataset (.msds)<input id="smartsFile" required></label><button type="button" id="smartsBrowse">Browse…</button>
  <label>SMILES column<input id="smartsColumn" value="SMILES" required></label>
  <label>SMARTS<input id="smartsQuery" placeholder="e.g. c1ccccc1" required></label>
  <p class="muted">Percentages use valid SMILES as the denominator. Unique molecules are deduplicated using canonical isomeric SMILES. Stereochemistry is not required for substructure matching.</p>
  <button type="button" id="smartsCopy">Copy Command</button> <button class="primary" id="smartsRun">Run CLI</button> <button type="button" id="smartsCancel" disabled>Cancel</button></form>
  <p id="smartsStatus" role="status"></p><pre id="smartsCommand" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre><div id="smartsResult"></div></section></div>`;
}

function script() { return `(${client.toString()})();`; }
function client() {
  const el = id => document.getElementById(id);
  const form = el('smartsForm');
  function busy(value) {
    for (const input of form.querySelectorAll('input,button')) input.disabled = value;
    el('smartsCancel').disabled = !value;
  }
  el('smartsBrowse').onclick = () => vscode.postMessage({ type: 'pickSmartsDataset' });
  el('smartsCancel').onclick = () => vscode.postMessage({ type: 'cancelSmartsSearch' });
  const payload = () => ({ file: el('smartsFile').value.trim(), column: el('smartsColumn').value.trim(), smarts: el('smartsQuery').value.trim() });
  el('smartsCopy').onclick = () => {
    if (form.reportValidity()) vscode.postMessage({ type: 'copySmartsCommand', payload: payload() });
  };
  form.onsubmit = event => {
    event.preventDefault();
    busy(true); el('smartsResult').textContent = ''; el('smartsStatus').textContent = 'Searching…';
    vscode.postMessage({ type: 'smartsSearch', payload: payload() });
  };
  window.addEventListener('message', event => {
    const m = event.data;
    if (m.type === 'smartsDataset') el('smartsFile').value = m.file;
    if (m.type === 'smartsCommandCopied') {
      el('smartsStatus').textContent = m.error || 'Command copied. Run from: ' + m.root;
      el('smartsCommand').textContent = m.command || '';
    }
    if (m.type !== 'smartsSearchResult') return;
    busy(false);
    el('smartsStatus').textContent = m.ok ? 'Search completed.' : m.error;
    if (!m.ok) return;
    const r = m.result, percent = v => v === null ? 'N/A' : v.toFixed(2) + '%';
    const table = document.createElement('table');
    for (const values of [['Counting basis', 'Detected', 'Valid total', 'Percentage'],
      ['Records (including duplicates)', r.matched, r.valid, percent(r.percent)],
      ['Unique molecules', r.uniqueMatched, r.unique, percent(r.uniquePercent)]]) {
      const row = table.insertRow();
      for (const value of values) row.insertCell().textContent = value;
    }
    const detail = document.createElement('p');
    detail.textContent = 'Total records: ' + r.total + ' · Missing SMILES: ' + r.missing + ' · Invalid SMILES: ' + r.invalid;
    el('smartsResult').replaceChildren(table, detail);
    for (const matched of [true, false]) {
      const compounds = (r.compounds || []).filter(compound => compound.matched === matched);
      const section = document.createElement('details');
      section.open = true;
      const title = document.createElement('summary');
      title.textContent = (matched ? 'Detected compounds' : 'Not detected compounds') + ' (' + compounds.length + ')';
      const filter = document.createElement('input');
      filter.placeholder = 'Filter by SMILES';
      filter.setAttribute('aria-label', title.textContent + ': filter by SMILES');
      const list = document.createElement('table');
      list.style.tableLayout = 'fixed';
      list.style.width = '100%';
      const header = list.createTHead().insertRow();
      for (const label of ['Canonical SMILES', 'Records']) {
        const cell = document.createElement('th'); cell.textContent = label; header.append(cell);
      }
      const body = list.createTBody();
      const previous = document.createElement('button'), next = document.createElement('button');
      previous.textContent = 'Previous'; next.textContent = 'Next';
      previous.type = next.type = 'button';
      const pageLabel = document.createElement('span');
      pageLabel.setAttribute('aria-live', 'polite');
      let page = 0, filtered = compounds;
      function renderPage() {
        body.replaceChildren();
        for (const compound of filtered.slice(page * 50, (page + 1) * 50)) {
          const row = body.insertRow();
          const smiles = row.insertCell();
          smiles.textContent = compound.smiles; smiles.style.overflowWrap = 'anywhere';
          row.insertCell().textContent = compound.records;
        }
        const pages = Math.max(1, Math.ceil(filtered.length / 50));
        pageLabel.textContent = ' Page ' + (page + 1) + ' / ' + pages + ' · ' + filtered.length + ' compounds ';
        previous.disabled = page === 0; next.disabled = page + 1 >= pages;
      }
      filter.oninput = () => { filtered = compounds.filter(c => c.smiles.includes(filter.value.trim())); page = 0; renderPage(); };
      previous.onclick = () => { page--; renderPage(); };
      next.onclick = () => { page++; renderPage(); };
      section.append(title, filter, list, previous, pageLabel, next);
      el('smartsResult').append(section);
      renderPage();
    }
  });
}
module.exports = { attach, html, script, buildArgs, shellDisplay };
