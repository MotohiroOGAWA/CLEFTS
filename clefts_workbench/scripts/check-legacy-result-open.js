const assert = require('assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const load = Module._load;
const executed = [];
const vscodeMock = {
  commands: { executeCommand: (...args) => { executed.push(args); return Promise.resolve(); } },
  Uri: { file: fsPath => ({ fsPath }) },
};
Module._load = function (name, parent, main) { if (name === 'vscode') return vscodeMock; return load.call(this, name, parent, main); };

const services = require('../src/workbench/services');

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'clefts-legacy-result-'));
  try {
    // A job recorded before the .pft.json -> .pft rename only has the old
    // file on disk; the "Results" button still guesses the new name, so
    // library/open must fall back to it instead of throwing ENOENT.
    const legacy = path.join(root, 'fragment-tree.pft.json');
    fs.writeFileSync(legacy, JSON.stringify({ status: 'completed' }));

    let received;
    const posted = [];
    const panel = { webview: { onDidReceiveMessage: fn => { received = fn; }, postMessage: m => posted.push(m) } };
    services.attach(panel, {}, () => root);
    await received({ type: 'library/open', path: path.join(root, 'fragment-tree.pft') });

    assert.equal(executed.length, 1);
    assert.equal(executed[0][0], 'vscode.openWith');
    assert.equal(executed[0][1].fsPath, legacy);
    assert.equal(executed[0][2], 'clefts.resultViewer');

    // A genuinely missing file (neither name exists) must still fail loudly.
    executed.length = 0; posted.length = 0;
    await received({ type: 'library/open', path: path.join(root, 'missing.pft') });
    assert.equal(executed.length, 0);
    assert.equal(posted.length, 1);
    assert.equal(posted[0].type, 'data/error');

    console.log('Legacy .pft.json result files still open via library/open checks passed.');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
