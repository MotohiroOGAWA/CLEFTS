const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, isMain);
};

const { resultHtml } = require('../src/extension');

(async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'clefts-webview-check-'));
  const resultFile = path.join(directory, 'fragment-tree.pft');
  fs.writeFileSync(resultFile, JSON.stringify({ status: 'completed', command: [] }));
  const html = await resultHtml(resultFile);
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);
  if (!scripts.length) throw new Error('Result viewer contains no script.');
  for (const source of scripts) new Function(source);
  fs.rmSync(directory, { recursive: true, force: true });
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
