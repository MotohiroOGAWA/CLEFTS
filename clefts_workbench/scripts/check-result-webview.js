const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, isMain);
};

const { resultHtml, summarizeStructureManifests } = require('../src/extension');

const statistics = summarizeStructureManifests([
  { directory: 'train_structures', rows: [
    { status: 'ok', num_input_records: '4', num_valid_samples: '3' },
    { status: 'error', num_input_records: '2', num_valid_samples: '' },
    { status: 'no_valid_samples', record_indexes: '[0,1,2]', sample_indexes: '[-1,-1,-1]' }
  ] },
  { directory: 'validation_structures', rows: [
    { status: 'ok', num_input_records: '2', num_valid_samples: '2' }
  ] }
]);
if (statistics.total.totalCompounds !== 4) throw new Error('Total compound statistics are incorrect.');
if (statistics.total.validCompounds !== 2) throw new Error('Valid compound statistics are incorrect.');
if (statistics.total.totalSpectra !== 11) throw new Error('Total spectrum statistics are incorrect.');
if (statistics.total.validSpectra !== 5) throw new Error('Valid spectrum statistics are incorrect.');
if (statistics.total.invalidCompounds !== 2) throw new Error('Rejected compound statistics are incorrect.');
if (statistics.total.invalidSpectra !== 6) throw new Error('Rejected spectrum statistics are incorrect.');

(async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'clefts-webview-check-'));
  const resultFile = path.join(directory, 'fragment-tree.pft');
  fs.writeFileSync(resultFile, JSON.stringify({ status: 'completed', command: [] }));
  const structureDir = path.join(directory, 'train_structures');
  fs.mkdirSync(structureDir);
  fs.writeFileSync(path.join(structureDir, 'manifest.tsv'), 'file\tstatus\tnum_input_records\tnum_valid_samples\ncompound.preft.pt\tok\t4\t3\nfailed.preft.pt\terror\t2\t\n');
  const html = await resultHtml(resultFile);
  for (const text of ['Dataset Statistics', 'Total compounds', 'Valid compounds', 'Rejected compounds', 'Total spectra', 'Valid spectra', 'Rejected spectra', 'Compound validity', 'Spectrum validity']) {
    if (!html.includes(text)) throw new Error(`Result viewer is missing ${text}.`);
  }
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);
  if (!scripts.length) throw new Error('Result viewer contains no script.');
  for (const source of scripts) new Function(source);
  fs.rmSync(directory, { recursive: true, force: true });
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
