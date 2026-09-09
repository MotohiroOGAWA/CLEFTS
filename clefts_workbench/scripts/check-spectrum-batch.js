const assert = require('assert/strict');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function(request, parent, main) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, main);
};
const feature = require('../src/features/spectrum-prediction/editor');

const args = feature.buildBatchArgs({
  input: '/data/source database.msds', outputDir: '/data/prediction output', outputName: 'result.msds',
  modelPath: '/models/model.pt', device: 'cuda', db: 'MoNA', batchSize: 16,
  precomputeWorkers: 6, keepTemp: true,
  specIdColumn: 'SpecID', smilesColumn: 'SMILES', precursorMzColumn: 'PrecursorMZ',
  adductTypeColumn: 'AdductType', collisionEnergyColumn: 'CollisionEnergy',
  instrumentColumn: 'InstrumentType', overwrite: true
});
assert.deepEqual(args.slice(0, 2), ['-m', 'clefts.ml.specgen.predict_spectrum']);
assert.deepEqual(args.slice(args.indexOf('--db'), args.indexOf('--db') + 2), ['--db', 'MoNA']);
assert.deepEqual(args.slice(args.indexOf('--batch-size'), args.indexOf('--batch-size') + 2), ['--batch-size', '16']);
assert.deepEqual(args.slice(args.indexOf('--output-dir'), args.indexOf('--output-dir') + 2), ['--output-dir', '/data/prediction output']);
assert.deepEqual(args.slice(args.indexOf('--output-name'), args.indexOf('--output-name') + 2), ['--output-name', 'result.msds']);
assert.deepEqual(args.slice(args.indexOf('--precompute-workers'), args.indexOf('--precompute-workers') + 2), ['--precompute-workers', '6']);
assert(args.includes('--overwrite'));
assert(args.includes('--keep-temp'));
assert(args.includes('--instrument-column'));
assert(feature.shellDisplay('python', args).includes("'/data/source database.msds'"));
assert(feature.html().includes('Predict MSDataset'));
assert(feature.html().includes('predictLoadConfig'));
assert(feature.html().includes('predictSaveConfig'));
assert(feature.html().includes('batchPrecomputeWorkers'));
new Function(feature.script());

const { workbenchHtml } = require('../src/extension');
Module._load = originalLoad;
const page = workbenchHtml({}, '{}', {});
assert(page.includes('id="predictBatchSection"'));
assert(page.includes('batchSpecIdColumn'));
assert(page.includes('batchOutputDir'));
for (const match of page.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
console.log('Batch spectrum prediction CLI and Workbench checks passed.');
