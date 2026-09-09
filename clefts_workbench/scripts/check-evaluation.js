const assert = require('assert/strict');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, isMain);
};
const {
  html, summaryArgs, groupedArgs, evaluationConfigDocument, parseEvaluationConfig
} = require('../src/features/evaluation/editor');
Module._load = originalLoad;

const page = html();
assert.match(page, /id="bins"[^>]*disabled/);
assert.doesNotMatch(page, /Aggregation|Value column/);
assert.match(page, /Box color/);
assert.match(page, /data-eval-tab="column"/);
assert.match(page, /data-eval-tab="grouped"/);
assert.match(page, /id="groupedAddGroup"/);
assert.match(page, /id="groupedAddSeries"/);
assert.match(page, /data-no-data/);
assert.match(page, /Transparent background/);
assert.match(page, /id="loadColumnConfig"/);
assert.match(page, /id="saveColumnConfig"/);
assert.match(page, /id="groupedLoadConfig"/);
assert.match(page, /id="groupedSaveConfig"/);
assert.match(page, /id="groupedGroupGap"/);
assert.match(page, /id="groupedSeriesGap"/);
assert.match(page, /id="groupedBoxWidth"/);
assert.match(page, /Collision Energy parser/);
assert.match(page, /SMILES chemical information/);
assert.match(page, /HeavyAtomCount/);
assert.match(page, /Changes are applied only when Generate box plot is pressed/);
for (const match of page.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);

const compare = groupedArgs({
  width: 800, height: 500, transparent: false, backgroundColor: '#ffffff',
  groupGap: 72, seriesGap: 5, boxWidth: 30,
  groups: [{ id: 'g1', name: 'MoNA' }],
  series: [{ id: 's1', name: 'FIORA', color: '#123456' }], entries: []
}, '/tmp/result.svg');
assert.equal(compare[0], 'compare');
assert.ok(compare.includes('--request-json'));
assert.ok(compare.includes('--opaque'));
assert.ok(compare.includes('/tmp/result.svg'));
const configDocument = evaluationConfigDocument('column', { input: '/tmp/results.mssim' });
assert.equal(configDocument.schema, 'clefts.evaluation.config');
assert.deepEqual(parseEvaluationConfig(JSON.parse(JSON.stringify(configDocument)), 'column'), configDocument.config);
assert.throws(() => parseEvaluationConfig(configDocument, 'grouped'), /Expected a grouped/);
assert.throws(() => parseEvaluationConfig({ schemaVersion: 1 }, 'column'), /not a supported/);
const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 2);

const args = summaryArgs({
  input: '/tmp/results.mssim', groupColumn: 'AdductType', mode: 'categorical',
  metadata: '/tmp/source.msds', joinColumn: 'SpecID',
  width: 500, height: 300, color: '#123456',
  transparent: true, include: [], order: []
});
assert.ok(args.includes('--include'));
assert.ok(args.includes('[]'));
assert.ok(!args.includes('--opaque'));
assert.deepEqual(args.slice(args.indexOf('--metadata'), args.indexOf('--metadata') + 2), ['--metadata', '/tmp/source.msds']);
assert.deepEqual(args.slice(args.indexOf('--join-column'), args.indexOf('--join-column') + 2), ['--join-column', 'SpecID']);
assert.ok(!args.includes('--bins'));
assert.deepEqual(args.slice(args.indexOf('--transform'), args.indexOf('--transform') + 2), ['--transform', 'none']);
const transformed = summaryArgs({
  input: '/tmp/results.mssim', groupColumn: 'CollisionEnergy', mode: 'numeric',
  transform: 'collision-energy', precursorMzColumn: 'PrecursorMZ',
  instrumentColumn: 'InstrumentType', bins: '0,10,20'
});
assert.ok(transformed.includes('--chemical-descriptor') === false);
assert.deepEqual(transformed.slice(transformed.indexOf('--transform'), transformed.indexOf('--transform') + 2), ['--transform', 'collision-energy']);
assert.ok(transformed.includes('--bins'));
console.log('Evaluation webview and CLI argument checks passed.');
