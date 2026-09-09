const assert = require('assert/strict');
const fs = require('fs');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, isMain);
};
const {
  html, summaryArgs, groupedArgs, evaluationConfigDocument, parseEvaluationConfig,
  evaluationConfigSuffix, ensureEvaluationConfigSuffix
} = require('../src/features/evaluation/editor');
Module._load = originalLoad;

const page = html();
assert.match(page, /id="bins"[^>]*disabled/);
assert.doesNotMatch(page, /Aggregation|Value column/);
assert.match(page, /Plot color/);
assert.match(page, /id="plotType"/);
assert.match(page, /id="groupedPlotType"/);
assert.match(page, /Violin plot/);
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
assert.match(page, /Changes are applied only when Generate plot is pressed/);
assert.match(page, /id="graphOpacity"/);
assert.match(page, /id="xLabelSize"/);
assert.match(page, /id="yLabelSize"/);
assert.match(page, /id="xAxisTitleSize"/);
assert.match(page, /id="yAxisTitleSize"/);
assert.match(page, /id="title" placeholder=/);
assert.match(page, /id="titleSize"/);
assert.match(page, /id="groupedGraphOpacity"/);
assert.match(page, /id="groupedXLabelSize"/);
assert.match(page, /id="groupedYLabelSize"/);
assert.match(page, /id="groupedXAxisTitleSize"/);
assert.match(page, /id="groupedYAxisTitleSize"/);
assert.match(page, /id="groupedTitleSize"/);
assert.match(page, /let groups = \[\{ id: 'g1', name: '' \}\]/);
assert.match(page, /id="input" data-path-kind="file"/);
assert.match(page, /data-result-path data-path-kind="file"/);
for (const match of page.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);

const compare = groupedArgs({
  width: 800, height: 500, transparent: false, backgroundColor: '#ffffff',
  groupGap: 72, seriesGap: 5, boxWidth: 30,
  xAxisTitleSize: 18, yAxisTitleSize: 19,
  groups: [{ id: 'g1', name: 'MoNA' }],
  series: [{ id: 's1', name: 'FIORA', color: '#123456' }], entries: []
}, '/tmp/result.svg');
assert.equal(compare[0], 'compare');
assert.ok(compare.includes('--request-json'));
assert.ok(compare.includes('--opaque'));
assert.ok(compare.includes('/tmp/result.svg'));
for (const option of ['--graph-opacity', '--x-label-size', '--y-label-size', '--x-axis-title-size', '--y-axis-title-size', '--title-size']) assert.ok(compare.includes(option));
assert.deepEqual(compare.slice(compare.indexOf('--x-axis-title-size'), compare.indexOf('--x-axis-title-size') + 2), ['--x-axis-title-size', '18']);
assert.deepEqual(compare.slice(compare.indexOf('--y-axis-title-size'), compare.indexOf('--y-axis-title-size') + 2), ['--y-axis-title-size', '19']);
assert.deepEqual(compare.slice(compare.indexOf('--plot-type'), compare.indexOf('--plot-type') + 2), ['--plot-type', 'box']);
const configDocument = evaluationConfigDocument('column', { input: '/tmp/results.mssim' });
assert.equal(configDocument.schema, 'clefts.evaluation.config');
assert.deepEqual(parseEvaluationConfig(JSON.parse(JSON.stringify(configDocument)), 'column'), configDocument.config);
assert.throws(() => parseEvaluationConfig(configDocument, 'grouped'), /Expected a grouped/);
assert.throws(() => parseEvaluationConfig({ schemaVersion: 1 }, 'column'), /not a supported/);
const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 2);
assert.equal(evaluationConfigSuffix('column'), '.evalcol.json');
assert.equal(evaluationConfigSuffix('grouped'), '.evalgroup.json');
assert.equal(ensureEvaluationConfigSuffix('/tmp/result.json', 'column'), '/tmp/result.evalcol.json');
assert.equal(ensureEvaluationConfigSuffix('/tmp/result.evalgroup.json', 'grouped'), '/tmp/result.evalgroup.json');
assert.equal(ensureEvaluationConfigSuffix('/tmp/result.evalgroup.json', 'column'), '/tmp/result.evalcol.json');
assert.match(page, /evaluationConfigOpened/);
assert.match(page, /evaluationReady/);
const manifest = require('../package.json');
const evaluationEditor = manifest.contributes.customEditors.find(item => item.viewType === 'clefts.evaluationConfigEditor');
assert.deepEqual(evaluationEditor.selector.map(item => item.filenamePattern), ['*.evalcol.json', '*.evalgroup.json']);
assert.match(fs.readFileSync(require.resolve('../src/extension'), 'utf8'), /evaluation\.register\(context, output, projectRoot\)/);

const args = summaryArgs({
  input: '/tmp/results.mssim', groupColumn: 'AdductType', mode: 'categorical',
  metadata: '/tmp/source.msds', joinColumn: 'SpecID',
  title: 'Adduct evaluation', width: 500, height: 300, color: '#123456',
  xAxisTitleSize: 20, yAxisTitleSize: 21,
  transparent: true, include: [], order: []
});
assert.ok(args.includes('--include'));
assert.ok(args.includes('[]'));
assert.ok(!args.includes('--opaque'));
assert.deepEqual(args.slice(args.indexOf('--metadata'), args.indexOf('--metadata') + 2), ['--metadata', '/tmp/source.msds']);
assert.deepEqual(args.slice(args.indexOf('--join-column'), args.indexOf('--join-column') + 2), ['--join-column', 'SpecID']);
assert.ok(!args.includes('--bins'));
for (const option of ['--graph-opacity', '--x-label-size', '--y-label-size', '--x-axis-title-size', '--y-axis-title-size', '--title-size', '--title']) assert.ok(args.includes(option));
assert.deepEqual(args.slice(args.indexOf('--title'), args.indexOf('--title') + 2), ['--title', 'Adduct evaluation']);
assert.deepEqual(args.slice(args.indexOf('--x-axis-title-size'), args.indexOf('--x-axis-title-size') + 2), ['--x-axis-title-size', '20']);
assert.deepEqual(args.slice(args.indexOf('--y-axis-title-size'), args.indexOf('--y-axis-title-size') + 2), ['--y-axis-title-size', '21']);
assert.deepEqual(args.slice(args.indexOf('--plot-type'), args.indexOf('--plot-type') + 2), ['--plot-type', 'box']);
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
