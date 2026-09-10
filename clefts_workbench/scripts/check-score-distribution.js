const assert = require('assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  histogram, readScores, distributionHtml, normalizeScoreChartSettings,
  SCORE_CHART_SCHEMA, SCORE_CHART_SUFFIX
} = require('../src/features/fragment-tree-result/score-distribution');
(async () => {
  const scores = [0, 0.099999, 0.1, 0.3, 0.6, 0.9, 1];
  const rows = histogram(scores, 0.1);
  assert.equal(rows.length, 10);
  assert.deepEqual(rows.map(row => row.count), [2, 1, 0, 1, 0, 0, 1, 0, 0, 2]);
  assert.equal(rows[9].cumulative, scores.length);
  assert.equal(rows[9].percent, 100);
  assert.equal(histogram(scores, 0.3).length, 4);
  assert.equal(histogram(scores, 0.3)[3].upper, 1);
  assert.equal(histogram(scores, 1)[0].count, scores.length);
  assert.equal(histogram([], 0.1)[9].percent, 0);
  for (const width of [0, NaN, -1, 1.1, 0.0001]) assert.throws(() => histogram(scores, width));
  assert.equal(SCORE_CHART_SUFFIX, '.scorechart.json');
  const settings = normalizeScoreChartSettings({
    chart: { title: { text: 'Comparison', show: false, size: 30, gap: 20 } },
    series: [{ resultPath: '/tmp/run.pft', subset: 'train_structures', label: 'Run A', opacity: 0.4, barWidth: 65 }]
  });
  assert.equal(settings.schema, SCORE_CHART_SCHEMA);
  assert.equal(settings.series[0].opacity, 0.4);
  assert.equal(settings.series[0].barWidth, 65);
  const clamped = normalizeScoreChartSettings({ series: [{ resultPath: '/tmp/run.pft', opacity: 4, barWidth: -2, color: 'bad' }] }).series[0];
  assert.equal(clamped.opacity, 1);
  assert.equal(clamped.barWidth, 5);
  assert.equal(clamped.barColor, '#4285d4');
  assert.equal(clamped.lineColor, '#4285d4');
  const defaults = normalizeScoreChartSettings({ series: [{ resultPath: '/tmp/run.pft', color: '#123456' }] }).series[0];
  assert.equal(defaults.opacity, 1);
  assert.equal(defaults.barWidth, 100);
  assert.equal(defaults.barColor, '#123456');
  assert.equal(defaults.lineColor, '#123456');
  assert.throws(() => normalizeScoreChartSettings({ schema: 'wrong' }), /not a score chart/);
  assert.throws(() => normalizeScoreChartSettings({ schemaVersion: 2 }), /Unsupported/);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'clefts-score-check-'));
  try {
    fs.mkdirSync(path.join(directory, 'train_structures'));
    fs.writeFileSync(path.join(directory, 'train_structures', 'assignment_scores.tsv'), '\uFEFFid\tassignment_score\r\na\t0\r\nb\t1\r\nc\t\r\nd\tNaN\r\ne\t1.1\r\nf\t-1\r\n');
    const resultPath = path.join(directory, 'fragment-tree.pft');
    const datasets = await readScores(directory, { source: 'run-a', resultPath });
    assert.deepEqual(datasets[0].scores, [0, 1]);
    assert.equal(datasets[0].skipped, 4);
    assert.equal(datasets[0].label, 'run-a — train_structures');
    assert.equal(datasets[0].resultPath, resultPath);
    assert.match(datasets[0].id, /fragment-tree\.pft::train_structures$/);
    const html = distributionHtml(datasets);
    assert.match(html, /id="scoreAdd"/);
    assert.match(html, /type: 'addScoreResults'/);
    assert.match(html, /scoreResultsAdded/);
    assert.match(html, /datasets overlaid/);
    assert.match(html, /dataset\\tbin_lower_inclusive/);
    assert.match(html, /dataset\.scoreField/);
    assert.match(html, /barWidth/);
    assert.match(html, /opacity/);
    assert.match(html, /barColor/);
    assert.match(html, /lineColor/);
    assert.match(html, /Bar color/);
    assert.match(html, /Line color/);
    for (const id of [
      'scoreLoadSettings', 'scoreSaveSettings', 'scoreTitleText', 'scoreTitleSize',
      'scoreTitleShow', 'scoreXTitleText', 'scoreLeftTitleText', 'scoreRightTitleText',
      'scoreLegendShow', 'scoreXTicksShow', 'scoreYTicksShow', 'scorePaddingTop',
      'scorePaddingRight', 'scorePaddingBottom', 'scorePaddingLeft', 'scoreBinCount',
      'scoreActionStatus'
    ]) assert.match(html, new RegExp(`id="${id}"`));
    assert.match(html, /id="scoreSaveSettings" type="button"/);
    assert.match(html, /Save cancelled/);
    assert.match(html, /id="scoreBinCount"[^>]*value="10"/);
    assert.match(html, /binCount, binWidth: 1 \/ binCount/);
    assert.match(html, /saveScoreDistributionSettings/);
    assert.match(html, /loadScoreDistributionSettings/);
    assert.match(html, /auto-expanded to protect labels/);
    assert.match(html, /id="scoreImageWidth"[^>]*value="1200"/);
    assert.match(html, /id="scoreImageHeight"[^>]*value="640"/);
    assert.match(html, /id="scoreTransparent"[^>]*checked/);
    assert.match(html, /ctx\.clearRect/);
    for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
    fs.writeFileSync(path.join(directory, 'assignment_scores.tsv'), 'id\tother\na\t1\n');
    assert.match((await readScores(directory))[0].error, /column is missing/);
    const extension = fs.readFileSync(path.join(__dirname, '..', 'src', 'extension.js'), 'utf8');
    assert.match(extension, /title: 'Save Assignment Score Chart Settings'/);
    assert.match(extension, /scoreSettingsSaved', cancelled: true/);
    assert.match(extension, /fs\.promises\.writeFile\(target\.fsPath/);
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
  console.log('Assignment score distribution checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
