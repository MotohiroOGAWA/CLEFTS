const assert = require('assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { histogram, readScores, distributionHtml } = require('../src/features/fragment-tree-result/score-distribution');
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
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'clefts-score-check-'));
  try {
    fs.mkdirSync(path.join(directory, 'train_structures'));
    fs.writeFileSync(path.join(directory, 'train_structures', 'assignment_scores.tsv'), '\uFEFFid\tassignment_score\r\na\t0\r\nb\t1\r\nc\t\r\nd\tNaN\r\ne\t1.1\r\nf\t-1\r\n');
    const datasets = await readScores(directory);
    assert.deepEqual(datasets[0].scores, [0, 1]);
    assert.equal(datasets[0].skipped, 4);
    const html = distributionHtml(datasets);
    assert.match(html, /id="scoreImageWidth"[^>]*value="1200"/);
    assert.match(html, /id="scoreImageHeight"[^>]*value="640"/);
    assert.match(html, /id="scoreTransparent"[^>]*checked/);
    assert.match(html, /ctx\.clearRect/);
    for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
    fs.writeFileSync(path.join(directory, 'assignment_scores.tsv'), 'id\tother\na\t1\n');
    assert.match((await readScores(directory))[0].error, /column is missing/);
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
  console.log('Assignment score distribution checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
