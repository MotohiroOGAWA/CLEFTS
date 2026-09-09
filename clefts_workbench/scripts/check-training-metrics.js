const assert = require('assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { readRun, groupSeries, script } = require('../src/features/training-metrics/editor');
(async () => {
  new Function('vscode', script());
  const fixture = [];
  for (const split of ['train', 'train_window', 'validation']) {
    for (const adduct of ['[M+H]+', '[M+Na]+']) {
      for (const stat of ['min', 'q1', 'median', 'q3', 'max', 'mean']) {
        fixture.push({ name: `distributions/${split}/peak_recall@by_adduct:${adduct}_${stat}`, points: [[10, .5]] });
      }
    }
  }
  fixture.push({ name: 'distributions/validation/peak_recall@by_ce_range:min-to-q1_mean', points: [[20, .6]] });
  fixture.push({ name: 'distributions/train/rollout_step_0/edge_selection/depth_1/recall_mean', points: [[10, .7]] });
  fixture.push({ name: 'metrics/train_loss', points: [[10, 3]] }, { name: 'metrics/val_loss', points: [[20, 2]] });
  const grouped = groupSeries(fixture);
  const distribution = grouped.find(c => c.name === 'distribution/peak_recall @by_adduct');
  assert.equal(distribution.lanes.length, 6);
  assert.deepEqual(Object.keys(distribution.lanes[0].stats).sort(), ['max', 'median', 'min', 'q1', 'q3']);
  const mean = grouped.find(c => c.name === 'mean/peak_recall @by_adduct');
  assert.equal(mean.lanes.length, 6);
  assert.deepEqual(Object.keys(mean.lanes[0].stats), ['mean']);
  assert.equal(grouped.find(c => c.name === 'mean/peak_recall @by_ce_range').lanes[0].category, 'min-to-q1');
  assert.ok(grouped.some(c => c.name === 'mean/rollout_step_0/edge_selection/depth_1/recall'));
  assert.deepEqual(grouped.find(c => c.name === 'metrics/loss').lanes.map(l => l.split), ['train', 'validation']);
  assert.equal(grouped.flatMap(c => c.sources).length, fixture.length);
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'clefts-metrics-'));
  try {
    await assert.rejects(readRun(dir), /No metrics\.tsv or metric_distributions\.tsv found/);
    await fs.writeFile(path.join(dir, 'metrics.tsv'), 'event\tepoch\tglobal_step\ttrain_loss\tval_loss\ntrain_step\t1\t10\t3\tnan\nvalidation\t1\t10\t3\t2\ntrain_step\t1\t20\tInfinity\t\ntrain_step\t1\t30');
    await fs.writeFile(path.join(dir, 'metric_distributions.tsv'), 'event\tepoch\tglobal_step\tsplit\tmetric\tvalue\r\ntrain_step\t1\t10\ttrain\tloss_q1\t0.5\r\n');
    const result = await readRun(dir);
    assert.deepEqual(result.series.find(s => s.name === 'metrics/val_loss').points, [[10, 2]]);
    assert.deepEqual(result.series.find(s => s.name === 'metrics/train_loss').points, [[10, 3]]);
    assert.deepEqual(result.series.find(s => s.name === 'distributions/train/loss_q1').points, [[10, .5]]);
    assert.equal(result.series.length, 3);
    console.log('Training metrics parsing and webview syntax checks passed.');
  } finally { await fs.rm(dir, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
