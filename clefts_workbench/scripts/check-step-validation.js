// Current training CLI and Workbench use validationIntervalSteps/validationFraction.
const assert = require('assert/strict');
const Module = require('module');
const originalLoad = Module._load;
Module._load = function(request, parent, main) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, main);
};
const { normalizeTrainingConfig, buildTrainingArgs, workbenchHtml } = require('../src/extension');
const { startTraining } = require('../src/workbench/panel');
Module._load = originalLoad;
(async () => {
  const base = { trainDir: '/train', valDir: '/val', outputDir: '/output', molEncoderCheckpoint: '/model.pt', epochs: 1, batchSize: 1, lr: .001 };
  for (const fraction of [1, .1, .001]) {
    const saved = JSON.parse(JSON.stringify(normalizeTrainingConfig({ ...base, validationIntervalSteps: 10, validationFraction: fraction })));
    const args = buildTrainingArgs(saved);
    assert.equal(args[args.indexOf('--validation-fraction') + 1], String(fraction));
    assert.equal(args[args.indexOf('--validation-interval-steps') + 1], '10');
  }
  for (const fraction of [0, -.1, 1.1, NaN, Infinity, '']) {
    await assert.rejects(startTraining({}, null, null, '/', 'python', { ...base, validationFraction: fraction }, buildTrainingArgs), /[Vv]alidation/);
  }
  const html = workbenchHtml({}, base);
  assert(html.includes('name="validationFraction"'));
  assert(html.includes('Full validation still runs at each epoch end.'));
  assert(html.includes('Validation generates spectra'));
  for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
  console.log('Current step validation flags, saved configuration, range checks and spectrum validation UI passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
