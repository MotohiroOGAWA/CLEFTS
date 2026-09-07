const assert = require('assert');
const Module = require('module');
const originalLoad = Module._load;
Module._load = function(request, parent, main) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, main);
};
const { normalizeTrainingConfig, buildTrainingArgs, workbenchHtml } = require('../src/extension');
Module._load = originalLoad;
const base = { trainDir: '/train', valDir: '/val', outputDir: '/output', molEncoderCheckpoint: '/model.pt' };
const defaults = normalizeTrainingConfig(base);
assert.strictEqual(defaults.stepValidationFraction, 1);
for (const fraction of [1, 0.1, 0.001]) {
  const saved = JSON.parse(JSON.stringify(normalizeTrainingConfig({ ...base, stepValidationFraction: fraction })));
  const args = buildTrainingArgs(saved);
  assert.strictEqual(args[args.indexOf('--step-validation-fraction') + 1], String(fraction));
}
for (const fraction of [0, -0.1, 1.1, NaN, Infinity, '']) {
  assert.throws(() => buildTrainingArgs({ ...base, stepValidationFraction: fraction }), /Step validation fraction/);
}
const html = workbenchHtml({}, '{}', defaults);
assert(html.includes('name="stepValidationFraction" type="number" step="any"'));
assert(html.includes('Step validation fraction'));
assert(html.includes('Epoch-end validation always evaluates all compounds.'));
for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new Function(match[1]);
console.log('Step validation defaults, command, saved config, validation, and form checks passed.');
