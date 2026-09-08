const assert = require('assert/strict');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return {};
  return originalLoad.call(this, request, parent, isMain);
};
const { html, summaryArgs } = require('../src/features/evaluation/editor');
Module._load = originalLoad;

const page = html();
assert.match(page, /id="bins"[^>]*disabled/);
assert.doesNotMatch(page, /Aggregation|Value column/);
assert.match(page, /Box color/);
const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1);
new Function(scripts[0][1]);

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
console.log('Evaluation webview and CLI argument checks passed.');
