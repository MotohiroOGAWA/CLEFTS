const assert = require('assert/strict');
const { pathFromDroppedValue, pathFromDataTransfer, script, css } = require('../src/webview-path-drop');

assert.equal(pathFromDroppedValue('file:///workspaces/CLEFTS/a%20file.msds'), '/workspaces/CLEFTS/a file.msds');
assert.equal(pathFromDroppedValue('file:///C:/Data/result.mssim'), 'C:/Data/result.mssim');
assert.equal(pathFromDroppedValue('vscode-remote://dev-container%2Babc/workspaces/CLEFTS/result.mssim'), '/workspaces/CLEFTS/result.mssim');
assert.equal(pathFromDroppedValue('/workspaces/CLEFTS/result.mssim'), '/workspaces/CLEFTS/result.mssim');
assert.equal(pathFromDroppedValue('result.mssim'), '');
assert.equal(pathFromDataTransfer({
  files: [],
  getData: type => type === 'text/uri-list' ? '# files\nfile:///tmp/first.mssim\nfile:///tmp/second.mssim' : ''
}), '/tmp/first.mssim');
assert.equal(pathFromDataTransfer({ files: [{ path: '/tmp/from-os.msds' }], getData: () => '' }), '/tmp/from-os.msds');
assert.match(script(), /data-path-kind/);
assert.match(css(), /path-drop-active/);
new Function(script());
console.log('Path drag-and-drop checks passed.');
