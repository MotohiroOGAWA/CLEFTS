const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const Module = require('module');

const load = Module._load;
let registered;
const vscodeMock = {
  ThemeColor: class ThemeColor { constructor(id) { this.id = id; } },
  window: { registerFileDecorationProvider: provider => { registered = provider; return { dispose() {} }; } },
};
Module._load = function (name, parent, main) { if (name === 'vscode') return vscodeMock; return load.call(this, name, parent, main); };

const fileBadges = require('../src/workbench/file-badges');
fileBadges.register({ subscriptions: [] });
assert.ok(registered, 'register() must call vscode.window.registerFileDecorationProvider');

const uri = value => ({ fsPath: value });
const decorate = value => registered.provideFileDecoration(uri(value));

// Every extension a customEditor actually listens on must get the badge --
// mirrors package.json's customEditors selectors one to one.
const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'package.json'), 'utf8'));
for (const editor of pkg.contributes.customEditors) {
  for (const { filenamePattern } of editor.selector) {
    const extension = filenamePattern.replace(/^\*\./, '');
    assert.ok(fileBadges.CLEFTS_EXTENSIONS.has(extension),
      `${editor.viewType}'s ${filenamePattern} has no matching file-badge extension`);
    const decoration = decorate(`sample.${extension}`);
    assert.ok(decoration, `sample.${extension} should be decorated`);
    assert.equal(decoration.badge, 'C');
    assert.ok(decoration.color instanceof vscodeMock.ThemeColor);
    assert.equal(decoration.color.id, 'clefts.fileBadge');
  }
}

// A compound extension must not leak onto unrelated files sharing its tail.
assert.equal(decorate('notes.json'), undefined, 'plain .json must be untouched');
assert.equal(decorate('script.py'), undefined, 'plain .py must be untouched');
assert.equal(decorate('README.md'), undefined, 'plain .md must be untouched');
assert.ok(decorate('result.pft.json'), 'compound .pft.json must still match');
assert.equal(decorate('result.json'), undefined, 'a bare .json must not match just because "pft.json" is registered');

// The custom color used above must actually be declared, or VS Code falls
// back to an undefined/invisible color.
const declaredColor = pkg.contributes.colors.find(color => color.id === 'clefts.fileBadge');
assert.ok(declaredColor, 'package.json must declare the clefts.fileBadge color');
assert.ok(declaredColor.defaults.dark && declaredColor.defaults.light);

console.log('File badges: decoration provider registration, extension coverage and no-leak checks passed.');
