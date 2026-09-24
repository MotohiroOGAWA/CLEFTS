// Marks files that open a CLEFTS Workbench viewer with a small Explorer/tab
// badge (File Decoration API), without touching any other file or folder
// icon. Unlike a File Icon Theme, this never replaces the user's own theme.
const vscode = require('vscode');
const path = require('path');

// Mirrors package.json's customEditors selectors: every extension listed
// there opens a CLEFTS viewer, so every one of them gets the same badge.
const CLEFTS_EXTENSIONS = new Set([
  'pft', 'pft.json', 'clefts-result',
  'evalcol.json', 'evalgroup.json',
  'cleavage.json', 'clevage.json', 'clevageset.json',
  'adduct.json', 'adductset.json',
]);

function cleftsExtension(uri) {
  const name = path.basename(uri.fsPath).toLowerCase();
  const parts = name.split('.');
  // Longest suffix first, so "x.pft.json" matches "pft.json" rather than
  // stopping at the unrelated bare "json" (which isn't in the set anyway).
  for (let i = 1; i < parts.length; i++) {
    const candidate = parts.slice(i).join('.');
    if (CLEFTS_EXTENSIONS.has(candidate)) return candidate;
  }
  return null;
}

function register(context) {
  context.subscriptions.push(vscode.window.registerFileDecorationProvider({
    provideFileDecoration(uri) {
      if (!cleftsExtension(uri)) return undefined;
      return { badge: 'C', color: new vscode.ThemeColor('clefts.fileBadge'), tooltip: 'Opens in a CLEFTS Workbench viewer' };
    },
  }));
}

module.exports = { register, cleftsExtension, CLEFTS_EXTENSIONS };
