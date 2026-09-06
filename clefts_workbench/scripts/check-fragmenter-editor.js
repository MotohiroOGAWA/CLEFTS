const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Module = require('module');
const { createFragmenterEditor } = require('../src/components/fragmenter-editor');
class Node {
  constructor(tag) { this.tag = tag; this.children = []; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
}
global.document = { createElement: tag => new Node(tag) };
const root = new Node('div');
const params = JSON.parse(fs.readFileSync(path.join(__dirname, '../../clefts/domain/fragment/presets/fragmenter_single_bond_pos.json')));
const editor = createFragmenterEditor(root, { value: params, editCleavagePatternSet: (value, apply) => { value.name = 'edited'; apply(value); } });
function find(node, predicate) {
  if (predicate(node)) return node;
  for (const child of node.children) { const found = find(child, predicate); if (found) return found; }
}
function click(text) { const button = find(root, n => n.tag === 'button' && n.textContent === text); assert(button, text); button.onclick(); }
assert.deepStrictEqual(editor.getValue(), params);
click('+ Add AdductType');
assert.equal(editor.getValue().fragment_ion_tree_builder.fragment_ion_adduct_rule_set.adduct_rules.length, 5);
click('Edit Cleavage Pattern Set');
assert.equal(editor.getValue().fragment_ion_tree_builder.cleavage_pattern_set.name, 'edited');
assert.equal(params.fragment_ion_tree_builder.cleavage_pattern_set.name, 'single_bond_cleavage_pattern_set');
const input = find(root, n => n.tag === 'input' && n.value === params.mass_tolerance);
input.value = '0.02Da'; input.oninput();
const saved = JSON.parse(JSON.stringify(editor.getValue())); editor.setValue(saved);
assert.equal(editor.getValue().mass_tolerance, '0.02Da');
const originalLoad = Module._load;
Module._load = function(request, ...args) { return request === 'vscode' ? {} : originalLoad.call(this, request, ...args); };
const filename = path.resolve(__dirname, '../src/extension.js');
const extension = new Module(filename, module); extension.filename = filename; extension.paths = module.paths;
extension._compile(fs.readFileSync(filename, 'utf8') + '\nmodule.exports.check={webviewScript,buildArgs,normalizeConfig};', filename);
const { webviewScript, buildArgs, normalizeConfig } = extension.exports.check;
new Function(webviewScript());
const config = normalizeConfig(JSON.parse(JSON.stringify({ trainInput: 'input file.msds', outputDir: 'out', symbols: ['C'], params: 'obsolete.json', fragmenterParams: saved })));
const args = buildArgs(config);
assert.deepStrictEqual(JSON.parse(args[args.indexOf('--params-json') + 1]), saved);
assert(!('params' in config));
console.log('Fragmenter editing, pattern callback, config round trip and generated Workbench script passed.');
