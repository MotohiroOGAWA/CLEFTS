const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const cleavageUi = require('./ui');
const { commonCss, formCss } = require('./styles');
const cleavageVisualEditor = require('./visual-editor');
const cleavageReactionPreview = require('./reaction-preview');
const pathDrop = require('../../webview-path-drop');
const host = require('./host');

const VIEW_TYPE = 'clefts.cleavagePatternSetEditor';
const PATTERN_VIEW_TYPE = 'clefts.cleavagePatternEditor';
// .pft (not .json) so VS Code's contributed language icon actually shows:
// every icon theme already claims plain .json, which always wins over a
// contributed language icon, but no theme claims .pft.
const FILE_SUFFIX = '.clevageset.pft';

function emptyDocument() {
  return {
    cleavage_pattern_set: {
      name: 'new_cleavage_pattern_set',
      patterns: []
    }
  };
}

function normalizeDocument(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('The JSON root must be an object.');
  const set = value.cleavage_pattern_set;
  if (!set || typeof set !== 'object' || Array.isArray(set)) throw new Error('The JSON must contain a cleavage_pattern_set object.');
  return {
    cleavage_pattern_set: {
      name: String(set.name || ''),
      patterns: Array.isArray(set.patterns) ? set.patterns.map(pattern => ({
        name: String(pattern && pattern.name || ''),
        reactant_smarts: String(pattern && pattern.reactant_smarts || ''),
        products: Array.isArray(pattern && pattern.products) ? pattern.products.map(product => ({
          name: String(product && product.name || ''),
          smarts: String(product && product.smarts || '')
        })) : []
      })) : []
    }
  };
}

function normalizePattern(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || 'cleavage_pattern_set' in value) throw new Error('The JSON must contain a single cleavage pattern object.');
  return normalizeDocument({cleavage_pattern_set:{name:'',patterns:[value]}}).cleavage_pattern_set.patterns[0];
}

class CleavagePatternSetEditorProvider {
  constructor(context, singlePattern = false) { this.context = context; this.singlePattern = singlePattern; }
  resolveCustomTextEditor(document, panel) {
    const context = this.context, singlePattern = this.singlePattern;
    panel.webview.options = { enableScripts: true };
    panel.webview.html = editorHtml(singlePattern);
    let applyingEdit = false;

    const currentValue = () => {
      const source = JSON.parse(document.getText());
      return singlePattern ? {cleavage_pattern_set:{name:'',patterns:[normalizePattern(source)]}} : normalizeDocument(source);
    };
    const sendDocument = () => {
      try {
        panel.webview.postMessage({ type: 'cleavagePatternSet', value: currentValue(), path: document.uri.fsPath });
      } catch (error) {
        panel.webview.postMessage({ type: 'cleavagePatternError', message: error.message });
      }
    };

    const changeSubscription = vscode.workspace.onDidChangeTextDocument(event => {
      if (event.document.uri.toString() === document.uri.toString() && !applyingEdit) sendDocument();
    });
    panel.onDidDispose(() => changeSubscription.dispose());

    async function writeDocument(rawValue) {
      const value = normalizeDocument(rawValue);
      if (singlePattern && value.cleavage_pattern_set.patterns.length !== 1) throw new Error('A cleavage pattern file must contain exactly one pattern.');
      const savedValue = singlePattern ? value.cleavage_pattern_set.patterns[0] : value;
      const text = `${JSON.stringify(savedValue, null, 2)}\n`;
      if (text === document.getText()) return;
      const fullRange = new vscode.Range(document.positionAt(0), document.positionAt(document.getText().length));
      const edit = new vscode.WorkspaceEdit();
      edit.replace(document.uri, fullRange, text);
      applyingEdit = true;
      const applied = await vscode.workspace.applyEdit(edit);
      applyingEdit = false;
      if (!applied) throw new Error('The document could not be updated.');
    }

    panel.webview.onDidReceiveMessage(async message => {
      try {
        if (message.type === 'ready') { sendDocument(); return; }
        if (message.type === 'update') { await writeDocument(message.value); return; }
        if (message.type === 'copyCleavageText') { await vscode.env.clipboard.writeText(String(message.text || '')); return; }
        if (message.type === 'chemistry') {
          const result = await host.runChemistryBackend(context, message.command, message.payload, panel);
          panel.webview.postMessage({ type: 'chemistryResult', requestId: message.requestId, result });
          return;
        }
        if (message.type === 'selectElements') {
          const elements = await host.showElementPicker(message.elements);
          if (elements) panel.webview.postMessage({ type: 'elementSelectionResult', requestId: message.requestId, elements });
          return;
        }
        if (message.type === 'loadCleavagePatternSet' || message.type === 'loadCleavagePattern') {
          const source = await host.readCleavageImport(context, message);
          if (!source) return;
          if (message.type === 'loadCleavagePatternSet') {
            panel.webview.postMessage({ type: 'cleavagePatternSet', value: normalizeDocument(source.value), path: source.path });
          } else {
            panel.webview.postMessage({ type: 'cleavagePatternLoaded', path: source.path, pattern: normalizePattern(source.value), index: message.index });
          }
          return;
        }
        if (message.type === 'saveCleavagePatternSet') {
          const value = normalizeDocument(message.value);
          const defaultUri = vscode.Uri.file(host.cleavagePatternSetSavePath(value.cleavage_pattern_set.name, message.path, FILE_SUFFIX));
          const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern Set': ['pft'] }, defaultUri });
          if (selected) {
            const target = vscode.Uri.file(host.ensureFileSuffix(selected.fsPath, FILE_SUFFIX));
            await fs.promises.writeFile(target.fsPath, `${JSON.stringify(value, null, 2)}\n`);
            panel.webview.postMessage({ type: 'cleavagePatternSetSaved', path: target.fsPath });
            vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
          }
          return;
        }
        if (message.type === 'saveCleavagePattern') {
          const pattern = normalizePattern(message.pattern);
          const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern': ['pft'] }, defaultUri: vscode.Uri.file(`${host.safeFileStem(pattern.name || 'pattern')}.cleavage.pft`) });
          if (selected) {
            const target = vscode.Uri.file(host.ensureFileSuffix(selected.fsPath, '.cleavage.pft', ['.clevage.pft']));
            await fs.promises.writeFile(target.fsPath, `${JSON.stringify(pattern, null, 2)}\n`);
            vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
          }
          return;
        }
      } catch (error) {
        panel.webview.postMessage({ type: 'cleavagePatternError', message: String(error.message || error) });
      }
    });
  }
}

function register(context) {
  for (const [viewType, singlePattern] of [[VIEW_TYPE, false], [PATTERN_VIEW_TYPE, true]]) {
    context.subscriptions.push(
      vscode.window.registerCustomEditorProvider(viewType, new CleavagePatternSetEditorProvider(context, singlePattern), {
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false
      })
    );
  }
}

// Reuses the exact same markup, styling and client script as the Workbench's
// Cleavage Pattern Set tab (ui.js/visual-editor.js/reaction-preview.js), so this
// standalone file editor can never fall out of sync with the Workbench's editor.
function editorHtml(singlePattern = false) {
  return `<!doctype html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${commonCss()}${formCss()}${pathDrop.css()}</style></head><body><main>
  ${cleavageUi.html(singlePattern)}
  <script>
  const vscode=acquireVsCodeApi();
  ${cleavageUi.state()}
  const htmlEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  ${cleavageUi.script(singlePattern)}
  ${cleavageVisualEditor.script()}
  ${cleavageReactionPreview.script()}
  ${pathDrop.script()}
  document.getElementById('cleavageApp').hidden=false;
  let cleavageSyncTimer;onCleavageChange=()=>{clearTimeout(cleavageSyncTimer);cleavageSyncTimer=setTimeout(()=>vscode.postMessage({type:'update',value:cleavageModel}),200);};
  vscode.postMessage({type:'ready'});
  </script>
  </main></body></html>`;
}

module.exports = { register, normalizeDocument, normalizePattern, emptyDocument, FILE_SUFFIX, PATTERN_VIEW_TYPE, VIEW_TYPE };
