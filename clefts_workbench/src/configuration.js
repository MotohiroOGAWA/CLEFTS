const vscode = require('vscode');

const SCHEMA = 'clefts.workbench.configuration';
const SCHEMA_VERSION = 1;

function configurationDocument(kind, config) {
  if (!kind || !config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('Invalid Workbench configuration.');
  }
  return { schema: SCHEMA, schemaVersion: SCHEMA_VERSION, kind, config };
}

function parseConfiguration(value, expectedKind) {
  if (!value || value.schema !== SCHEMA || value.schemaVersion !== SCHEMA_VERSION) {
    throw new Error('This is not a supported CLEFTS Workbench configuration.');
  }
  if (value.kind !== expectedKind) {
    throw new Error(`Expected ${expectedKind} settings, but found ${value.kind || 'unknown'}.`);
  }
  if (!value.config || typeof value.config !== 'object' || Array.isArray(value.config)) {
    throw new Error('Workbench configuration payload is missing.');
  }
  return value.config;
}

async function saveConfiguration(kind, config, defaultName) {
  const selected = await vscode.window.showSaveDialog({
    filters: { 'CLEFTS Workbench configuration': ['json'] },
    defaultUri: vscode.Uri.file(defaultName)
  });
  if (!selected) return undefined;
  const target = selected.fsPath.toLowerCase().endsWith('.json')
    ? selected : vscode.Uri.file(selected.fsPath + '.json');
  const content = JSON.stringify(configurationDocument(kind, config), null, 2) + '\n';
  await vscode.workspace.fs.writeFile(target, Buffer.from(content, 'utf8'));
  return target.fsPath;
}

async function loadConfiguration(kind) {
  const selected = await vscode.window.showOpenDialog({
    canSelectMany: false, filters: { 'CLEFTS Workbench configuration': ['json'] }
  });
  if (!selected?.[0]) return undefined;
  const bytes = await vscode.workspace.fs.readFile(selected[0]);
  return {
    path: selected[0].fsPath,
    config: parseConfiguration(JSON.parse(Buffer.from(bytes).toString('utf8')), kind)
  };
}

module.exports = {
  SCHEMA, SCHEMA_VERSION, configurationDocument, parseConfiguration,
  saveConfiguration, loadConfiguration
};
