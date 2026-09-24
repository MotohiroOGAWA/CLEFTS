const cleavageUi = require('./features/cleavage-pattern-set/ui');
const {commonCss,formCss} = require('./features/cleavage-pattern-set/styles');
const projects = require('./workbench/project');
const cleavageVisualEditor = require('./features/cleavage-pattern-set/visual-editor');
const cleavageReactionPreview = require('./features/cleavage-pattern-set/reaction-preview');
const molTraining = require('./workbench/mol-training');
const trainingWorkbench = require('./workbench/training');
const resultView = require('./features/fragment-tree-result/view');
const layout = require('./workbench/layout');
const preparation = require('./workbench/preparation');
const datasetUpload = require('./workbench/dataset-upload');
const workbenchServices = require('./workbench/services');
const parameterEditor = require('./workbench/parameter-editor');
const workbenchDefaults = require('./workbench/defaults');
const parameterService = require('./workbench/parameter-service');
const workbench = require('./workbench/panel');
const trainingMetrics = require('./features/training-metrics/editor');
const trainingReport = require('./features/training-report/view');
const smartsSearch = require('./features/smarts-search/editor');
const evaluation = require('./features/evaluation/editor');
const spectrumPrediction = require('./features/spectrum-prediction/editor');
const pathDrop = require('./webview-path-drop');
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const cleavagePatternSetEditor = require('./features/cleavage-pattern-set/editor');
const adductUi = require('./features/adduct-rule-set/ui');
const adductRuleSetEditor = require('./features/adduct-rule-set/editor');
const cleavageHost = require('./features/cleavage-pattern-set/host');
const { projectRoot, isCleftsRoot, runChemistryBackend, showElementPicker, readCleavageImport, safeFileStem, ensureFileSuffix } = cleavageHost;

const RESULT_NAME = 'fragment-tree.pft';
let runningProcess;

function activate(context) {
  const output = vscode.window.createOutputChannel('CLEFTS');
  const provider = new ResultEditorProvider(context);
  context.subscriptions.push(
    output,
    vscode.commands.registerCommand('clefts.openWorkbench', () => openWorkbench(context, output)),
    vscode.commands.registerCommand('clefts.openResult', openResultPicker),
    vscode.commands.registerCommand('clefts.openEvaluation', () => evaluation.open(context, output, projectRoot)),
    vscode.window.registerCustomEditorProvider('clefts.resultViewer', provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: true
    })
  );
  evaluation.register(context, output, projectRoot);
  cleavagePatternSetEditor.register(context);
  adductRuleSetEditor.register(context);
  workbench.register(context, output, projectRoot, openWorkbench);
  enableDevelopmentReload(context, output);
}

function enableDevelopmentReload(context, output) {
  if (context.extensionMode !== vscode.ExtensionMode.Development) return;
  const pattern = new vscode.RelativePattern(context.extensionPath, '{src,media}/**/*');
  const watcher = vscode.workspace.createFileSystemWatcher(pattern);
  let reloadTimer;
  const reload = uri => {
    if (!/\.(js|css|html|json)$/.test(uri.fsPath)) return;
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(() => {
      output.appendLine(`[development] ${path.relative(context.extensionPath, uri.fsPath)} changed; reloading Extension Host.`);
      vscode.commands.executeCommand('workbench.action.reloadWindow');
    }, 250);
  };
  watcher.onDidChange(reload);
  watcher.onDidCreate(reload);
  watcher.onDidDelete(reload);
  context.subscriptions.push(watcher, { dispose: () => clearTimeout(reloadTimer) });
  output.appendLine('[development] Auto reload is enabled.');
}

function defaultConfig(context) {
  const root = projectRoot(context);
  const defaultParams = path.join(root, 'clefts', 'presets', 'spectrum_generator_params', 'source_anchored_pos_model_config.json');
  return workbenchDefaults.workflowDefaults('data',{
    application: 'fragment-tree-data-preparation', modelConfig: parameterService.defaults(root),
    input: '', params: defaultParams, smilesColumn:'SMILES', adductTypeColumn:'AdductType', collisionEnergyColumn:'CollisionEnergy', precursorMzColumn:'PrecursorMZ',
    validationSmilesColumn:'', validationAdductTypeColumn:'', validationCollisionEnergyColumn:'', validationPrecursorMzColumn:'',
    validationInput:'', validationRatio:0.1, validationSeed:0, minimumRelativeIntensity:0, normalizeIntensities:true, overwrite:false, numWorkers:1, chunkSize:1,
    outputDir: ''
  },root);
}

function defaultTrainingConfig(context) {
  const root = projectRoot(context);
  const defaultParams = path.join(root, 'clefts', 'presets', 'spectrum_generator_params', 'source_anchored_pos_model_config.json');
  return workbenchDefaults.workflowDefaults('training',{
    application: 'fragment-tree-training', modelConfig: parameterService.defaults(root), params: defaultParams, trainDir: '', valDir: '',
    outputDir: '',
    experimentName: 'exp_main', molEncoderCheckpoint: '', epochs: 1, batchSize: 4, device: 'cuda', maxSamples:128, seed:42, warmupSteps:100, validationIntervalSteps:0, validationFraction:0.1, lrPatience:3, earlyStoppingPatience:10, minLr:0.000001, branchWeight:1, negativeWeight:0.2, branchMilTemperature:0.1, intensityWeight:1, gradientClip:1, lr: 0.0001, dropout: 0.5, resume: '', minimumAssignmentScore:0, minimumAssignmentScoreWithoutPrecursor:0,
    fineTuneCheckpoint: '', adapterWidth: 8
  },root);
}

function openWorkbench(context, output, initialPage = "home") {
  context = projects.createContext(context);
  const panel = vscode.window.createWebviewPanel('clefts.workbench', 'CLEFTS Workbench', vscode.ViewColumn.One, {
    enableScripts: true, retainContextWhenHidden: true, localResourceRoots: []
  });
  projects.attach(panel, context, projectRoot);
  smartsSearch.attach(panel, context, projectRoot);
  spectrumPrediction.attach(panel, context, projectRoot, output);
  trainingMetrics.attach(panel);
  workbench.attach(panel, context, projectRoot, output, initialPage);
  parameterService.attach(panel, context, projectRoot);
  workbenchServices.attach(panel, context, projectRoot);
  workbenchDefaults.attach(panel);
  datasetUpload.attach(panel,context);
  molTraining.attach(panel,context,projectRoot,output);
  const initialConfig = defaultConfig(context);
  const initialTrainingConfig = defaultTrainingConfig(context);
  panel.webview.html = workbench.secure(workbenchHtml(initialConfig, initialTrainingConfig,workbenchDefaults.workflowDefaults('prediction',{},projectRoot(context)),workbenchDefaults.workflowDefaults('molTraining',molTraining.defaults(),projectRoot(context))), panel.webview);
  const panelContext = context;
  panel.webview.onDidReceiveMessage(async message => {
    const context = projects.capture(panelContext);
    try {
      if (message.type === 'pick') {
        const folders = message.kind === 'folder';
        const picked = await vscode.window.showOpenDialog({ canSelectFiles: !folders, canSelectFolders: folders, canSelectMany: false });
        if (picked && picked[0]) {
          panel.webview.postMessage({ type: 'picked', form: message.form, field: message.field, value: picked[0].fsPath });
        }
      } else if (message.type === 'saveConfig') {
        const target = await vscode.window.showSaveDialog({ filters: { 'CLEFTS run configuration': ['pft.json', 'json'] }, defaultUri: vscode.Uri.file('fragment-tree.pft.json') });
        if (target) {await fs.promises.writeFile(target.fsPath, JSON.stringify(normalizeConfig(message.config), null, 2) + '\n');projects.record(context,'data',message.config,{label:path.basename(target.fsPath),reason:'exported',sourcePath:target.fsPath,base:projectRoot(context)});}
      } else if (['loadConfig','loadConfigFile','loadConfigJSON'].includes(message.type)) {
        const picked = message.type==='loadConfigFile'?[vscode.Uri.file(message.path)]:message.type==='loadConfigJSON'?[{fsPath:message.name||'fragment-tree.pft.json'}]:await vscode.window.showOpenDialog({ filters: { 'CLEFTS run configuration': ['pft.json', 'json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const config = normalizeConfig(JSON.parse(message.type==='loadConfigJSON'?message.json:await fs.promises.readFile(picked[0].fsPath, 'utf8')));
          if (!config.modelConfig && config.params) config.modelConfig = parameterService.merge(parameterService.defaults(projectRoot(context)), parameterService.unpack(JSON.parse(await fs.promises.readFile(path.resolve(path.dirname(picked[0].fsPath), config.params), 'utf8'))));
          if(!config.modelConfig && config.fragmenterParams && !config.symbols) config.modelConfig=parameterService.merge(parameterService.defaults(projectRoot(context)),{fragmenter_params:config.fragmenterParams});
          panel.webview.postMessage({ type: 'config', path: message.type==='loadConfigJSON'?undefined:picked[0].fsPath, config: normalizeConfig({ ...defaultConfig(context), ...config }) });
        }
      } else if (message.type === 'saveTrainingConfig') {
        const target = await vscode.window.showSaveDialog({ filters: { 'CLEFTS training configuration': ['pfttrain.json'] }, defaultUri: vscode.Uri.file('fragment_tree.pfttrain.json') });
        if (target) {const config=normalizeTrainingConfig(message.config);await fs.promises.writeFile(target.fsPath, JSON.stringify(config, null, 2) + '\n');projects.record(context,'training',config,{label:path.basename(target.fsPath),reason:'exported',sourcePath:target.fsPath,base:projectRoot(context)});}
      } else if (['loadTrainingConfig','loadTrainingConfigFile','loadTrainingConfigJSON'].includes(message.type)) {
        if(message.type==='loadTrainingConfigJSON'&&(typeof message.json!=='string'||message.json.length>2000000))throw new Error('Training configuration is too large or invalid.');
        if(message.type==='loadTrainingConfigFile'&&(typeof message.path!=='string'||!message.path.trim()))throw new Error('Select a training configuration file.');
        const picked = message.type==='loadTrainingConfigFile'?[vscode.Uri.file(message.path)]:message.type==='loadTrainingConfigJSON'?[{fsPath:message.name||'fragment_tree.pfttrain.json'}]:await vscode.window.showOpenDialog({ filters: { 'CLEFTS training configuration': ['pfttrain.json'] }, canSelectMany: false });
        if (picked && picked[0]) {
          const raw=JSON.parse(message.type==='loadTrainingConfigJSON'?message.json:await fs.promises.readFile(picked[0].fsPath, 'utf8'));
          if(!raw.modelConfig&&raw.params){if(message.type==='loadTrainingConfigJSON')throw new Error('Dropped training configurations must contain their model settings.');raw.modelConfig=parameterService.merge(parameterService.defaults(projectRoot(context)),parameterService.unpack(JSON.parse(await fs.promises.readFile(path.resolve(path.dirname(picked[0].fsPath),raw.params),'utf8'))));}
          const config=normalizeTrainingConfig(raw);
          panel.webview.postMessage({ type: 'trainingConfig', path: message.type==='loadTrainingConfigJSON'?undefined:picked[0].fsPath, config: { ...defaultTrainingConfig(context), ...config } });
        }
      } else if (message.type === 'openResult') {
        await openResultPicker();
      } else if (message.type === 'openEvaluation') {
        evaluation.open(context, output, projectRoot);
      } else if (message.type === 'copyCleavageText') {
        await vscode.env.clipboard.writeText(String(message.text || ''));
      } else if (message.type === 'copyCommand') {
        const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
        const command = shellDisplay(python, buildArgs(normalizeConfig(message.config)));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'status', status: 'idle', text: 'Command copied.', command });
      } else if (message.type === 'run') {
        const config=normalizeConfig(message.config),root=projectRoot(context),target=path.resolve(root,config.outputDir||'');
        if(!config.overwrite&&config.outputDir&&fs.existsSync(target)){
          const choice=await vscode.window.showWarningMessage(`Output directory already exists: ${target}\nOverwrite it before processing?`,{modal:true},'Overwrite');
          if(choice!=='Overwrite'){panel.webview.postMessage({type:'status',status:'idle',text:'Run cancelled. The output directory was not changed.'});return;}
          config.overwrite=true;
        }
        panel.webview.postMessage({type:'status',status:'running',text:'Validating datasets and configuration…'});
        await workbenchServices.validateRun(context,root,config);
        await runFragmentTree(context, output, config, panel);
      } else if (message.type === 'copyTrainingCommand') {
        const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
        const command = shellDisplay(python, buildTrainingArgs(normalizeTrainingConfig(message.config)));
        await vscode.env.clipboard.writeText(command);
        panel.webview.postMessage({ type: 'trainingStatus', status: 'idle', text: 'Command copied.', command });
      } else if (message.type === 'runTraining') {
        await runTraining(context, output, normalizeTrainingConfig(message.config), panel);
      } else if (message.type === 'stop') {
        if (runningProcess) workbench.stopChild(runningProcess,process.platform!=='win32');
      } else if (message.type === 'loadCleavagePatternSet') {
        const source = await readCleavageImport(context, message);
        if (source) {
          const value = cleavagePatternSetEditor.normalizeDocument(source.value);
          panel.webview.postMessage({ type: 'cleavagePatternSet', value, path: source.path });
        }
      } else if (message.type === 'saveCleavagePatternSet') {
        const value = cleavagePatternSetEditor.normalizeDocument(message.value);
        const defaultUri = vscode.Uri.file(cleavagePatternSetSavePath(value.cleavage_pattern_set.name, message.path));
        const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern Set': ['json'] }, defaultUri });
        if (selected) {
          const target = vscode.Uri.file(ensureFileSuffix(selected.fsPath, cleavagePatternSetEditor.FILE_SUFFIX));
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(value, null, 2)}\n`);
          panel.webview.postMessage({ type: 'cleavagePatternSetSaved', path: target.fsPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'loadCleavagePattern') {
        const source = await readCleavageImport(context, message);
        if (source) {
          const pattern = normalizePattern(source.value);
          panel.webview.postMessage({ type: 'cleavagePatternLoaded', path: source.path, pattern, index: message.index });
        }
      } else if (message.type === 'saveCleavagePattern') {
        const pattern = normalizePattern(message.pattern);
        const selected = await vscode.window.showSaveDialog({ filters: { 'CLEFTS Cleavage Pattern': ['json'] }, defaultUri: vscode.Uri.file(`${safeFileStem(pattern.name || 'pattern')}.cleavage.json`) });
        if (selected) {
          const target = vscode.Uri.file(ensureFileSuffix(selected.fsPath, '.cleavage.json', ['.clevage.json']));
          await fs.promises.writeFile(target.fsPath, `${JSON.stringify(pattern, null, 2)}\n`);
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      } else if (message.type === 'loadAdductRuleSet' || message.type === 'loadAdductRule') {
        const source = await readCleavageImport(context, message);
        if (source) {
          if (message.type === 'loadAdductRuleSet') panel.webview.postMessage({type:'adductRuleSet',value:adductRuleSetEditor.normalizeDocument(source.value),path:source.path});
          else panel.webview.postMessage({type:'adductRuleLoaded',rule:adductRuleSetEditor.normalizeRule(source.value),index:message.index,path:source.path});
        }
      } else if (message.type === 'saveAdductRuleSet') {
        const value=adductRuleSetEditor.normalizeDocument(message.value),defaultUri=vscode.Uri.file(cleavageHost.cleavagePatternSetSavePath(value.fragment_ion_adduct_rule_set.name,message.path,adductRuleSetEditor.FILE_SUFFIX));
        const selected=await vscode.window.showSaveDialog({filters:{'CLEFTS Fragment Ion Adduct Rule Set':['json']},defaultUri});
        if(selected){const target=vscode.Uri.file(ensureFileSuffix(selected.fsPath,adductRuleSetEditor.FILE_SUFFIX));await fs.promises.writeFile(target.fsPath,JSON.stringify(value,null,2)+'\n');panel.webview.postMessage({type:'adductRuleSetSaved',path:target.fsPath});vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);}
      } else if (message.type === 'saveAdductRule') {
        const rule=adductRuleSetEditor.normalizeRule(message.rule),selected=await vscode.window.showSaveDialog({filters:{'CLEFTS Fragment Ion Adduct Rule':['json']},defaultUri:vscode.Uri.file(safeFileStem(rule.name||'adduct_rule')+adductRuleSetEditor.RULE_SUFFIX)});
        if(selected){const target=vscode.Uri.file(ensureFileSuffix(selected.fsPath,adductRuleSetEditor.RULE_SUFFIX));await fs.promises.writeFile(target.fsPath,JSON.stringify(rule,null,2)+'\n');vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);}
      } else if (message.type === 'selectElements') {
        const elements = await showElementPicker(message.elements);
        if (elements) panel.webview.postMessage({ type: 'elementSelectionResult', requestId: message.requestId, elements });
      } else if (message.type === 'chemistry') {
        const result = await runChemistryBackend(context, message.command, message.payload, panel);
        panel.webview.postMessage({ type: 'chemistryResult', requestId: message.requestId, result });
      } else if (message.type === 'predictSpectrum') {
        try {
          const result = await runSpectrumPredictionBackend(context, { command: 'predict', payload: message.payload });
          panel.webview.postMessage({ type: 'predictSpectrumResult', requestId: message.requestId, result });
        } catch (error) {
          panel.webview.postMessage({ type: 'predictSpectrumError', requestId: message.requestId, message: String(error.message || error) });
        }
      } else if (message.type === 'listPredictAdducts') {
        try {
          const result = await runSpectrumPredictionBackend(context, { command: 'listAdducts', payload: message.payload });
          panel.webview.postMessage({ type: 'listPredictAdductsResult', requestId: message.requestId, result });
        } catch (error) {
          panel.webview.postMessage({ type: 'listPredictAdductsError', requestId: message.requestId, message: String(error.message || error) });
        }
      }
    } catch (error) {
      if (/^(load|save)Cleavage/.test(message.type)) panel.webview.postMessage({ type: 'cleavagePatternError', message: String(error.message || error) });
      if (/^(load|save)Adduct/.test(message.type)) panel.webview.postMessage({ type: 'adductRuleError', message: String(error.message || error) });
      if (message.type === 'chemistry') panel.webview.postMessage({ type: 'chemistryError', requestId: message.requestId, message: String(error.message || error) });
      const trainingMessage = ['runTraining', 'copyTrainingCommand', 'saveTrainingConfig', 'loadTrainingConfig', 'loadTrainingConfigFile', 'loadTrainingConfigJSON'].includes(message.type);
      panel.webview.postMessage({ type: trainingMessage ? 'trainingStatus' : 'status', status: 'error', text: String(error.message || error) });
      vscode.window.showErrorMessage(`CLEFTS: ${error.message || error}`);
    }
  });
}

function normalizePattern(value) { return cleavagePatternSetEditor.normalizePattern(value); }

function cleavagePatternSetSavePath(name, previousPath) {
  return cleavageHost.cleavagePatternSetSavePath(name, previousPath, cleavagePatternSetEditor.FILE_SUFFIX);
}

function runSpectrumPredictionBackend(context, payload) {
  return new Promise((resolve, reject) => {
    const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'spectrum-prediction', 'backend.py');
    let root;
    try { root = projectRoot(context); }
    catch (error) { reject(error); return; }
    if (!isCleftsRoot(root)) { reject(new Error('The detected working directory is not a CLEFTS application.')); return; }
    const pythonPath = ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const child = spawn(python, [script], { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath } });
    const observed=payload.command==='predict'?workbench.observeJob(context,{type:'prediction',name:String(payload.payload?.smiles||'Spectrum prediction').slice(0,72),child,command:[python,script],config:payload.payload,workflow:'prediction',base:projectRoot(context)}):null;
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { const text=chunk.toString();stderr += text;if(observed)observed.log(text); });
    child.on('error', error=>{if(observed)observed.finish({error:error.message});reject(error);});
    child.on('close', (code,signal) => {
      try { const response = JSON.parse(stdout.trim().split('\n').reverse().find(line=>line.startsWith('{'))||stdout);if(!response.ok)throw new Error(response.error);if(code!==0)throw new Error(stderr||'Prediction backend failed.');if(observed){observed.log('Predicted '+response.result.peaks.length+' peaks.\n');observed.finish({code:0,result:response.result});}resolve(response.result); }
      catch (error) {if(observed){observed.log(stderr||stdout||error.message);observed.finish({code,signal,error:signal?undefined:error.message});}reject(new Error(error.message||stderr||stdout||'The spectrum prediction backend did not return a response.'));}
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

function normalizeConfig(config) {
  const result={...config,...config.workbench_config};
  delete result.workbench_config;
  const aliases={validation_input:'validationInput',validation_ratio:'validationRatio',validation_seed:'validationSeed',
    output_dir:'outputDir',smiles_column:'smilesColumn',adduct_type_column:'adductTypeColumn',collision_energy_column:'collisionEnergyColumn',
    precursor_mz_column:'precursorMzColumn',validation_smiles_column:'validationSmilesColumn',validation_adduct_type_column:'validationAdductTypeColumn',
    validation_collision_energy_column:'validationCollisionEnergyColumn',validation_precursor_mz_column:'validationPrecursorMzColumn',
    minimum_relative_intensity:'minimumRelativeIntensity',normalize_intensities:'normalizeIntensities',
    num_workers:'numWorkers',chunk_size:'chunkSize',max_node:'maxNode',max_edge:'maxEdge',model_config:'modelConfig'};
  for(const [from,to] of Object.entries(aliases)){if(result[from]!==undefined&&result[to]===undefined)result[to]=result[from];delete result[from];}
  if(result.modelConfig){const model=result.modelConfig.params||result.modelConfig;result.fragmenterParams??=model.fragmenter_params;result.symbols??=model.symbols||model.mol_encoder_params?.symbols;result.maxNode??=model.max_node;result.maxEdge??=model.max_edge;if(result.fragmenterParams)delete result.modelConfig;}
  if(result.validationRatio==null)delete result.validationRatio;
  if(result.validationInput==null)result.validationInput='';
  if (result.modelConfig || result.fragmenterParams) delete result.params;
  result.input ||= result.trainInput;
  return result;
}

function normalizeTrainingConfig(config) {
  const result=JSON.parse(JSON.stringify(config||{}));
  const model=result.modelConfig?.params||result.modelConfig;
  if(model&&typeof model==='object'&&!Array.isArray(model)){
    if(!result.molEncoderCheckpoint&&typeof model.mol_encoder_checkpoint==='string')result.molEncoderCheckpoint=model.mol_encoder_checkpoint;
    result.modelConfig={};
    for(const key of ['action_model_params','post_model_params'])if(model[key]&&typeof model[key]==='object'&&!Array.isArray(model[key]))result.modelConfig[key]=model[key];
    if(!Object.keys(result.modelConfig).length)delete result.modelConfig;
  }
  for(const key of ['adduct_type_strs','fragmenter_params','mol_encoder_params','symbols'])delete result[key];
  delete result.params;
  delete result.initializeFrom;
  return result;
}

async function runTraining(context, output, config, panel) {
  if (runningProcess) throw new Error('Another CLEFTS process is already running.');
  for (const key of ['trainDir', 'valDir', 'outputDir']) {
    if (!config[key]) throw new Error(`${key} is required.`);
  }
  if (!config.resume && !config.fineTuneCheckpoint && !String(config.molEncoderCheckpoint || '').trim()) throw new Error('Mol encoder checkpoint is required for new training.');
  if (config.initializeFrom && (config.resume || config.fineTuneCheckpoint)) throw new Error('Use pretrained weight initialization, resume, or frozen-base expansion separately.');
  const root = projectRoot(context);
  const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
  await workbench.startTraining(context, panel, output, root, python, config, buildTrainingArgs);

}

async function runFragmentTree(context, output, config, panel) {
  if (runningProcess) throw new Error('Another CLEFTS process is already running.');
  for (const key of ['input', 'outputDir']) if (!config[key]) throw new Error(`${key} is required.`);
  const root = projectRoot(context);
  const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
  const resultPath = path.join(config.outputDir, 'train_structures', 'fragment-tree.pft.json');
  const args = buildArgs(config);
  const command = shellDisplay(python, args);
  output.clear(); output.show(true); output.appendLine(`$ ${command}`);
  panel.webview.postMessage({ type: 'status', status: 'running', text: 'Running CLI…', command });
  const startedAt = new Date().toISOString();
  runningProcess = spawn(python, args, { cwd: root, detached:process.platform!=='win32', env: { ...process.env, PYTHONUNBUFFERED:'1', PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) } });
  const observed=workbench.observeJob(context,{type:'preparation',name:path.basename(config.outputDir),outputDir:config.outputDir,child:runningProcess,detached:process.platform!=='win32',command:[python,...args],config,workflow:'data',base:root});
  const stream=chunk=>{const text=chunk.toString();observed.log(text);output.append(text);panel.webview.postMessage({type:'data/log',text});};
  runningProcess.stdout.on('data', stream);
  runningProcess.stderr.on('data', stream);
  runningProcess.on('error', async error => {
    observed.finish({error:error.message});
    runningProcess = undefined;
    panel.webview.postMessage({ type: 'status', status: 'error', text: error.message });
  });
  runningProcess.on('close', async (code,signal) => {
    runningProcess = undefined;
    observed.finish({code,signal});
    const status = observed.job.status;
    if (code === 0) {
      for(const split of ['train','validation']){
        const target=path.join(config.outputDir,split+'_structures','fragment-tree.pft.json');
        if(!fs.existsSync(target))continue;
        const saved=JSON.parse(await fs.promises.readFile(target,'utf8'));
        await fs.promises.writeFile(target,JSON.stringify({...saved,status:'completed',startedAt,finishedAt:new Date().toISOString(),exitCode:code,command:args},null,2)+'\n');
      }
    }
    panel.webview.postMessage({ type: 'status', status, text: signal ? 'Cancelled.' : code === 0 ? 'Completed.' : `Failed with exit code ${code}.`, resultPath });
    if (code === 0) {const validationPath=path.join(config.outputDir,'validation_structures','fragment-tree.pft.json');vscode.window.showInformationMessage('CLEFTS fragment tree data preparation completed.', 'Open Train Result',...(fs.existsSync(validationPath)?['Open Validation Result']:[])).then(choice => { if (choice) openResult(vscode.Uri.file(choice==='Open Validation Result'?validationPath:resultPath)); });}
  });
}

function buildArgs(c) {
  const args=['-m', 'clefts.cli', 'train', 'create-fragment-tree-data',
    '--input', c.input || c.trainInput, '--output-dir', c.outputDir];
  if(c.modelConfig || c.fragmenterParams) args.push('--params-json',JSON.stringify(c.modelConfig || c.fragmenterParams));
  else if(c.params) args.push('--params',c.params);
  if(c.symbols)args.push('--symbols-json',JSON.stringify(c.symbols));
  for(const [key,flag]of Object.entries({maxNode:'--max-node',maxEdge:'--max-edge'}))if(c[key]!==undefined)args.push(flag,String(c[key]));
  for(const [key,flag] of Object.entries({smilesColumn:'--smiles-column',adductTypeColumn:'--adduct-type-column',collisionEnergyColumn:'--collision-energy-column',precursorMzColumn:'--precursor-mz-column'}))if(c[key])args.push(flag,c[key]);
  // Only emitted when the validation dataset genuinely needs a different column name than training.
  if(c.validationInput)for(const [key,flag,trainKey] of [['validationSmilesColumn','--validation-smiles-column','smilesColumn'],['validationAdductTypeColumn','--validation-adduct-type-column','adductTypeColumn'],['validationCollisionEnergyColumn','--validation-collision-energy-column','collisionEnergyColumn'],['validationPrecursorMzColumn','--validation-precursor-mz-column','precursorMzColumn']])if(c[key]&&c[key]!==c[trainKey])args.push(flag,c[key]);
  if(c.validationInput) args.push('--validation-input',c.validationInput);
  // With --validation-input this caps validation to this fraction of training's
  // unique SMILES after removing molecules shared with training, instead of
  // splitting a single dataset (which is what it does without --validation-input).
  if(c.validationRatio !== undefined && c.validationRatio !== '') args.push('--validation-ratio',String(c.validationRatio));
  for(const [key,flag] of Object.entries({validationSeed:'--validation-seed',minimumRelativeIntensity:'--minimum-relative-intensity',numWorkers:'--num-workers',chunkSize:'--chunk-size'}))if(c[key] !== undefined && c[key] !== '')args.push(flag,String(c[key]));
  for(const [key,flag] of Object.entries({normalizeIntensities:'--normalize-intensities',overwrite:'--overwrite'}))if(c[key]!==undefined)args.push(flag,c[key]?'1':'0');
  return args;
}

function buildTrainingArgs(c) {
  const a = ['-m', 'clefts.cli', 'train', 'fragment-tree',
    '--train-dir', c.trainDir, '--val-dir', c.valDir, '--output-dir', c.outputDir,
    '--epochs', String(c.epochs || 1), '--batch-size', String(c.batchSize || 4),
    '--device', c.device || 'cuda', '--lr', String(c.lr ?? 0.0001)];
  const encoder=c.molEncoderCheckpoint;
  if(encoder&&!c.resume&&!c.fineTuneCheckpoint)a.push('--mol-encoder-checkpoint',encoder);
  if(!c.resume&&!c.fineTuneCheckpoint){
    const sections={action_model_params:{hidden_dim:'action-hidden-dim',branch_main_adduct_dim:'action-main-adduct-dim',num_heads:'action-num-heads',branch_path_threshold:'branch-path-threshold',max_fragment_nodes:'max-fragment-nodes',state_num_layers:'action-state-layers',action_neighborhood_mode:'action-neighborhood-mode',action_neighborhood_max_hop:'action-neighborhood-max-hop'},post_model_params:{hidden_dim:'post-hidden-dim',num_layers:'post-num-layers',num_heads:'post-num-heads',ion_embedding_dim:'ion-embedding-dim',unsaturation_embedding_dim:'unsaturation-embedding-dim',radical_embedding_dim:'radical-embedding-dim',state_hidden_dim:'ion-state-hidden-dim',main_adduct_dim:'main-adduct-embedding-dim',collision_energy_dim:'collision-energy-feature-dim',cosine_loss_weight:'post-cosine-loss-weight',ion_loss_weight:'post-ion-loss-weight',ion_prediction_threshold:'post-ion-threshold',peak_intensity_threshold:'post-peak-intensity-threshold',intensity_power:'post-intensity-power',precursor_free_loss_weight:'post-precursor-free-weight'}};
    for(const [section,flags]of Object.entries(sections))for(const [key,flag]of Object.entries(flags)){const value=c.modelConfig?.[section]?.[key];if(value!==undefined&&value!==null)a.push('--'+flag,String(value));}
  }
  for (const [key, flag] of Object.entries(workbench.trainingFlags)) if (c[key] !== undefined && c[key] !== '') a.push(flag, String(c[key]));
  if(c.trainMolEncoder)a.push('--train-mol-encoder');
  if(c.overwrite)a.push('--overwrite');
  if (c.initializeFrom) a.push('--initialize-from', c.initializeFrom);
  if (c.resume) a.push('--resume', c.resume);
  if (c.fineTuneCheckpoint && !c.resume) a.push('--fine-tune-checkpoint', c.fineTuneCheckpoint, '--adapter-width', String(c.adapterWidth || 8));
  return a;
}

function shellDisplay(program, args) { return [program, ...args].map(v => /^[A-Za-z0-9_./:=,-]+$/.test(v) ? v : `'${v.replace(/'/g, "'\\''")}'`).join(' '); }
async function openResultPicker() {
  const picked = await vscode.window.showOpenDialog({ filters: { 'CLEFTS result': ['pft.json', 'pft', 'clefts-result'] }, canSelectMany: false });
  if (picked && picked[0]) openResult(picked[0]);
}
function openResult(uri) { return vscode.commands.executeCommand('vscode.openWith', uri, 'clefts.resultViewer'); }

class ResultEditorProvider {
  constructor(context) { this.context = context; }
  async openCustomDocument(uri) { return { uri, dispose() {} }; }
  async resolveCustomEditor(document, panel) {
    panel.webview.options = { enableScripts: true };
    trainingMetrics.attach(panel);
    const refresh = async () => {
      try { panel.webview.html = await resultHtml(document.uri.fsPath); }
      catch (e) { panel.webview.html = errorHtml(e.message); }
    };
    let moleculePanel,selection=0;
    panel.onDidDispose(()=>{selection++;moleculePanel?.dispose();});
    panel.webview.onDidReceiveMessage(async m => {
      if (m.type === 'refresh') refresh();
      if (m.type === 'openFile') {
        const base = path.dirname(document.uri.fsPath), target = path.resolve(base, m.path);
        if (!(target === base || target.startsWith(base + path.sep))) return;
        if (target.endsWith('.preft.pt') || target.endsWith('.preft') || target.endsWith('.pt')) {
          panel.webview.postMessage({ type: 'structureLoading', path: m.path });
          try { panel.webview.postMessage({ type: 'structureDetail', result: await inspectFragmentTreeResult(this.context, target) }); }
          catch (error) { panel.webview.postMessage({ type: 'structureError', message: String(error.message || error) }); }
        } else vscode.commands.executeCommand('vscode.open', vscode.Uri.file(target));
      } else if (m.type === 'inspectMolecule') {
        if(!m.smiles)return;
        const request=++selection;
        try {
          const result=await runChemistryBackend(this.context,'depict',{smiles:m.smiles});
          if(request!==selection)return;
          if(!moleculePanel){moleculePanel=vscode.window.createWebviewPanel('clefts.fragmentMolecule','Fragment Molecule',{viewColumn:vscode.ViewColumn.Beside,preserveFocus:true},{enableScripts:true,retainContextWhenHidden:true});moleculePanel.onDidDispose(()=>{moleculePanel=undefined;});}
          moleculePanel.title='Fragment Node '+m.node;
          moleculePanel.webview.html=resultView.moleculeHtml(result,m.smiles,m.node,m.theme);
        }catch(error){panel.webview.postMessage({type:'structureError',message:String(error.message||error)});}
      } else if(m.type==='resultTheme') {
        moleculePanel?.webview.postMessage({type:'theme',theme:m.theme});
      } else if(m.type==='shell/document') {
        await workbenchServices.openDocumentation(this.context,projectRoot(this.context));
      } else if(m.type==='shell/github') {
        await vscode.env.openExternal(vscode.Uri.parse('https://github.com/MotohiroOGAWA/CLEFTS'));
      }

    });
    await refresh();
  }
}

async function resultHtml(manifestPath) {
  const manifest = JSON.parse(await fs.promises.readFile(manifestPath, 'utf8'));
  if (manifest.schema === 'clefts.training-report') {
    return trainingReport.html({ manifest, root: path.dirname(manifestPath), reportPath: manifestPath }, commonCss());
  }
  const root = path.dirname(manifestPath);
  const files = await scan(root, root, 3, 500);
  const manifests = await readStructureManifests(root);
  const summary = summarize(files);
  summary.preft = manifests.reduce((total, item) => total + item.rows.filter(row => row.exists).length, 0);
  const assignmentScores=[];
  for(const item of manifests){
    let scoreFile=path.join(root,item.relativeBase,'assignment_scores.tsv');
    if(!fs.existsSync(scoreFile))scoreFile=path.join(root,item.relativeBase,'data','assignment_scores.tsv');
    if(!fs.existsSync(scoreFile))continue;
    const lines=(await fs.promises.readFile(scoreFile,'utf8')).trim().split(/\r?\n/),columns=lines.shift().split('\t'),byFile=new Map();
    for(const line of lines){const cells=line.split('\t'),value=key=>{const raw=cells[columns.indexOf(key)];if(!raw?.trim())return null;const number=Number(raw);return Number.isFinite(number)&&number>=0&&number<=1?number:null;};const entry={score:value('assignment_score'),withoutPrecursor:value('assignment_score_without_precursor')};assignmentScores.push(entry);const file=path.basename(cells[columns.indexOf('structure_file')]||'');if(!byFile.has(file))byFile.set(file,[]);byFile.get(file).push(entry);}
    for(const row of item.rows){const entries=byFile.get(path.basename(row.file||''))||[];for(const [field,key]of [['assignment_score','score'],['assignment_score_without_precursor','withoutPrecursor']])if(row[field]==null||row[field]===''){const values=entries.map(entry=>entry[key]).filter(value=>value!=null);row[field]=values.length?values.reduce((a,b)=>a+b,0)/values.length:null;}}
  }
  const datasetStats = summarizeStructureManifests(manifests);
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}:root{--detected:#ff8a3d}.result-main{max-width:none;width:100%}.dataset-stat-cards{grid-template-columns:repeat(4,minmax(150px,1fr))}.dataset-stat-cards article small{display:block;margin-top:5px}.dataset-breakdown th:not(:first-child),.dataset-breakdown td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}.tree-scroll{overflow:auto;max-height:70vh;overscroll-behavior:contain;border:1px solid var(--border);border-radius:8px;background:var(--vscode-editor-background);scrollbar-gutter:stable}.fragment-tree{display:block;width:100%;min-width:700px}.tree-edge{stroke:color-mix(in srgb,var(--vscode-editor-foreground) 35%,transparent);stroke-width:1.5}.tree-edge.detected{stroke:var(--detected);stroke-width:3}.tree-node circle{fill:var(--vscode-editor-background);stroke:var(--vscode-editor-foreground);stroke-width:1.5}.tree-node.detected circle{fill:color-mix(in srgb,var(--detected) 28%,var(--vscode-editor-background));stroke:var(--detected);stroke-width:3}.tree-node text{fill:var(--vscode-editor-foreground);font-size:9px;pointer-events:none}.tree-node{cursor:pointer}.molecule-preview{width:100%;max-height:360px;overflow:hidden;background:#fff;border-radius:6px}.molecule-preview svg{display:block;width:100%;height:auto}.chem-info{width:100%;font-size:11px}.chem-info th,.chem-info td{padding:5px}.used-adduct{background:color-mix(in srgb,var(--accent) 24%,transparent);outline:1px solid var(--accent)}.used-adduct.modified-state{background:color-mix(in srgb,#b06cff 30%,transparent);outline-color:#b06cff}.used-adduct td:first-child::before{content:'✓ ';color:var(--accent);font-weight:700}.used-adduct.modified-state td:first-child::before{color:#b06cff}.manifest-grid{margin-top:18px}.manifest-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}.manifest-controls input{min-width:260px;flex:1;padding:7px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--border);border-radius:5px}.manifest-controls label{display:flex;align-items:center;gap:6px}.manifest-controls select{width:auto;margin:0}.manifest-scroll{max-height:none;overflow-x:auto;overflow-y:visible}.column-filters input{width:100%;min-width:80px;padding:4px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--border)}.missing-file{display:block;color:var(--vscode-errorForeground)}.tree-controls,.detail-controls{display:flex;justify-content:flex-end;align-items:center;gap:8px;flex-wrap:wrap;margin:8px 0}.tree-controls select{width:auto}.tree-controls button{padding:5px 9px}.detail-controls input{width:180px;margin:0}.sortable{cursor:pointer;user-select:none}.sortable:hover{color:var(--accent)}.detail-layout{display:grid;grid-template-columns:minmax(0,1fr) var(--molecule-width,35%);gap:18px;align-items:start}.molecule-side{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto;border:1px solid var(--border);border-radius:8px;padding:14px}.molecule-side[hidden]{display:block;visibility:hidden}.molecule-side code{display:block;overflow-wrap:anywhere;margin:8px 0}.target-diagnostic{font-size:11px;padding:3px 0;border:0}.target-diagnostic summary{font-weight:400}.target-diagnostic code{display:block;white-space:normal;margin:2px 0}@media(max-width:900px){.dataset-stat-cards{grid-template-columns:repeat(2,minmax(140px,1fr))}.detail-layout{grid-template-columns:1fr}.molecule-side{position:static;max-height:none}.molecule-side[hidden]{display:none}}${resultView.css}</style></head><body><div class="result-toolbar"><strong>CLEFTS Workbench <small> / Fragment Tree Dataset Results</small></strong><button id="resultDocs">Documentation</button><button id="resultGithub">GitHub</button><button id="resultTheme">☀ / ☾</button><button id="resultHelp">Help</button></div><main class="result-main"><p id="resultHelpText" class="result-help" hidden>Select an output file to inspect its saved fragment trees. Node labels are fragment node IDs; diamonds mark precursors. Click nodes to open the molecule window, or edges to inspect the added action. Expand action sets and follow Action links to the atom-map reference.</p>
    <header><div><span class="eyebrow">CLEFTS RESULT</span><h1>Fragment Tree Dataset Results</h1><p class="muted">${escapeHtml(root)}</p></div><button onclick="vscode.postMessage({type:'refresh'})">Refresh</button></header>
    <section class="cards result-overview"><article><b>${summary.preft}</b><span>Generated structures</span></article><article><b>${datasetStats.total.totalSpectra}</b><span>Total records</span></article><article><b>${datasetStats.total.totalCompounds}</b><span>Sources / SMILES</span></article><article><b>${datasetStats.total.invalidSpectra}</b><span>Skipped records</span></article><article><b>${datasetStats.datasets.length}</b><span>Datasets</span></article><article><b>${escapeHtml(manifest.status||'unknown')}</b><span>Job status</span></article></section>
    ${datasetStatisticsHtml(datasetStats)}
    ${resultView.histogramHtml(assignmentScores)}
    <section><details><summary>Run Information</summary><dl><dt>Application</dt><dd>${escapeHtml(manifest.application || '')}</dd><dt>Finished</dt><dd>${escapeHtml(manifest.finishedAt || '—')}</dd><dt>Exit code</dt><dd>${escapeHtml(String(manifest.exitCode ?? '—'))}</dd><dt>Command</dt><dd><code>${escapeHtml((manifest.command || []).join(' '))}</code></dd></dl></details></section>
    <section><h2>Output Files</h2><p class="muted">Select a generated structure to inspect its Source, fragment tree and action reference.</p>${manifests.map((manifest,index)=>manifestTableHtml(manifest,index)).join('') || '<p>No structure manifest was found.</p>'}</section>
    <section><details><summary>Supporting Files <small>${files.length} files</small></summary><div class="files">${files.map(f => `<button class="file" data-open-file="${encodeURIComponent(f.relative)}"><span>${escapeHtml(f.relative)}</span><em>${formatBytes(f.size)}</em></button>`).join('')}</div></details></section>
    <section id="detail" hidden></section>
    <script>const vscode=acquireVsCodeApi(),detail=document.getElementById('detail'),sortState=new Map();const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    ${resultView.script()}
    document.addEventListener('click',e=>{const file=e.target.closest('[data-open-file]');if(file)vscode.postMessage({type:'openFile',path:decodeURIComponent(file.dataset.openFile)})});
    const manifestStates=new WeakMap();
    function updateManifest(grid,resetPage=false){const state=manifestStates.get(grid)||{page:0,sortColumn:-1,sortDirection:1};if(resetPage)state.page=0;const search=grid.querySelector('[data-manifest-search]').value.toLowerCase(),filters=[...grid.querySelectorAll('[data-column-filter]')].map(input=>input.value.toLowerCase()),body=grid.querySelector('tbody'),rows=[...body.rows].filter(row=>{const values=[...row.cells].map(cell=>(cell.dataset.value||'').toLowerCase());return(!search||values.some(value=>value.includes(search)))&&filters.every((filter,index)=>!filter||values[index].includes(filter))});if(state.sortColumn>=0)rows.sort((a,b)=>{const left=a.cells[state.sortColumn].dataset.value||'',right=b.cells[state.sortColumn].dataset.value||'',ln=Number(left),rn=Number(right),result=left!==''&&right!==''&&Number.isFinite(ln)&&Number.isFinite(rn)?ln-rn:left.localeCompare(right,undefined,{numeric:true,sensitivity:'base'});return state.sortDirection*result});for(const row of rows)body.appendChild(row);const pageSize=Number(grid.querySelector('[data-page-size]').value),pages=Math.max(1,Math.ceil(rows.length/pageSize));state.page=Math.min(state.page,pages-1);const pageRows=new Set(rows.slice(state.page*pageSize,(state.page+1)*pageSize));for(const row of body.rows)row.hidden=!pageRows.has(row);grid.querySelector('[data-page-label]').textContent=(rows.length?state.page*pageSize+1:0)+'–'+Math.min((state.page+1)*pageSize,rows.length)+' / '+rows.length+' · page '+(state.page+1)+'/'+pages;grid.querySelector('[data-page-prev]').disabled=state.page===0;grid.querySelector('[data-page-next]').disabled=state.page>=pages-1;manifestStates.set(grid,state)}
    document.querySelectorAll('[data-manifest-grid]').forEach(grid=>updateManifest(grid));
    document.addEventListener('input',e=>{const grid=e.target.closest('[data-manifest-grid]');if(grid&&(e.target.matches('[data-manifest-search]')||e.target.matches('[data-column-filter]')))updateManifest(grid,true)});
    document.addEventListener('change',e=>{const grid=e.target.closest('[data-manifest-grid]');if(grid&&e.target.matches('[data-page-size]'))updateManifest(grid,true)});
    document.addEventListener('click',e=>{const grid=e.target.closest('[data-manifest-grid]');if(!grid)return;const state=manifestStates.get(grid);if(e.target.closest('[data-page-prev]'))state.page--;else if(e.target.closest('[data-page-next]'))state.page++;else{const header=e.target.closest('[data-manifest-sort]');if(!header)return;const column=Number(header.dataset.manifestSort);state.sortDirection=state.sortColumn===column?-state.sortDirection:1;state.sortColumn=column;grid.querySelectorAll('[data-manifest-sort]').forEach(item=>item.textContent=item.dataset.label+' ↕');header.textContent=header.dataset.label+(state.sortDirection>0?' ↑':' ↓');state.page=0}updateManifest(grid)});
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='structureLoading'){detail.hidden=false;detail.innerHTML='<h2>Loading structure…</h2><p class="muted">'+esc(m.path)+'</p>'}if(m.type==='structureDetail')render(m.result);if(m.type==='structureError')detail.innerHTML='<h2>Unable to inspect structure</h2><pre>'+esc(m.message)+'</pre>'});</script></main></body></html>`;
}

async function readStructureManifests(root) {
  const results = [];
  const localManifest = path.join(root, 'manifest.tsv');
  if (fs.existsSync(localManifest)) {
    const lines = (await fs.promises.readFile(localManifest, 'utf8')).trim().split(/\r?\n/);
    if (lines.length && lines[0]) {
      const columns = lines[0].split('\t');
      const rows = lines.slice(1).map(line => {
        const values = line.split('\t'), row = Object.fromEntries(columns.map((column, index) => [column, values[index] || '']));
        row.relative = row.file ? path.join('data', row.file) : '';
        row.exists = Boolean(row.relative && fs.existsSync(path.join(root, row.relative)));
        return row;
      });
      results.push({ directory: path.basename(root), relativeBase: '', rows });
    }
  }
  for (const directory of ['train_structures', 'validation_structures']) {
    const file = path.join(root, directory, 'manifest.tsv');
    if (!fs.existsSync(file)) continue;
    const lines = (await fs.promises.readFile(file, 'utf8')).trim().split(/\r?\n/);
    if (!lines.length || !lines[0]) continue;
    const columns = lines[0].split('\t');
    const rows = lines.slice(1).map(line => {
      const values = line.split('\t');
      const row = Object.fromEntries(columns.map((column, index) => [column, values[index] || '']));
      row.relative = row.file ? path.join(directory, 'data', row.file) : '';
      row.exists = Boolean(row.relative && fs.existsSync(path.join(root, row.relative)));
      return row;
    });
    results.push({ directory, relativeBase: directory, rows });
  }
  // Older preparation outputs did not write manifest.tsv. Inspect their saved metadata.
  for(const directory of ['', 'train_structures','validation_structures']){
    if(results.some(item=>item.relativeBase===directory))continue;
    const folder=path.join(root,directory);if(!fs.existsSync(folder))continue;
    const locations=[folder,path.join(folder,'data')];let hasStructures=false;
    for(const location of locations)if(fs.existsSync(location)&&(await fs.promises.readdir(location)).some(name=>name.endsWith('.preft.pt')))hasStructures=true;
    if(!hasStructures)continue;
    const context={extensionPath:path.resolve(__dirname,'..')};
    const rows=await workbenchServices.backend(context,projectRoot(context),'structure-manifest',{directory:folder});
    for(const row of rows){row.relative=path.join(directory,row.file);row.exists=fs.existsSync(path.join(root,row.relative));}
    results.push({directory:directory||path.basename(root),relativeBase:directory,rows});
  }
  for(const manifest of results){
    const missing=manifest.rows.filter(row=>row.exists&&['num_nodes','num_edges'].some(key=>row[key]===undefined||row[key]===''));
    if(missing.length){
      const context={extensionPath:path.resolve(__dirname,'..')};
      const summaries=await workbenchServices.backend(context,projectRoot(context),'structure-manifest',{directory:path.join(root,manifest.relativeBase)});
      const byFile=new Map(summaries.map(row=>[path.basename(row.file),row]));
      for(const row of missing){const summary=byFile.get(path.basename(row.file));if(summary)for(const [key,value] of Object.entries(summary))if(row[key]===undefined||row[key]==='')row[key]=value;}
    }
    for(const row of manifest.rows){
      if(row.rejected_sample_count==null||row.rejected_sample_count==='')row.rejected_sample_count=Math.max(0,manifestCount(row,'num_input_records','record_indexes')-manifestCount(row,'num_valid_samples','sample_indexes',true));
      if(!row.rejection_log&&Number(row.rejected_sample_count)>0&&fs.existsSync(path.join(root,manifest.relativeBase,'skipped_sources.json')))row.rejection_log='skipped_sources.json';
      if(!row.exists&&row.status==='skipped'){for(const key of ['num_nodes','num_edges'])if(row[key]==null||row[key]==='')row[key]=0;}
    }
  }
  return results;
}

function manifestListLength(value, validOnly = false) {
  if (!value) return 0;
  try {
    const items = JSON.parse(value);
    if (!Array.isArray(items)) return 0;
    return validOnly ? items.filter(item => Number(item) >= 0).length : items.length;
  } catch (_) { return 0; }
}

function manifestCount(row, key, fallbackList, validOnly = false) {
  const raw = row[key], value = Number(raw);
  return raw !== '' && raw !== null && raw !== undefined && Number.isFinite(value) && value >= 0
    ? value : manifestListLength(row[fallbackList], validOnly);
}

function summarizeStructureManifests(manifests) {
  const summarizeRows = (dataset, rows) => {
    let validCompounds = 0, totalSpectra = 0, validSpectra = 0;
    const validSpectraPerCompound = [];
    for (const row of rows) {
      const inputCount = manifestCount(row, 'num_input_records', 'record_indexes');
      const validCount = manifestCount(row, 'num_valid_samples', 'sample_indexes', true);
      const status = String(row.status || '').toLowerCase();
      const validCompound = validCount > 0 && !['error', 'failed', 'no_valid_samples'].includes(status);
      totalSpectra += inputCount;
      validSpectra += validCount;
      if (validCompound) {
        validCompounds++;
        validSpectraPerCompound.push(validCount);
      }
    }
    validSpectraPerCompound.sort((a, b) => a - b);
    const middle = Math.floor(validSpectraPerCompound.length / 2);
    const median = validSpectraPerCompound.length
      ? (validSpectraPerCompound.length % 2 ? validSpectraPerCompound[middle] : (validSpectraPerCompound[middle - 1] + validSpectraPerCompound[middle]) / 2)
      : 0;
    return {
      dataset, totalCompounds: rows.length, validCompounds,
      invalidCompounds: rows.length - validCompounds,
      totalSpectra, validSpectra, invalidSpectra: Math.max(0, totalSpectra - validSpectra),
      spectraPerCompoundMedian: median,
      spectraPerCompoundMean: validSpectraPerCompound.length ? validSpectra / validSpectraPerCompound.length : 0
    };
  };
  const datasets = manifests.map(manifest => summarizeRows(manifest.directory, manifest.rows));
  const total = summarizeRows('All datasets', manifests.flatMap(manifest => manifest.rows));
  return { total, datasets };
}

function statsPercent(valid, total) { return total ? `${(valid / total * 100).toFixed(1)}%` : '—'; }
function statsNumber(value) { return Number(value).toLocaleString('en-US', { maximumFractionDigits: 1 }); }
function datasetStatisticsHtml(stats) {
  const total = stats.total;
  const rows = [total, ...stats.datasets].map(item => `<tr><td>${escapeHtml(item.dataset)}</td><td>${statsNumber(item.totalCompounds)}</td><td>${statsNumber(item.validCompounds)}</td><td>${statsNumber(item.invalidCompounds)}</td><td>${statsPercent(item.validCompounds, item.totalCompounds)}</td><td>${statsNumber(item.totalSpectra)}</td><td>${statsNumber(item.validSpectra)}</td><td>${statsNumber(item.invalidSpectra)}</td><td>${statsPercent(item.validSpectra, item.totalSpectra)}</td><td>${statsNumber(item.spectraPerCompoundMedian)} / ${statsNumber(item.spectraPerCompoundMean)}</td></tr>`).join('');
  return `<section class="dataset-statistics"><h2>Dataset Statistics</h2><p class="muted">A valid compound has at least one valid spectrum and no failed status. Spectrum counts are summed from each structure-manifest row.</p><div class="cards dataset-stat-cards"><article><b>${statsNumber(total.totalCompounds)}</b><span>Total compounds</span></article><article><b>${statsNumber(total.validCompounds)}</b><span>Valid compounds</span><small>${statsPercent(total.validCompounds, total.totalCompounds)} of total</small></article><article><b>${statsNumber(total.totalSpectra)}</b><span>Total spectra</span></article><article><b>${statsNumber(total.validSpectra)}</b><span>Valid spectra</span><small>${statsPercent(total.validSpectra, total.totalSpectra)} of total</small></article></div><div class="table-scroll"><table class="dataset-breakdown"><thead><tr><th>Dataset</th><th>Total compounds</th><th>Valid compounds</th><th>Rejected compounds</th><th>Compound validity</th><th>Total spectra</th><th>Valid spectra</th><th>Rejected spectra</th><th>Spectrum validity</th><th>Valid spectra / compound<br><small>median / mean</small></th></tr></thead><tbody>${rows}</tbody></table></div>${stats.datasets.length ? '' : '<p class="muted">No structure manifest was found, so dataset counts are unavailable.</p>'}</section>`;
}

function manifestTableHtml(manifest, index) {
  const columns = ['file', 'status', 'smiles', 'num_input_records', 'num_valid_samples', 'rejected_sample_count', 'rejection_log', 'num_branch_groups', 'num_teacher_nodes', 'num_transition_states', 'num_positive_transitions', 'num_physical_ion_candidates', 'num_ion_explanations', 'max_ms2_depth', 'num_nodes', 'num_edges', 'assignment_score', 'assignment_score_without_precursor'];
  const rows = manifest.rows.map(row => `<tr>${columns.map(column => { const value = row[column] ?? ''; if(column.startsWith('assignment_score')) return `<td data-value="${escapeHtml(value)}" title="Mean per-sample intensity coverage">${value!==''&&Number.isFinite(Number(value))?(Number(value)*100).toFixed(2)+'%':'Unavailable'}</td>`; if(column==='rejection_log'&&!value)return '<td data-value="">—</td>'; if (column === 'file' && value && row.exists) return `<td data-value="${escapeHtml(value)}"><button class="manifest-file" data-open-file="${encodeURIComponent(row.relative)}">${escapeHtml(value)}</button></td>`; if (column === 'file' && value) return `<td data-value="${escapeHtml(value)}"><span>${escapeHtml(value)}</span><small class="missing-file">not generated (${escapeHtml(row.status || 'missing')})</small></td>`; if (column === 'rejection_log' && value) return `<td data-value="${escapeHtml(value)}"><button class="manifest-file" data-open-file="${encodeURIComponent(path.join(manifest.relativeBase || '', value))}">${escapeHtml(value)}</button></td>`; return `<td data-value="${escapeHtml(value)}">${escapeHtml(value)}</td>`; }).join('')}</tr>`).join('');
  return `<div class="manifest-grid" data-manifest-grid="${index}"><h3>${escapeHtml(manifest.directory)}</h3><div class="manifest-controls"><input type="search" data-manifest-search placeholder="Filter all columns…"><label>Rows <select data-page-size><option>20</option><option selected>50</option><option>100</option><option>250</option></select></label><button data-page-prev>Previous</button><span data-page-label></span><button data-page-next>Next</button></div><div class="table-scroll manifest-scroll"><table><thead><tr>${columns.map((column,columnIndex) => `<th class="sortable" data-manifest-sort="${columnIndex}" data-label="${escapeHtml(column)}">${escapeHtml(column)} ↕</th>`).join('')}</tr><tr class="column-filters">${columns.map((column,columnIndex) => `<th><input data-column-filter="${columnIndex}" placeholder="Filter…" aria-label="Filter ${escapeHtml(column)}"></th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function inspectFragmentTreeResult(context, target) {
  return new Promise((resolve, reject) => {
    const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
    const script = path.join(context.extensionPath, 'src', 'features', 'fragment-tree-result', 'backend.py');
    let root; try { root = projectRoot(context); } catch (error) { reject(error); return; }
    const pythonPath = ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const child = spawn(python, [script], { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath } });
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('error', reject);
    child.on('close', () => { try { const response = JSON.parse(stdout); response.ok ? resolve(response.result) : reject(new Error(response.error)); } catch (_) { reject(new Error(stderr || stdout || 'Structure inspector returned no JSON response.')); } });
    child.stdin.end(JSON.stringify({ path: target }));
  });
}

async function scan(root, dir, depth, limit, out = []) {
  if (depth < 0 || out.length >= limit) return out;
  for (const entry of await fs.promises.readdir(dir, { withFileTypes: true })) {
    if (entry.name === RESULT_NAME || entry.name === 'fragment-tree.clefts-result' || entry.name.startsWith('.')) continue;
    if (entry.isDirectory() && entry.name === 'data' && /(?:train|validation)_structures$/.test(path.basename(dir))) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await scan(root, full, depth - 1, limit, out);
    else { const stat = await fs.promises.stat(full); out.push({ relative: path.relative(root, full), size: stat.size }); }
    if (out.length >= limit) break;
  }
  return out.sort((a,b) => a.relative.localeCompare(b.relative));
}
function summarize(files) { return files.reduce((s,f) => { s.bytes += f.size; if (f.relative.endsWith('.preft.pt') || f.relative.endsWith('.preft') || f.relative.endsWith('.pt')) s.preft++; if (f.relative.endsWith('.tsv')) s.tsv++; return s; }, {preft:0,tsv:0,bytes:0}); }
function formatBytes(n) { if (n < 1024) return `${n} B`; if (n < 1048576) return `${(n/1024).toFixed(1)} KB`; return `${(n/1048576).toFixed(1)} MB`; }
function escapeHtml(v) { return String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function errorHtml(message) { return `<!doctype html><style>${commonCss()}</style><main><h1>Unable to display result</h1><pre>${escapeHtml(message)}</pre></main>`; }

const HELP = {
  trainInput: 'Training input MSDataset path. Required.', validationInput: 'Optional validation input MSDataset path.',
  params: 'Fragmenter parameter JSON used for preprocessing. Required.', outputDir: 'Output root directory. Required.',
  symbols: 'Element symbols that define atom-feature columns. Required and immutable for generated data.',
  maxNode: 'Maximum fragment-tree nodes. Use -1 for no limit.', maxEdge: 'Maximum fragment-tree edges. Use -1 for no limit.',
  smilesColumn: 'Metadata column containing SMILES strings. Default: SMILES.', precursorMzColumn: 'Metadata column containing precursor m/z values. Default: PrecursorMZ.',
  adductTypeColumn: 'Metadata column containing adduct types. Default: AdductType.', collisionEnergyColumn: 'Metadata column containing collision energies. Default: CollisionEnergy.',
  instrumentColumn: 'Optional metadata column containing instrument names. Leave blank when unavailable; for NIST-style data it is commonly InstrumentType.', validationSmilesRatio: 'Target validation SMILES count relative to unique training SMILES. Default: 0.1.',
  tanimotoNumBins: 'Number of similarity intervals used for balanced validation sampling. Default: 10.', tanimotoRadius: 'Morgan fingerprint radius used for validation similarity. Default: 2.',
  tanimotoNBits: 'Morgan fingerprint bit count. Default: 2048.', validationSamplingSeed: 'Random seed for deterministic validation sampling. Default: 0.',
  validationStructuresInputDir: 'Optional existing validation structure directory; mutually exclusive with validation input.',
  numWorkers: 'Number of parallel subprocess workers. Use 1 to disable parallel processing.', chunkSize: 'Number of SMILES groups assigned to each parallel chunk.',
  structureRebuildPolicy: 'When reusing an existing structure directory, decide whether newly added cleavage patterns require rebuilding: match any saved fragment (recommended), match only the root molecule, or always rebuild. An existing output is still skipped unless Overwrite structures is enabled.',
  requirePrecursorPathTargets: 'When enabled, only peak pathways that explicitly pass through a precursor node are saved as supervised assignments.',
  overwrite: 'Overwrite existing .preft.pt structure files.', overwritePreprocessingConfig: 'Overwrite an existing preprocessing configuration without prompting.',
  saveTrainValidRecords: 'Save successfully processed training records as an MSDataset.', saveValidationValidRecords: 'Save successfully processed validation records as an MSDataset.',
  keepParallelTemp: 'Keep temporary parallel-processing files after merging.'
  ,trainDir: 'Directory containing generated training structures and preprocessing_config.pftprep.json.',
  valDir: 'Directory containing generated validation structures with compatible preprocessing settings.',
  initializeFrom: 'Initialize compatible Source-anchored model weights for fine-tuning with a fresh optimizer.',
  molEncoderCheckpoint: 'Pretrained molecular encoder checkpoint used by FragmentTreeFeatureModel.',
  dropout: 'Dropout shared by the constructed fragment-tree model.',
  assignmentScoreThreshold: 'Minimum assignment score accepted for training and the filtered validation view. Reads assignment_scores.tsv from each split. Default: 0.8. Validation also evaluates all samples by combining disjoint above- and below-threshold results without repeating inference.',
  maxSamples: 'Maximum spectra packed into one loaded compound batch.',
  minimumAssignmentScore: 'Drop a prepared sample at training load time if its assignment score (fraction of matched peak intensity, all peaks) is below this value. Data preparation keeps every sample; this only affects what training actually loads. Use 0 to keep every sample.',
  minimumAssignmentScoreWithoutPrecursor: 'Drop a prepared sample at training load time if its assignment score excluding the precursor peak is below this value. Use 0 to keep every sample.',
  experimentName: 'Checkpoint experiment directory name.', ckptId: 'Optional checkpoint identifier to resume.',
  batchSize: 'Number of prepared compound batches per optimizer step.', device: 'PyTorch execution device.',
  epochs: 'Number of complete training epochs.', validationIntervalSteps: 'Run validation after this many optimizer steps; use 0 to disable step validation.',
  stepValidationFraction: 'Fraction of validation compounds (SMILES) used at step intervals: 1 = all, 0.1 = 10%. Uses a fixed subset, rounded up to at least one compound. Epoch-end validation always evaluates all compounds.',
  trainLogIntervalSteps: 'Write training metrics after this many steps.', saveIntervalEpochs: 'Save a checkpoint after this many epochs.',
  saveIntervalSteps: 'Save a checkpoint after this many steps; use 0 to disable.', earlyStoppingPatience: 'Stop after this many non-improving validations; blank disables it.',
  optimizer: 'PyTorch optimizer class name.', lr: 'Optimizer learning rate.', weightDecay: 'Optimizer weight decay.', gradClipNorm: 'Maximum gradient norm; non-positive disables clipping.',
  validateAtStart: 'Run validation before the first training update.', detectAnomaly: 'Enable PyTorch autograd anomaly detection.',
  profilePerformance: 'Write detailed runtime performance profiles.', shuffle: 'Shuffle training structure batches each epoch.'
  ,modelPath: 'Trained fragment-tree model checkpoint (model.pt or a raw state_dict) used for spectrum prediction.'
};

function workbenchHtml(config, trainingConfig, predictionConfig = {}, molConfig = {}) {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss()}${formCss()}${trainingCss()}${trainingWorkbench.css()}${molTraining.css()}${pathDrop.css()}${workbench.css()}${parameterEditor.css()}</style></head><body><main>
  ${layout.header()}
  ${workbench.html()}<nav id="legacyNavigation"><button class="tab" data-app="smarts">SMARTS Search</button><button class="tab" data-app="cleavageViewer">Cleavage Viewer</button><button class="tab" data-app="cleavage">Cleavage Pattern Set</button><button class="tab" data-app="adduct">Adduct Rule Set</button><button class="tab active" data-app="data">Data Preparation</button><button class="tab" data-app="training">Training</button><button class="tab" data-app="molTraining">Mol Training</button><button class="tab" data-app="metrics">Metrics</button><button class="tab" data-app="predict">Predict Spectrum</button></nav>
  ${trainingMetrics.html()}
  ${smartsSearch.html()}
  <div id="cleavageViewerApp" data-app-panel="cleavageViewer" hidden><section><div class="section-title"><div><h2>Single-SMILES Cleavage Viewer</h2><p class="muted">Use the patterns currently loaded in Cleavage Patterns. Enumerate valid simultaneous action sets or add compatible actions interactively.</p></div><button type="button" id="cleavageViewerLoadPatterns">Load Pattern Set</button></div></section><div id="cleavageViewerMount"></div></div>
  <div id="cleavagePageMount">${cleavageUi.html()}</div>
  <div id="adductPageMount">${adductUi.html()}</div>
  ${preparation.html()}
  ${trainingWorkbench.html({pathField,field,optimization:workbench.trainingHtml(),parameters:parameterEditor.importHtml('training'),defaults:workbenchDefaults.html('training')})}${molTraining.html(molConfig)}
  <form id="predictForm" data-app-panel="predict" hidden>
    <section><h2>Model</h2><p class="muted">Points at a checkpoint produced by fragment-tree training (a full <code>model.pt</code>, or a raw generator state_dict). Apply it to load its trained main adduct types before specifying a molecule.</p>${pathField('modelPath','Model checkpoint *','file','predict')}<label>Device<select name="device"><option value="cpu">cpu</option><option value="cuda">cuda</option><option value="mps">mps</option></select></label><div class="actions"><button type="button" id="predictApplyModel" class="primary">Apply Model</button></div><div id="predictModelStatus" class="status idle">No model applied yet.</div></section>
    ${spectrumPrediction.html()}${workbenchDefaults.html('prediction')}
    <section id="predictSingleSection" data-predict-mode="single"><h2>Molecule and conditions</h2><label>SMILES *<input name="smiles" placeholder="CC(=O)Oc1ccccc1C(=O)O"></label><div class="grid">${field('ce','Collision energy (eV) *','text')}<label>Adduct type *<select name="adductType"><option value="">Apply a model first…</option></select></label></div><div class="actions"><button type="button" id="predictPreview">Preview Molecule</button></div><div id="predictMoleculePreview" class="molecule-preview" hidden></div></section>
    <section id="predictResultSection" data-predict-mode="single" hidden><h2>Predicted Spectrum</h2><div id="predictResult"></div></section>
    <footer data-predict-mode="single"><div><div id="predictStatus" class="status idle">Ready</div></div><div class="actions"><button type="submit" class="primary" id="predictSubmit">Predict Spectrum</button></div></footer>
  </form><div id="helpTooltip" role="tooltip"></div><script>const vscode=acquireVsCodeApi(); const initial=${safeJson(config)}; const initialTraining=${safeJson(trainingConfig)};const initialPrediction=${safeJson(predictionConfig)}; ${webviewScript()}${cleavageVisualEditor.script()}${cleavageReactionPreview.script()}${spectrumPrediction.script()}${pathDrop.script()}${workbench.script()}${parameterEditor.script()}${layout.script()}${datasetUpload.script()}${preparation.script()}${workbenchDefaults.script()}${trainingWorkbench.script()}${molTraining.script(molConfig)}${projects.script()}</script></main></body></html>`;
}
function pathField(name,label,kind,form='data') { return `<label data-help="${HELP[name] || ''}">${label}<div class="path"><input name="${name}" data-path-kind="${kind}"><button type="button" data-pick="${name}" data-kind="${kind}" data-form="${form}">Browse</button></div>${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</label>`; }
function field(name,label,type,step='1') { return `<label data-help="${HELP[name] || ''}">${label}<input name="${name}" type="${type}"${type==='number'?` step="${step}"`:''}>${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</label>`; }
function check(name,label) { return `<label class="check" data-help="${HELP[name] || ''}"><input name="${name}" type="checkbox"><span>${label}${HELP[name]?`<small class="field-help">${HELP[name]}</small>`:''}</span></label>`; }
function safeJson(value) { return JSON.stringify(value).replace(/</g, '\\u003c'); }
function webviewScript() { return `
    const form=document.getElementById('form'), trainingForm=document.getElementById('trainingForm'), predictForm=document.getElementById('predictForm'), statusEl=document.getElementById('status'), stop=document.getElementById('stop'),trainingStatusEl=document.getElementById('trainingStatus'),trainingStop=document.getElementById('trainingStop');
    ${cleavageUi.state()}
    ${adductUi.state()}
    ${trainingMetrics.script()}
    ${smartsSearch.script()}
    const htmlEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function setFormConfig(target,c){for(const [k,v] of Object.entries(c)){const el=target.elements[k];if(!el)continue;if(el.type==='checkbox')el.checked=!!v;else el.value=Array.isArray(v)?v.join(target===trainingForm?',':' '):v??'';}}
    function readForm(target,application){const c={application};for(const el of target.elements){if(!el.name)continue;if(el.type==='checkbox')c[el.name]=el.checked;else if(el.type==='number')c[el.name]=el.value===''?'':Number(el.value);else c[el.name]=el.value;}return c;}
    let extraConfig={};
    function setConfig(c){extraConfig={...c};setFormConfig(form,c)}
    function getConfig(){return {...extraConfig,...readForm(form,'fragment-tree-data-preparation')};}
    function getTrainingConfig(){return readForm(trainingForm,'fragment-tree-training')}
    const tooltip=document.getElementById('helpTooltip'); let tooltipTimer;
    document.querySelectorAll('[data-help]').forEach(el=>{el.addEventListener('mouseenter',()=>{tooltipTimer=setTimeout(()=>{const r=el.getBoundingClientRect();tooltip.textContent=el.dataset.help;tooltip.style.left=Math.min(r.left,window.innerWidth-390)+'px';tooltip.style.top=(r.bottom+7)+'px';tooltip.classList.add('visible');},500);});el.addEventListener('mouseleave',()=>{clearTimeout(tooltipTimer);tooltip.classList.remove('visible');});});
    setConfig(initial);setFormConfig(trainingForm,initialTraining);setFormConfig(predictForm,initialPrediction);if(initialPrediction.adductType)predictForm.elements.adductType.dataset.restoreValue=initialPrediction.adductType;setPredictEnabled(false); document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>vscode.postMessage({type:'pick',form:b.dataset.form||'data',field:b.dataset.pick,kind:b.dataset.kind}));
    document.querySelectorAll('[data-app]').forEach(button=>button.onclick=()=>{document.querySelectorAll('[data-app]').forEach(x=>x.classList.toggle('active',x===button));const app=button.dataset.app;if(app==='cleavage'){document.getElementById('cleavagePageMount').append(document.getElementById('cleavageApp'));onCleavageChange=()=>{};}if(app==='adduct'){document.getElementById('adductPageMount').append(document.getElementById('adductApp'));onAdductChange=()=>{};}document.getElementById('homePage').hidden=true;document.getElementById('jobPage').hidden=true;document.getElementById('environmentPage').hidden=true;document.getElementById('metricsApp').hidden=app!=='metrics';document.getElementById('smartsApp').hidden=app!=='smarts';document.getElementById('cleavageViewerApp').hidden=app!=='cleavageViewer';document.getElementById('cleavageApp').hidden=app!=='cleavage';document.getElementById('adductApp').hidden=app!=='adduct';form.hidden=app!=='data';trainingForm.hidden=app!=='training';document.getElementById('molTrainingForm').hidden=app!=='molTraining';predictForm.hidden=app!=='predict';document.getElementById('dataActions').hidden=app!=='data';document.getElementById('appSubtitle').textContent=app==='metrics'?'Training Metrics':app==='smarts'?'SMARTS Search':app==='cleavageViewer'?'Cleavage Viewer':app==='cleavage'?'Cleavage Pattern Set Editor':app==='adduct'?'Fragment Ion Adduct Rule Set Editor':app==='training'?'Fragment Tree Training':app==='molTraining'?'Mol Training':app==='predict'?'Predict Spectrum':'Fragment Tree Data Preparation';if(app==='cleavageViewer')window.dispatchEvent(new Event('cleavageViewer/open'));});
    document.getElementById('cleavageViewerLoadPatterns').onclick=()=>vscode.postMessage({type:'loadCleavagePatternSet'});
    document.getElementById('save').onclick=()=>vscode.postMessage({type:'saveConfig',config:getConfig()}); document.getElementById('load').onclick=()=>vscode.postMessage({type:'loadConfig'}); document.getElementById('openResult').onclick=()=>vscode.postMessage({type:'openResult'}); document.getElementById('openEvaluation').onclick=()=>vscode.postMessage({type:'openEvaluation'});
    ${cleavageUi.script()}
    ${adductUi.script()}
    const initialAdductSet=initial.modelConfig?.fragmenter_params?.fragment_ion_tree_builder?.fragment_ion_adduct_rule_set;if(initialAdductSet)window.dispatchEvent(new MessageEvent('message',{data:{type:'adductRuleSet',value:{fragment_ion_adduct_rule_set:JSON.parse(JSON.stringify(initialAdductSet))},path:''}}));
    form.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'run',config:getConfig()});}; document.getElementById('copyCommand').onclick=()=>vscode.postMessage({type:'copyCommand',config:getConfig()}); stop.onclick=()=>vscode.postMessage({type:'stop'});
    trainingForm.onsubmit=e=>{e.preventDefault();vscode.postMessage({type:'runTraining',config:getTrainingConfig()})};document.getElementById('trainingCopyCommand').onclick=()=>vscode.postMessage({type:'copyTrainingCommand',config:getTrainingConfig()});trainingStop.onclick=()=>vscode.postMessage({type:'stop'});document.getElementById('saveTraining').onclick=()=>vscode.postMessage({type:'saveTrainingConfig',config:getTrainingConfig()});document.getElementById('loadTraining').onclick=()=>vscode.postMessage({type:'loadTrainingConfig'});
    let predictSequence=0,predictWaiters=new Map(),predictListSequence=0,predictListWaiters=new Map();
    function predictSpectrumRequest(payload){return new Promise((resolve,reject)=>{const requestId=++predictSequence;predictWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'predictSpectrum',requestId,payload})})}
    function listPredictAdductsRequest(payload){return new Promise((resolve,reject)=>{const requestId=++predictListSequence;predictListWaiters.set(requestId,{resolve,reject});vscode.postMessage({type:'listPredictAdducts',requestId,payload})})}
    function setPredictEnabled(enabled){predictForm.elements.smiles.disabled=!enabled;predictForm.elements.ce.disabled=!enabled;predictForm.elements.adductType.disabled=!enabled;document.getElementById('predictPreview').disabled=!enabled;document.getElementById('predictSubmit').disabled=!enabled;}
    function renderAdductOptions(adducts){const select=predictForm.elements.adductType,restore=select.dataset.restoreValue;select.innerHTML=adducts.length?adducts.map(a=>'<option value="'+htmlEscape(a)+'">'+htmlEscape(a)+'</option>').join(''):'<option value="">No adducts available</option>';if(restore&&adducts.includes(restore))select.value=restore;delete select.dataset.restoreValue;}
    function resetPredictModelState(){document.getElementById('predictModelStatus').textContent='Model checkpoint changed — apply it again to refresh adduct types.';document.getElementById('predictModelStatus').className='status idle';renderAdductOptions([]);predictForm.elements.adductType.innerHTML='<option value="">Apply a model first…</option>';setPredictEnabled(false);}
    document.getElementById('predictApplyModel').onclick=async()=>{const modelPath=predictForm.elements.modelPath.value.trim(),device=predictForm.elements.device.value,modelStatusEl=document.getElementById('predictModelStatus');if(!modelPath){modelStatusEl.textContent='Model checkpoint path is required.';modelStatusEl.className='status error';return}modelStatusEl.textContent='Loading model…';modelStatusEl.className='status running';setPredictEnabled(false);try{const result=await listPredictAdductsRequest({modelPath,device});renderAdductOptions(result.adducts);modelStatusEl.textContent='Model applied. '+result.adducts.length+' main adduct type(s) available.';modelStatusEl.className='status completed';setPredictEnabled(true);}catch(error){modelStatusEl.textContent=String(error.message||error);modelStatusEl.className='status error';setPredictEnabled(false);}};
    predictForm.elements.modelPath.addEventListener('input',resetPredictModelState);
    function spectrumSvg(peaks,width=900,height=260){if(!peaks.length)return '<p class="muted">No peaks were predicted.</p>';const maxMz=Math.max(...peaks.map(p=>p.mz))*1.05,maxIntensity=Math.max(...peaks.map(p=>p.intensity))||1,padL=55,padB=30,padT=12,padR=20,x=mz=>padL+(mz/maxMz)*(width-padL-padR),y=intensity=>height-padB-(intensity/maxIntensity)*(height-padT-padB);const bars=peaks.map(p=>'<line class="spectrum-bar" x1="'+x(p.mz)+'" x2="'+x(p.mz)+'" y1="'+y(0)+'" y2="'+y(p.intensity)+'"><title>m/z '+p.mz.toFixed(4)+' · intensity '+p.intensity.toPrecision(4)+(p.formula?(' · '+p.formula):'')+'</title></line>').join('');const ticks=[0,.25,.5,.75,1].map(f=>'<text x="'+x(maxMz*f)+'" y="'+(height-8)+'" text-anchor="middle" class="spectrum-tick">'+(maxMz*f).toFixed(0)+'</text>').join('');return '<svg class="spectrum-chart" viewBox="0 0 '+width+' '+height+'"><line x1="'+padL+'" y1="'+padT+'" x2="'+padL+'" y2="'+y(0)+'" class="spectrum-axis"/><line x1="'+padL+'" y1="'+y(0)+'" x2="'+(width-padR)+'" y2="'+y(0)+'" class="spectrum-axis"/>'+bars+ticks+'</svg>'}
    function peakTable(peaks){return '<table><thead><tr><th>#</th><th>m/z</th><th>intensity</th><th>formula</th></tr></thead><tbody>'+peaks.map((p,i)=>'<tr><td>'+(i+1)+'</td><td>'+p.mz.toFixed(6)+'</td><td>'+p.intensity.toPrecision(5)+'</td><td>'+htmlEscape(p.formula||'')+'</td></tr>').join('')+'</tbody></table>'}
    let lastPredictTree=null;
    function predictTreeSvg(tree){if(!tree||!tree.nodes.length)return '<p class="muted">No fragment tree was produced.</p>';const detectedNodes=new Set(tree.nodes.filter(n=>n.annotations&&n.annotations.length).map(n=>n.id)),depth=new Map(tree.nodes.map(n=>[n.id,Math.max(0,n.depth)]));const levels={};for(const n of tree.nodes)(levels[depth.get(n.id)]||(levels[depth.get(n.id)]=[])).push(n);for(const nodes of Object.values(levels))nodes.sort((a,b)=>a.id-b.id);const maxDepth=Math.max(...depth.values(),1),width=Math.max(900,(maxDepth+1)*220),maxRows=Math.max(...Object.values(levels).map(x=>x.length)),height=Math.max(240,maxRows*58+50),pos=new Map();for(const [d,nodes] of Object.entries(levels))nodes.forEach((n,i)=>pos.set(n.id,{x:45+Number(d)/maxDepth*(width-90),y:30+(i+1)*(height-40)/(nodes.length+1)}));const lines=tree.edges.map(e=>{const a=pos.get(e.source),b=pos.get(e.target);return '<line class="tree-edge" x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y+'"><title>edge '+e.id+'</title></line>'}).join('');const nodes=tree.nodes.map(n=>{const p=pos.get(n.id);const shape=n.precursor?'<polygon points="0,-17 17,0 0,17 -17,0"><title>'+htmlEscape(n.smiles)+'</title></polygon>':'<circle r="13"><title>'+htmlEscape(n.smiles)+'</title></circle>';return '<g class="tree-node '+(n.precursor?'precursor ':'')+(detectedNodes.has(n.id)?'detected':'')+'" data-node="'+n.id+'" transform="translate('+p.x+' '+p.y+')">'+shape+'<text text-anchor="middle" dominant-baseline="central">'+n.id+'</text></g>'}).join('');return '<div class="tree-scroll"><svg class="fragment-tree" viewBox="0 0 '+width+' '+height+'">'+lines+nodes+'</svg></div>'}
    document.getElementById('predictPreview').onclick=async()=>{const smiles=predictForm.elements.smiles.value.trim(),adductType=predictForm.elements.adductType.value.trim(),box=document.getElementById('predictMoleculePreview');if(!smiles){box.hidden=true;return}try{const result=await chemistry('depict',{smiles,adducts:adductType?[adductType]:[],usedAdduct:adductType});box.hidden=false;const row=result.adducts[0];box.innerHTML='<div>'+result.svg+'</div><dl><dt>Formula</dt><dd>'+htmlEscape(result.formula)+'</dd><dt>Exact mass</dt><dd>'+result.exactMass.toFixed(6)+'</dd>'+(row?'<dt>Precursor m/z ('+htmlEscape(row.adduct)+')</dt><dd>'+(row.error?('<span class="warning">'+htmlEscape(row.error)+'</span>'):row.mz.toFixed(6))+'</dd>':'')+'</dl>'}catch(e){box.hidden=false;box.innerHTML='<p class="warning">'+htmlEscape(e.message||String(e))+'</p>'}};
    function renderPredictionResult(result){const resultSection=document.getElementById('predictResultSection'),resultBox=document.getElementById('predictResult');document.getElementById('predictStatus').textContent='Predicted '+result.peaks.length+' peaks.';resultSection.hidden=false;lastPredictTree=result.tree;resultBox.innerHTML='<div class="predict-layout"><div>'+spectrumSvg(result.peaks)+peakTable(result.peaks)+'<h3>Fragment Tree</h3><p class="muted">Circles are candidate fragments explored while predicting. Accent-colored fragments back at least one predicted peak. Click a fragment to inspect its structure.</p>'+predictTreeSvg(result.tree)+'<div id="predictFragmentDetail" hidden></div></div><aside class="predict-side"><div class="molecule-preview">'+result.svg+'</div><dl><dt>Formula</dt><dd>'+htmlEscape(result.formula)+'</dd><dt>Exact mass</dt><dd>'+result.exactMass.toFixed(6)+'</dd><dt>Adduct</dt><dd>'+htmlEscape(result.adduct)+'</dd><dt>Precursor m/z</dt><dd>'+result.precursorMz.toFixed(6)+'</dd></dl></aside></div>'}
    window.addEventListener('message',event=>{if(event.data.type==='workbench/predictionResult')renderPredictionResult(event.data.result);});
    predictForm.onsubmit=async e=>{e.preventDefault();const modelPath=predictForm.elements.modelPath.value.trim(),smiles=predictForm.elements.smiles.value.trim(),ce=predictForm.elements.ce.value.trim(),adductType=predictForm.elements.adductType.value.trim(),device=predictForm.elements.device.value,predictStatusEl=document.getElementById('predictStatus'),resultSection=document.getElementById('predictResultSection'),resultBox=document.getElementById('predictResult');if(!modelPath||!smiles||!ce||!adductType){predictStatusEl.textContent='Model checkpoint, SMILES, collision energy, and adduct type are required.';predictStatusEl.className='status error';return}predictStatusEl.textContent='Predicting… this can take a while for large models.';predictStatusEl.className='status running';lastPredictTree=null;try{const result=await predictSpectrumRequest({modelPath,smiles,ce,adductType,device});predictStatusEl.textContent='Predicted '+result.peaks.length+' peaks.';predictStatusEl.className='status completed';renderPredictionResult(result)}catch(error){predictStatusEl.textContent=String(error.message||error);predictStatusEl.className='status error';resultSection.hidden=true}};
    document.getElementById('predictResult').addEventListener('click',async e=>{const nodeEl=e.target.closest('[data-node]');if(!nodeEl||!lastPredictTree)return;const nodeId=Number(nodeEl.dataset.node),node=lastPredictTree.nodes.find(n=>n.id===nodeId);if(!node)return;const detail=document.getElementById('predictFragmentDetail');detail.hidden=false;detail.innerHTML='<p class="muted">Loading…</p>';detail.scrollIntoView({behavior:'smooth',block:'nearest'});try{const adducts=(node.annotations||[]).map(a=>a.adduct);const depicted=await chemistry('depict',{smiles:node.smiles,adducts,usedAdduct:adducts[0]||''});const rows=(node.annotations||[]).map(a=>'<tr><td>'+htmlEscape(a.adduct)+'</td><td>'+htmlEscape(a.ionFormula||'')+'</td><td>'+a.peakMz.toFixed(6)+'</td><td>'+a.peakIntensity.toPrecision(4)+'</td><td>'+(a.probability*100).toFixed(1)+'%</td></tr>').join('');detail.innerHTML='<h4>Fragment node '+nodeId+'</h4><div class="predict-layout"><div class="molecule-preview">'+depicted.svg+'</div><div><dl><dt>SMILES</dt><dd>'+htmlEscape(node.smiles)+'</dd><dt>Formula</dt><dd>'+htmlEscape(node.formula)+'</dd><dt>Exact mass</dt><dd>'+node.exactMass.toFixed(6)+'</dd></dl>'+(rows?('<table><thead><tr><th>Adduct</th><th>Ion formula</th><th>m/z</th><th>intensity</th><th>probability</th></tr></thead><tbody>'+rows+'</tbody></table>'):'<p class="muted">This fragment did not directly back a predicted peak.</p>')+'</div></div>'}catch(error){detail.innerHTML='<p class="warning">'+htmlEscape(error.message||String(error))+'</p>'}});
    window.addEventListener('message',e=>{const m=e.data;if(m.type==='picked'){const target=m.form==='training'?trainingForm:m.form==='predict'?predictForm:form;if(target.elements[m.field])target.elements[m.field].value=m.value;if(m.form==='predict'&&m.field==='modelPath')resetPredictModelState()}if(m.type==='config')setConfig(m.config);if(m.type==='trainingConfig')setFormConfig(trainingForm,m.config);if(m.type==='predictSpectrumResult'){const waiter=predictWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);predictWaiters.delete(m.requestId)}}if(m.type==='predictSpectrumError'){const waiter=predictWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));predictWaiters.delete(m.requestId)}}if(m.type==='listPredictAdductsResult'){const waiter=predictListWaiters.get(m.requestId);if(waiter){waiter.resolve(m.result);predictListWaiters.delete(m.requestId)}}if(m.type==='listPredictAdductsError'){const waiter=predictListWaiters.get(m.requestId);if(waiter){waiter.reject(new Error(m.message));predictListWaiters.delete(m.requestId)}}if(m.type==='status'){statusEl.textContent=m.text;statusEl.className='status '+m.status;stop.disabled=m.status!=='running';if(m.command)document.getElementById('command').textContent=m.command;}if(m.type==='trainingStatus'){trainingStatusEl.textContent=m.text;trainingStatusEl.className='status '+m.status;trainingStop.disabled=m.status!=='running';if(m.command)document.getElementById('trainingCommand').textContent=m.command;}});`;
}

function trainingCss() { return `.model-map{background:color-mix(in srgb,var(--vscode-editor-background) 94%,var(--accent))}.architecture{display:grid;gap:13px;margin-top:15px}.architecture-input,.architecture-merge,.architecture-branches{display:grid;gap:9px}.architecture-branches{grid-template-columns:1fr 1fr 1.35fr}.architecture-modes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mode{display:grid;grid-template-columns:1fr 1fr;gap:8px;border:1px solid var(--border);border-radius:8px;padding:10px}.mode strong{grid-column:1/-1;font-size:10px;letter-spacing:.14em}.training-mode strong{color:#e9b949}.prediction-mode strong{color:#63a8ff}.architecture button{position:relative;text-align:left;min-height:78px;background:var(--vscode-editor-background)}.architecture button b,.architecture button code,.architecture button span{display:block}.architecture button b{color:var(--accent);margin-bottom:5px}.architecture button code{font-size:11px;margin-bottom:5px;white-space:normal}.architecture button span{font-size:11px;opacity:.68}.architecture button.selected{outline:2px solid var(--accent);background:color-mix(in srgb,var(--accent) 14%,var(--vscode-editor-background))}.architecture-input button::after,.architecture-branches button::after,.architecture-merge button::after{content:'↓';position:absolute;left:50%;bottom:-22px;color:var(--accent);z-index:2;font-size:17px}.model-hint{margin:13px 0 0;font-size:12px;opacity:.75}[data-model-block]{scroll-margin-top:12px;transition:border-color .15s,box-shadow .15s}[data-model-block].selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}.model-losses{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}.model-losses span:nth-child(odd){padding:8px 11px;border:1px solid var(--border);border-radius:18px;background:var(--vscode-editor-background)}.model-notes{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0 16px}.model-notes p{margin:0;padding:10px;border-left:3px solid var(--accent);background:var(--vscode-editor-background);font-size:12px;line-height:1.45}.field-help{display:block;margin-top:5px;line-height:1.35;opacity:.7}.check .field-help{margin-top:2px}.check{align-items:flex-start}@media(max-width:850px){.architecture-branches,.architecture-modes,.mode,.model-notes{grid-template-columns:1fr}.mode strong{grid-column:1}.architecture button::after{display:none}}`; }


function deactivate() {}
module.exports = { ResultEditorProvider, activate, deactivate, buildArgs, normalizeConfig, buildTrainingArgs, normalizeTrainingConfig, resultHtml, workbenchHtml, ensureFileSuffix, summarizeStructureManifests };
