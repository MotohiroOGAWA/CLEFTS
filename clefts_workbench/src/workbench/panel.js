const layout=require('./layout');
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');
const sessions = new Map();
const trainingFlags = { weightDecay:'--weight-decay', gradientClip:'--gradient-clip', absoluteWeight:'--absolute-weight', nextWeight:'--next-weight', negativeWeight:'--negative-weight', intensityWeight:'--intensity-weight' };
const pages = [['HOME','home','Overview'],['DATA','data','Training Data'],['TRAINING','training','New Training'],['TRAINING','molTraining','Mol Training'],['TRAINING','jobs','Training Jobs'],['TRAINING','models','Models'],['TRAINING','metrics','Metrics'],['TRAINING','finetune','Fine-tuning'],['PREDICTION','predict','Single / Batch Prediction'],['VISUALIZATION','viewer','Spectrum / Fragment Viewer'],['VISUALIZATION','smarts','Structure / SMARTS Search'],['SETTINGS','cleavage','Cleavage Patterns'],['SETTINGS','environment','Environment']];
function session(context) {
  const dir=path.join((context.storageUri || context.globalStorageUri).fsPath,'jobs');
  if(sessions.has(dir)) return sessions.get(dir);
  fs.mkdirSync(dir,{recursive:true});
  const value={dir,jobs:[],owned:new Map(),listeners:new Set()};
  for(const file of fs.readdirSync(dir).filter(f=>f.endsWith('.json')))try{const job=JSON.parse(fs.readFileSync(path.join(dir,file),'utf8'));if(!job.id)continue;if(job.status==='running')try{process.kill(job.pid,0);}catch{job.status='failed';job.error='Process ended while the extension was unavailable; exit status is unknown.';}value.jobs.push(job);}catch{}
  value.jobs.sort((a,b)=>b.startedAt.localeCompare(a.startedAt));sessions.set(dir,value);return value;
}
function save(s,job){const file=path.join(s.dir,job.id+'.json');fs.writeFileSync(file+'.tmp',JSON.stringify(job,null,2));fs.renameSync(file+'.tmp',file);for(const listener of s.listeners)listener();}
function register(context, output, projectRoot, open) {
  const commands={importDataset:'data',prepareTrainingData:'data',startTraining:'training',startMolTraining:'molTraining',predictSpectrum:'predict',batchPrediction:'predict',openSpectrumViewer:'viewer',openFragmentExplorer:'viewer',checkEnvironment:'environment'};
  for(const [name,page] of Object.entries(commands))context.subscriptions.push(vscode.commands.registerCommand('clefts.'+name,()=>open(context,output,page)));
  context.subscriptions.push(vscode.commands.registerCommand('clefts.navigate',page=>open(context,output,page)),vscode.window.registerTreeDataProvider('clefts.navigation',{getTreeItem:item=>item,getChildren:item=>item?pages.filter(p=>p[0]===item.label).map(p=>({label:p[2],command:{command:'clefts.navigate',title:p[2],arguments:[p[1]]}})):[...new Set(pages.map(p=>p[0]))].map(label=>({label,collapsibleState:1}))}));
}
async function startTraining(context,panel,output,root,python,config,buildArgs){
  for(const key of Object.keys(trainingFlags))if(config[key]!==undefined&&(!Number.isFinite(Number(config[key]))||Number(config[key])<0))throw new Error(key+' must be non-negative.');
  for(const key of ['epochs','batchSize'])if(!Number.isInteger(Number(config[key]))||Number(config[key])<1)throw new Error(key+' must be a positive integer.');
  if(!(Number(config.lr)>0))throw new Error('Learning rate must be positive.');
  const s=session(context);if(s.jobs.some(j=>j.status==='running'))throw new Error('A training job is already running.');
  const args=buildArgs(config),id=crypto.randomUUID(),logPath=path.join(s.dir,id+'.log');
  const fd=fs.openSync(logPath,'a');let child;
  try{child=spawn(python,args,{cwd:root,detached:process.platform!=='win32',stdio:['ignore',fd,fd],env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)}});}finally{fs.closeSync(fd);}
  const job={id,type:config.workflow==='mol'?'mol-training':'training',name:config.experimentName||path.basename(config.outputDir),status:'running',pid:child.pid,detached:process.platform!=='win32',startedAt:new Date().toISOString(),outputDir:config.outputDir,logPath,command:[python,...args],epochs:config.epochs};
  s.jobs.unshift(job);s.owned.set(id,child);save(s,job);
  child.on('error',error=>{job.status='failed';job.error=error.message;job.finishedAt=new Date().toISOString();save(s,job);});
  child.on('close',(code,signal)=>{if(job.status!=='cancelled')job.status=code===0?'completed':'failed';job.exitCode=code;job.signal=signal;job.finishedAt=new Date().toISOString();s.owned.delete(id);save(s,job);vscode.window.showInformationMessage('CLEFTS training '+job.status+': '+job.name);});child.unref();
  output.appendLine('Started '+job.name);panel.webview.postMessage({type:'workbench/jobSelected',jobId:id});
}
async function trainingDataset(directory) {
  let count=0,bytes=0;const files=[];const pending=[directory];
  while(pending.length){const current=pending.pop();for(const entry of await fs.promises.readdir(current,{withFileTypes:true})){const file=path.join(current,entry.name);if(entry.isDirectory())pending.push(file);else if(entry.isFile()&&entry.name.endsWith('.preft.pt')){const stat=await fs.promises.stat(file);count++;bytes+=stat.size;if(files.length<10)files.push(path.relative(directory,file));}}}
  let summary;try{summary=JSON.parse(await fs.promises.readFile(path.join(directory,'action_statistics.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
  return {count,bytes,files,summary};
}
function tail(file){let fd;try{fd=fs.openSync(file,'r');const size=fs.fstatSync(fd).size,buffer=Buffer.alloc(Math.min(size,64000));fs.readSync(fd,buffer,0,buffer.length,Math.max(0,size-buffer.length));return buffer.toString();}catch{return '';}finally{if(fd!==undefined)fs.closeSync(fd);}}
function attach(panel,context,projectRoot,output,initialPage){
  const s=session(context),post=data=>panel.webview.postMessage(data),update=()=>post({type:'workbench/jobs',jobs:s.jobs});s.listeners.add(update);const timer=setInterval(update,2000);panel.onDidDispose(()=>{clearInterval(timer);s.listeners.delete(update);});
  panel.webview.onDidReceiveMessage(async (/** @type {import('./messages').WorkbenchRequest} */ message)=>{
    if(typeof message?.type!=='string'||!message.type.startsWith('workbench/'))return;
    try{
      if(message.type==='workbench/ready'){update();post({type:'workbench/navigate',page:initialPage});}
      if(message.type==='workbench/logs'){if(typeof message.jobId!=='string')throw new Error('Invalid job ID.');const job=s.jobs.find(j=>j.id===message.jobId);if(job)post({type:'workbench/logs',job,text:tail(job.logPath)});}
      if(message.type==='workbench/stop'){const job=s.jobs.find(j=>j.id===message.jobId);if(!job||job.status!=='running')return;if(!s.owned.has(job.id))throw new Error('Restored jobs must be verified in the terminal before stopping: a PID may have been reused.');stopChild(s.owned.get(job.id),job.detached);job.status='cancelled';save(s,job);}
      if(message.type==='workbench/predictionResult'){const job=s.jobs.find(j=>j.id===message.jobId);if(job?.resultsPath&&job.type==='prediction')post({type:'workbench/predictionResult',result:JSON.parse(await fs.promises.readFile(job.resultsPath,'utf8'))});}
      if(message.type==='workbench/openOutput'){const job=s.jobs.find(j=>j.id===message.jobId);if(job)await vscode.commands.executeCommand('revealInExplorer',vscode.Uri.file(job.outputDir));}
      if(message.type==='workbench/settings')await vscode.commands.executeCommand('workbench.action.openSettings','clefts');
      if(message.type==='workbench/terminal')vscode.window.createTerminal({name:'CLEFTS',cwd:projectRoot(context)}).show();
      if(message.type==='workbench/trainingInspect'){
        if(typeof message.path!=='string'||!message.path.trim())throw new Error('Select a dataset directory.');
        if(!['trainDir','valDir'].includes(message.field))throw new Error('Select a training or validation dataset.');
        try{const stats=await trainingDataset(path.resolve(projectRoot(context),message.path));post({type:'workbench/trainingDataset',field:message.field,path:message.path,...stats});}catch(error){post({type:'workbench/trainingDataset',field:message.field,path:message.path,error:error.message});}
      }
      if(message.type==='workbench/trainingDataset'){
        if(!['trainDir','valDir'].includes(message.field)||typeof message.path!=='string'||!message.path.trim())throw new Error('Select a dataset directory.');
        try{const stats=await trainingDataset(path.resolve(projectRoot(context),message.path));post({type:'workbench/trainingDataset',field:message.field,path:message.path,...stats});}catch(error){post({type:'workbench/trainingDataset',field:message.field,path:message.path,error:error.message});}
      }
      if(message.type==='workbench/modelConfig'){if(typeof message.path!=='string'||!message.path.trim())throw new Error('Select a model config JSON first.');const file=path.resolve(projectRoot(context),message.path);if(message.open)await vscode.window.showTextDocument(vscode.Uri.file(file));else post({type:'workbench/modelConfig',config:JSON.parse(await fs.promises.readFile(file,'utf8'))});}
      if(message.type==='workbench/saveModelConfig'){
        if(typeof message.json!=='string' || message.json.length>2000000)throw new Error('Invalid model JSON payload.');
        const config=JSON.parse(message.json);
        if (!config || typeof config !== 'object' || Array.isArray(config)) throw new Error('Model configuration must be a JSON object.');
        const target=await vscode.window.showSaveDialog({filters:{JSON:['json']},defaultUri:vscode.Uri.file(path.join(projectRoot(context),'workbench-model.json'))});
        if(target){await fs.promises.writeFile(target.fsPath,JSON.stringify(config,null,2)+'\n');post({type:'picked',form:'training',field:'params',value:target.fsPath});}
      }
      if(message.type==='workbench/environment'){
        const python=vscode.workspace.getConfiguration('clefts').get('pythonPath','python'),root=projectRoot(context);
        const child=spawn(python,[path.join(context.extensionPath,'src/workbench/environment.py')],{cwd:root,env:{...process.env,PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)}});let stdout='',stderr='';const timeout=setTimeout(()=>child.kill(),20000);
        child.stdout.on('data',chunk=>{stdout=(stdout+chunk).slice(-32000);});child.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-8000);});child.on('error',error=>{clearTimeout(timeout);post({type:'workbench/environment',error:error.message});});child.on('close',code=>{clearTimeout(timeout);try{if(code!==0)throw new Error(stderr||'Environment check failed.');post({type:'workbench/environment',data:{...JSON.parse(stdout),root,python}});}catch(error){post({type:'workbench/environment',error:error.message});}});
      }
    }catch(error){post({type:'workbench/error',error:error.message});}
  });
}
function secure(html,webview){const nonce=crypto.randomBytes(18).toString('base64');return html.replace('<head>',`<head><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data:; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">`).replace(/<script>/g,`<script nonce="${nonce}">`);}
function html(){return layout.sidebar()+`<p id="workbenchError" class="danger" role="alert"></p><div id="homePage" hidden>${layout.home()}</div><div id="jobPage" hidden><section><h2>Training Jobs</h2><div id="allJobs"></div></section><section id="jobDetail" hidden><h2 id="jobTitle"></h2><p id="jobProgress"></p><progress id="jobProgressBar" max="100"></progress><div class="actions"><button id="jobStop" class="danger">Stop Training</button><button id="jobMetrics">Metrics</button><button id="jobOutput">Reveal Output</button></div><div id="jobMetricsPanel"><h3>Loss · Train / Validation</h3><div id="jobLossChart"></div><div id="jobActionMetrics" class="grid"></div></div><h3>Logs</h3><input id="logFilter" type="search" placeholder="Filter logs…" aria-label="Filter logs"><label class="check"><input id="logAuto" type="checkbox" checked>Auto Scroll</label><button id="logCopy">Copy</button><button id="logClear">Clear</button><pre id="jobLogs" role="log"></pre></section></div><div id="environmentPage" hidden><section><h2>Environment</h2><div class="actions"><button id="testEnvironment" class="primary">Test Environment</button><button id="openCleftsTerminal">Open Terminal</button></div><pre id="environmentDetails">Test Python, CLEFTS, PyTorch, RDKit and CUDA.</pre></section></div>`;}
function trainingHtml(){return `<section><h2>Optimization and Loss</h2><label>Experiment Name<input name="experimentName" value="exp_main" placeholder="exp_main"></label><button type="button" id="trainingLossToggle" aria-expanded="true">Optimization and Loss Settings</button><div id="trainingLossFields"><div class="grid">${Object.keys(trainingFlags).map(key=>`<label><span data-help="${({weightDecay: 'AdamW L2 weight decay applied during optimizer updates.', gradientClip: 'Maximum gradient norm; zero disables clipping.', absoluteWeight: 'Multiplier for primitive absolute action filtering loss.', nextWeight: 'Multiplier for the multi-positive next action and EOS loss.', negativeWeight: 'Relative contribution of eligible negative primitive actions.', intensityWeight: 'Multiplier for the materialized fragment formula intensity loss.'})[key]}">${({weightDecay:'Weight decay',gradientClip:'Gradient clip (0 disables)',absoluteWeight:'Absolute action loss weight',nextWeight:'Next action loss weight',negativeWeight:'Negative action weight',intensityWeight:'Fragment intensity loss weight'})[key]}</span><input name="${key}" type="number" min="0" step="any" value="${key==='weightDecay'?0.01:key==='gradientClip'?0:1}"></label>`).join('')}</div><details><summary>How the losses work</summary><p>Absolute action loss learns which primitive actions survive the first filter. Next action loss learns valid actions and EOS from normalized states. Intensity loss fits the materialized fragment spectrum.</p></details></div></section>`;}
function css(){return layout.css();}
function script(){return `(${client.toString()})();`;}
function client(){
  const el=id=>document.getElementById(id);let jobs=[],selected='',logs='';
  function navigate(page){if(page==='models'){navigate('home');document.querySelector('[data-resource-tab=models]').click();document.querySelector('#navigationRail [data-page=home]').classList.remove('active');document.querySelector('#navigationRail [data-page=models]').classList.add('active');return;}document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===page));for(const id of ['homePage','jobPage','environmentPage'])el(id).hidden=true;if(['home','jobs','environment','viewer'].includes(page)){document.querySelectorAll('[data-app-panel]').forEach(n=>n.hidden=true);for(const id of ['metricsApp','fineTuneApp','smartsApp','cleavageApp'])el(id).hidden=true;el('dataActions').hidden=true;if(page==='viewer'){vscode.postMessage({type:'openResult'});page='home';}el(page==='home'?'homePage':page==='jobs'?'jobPage':'environmentPage').hidden=false;el('appSubtitle').textContent=page==='home'?'Overview':page==='jobs'?'Training Jobs':'Environment';}else document.querySelector('#legacyNavigation [data-app="'+page+'"]').click();vscode.setState({...vscode.getState(),page});}
  document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>navigate(b.dataset.page));
  function table(target){const box=el(target);box.replaceChildren();if(!jobs.length){box.textContent='No training jobs yet. Start your first training from New Training.';return;}const t=document.createElement('table'),head=document.createElement('tr');for(const title of ['Name','Type','Status','Started','Duration','Actions']){const th=document.createElement('th');th.textContent=title;head.append(th);}t.append(head);for(const job of jobs){const row=document.createElement('tr');for(const value of [job.name,job.type,job.status,new Date(job.startedAt).toLocaleString(),Math.round((new Date(job.finishedAt||Date.now())-new Date(job.startedAt))/1000)+' s']){const td=document.createElement('td');td.textContent=value;row.append(td);}row.children[2].className='job-'+job.status;const td=document.createElement('td'),button=document.createElement('button');button.textContent='Open / Logs';button.onclick=()=>{selected=job.id;navigate('jobs');load();};td.append(button);const copy=document.createElement('button');copy.textContent='Copy Path';copy.onclick=()=>vscode.postMessage({type:'library/copy',path:job.outputDir});td.append(copy);const reveal=document.createElement('button');reveal.textContent='Results';reveal.onclick=()=>{if(job.type==='preparation'&&job.status==='completed'){vscode.postMessage({type:'library/open',path:job.outputDir+'/train_structures/fragment-tree.pft.json'});}else if(job.type==='prediction'&&job.resultsPath){navigate('predict');vscode.postMessage({type:'workbench/predictionResult',jobId:job.id});}else{selected=job.id;navigate('jobs');load();}};td.append(reveal);row.append(td);t.append(row);}box.append(t);}
  function chart(events){
    const box=el('jobLossChart');box.replaceChildren();
    const series=['train_loss','validation_loss'];const values=events.flatMap(e=>series.map(k=>Number(e[k]))).filter(Number.isFinite);
    if(!values.length){box.textContent='Loss charts appear after the first epoch.';return;}
    const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 600 180');svg.style.width='100%';svg.style.maxHeight='220px';
    const max=Math.max(...values,0.001),count=Math.max(events.length-1,1);
    series.forEach((key,i)=>{const line=document.createElementNS(ns,'polyline');line.setAttribute('points',events.filter(e=>Number.isFinite(Number(e[key]))).map((e,j)=>(20+560*j/count)+','+(155-130*Number(e[key])/max)).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',i?'var(--vscode-charts-orange,#d18616)':'var(--vscode-charts-blue,#3794ff)');line.setAttribute('stroke-width','2');svg.append(line);const label=document.createElementNS(ns,'text');label.setAttribute('x',String(20+i*220));label.setAttribute('y','175');label.setAttribute('fill','var(--vscode-foreground)');label.textContent=key;svg.append(label);});box.append(svg);
    const last=events.at(-1);el('jobActionMetrics').replaceChildren();for(const [key,value] of Object.entries(last||{}).filter(([key])=>/validation.*(recall|precision|accuracy)/.test(key)).slice(0,12)){const card=document.createElement('p');card.textContent=key+': '+Number(value).toFixed(4);el('jobActionMetrics').append(card);}
  }
  function load(){if(selected)vscode.postMessage({type:'workbench/logs',jobId:selected});}
  function renderLogs(){el('jobLogs').textContent=logs.split('\n').filter(line=>line.toLowerCase().includes(el('logFilter').value.toLowerCase())).join('\n');if(el('logAuto').checked)el('jobLogs').scrollTop=el('jobLogs').scrollHeight;}
  el('logFilter').oninput=renderLogs;el('logCopy').onclick=()=>navigator.clipboard.writeText(el('jobLogs').textContent);el('logClear').onclick=()=>{logs='';renderLogs();};el('jobStop').onclick=()=>vscode.postMessage({type:'workbench/stop',jobId:selected});el('jobOutput').onclick=()=>vscode.postMessage({type:'workbench/openOutput',jobId:selected});
  el('jobMetrics').onclick=()=>{const job=jobs.find(j=>j.id===selected);if(job){el('metricsDirectory').value=job.outputDir;navigate('metrics');el('metricsLoad').click();}};
  el('testEnvironment').onclick=()=>{el('environmentDetails').textContent='Checking…';vscode.postMessage({type:'workbench/environment'});};el('workbenchSettings').onclick=()=>vscode.postMessage({type:'workbench/settings'});el('openCleftsTerminal').onclick=()=>vscode.postMessage({type:'workbench/terminal'});
  el('trainingStop').onclick=()=>vscode.postMessage({type:'workbench/stop',jobId:selected});
  el('trainingLossToggle').onclick=()=>{const fields=el('trainingLossFields');fields.hidden=!fields.hidden;el('trainingLossToggle').setAttribute('aria-expanded',String(!fields.hidden));};
  el('trainingFineTuneToggle').onclick=()=>{const fields=el('trainingFineTuneFields');fields.hidden=!fields.hidden;el('trainingFineTuneToggle').setAttribute('aria-expanded',String(!fields.hidden));};
  window.addEventListener('message',event=>{const m=event.data;if(m.type==='workbench/navigate')navigate(m.page);if(m.type==='workbench/jobs'){jobs=m.jobs;table('recentJobs');table('allJobs');load();}if(m.type==='workbench/jobSelected'){selected=m.jobId;navigate('jobs');load();}if(m.type==='workbench/logs'&&m.job.id===selected){logs=m.text;renderLogs();el('jobDetail').hidden=false;el('jobTitle').textContent=m.job.name+' · '+m.job.status;el('jobMetricsPanel').hidden=m.job.type!=='training';el('jobMetrics').hidden=m.job.type!=='training';el('jobStop').disabled=m.job.status!=='running';const events=logs.split('\n').flatMap(line=>{try{const item=JSON.parse(line);return item.event==='epoch_end'?[item]:[];}catch{return [];}}),last=events.at(-1);chart(events);el('jobProgress').textContent=m.job.error||(m.job.type!=='training'?m.job.type+' · '+m.job.status:last?'Epoch '+last.epoch+' / '+m.job.epochs:'Waiting for epoch metrics…');if(last)el('jobProgressBar').value=100*last.current/last.total;else el('jobProgressBar').removeAttribute('value');}if(m.type==='workbench/environment'){el('environmentDetails').textContent=m.error||JSON.stringify(m.data,null,2);el('homeEnvironment').textContent=m.error?'CLEFTS environment unavailable: '+m.error:'CLEFTS environment ready · PyTorch '+m.data.torch+' · '+(m.data.cuda?'CUDA '+m.data.gpu:'CPU');}if(m.type==='workbench/error')el('workbenchError').textContent=m.error;});
  navigate('home');vscode.postMessage({type:'workbench/ready'});vscode.postMessage({type:'workbench/environment'});
}
function stopChild(child,detached=false){child.cleftsCancelled=true;if(process.platform==='win32')spawn('taskkill',['/pid',String(child.pid),'/T','/F'],{windowsHide:true});else if(detached)process.kill(-child.pid,'SIGTERM');else child.kill('SIGTERM');}
function observeJob(context,info){
 const s=session(context),id=crypto.randomUUID();const job={id,name:info.name,type:info.type,status:'running',startedAt:new Date().toISOString(),outputDir:info.outputDir||s.dir,command:info.command||[],pid:info.child?.pid,logPath:path.join(s.dir,id+'.log'),detached:!!info.detached};
 fs.writeFileSync(job.logPath,'');if(info.child)s.owned.set(id,info.child);s.jobs.unshift(job);save(s,job);
 let finished=false;
 return {job,log:text=>fs.appendFileSync(job.logPath,text),finish:({code,signal,error,result}={})=>{if(finished)return;finished=true;job.status=info.child?.cleftsCancelled||job.status==='cancelled'?'cancelled':error?'failed':signal?'cancelled':code===0?'completed':'failed';job.exitCode=code;job.error=error;job.finishedAt=new Date().toISOString();if(result){job.resultsPath=path.join(s.dir,id+'.result.json');fs.writeFileSync(job.resultsPath,JSON.stringify(result));}s.owned.delete(id);save(s,job);}};
}
module.exports={register,attach,startTraining,html,trainingHtml,css,script,secure,trainingFlags,observeJob,stopChild,trainingDataset};
