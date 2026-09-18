const fs=require('fs');
const path=require('path');
const crypto=require('crypto');
const vscode=require('vscode');
const {spawn}=require('child_process');
function backend(context,root,command,payload){return new Promise((resolve,reject)=>{
 const python=vscode.workspace.getConfiguration('clefts').get('pythonPath','python');
 const child=spawn(python,[path.join(context.extensionPath,'src/workbench/dataset_backend.py')],{cwd:root,env:{...process.env,PYTHONPATH:[root,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)}});
 let output='',errors='',settled=false;const finish=(error,result)=>{if(settled)return;settled=true;clearTimeout(timeout);error?reject(error):resolve(result);};
 const timeout=setTimeout(()=>{child.kill();finish(new Error('Dataset inspection timed out. Try a smaller dataset or inspect the Python environment.'));},120000);
 child.stdout.on('data',data=>{output+=data.toString();if(output.length>8000000){child.kill();finish(new Error('Backend response exceeded the preview limit.'));}});child.stderr.on('data',data=>{errors=(errors+data).slice(-8000);});
 child.on('error',error=>finish(error));child.on('close',code=>{if(settled)return;try{const lines=output.trim().split('\n');const response=JSON.parse(lines.reverse().find(line=>line.startsWith('{'))||'');if(!response.ok)throw new Error(response.error);if(code!==0)throw new Error(errors||'Dataset backend failed.');finish(null,response.result);}catch(error){finish(new Error(error.message||errors));}});
 child.stdin.on('error',()=>{});child.stdin.end(JSON.stringify({command,payload}));
 });}
async function inspectFiles(directory,extensions,limit,depth=5){
 const files=[];let directories=0;
 async function visit(dir,remaining){if(files.length>=limit||remaining<0||++directories>800)return;let entries;try{entries=await fs.promises.readdir(dir,{withFileTypes:true});}catch{return;}
 for(const entry of entries){if(files.length>=limit)return;if(entry.name.startsWith('.')||entry.isSymbolicLink())continue;const file=path.join(dir,entry.name);if(entry.isDirectory())await visit(file,remaining-1);else if(extensions(entry.name)){const stat=await fs.promises.stat(file);files.push({path:file,name:entry.name,size:stat.size,createdAt:stat.mtime.toISOString()});}}}
 await visit(directory,depth);return files;
}
async function library(root){const [samples,models]=await Promise.all([
 inspectFiles(path.join(root,'data/minidata'),name=>/\.(msds|msp|mgf|tsv|csv|parquet)$/i.test(name),24,2),
 inspectFiles(path.join(root,'data/training'),name=>/^(last|best|model)\.pt$/i.test(name),40,9)
 ]);return {samples,models};}
async function openDocumentation(context,root){
 const build=path.join(root,'docs/_build/html'),index=path.join(build,'index.html');
 if(!fs.existsSync(index)){await vscode.commands.executeCommand('markdown.showPreview',vscode.Uri.file(path.join(context.extensionPath,'content/overview.md')));return;}
 const panel=vscode.window.createWebviewPanel('clefts.documentation','CLEFTS Documentation',vscode.ViewColumn.Beside,{enableScripts:true,localResourceRoots:[vscode.Uri.file(build)]});
 const render=async file=>{
 if(file!==build&&!file.startsWith(build+path.sep))throw new Error('Documentation path is outside the build.');
 const nonce=crypto.randomBytes(18).toString('base64');let html=await fs.promises.readFile(file,'utf8');
 html=html.replace(/(src|href)="([^"#]+)"/g,(match,attribute,value)=>{
 if(/^(https?:|data:|mailto:)/.test(value))return match;
 const [relative,hash]=value.split('#'),target=path.resolve(path.dirname(file),relative);
 if(!target.startsWith(build+path.sep))return match;
 if(attribute==='href'&&target.endsWith('.html'))return `href="#${hash||''}" data-doc="${encodeURIComponent(target)}"`;
 return `${attribute}="${panel.webview.asWebviewUri(vscode.Uri.file(target))}${hash?'#'+hash:''}"`;
 });
 html=html.replace('<head>',`<head><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${panel.webview.cspSource} data:; font-src ${panel.webview.cspSource}; style-src ${panel.webview.cspSource} 'unsafe-inline'; script-src ${panel.webview.cspSource} 'nonce-${nonce}';">`).replace(/<script(?=\s|>)/g,`<script nonce="${nonce}"`);
 html=html.replace('</body>',`<script nonce="${nonce}">const vscode=acquireVsCodeApi();document.addEventListener('click',event=>{const link=event.target.closest('a');if(!link)return;if(link.dataset.doc){event.preventDefault();vscode.postMessage({type:'doc',path:decodeURIComponent(link.dataset.doc)});}else if(/^https?:/.test(link.href)){event.preventDefault();vscode.postMessage({type:'external',url:link.href});}});</script></body>`);panel.webview.html=html;
 };
 panel.webview.onDidReceiveMessage(async message=>{try{if(message.type==='doc'&&typeof message.path==='string')await render(message.path);if(message.type==='external'&&/^https:\/\//.test(message.url))await vscode.env.openExternal(vscode.Uri.parse(message.url));}catch(error){vscode.window.showErrorMessage(error.message);}});
 await render(index);
}
function attach(panel,context,projectRoot){
 const post=data=>panel.webview.postMessage(data),latest=new Map();
 panel.webview.onDidReceiveMessage(async message=>{
 if(!message||!['data/preview','data/check','data/preflight','data/model','library/load','library/pick','library/copy','library/open','shell/document','shell/github'].includes(message.type))return;
 try{
 const root=projectRoot(context);
 if(message.type==='shell/document')await openDocumentation(context,root);
 if(message.type==='shell/github')await vscode.env.openExternal(vscode.Uri.parse('https://github.com/MotohiroOGAWA/CLEFTS'));
 if(message.type==='library/load')post({type:'library/data',...await library(root)});
 if(message.type==='library/copy'){if(typeof message.path==='string')await vscode.env.clipboard.writeText(message.path);}
 if(message.type==='library/open'){
 if(typeof message.path!=='string')throw new Error('Invalid resource path.');const stat=await fs.promises.stat(message.path);
 if(stat.isDirectory())await vscode.commands.executeCommand('revealInExplorer',vscode.Uri.file(message.path));else if(/\.(pft(?:\.json)?|clefts-result)$/.test(message.path))await vscode.commands.executeCommand('vscode.openWith',vscode.Uri.file(message.path),'clefts.resultViewer');else await vscode.commands.executeCommand('revealInExplorer',vscode.Uri.file(message.path));
 }
 if(message.type==='library/pick'){
 const allowed=['quickTrainDir','quickValDir','quickTrainOutput','quickCheckpoint','quickPredictModel'];if(!allowed.includes(message.target))return;
 const folder=/Dir|Output/.test(message.target);const selected=await vscode.window.showOpenDialog({canSelectMany:false,canSelectFolders:folder,canSelectFiles:!folder,...(!folder?{filters:{Checkpoint:['pt']}}:{})});if(selected?.[0])post({type:'library/picked',target:message.target,path:selected[0].fsPath});
 }
 if(message.type==='data/model')post({type:'data/model',requestId:message.requestId,...await backend(context,root,'model',{path:message.path})});
 if(['data/preview','data/check'].includes(message.type)){
 if(!['train','validation','home'].includes(message.target)||typeof message.path!=='string'||!message.path.trim())return;
 latest.set(message.target,message.requestId);
 const data=await backend(context,root,'preview',{path:message.path,mapping:message.mapping,validateValues:message.type==='data/check',fragmenterParams:message.fragmenterParams});if(latest.get(message.target)===message.requestId)post({type:message.type,target:message.target,requestId:message.requestId,data});
 }
 if(message.type==='data/preflight')post({type:'data/preflight',requestId:message.requestId,...await backend(context,root,'validate',message.config)});
 }catch(error){post({type:'data/error',source:message.type,target:message.target,requestId:message.requestId,error:error.message});}
 });
}
async function validateRun(context,root,config){return backend(context,root,'validate-config',config);}
module.exports={attach,backend,library,inspectFiles,openDocumentation,validateRun};
