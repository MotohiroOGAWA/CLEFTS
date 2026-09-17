const assert=require('assert/strict');
const fs=require('fs');
const os=require('os');
const path=require('path');
const Module=require('module');
const {EventEmitter}=require('events');
const load=Module._load,children=[];
Module._load=function(name,parent,main){if(name==='vscode')return {window:{showInformationMessage(){}},workspace:{getConfiguration:()=>({get:()=> 'python'})}};if(name==='child_process')return {spawn:(python,args,options)=>{const child=new EventEmitter();child.pid=12345;child.unref=()=>{};children.push({child,options,args});return child;}};return load.call(this,name,parent,main);};
const panel=require('../src/workbench/panel');
(async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'clefts-jobs-')),posted=[];
 const context={storageUri:{fsPath:root}},webview={postMessage:data=>posted.push(data)};
 try{
 await panel.startTraining(context,{webview},{appendLine(){}},root,'python',{outputDir:root,epochs:2,batchSize:4,lr:0.001},()=>['-m','clefts.cli']);
 const record=children[0];assert.equal(record.options.stdio[0],'ignore');assert.equal(record.options.detached,process.platform!=='win32');
 let handler,dispose;panel.attach({webview:{...webview,onDidReceiveMessage:fn=>handler=fn},onDidDispose:fn=>dispose=fn},context,()=>root,{},'home');
 try{
 await handler({type:'workbench/ready'});const jobs=posted.find(m=>m.type==='workbench/jobs').jobs;assert.equal(jobs[0].status,'running');assert(fs.existsSync(path.join(root,'jobs',jobs[0].id+'.json')));
 fs.appendFileSync(jobs[0].logPath,'stdout\nstderr\n');await handler({type:'workbench/logs',jobId:jobs[0].id});assert.equal(posted.at(-1).text,'stdout\nstderr\n');
 record.child.emit('close',0,null);const saved=JSON.parse(fs.readFileSync(path.join(root,'jobs',jobs[0].id+'.json')));assert.equal(saved.status,'completed');assert(saved.finishedAt);
 }finally{dispose();}
 await assert.rejects(()=>panel.startTraining(context,{webview},{appendLine(){}},root,'python',{outputDir:root,epochs:0,batchSize:4,lr:0.001},()=>[]),/positive integer/);
 console.log('Training job persistence, process options, logs and completion checks passed.');
 }finally{fs.rmSync(root,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1;});
