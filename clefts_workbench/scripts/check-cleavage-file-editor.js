const assert=require('assert/strict'),fs=require('fs'),path=require('path'),Module=require('module');
const {JSDOM,VirtualConsole}=require('jsdom');
const providers=new Map();let changed,document,currentText,allowEdit=true;
const vscode={window:{registerCustomEditorProvider:(name,provider)=>{providers.set(name,provider);return {dispose(){}};}},workspace:{onDidChangeTextDocument:fn=>{changed=fn;return {dispose(){}};},applyEdit:async edit=>{if(!allowEdit)return false;currentText=edit.text;changed({document});return true;}},Range:class{},WorkspaceEdit:class{replace(uri,range,text){this.text=text;}}};
const load=Module._load;Module._load=function(name,parent,main){return name==='vscode'?vscode:load.call(this,name,parent,main);};
const feature=require('../src/features/cleavage-pattern-set/editor');Module._load=load;
feature.register({subscriptions:[]});
const manifest=require('../package.json');
for(const suffix of ['*.cleavage.json','*.clevage.json'])assert(manifest.contributes.customEditors.some(item=>item.viewType==='clefts.cleavagePatternEditor'&&item.priority==='default'&&item.selector.some(selector=>selector.filenamePattern===suffix)));
assert(manifest.activationEvents.includes('onCustomEditor:clefts.cleavagePatternEditor'));
async function open(viewType,value){
 currentText=JSON.stringify(value);const messages=[];let handler,dispose;
 document={uri:{toString:()=>'/tmp/pattern.json'},getText:()=>currentText,positionAt:value=>value};
 const panel={webview:{postMessage:message=>messages.push(message),onDidReceiveMessage:fn=>handler=fn},onDidDispose:fn=>dispose=fn};
 providers.get(viewType).resolveCustomTextEditor(document,panel);await handler({type:'ready'});
 return {panel,messages,send:message=>handler(message),dispose};
}
(async()=>{
 const original=JSON.parse(fs.readFileSync(path.join(__dirname,'../../clefts/presets/fragment/cleavage_pattern_sets/cleavage_patterns/nonring_c_single_bond.cleavage.json'),'utf8'));
 const single=await open('clefts.cleavagePatternEditor',original);let dom;
 try{
  assert.equal(single.messages.at(-1).type,'document');
  const errors=[],sent=[],console=new VirtualConsole();console.on('jsdomError',error=>errors.push(error.message));
  dom=new JSDOM(single.panel.webview.html,{runScripts:'dangerously',virtualConsole:console,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:message=>sent.push(JSON.parse(JSON.stringify(message)))});}});
  const w=dom.window,d=w.document,post=message=>w.dispatchEvent(new w.MessageEvent('message',{data:message}));
  post(single.messages.at(-1));assert.equal(d.querySelector('h1').textContent,'Cleavage Pattern');assert(d.getElementById('addPattern').hidden);assert(d.getElementById('setName').closest('section').hidden);assert.equal(d.querySelectorAll('[data-remove-pattern]').length,0);assert.equal(d.querySelectorAll('.pattern').length,1);
  const input=d.querySelector('[data-key="name"]');input.value='Edited <pattern>';input.dispatchEvent(new w.Event('input',{bubbles:true}));
  d.querySelector('[data-add-product]').click();const product=d.querySelector('[data-product="2"][data-key="smarts"]');product.value='[!#1:2]';product.dispatchEvent(new w.Event('input',{bubbles:true}));
  await new Promise(resolve=>setTimeout(resolve,210));await single.send(sent.at(-1));const saved=JSON.parse(currentText);
  assert.equal(saved.name,'Edited <pattern>');assert.equal(saved.reactant_smarts,original.reactant_smarts);assert.equal(saved.products.length,3);assert.equal(saved.products[2].smarts,'[!#1:2]');assert(!('cleavage_pattern_set' in saved));
  // Undo/external edits are reflected back into the form.
  currentText=JSON.stringify(original);changed({document});post(single.messages.at(-1));assert.equal(d.querySelector('[data-key="name"]').value,original.name);assert.equal(d.querySelectorAll('.product').length,2);
  await single.send({type:'update',value:{cleavage_pattern_set:{name:'',patterns:[]}}});assert.equal(single.messages.at(-1).type,'error');assert.deepEqual(JSON.parse(currentText),original);
  allowEdit=false;await single.send({type:'update',value:{cleavage_pattern_set:{name:'',patterns:[{...original,name:'Rejected'}]}}});assert.equal(single.messages.at(-1).type,'error');assert.deepEqual(JSON.parse(currentText),original);allowEdit=true;
  currentText='{"cleavage_pattern_set":{"patterns":[]}}';changed({document});assert.equal(single.messages.at(-1).type,'error');assert.deepEqual(errors,[]);
 }finally{dom?.window.close();single.dispose();}
 const value={cleavage_pattern_set:{name:'Original set',patterns:[original]}},set=await open('clefts.cleavagePatternSetEditor',value);
 try{assert.deepEqual(set.messages.at(-1).value,value);await set.send({type:'update',value});assert.deepEqual(JSON.parse(currentText),value);assert(set.panel.webview.html.includes('<h1>Cleavage Pattern Set</h1>'));}finally{set.dispose();}
 console.log('Single cleavage editor registration, form editing, flat JSON persistence, products, external edits, invalid documents and existing set editor passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
