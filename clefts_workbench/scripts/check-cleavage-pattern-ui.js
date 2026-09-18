const assert=require('assert/strict'),fs=require('fs'),path=require('path'),os=require('os'),Module=require('module');
const {JSDOM,VirtualConsole}=require('jsdom');
const project=path.resolve(__dirname,'../..');let selected;
const mock={workspace:{workspaceFolders:[{uri:{fsPath:project}}],getConfiguration:()=>({get:(key,fallback)=>fallback})},window:{showOpenDialog:async()=>selected}};
const load=Module._load;Module._load=function(name,parent,main){return name==='vscode'?mock:load.call(this,name,parent,main);};
const extension=require('../src/extension'),preset=require('../../clefts/presets/spectrum_generator_params/source_anchored_pos_model_config.json');
const messages=[],errors=[],console=new VirtualConsole();console.on('jsdomError',error=>errors.push(error.detail?.stack||error.message));
const dom=new JSDOM(extension.workbenchHtml({modelConfig:preset},{modelConfig:preset}),{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:console,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){}disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:s=>s};}});
(async()=>{
 const w=dom.window,d=w.document,post=data=>w.dispatchEvent(new w.MessageEvent('message',{data}));
 try{
  const nav=d.querySelector('#navigationRail [data-page=cleavage]');assert(nav);
  assert.equal(nav.closest('details').querySelector('summary span').textContent,'Data');
  nav.click();assert(!d.getElementById('cleavageApp').hidden);assert(d.getElementById('form').hidden);
  assert(!d.getElementById('cleavageEmpty').hidden);
  d.querySelector('#navigationRail [data-page=data]').click();d.getElementById('dataEditCleavagePatterns').click();
  assert(!d.getElementById('cleavageApp').hidden);assert(nav.classList.contains('active'));
  const original=preset.fragmenter_params.fragment_ion_tree_builder.cleavage_pattern_set.patterns.length;
  assert.equal(d.querySelectorAll('#cleavagePatterns .pattern-card').length,original);
  d.getElementById('addCleavagePattern').click();const index=original;
  function input(selector,value){const element=d.querySelector(selector);assert(element,selector);element.value=value;element.dispatchEvent(new w.Event('input',{bubbles:true}));}
  input('[data-pattern="'+index+'"][data-key=name]','added_carbon_oxygen');
  input('[data-pattern="'+index+'"][data-key=reactant_smarts]','[#6:1]-[#8:2]');
  d.querySelector('[data-add-product="'+index+'"]').click();
  input('[data-pattern="'+index+'"][data-product="0"][data-key=name]','carbon');
  input('[data-pattern="'+index+'"][data-product="0"][data-key=smarts]','[#6:1]');
  d.getElementById('saveCleavage').click();assert.equal(messages.at(-1).type,'saveCleavagePatternSet');
  assert.equal(messages.at(-1).value.cleavage_pattern_set.patterns[index].name,'added_carbon_oxygen');
  d.getElementById('applyParameterPatterns').click();assert(!d.getElementById('form').hidden);
  const config=w.eval('getConfig()'),added=config.fragmenterParams.fragment_ion_tree_builder.cleavage_pattern_set.patterns[index];
  assert.equal(config.fragmenterParams.fragment_ion_tree_builder.cleavage_pattern_set.patterns.length,original+1);
  assert.deepEqual(JSON.parse(JSON.stringify(added)),{name:'added_carbon_oxygen',reactant_smarts:'[#6:1]-[#8:2]',products:[{name:'carbon',smarts:'[#6:1]'}]});
  const args=extension.buildArgs(config);assert.deepEqual(JSON.parse(args[args.indexOf('--params-json')+1]).fragment_ion_tree_builder.cleavage_pattern_set.patterns[index],JSON.parse(JSON.stringify(added)));
  assert.equal(preset.fragmenter_params.fragment_ion_tree_builder.cleavage_pattern_set.patterns.length,original);
  d.getElementById('dataAddVisualPattern').click();assert(!d.getElementById('visualBuilder').hidden);assert.equal(d.activeElement.id,'builderSmiles');
  assert(!d.getElementById('applyParameterPatterns').hidden);
  const graph={atoms:[{index:0,x:0,y:0,symbol:'C'},{index:1,x:1,y:1,symbol:'O'}],bonds:[{index:0,begin:0,end:1,order:1}]};
  const drawing=d.getElementById('drawMolecule').onclick(),request=messages.at(-1);
  post({type:'chemistryResult',requestId:request.requestId,result:graph});await drawing;
  w.eval('productGraph=JSON.parse(JSON.stringify(molGraph));renderProduct()');
  for(const id of ['moleculeCanvas','productCanvas']){
   const canvas=d.getElementById(id),svg=canvas.querySelector('svg'),lasso=svg.querySelector('.selection-lasso');
   svg.createSVGPoint=()=>({x:0,y:0,matrixTransform(){return {x:this.x,y:this.y};}});
   svg.getScreenCTM=()=>({inverse:()=>({})});svg.setPointerCapture=()=>{};
   assert(lasso.hasAttribute('hidden'));assert.equal(w.getComputedStyle(lasso).display,'none');
   canvas.onpointerdown({button:0,clientX:10,clientY:20,pointerId:1,target:svg,preventDefault(){}});
   canvas.onpointermove({clientX:30,clientY:40});
   assert(!lasso.hasAttribute('hidden'));assert.equal(w.getComputedStyle(lasso).display,'block');
   assert.equal(lasso.getAttribute('points'),'10,20 30,40');
   canvas.onpointermove({clientX:50,clientY:60});assert.equal(lasso.getAttribute('points'),'10,20 30,40 50,60');
   canvas.onpointercancel();assert(canvas.querySelector('.selection-lasso').hasAttribute('hidden'));
  }
  await d.getElementById('loadCleavage').ondrop({preventDefault(){},stopPropagation(){},dataTransfer:{getData:type=>type==='text/uri-list'?'file:///tmp/imported%20set.clevageset.json':'',files:[]}});
  assert.equal(messages.at(-1).type,'loadCleavagePatternSet');assert.equal(messages.at(-1).path,'/tmp/imported set.clevageset.json');
  const pattern={name:'imported',reactant_smarts:'[#6:1]-[#8:2]',products:[{name:'carbon',smarts:'[#6:1]'}]};
  await d.getElementById('loadSinglePattern').ondrop({preventDefault(){},stopPropagation(){},dataTransfer:{getData:()=>'',files:[{text:async()=>JSON.stringify(pattern)}]}});
  assert.equal(messages.at(-1).type,'loadCleavagePattern');assert.deepEqual(JSON.parse(messages.at(-1).json),pattern);
  post({type:'cleavagePatternLoaded',pattern,index:-1});assert.equal(d.querySelectorAll('#cleavagePatterns .pattern-card').length,original+2);
  post({type:'cleavagePatternError',message:'Invalid JSON'});assert.equal(d.getElementById('cleavageError').textContent,'Invalid JSON');
  post({type:'cleavagePatternLoaded',pattern,index:-1});assert.equal(d.getElementById('cleavageError').textContent,'');assert.deepEqual(errors,[]);
 }finally{dom.window.close();}
 const filename=require.resolve('../src/extension'),compiled=new Module(filename,module);compiled.filename=filename;compiled.paths=module.paths;
 compiled._compile(fs.readFileSync(filename,'utf8')+'\nmodule.exports.readCleavageImport=readCleavageImport;',filename);
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'clefts-pattern-import-'));
 try{
  const file=path.join(root,'set.clevageset.json'),value={cleavage_pattern_set:{name:'imported_set',patterns:[]}};
  fs.writeFileSync(file,JSON.stringify(value));const context={extensionPath:path.dirname(filename)};
  assert.deepEqual(await compiled.exports.readCleavageImport(context,{path:file}),{value,path:file});
  assert.deepEqual(await compiled.exports.readCleavageImport(context,{json:JSON.stringify(value)}),{value,path:''});
  selected=[{fsPath:file}];assert.deepEqual(await compiled.exports.readCleavageImport(context,{}),{value,path:file});
  await assert.rejects(()=>compiled.exports.readCleavageImport(context,{json:'invalid JSON'}),SyntaxError);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
 process.stdout.write('Workbench cleavage navigation, addition, preparation apply, visible source/product drag trails, command serialization and import checks passed.\n');
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
