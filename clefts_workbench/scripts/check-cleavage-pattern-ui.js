const assert=require('assert/strict'),fs=require('fs'),path=require('path'),os=require('os'),Module=require('module');
const {JSDOM,VirtualConsole}=require('jsdom');
const project=path.resolve(__dirname,'../..');let selected;
const mock={workspace:{workspaceFolders:[{uri:{fsPath:project}}],getConfiguration:()=>({get:(key,fallback)=>fallback})},window:{showOpenDialog:async()=>selected}};
const load=Module._load;Module._load=function(name,parent,main){return name==='vscode'?mock:load.call(this,name,parent,main);};
const periodicRows=require('../src/features/cleavage-pattern-set/periodic-table').rows;assert(periodicRows.every(row=>row.length===18));assert.equal(periodicRows[1][12],'B');assert.equal(periodicRows[1][17],'Ne');assert.equal(periodicRows[2][12],'Al');assert.equal(periodicRows[2][17],'Ar');
const extension=require('../src/extension'),preset=require('../../clefts/presets/spectrum_generator_params/source_anchored_pos_model_config.json');
const messages=[],errors=[],console=new VirtualConsole();console.on('jsdomError',error=>errors.push(error.detail?.stack||error.message));
const dom=new JSDOM(extension.workbenchHtml({modelConfig:preset},{modelConfig:preset}),{runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:console,beforeParse(w){w.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>({}),setState(){}});w.HTMLElement.prototype.scrollIntoView=function(){};w.ResizeObserver=class{observe(){}disconnect(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});w.CSS={escape:s=>s};}});
(async()=>{
 const w=dom.window,d=w.document,post=data=>w.dispatchEvent(new w.MessageEvent('message',{data}));
 try{
  assert.deepEqual(errors,[]);
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
  d.getElementById('dataAddVisualPattern').click();assert(!d.getElementById('visualBuilder').hidden);assert.equal(d.activeElement.id,'builderSmiles');assert.equal(d.getElementById('builderSourceType').value,'smiles');
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
  // Both canvases share toggle/deselect, shape selection, pan and zoom handling.
  function pointer(canvas,kind,extra={}){
   const svg=canvas.querySelector('svg');
   svg.createSVGPoint=()=>({x:0,y:0,matrixTransform(){return {x:this.x,y:this.y};}});
   svg.getScreenCTM=()=>({inverse:()=>({})});svg.setPointerCapture=()=>{};
   canvas['onpointer'+kind]({button:0,clientX:10,clientY:20,pointerId:1,target:svg,preventDefault(){},...extra});
  }
  for(const id of ['moleculeCanvas','productCanvas']){
   const canvas=d.getElementById(id),step=canvas.closest('.ce-step');
   function clickAtom(shift=false){pointer(canvas,'down',{target:canvas.querySelector('[data-mol-atom="0"]'),shiftKey:shift});pointer(canvas,'up');}
   clickAtom();assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   clickAtom();assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   clickAtom();clickAtom(true);assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   const box=step.querySelector('[data-shape="box"]');box.click();assert.equal(box.getAttribute('aria-pressed'),'true');
   pointer(canvas,'down');pointer(canvas,'move',{clientX:100,clientY:100});assert.equal(canvas.querySelector('.selection-lasso').getAttribute('points'),'10,20 100,20 100,100 10,100 10,20');canvas.onpointercancel();
   pointer(canvas,'down',{clientX:0,clientY:0});pointer(canvas,'move',{clientX:700,clientY:440});pointer(canvas,'up');assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   pointer(canvas,'down',{clientX:0,clientY:0,shiftKey:true});pointer(canvas,'move',{clientX:700,clientY:440});pointer(canvas,'up');assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   const before=canvas.querySelector('svg').getAttribute('viewBox');
   pointer(canvas,'down',{button:2,target:canvas.querySelector('[data-mol-atom="0"]')});pointer(canvas,'move',{clientX:50,clientY:60});pointer(canvas,'up');
   assert.notEqual(canvas.querySelector('svg').getAttribute('viewBox'),before);assert(!canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   step.querySelector('[data-zoom="fit"]').click();step.querySelector('[data-shape="lasso"]').click();
   clickAtom();step.querySelector('[data-mode="edit"]').click();
   pointer(canvas,'down',{target:canvas.querySelector('[data-mol-atom="0"]')});pointer(canvas,'up');
   assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));
   pointer(canvas,'down',{target:canvas.querySelector('[data-mol-atom="1"]')});pointer(canvas,'up');
   assert(canvas.querySelector('[data-mol-atom="0"]').classList.contains('selected'));assert(!canvas.querySelector('[data-mol-atom="1"]').classList.contains('selected'));
   assert.equal(canvas.nextElementSibling.textContent,'1 atoms · 0 bonds');
   step.querySelector('[data-mode="selection"]').click();clickAtom();assert(step.querySelector('.ce-help').textContent.includes('Shift'));

  }
  const sourceCanvas=d.getElementById('moleculeCanvas');pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-atom="0"]')});pointer(sourceCanvas,'up');
  const transforms=[...sourceCanvas.querySelectorAll('.mol-atom')].map(a=>a.getAttribute('transform').match(/[-\d.]+/g).map(Number));assert.equal(transforms[1][0]-transforms[0][0],65);assert.equal(transforms[1][1]-transforms[0][1],65);
  const periodic=d.querySelector('[data-periodic]');assert(periodic);const dialog=d.querySelector('.ce-dialog');dialog.showModal=()=>dialog.setAttribute('open','');dialog.close=()=>dialog.removeAttribute('open');periodic.click();assert(dialog.hasAttribute('open'));
  const element=symbol=>[...dialog.querySelectorAll('.ce-periodic button')].find(b=>b.textContent===symbol);
  assert.equal(element('C').style.gridColumn,'14');assert.equal(element('C').style.gridRow,'2');assert.equal(element('Ne').style.gridColumn,'18');assert.equal(element('Na').style.gridRow,'3');assert.equal(element('Na').style.gridColumn,'1');assert.equal(element('C').getAttribute('aria-pressed'),'true');assert.equal(w.getComputedStyle(element('C')).backgroundColor,'rgb(8, 125, 221)');
  element('N').click();assert.equal(element('N').getAttribute('aria-pressed'),'true');assert.equal(w.getComputedStyle(element('N')).backgroundColor,'rgb(8, 125, 221)');
  dialog.querySelector('.actions button').click();d.querySelector('[data-periodic]').click();assert.equal(element('N').getAttribute('aria-pressed'),'true');element('N').click();assert.equal(element('N').getAttribute('aria-pressed'),'false');dialog.close();
  const any=d.querySelector('#atomConstraints [data-am="any"]');any.checked=true;any.dispatchEvent(new w.Event('change',{bubbles:true}));assert(d.querySelector('#atomConstraints [data-non-h]'));assert(d.querySelector('#atomConstraints [data-non-h]').checked);
  const nonH=d.querySelector('#atomConstraints [data-non-h]');nonH.checked=false;nonH.dispatchEvent(new w.Event('change',{bubbles:true}));assert.equal(w.eval('atomConstraintState[0].nonHydrogen'),false);assert.equal(w.eval('atomConstraintState[0].mode'),'any');

  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="0"]')});pointer(sourceCanvas,'up');
  d.querySelector('#atomConstraints').closest('.ce-step').querySelector('[data-mode="edit"]').click();
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="0"]')});pointer(sourceCanvas,'up');
  d.querySelector('#bondConstraints [data-rs="inRing"]').click();
  assert.equal(w.eval('bondConstraintState[0].ringStatus'),'inRing');assert(d.querySelector('#bondConstraints [data-rs="inRing"]').checked);
  w.eval("chemistry=async(command,payload)=>command==='reactant'?{smarts:'[#6:1]-[#8:2]',atomMapBySource:{0:1,1:2}}:command==='molecule'?"+JSON.stringify(graph)+":command==='productFromStructure'?{smarts:'[#6:1].[#8:2]'}:{valid:true}");
  const patternCount=d.querySelectorAll('#cleavagePatterns .pattern-card').length;await d.getElementById('generateReactant').onclick();assert(!d.querySelector('.ce-step .ce-body').hidden);assert.equal(d.querySelectorAll('#cleavagePatterns .pattern-card').length,patternCount);assert.equal(d.querySelector('[aria-label="Reactant SMARTS"]').value,'[#6:1]-[#8:2]');
  await d.getElementById('applyReactant').onclick();
  const steps=d.querySelectorAll('.ce-step');assert(steps[0].querySelector('.ce-body').hidden);assert(!steps[1].querySelector('.ce-body').hidden);assert.equal(steps[0].querySelector('.ce-status').textContent,'Completed');
  pointer(sourceCanvas,'down',{button:2});pointer(sourceCanvas,'move',{clientX:50,clientY:60});pointer(sourceCanvas,'up');assert.equal(steps[0].querySelector('.ce-status').textContent,'Completed');
  steps[0].querySelector('.ce-header button').click();assert(!steps[0].querySelector('.ce-body').hidden);assert(steps[1].querySelector('.ce-body').hidden);
  steps[1].querySelector('.ce-header button').click();assert(steps[0].querySelector('.ce-body').hidden);assert(!steps[1].querySelector('.ce-body').hidden);
  steps[0].querySelector('.ce-header [aria-label="Copy Reactant SMARTS"]').click();assert.equal(messages.at(-1).type,'copyCleavageText');assert.equal(messages.at(-1).text,'[#6:1]-[#8:2]');
  const pc=d.getElementById('productCanvas');steps[1].querySelector('[data-tool="cut"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');assert(pc.querySelector('[data-product-bond="base:0"]').classList.contains('deleted'));
  w.eval('productBondOverrides={}');steps[1].querySelector('[data-tool="deleteAtom"]').click();pointer(pc,'down',{target:pc.querySelector('[data-mol-atom="1"]')});pointer(pc,'up');assert(pc.querySelector('[data-mol-atom="1"]').classList.contains('deleted'));
  steps[1].querySelector('[data-undo-product]').click();assert(!pc.querySelector('[data-mol-atom="1"]').classList.contains('deleted'));
  steps[1].querySelector('[data-tool="deleteAtom"]').click(); // toggles the tool off
  steps[1].querySelector('[data-mode="selection"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');
  steps[1].querySelector('[data-mode="edit"]').click();pointer(pc,'down',{target:pc.querySelector('[data-product-bond="base:0"]')});pointer(pc,'up');
  const double=steps[1].querySelector('[data-product-order="2"]');assert(double);double.checked=true;double.dispatchEvent(new w.Event('change',{bubbles:true}));assert(pc.querySelector('[data-product-bond="base:0"]').classList.contains('bond-double'));assert.equal(w.eval('productBondOverrides[0]'),'2');
  assert(!steps[1].querySelector('.ce-reference'));assert(!steps[1].querySelector('[data-tool="change"]'));assert(!steps[1].querySelector('[aria-label="New bond type"]'));
  await d.getElementById('newVisualProduct').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,1);
  await d.getElementById('newVisualProduct').onclick();assert.equal(d.querySelectorAll('[data-visual-product]').length,2);assert(d.querySelector('[data-visual-product="1"]').classList.contains('active'));
  await d.querySelector('[data-visual-product="0"]').onclick();assert(d.querySelector('[data-visual-product="0"]').classList.contains('active'));assert.equal(w.eval('cleavageModel.cleavage_pattern_set.patterns[visualPatternIndex].products.length'),2);
  // Regression with the user's exact SMILES and the real RDKit messaging backend.
  const {execFileSync}=require('child_process');
  w.testChemistry=(command,payload)=>JSON.parse(execFileSync('python',[path.join(project,'clefts_workbench/src/features/cleavage-pattern-set/backend.py')],{cwd:project,input:JSON.stringify({command,payload}),encoding:'utf8'})).result;
  w.eval('chemistry=async(command,payload)=>window.testChemistry(command,payload)');
  d.getElementById('addVisualPattern').click();assert.equal(d.getElementById('builderSourceType').value,'smiles');
  d.getElementById('builderSmiles').value='CC=CC1CCCCC1';await d.getElementById('drawMolecule').onclick();
  const actual=w.eval('molGraph'),ringBond=actual.bonds.find(b=>b.inRing),chainBond=actual.bonds.find(b=>!b.inRing);
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+chainBond.index+'"]')});pointer(sourceCanvas,'up');assert(d.querySelector('#bondConstraints [data-rs="notInRing"]').checked);
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+chainBond.index+'"]'),shiftKey:true});pointer(sourceCanvas,'up');
  pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+ringBond.index+'"]')});pointer(sourceCanvas,'up');assert(d.querySelector('#bondConstraints [data-rs="inRing"]').checked);
  for(const id of [ringBond.begin,ringBond.end]){pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-atom="'+id+'"]')});pointer(sourceCanvas,'up');}
  steps[0].querySelector('[data-mode="edit"]').click();pointer(sourceCanvas,'down',{target:sourceCanvas.querySelector('[data-mol-bond="'+ringBond.index+'"]')});pointer(sourceCanvas,'up');
  d.querySelector('#bondConstraints [data-rs="notInRing"]').click();assert(d.querySelector('#bondConstraints [data-rs="notInRing"]').checked);
  d.querySelector('#bondConstraints [data-rs="inRing"]').click();assert(d.querySelector('#bondConstraints [data-rs="inRing"]').checked);
  await d.getElementById('generateReactant').onclick();assert(!steps[0].querySelector('.ce-body').hidden);assert(steps[1].querySelector('.ce-body').hidden);assert(d.querySelector('[aria-label="Reactant SMARTS"]').value.includes('@'));assert.equal(d.querySelectorAll('#cleavagePatterns .pattern-card').length,patternCount+1);
  await d.getElementById('loadCleavage').ondrop({preventDefault(){},stopPropagation(){},dataTransfer:{getData:type=>type==='text/uri-list'?'file:///tmp/imported%20set.clevageset.json':'',files:[]}});
  assert.equal(messages.at(-1).type,'loadCleavagePatternSet');assert.equal(messages.at(-1).path,'/tmp/imported set.clevageset.json');
  const pattern={name:'imported',reactant_smarts:'[#6:1]-[#8:2]',products:[{name:'carbon',smarts:'[#6:1]'}]};
  await d.getElementById('loadSinglePattern').ondrop({preventDefault(){},stopPropagation(){},dataTransfer:{getData:()=>'',files:[{text:async()=>JSON.stringify(pattern)}]}});
  assert.equal(messages.at(-1).type,'loadCleavagePattern');assert.deepEqual(JSON.parse(messages.at(-1).json),pattern);
  post({type:'cleavagePatternLoaded',pattern,index:-1});assert.equal(d.querySelectorAll('#cleavagePatterns .pattern-card').length,original+3);
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
 process.stdout.write('Workbench cleavage navigation, addition, preparation apply, visible source/product drag trails, shared selection, Shift box removal, pan isolation, accordion transitions, Copy, periodic table, product operations and import checks passed.\n');
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
