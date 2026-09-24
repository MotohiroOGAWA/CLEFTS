const assert=require('assert/strict');
const fs=require('fs');
const path=require('path');
const os=require('os');
const Module=require('module');
const load=Module._load;
const vscodeMock={window:{},Uri:{file:fsPath=>({fsPath})}};
Module._load=function(name,parent,main){if(name==='vscode')return vscodeMock;return load.call(this,name,parent,main);};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.hidden=false;}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;}
 setAttribute(key,value){this[key]=value;}
 setCustomValidity(message){this.validationMessage=message;}
 querySelectorAll(){return walk(this).filter(node=>'advanced' in node.dataset);}
}
function walk(node){return [node,...(node.children||[]).filter(child=>child&&typeof child==='object').flatMap(walk)];}
global.document={createElement:tag=>new Element(tag)};
const {createParameterEditor,importHtml}=require('../src/workbench/parameter-editor');
const service=require('../src/workbench/parameter-service');
const {buildArgs,buildTrainingArgs}=require('../src/extension');
const app=path.resolve(__dirname,'../..'),preset=service.defaults(app);
(async()=>{
 const root=new Element('div'),editor=createParameterEditor(root,preset,'data');
 assert(!walk(root).some(node=>node.dataset.parameterPath?.includes('hidden_dim')));assert(!walk(root).some(node=>node.dataset.parameterPath?.includes('adduct_type_strs')));assert.equal(walk(root).filter(node=>node.dataset.symbol).length,118);
 const hydrogen=walk(root).find(node=>node.dataset.symbol==='H');hydrogen.onclick();assert(editor.getValue().symbols.includes('H'));
 const maxNode=walk(root).find(node=>node.dataset.parameterPath==='max_node');maxNode.value='500';maxNode.oninput();assert.equal(editor.getValue().max_node,500);
 const maxEdge=walk(root).find(node=>node.dataset.parameterPath==='max_edge');maxEdge.value='1000';maxEdge.oninput();assert.equal(editor.getValue().max_edge,1000);
 const find=predicate=>walk(root).find(predicate);
 const mass=find(node=>node.dataset.parameterPath==='fragmenter_params.mass_tolerance');mass.value='0.03Da';mass.oninput();assert.equal(editor.getValue().fragmenter_params.mass_tolerance,'0.03Da');assert.equal(preset.fragmenter_params.mass_tolerance,'0.01Da,10ppm');
 const basic=find(node=>node.dataset.parameterPath==='fragmenter_params.fragment_ion_tree_builder.max_action_count');basic.value='';basic.oninput();assert(basic.validationMessage);assert.equal(editor.getValue().fragmenter_params.fragment_ion_tree_builder.max_action_count,3);
 assert(!find(node=>node.dataset.advanced==='fragmenter_params'));assert(!find(node=>node.tag==='button'&&node.textContent==='Advanced Fragmenter Params'));
 const fragmenterHtml=importHtml('data');assert(fragmenterHtml.includes('dataParameterLoad'));assert(fragmenterHtml.includes('dataParameterSave'));assert(!fragmenterHtml.includes('Import Parameters'));assert(!fragmenterHtml.includes('model JSON'));assert(!fragmenterHtml.includes('dataParameterFile'));
 find(node=>node.tag==='button'&&node.textContent==='Add Adduct Rules').onclick();assert.equal(editor.getValue().fragmenter_params.fragment_ion_tree_builder.fragment_ion_adduct_rule_set.adduct_rules.length,5);
 const current=editor.getValue(),args=buildArgs({input:'input.msds',outputDir:'output',modelConfig:current,params:'ignored.json'});assert(!args.includes('--params'));assert.deepEqual(JSON.parse(args[args.indexOf('--params-json')+1]),current);
 const dataArgs=buildArgs({input:'input.msds',outputDir:'out',fragmenterParams:current.fragmenter_params,symbols:current.symbols,maxNode:500,maxEdge:1000});assert(dataArgs.includes('--symbols-json'));assert.equal(dataArgs[dataArgs.indexOf('--max-edge')+1],'1000');
 const trainingArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:current});assert(!trainingArgs.includes('--params'));assert(!trainingArgs.includes('--params-json'));
 // A resumed config always carries action_model_params.state_dropout and
 // post_model_params.dropout (training_model_config writes both from the one
 // shared --dropout), but only that one shared field may ever be edited.
 const dropoutConfigRoot=new Element('div');
 createParameterEditor(dropoutConfigRoot,{action_model_params:{state_dropout:0.3},post_model_params:{dropout:0.3}},'training');
 assert(!walk(dropoutConfigRoot).some(node=>node.dataset.parameterPath==='action_model_params.state_dropout'));
 assert(!walk(dropoutConfigRoot).some(node=>node.dataset.parameterPath==='post_model_params.dropout'));
 // The balanced ion-score loss weight is a per-component model_config field, like
 // cosine_loss_weight, and (like every other optional field) is filled in with
 // its default immediately -- no "Override" button to click first.
 const ionRoot=new Element('div'),ionEditor=createParameterEditor(ionRoot,{post_model_params:{}},'training');
 assert(!walk(ionRoot).some(node=>node.tag==='button'&&node.textContent.startsWith('Override ')));
 assert.equal(walk(ionRoot).find(node=>node.dataset.parameterPath==='post_model_params.ion_loss_weight').value,0.5);
 const aggregation=walk(ionRoot).find(node=>node.dataset.parameterPath==='post_model_params.equivalent_state_aggregation');
 assert.equal(aggregation.tag,'select');assert.deepEqual(aggregation.children.map(option=>option.value),['attention']);assert.equal(aggregation.value,'attention');
 assert(walk(ionRoot).some(node=>node.className==='parameter-group-heading'&&node.textContent==='Loss weights'));
 const ionArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:ionEditor.getValue()});
 assert.equal(ionArgs[ionArgs.indexOf('--post-ion-loss-weight')+1],'0.5');
 // action_neighborhood_mode is a closed-choice Primitive Action Encoder setting,
 // defaulted like every other optional action_model_params field.
 const neighborhoodRoot=new Element('div'),neighborhoodEditor=createParameterEditor(neighborhoodRoot,{action_model_params:{}},'training');
 const neighborhoodMode=walk(neighborhoodRoot).find(node=>node.dataset.parameterPath==='action_model_params.action_neighborhood_mode');
 assert.equal(neighborhoodMode.tag,'select');assert.deepEqual(neighborhoodMode.children.map(option=>option.value),['hop_pooling','none']);assert.equal(neighborhoodMode.value,'hop_pooling');
 // Neighborhood Max Hop defaults to 3 and starts enabled alongside hop_pooling.
 const maxHop=walk(neighborhoodRoot).find(node=>node.dataset.parameterPath==='action_model_params.action_neighborhood_max_hop');
 assert.equal(maxHop.value,3);assert(!maxHop.disabled);
 maxHop.value='2';maxHop.oninput();assert.equal(neighborhoodEditor.getValue().action_model_params.action_neighborhood_max_hop,2);
 neighborhoodMode.value='none';neighborhoodMode.onchange();assert.equal(neighborhoodEditor.getValue().action_model_params.action_neighborhood_mode,'none');
 // Switching to Mode=none re-renders and disables the now-meaningless Max Hop field.
 assert(walk(neighborhoodRoot).find(node=>node.dataset.parameterPath==='action_model_params.action_neighborhood_max_hop').disabled);
 const neighborhoodArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:neighborhoodEditor.getValue()});
 assert.equal(neighborhoodArgs[neighborhoodArgs.indexOf('--action-neighborhood-mode')+1],'none');
 assert.equal(neighborhoodArgs[neighborhoodArgs.indexOf('--action-neighborhood-max-hop')+1],'2');
 // max_roles is sized from the prepared dataset, never a form field, even
 // when a resumed/loaded config already carries one.
 const maxRolesRoot=new Element('div');
 createParameterEditor(maxRolesRoot,{action_model_params:{max_roles:96}},'training');
 assert(!walk(maxRolesRoot).some(node=>node.dataset.parameterPath==='action_model_params.max_roles'));
 // Dropout is one shared training-level flag, not a per-component model_config override.
 const dropoutArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',dropout:0.3});
 assert.equal(dropoutArgs[dropoutArgs.indexOf('--dropout')+1],'0.3');
 const legacyConfigArgs=buildTrainingArgs({trainDir:'train',valDir:'val',outputDir:'out',modelConfig:{action_model_params:{state_dropout:0.1},post_model_params:{dropout:0.1}},molEncoderCheckpoint:'encoder.pt'});
 assert(!legacyConfigArgs.includes('--action-state-dropout'));assert(!legacyConfigArgs.includes('--post-dropout'));
 const temp=fs.mkdtempSync(path.join(os.tmpdir(),'clefts-parameter-import-'));try{
 const file=path.join(temp,'fragmenter config.json');fs.writeFileSync(file,JSON.stringify({fragment_ion_tree_builder:{max_action_count:2},mass_tolerance:'0.04Da'}));let handler;const messages=[];service.attach({webview:{onDidReceiveMessage:fn=>handler=fn,postMessage:data=>messages.push(data)}},{},()=>app);
 await handler({type:'parameters/load',target:'data',path:file});assert.equal(messages.at(-1).type,'parameters/loaded');assert.equal(messages.at(-1).modelConfig.fragmenter_params.mass_tolerance,'0.04Da');assert.equal(messages.at(-1).modelConfig.fragmenter_params.fragment_ion_tree_builder.max_action_count,2);assert(messages.at(-1).modelConfig.fragmenter_params.fragment_ion_tree_builder.cleavage_pattern_set);
 await handler({type:'parameters/import',target:'data',name:'dropped.json',json:JSON.stringify({cleavage_pattern_set:{name:'imported',patterns:[]},fragment_ion_adduct_rule_set:{name:'imported rules',adduct_rules:[]}})});assert.equal(messages.at(-1).modelConfig.fragmenter_params.fragment_ion_tree_builder.cleavage_pattern_set.name,'imported');assert.equal(messages.at(-1).modelConfig.fragmenter_params.fragment_ion_tree_builder.fragment_ion_adduct_rule_set.name,'imported rules');
 await handler({type:'parameters/import',target:'data',name:'model.json',json:JSON.stringify({fragmenter_params:{fragment_ion_tree_builder:{}},mol_encoder_params:{hidden_dim:128}})});assert.equal(messages.at(-1).type,'parameters/error');assert.match(messages.at(-1).error,/not a model config/);
 const saved=path.join(temp,'saved.fragmenter.json');vscodeMock.window.showSaveDialog=async()=>({fsPath:saved});await handler({type:'parameters/save',target:'data',fragmenterParams:{fragment_ion_tree_builder:{max_action_count:4},mass_tolerance:'0.02Da'}});assert.equal(messages.at(-1).type,'parameters/saved');assert.deepEqual(JSON.parse(fs.readFileSync(saved,'utf8')),{fragment_ion_tree_builder:{max_action_count:4},mass_tolerance:'0.02Da'});
 await handler({type:'parameters/load',target:'training',path:file+'missing'});assert.equal(messages.at(-1).type,'parameters/error');
 }finally{fs.rmSync(temp,{recursive:true,force:true});}
 console.log('Parameter form editing, array updates, advanced settings, execution snapshots and file expansion checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
