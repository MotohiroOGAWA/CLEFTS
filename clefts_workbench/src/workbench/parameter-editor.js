// A DOM-only configuration editor shared by data preparation and training.
function createParameterEditor(container, initial, kind = "training", editCleavagePatternSet) {
  const clone=value=>JSON.parse(JSON.stringify(value));
  const dataValue=value=>({fragmenter_params:clone(value.fragmenter_params||value),symbols:clone(value.symbols||value.mol_encoder_params?.symbols||initial.mol_encoder_params?.symbols||['C','N','O','P','S','F','Cl','Br','I']),max_node:value.max_node??-1,max_edge:value.max_edge??-1});
  let model=kind==='data'?dataValue(initial):clone(initial);
  const descriptions={
    architecture:'Spectrum generator architecture. Source-anchored action models operate on primitive cleavage actions.',
    max_action_count:'Maximum number of primitive actions in a normalized fragment action state.',
    precursor_candidate_max_action_count:'Maximum action count used to find candidates for the observed precursor.',
    mass_tolerance:'Peak assignment mass tolerance, e.g. 0.01Da,10ppm.',
    cleavage_pattern_set:'Patterns defining eligible primitive cleavage actions from the source molecule.',
    reactant_smarts:'SMARTS query that matches a cleavage site in the source molecule.',
    smarts:'Product SMARTS describing retained atoms and bond changes.',
    adduct_type:'Observed precursor adduct associated with this fragment ion rule.',
    radical:'Allow radical fragment ions for this adduct rule.',
    unsaturation:'Maximum hydrogen rearrangement / unsaturation allowed by this rule.',
    ion_shift:'Fragment ion composition shift, e.g. [M+H]+.',
    atoms:'Optional element restrictions. Remove this parameter to allow unrestricted elements.',
    symbols:'Elements included in prepared atom and formula features. Select elements using the periodic table.',
    max_node:'Maximum unique fragment nodes after merging identical fragments, including the source. -1 means unlimited.',
    max_edge:'Maximum total distinct cleavage transitions across all node pairs. Multiple routes to the same merged node count separately. -1 means unlimited.',
    hidden_dim:'Hidden representation dimension for this model component.',
    condition_dim:'Dimension of the encoded spectrum conditions.',
    node_dim:'Molecular encoder atom representation dimension.',graph_dim:'Molecular encoder graph representation dimension.',
    dropout:'Fraction of molecular encoder activations dropped during training.',
    max_roles:'Maximum SMARTS query role embeddings; leave the override unset to infer a safe size from the patterns.',
    num_layers:'Number of transformer layers.',num_heads:'Number of attention heads. Must divide the corresponding hidden dimension.',
    max_degree:'Maximum degree represented in graph structural embeddings.',max_spatial_dist:'Maximum atom graph distance encoded by the molecular encoder.',max_edge_dist:'Maximum bond distance encoded by the molecular encoder.',
    action_prefilter_top_k:'Number of primitive source actions retained by the absolute action scorer.',
    action_prefilter_max_k:'Maximum action pool size during teacher-forced training, including forced positive actions.',
    action_prefilter_threshold_logit:'Absolute scorer logit threshold used before top-K action selection.',
    prediction_threshold:'Independent sigmoid probability required to generate a branch (0 < value < 1).',
    beam_size:'Maximum candidate normalized action states retained at each decoding step.',
    state_num_layers:'Number of Set Transformer layers encoding the current action set.',state_dropout:'Dropout in the action state Set Transformer.',cosine_loss_weight:'Spectrum cosine loss weight, added to log intensity MSE.',
    max_decode_steps:'Maximum MS2 branch expansion depth from each precursor seed.',
    adduct_type_strs:'Supported precursor adduct types, in model embedding order.',
    name:'Human-readable name stored in this configuration.',
    mol_encoder_checkpoint:'Pretrained molecular encoder checkpoint path.',
    freeze_mol_encoder:'Freeze the molecular encoder during training.',
  };
  const templates={patterns:{name:'new_pattern',reactant_smarts:'[!#1:1]-[!#1:2]',products:[{name:'product',smarts:'[!#1:1]'}]},products:{name:'product',smarts:'[!#1:1]'},adduct_rules:{name:'new_rule',adduct_type:'[M+H]+',radical:false,unsaturation:0,ion_shifts:[]},ion_shifts:{ion_shift:'[M+H]+'},atoms:'C',symbols:'C',adduct_type_strs:'[M+H]+'};
  const optional={action_model_params:{num_heads:4,max_roles:64,state_num_layers:2,prediction_threshold:0.5},mol_encoder_params:{dropout:0},post_model_params:{cosine_loss_weight:0.5}};
  const basic=new Set(['fragmenter_params.fragment_ion_tree_builder.max_action_count','fragmenter_params.precursor_candidate_max_action_count','fragmenter_params.mass_tolerance']);
  const node=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const button=(text,fn)=>{const b=node('button',text);b.type='button';b.onclick=fn;return b;};
  const title=key=>String(key).replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase()).replace('Smarts','SMARTS');
  function help(key,path){return descriptions[key]||'Configure '+path.replaceAll('_',' ').replaceAll('.',' / ')+'. This value is passed to the CLEFTS Python workflow.';}
  function renderValue(parent,key,host,path,excludeBasic=false){
    const full=path?path+'.'+key:String(key),value=parent[key];if(excludeBasic&&basic.has(full))return;
    if(Array.isArray(value)||value&&typeof value==='object'){
      const group=node('fieldset'),legend=node('legend',title(key));legend.dataset.help=help(key,full);group.append(legend);
      if(Array.isArray(value)){
        value.forEach((item,index)=>{const row=node('div');row.className='parameter-array-item';renderValue(value,index,row,full);row.append(button('Remove',()=>{value.splice(index,1);render();}));group.append(row);});
        group.append(button('Add '+title(key),()=>{value.push(clone(templates[key]??value[0]??''));render();}));
      }else{
        Object.keys(value).forEach(child=>renderValue(value,child,group,full,excludeBasic));
        for(const [field,defaultValue]of Object.entries(optional[key]||{}))if(!(field in value))group.append(button('Override '+title(field),()=>{value[field]=defaultValue;render();}));
        if('ion_shift' in value&&!('atoms' in value))group.append(button('Add Atom Restrictions',()=>{value.atoms=[];render();}));
        if('ion_shift' in value&&'atoms' in value)group.append(button('Remove Atom Restrictions',()=>{delete value.atoms;render();}));
      }
      if(key==='cleavage_pattern_set'&&editCleavagePatternSet)group.append(button('Edit Cleavage Pattern Set Visually',()=>editCleavagePatternSet(clone(value),updated=>{parent[key]=clone(updated);render();})));
      host.append(group);return;
    }
    const fieldKey=typeof key==='number'?path.split('.').at(-1):key;
    const wrapper=node('label'),label=node('span',typeof key==='number'?title(fieldKey)+' '+(key+1):title(key));label.dataset.help=help(fieldKey,full);const input=node('input');input.dataset.parameterPath=full;input.type=typeof value==='boolean'?'checkbox':typeof value==='number'?'number':'text';
    if(input.type==='number')input.step='any';if(input.type==='checkbox'){input.checked=value;wrapper.className='check';}else input.value=value??'';
    input.oninput=()=>{if(input.type==='number'&&(!input.value.trim()||!Number.isFinite(Number(input.value)))){input.setCustomValidity('Enter a finite number.');return;}input.setCustomValidity('');parent[key]=input.type==='checkbox'?input.checked:input.type==='number'?Number(input.value):input.value;};
    wrapper.append(label,input);if(optional[path]?.[key]!==undefined)wrapper.append(button('Use Default',()=>{delete parent[key];render();}));host.append(wrapper);
  }
  function renderSymbols(section){
    const rows=['H . . . . . . . . . . . . . . . . He','Li Be . . . . . . . . . . B C N O F Ne','Na Mg . . . . . . . . . . Al Si P S Cl Ar','K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr','Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe','Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn','Fr Ra Ac Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og','. . Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu . .','. . Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr . .'];
    const list=node('p','Selected: '+model.symbols.join(', '));section.append(list);const grid=node('div');grid.className='periodic-table';grid.setAttribute('aria-label','Select element symbols');
    for(const row of rows)for(const symbol of row.split(' ')){if(symbol==='.'){grid.append(node('span'));continue;}const selected=model.symbols.includes(symbol);const tile=button(symbol,()=>{model.symbols=selected?model.symbols.filter(item=>item!==symbol):[...model.symbols,symbol];render();if(container.dispatchEvent)container.dispatchEvent(new Event('input',{bubbles:true}));});tile.dataset.symbol=symbol;tile.setAttribute('aria-pressed',String(selected));tile.title=symbol+(selected?' — enabled':' — disabled');grid.append(tile);}section.append(grid);
  }
  function render(){
    const expanded=new Set([...container.querySelectorAll('[data-advanced]')].filter(n=>!n.hidden).map(n=>n.dataset.advanced));container.replaceChildren();
    for(const [key,value]of Object.entries(model)){
      if(kind==='data'&&key==='symbols'){const section=node('section');section.append(node('h2','Symbols'));renderSymbols(section);container.append(section);continue;}
      if(kind==='training'&&['architecture','mol_encoder_checkpoint','fine_tuning','fragmenter_params','mol_encoder_params','adduct_type_strs','symbols','max_node','max_edge'].includes(key))continue;
      const section=node('section'),heading=node('h2',key==='fragmenter_params'?'Fragmenter Parameters':title(key));section.dataset.modelBlock=key;section.append(heading);
      if(kind==='data'&&['max_node','max_edge'].includes(key))continue;
      if(kind==='data'&&key==='fragmenter_params'){renderValue(model,'max_node',section,'');renderValue(model,'max_edge',section,'');}
      if(key==='fragmenter_params')for(const path of basic){const parts=path.split('.');let parent=model;for(const part of parts.slice(0,-1))parent=parent?.[part];if(parent&&parts.at(-1) in parent)renderValue(parent,parts.at(-1),section,parts.slice(0,-1).join('.'));}
      const advanced=node('div');advanced.dataset.advanced=key;advanced.hidden=!expanded.has(key);
      if(key==='fragmenter_params'||['mol_encoder_params','post_model_params'].includes(key)||(kind==='data'&&key!=='architecture')){const toggle=button('Advanced '+title(key),()=>{advanced.hidden=!advanced.hidden;toggle.setAttribute('aria-expanded',String(!advanced.hidden));});toggle.dataset.parameterToggle=key;toggle.setAttribute('aria-expanded',String(!advanced.hidden));section.append(toggle);renderValue(model,key,advanced,'',true);section.append(advanced);}
      else renderValue(model,key,section,'');container.append(section);
    }
  }
  render();return {getValue:()=>clone(model),setValue:value=>{model=kind==='data'?dataValue(value):clone(value);render();}};
}
function client(){
  const el=id=>document.getElementById(id),form=el('form'),training=el('trainingForm');
  const defaultModel=initial.modelConfig||{fragmenter_params:initial.fragmenterParams||initialTraining.modelConfig?.fragmenter_params,symbols:initial.symbols||initialTraining.modelConfig?.mol_encoder_params?.symbols};
  const dataInitial={...defaultModel,symbols:initial.symbols||defaultModel.symbols||defaultModel.mol_encoder_params?.symbols,max_node:initial.maxNode??defaultModel.max_node,max_edge:initial.maxEdge??defaultModel.max_edge};
  let applyPatterns,patternTarget;
  function editPatterns(target){return (value,apply)=>{applyPatterns=apply;patternTarget=target;cleavageModel={cleavage_pattern_set:value};cleavagePath='';renderCleavage();el('cleavagePath').textContent='Editing '+(target==='data'?'Data Preparation':'Training')+' Cleavage Patterns';el('applyParameterPatterns').hidden=false;document.querySelector('#navigationRail [data-page=cleavage]').click();};}
  const editors={data:createParameterEditor(el('dataParameterEditor'),dataInitial,'data',editPatterns('data')),training:createParameterEditor(el('trainingParameterEditor'),initialTraining.modelConfig||defaultModel,'training',editPatterns('training'))};
  function openDataPatterns(visually=false){
    const model=editors.data.getValue();
    const value=model.fragmenter_params?.fragment_ion_tree_builder?.cleavage_pattern_set||{name:'new_cleavage_pattern_set',patterns:[]};
    editPatterns('data')(JSON.parse(JSON.stringify(value)),updated=>{
      const current=editors.data.getValue();
      current.fragmenter_params.fragment_ion_tree_builder.cleavage_pattern_set=JSON.parse(JSON.stringify(updated));
      editors.data.setValue(current);form.dispatchEvent(new Event('input',{bubbles:true}));
    });
    if(visually)el('addVisualPattern').click();
  }
  el('dataEditCleavagePatterns').onclick=()=>openDataPatterns();
  el('dataAddVisualPattern').onclick=()=>openDataPatterns(true);
  el('applyParameterPatterns').onclick=()=>{if(applyPatterns){applyPatterns(cleavageModel.cleavage_pattern_set);el('applyParameterPatterns').hidden=true;document.querySelector('[data-page='+patternTarget+']').click();}};
  function snapshot(target,application){const config={application};for(const input of target.elements){if(input.name)config[input.name]=input.type==='number'?Number(input.value):input.type==='checkbox'?input.checked:input.value;}return config;}
  getConfig=()=>{const value=editors.data.getValue();return {...snapshot(form,'fragment-tree-data-preparation'),fragmenterParams:value.fragmenter_params,symbols:value.symbols,maxNode:value.max_node,maxEdge:value.max_edge};};
  getTrainingConfig=()=>({...snapshot(training,'fragment-tree-training'),modelConfig:editors.training.getValue()});
  function importFile(target){const input=el(target+'ParameterFile');if(input.value.trim())vscode.postMessage({type:'parameters/load',target,path:input.value});}
  for(const target of ['data','training']){const zone=el(target+'ParameterDrop');zone.onclick=()=>vscode.postMessage({type:'parameters/pick',target});zone.ondragover=event=>{event.preventDefault();zone.classList.add('path-drop-active');};zone.ondragleave=()=>zone.classList.remove('path-drop-active');zone.ondrop=async event=>{event.preventDefault();if(zone.disabled)return;zone.classList.remove('path-drop-active');const path=pathFromDataTransfer(event.dataTransfer);if(path){el(target+'ParameterFile').value=path;importFile(target);}else if(event.dataTransfer?.files?.[0]){const file=event.dataTransfer.files[0];try{if(file.size>2000000)throw new Error('Parameter file is too large.');vscode.postMessage({type:'parameters/import',target,name:file.name,json:await file.text()});}catch(error){el(target+'ParameterStatus').textContent=error.message;}}else el(target+'ParameterStatus').textContent='Drop a JSON file from the system file manager or VS Code Explorer.';};el(target+'ParameterBrowse').onclick=()=>vscode.postMessage({type:'parameters/pick',target});el(target+'ParameterFile').onchange=()=>importFile(target);el(target+'ParameterLoad').onclick=()=>importFile(target);}
  window.addEventListener('message',event=>{const m=event.data;if(m.type==='training/sources'&&!m.error&&m.requestId===training.dataset.sourcesRequest){const current=editors.training.getValue();for(const key of ['fragmenter_params','mol_encoder_params','adduct_type_strs'])current[key]=m.modelConfig[key];editors.training.setValue(current);}if(m.type==='training/checkpoint'&&!m.error&&[training.elements.resume.value,training.elements.fineTuneResume.value].includes(m.path)&&['resume','patterns'].includes(training.querySelector('input[name=trainingMode]:checked')?.value)){editors.training.setValue(m.modelConfig);training.elements.lr.value=m.lr;training.elements.weightDecay.value=m.weightDecay;for(const [key,value]of Object.entries(m.trainingSettings||{})){const input=training.elements[key];if(input){if(input.type==='checkbox')input.checked=value;else input.value=value;}}const fine=m.modelConfig.fine_tuning;if(fine){training.elements.fineTuneCheckpoint.value=fine.source_checkpoint||fine.base_checkpoint||fine.checkpoint_path||'';training.elements.adapterWidth.value=fine.adapter_width??fine.width??8;}}if(m.type==='parameters/loaded'&&editors[m.target]&&!(m.target==='training'&&(training.querySelector('input[name=trainingMode]:checked')?.value==='resume'||training.querySelector('input[name=trainingMode]:checked')?.value==='patterns'&&training.elements.fineTuneResume.value))){editors[m.target].setValue(m.modelConfig);el(m.target+'ParameterFile').value=m.path;el(m.target+'ParameterStatus').textContent='Parameters loaded. Form values will be used for execution.';}if(m.type==='parameters/error')el(m.target+'ParameterStatus').textContent=m.error;if(m.type==='config'&&(m.config.modelConfig||m.config.fragmenterParams))editors.data.setValue(m.config.modelConfig||{fragmenter_params:m.config.fragmenterParams,symbols:m.config.symbols,max_node:m.config.maxNode,max_edge:m.config.maxEdge});if(m.type==='trainingConfig'&&m.config.modelConfig)editors.training.setValue(m.config.modelConfig);});
  let timer;const tooltip=el('helpTooltip');document.addEventListener('mouseover',event=>{const label=event.target.closest('[data-help]');if(!label||!label.dataset.help)return;clearTimeout(timer);timer=setTimeout(()=>{const rect=label.getBoundingClientRect();tooltip.textContent=label.dataset.help;tooltip.style.left=Math.max(8,Math.min(rect.left,innerWidth-390))+'px';tooltip.style.top=Math.min(rect.bottom+7,innerHeight-110)+'px';tooltip.classList.add('visible');},500);});document.addEventListener('mouseout',event=>{if(event.target.closest('[data-help]')){clearTimeout(timer);tooltip.classList.remove('visible');}});
}
function importHtml(target){return `<section><h2>Import Parameters</h2><p class="muted">${target==='training'?'Drop an Action Model / Post Model parameter JSON to populate the trainable model settings. Dataset and encoder settings are loaded automatically.':'Optionally browse or drop a fragmenter / model JSON file to populate the fields below. Execution uses the current form values.'}</p><button type="button" id="${target}ParameterDrop" class="parameter-drop">Drop a parameter JSON file here or click to Browse</button><label><span data-help="Load a configuration into individual parameter fields. Later edits to the form override the imported file.">Parameter File</span><div class="path"><input id="${target}ParameterFile" data-path-kind="file" placeholder="Drop a JSON file here"><button type="button" id="${target}ParameterBrowse">Browse</button><button type="button" id="${target}ParameterLoad">Load Parameters</button></div></label><p id="${target}ParameterStatus" role="status"></p></section><div id="${target}ParameterEditor" class="parameter-editor"></div>`;}
module.exports={createParameterEditor,importHtml,script:()=>`${createParameterEditor.toString()}(${client.toString()})();`,css:()=>`.parameter-editor fieldset{border:1px solid var(--border);border-radius:5px;margin:12px 0;padding:14px;min-width:0}.parameter-editor legend{font-weight:600}.parameter-editor input[type=checkbox]{width:auto}.parameter-array-item{border-left:2px solid var(--border);margin:10px 0;padding:0 12px}.parameter-editor fieldset>label{display:inline-block;vertical-align:top;width:calc(50% - 16px);margin-right:16px}.parameter-editor .check{display:inline-flex}#helpTooltip{pointer-events:none}@media(max-width:700px){.parameter-editor fieldset>label{width:100%}}`};
