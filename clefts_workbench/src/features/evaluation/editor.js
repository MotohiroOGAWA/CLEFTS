const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');

function runCli(context, projectRoot, args, output) {
  return new Promise((resolve, reject) => {
    const root = projectRoot(context);
    const python = vscode.workspace.getConfiguration('clefts').get('pythonPath', 'python');
    const commandArgs = ['-m', 'evaluation', ...args];
    output.appendLine(`$ ${[python, ...commandArgs].join(' ')}`);
    const child = spawn(python, commandArgs, {
      cwd: root,
      env: { ...process.env, PYTHONPATH: ['.', process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
    });
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); output.append(chunk.toString()); });
    child.on('error', reject);
    child.on('close', code => {
      try {
        const response = JSON.parse(stdout.trim());
        if (code === 0 && response.ok) resolve(response.result);
        else reject(new Error(response.error || stderr || `Evaluation CLI exited with code ${code}.`));
      } catch (_) {
        reject(new Error(stderr || stdout || `Evaluation CLI exited with code ${code}.`));
      }
    });
  });
}

function summaryArgs(request, outputImage) {
  const args = ['summarize', '--input', request.input, '--group-column', request.groupColumn,
    '--mode', request.mode || 'auto',
    '--width', String(request.width || 900), '--height', String(request.height || 520),
    '--color', request.color || '#36c5a2'];
  if (request.metadata) args.push('--metadata', request.metadata);
  if (request.joinColumn) args.push('--join-column', request.joinColumn);
  if (request.mode === 'numeric' && request.bins) args.push('--bins', request.bins);
  if (Array.isArray(request.include)) args.push('--include', JSON.stringify(request.include));
  if (Array.isArray(request.order)) args.push('--order', JSON.stringify(request.order));
  if (!request.transparent) args.push('--opaque');
  if (outputImage) args.push('--output-image', outputImage);
  return args;
}

function open(context, output, projectRoot) {
  const panel = vscode.window.createWebviewPanel('clefts.evaluation', 'CLEFTS Evaluation', vscode.ViewColumn.One, {
    enableScripts: true, retainContextWhenHidden: true
  });
  panel.webview.html = html();
  panel.webview.onDidReceiveMessage(async message => {
    try {
      if (message.type === 'pickInput' || message.type === 'pickMetadata') {
        const metadata = message.type === 'pickMetadata';
        const picked = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: metadata
            ? { 'Spectrum metadata': ['msds', 'csv', 'tsv', 'json', 'jsonl', 'ndjson', 'parquet'] }
            : { 'Evaluation data': ['mssim', 'csv', 'tsv', 'json', 'jsonl', 'ndjson', 'parquet'] }
        });
        if (picked && picked[0]) panel.webview.postMessage({ type: metadata ? 'metadataPicked' : 'inputPicked', path: picked[0].fsPath });
      } else if (message.type === 'inspect') {
        panel.webview.postMessage({ type: 'busy', text: 'Inspecting columns…' });
        const args = ['inspect', '--input', message.input];
        if (message.metadata) args.push('--metadata', message.metadata);
        if (message.joinColumn) args.push('--join-column', message.joinColumn);
        const result = await runCli(context, projectRoot, args, output);
        panel.webview.postMessage({ type: 'inspected', result });
      } else if (message.type === 'summarize') {
        panel.webview.postMessage({ type: 'busy', text: 'Generating evaluation…' });
        const result = await runCli(context, projectRoot, summaryArgs(message.request), output);
        panel.webview.postMessage({ type: 'summary', result });
      } else if (message.type === 'saveImage') {
        const target = await vscode.window.showSaveDialog({
          filters: { 'Scalable Vector Graphics': ['svg'] },
          defaultUri: vscode.Uri.file('clefts-evaluation.svg')
        });
        if (target) {
          const result = await runCli(context, projectRoot, summaryArgs(message.request, target.fsPath), output);
          panel.webview.postMessage({ type: 'saved', path: result.imagePath || target.fsPath });
          vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
        }
      }
    } catch (error) {
      panel.webview.postMessage({ type: 'error', message: String(error.message || error) });
    }
  });
}

function html() {
  return `<!doctype html><html><head><meta charset="UTF-8"><style>
  :root{color-scheme:light dark;--accent:#36c5a2;--border:color-mix(in srgb,var(--vscode-editor-foreground) 18%,transparent);--panel:color-mix(in srgb,var(--vscode-editor-background) 90%,var(--vscode-editor-foreground))}
  *{box-sizing:border-box}body{margin:0;color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);font-family:var(--vscode-font-family)}main{max-width:1200px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;align-items:start;margin-bottom:20px}h1{margin:3px 0;font-size:28px}h2{font-size:16px;margin:0 0 14px}.eyebrow{font-size:11px;letter-spacing:.14em;font-weight:700;color:var(--accent)}.muted{opacity:.65;margin:4px 0}.layout{display:grid;grid-template-columns:340px minmax(0,1fr);gap:16px}.panel{padding:18px;border:1px solid var(--border);border-radius:9px;background:var(--panel);margin-bottom:14px}label{font-size:12px;display:block;margin:11px 0}input,select{width:100%;display:block;margin-top:5px;padding:8px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,var(--border));border-radius:5px}.path{display:flex;gap:6px}.path input{flex:1}.path button{margin-top:5px}button{font:inherit;color:inherit;background:var(--vscode-button-secondaryBackground);border:1px solid var(--border);border-radius:6px;padding:7px 11px;cursor:pointer}.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground);font-weight:600}.actions{display:flex;gap:7px;flex-wrap:wrap}.check{display:flex;align-items:center;gap:7px}.check input{width:auto;margin:0}.category-list{max-height:285px;overflow:auto;border:1px solid var(--border);border-radius:6px}.category{display:grid;grid-template-columns:auto minmax(0,1fr) auto auto;align-items:center;gap:7px;padding:6px;border-bottom:1px solid var(--border)}.category input{width:auto;margin:0}.category button{padding:2px 7px}.chart{overflow:auto;min-height:260px;border:1px dashed var(--border);border-radius:7px;display:grid;place-items:center}.chart svg{max-width:100%;height:auto}.status{font-weight:600}.error{color:var(--vscode-errorForeground)}table{width:100%;border-collapse:collapse;margin-top:14px;font-size:12px}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--border)}.size-row{display:grid;grid-template-columns:1fr 1fr;gap:9px}@media(max-width:800px){.layout{grid-template-columns:1fr}}
  </style></head><body><main><header><div><span class="eyebrow">CLEFTS EVALUATION</span><h1>Column Evaluation</h1><p class="muted">Group a result table by categories or numeric ranges.</p></div><button id="saveImage" disabled>Save SVG</button></header>
  <div class="layout"><aside>
    <section class="panel"><h2>Dataset</h2><label>Similarity result (.mssim)<div class="path"><input id="input" placeholder="results.mssim"><button id="browse">Browse</button></div></label><label>Metadata source (.msds or table)<div class="path"><input id="metadata" placeholder="metadata.msds"><button data-browse-metadata>Browse</button></div></label><label>Join column<input id="joinColumn" value="SpecID" placeholder="SpecID"></label><p class="muted">The join column must exist in both files. SpecID is used by default when available.</p><button id="load" class="primary">Load columns</button><p id="datasetInfo" class="muted"></p></section>
    <section class="panel"><h2>Grouping</h2><label>Column<select id="groupColumn" disabled></select></label><label>Mode<select id="mode"><option value="auto">Auto</option><option value="categorical">Category</option><option value="numeric">Numeric ranges</option></select></label><label id="binsLabel" hidden>Range boundaries<input id="bins" value="0,10,20" placeholder="0,10,20" disabled><small class="muted">0,10,20 creates [0,10), [10,20), [20,~).</small></label><p class="muted">Each group shows the distribution of cosine_similarity from the .mssim file.</p><button id="generate" class="primary" disabled>Generate box plot</button></section>
    <section class="panel"><h2>Categories</h2><div class="actions"><button id="all">All</button><button id="none">None</button></div><div id="categories" class="category-list"><p class="muted" style="padding:8px">Generate once to list categories.</p></div></section>
    <section class="panel"><h2>Image</h2><div class="size-row"><label>Width<input id="width" type="number" min="240" value="900"></label><label>Height<input id="height" type="number" min="200" value="520"></label></div><label>Box color<input id="color" type="color" value="#36c5a2"></label><label class="check"><input id="transparent" type="checkbox" checked>Transparent background</label></section>
  </aside><section class="panel"><div class="actions"><span id="status" class="status">Choose a dataset.</span></div><div id="chart" class="chart"><p class="muted">The box plot will appear here.</p></div><table id="table" hidden><thead><tr><th>Category / range</th><th>n</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th></tr></thead><tbody></tbody></table></section></div>
  <script>
  const vscode=acquireVsCodeApi(),input=document.getElementById('input'),group=document.getElementById('groupColumn'),mode=document.getElementById('mode'),binsLabel=document.getElementById('binsLabel'),bins=document.getElementById('bins'),categories=document.getElementById('categories'),status=document.getElementById('status'),chart=document.getElementById('chart'),table=document.getElementById('table');let categoryState=[];
  const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function setStatus(text,error=false){status.textContent=text;status.className='status'+(error?' error':'')}
  function request(){return{input:input.value,metadata:document.getElementById('metadata').value,joinColumn:document.getElementById('joinColumn').value,groupColumn:group.value,mode:mode.value,bins:mode.value==='numeric'?bins.value:'',width:Number(document.getElementById('width').value),height:Number(document.getElementById('height').value),color:document.getElementById('color').value,transparent:document.getElementById('transparent').checked,include:categoryState.filter(x=>x.checked).map(x=>x.name),order:categoryState.map(x=>x.name)}}
  function generate(preserve=true){if(!input.value||!group.value)return;vscode.postMessage({type:'summarize',request:{...request(),include:preserve&&categoryState.length?request().include:undefined,order:preserve&&categoryState.length?request().order:undefined}})}
  function renderCategories(rows){const old=new Map(categoryState.map(x=>[x.name,x.checked]));categoryState=rows.map(r=>({name:r.category,checked:old.has(r.category)?old.get(r.category):true}));categories.innerHTML=categoryState.map((x,i)=>'<div class="category" data-i="'+i+'"><input type="checkbox" '+(x.checked?'checked':'')+'><span>'+esc(x.name)+'</span><button data-move="-1" title="Move up">↑</button><button data-move="1" title="Move down">↓</button></div>').join('')||'<p class="muted" style="padding:8px">No categories.</p>'}
  document.getElementById('browse').onclick=()=>vscode.postMessage({type:'pickInput'});document.querySelector('[data-browse-metadata]').onclick=()=>vscode.postMessage({type:'pickMetadata'});document.getElementById('load').onclick=()=>{if(input.value){const r=request();vscode.postMessage({type:'inspect',input:r.input,metadata:r.metadata,joinColumn:r.joinColumn})}};input.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();document.getElementById('load').click()}};mode.onchange=()=>{const numeric=mode.value==='numeric';binsLabel.hidden=!numeric;bins.disabled=!numeric};document.getElementById('generate').onclick=()=>generate();group.onchange=()=>{categoryState=[];mode.value=group.selectedOptions[0].dataset.kind==='numeric'?'numeric':'categorical';mode.onchange();generate(false)};
  categories.onchange=e=>{const row=e.target.closest('[data-i]');if(row){categoryState[Number(row.dataset.i)].checked=e.target.checked;generate()}};categories.onclick=e=>{const button=e.target.closest('[data-move]');if(!button)return;const row=button.closest('[data-i]'),from=Number(row.dataset.i),to=Math.max(0,Math.min(categoryState.length-1,from+Number(button.dataset.move)));if(from===to)return;categoryState.splice(to,0,categoryState.splice(from,1)[0]);renderCategories(categoryState.map(x=>({category:x.name})));generate()};document.getElementById('all').onclick=()=>{categoryState.forEach(x=>x.checked=true);renderCategories(categoryState.map(x=>({category:x.name})));generate()};document.getElementById('none').onclick=()=>{categoryState.forEach(x=>x.checked=false);renderCategories(categoryState.map(x=>({category:x.name})));generate()};document.getElementById('saveImage').onclick=()=>vscode.postMessage({type:'saveImage',request:request()});
  window.addEventListener('message',e=>{const m=e.data;if(m.type==='inputPicked'){input.value=m.path}else if(m.type==='metadataPicked'){document.getElementById('metadata').value=m.path}else if(m.type==='busy')setStatus(m.text);else if(m.type==='inspected'){const cols=m.result.columns.filter(c=>!['index1','index2','cosine_similarity'].includes(c.name));group.innerHTML=cols.map(c=>'<option value="'+esc(c.name)+'" data-kind="'+c.kind+'">'+esc(c.name)+' ('+c.kind+')</option>').join('');group.disabled=document.getElementById('generate').disabled=!cols.length;document.getElementById('datasetInfo').textContent=m.result.rows+' pairs · '+cols.length+' grouping columns';if(cols.length){mode.value=group.selectedOptions[0].dataset.kind==='numeric'?'numeric':'categorical';mode.onchange();categoryState=[];generate(false)}}else if(m.type==='summary'){if(!categoryState.length)renderCategories(m.result.rows);chart.innerHTML=m.result.svg;table.hidden=false;const f=v=>v===null?'—':Number(v).toFixed(4);table.querySelector('tbody').innerHTML=m.result.rows.map(r=>'<tr><td>'+esc(r.category)+'</td><td>'+r.count+'</td><td>'+f(r.min)+'</td><td>'+f(r.q1)+'</td><td>'+f(r.median)+'</td><td>'+f(r.q3)+'</td><td>'+f(r.max)+'</td></tr>').join('');document.getElementById('saveImage').disabled=false;setStatus(m.result.rows.length+' groups · '+m.result.sourceRows+' similarity pairs')}else if(m.type==='saved')setStatus('Saved '+m.path);else if(m.type==='error')setStatus(m.message,true)});
  </script></main></body></html>`;
}

module.exports = { open, html, summaryArgs };
