const vscode = require('vscode');

const VIEW_TYPE = 'clefts.cleavagePatternSetEditor';
const FILE_SUFFIX = '.clevageset.json';

function emptyDocument() {
  return {
    cleavage_pattern_set: {
      name: 'new_cleavage_pattern_set',
      patterns: []
    }
  };
}

function normalizeDocument(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('The JSON root must be an object.');
  const set = value.cleavage_pattern_set;
  if (!set || typeof set !== 'object' || Array.isArray(set)) throw new Error('The JSON must contain a cleavage_pattern_set object.');
  return {
    cleavage_pattern_set: {
      name: String(set.name || ''),
      patterns: Array.isArray(set.patterns) ? set.patterns.map(pattern => ({
        name: String(pattern && pattern.name || ''),
        reactant_smarts: String(pattern && pattern.reactant_smarts || ''),
        products: Array.isArray(pattern && pattern.products) ? pattern.products.map(product => ({
          name: String(product && product.name || ''),
          smarts: String(product && product.smarts || '')
        })) : []
      })) : []
    }
  };
}

class CleavagePatternSetEditorProvider {
  resolveCustomTextEditor(document, panel) {
    panel.webview.options = { enableScripts: true };
    panel.webview.html = editorHtml();
    let applyingEdit = false;

    const sendDocument = () => {
      try {
        const value = normalizeDocument(JSON.parse(document.getText()));
        panel.webview.postMessage({ type: 'document', value });
      } catch (error) {
        panel.webview.postMessage({ type: 'error', message: error.message });
      }
    };

    const changeSubscription = vscode.workspace.onDidChangeTextDocument(event => {
      if (event.document.uri.toString() === document.uri.toString() && !applyingEdit) sendDocument();
    });
    panel.onDidDispose(() => changeSubscription.dispose());
    panel.webview.onDidReceiveMessage(async message => {
      if (message.type === 'ready') return sendDocument();
      if (message.type !== 'update') return;
      try {
        const value = normalizeDocument(message.value);
        const text = `${JSON.stringify(value, null, 2)}\n`;
        const fullRange = new vscode.Range(document.positionAt(0), document.positionAt(document.getText().length));
        const edit = new vscode.WorkspaceEdit();
        edit.replace(document.uri, fullRange, text);
        applyingEdit = true;
        await vscode.workspace.applyEdit(edit);
        applyingEdit = false;
        panel.webview.postMessage({ type: 'savedState', valid: true });
      } catch (error) {
        applyingEdit = false;
        panel.webview.postMessage({ type: 'error', message: error.message });
      }
    });
  }
}

function register(context) {
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, new CleavagePatternSetEditorProvider(), {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: false
    })
  );
}

function editorHtml() {
  return `<!doctype html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
  :root{color-scheme:light dark;--border:color-mix(in srgb,var(--vscode-editor-foreground) 18%,transparent);--panel:color-mix(in srgb,var(--vscode-editor-background) 90%,var(--vscode-editor-foreground));--accent:#36c5a2}*{box-sizing:border-box}body{margin:0;background:var(--vscode-editor-background);color:var(--vscode-editor-foreground);font-family:var(--vscode-font-family)}main{max-width:1050px;margin:auto;padding:32px}header,.row,.section-title,.card-title{display:flex;align-items:center;justify-content:space-between;gap:14px}header{margin-bottom:28px}.eyebrow{font-size:11px;letter-spacing:.14em;color:var(--accent);font-weight:700}h1{margin:4px 0;font-size:30px}h2{font-size:18px;margin:0}.muted{opacity:.65}.panel,.pattern{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;margin:14px 0}.pattern{border-left:3px solid var(--accent)}label{display:block;font-size:12px;opacity:.8;margin:10px 0}input{display:block;width:100%;margin-top:6px;padding:9px;border-radius:5px;border:1px solid var(--vscode-input-border,var(--border));background:var(--vscode-input-background);color:var(--vscode-input-foreground);font:inherit}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.products{margin:16px 0 0 18px}.product{display:grid;grid-template-columns:1fr 1fr auto;gap:10px;align-items:end;border-top:1px solid var(--border);padding:8px 0}button{font:inherit;color:inherit;background:var(--vscode-button-secondaryBackground);border:1px solid var(--border);border-radius:6px;padding:8px 12px;cursor:pointer}button:hover{background:var(--vscode-button-secondaryHoverBackground)}button.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground)}button.danger{color:var(--vscode-errorForeground)}#message{min-height:18px;font-size:12px}.error{color:var(--vscode-errorForeground)}@media(max-width:650px){main{padding:18px}.grid,.product{grid-template-columns:1fr}}
  </style></head><body><main><header><div><span class="eyebrow">CLEFTS EDITOR</span><h1>Cleavage Pattern Set</h1><div id="message" class="muted">Changes are stored in the JSON document. Use Ctrl/Cmd+S to save.</div></div><button id="addPattern" class="primary">Add Pattern</button></header><section class="panel"><label>Pattern set name<input id="setName" placeholder="single_bond_cleavage_pattern_set"></label></section><div id="patterns"></div><script>
  const vscode=acquireVsCodeApi();let model={cleavage_pattern_set:{name:'',patterns:[]}},timer;
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function render(){document.getElementById('setName').value=model.cleavage_pattern_set.name;document.getElementById('patterns').innerHTML=model.cleavage_pattern_set.patterns.map((p,pi)=>\`<section class="pattern"><div class="card-title"><h2>Pattern \${pi+1}</h2><button class="danger" data-remove-pattern="\${pi}">Remove Pattern</button></div><div class="grid"><label>Pattern name<input data-pattern="\${pi}" data-key="name" value="\${esc(p.name)}" placeholder="single_bond_cleavage"></label><label>Reactant SMARTS<input data-pattern="\${pi}" data-key="reactant_smarts" value="\${esc(p.reactant_smarts)}" placeholder="[!#1:1]-[!#1:2]"></label></div><div class="section-title"><h3>Products</h3><button data-add-product="\${pi}">Add Product</button></div><div class="products">\${p.products.map((x,xi)=>\`<div class="product"><label>Product name<input data-pattern="\${pi}" data-product="\${xi}" data-key="name" value="\${esc(x.name)}"></label><label>Product SMARTS<input data-pattern="\${pi}" data-product="\${xi}" data-key="smarts" value="\${esc(x.smarts)}" placeholder="[!#1:1]"></label><button class="danger" data-remove-product="\${pi}:\${xi}">Remove</button></div>\`).join('')}</div></section>\`).join('');}
  function update(){clearTimeout(timer);timer=setTimeout(()=>vscode.postMessage({type:'update',value:model}),180)}
  document.getElementById('setName').addEventListener('input',e=>{model.cleavage_pattern_set.name=e.target.value;update()});document.getElementById('addPattern').onclick=()=>{model.cleavage_pattern_set.patterns.push({name:'',reactant_smarts:'',products:[]});render();update()};
  document.getElementById('patterns').addEventListener('input',e=>{const pi=Number(e.target.dataset.pattern);if(!Number.isInteger(pi))return;const xi=e.target.dataset.product;if(xi===undefined)model.cleavage_pattern_set.patterns[pi][e.target.dataset.key]=e.target.value;else model.cleavage_pattern_set.patterns[pi].products[Number(xi)][e.target.dataset.key]=e.target.value;update()});
  document.getElementById('patterns').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.addProduct!==undefined){model.cleavage_pattern_set.patterns[Number(b.dataset.addProduct)].products.push({name:'',smarts:''})}else if(b.dataset.removePattern!==undefined){model.cleavage_pattern_set.patterns.splice(Number(b.dataset.removePattern),1)}else if(b.dataset.removeProduct){const [pi,xi]=b.dataset.removeProduct.split(':').map(Number);model.cleavage_pattern_set.patterns[pi].products.splice(xi,1)}else return;render();update()});
  window.addEventListener('message',e=>{const m=e.data;if(m.type==='document'){model=m.value;render();document.getElementById('message').textContent='Changes are stored in the JSON document. Use Ctrl/Cmd+S to save.';document.getElementById('message').className='muted'}if(m.type==='error'){document.getElementById('message').textContent=m.message;document.getElementById('message').className='error'}});vscode.postMessage({type:'ready'});
  </script></main></body></html>`;
}

module.exports = { register, normalizeDocument, emptyDocument, FILE_SUFFIX };
