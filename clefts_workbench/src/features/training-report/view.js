const fs = require('fs');
const path = require('path');
const trainingMetrics = require('../training-metrics/editor');

const STATS = new Set(['q10','q25','median','q75','q90','mean','value']);

function reportCharts(series, maxDepth = null) {
  const charts = new Map();
  for (const source of series) {
    let split = 'all', metric, statistic = 'value';
    if (source.name.startsWith('distributions/')) {
      const match = source.name.match(/^distributions\/([^/]+)\/(.*)_(min|q1|q10|q25|median|q3|q75|q90|max|mean)$/);
      if (!match) continue;
      [, split, metric, statistic] = match;
      statistic = ({ min:'q10', q1:'q25', q3:'q75', max:'q90' })[statistic] || statistic;
    } else {
      metric = source.name.replace(/^metrics\//, '');
      const match = metric.match(/^(train_window|train|validation|val)_(.*)$/);
      if (match) { split = match[1]; metric = match[2]; }
    }
    if (split === 'val') split = 'validation';
    const facet = metric.match(/^(.*?)@([^:]+):(.+)$/);
    const base = facet ? facet[1] : metric, dimension = facet ? facet[2] : '', category = facet ? facet[3] : 'all';
    const depth = dimension === 'depth' ? Number(category) : Number(base.match(/(?:^|_)depth_(\d+)$/)?.[1]);
    if (maxDepth !== null && Number.isFinite(depth) && depth > maxDepth) continue;
    const key = base + '@@' + dimension;
    if (!charts.has(key)) charts.set(key, { metric: base, dimension, lanes: [] });
    const chart = charts.get(key);
    let lane = chart.lanes.find(item => item.split === split && item.category === category);
    if (!lane) { lane = { split, category, stats: {} }; chart.lanes.push(lane); }
    lane.stats[statistic] = source.points;
  }
  return [...charts.values()].sort((a,b) => (a.dimension+a.metric).localeCompare(b.dimension+b.metric));
}

function safeResolve(root, relative) {
  if (!relative) return null;
  const target = path.resolve(root, relative), prefix = root.endsWith(path.sep) ? root : root + path.sep;
  return target === root || target.startsWith(prefix) ? target : null;
}

async function validationResult(root, manifest) {
  let target = safeResolve(root, manifest.latestValidation);
  if (!target || !fs.existsSync(target)) {
    const directory = path.join(root, manifest.artifacts?.validationDirectory || 'spectrum_validation');
    let files = [];
    try { files = (await fs.promises.readdir(directory)).filter(file => /^(epoch|step)_\d+\.json$/.test(file)); } catch {}
    files.sort((a,b) => (Number(a.match(/\d+/)?.[0]) || 0) - (Number(b.match(/\d+/)?.[0]) || 0));
    target = files.length ? path.join(directory, files.at(-1)) : null;
  }
  if (!target) return null;
  const value = JSON.parse(await fs.promises.readFile(target, 'utf8'));
  const spectra = value.spectra || [], representatives = {};
  for (const [metric, entries] of Object.entries(value.representatives || {})) {
    representatives[metric] = {};
    for (const [quantile, item] of Object.entries(entries)) {
      const spectrum = spectra[item.spectrum_index];
      if (spectrum) representatives[metric][quantile] = { ...item, spectrum };
    }
  }
  return { file: path.relative(root, target), summary: value.summary || {}, representatives };
}

async function readTrainingReport(reportPath) {
  const manifest = JSON.parse(await fs.promises.readFile(reportPath, 'utf8'));
  if (manifest.schema !== 'clefts.training-report') throw new Error('Unsupported training report schema.');
  let run = { series: [] };
  try { run = await trainingMetrics.readRun(reportPath); }
  catch (error) { if (!/No metrics\.tsv/.test(error.message)) throw error; }
  const root = path.dirname(reportPath);
  return { manifest, root, charts: reportCharts(run.series || [], run.maxDepth ?? null), validation: await validationResult(root, manifest) };
}

function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char])); }

function client() {
  const data = JSON.parse(document.getElementById('trainingReportData').textContent), el = id => document.getElementById(id);
  const saved = vscode.getState() || {};
  const splitColors={train:'#e76f51',train_window:'#f4a261',validation:'#3a86ff',intermediate_validation:'#2a9d8f',all:'#8b8f98'};
  const splitDashes={train:'',train_window:'4 2',validation:'',intermediate_validation:'6 3',all:''};
  const categoryColors=['#e76f51','#2a9d8f','#e9c46a','#264653','#8ab17d','#f4a261','#577590','#b56576','#6d597a','#ee6c4d'];
  const METRIC_GROUPS=[
    ['Spectrum similarity',/cosine_similarity|assignment_score|spectrum_nonempty_fraction|teacher_spectrum/],
    ['Loss',/loss/],
    ['Action recall & precision',/recall|precision|positive_action|branch_positive|positive_fraction|negative_fraction|valid_candidate_count|normalized_replacement_rate|actions_(before|after)_filter|training_pool_actions/],
    ['Optimization',/^(learning_rate|gradient_norm)$/],
    ['Dataset & throughput',/samples|batches|fragment_nodes|validation_fraction/],
    ['Performance & resources',/seconds|memory/],
  ];
  const GROUP_ORDER=METRIC_GROUPS.map(item=>item[0]).concat('Other');
  const metricGroup=metric=>{const base=metric.replace(/_without_precursor$/,'');for(const [label,pattern] of METRIC_GROUPS)if(pattern.test(base))return label;return 'Other';};
  const without = metric => metric.endsWith('_without_precursor');
  const labelMetric = metric => metric.replace(/_without_precursor$/, '') + (without(metric) ? ' (without precursor)' : '');
  const dimensions = [...new Set(data.charts.map(chart => chart.dimension))];
  for (const dimension of dimensions) { const option=document.createElement('option'); option.value=dimension; option.textContent=dimension ? dimension.replaceAll('_',' ') : 'Overall'; el('reportFacet').append(option); }
  if (['mean','median','distribution'].includes(saved.mode)) el('reportMode').value=saved.mode;
  if (dimensions.includes(saved.facet)) el('reportFacet').value=saved.facet;
  if (typeof saved.includePrecursor==='boolean') el('reportPrecursor').checked=saved.includePrecursor;
  if (typeof saved.filter==='string') el('reportFilter').value=saved.filter;
  if (['cosine_similarity','assignment_score'].includes(saved.representativeMetric)) el('representativeMetric').value=saved.representativeMetric;
  el('reportAuto').checked=typeof saved.auto==='boolean'?saved.auto:data.manifest.status==='running';
  const save=()=>vscode.setState({mode:el('reportMode').value,facet:el('reportFacet').value,includePrecursor:el('reportPrecursor').checked,filter:el('reportFilter').value,representativeMetric:el('representativeMetric').value,representativeQuantile:el('representativeLevel')?.value||saved.representativeQuantile,auto:el('reportAuto').checked});
  function svgNode(svg, tag, attributes, text) { const node=document.createElementNS('http://www.w3.org/2000/svg',tag); for(const [key,value] of Object.entries(attributes))node.setAttribute(key,value); if(text!==undefined)node.textContent=text; svg.append(node); return node; }
  function renderChart(chart) {
    const card=document.createElement('section'), title=document.createElement('h3'); title.textContent=labelMetric(chart.metric)+(chart.dimension?' · '+chart.dimension.replaceAll('_',' '):''); card.append(title);
    const lanes=chart.lanes, categories=[...new Set(lanes.map(lane=>lane.category))], legend=document.createElement('div'); legend.className='report-legend';
    const faceted=!!chart.dimension&&categories.length>1;
    const color=lane=>faceted?categoryColors[categories.indexOf(lane.category)%categoryColors.length]:(splitColors[lane.split]||splitColors.all);
    const dash=lane=>faceted?(splitDashes[lane.split]||''):'';
    for(const lane of lanes){
      const item=document.createElement('span');item.className='report-legend-item';
      const swatch=document.createElementNS('http://www.w3.org/2000/svg','svg');swatch.setAttribute('viewBox','0 0 28 10');swatch.setAttribute('class','legend-swatch');
      const ln=document.createElementNS('http://www.w3.org/2000/svg','line');ln.setAttribute('x1',1);ln.setAttribute('x2',27);ln.setAttribute('y1',5);ln.setAttribute('y2',5);
      ln.setAttribute('stroke',color(lane));ln.setAttribute('stroke-width',3);ln.setAttribute('stroke-dasharray',dash(lane));ln.setAttribute('stroke-linecap','round');swatch.append(ln);
      const label=document.createElement('span');label.textContent=(chart.dimension?lane.category+' · ':'')+lane.split;
      item.append(swatch,label);legend.append(item);
    }
    card.append(legend);
    const mode=el('reportMode').value;
    const visibleStats=lane=>mode==='distribution'?['q10','q25','median','q75','q90']:mode==='median'?['median','value']:['mean','value'];
    const points=lanes.flatMap(lane=>visibleStats(lane).flatMap(stat=>lane.stats[stat]||[]));
    if(!points.length){const p=document.createElement('p');p.className='muted';p.textContent='No '+mode+' measurements are available.';card.append(p);return card;}
    let min=Math.min(...points.map(p=>p[1])),max=Math.max(...points.map(p=>p[1])),first=Math.min(...points.map(p=>p[0])),last=Math.max(...points.map(p=>p[0]));if(min===max){min-=Math.abs(min)*.05||1;max+=Math.abs(max)*.05||1;}
    const x=step=>68+(step-first)/(last-first||1)*510,y=value=>220-(value-min)/(max-min)*190;
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 600 255');svg.setAttribute('role','img');svg.setAttribute('aria-label',title.textContent);
    for(let i=0;i<=4;i++){const value=min+(max-min)*i/4;svgNode(svg,'line',{x1:68,x2:578,y1:y(value),y2:y(value),stroke:'currentColor',opacity:'.14'});svgNode(svg,'text',{x:62,y:y(value)+4,fill:'currentColor','font-size':10,'text-anchor':'end'},value.toPrecision(4));}
    const coords=values=>values.map(p=>x(p[0])+','+y(p[1])).join(' ');
    for(const lane of lanes){const line=(values,width=2,opacity=1)=>{if(!values?.length)return;svgNode(svg,'polyline',{points:coords(values),fill:'none',stroke:color(lane),'stroke-width':width,'stroke-dasharray':dash(lane),opacity});};const band=(lower,upper,opacity)=>{const map=new Map(upper||[]),paired=(lower||[]).filter(p=>map.has(p[0]));if(paired.length)svgNode(svg,'polygon',{points:coords(paired)+' '+coords(paired.map(p=>[p[0],map.get(p[0])]).reverse()),fill:color(lane),opacity});};if(mode==='distribution'){band(lane.stats.q10,lane.stats.q90,.11);band(lane.stats.q25,lane.stats.q75,.25);line(lane.stats.median);}else line(lane.stats[mode]||lane.stats.value);}
    svgNode(svg,'text',{x:68,y:243,fill:'currentColor','font-size':11},String(first));svgNode(svg,'text',{x:578,y:243,fill:'currentColor','font-size':11,'text-anchor':'end'},String(last));card.append(svg);return card;
  }
  function renderCharts(){
    const query=el('reportFilter').value.trim().toLowerCase(),facet=el('reportFacet').value,include=el('reportPrecursor').checked,names=new Set(data.charts.filter(chart=>chart.dimension===facet).map(chart=>chart.metric));
    const visible=data.charts.filter(chart=>{
      const paired=without(chart.metric)?names.has(chart.metric.replace(/_without_precursor$/,'')):names.has(chart.metric+'_without_precursor');
      return chart.dimension===facet&&!(paired&&without(chart.metric)===include)&&labelMetric(chart.metric).toLowerCase().includes(query);
    }).sort((a,b)=>{
      const ga=GROUP_ORDER.indexOf(metricGroup(a.metric)),gb=GROUP_ORDER.indexOf(metricGroup(b.metric));
      return ga!==gb?ga-gb:labelMetric(a.metric).localeCompare(labelMetric(b.metric));
    });
    el('reportCharts').replaceChildren();
    const showGroups=new Set(visible.map(chart=>metricGroup(chart.metric))).size>1;
    let lastGroup=null;
    for(const chart of visible){
      const group=metricGroup(chart.metric);
      if(showGroups&&group!==lastGroup){
        const header=document.createElement('div');header.className='report-group-header';header.textContent=group;
        el('reportCharts').append(header);lastGroup=group;
      }
      el('reportCharts').append(renderChart(chart));
    }
    el('chartCount').textContent=visible.length+' metric cards';
  }
  function peaks(record, kind){const values=record[kind+'_peaks'];if(values)return values;return (record[kind+'_mz']||[]).map((mz,index)=>({mz,intensity:record[kind+'_intensity'][index],precursor:false}));}
  function renderSpectra(){const metric=el('representativeMetric').value+(el('reportPrecursor').checked?'':'_without_precursor'),entries=data.validation?.representatives?.[metric]||{},levels=['q10','q25','median','q75','q90'].filter(level=>entries[level]);el('representativeSpectra').replaceChildren();if(levels.length){const card=document.createElement('article'),heading=document.createElement('h4'),level=document.createElement('select');level.id='representativeLevel';for(const value of levels){const option=document.createElement('option');option.value=value;option.textContent=value;level.append(option);}if(levels.includes(saved.representativeQuantile))level.value=saved.representativeQuantile;const control=document.createElement('label');control.append(document.createTextNode('Representative level '),level);card.append(heading,control);const plot=document.createElement('div'),meta=document.createElement('p');meta.className='muted';card.append(plot,meta);const draw=()=>{saved.representativeQuantile=level.value;save();const quantile=level.value,item=entries[quantile],record=item.spectrum,include=el('reportPrecursor').checked,generated=peaks(record,'generated').filter(p=>include||!p.precursor),original=(record.original_peaks||[]).filter(p=>include||!p.precursor),all=[...generated,...original],maximum=Math.max(...all.map(p=>Number(p.mz)),1),gmax=Math.max(...generated.map(p=>Number(p.intensity)),1e-8),omax=Math.max(...original.map(p=>Number(p.intensity)),1e-8),x=mz=>42+Number(mz)/maximum*516;heading.innerHTML='';const levelLabel=document.createElement('span');levelLabel.className='rep-level-label';levelLabel.textContent=quantile+' representative';const badge=document.createElement('span');badge.className='similarity-badge';badge.textContent=el('representativeMetric').selectedOptions[0].textContent+' '+Number(item.value).toFixed(4);heading.append(levelLabel,badge);const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 570 270');svg.setAttribute('role','img');svg.setAttribute('aria-label',quantile+' representative mirror spectrum');svgNode(svg,'line',{x1:42,x2:558,y1:130,y2:130,stroke:'currentColor'});for(const peak of generated)svgNode(svg,'line',{x1:x(peak.mz),x2:x(peak.mz),y1:130,y2:130-105*peak.intensity/gmax,stroke:peak.precursor?'#f0883e':'#58a6ff','stroke-width':2});for(const peak of original)svgNode(svg,'line',{x1:x(peak.mz),x2:x(peak.mz),y1:130,y2:130+105*peak.intensity/omax,stroke:peak.precursor?'#f0883e':'#3fb950','stroke-width':2});svgNode(svg,'text',{x:45,y:18,fill:'#58a6ff','font-size':11},'Generated');svgNode(svg,'text',{x:45,y:258,fill:'#3fb950','font-size':11},'Observed');svgNode(svg,'text',{x:558,y:148,fill:'currentColor','font-size':10,'text-anchor':'end'},maximum.toFixed(2)+' m/z');plot.replaceChildren(svg);meta.textContent=record.main_adduct+' · '+record.collision_energy+' eV · '+generated.length+' / '+original.length+' peaks';};level.onchange=draw;draw();el('representativeSpectra').append(card);}el('representativeStatus').textContent=data.validation?data.validation.file+' · '+(data.validation.summary.samples||0)+' spectra':'No validation spectrum artifact is available yet.';}
  for(const id of ['reportMode','reportFacet','reportPrecursor','representativeMetric'])el(id).addEventListener('change',()=>{save();renderCharts();renderSpectra();});el('reportAuto').onchange=save;el('reportFilter').oninput=()=>{save();renderCharts();};el('reportRefresh').onclick=()=>vscode.postMessage({type:'refresh'});setInterval(()=>{if(el('reportAuto').checked)vscode.postMessage({type:'refresh'});},15000);renderCharts();renderSpectra();
}

function html(data, commonCss) {
  const manifest=data.manifest, training=manifest.datasets?.train?.samples??'—', validation=manifest.datasets?.validation?.samples??'—';
  const serialized=JSON.stringify(data).replace(/</g,'\\u003c');
  return `<!doctype html><html><head><meta charset="UTF-8"><style>${commonCss}
  body{margin:0;background:var(--vscode-editor-background);color:var(--vscode-foreground)}.report-main{padding:24px 28px}.report-toolbar{display:flex;gap:12px;align-items:center;padding:14px 24px;border-bottom:1px solid var(--border);background:var(--vscode-input-background)}.report-toolbar strong{margin-right:auto}.report-controls,.report-legend{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.report-controls input[type=search]{min-width:250px}.report-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px;margin-top:16px;align-items:start}.report-grid section{margin:0;content-visibility:auto;contain-intrinsic-size:auto 360px}.report-grid svg,.spectrum-grid svg{display:block;width:100%}.report-group-header{grid-column:1/-1;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;opacity:.7;margin:18px 0 -4px;padding-top:14px;border-top:1px solid var(--border)}.report-group-header:first-child{margin-top:0;padding-top:0;border-top:0}.report-legend{font-size:12px;margin:8px 0;row-gap:6px}.report-legend-item{display:inline-flex;align-items:center;gap:6px}.legend-swatch{width:26px;height:10px;flex:0 0 auto}.spectrum-grid{display:grid;grid-template-columns:minmax(min(100%,760px),1fr);gap:12px}.spectrum-grid article{border:1px solid var(--border);border-radius:9px;padding:12px}.spectrum-grid h4{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:0 0 6px}.rep-level-label{font-weight:600}.similarity-badge{background:color-mix(in srgb,var(--accent) 22%,transparent);color:var(--accent);padding:2px 10px;border-radius:999px;font-weight:700;font-size:12px}.status-${escapeHtml(manifest.status)}{color:var(--accent)}</style></head><body><div class="report-toolbar"><strong>CLEFTS Workbench <small>/ Training Result</small></strong><span class="status-${escapeHtml(manifest.status)}">${escapeHtml(manifest.status)}</span><label><input id="reportAuto" type="checkbox"> Auto refresh (15s)</label><button id="reportRefresh">Refresh</button></div><main class="report-main"><header><div><span class="eyebrow">${escapeHtml(manifest.project)} / ${escapeHtml(manifest.run)}</span><h1>Fragment Tree Training Report</h1><p class="muted">${escapeHtml(data.root)} · updated ${escapeHtml(manifest.updatedAt)}</p></div></header><section class="cards"><article><b>${manifest.completedEpochs||0}</b><span>Completed epochs</span></article><article><b>${manifest.globalStep||0}</b><span>Global step</span></article><article><b>${training}</b><span>Train samples</span></article><article><b>${validation}</b><span>Validation samples</span></article></section>${manifest.error?`<section><h2>Training failed</h2><pre>${escapeHtml(manifest.error)}</pre></section>`:''}<section><h2>Learning curves</h2><div class="report-controls"><label>Display <select id="reportMode"><option value="mean">Mean only</option><option value="median">Median only</option><option value="distribution">Distribution (q10–q90)</option></select></label><label>Breakdown <select id="reportFacet"></select></label><label><input id="reportPrecursor" type="checkbox" checked> Include precursor</label><input id="reportFilter" type="search" placeholder="Filter metrics"><span id="chartCount" class="muted"></span></div><p class="muted">Train and validation share each card. Distribution shows q10–q90 and q25–q75 bands with the median line.</p><div id="reportCharts" class="report-grid"></div></section><section><h2>Representative validation spectra</h2><div class="report-controls"><label>Rank by <select id="representativeMetric"><option value="cosine_similarity">Cosine similarity</option><option value="assignment_score">Assignment score (m/z coverage)</option></select></label><span id="representativeStatus" class="muted"></span></div><p class="muted">Compare the generated and observed spectrum in one card, switching its representative level between q10, q25, median, q75, and q90.</p><div id="representativeSpectra" class="spectrum-grid"></div></section><script type="application/json" id="trainingReportData">${serialized}</script><script>const vscode=acquireVsCodeApi();(${client.toString()})();</script></main></body></html>`;
}

module.exports = { readTrainingReport, reportCharts, html, STATS };
