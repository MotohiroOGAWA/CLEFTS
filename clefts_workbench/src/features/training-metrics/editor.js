const fs = require('fs');
const path = require('path');
const readline = require('readline');

async function readRun(directory) {
  let report;
  if (directory.endsWith('.pft.json')) {
    report = JSON.parse(await fs.promises.readFile(directory, 'utf8'));
    if (report.schema !== 'clefts.training-report') throw new Error('This .pft.json is not a training report.');
    directory = path.dirname(directory);
  }
  const series = new Map();
  const files = [];
  for (const file of ['metrics.tsv', 'metric_distributions.tsv']) {
    const filename = path.join(directory, file);
    try { await fs.promises.access(filename); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    files.push(file);
    const input = fs.createReadStream(filename);
    const lines = readline.createInterface({ input, crlfDelay: Infinity });
    let headers;
    try {
      for await (const line of lines) {
        const cells = line.split('\t');
        if (!headers) { headers = cells; continue; }
        if (cells.length !== headers.length) continue; // A live writer may leave an incomplete final row.
        const row = Object.fromEntries(headers.map((key, i) => [key, cells[i]]));
        const step = Number(row.global_step);
        if (!row.global_step || !Number.isFinite(step)) continue;
        const add = (key, raw) => {
          const value = Number(raw);
          if (!raw?.trim() || !Number.isFinite(value)) return;
          if (!series.has(key)) series.set(key, new Map());
          series.get(key).set(step, value);
        };
        if (file === 'metric_distributions.tsv') add('distributions/' + row.split + '/' + row.metric, row.value);
        else for (const key of headers) if (!['event', 'epoch', 'global_step', 'branch_id'].includes(key)) add('metrics/' + key, row[key]);
      }
    } finally { lines.close(); input.destroy(); }
  }
  if (!files.length) throw new Error('No metrics.tsv or metric_distributions.tsv found. Select a training run directory.');
  let maxDepth = null;
  try {
    const config = JSON.parse(await fs.promises.readFile(path.join(directory, 'training_config.json'), 'utf8'));
    const value = config.model_config?.fragmenter_params?.fragment_ion_tree_builder?.max_action_count;
    if (Number.isInteger(value) && value >= 0) maxDepth = value;
  } catch {}
  return { directory, report, files, maxDepth, series: [...series].sort(([a], [b]) => a.localeCompare(b)).map(([name, values]) => ({ name, points: [...values].sort((a, b) => a[0] - b[0]) })) };
}
// Keep grouping independent of the webview so recorded naming conventions are testable.
function groupSeries(series, maxDepth = null) {
  const charts = new Map();
  for (const source of series) {
    let metric, split, statistic = 'value';
    if (source.name.startsWith('distributions/')) {
      const match = source.name.match(/^distributions\/([^/]+)\/(.*)_(min|q1|q10|q25|median|q3|q75|q90|max|mean)$/);
      if (!match) continue;
      [, split, metric, statistic] = match;
      statistic = ({q1:'q25',q3:'q75',min:'q10',max:'q90'})[statistic] || statistic;
    } else {
      metric = source.name.replace(/^metrics\//, '');
      const match = metric.match(/^(train_window|train|validation|val)_(.*)$/);
      split = match ? match[1] : 'all';
      if (match) metric = match[2];
    }
    if (split === 'val') split = 'validation';
    const facet = metric.match(/^(.*?)@([^:]+):(.+)$/);
    const dimension = facet ? facet[2] : '';
    const category = facet ? facet[3] : 'all';
    let base = facet ? facet[1] : metric;
    const depth = dimension === 'depth' ? Number(category) : Number(base.match(/(?:^|_)depth_(\d+)$/)?.[1]);
    if (maxDepth !== null && Number.isFinite(depth) && depth > maxDepth) continue;
    const precursor = base.endsWith('_without_precursor') ? 'withoutPrecursor' : 'withPrecursor';
    if (precursor === 'withoutPrecursor') base = base.replace(/_without_precursor$/, '');
    const name = base + (dimension ? ' @' + dimension : '');
    if (!charts.has(name)) charts.set(name, { name, metric: base, dimension, sources: [], variants: { withPrecursor: [], withoutPrecursor: [] } });
    const chart = charts.get(name);
    chart.sources.push(source.name);
    const lanes = chart.variants[precursor];
    let lane = lanes.find(item => item.split === split && item.category === category);
    if (!lane) { lane = { split, category, label: (dimension ? category + ' · ' : '') + split, stats: {} }; lanes.push(lane); }
    lane.stats[statistic] = source.points;
  }
  return [...charts.values()].sort((a,b) => a.name.localeCompare(b.name)).map(chart => {
    for (const lanes of Object.values(chart.variants)) lanes.sort((a,b) => a.category.localeCompare(b.category) || a.split.localeCompare(b.split));
    return chart;
  });
}
function attach(panel) {
  const vscode = require('vscode');
  panel.webview.onDidReceiveMessage(async message => {
    if (message.type === 'metricsPick') {
      const picked = await vscode.window.showOpenDialog({ canSelectFiles: true, canSelectFolders: true, canSelectMany: false,
        filters: { 'Training report or run': ['pft.json', 'json', 'tsv'] } });
      if (picked?.[0]) panel.webview.postMessage({ type: 'metricsPicked', directory: picked[0].fsPath });
    }
    if (message.type !== 'metricsLoad') return;
    try {
      if (typeof message.directory !== 'string' || !path.isAbsolute(message.directory.trim())) throw new Error('Enter an absolute path to the training run or training.pft.json.');
      const data = await readRun(message.directory.trim());
      panel.webview.postMessage({ type: 'metricsData', request: message.request, data });
    } catch (error) { panel.webview.postMessage({ type: 'metricsData', request: message.request, error: error.message }); }
  });
}
function html() {
  return `<div id="metricsApp" hidden><section><h2>Training Report</h2><label>Run directory or training.pft.json<input id="metricsDirectory" type="text" data-path-kind="folder" placeholder="/path/to/run/training.pft.json" style="width:100%"></label><div class="actions"><button id="metricsBrowse">Browse…</button><button id="metricsLoad">Load / Refresh</button><label><input id="metricsAuto" type="checkbox"> Auto refresh (15s)</label></div><p id="metricsStatus" role="status"></p><p class="muted">Each metric appears once. After adding it, switch that card between mean, median, and q10–q90 distribution views, or exclude precursor peaks. Train uses warm colors and validation uses cool colors.</p><div class="actions"><input id="metricsFilter" type="search" placeholder="Filter charts" aria-label="Filter charts"><select id="metricsChoice" aria-label="Chart metric" style="max-width:80%"></select><button id="metricsAdd" disabled>＋ Add chart</button><button id="metricsAddAll" disabled>Add all charts</button><button id="metricsClear">Clear charts</button><label>Layout <select id="metricsLayout"><option value="grid">Grid</option><option value="vertical">Vertical</option></select></label></div></section><div id="metricsCharts" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px"></div></div>`;
}
function client(groupSeries) {
  const el = id => document.getElementById(id);
  const state = vscode.getState() || {};
  let selected = state.metricsSelected || [], data = [], request = 0, busy = false;
  const cardSettings = new Map(Object.entries(state.metricsCardSettings || {}));
  const hiddenLanes = new Map();
  el('metricsDirectory').value = state.metricsDirectory || '';
  el('metricsLayout').value = state.metricsLayout === 'vertical' ? 'vertical' : 'grid';
  const applyLayout = () => { el('metricsCharts').style.gridTemplateColumns = el('metricsLayout').value === 'vertical' ? 'minmax(0, 1fr)' : 'repeat(auto-fit,minmax(min(100%,440px),1fr))'; };
  applyLayout();
  const save = () => vscode.setState({ ...(vscode.getState() || {}), metricsSelected: selected, metricsCardSettings:Object.fromEntries(cardSettings), metricsLayout: el('metricsLayout').value, metricsDirectory: el('metricsDirectory').value });
  function load() {
    if (busy) return;
    busy = true; el('metricsLoad').disabled = true; el('metricsStatus').textContent = 'Loading…'; save();
    vscode.postMessage({ type: 'metricsLoad', directory: el('metricsDirectory').value, request: ++request });
  }
  const matchingCharts = () => {
    const query = el('metricsFilter').value.trim().toLowerCase();
    return data.filter(chart => chart.name.toLowerCase().includes(query));
  };
  function updateChoices() {
    const previous = el('metricsChoice').value;
    const matching = matchingCharts();
    el('metricsChoice').replaceChildren();
    for (const chart of matching) {
      const option = document.createElement('option'); option.value = option.textContent = chart.name;
      el('metricsChoice').append(option);
    }
    if (matching.some(chart => chart.name === previous)) el('metricsChoice').value = previous;
    el('metricsAdd').disabled = !matching.length;
    el('metricsAddAll').disabled = !matching.length;
    el('metricsAddAll').textContent = (el('metricsFilter').value.trim() ? 'Add matching charts' : 'Add all charts') + ' (' + matching.length + ')';
  }
  let renderVersion = 0;
  async function render() {
    const version = ++renderVersion;
    let rendered = 0;
    el('metricsCharts').replaceChildren();
    const chartsByName = new Map(data.map(chart => [chart.name, chart]));
    for (const name of [...selected]) {
      if (rendered > 0 && rendered % 8 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
        if (version !== renderVersion) return;
      }
      const card = document.createElement('section'); card.style.cssText = 'display:block;content-visibility:auto;contain-intrinsic-size:auto 380px';
      const title = document.createElement('strong'); title.textContent = name;
      const remove = document.createElement('button'); remove.textContent = '×'; remove.title = 'Remove chart'; remove.style.float = 'right';
      remove.onclick = () => { selected = selected.filter(n => n !== name); save(); render(); };
      card.append(remove, title);
      const chart = chartsByName.get(name);
      if (!chart) continue;
      const availableWith = chart.variants.withPrecursor.length > 0, availableWithout = chart.variants.withoutPrecursor.length > 0;
      const settings = { mode:'mean', withoutPrecursor:!availableWith && availableWithout, ...(cardSettings.get(name) || {}) };
      if (settings.withoutPrecursor && !availableWithout) settings.withoutPrecursor=false;
      if (!settings.withoutPrecursor && !availableWith) settings.withoutPrecursor=true;
      cardSettings.set(name,settings);
      const controls=document.createElement('div');controls.style.cssText='display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0';
      const modeLabel=document.createElement('label');modeLabel.append(document.createTextNode('Display '));const mode=document.createElement('select');
      for(const [value,labelText] of [['mean','Mean'],['median','Median'],['distribution','Distribution']]){const option=document.createElement('option');option.value=value;option.textContent=labelText;mode.append(option);}mode.value=settings.mode;mode.onchange=()=>{settings.mode=mode.value;save();render();};modeLabel.append(mode);controls.append(modeLabel);
      const precursorLabel=document.createElement('label'),precursor=document.createElement('input');precursor.type='checkbox';precursor.checked=settings.withoutPrecursor;precursor.disabled=!availableWith||!availableWithout;precursor.onchange=()=>{settings.withoutPrecursor=precursor.checked;save();render();};precursorLabel.append(precursor,document.createTextNode(' Exclude precursor'));controls.append(precursorLabel);card.append(controls);
      const hidden = hiddenLanes.get(name) || new Set();
      hiddenLanes.set(name, hidden);
      const allLanes = chart.variants[settings.withoutPrecursor?'withoutPrecursor':'withPrecursor'];
      const lanes = allLanes.filter(lane => !hidden.has(lane.label));
      const splitColors={train:'#e76f51',train_window:'#f4a261',validation:'#3a86ff',intermediate_validation:'#2a9d8f',all:'#8b8f98'};
      const categoryDashes=['','8 3','3 3','10 3 2 3','2 5'];
      const categories=[...new Set(allLanes.map(lane=>lane.category))];
      const color = lane => splitColors[lane.split] || splitColors.all;
      const dash = lane => categoryDashes[categories.indexOf(lane.category)%categoryDashes.length];
      const legend = document.createElement('div'); legend.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;margin:12px 0';
      for (const lane of allLanes) {
        const label = document.createElement('label'); label.style.color = color(lane);
        const toggle = document.createElement('input'); toggle.type = 'checkbox'; toggle.checked = !hidden.has(lane.label);
        toggle.onchange = () => { if (toggle.checked) hidden.delete(lane.label); else hidden.add(lane.label); render(); };
        label.append(toggle, document.createTextNode(lane.label)); legend.append(label);
      }
      card.append(legend);
      if (settings.mode === 'distribution') { const hint = document.createElement('p'); hint.className = 'muted'; hint.textContent = 'Light band: q10–q90 · Dark band: q25–q75 · Line: median'; card.append(hint); }
      const visibleStats=lane=>settings.mode==='distribution'?['q10','q25','median','q75','q90']:settings.mode==='median'?['median','value']:['mean','value'];
      const points = lanes.flatMap(lane => visibleStats(lane).flatMap(stat=>lane.stats[stat]||[]));
      if (!points.length) { const p = document.createElement('p'); p.textContent = 'No measurements available to display.'; card.append(p); }
      else {
        const ns = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 600 260'); svg.style.width = '100%'; svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', name);
        const node = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
        let min = Infinity, max = -Infinity, first = Infinity, last = -Infinity;
        for (const [step, value] of points) { min = Math.min(min, value); max = Math.max(max, value); first = Math.min(first, step); last = Math.max(last, step); }
        if (min === max) { min -= Math.abs(min) * .05 || 1; max += Math.abs(max) * .05 || 1; }
        const x = step => 75 + (step - first) / (last - first || 1) * 500;
        const y = value => 215 - (value - min) / (max - min) * 190;
        const coords = values => values.map(p => x(p[0])+','+y(p[1])).join(' ');
        for (let i = 0; i <= 4; i++) {
          const value = min + (max-min)*i/4;
          node('line', { x1:75, x2:575, y1:y(value), y2:y(value), stroke:'currentColor', opacity:'.15' });
          node('text', { x:70, y:y(value)+4, fill:'currentColor', 'font-size':11, 'text-anchor':'end' }, value.toPrecision(4));
        }
        for (const lane of lanes) {
          const line = (values, opacity = 1, width = 2) => {
            if (!values?.length) return;
            node('polyline', { points:coords(values), fill:'none', stroke:color(lane), 'stroke-width':width, 'stroke-dasharray':dash(lane), opacity });
            if (values.length === 1) node('circle', { cx:x(values[0][0]), cy:y(values[0][1]), r:3, fill:color(lane), opacity });
          };
          const band = (lower, upper, opacity) => {
            const upperMap = new Map(upper || []);
            const paired = (lower || []).filter(p => upperMap.has(p[0]));
            if (paired.length === 1) node('line', { x1:x(paired[0][0]), x2:x(paired[0][0]), y1:y(paired[0][1]), y2:y(upperMap.get(paired[0][0])), stroke:color(lane), 'stroke-width':8, opacity });
            else if (paired.length) node('polygon', { points:coords(paired)+' '+coords(paired.map(p => [p[0], upperMap.get(p[0])]).reverse()), fill:color(lane), opacity });
          };
          if (settings.mode === 'distribution') {
            band(lane.stats.q10, lane.stats.q90, .10); band(lane.stats.q25, lane.stats.q75, .25);
            for (const stat of ['q10', 'q90', 'q25', 'q75']) line(lane.stats[stat], .45, 1);
            line(lane.stats.median);
          } else line(lane.stats[settings.mode] || lane.stats.value);
        }
        node('text', { x:75, y:240, fill:'currentColor', 'font-size':12 }, String(first));
        node('text', { x:575, y:240, fill:'currentColor', 'font-size':12, 'text-anchor':'end' }, String(last));
        const info = document.createElement('p'); info.className = 'muted'; info.style.whiteSpace = 'pre-line'; info.textContent = 'Solid: train · Dotted: train_window · Dashed: validation · Hover to inspect';
        svg.onmousemove = event => {
          const rect = svg.getBoundingClientRect(); const step = first + ((event.clientX-rect.left)/rect.width*600-75)/500*(last-first);
          info.textContent = lanes.map(lane => {
            const values = visibleStats(lane).filter(stat=>lane.stats[stat]?.length).map(stat => { const samples=lane.stats[stat];
              let closest = samples[0]; for (const p of samples) if (Math.abs(p[0]-step)<Math.abs(closest[0]-step)) closest=p;
              return stat+'='+closest[1].toPrecision(5)+' (step '+closest[0]+')';
            });
            return lane.label+': '+values.join(' · ');
          }).join('\n');
        };
        card.append(svg, info);
      }
      el('metricsCharts').append(card);
      rendered++;
    }
  }
  el('metricsFilter').oninput = updateChoices;
  el('metricsAddAll').onclick = () => {
    selected = [...new Set([...selected, ...matchingCharts().map(chart => chart.name)])];
    save(); render();
  };
  el('metricsClear').onclick = () => { selected = []; save(); render(); };
  el('metricsLayout').onchange = () => { applyLayout(); save(); };
  el('metricsLoad').onclick = load;
  el('metricsBrowse').onclick = () => vscode.postMessage({ type:'metricsPick' });
  el('metricsAdd').onclick = () => { const name = el('metricsChoice').value; if (name && !selected.includes(name)) { selected.push(name); save(); render(); } };
  window.addEventListener('message', ({ data: message }) => {
    if (message.type === 'metricsPicked') { el('metricsDirectory').value = message.directory; save(); }
    if (message.type !== 'metricsData' || message.request !== request) return;
    busy = false; el('metricsLoad').disabled = false;
    data = groupSeries(message.data?.series || [],message.data?.maxDepth ?? null);
    if (!message.error) {
      selected = [...new Set(selected.map(name => data.find(chart => chart.name===name||chart.sources.includes(name))?.name || name.replace(/^(mean|distribution|metrics)\//,'')))].filter(name=>data.some(chart=>chart.name===name));
      save();
    }
    el('metricsStatus').textContent = message.error || message.data.directory+' · '+data.length+' charts · '+new Date().toLocaleTimeString();
    updateChoices();
    render();
  });
  setInterval(() => { if (el('metricsAuto').checked && !el('metricsApp').hidden && el('metricsDirectory').value.trim()) load(); }, 15000);
  render();
}
function script() { return '(' + client.toString() + ')(' + groupSeries.toString() + ');'; }
module.exports = { readRun, groupSeries, attach, html, script };
