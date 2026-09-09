function groupedArgs(request, outputImage) {
  const args = ['compare', '--request-json', JSON.stringify(request),
    '--width', String(request.width || 1000), '--height', String(request.height || 600),
    '--background-color', request.backgroundColor || '#ffffff',
    '--text-color', request.textColor || '#555555',
    '--graph-opacity', String(request.graphOpacity ?? 1),
    '--x-label-size', String(request.xLabelSize ?? 12),
    '--y-label-size', String(request.yLabelSize ?? 11),
    '--title-size', String(request.titleSize ?? 18)];
  if (!request.transparent) args.push('--opaque');
  if (outputImage) args.push('--output-image', outputImage);
  return args;
}

function groupedHtml() {
  return `<section id="groupedEvaluation" hidden>
  <header><div><span class="eyebrow">CLEFTS EVALUATION</span><h1>Grouped Box Plot</h1><p class="muted">Compare multiple .mssim results by database group and tool series.</p></div><div class="actions"><button id="groupedLoadConfig">Load Settings</button><button id="groupedSaveConfig">Save Settings</button><button id="groupedSave" disabled>Save SVG</button></div></header>
  <div class="grouped-layout"><aside>
    <section class="panel"><h2>Groups (x-axis)</h2><p class="muted">For example: MoNA, MassBank, or an evaluation subset.</p><div id="groupedGroups" class="edit-list"></div><button id="groupedAddGroup">＋ Add group</button></section>
    <section class="panel"><h2>Series</h2><p class="muted">For example: CLEFTS or FIORA. A series keeps the same color in every group.</p><div id="groupedSeries" class="edit-list"></div><button id="groupedAddSeries">＋ Add series</button></section>
    <section class="panel"><h2>Image</h2><label>Title<input id="groupedTitle" value="MS/MS similarity comparison"></label><div class="size-row"><label>Width<input id="groupedWidth" type="number" min="400" max="8192" value="1000"></label><label>Height<input id="groupedHeight" type="number" min="300" max="8192" value="600"></label></div><label>Graph opacity<input id="groupedGraphOpacity" type="number" min="0" max="1" step="0.05" value="1"></label><div class="size-row"><label>X-axis label size<input id="groupedXLabelSize" type="number" min="6" max="96" value="12"></label><label>Y-axis label size<input id="groupedYLabelSize" type="number" min="6" max="96" value="11"></label></div><label>Title size<input id="groupedTitleSize" type="number" min="6" max="96" value="18"></label><div class="size-row"><label>Group gap (px)<input id="groupedGroupGap" type="number" min="0" step="1" value="56"></label><label>Within-group gap (px)<input id="groupedSeriesGap" type="number" min="0" step="1" value="6"></label></div><label>Box width (px, 0 = auto)<input id="groupedBoxWidth" type="number" min="0" step="1" value="0"><small class="muted">Auto expands boxes with the image width. A fixed value keeps boxes narrow and centers each group.</small></label><label>Text and axis color<input id="groupedText" type="color" value="#555555"></label><label class="check"><input id="groupedTransparent" type="checkbox" checked>Transparent background</label><label>Background color<input id="groupedBackground" type="color" value="#ffffff" disabled></label><button id="groupedGenerate" class="primary">Generate box plot</button></section>
  </aside><section>
    <section class="panel"><h2>Similarity results</h2><p class="muted">Assign one .mssim file to each group/series pair. Enable “No data” when a tool produced no result for that group.</p><div class="result-matrix-scroll"><table class="result-matrix"><thead><tr><th>Group</th><th>Series</th><th>.mssim file</th><th>No data</th></tr></thead><tbody id="groupedEntries"></tbody></table></div></section>
    <section class="panel"><div class="actions"><span id="groupedStatus" class="status">Configure groups and result files.</span></div><div id="groupedChart" class="chart"><p class="muted">The grouped box plot will appear here.</p></div><div class="result-matrix-scroll"><table id="groupedTable" hidden><thead><tr><th>Group</th><th>Series</th><th>Status</th><th>n</th><th>Invalid</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th></tr></thead><tbody></tbody></table></div></section>
  </section></div></section>`;
}

function groupedClient() {
  const root = document.getElementById('groupedEvaluation');
  const get = id => root.querySelector('#' + id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  let nextGroup = 2, nextSeries = 2;
  let groups = [{ id: 'g1', name: 'MoNA' }];
  let series = [{ id: 's1', name: 'CLEFTS', color: '#36c5a2' }];
  const entries = new Map();
  const key = (groupId, seriesId) => groupId + '::' + seriesId;
  const entry = (groupId, seriesId) => {
    const pair = key(groupId, seriesId);
    if (!entries.has(pair)) entries.set(pair, { groupId, seriesId, path: '', noData: false });
    return entries.get(pair);
  };
  function move(items, index, delta) {
    const target = Math.max(0, Math.min(items.length - 1, index + delta));
    if (target !== index) items.splice(target, 0, items.splice(index, 1)[0]);
  }
  function renderDefinitions() {
    get('groupedGroups').innerHTML = groups.map((item, index) => `<div class="edit-row" data-group="${item.id}"><input value="${esc(item.name)}" aria-label="Group name"><button data-move="-1" title="Move up">↑</button><button data-move="1" title="Move down">↓</button><button data-remove title="Remove">×</button></div>`).join('');
    get('groupedSeries').innerHTML = series.map((item, index) => `<div class="edit-row series-row" data-series="${item.id}"><input value="${esc(item.name)}" aria-label="Series name"><input type="color" value="${item.color}" aria-label="Series color"><button data-move="-1" title="Move up">↑</button><button data-move="1" title="Move down">↓</button><button data-remove title="Remove">×</button></div>`).join('');
    renderEntries();
  }
  function renderEntries() {
    get('groupedEntries').innerHTML = groups.flatMap(group => series.map(item => {
      const value = entry(group.id, item.id);
      return `<tr data-group="${group.id}" data-series="${item.id}"><td>${esc(group.name)}</td><td><span class="series-dot" style="background:${item.color}"></span>${esc(item.name)}</td><td><div class="path"><input data-result-path data-path-kind="file" value="${esc(value.path)}" placeholder="result.mssim" ${value.noData ? 'disabled' : ''}><button data-result-browse ${value.noData ? 'disabled' : ''}>Browse</button></div></td><td><label class="check"><input data-no-data type="checkbox" ${value.noData ? 'checked' : ''}>No data</label></td></tr>`;
    })).join('');
  }
  function request() {
    return {
      title: get('groupedTitle').value,
      groups: groups.map(item => ({ ...item })), series: series.map(item => ({ ...item })),
      entries: groups.flatMap(group => series.map(item => ({ ...entry(group.id, item.id) }))),
      width: Number(get('groupedWidth').value), height: Number(get('groupedHeight').value),
      graphOpacity: Number(get('groupedGraphOpacity').value),
      xLabelSize: Number(get('groupedXLabelSize').value),
      yLabelSize: Number(get('groupedYLabelSize').value),
      titleSize: Number(get('groupedTitleSize').value),
      groupGap: Number(get('groupedGroupGap').value), seriesGap: Number(get('groupedSeriesGap').value),
      boxWidth: Number(get('groupedBoxWidth').value),
      transparent: get('groupedTransparent').checked, backgroundColor: get('groupedBackground').value,
      textColor: get('groupedText').value,
    };
  }
  function status(text, error = false) { get('groupedStatus').textContent = text; get('groupedStatus').className = 'status' + (error ? ' error' : ''); }
  get('groupedAddGroup').onclick = () => { const number = nextGroup++; groups.push({ id: 'g' + number, name: 'Group ' + number }); renderDefinitions(); };
  get('groupedAddSeries').onclick = () => { const number = nextSeries++; series.push({ id: 's' + number, name: 'Series ' + number, color: '#e67722' }); renderDefinitions(); };
  get('groupedGroups').oninput = event => { const row = event.target.closest('[data-group]'); if (row) { groups.find(item => item.id === row.dataset.group).name = event.target.value; renderEntries(); } };
  get('groupedSeries').oninput = event => { const row = event.target.closest('[data-series]'); if (!row) return; const item = series.find(value => value.id === row.dataset.series); if (event.target.type === 'color') item.color = event.target.value; else item.name = event.target.value; renderEntries(); };
  function definitionClick(event, items, attribute) {
    const button = event.target.closest('button'), row = event.target.closest('[' + attribute + ']');
    if (!button || !row) return;
    const index = items.findIndex(item => item.id === row.dataset[attribute.replace('data-', '')]);
    if (button.hasAttribute('data-remove')) {
      if (items.length === 1) return status('At least one group and one series are required.', true);
      items.splice(index, 1);
    } else move(items, index, Number(button.dataset.move));
    renderDefinitions();
  }
  get('groupedGroups').onclick = event => definitionClick(event, groups, 'data-group');
  get('groupedSeries').onclick = event => definitionClick(event, series, 'data-series');
  get('groupedEntries').onchange = event => {
    const row = event.target.closest('tr'); if (!row) return;
    const value = entry(row.dataset.group, row.dataset.series);
    if (event.target.matches('[data-no-data]')) { value.noData = event.target.checked; renderEntries(); }
    else if (event.target.matches('[data-result-path]')) value.path = event.target.value;
  };
  get('groupedEntries').onclick = event => {
    const button = event.target.closest('[data-result-browse]'); if (!button) return;
    const row = button.closest('tr'); vscode.postMessage({ type: 'pickGroupedInput', groupId: row.dataset.group, seriesId: row.dataset.series });
  };
  get('groupedTransparent').onchange = () => { get('groupedBackground').disabled = get('groupedTransparent').checked; };
  get('groupedGenerate').onclick = () => { status('Generating grouped box plot…'); vscode.postMessage({ type: 'groupedSummarize', request: request() }); };
  get('groupedSave').onclick = () => vscode.postMessage({ type: 'saveGroupedImage', request: request() });
  get('groupedSaveConfig').onclick = () => vscode.postMessage({ type: 'saveEvaluationConfig', kind: 'grouped', config: request() });
  get('groupedLoadConfig').onclick = () => vscode.postMessage({ type: 'loadEvaluationConfig', kind: 'grouped' });
  window.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'groupedInputPicked') {
      const value = entry(message.groupId, message.seriesId); value.path = message.path; value.noData = false; renderEntries();
    } else if (message.type === 'groupedConfigLoaded') {
      const config = message.config;
      if (!Array.isArray(config.groups) || !config.groups.length || !Array.isArray(config.series) || !config.series.length) return status('The settings file requires at least one group and one series.', true);
      groups = config.groups.map(item => ({ id: String(item.id), name: String(item.name) }));
      series = config.series.map(item => ({ id: String(item.id), name: String(item.name), color: item.color || '#36c5a2' }));
      entries.clear();
      if (Array.isArray(config.entries)) config.entries.forEach(item => entries.set(key(String(item.groupId), String(item.seriesId)), { groupId: String(item.groupId), seriesId: String(item.seriesId), path: item.path || '', noData: Boolean(item.noData) }));
      nextGroup = 1; while (groups.some(item => item.id === 'g' + nextGroup)) nextGroup++;
      nextSeries = 1; while (series.some(item => item.id === 's' + nextSeries)) nextSeries++;
      get('groupedTitle').value = config.title || 'MS/MS similarity comparison';
      get('groupedWidth').value = config.width || 1000; get('groupedHeight').value = config.height || 600;
      get('groupedGraphOpacity').value = config.graphOpacity ?? 1;
      get('groupedXLabelSize').value = config.xLabelSize ?? 12;
      get('groupedYLabelSize').value = config.yLabelSize ?? 11;
      get('groupedTitleSize').value = config.titleSize ?? 18;
      get('groupedGroupGap').value = config.groupGap ?? 56;
      get('groupedSeriesGap').value = config.seriesGap ?? 6;
      get('groupedBoxWidth').value = config.boxWidth ?? 0;
      get('groupedTransparent').checked = config.transparent !== false;
      get('groupedBackground').value = config.backgroundColor || '#ffffff';
      get('groupedText').value = config.textColor || '#555555';
      get('groupedBackground').disabled = get('groupedTransparent').checked;
      get('groupedSave').disabled = true; renderDefinitions(); status('Loaded settings ' + message.path);
    } else if (message.type === 'groupedConfigSaved') status('Saved settings ' + message.path);
    else if (message.type === 'groupedBusy') status(message.text);
    else if (message.type === 'groupedSummary') {
      get('groupedChart').innerHTML = message.result.svg; get('groupedSave').disabled = false;
      const groupNames = new Map(message.result.groups.map(item => [item.id, item.name]));
      const seriesNames = new Map(message.result.series.map(item => [item.id, item.name]));
      const f = value => value === null ? '—' : Number(value).toFixed(4);
      get('groupedTable').hidden = false;
      get('groupedTable').querySelector('tbody').innerHTML = message.result.cells.map(cell => `<tr><td>${esc(groupNames.get(cell.groupId))}</td><td>${esc(seriesNames.get(cell.seriesId))}</td><td>${cell.status === 'ok' ? 'Available' : 'No data'}</td><td>${cell.count}</td><td>${cell.invalidCount}</td><td>${f(cell.min)}</td><td>${f(cell.q1)}</td><td>${f(cell.median)}</td><td>${f(cell.q3)}</td><td>${f(cell.max)}</td></tr>`).join('');
      const available = message.result.cells.filter(cell => cell.status === 'ok').length;
      status(`${available} available · ${message.result.cells.length - available} no-data combinations`);
    } else if (message.type === 'groupedSaved') status('Saved ' + message.path);
    else if (message.type === 'groupedError') status(message.message, true);
  });
  renderDefinitions();
}

module.exports = { groupedArgs, groupedHtml, groupedClient };
