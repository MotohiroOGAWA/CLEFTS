const fs = require('fs');
const path = require('path');
const readline = require('readline');

async function readScores(root, options = {}) {
  const datasets = [];
  const source = options.source || path.basename(root) || 'Result';
  const resultPath = options.resultPath || root;
  for (const directory of ['', 'train_structures', 'validation_structures']) {
    const file = path.join(root, directory, 'assignment_scores.tsv');
    if (!fs.existsSync(file)) continue;
    const scores = [];
    let column = -1, first = true, skipped = 0;
    const lines = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
    for await (const line of lines) {
      if (first) {
        column = line.replace(/^\uFEFF/, '').split('\t').indexOf('assignment_score');
        first = false;
        continue;
      }
      if (!line.trim()) continue;
      const value = (line.split('\t')[column] || '').trim(), score = Number(value);
      if (!value || !Number.isFinite(score) || score < 0 || score > 1) skipped++;
      else scores.push(score);
    }
    const subset = directory || 'Dataset';
    datasets.push({
      id: `${path.resolve(resultPath)}::${directory || '.'}`,
      source,
      subset,
      resultPath: path.resolve(resultPath),
      label: `${source} — ${subset}`,
      scores,
      skipped,
      error: column < 0 ? 'assignment_score column is missing.' : ''
    });
  }
  return datasets;
}

function histogram(scores, width) {
  if (!Number.isFinite(width) || width < 0.001 || width > 1) throw new Error('Bin width must be between 0.001 and 1.');
  const clean = value => Number(value.toFixed(12));
  const rows = Array.from({ length: Math.ceil(clean(1 / width)) }, (_, i) => ({ lower: clean(i * width), upper: Math.min(1, clean((i + 1) * width)), count: 0 }));
  for (const score of scores) {
    // Compare with the same rounded boundaries that are displayed/exported.
    let low = 0, high = rows.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (score < rows[middle].upper) high = middle;
      else low = middle + 1;
    }
    rows[low].count++;
  }
  let cumulative = 0;
  return rows.map(row => ({ ...row, cumulative: cumulative += row.count, percent: scores.length ? cumulative / scores.length * 100 : 0 }));
}

function distributionClient(initialDatasets, histogram) {
  const section = document.getElementById('scoreDistribution');
  const get = id => section.querySelector('#' + id);
  const palette = ['#4285d4', '#e67722', '#2f9e44', '#9c36b5', '#e03131', '#1098ad', '#f08c00', '#5f3dc4'];
  let colorIndex = 0;
  let datasets = [];
  let selectedRows = [];
  const prepare = data => ({ ...data, enabled: data.enabled !== false, color: data.color || palette[colorIndex++ % palette.length] });
  for (const data of initialDatasets) datasets.push(prepare(data));
  get('scoreControls').hidden = false;

  function renderSeries() {
    const list = get('scoreSeries');
    list.textContent = '';
    for (const data of datasets) {
      const row = document.createElement('div');
      row.className = 'score-series-row';
      const visible = document.createElement('input');
      visible.type = 'checkbox'; visible.checked = data.enabled; visible.dataset.scoreVisible = data.id;
      visible.setAttribute('aria-label', 'Show ' + data.label);
      const color = document.createElement('input');
      color.type = 'color'; color.value = data.color; color.dataset.scoreColor = data.id;
      color.setAttribute('aria-label', 'Color for ' + data.label);
      const label = document.createElement('span');
      label.textContent = `${data.label} (${data.scores.length})`; label.title = data.resultPath || data.label;
      const remove = document.createElement('button');
      remove.type = 'button'; remove.textContent = 'Remove'; remove.dataset.scoreRemove = data.id;
      row.append(visible, color, label, remove);
      list.appendChild(row);
    }
    get('scoreEmpty').hidden = Boolean(datasets.length);
  }

  function draw() {
    const width = Number(get('scoreWidth').value);
    const imageWidth = Number(get('scoreImageWidth').value);
    const imageHeight = Number(get('scoreImageHeight').value);
    const validImageSize = Number.isInteger(imageWidth) && imageWidth >= 320 && imageWidth <= 8192
      && Number.isInteger(imageHeight) && imageHeight >= 240 && imageHeight <= 8192;
    get('scoreBackground').disabled = get('scoreTransparent').checked;
    if (!validImageSize) {
      get('scoreStatus').textContent = 'PNG width must be 320–8192 px and height must be 240–8192 px.';
      get('scorePng').disabled = get('scoreTsv').disabled = true;
      return;
    }
    try {
      selectedRows = datasets.filter(data => data.enabled && !data.error && data.scores.length)
        .map(data => ({ data, rows: histogram(data.scores, width) }));
    } catch (error) {
      get('scoreStatus').textContent = error.message;
      get('scorePng').disabled = get('scoreTsv').disabled = true;
      return;
    }
    const enabled = datasets.filter(data => data.enabled);
    const problems = enabled.filter(data => data.error || !data.scores.length)
      .map(data => `${data.label}: ${data.error || 'no valid scores'}`);
    const total = selectedRows.reduce((sum, item) => sum + item.data.scores.length, 0);
    const skipped = selectedRows.reduce((sum, item) => sum + item.data.skipped, 0);
    get('scoreStatus').textContent = selectedRows.length
      ? `${selectedRows.length} datasets overlaid · ${total} samples · ${selectedRows[0].rows.length} bins · ${skipped} invalid/missing scores excluded.${problems.length ? ' Excluded: ' + problems.join('; ') : ''}`
      : datasets.length ? `Select at least one dataset with valid scores.${problems.length ? ' ' + problems.join('; ') : ''}` : 'No assignment_scores.tsv found. Add a Data Preparation result to compare.';
    get('scorePng').disabled = get('scoreTsv').disabled = !selectedRows.length;
    const canvas = get('scoreCanvas'), ctx = canvas.getContext('2d');
    if (!selectedRows.length) { canvas.hidden = true; get('scoreTable').textContent = ''; return; }
    canvas.hidden = false;
    canvas.width = imageWidth; canvas.height = imageHeight;
    const sx = imageWidth / 1200, sy = imageHeight / 640, scale = Math.min(sx, sy);
    const x = value => value * sx, y = value => value * sy;
    const foreground = get('scoreText').value;
    ctx.clearRect(0, 0, imageWidth, imageHeight);
    if (!get('scoreTransparent').checked) { ctx.fillStyle = get('scoreBackground').value; ctx.fillRect(0, 0, imageWidth, imageHeight); }
    ctx.fillStyle = foreground; ctx.font = `bold ${Math.max(11, 22 * scale)}px sans-serif`; ctx.textAlign = 'center';
    ctx.fillText(selectedRows.length === 1 ? 'Assignment score distribution — ' + selectedRows[0].data.label : 'Assignment score distribution comparison', x(600), y(36));
    ctx.font = `${Math.max(8, 16 * scale)}px sans-serif`; ctx.fillText(`${selectedRows.length} datasets · n = ${total} · bin width = ${width}`, x(600), y(63));
    const legendItems = selectedRows.slice(0, 12), legendColumns = 3;
    ctx.font = `${Math.max(8, 13 * scale)}px sans-serif`; ctx.textAlign = 'left';
    legendItems.forEach((item, index) => {
      const column = index % legendColumns, row = Math.floor(index / legendColumns);
      const lx = x(75 + column * 375), ly = y(88 + row * 20);
      ctx.fillStyle = item.data.color; ctx.fillRect(lx, ly - y(10), x(18), y(10));
      ctx.fillStyle = foreground;
      const suffix = index === 11 && selectedRows.length > 12 ? ` (+${selectedRows.length - 12} more)` : '';
      ctx.fillText(item.data.label.slice(0, 38) + suffix, lx + x(25), ly);
    });
    const legendRows = Math.ceil(legendItems.length / legendColumns);
    const left = x(100), top = y(Math.max(110, 90 + legendRows * 20)), w = x(1000), bottom = y(535), h = Math.max(y(120), bottom - top);
    const max = Math.max(1, ...selectedRows.flatMap(item => item.rows.map(row => row.count))), step = Math.max(1, Math.ceil(max / 5)), ceiling = step * 5;
    ctx.lineWidth = Math.max(1, scale); ctx.font = `${Math.max(8, 16 * scale)}px sans-serif`;
    for (let i = 0; i <= 5; i++) {
      const lineY = bottom - i * h / 5;
      ctx.strokeStyle = foreground; ctx.globalAlpha = 0.15; ctx.beginPath(); ctx.moveTo(left, lineY); ctx.lineTo(left + w, lineY); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = foreground; ctx.textAlign = 'right'; ctx.fillText(String(i * step), left - x(12), lineY + sy * 5);
      ctx.textAlign = 'left'; ctx.fillText(i * 20 + '%', left + w + x(12), lineY + sy * 5);
    }
    for (const { data, rows } of selectedRows) {
      ctx.fillStyle = data.color; ctx.strokeStyle = data.color; ctx.globalAlpha = 0.28;
      for (const row of rows) {
        const height = row.count / ceiling * h, binPixels = (row.upper - row.lower) * w;
        const gap = Math.min(Math.max(0.5, scale), binPixels / 5);
        ctx.fillRect(left + row.lower * w + gap / 2, bottom - height, Math.max(0.5, binPixels - gap), height);
      }
      ctx.globalAlpha = 1; ctx.lineWidth = Math.max(1, 3 * scale); ctx.beginPath(); ctx.moveTo(left, bottom);
      for (const row of rows) ctx.lineTo(left + row.upper * w, bottom - row.percent / 100 * h);
      ctx.stroke();
    }
    ctx.fillStyle = foreground; ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) ctx.fillText((i / 10).toFixed(1), left + i * w / 10, bottom + y(28));
    ctx.fillText('Assignment score', x(600), y(592));
    ctx.save(); ctx.translate(x(28), top + h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('Sample count', 0, 0); ctx.restore();
    ctx.save(); ctx.translate(x(1180), top + h / 2); ctx.rotate(Math.PI / 2); ctx.fillText('Cumulative percentage (%)', 0, 0); ctx.restore();
    get('scoreTable').textContent = '';
    for (const { data, rows } of selectedRows) for (const row of rows) {
      const tr = document.createElement('tr');
      [data.label, row.lower, row.upper, row.count, row.cumulative, row.percent.toFixed(6)].forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.appendChild(td); });
      get('scoreTable').appendChild(tr);
    }
  }

  section.addEventListener('input', event => {
    const visible = event.target.dataset.scoreVisible, color = event.target.dataset.scoreColor;
    if (visible) datasets.find(data => data.id === visible).enabled = event.target.checked;
    if (color) datasets.find(data => data.id === color).color = event.target.value;
    draw();
  });
  section.addEventListener('click', event => {
    const remove = event.target.closest('[data-score-remove]');
    if (remove) { datasets = datasets.filter(data => data.id !== remove.dataset.scoreRemove); renderSeries(); draw(); }
  });
  get('scoreAdd').onclick = () => { get('scoreAdd').disabled = true; get('scoreStatus').textContent = 'Choose one or more Data Preparation results…'; vscode.postMessage({ type: 'addScoreResults' }); };
  get('scorePng').onclick = () => vscode.postMessage({ type: 'exportScoreDistribution', format: 'png', data: get('scoreCanvas').toDataURL('image/png').split(',')[1] });
  get('scoreTsv').onclick = () => {
    const clean = value => String(value).replace(/[\t\r\n]+/g, ' ');
    const data = 'dataset\tbin_lower_inclusive\tbin_upper\tcount\tcumulative_count\tcumulative_percent\n'
      + selectedRows.flatMap(item => item.rows.map(row => [clean(item.data.label), row.lower, row.upper, row.count, row.cumulative, row.percent].join('\t'))).join('\n') + '\n';
    vscode.postMessage({ type: 'exportScoreDistribution', format: 'tsv', data });
  };
  window.addEventListener('message', event => {
    const message = event.data;
    if (message.type !== 'scoreResultsAdded') return;
    get('scoreAdd').disabled = false;
    const existing = new Set(datasets.map(data => data.id));
    const added = (message.datasets || []).filter(data => !existing.has(data.id));
    datasets.push(...added.map(prepare));
    renderSeries(); draw();
    if (message.error) get('scoreStatus').textContent = message.error;
    else if (!added.length && !message.cancelled) get('scoreStatus').textContent = 'No new assignment score datasets were found.';
  });
  renderSeries(); draw();
}

function distributionHtml(datasets) {
  return `<section id="scoreDistribution"><h2>Assignment Score Distribution</h2>
  <p>Overlay assignment-score charts from multiple Data Preparation results. Use the checkboxes to choose visible datasets.</p>
  <div id="scoreControls" hidden><div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
  <button id="scoreAdd" type="button">Add Result…</button><label>Bin width<input id="scoreWidth" type="number" min="0.001" max="1" step="any" value="0.1"></label>
  ${[['Background', '#ffffff'], ['Text', '#222222']].map(([name, color]) => `<label>${name} color<input id="score${name}" type="color" value="${color}"></label>`).join('')}
  <label>PNG width (px)<input id="scoreImageWidth" type="number" min="320" max="8192" step="1" value="1200"></label><label>PNG height (px)<input id="scoreImageHeight" type="number" min="240" max="8192" step="1" value="640"></label>
  <label><input id="scoreTransparent" type="checkbox" checked> Transparent background</label>
  <button id="scorePng">Save PNG</button><button id="scoreTsv">Save TSV</button></div>
  <div id="scoreSeries" class="score-series"></div><p id="scoreEmpty" class="muted">No score datasets loaded.</p></div>
  <p id="scoreStatus" role="status"></p><canvas id="scoreCanvas" width="1200" height="640" style="width:100%;max-width:1200px;height:auto" aria-label="Overlaid assignment score histograms and cumulative percentages" role="img" hidden></canvas>
  <details><summary>Distribution table</summary><p>Upper bounds are exclusive, except the last bin which includes 1.</p><div style="max-height:360px;overflow:auto"><table><thead><tr><th>Dataset</th><th>Lower (inclusive)</th><th>Upper</th><th>Count</th><th>Cumulative count</th><th>Cumulative (%)</th></tr></thead><tbody id="scoreTable"></tbody></table></div></details></section>
  <style>.score-series{display:grid;gap:6px;margin-top:12px}.score-series-row{display:grid;grid-template-columns:auto auto minmax(180px,1fr) auto;gap:8px;align-items:center;padding:6px 8px;border:1px solid var(--border);border-radius:6px}.score-series-row input[type=color]{width:36px;height:26px;padding:1px}.score-series-row span{overflow-wrap:anywhere}.score-series-row button{padding:4px 8px}</style>
  <script>(${distributionClient.toString()})(${JSON.stringify(datasets).replace(/</g, '\\u003c')}, ${histogram.toString()});</script>`;
}
module.exports = { readScores, histogram, distributionHtml };
