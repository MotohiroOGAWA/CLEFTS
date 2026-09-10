const fs = require('fs');
const path = require('path');
const readline = require('readline');

const SCORE_CHART_SCHEMA = 'clefts.score-chart';
const SCORE_CHART_SUFFIX = '.scorechart.json';
const SCORE_CHART_DEFAULTS = {
  binCount: 10,
  imageWidth: 1200, imageHeight: 640,
  transparent: true, backgroundColor: '#ffffff', textColor: '#222222',
  title: { show: true, text: 'Assignment score distribution comparison', size: 22, gap: 8 },
  subtitle: { show: true, text: '{datasets} datasets · n = {samples}', size: 16, gap: 12 },
  legend: { show: true, size: 13, gap: 14 },
  xTitle: { show: true, text: 'Assignment score', size: 16, gap: 12 },
  leftTitle: { show: true, text: 'Sample count', size: 16, gap: 12 },
  rightTitle: { show: true, text: 'Cumulative percentage (%)', size: 16, gap: 12 },
  xTicks: { show: true, size: 16, gap: 8 },
  yTicks: { show: true, size: 16, gap: 8 },
  padding: { top: 20, right: 20, bottom: 20, left: 20 }
};

function normalizeScoreChartSettings(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Score chart settings must be a JSON object.');
  if (value.schema && value.schema !== SCORE_CHART_SCHEMA) throw new Error('This is not a score chart settings file.');
  if (value.schemaVersion !== undefined && value.schemaVersion !== 1) throw new Error(`Unsupported score chart settings version: ${value.schemaVersion}`);
  const chart = value.chart && typeof value.chart === 'object' && !Array.isArray(value.chart) ? value.chart : {};
  const series = Array.isArray(value.series) ? value.series.map(item => {
    if (!item || typeof item !== 'object' || !String(item.resultPath || '').trim()) throw new Error('Every saved series requires a resultPath.');
    const opacity = Number(item.opacity ?? 1), barWidth = Number(item.barWidth ?? 100);
    const legacyColor = /^#[0-9a-f]{6}$/i.test(String(item.color || '')) ? String(item.color) : '#4285d4';
    return {
      resultPath: path.resolve(String(item.resultPath)), subset: String(item.subset || 'Dataset'),
      source: String(item.source || ''), label: String(item.label || ''), enabled: item.enabled !== false,
      barColor: /^#[0-9a-f]{6}$/i.test(String(item.barColor || '')) ? String(item.barColor) : legacyColor,
      lineColor: /^#[0-9a-f]{6}$/i.test(String(item.lineColor || '')) ? String(item.lineColor) : legacyColor,
      opacity: Number.isFinite(opacity) ? Math.max(0, Math.min(1, opacity)) : 1,
      barWidth: Number.isFinite(barWidth) ? Math.max(5, Math.min(100, barWidth)) : 100
    };
  }) : [];
  return { schema: SCORE_CHART_SCHEMA, schemaVersion: 1, chart, series };
}

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
      if (first) { column = line.replace(/^\uFEFF/, '').split('\t').indexOf('assignment_score'); first = false; continue; }
      if (!line.trim()) continue;
      const value = (line.split('\t')[column] || '').trim(), score = Number(value);
      if (!value || !Number.isFinite(score) || score < 0 || score > 1) skipped++;
      else scores.push(score);
    }
    const subset = directory || 'Dataset';
    datasets.push({
      id: `${path.resolve(resultPath)}::${subset}`, source, subset,
      resultPath: path.resolve(resultPath), label: `${source} — ${subset}`,
      scores, skipped, error: column < 0 ? 'assignment_score column is missing.' : ''
    });
  }
  return datasets;
}

function histogram(scores, width) {
  if (!Number.isFinite(width) || width < 0.001 || width > 1) throw new Error('Bin width must be between 0.001 and 1.');
  const clean = value => Number(value.toFixed(12));
  const rows = Array.from({ length: Math.ceil(clean(1 / width)) }, (_, i) => ({ lower: clean(i * width), upper: Math.min(1, clean((i + 1) * width)), count: 0 }));
  for (const score of scores) {
    let low = 0, high = rows.length - 1;
    while (low < high) { const middle = Math.floor((low + high) / 2); if (score < rows[middle].upper) high = middle; else low = middle + 1; }
    rows[low].count++;
  }
  let cumulative = 0;
  return rows.map(row => ({ ...row, cumulative: cumulative += row.count, percent: scores.length ? cumulative / scores.length * 100 : 0 }));
}

function distributionClient(initialDatasets, histogram, defaults) {
  const section = document.getElementById('scoreDistribution'), get = id => section.querySelector('#' + id);
  const palette = ['#4285d4', '#e67722', '#2f9e44', '#9c36b5', '#e03131', '#1098ad', '#f08c00', '#5f3dc4'];
  let colorIndex = 0, datasets = [], selectedRows = [], settingsPath = '';
  const number = (id, min, max, fallback) => { const value = Number(get(id).value); return Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback; };
  const prepare = data => {
    const legacyColor = data.color || palette[colorIndex++ % palette.length];
    return {
      ...data, enabled: data.enabled !== false,
      barColor: data.barColor || legacyColor, lineColor: data.lineColor || legacyColor,
      opacity: Number.isFinite(Number(data.opacity)) ? Math.max(0, Math.min(1, Number(data.opacity))) : 1,
      barWidth: Number.isFinite(Number(data.barWidth)) ? Math.max(5, Math.min(100, Number(data.barWidth))) : 100
    };
  };
  datasets = initialDatasets.map(prepare);
  get('scoreControls').hidden = false;

  const labelControls = {
    title: ['scoreTitleShow', 'scoreTitleText', 'scoreTitleSize', 'scoreTitleGap'],
    subtitle: ['scoreSubtitleShow', 'scoreSubtitleText', 'scoreSubtitleSize', 'scoreSubtitleGap'],
    xTitle: ['scoreXTitleShow', 'scoreXTitleText', 'scoreXTitleSize', 'scoreXTitleGap'],
    leftTitle: ['scoreLeftTitleShow', 'scoreLeftTitleText', 'scoreLeftTitleSize', 'scoreLeftTitleGap'],
    rightTitle: ['scoreRightTitleShow', 'scoreRightTitleText', 'scoreRightTitleSize', 'scoreRightTitleGap']
  };
  function chartSettings() {
    const binCount = Math.round(number('scoreBinCount', 1, 1000, defaults.binCount));
    const chart = {
      binCount, binWidth: 1 / binCount,
      imageWidth: number('scoreImageWidth', 320, 8192, defaults.imageWidth),
      imageHeight: number('scoreImageHeight', 240, 8192, defaults.imageHeight),
      transparent: get('scoreTransparent').checked,
      backgroundColor: get('scoreBackground').value, textColor: get('scoreText').value,
      legend: { show: get('scoreLegendShow').checked, size: number('scoreLegendSize', 6, 96, 13), gap: number('scoreLegendGap', 0, 200, 14) },
      xTicks: { show: get('scoreXTicksShow').checked, size: number('scoreXTicksSize', 6, 96, 16), gap: number('scoreXTicksGap', 0, 200, 8) },
      yTicks: { show: get('scoreYTicksShow').checked, size: number('scoreYTicksSize', 6, 96, 16), gap: number('scoreYTicksGap', 0, 200, 8) },
      padding: {
        top: number('scorePaddingTop', 0, 500, 20), right: number('scorePaddingRight', 0, 500, 20),
        bottom: number('scorePaddingBottom', 0, 500, 20), left: number('scorePaddingLeft', 0, 500, 20)
      }
    };
    for (const [key, ids] of Object.entries(labelControls)) chart[key] = {
      show: get(ids[0]).checked, text: get(ids[1]).value,
      size: number(ids[2], 6, 96, defaults[key].size), gap: number(ids[3], 0, 200, defaults[key].gap)
    };
    return chart;
  }
  function applyChart(chart = {}) {
    const merged = { ...defaults, ...chart, padding: { ...defaults.padding, ...(chart.padding || {}) } };
    for (const [key, ids] of Object.entries(labelControls)) {
      const value = { ...defaults[key], ...(chart[key] || {}) };
      get(ids[0]).checked = value.show !== false; get(ids[1]).value = value.text; get(ids[2]).value = value.size; get(ids[3]).value = value.gap;
    }
    for (const [key, prefix] of [['legend', 'scoreLegend'], ['xTicks', 'scoreXTicks'], ['yTicks', 'scoreYTicks']]) {
      const value = { ...defaults[key], ...(chart[key] || {}) };
      get(prefix + 'Show').checked = value.show !== false; get(prefix + 'Size').value = value.size; get(prefix + 'Gap').value = value.gap;
    }
    get('scoreImageWidth').value = merged.imageWidth; get('scoreImageHeight').value = merged.imageHeight;
    get('scoreBinCount').value = Number.isFinite(Number(merged.binCount)) ? Math.max(1, Math.min(1000, Math.round(Number(merged.binCount)))) : Math.max(1, Math.round(1 / Number(merged.binWidth || 0.1)));
    get('scoreTransparent').checked = merged.transparent !== false; get('scoreBackground').value = merged.backgroundColor; get('scoreText').value = merged.textColor;
    for (const side of ['Top', 'Right', 'Bottom', 'Left']) get('scorePadding' + side).value = merged.padding[side.toLowerCase()];
    get('scoreBackground').disabled = get('scoreTransparent').checked;
  }
  function savedSettings() {
    return {
      schema: 'clefts.score-chart', schemaVersion: 1, chart: chartSettings(),
      series: datasets.map(data => ({
        resultPath: data.resultPath, subset: data.subset, source: data.source, label: data.label,
        enabled: data.enabled, barColor: data.barColor, lineColor: data.lineColor,
        opacity: data.opacity, barWidth: data.barWidth
      }))
    };
  }
  function renderSeries() {
    const list = get('scoreSeries'); list.textContent = '';
    for (const data of datasets) {
      const row = document.createElement('div'); row.className = 'score-series-row'; row.dataset.seriesId = data.id;
      const visible = document.createElement('input'); visible.type = 'checkbox'; visible.checked = data.enabled; visible.dataset.scoreField = 'enabled'; visible.setAttribute('aria-label', 'Show ' + data.label);
      const label = document.createElement('input'); label.type = 'text'; label.value = data.label; label.dataset.scoreField = 'label'; label.title = data.resultPath || data.label; label.setAttribute('aria-label', 'Legend label');
      const barColor = document.createElement('label'); barColor.textContent = 'Bar color'; const barColorInput = document.createElement('input'); barColorInput.type = 'color'; barColorInput.value = data.barColor; barColorInput.dataset.scoreField = 'barColor'; barColor.appendChild(barColorInput);
      const lineColor = document.createElement('label'); lineColor.textContent = 'Line color'; const lineColorInput = document.createElement('input'); lineColorInput.type = 'color'; lineColorInput.value = data.lineColor; lineColorInput.dataset.scoreField = 'lineColor'; lineColor.appendChild(lineColorInput);
      const opacity = document.createElement('label'); opacity.textContent = 'Opacity'; const opacityInput = document.createElement('input'); opacityInput.type = 'number'; opacityInput.min = '0'; opacityInput.max = '1'; opacityInput.step = '0.05'; opacityInput.value = data.opacity; opacityInput.dataset.scoreField = 'opacity'; opacity.appendChild(opacityInput);
      const bar = document.createElement('label'); bar.textContent = 'Bar width (%)'; const barInput = document.createElement('input'); barInput.type = 'number'; barInput.min = '5'; barInput.max = '100'; barInput.step = '1'; barInput.value = data.barWidth; barInput.dataset.scoreField = 'barWidth'; bar.appendChild(barInput);
      const count = document.createElement('small'); count.textContent = `${data.scores.length} scores`; count.className = 'muted';
      const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove'; remove.dataset.scoreRemove = 'true';
      row.append(visible, label, barColor, lineColor, opacity, bar, count, remove); list.appendChild(row);
    }
    get('scoreEmpty').hidden = Boolean(datasets.length);
  }
  function fitText(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let value = text; while (value.length && ctx.measureText(value + '…').width > maxWidth) value = value.slice(0, -1);
    return value + '…';
  }
  function template(text, values) { return String(text).replace(/\{(datasets|samples|bins|binWidth)\}/g, (_, key) => values[key]); }

  function draw() {
    const chart = chartSettings(), visible = datasets.filter(data => data.enabled && !data.error && data.scores.length);
    try { selectedRows = visible.map(data => ({ data, rows: histogram(data.scores, chart.binWidth) })); }
    catch (error) { get('scoreStatus').textContent = error.message; get('scorePng').disabled = get('scoreTsv').disabled = true; return; }
    const problems = datasets.filter(data => data.enabled && (data.error || !data.scores.length)).map(data => `${data.label}: ${data.error || 'no valid scores'}`);
    const total = selectedRows.reduce((sum, item) => sum + item.data.scores.length, 0), skipped = selectedRows.reduce((sum, item) => sum + item.data.skipped, 0);
    get('scorePng').disabled = get('scoreTsv').disabled = !selectedRows.length;
    const canvas = get('scoreCanvas'), ctx = canvas.getContext('2d');
    if (!selectedRows.length) {
      canvas.hidden = true; get('scoreTable').textContent = '';
      get('scoreStatus').textContent = datasets.length ? `Select at least one dataset with valid scores.${problems.length ? ' ' + problems.join('; ') : ''}` : 'No assignment_scores.tsv found. Add a Data Preparation result to compare.';
      return;
    }
    const values = { datasets: selectedRows.length, samples: total, bins: chart.binCount, binWidth: chart.binWidth };
    const shown = key => chart[key].show && String(chart[key].text).trim();
    const max = Math.max(1, ...selectedRows.flatMap(item => item.rows.map(row => row.count)));
    const step = Math.max(1, Math.ceil(max / 5)), ceiling = step * 5;
    canvas.width = chart.imageWidth; canvas.height = chart.imageHeight;
    const font = (size, bold = false) => `${bold ? 'bold ' : ''}${size}px sans-serif`;
    const measure = (text, size, bold = false) => { ctx.font = font(size, bold); return ctx.measureText(text).width; };
    const titleText = template(chart.title.text, values), subtitleText = template(chart.subtitle.text, values);
    const leftZone = chart.padding.left + (shown('leftTitle') ? chart.leftTitle.size + chart.leftTitle.gap : 0) + (chart.yTicks.show ? measure(String(ceiling), chart.yTicks.size) + chart.yTicks.gap : 0);
    const rightZone = chart.padding.right + (shown('rightTitle') ? chart.rightTitle.size + chart.rightTitle.gap : 0) + (chart.yTicks.show ? measure('100%', chart.yTicks.size) + chart.yTicks.gap : 0);
    const widestHeading = Math.max(shown('title') ? measure(titleText, chart.title.size, true) : 0, shown('subtitle') ? measure(subtitleText, chart.subtitle.size) : 0, shown('xTitle') ? measure(chart.xTitle.text, chart.xTitle.size) : 0);
    const widestLegend = chart.legend.show ? Math.max(0, ...selectedRows.map(item => measure(item.data.label, chart.legend.size) + 34)) : 0;
    const minimumPlotHeight = Math.max(180, shown('leftTitle') ? measure(chart.leftTitle.text, chart.leftTitle.size) : 0, shown('rightTitle') ? measure(chart.rightTitle.text, chart.rightTitle.size) : 0);
    let actualWidth = Math.min(8192, Math.max(chart.imageWidth, leftZone + rightZone + 320, chart.padding.left + chart.padding.right + widestHeading, chart.padding.left + chart.padding.right + widestLegend));
    canvas.width = actualWidth;
    const legendRows = [];
    if (chart.legend.show) {
      let row = [], used = 0; const available = actualWidth - chart.padding.left - chart.padding.right;
      for (const item of selectedRows) {
        const itemWidth = Math.min(available, measure(item.data.label, chart.legend.size) + 34);
        if (row.length && used + itemWidth > available) { legendRows.push(row); row = []; used = 0; }
        row.push({ item, x: used, width: itemWidth }); used += itemWidth;
      }
      if (row.length) legendRows.push(row);
    }
    let topZone = chart.padding.top;
    if (shown('title')) topZone += chart.title.size + chart.title.gap;
    if (shown('subtitle')) topZone += chart.subtitle.size + chart.subtitle.gap;
    if (chart.legend.show && legendRows.length) topZone += legendRows.length * (chart.legend.size + 8) + chart.legend.gap;
    const bottomZone = chart.padding.bottom + (chart.xTicks.show ? chart.xTicks.size + chart.xTicks.gap : 0) + (shown('xTitle') ? chart.xTitle.size + chart.xTitle.gap : 0);
    const actualHeight = Math.min(8192, Math.max(chart.imageHeight, topZone + bottomZone + minimumPlotHeight));
    canvas.height = actualHeight; canvas.hidden = false;
    ctx.clearRect(0, 0, actualWidth, actualHeight);
    if (!chart.transparent) { ctx.fillStyle = chart.backgroundColor; ctx.fillRect(0, 0, actualWidth, actualHeight); }
    ctx.fillStyle = chart.textColor; ctx.textAlign = 'center';
    let cursor = chart.padding.top;
    if (shown('title')) { ctx.font = font(chart.title.size, true); cursor += chart.title.size; ctx.fillText(fitText(ctx, titleText, actualWidth - chart.padding.left - chart.padding.right), actualWidth / 2, cursor); cursor += chart.title.gap; }
    if (shown('subtitle')) { ctx.font = font(chart.subtitle.size); cursor += chart.subtitle.size; ctx.fillText(fitText(ctx, subtitleText, actualWidth - chart.padding.left - chart.padding.right), actualWidth / 2, cursor); cursor += chart.subtitle.gap; }
    if (chart.legend.show && legendRows.length) {
      ctx.font = font(chart.legend.size); ctx.textAlign = 'left';
      for (const row of legendRows) {
        cursor += chart.legend.size;
        for (const entry of row) {
          const lx = chart.padding.left + entry.x; ctx.fillStyle = entry.item.data.barColor; ctx.globalAlpha = entry.item.data.opacity; ctx.fillRect(lx, cursor - chart.legend.size * .75, 18, Math.max(4, chart.legend.size * .65));
          ctx.strokeStyle = entry.item.data.lineColor; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(lx, cursor - chart.legend.size * .35); ctx.lineTo(lx + 18, cursor - chart.legend.size * .35); ctx.stroke();
          ctx.globalAlpha = 1; ctx.fillStyle = chart.textColor; ctx.fillText(fitText(ctx, entry.item.data.label, entry.width - 30), lx + 25, cursor);
        }
        cursor += 8;
      }
      cursor += chart.legend.gap;
    }
    const plotTop = cursor, plotBottom = actualHeight - bottomZone, plotLeft = leftZone, plotRight = actualWidth - rightZone;
    const plotWidth = Math.max(1, plotRight - plotLeft), plotHeight = Math.max(1, plotBottom - plotTop);
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const lineY = plotBottom - i * plotHeight / 5;
      ctx.strokeStyle = chart.textColor; ctx.globalAlpha = 0.15; ctx.beginPath(); ctx.moveTo(plotLeft, lineY); ctx.lineTo(plotRight, lineY); ctx.stroke(); ctx.globalAlpha = 1;
      if (chart.yTicks.show) {
        ctx.font = font(chart.yTicks.size); ctx.fillStyle = chart.textColor; ctx.textAlign = 'right'; ctx.fillText(String(i * step), plotLeft - chart.yTicks.gap, lineY + chart.yTicks.size * .35);
        ctx.textAlign = 'left'; ctx.fillText(i * 20 + '%', plotRight + chart.yTicks.gap, lineY + chart.yTicks.size * .35);
      }
    }
    ctx.strokeStyle = chart.textColor; ctx.globalAlpha = 1; ctx.beginPath(); ctx.moveTo(plotLeft, plotTop); ctx.lineTo(plotLeft, plotBottom); ctx.lineTo(plotRight, plotBottom); ctx.stroke();
    for (const { data, rows } of selectedRows) {
      ctx.fillStyle = data.barColor; ctx.globalAlpha = data.opacity;
      for (const row of rows) {
        const binPixels = (row.upper - row.lower) * plotWidth, visualWidth = Math.max(0.5, binPixels * data.barWidth / 100), center = plotLeft + (row.lower + row.upper) / 2 * plotWidth;
        ctx.fillRect(center - visualWidth / 2, plotBottom - row.count / ceiling * plotHeight, visualWidth, row.count / ceiling * plotHeight);
      }
      ctx.strokeStyle = data.lineColor; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(plotLeft, plotBottom);
      for (const row of rows) ctx.lineTo(plotLeft + row.upper * plotWidth, plotBottom - row.percent / 100 * plotHeight);
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    if (chart.xTicks.show) {
      ctx.font = font(chart.xTicks.size); ctx.fillStyle = chart.textColor; ctx.textAlign = 'center';
      for (let i = 0; i <= 10; i++) ctx.fillText((i / 10).toFixed(1), plotLeft + i * plotWidth / 10, plotBottom + chart.xTicks.gap + chart.xTicks.size);
    }
    if (shown('xTitle')) {
      ctx.font = font(chart.xTitle.size); ctx.fillStyle = chart.textColor; ctx.textAlign = 'center';
      const tickBlock = chart.xTicks.show ? chart.xTicks.gap + chart.xTicks.size : 0;
      ctx.fillText(fitText(ctx, chart.xTitle.text, plotWidth), plotLeft + plotWidth / 2, plotBottom + tickBlock + chart.xTitle.gap + chart.xTitle.size);
    }
    for (const [key, side] of [['leftTitle', 'left'], ['rightTitle', 'right']]) if (shown(key)) {
      ctx.save(); ctx.font = font(chart[key].size); ctx.fillStyle = chart.textColor; ctx.textAlign = 'center';
      const axisX = side === 'left' ? chart.padding.left + chart[key].size : actualWidth - chart.padding.right - chart[key].size;
      ctx.translate(axisX, plotTop + plotHeight / 2); ctx.rotate(side === 'left' ? -Math.PI / 2 : Math.PI / 2);
      ctx.fillText(fitText(ctx, chart[key].text, plotHeight), 0, 0); ctx.restore();
    }
    get('scoreStatus').textContent = `${selectedRows.length} datasets overlaid · ${total} samples · ${chart.binCount} bins · ${skipped} invalid/missing excluded · ${actualWidth} × ${actualHeight}px${actualWidth !== chart.imageWidth || actualHeight !== chart.imageHeight ? ' (auto-expanded to protect labels)' : ''}.${problems.length ? ' Excluded: ' + problems.join('; ') : ''}`;
    get('scoreTable').textContent = '';
    for (const { data, rows } of selectedRows) for (const row of rows) {
      const tr = document.createElement('tr'); [data.label, row.lower, row.upper, row.count, row.cumulative, row.percent.toFixed(6)].forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.appendChild(td); }); get('scoreTable').appendChild(tr);
    }
  }

  section.addEventListener('input', event => {
    const row = event.target.closest('[data-series-id]'), field = event.target.dataset.scoreField;
    if (row && field) {
      const data = datasets.find(item => item.id === row.dataset.seriesId);
      if (field === 'enabled') data.enabled = event.target.checked;
      else if (field === 'opacity') data.opacity = Math.max(0, Math.min(1, Number(event.target.value)));
      else if (field === 'barWidth') data.barWidth = Math.max(5, Math.min(100, Number(event.target.value)));
      else data[field] = event.target.value;
    }
    get('scoreBackground').disabled = get('scoreTransparent').checked; draw();
  });
  section.addEventListener('click', event => { const button = event.target.closest('[data-score-remove]'); if (!button) return; const row = button.closest('[data-series-id]'); datasets = datasets.filter(data => data.id !== row.dataset.seriesId); renderSeries(); draw(); });
  get('scoreAdd').onclick = () => { get('scoreAdd').disabled = true; get('scoreStatus').textContent = 'Choose one or more Data Preparation results…'; vscode.postMessage({ type: 'addScoreResults' }); };
  get('scoreSaveSettings').onclick = () => {
    get('scoreSaveSettings').disabled = true; get('scoreActionStatus').textContent = 'Choose where to save the chart settings…';
    vscode.postMessage({ type: 'saveScoreDistributionSettings', config: savedSettings(), path: settingsPath });
  };
  get('scoreLoadSettings').onclick = () => { get('scoreLoadSettings').disabled = true; vscode.postMessage({ type: 'loadScoreDistributionSettings' }); };
  get('scorePng').onclick = () => vscode.postMessage({ type: 'exportScoreDistribution', format: 'png', data: get('scoreCanvas').toDataURL('image/png').split(',')[1] });
  get('scoreTsv').onclick = () => { const clean = value => String(value).replace(/[\t\r\n]+/g, ' '), data = 'dataset\tbin_lower_inclusive\tbin_upper\tcount\tcumulative_count\tcumulative_percent\n' + selectedRows.flatMap(item => item.rows.map(row => [clean(item.data.label), row.lower, row.upper, row.count, row.cumulative, row.percent].join('\t'))).join('\n') + '\n'; vscode.postMessage({ type: 'exportScoreDistribution', format: 'tsv', data }); };
  window.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'scoreResultsAdded') {
      get('scoreAdd').disabled = false; const existing = new Set(datasets.map(data => data.id)), added = (message.datasets || []).filter(data => !existing.has(data.id)); datasets.push(...added.map(prepare)); renderSeries(); draw();
      if (message.error) get('scoreStatus').textContent = message.error; else if (!added.length && !message.cancelled) get('scoreStatus').textContent = 'No new assignment score datasets were found.';
    } else if (message.type === 'scoreSettingsSaved') {
      get('scoreSaveSettings').disabled = false;
      if (message.error) get('scoreActionStatus').textContent = message.error;
      else if (message.cancelled) get('scoreActionStatus').textContent = 'Save cancelled.';
      else { settingsPath = message.path; get('scoreActionStatus').textContent = 'Saved: ' + message.path; }
    } else if (message.type === 'scoreSettingsLoaded') {
      get('scoreLoadSettings').disabled = false;
      if (message.failed) get('scoreStatus').textContent = message.error;
      else if (!message.cancelled) {
        const byKey = new Map((message.config.series || []).map(item => [`${item.resultPath}::${item.subset}`, item]));
        datasets = (message.datasets || []).map(raw => prepare({ ...raw, ...(byKey.get(`${raw.resultPath}::${raw.subset}`) || {}) }));
        applyChart(message.config.chart); settingsPath = message.path || ''; renderSeries(); draw();
        if (message.error) get('scoreStatus').textContent = message.error;
      }
    }
  });
  applyChart(defaults); renderSeries(); draw();
}

function labelRow(prefix, name, text, size, gap) {
  return `<div class="score-label-row"><label class="score-check"><input id="score${prefix}Show" type="checkbox" checked>Show ${name}</label><label>${name}<input id="score${prefix}Text" value="${text}"></label><label>Size<input id="score${prefix}Size" type="number" min="6" max="96" value="${size}"></label><label>Gap<input id="score${prefix}Gap" type="number" min="0" max="200" value="${gap}"></label></div>`;
}

function distributionHtml(datasets) {
  return `<section id="scoreDistribution"><h2>Assignment Score Distribution</h2><p>Overlay results and customize each series, label, and spacing.</p>
  <div id="scoreControls" hidden><div class="score-actions"><button id="scoreAdd" type="button">Add Result…</button><button id="scoreLoadSettings" type="button">Load Settings…</button><button id="scoreSaveSettings" type="button">Save Settings…</button><button id="scorePng" type="button">Save PNG</button><button id="scoreTsv" type="button">Save TSV</button><span id="scoreActionStatus" class="muted" role="status"></span></div>
  <h3>Series</h3><div id="scoreSeries" class="score-series"></div><p id="scoreEmpty" class="muted">No score datasets loaded.</p>
  <details open><summary>Chart and labels</summary><div class="score-basic"><label>Number of bins<input id="scoreBinCount" type="number" min="1" max="1000" step="1" value="10"></label><label>Minimum PNG width<input id="scoreImageWidth" type="number" min="320" max="8192" value="1200"></label><label>Minimum PNG height<input id="scoreImageHeight" type="number" min="240" max="8192" value="640"></label><label>Text and axis color<input id="scoreText" type="color" value="#222222"></label><label class="score-check"><input id="scoreTransparent" type="checkbox" checked>Transparent background</label><label>Background color<input id="scoreBackground" type="color" value="#ffffff" disabled></label></div>
  ${labelRow('Title', 'title', 'Assignment score distribution comparison', 22, 8)}${labelRow('Subtitle', 'subtitle', '{datasets} datasets · n = {samples}', 16, 12)}${labelRow('XTitle', 'X-axis title', 'Assignment score', 16, 12)}${labelRow('LeftTitle', 'left Y-axis title', 'Sample count', 16, 12)}${labelRow('RightTitle', 'right Y-axis title', 'Cumulative percentage (%)', 16, 12)}
  <div class="score-label-row"><label class="score-check"><input id="scoreLegendShow" type="checkbox" checked>Show legend</label><span></span><label>Size<input id="scoreLegendSize" type="number" min="6" max="96" value="13"></label><label>Gap<input id="scoreLegendGap" type="number" min="0" max="200" value="14"></label></div>
  <div class="score-label-row"><label class="score-check"><input id="scoreXTicksShow" type="checkbox" checked>Show X ticks</label><span></span><label>Size<input id="scoreXTicksSize" type="number" min="6" max="96" value="16"></label><label>Gap<input id="scoreXTicksGap" type="number" min="0" max="200" value="8"></label></div>
  <div class="score-label-row"><label class="score-check"><input id="scoreYTicksShow" type="checkbox" checked>Show Y ticks</label><span></span><label>Size<input id="scoreYTicksSize" type="number" min="6" max="96" value="16"></label><label>Gap<input id="scoreYTicksGap" type="number" min="0" max="200" value="8"></label></div>
  <h4>Minimum outer spacing</h4><div class="score-basic">${['Top', 'Right', 'Bottom', 'Left'].map(side => `<label>${side}<input id="scorePadding${side}" type="number" min="0" max="500" value="20"></label>`).join('')}</div><p class="muted">The canvas grows automatically when these minimums are too small for visible labels. Subtitle placeholders: {datasets}, {samples}, {bins}, {binWidth}.</p></details></div>
  <p id="scoreStatus" role="status"></p><canvas id="scoreCanvas" width="1200" height="640" style="width:100%;max-width:1200px;height:auto" aria-label="Overlaid assignment score distributions" role="img" hidden></canvas>
  <details><summary>Distribution table</summary><div style="max-height:360px;overflow:auto"><table><thead><tr><th>Dataset</th><th>Lower</th><th>Upper</th><th>Count</th><th>Cumulative</th><th>Cumulative (%)</th></tr></thead><tbody id="scoreTable"></tbody></table></div></details></section>
  <style>.score-actions,.score-basic{display:flex;gap:12px;flex-wrap:wrap;align-items:end}.score-basic>label{min-width:145px;flex:1}.score-series{display:grid;gap:7px}.score-series-row{display:grid;grid-template-columns:auto minmax(180px,2fr) minmax(80px,.5fr) minmax(80px,.5fr) minmax(90px,.5fr) minmax(105px,.6fr) auto auto;gap:8px;align-items:end;padding:8px;border:1px solid var(--border);border-radius:6px}.score-series-row>input{margin:0}.score-series-row input[type=color]{height:34px;padding:3px}.score-series-row label{margin:0}.score-label-row{display:grid;grid-template-columns:minmax(130px,.7fr) minmax(220px,2fr) minmax(80px,.5fr) minmax(80px,.5fr);gap:10px;align-items:end;border-top:1px solid var(--border);padding:8px 0}.score-label-row label{margin:0}.score-check{display:flex!important;align-items:center;gap:7px}.score-check input{width:auto;margin:0}@media(max-width:900px){.score-series-row,.score-label-row{grid-template-columns:1fr 1fr}.score-series-row>input[type=text]{grid-column:1/-1}}</style>
  <script>(${distributionClient.toString()})(${JSON.stringify(datasets).replace(/</g, '\\u003c')}, ${histogram.toString()}, ${JSON.stringify(SCORE_CHART_DEFAULTS)});</script>`;
}

module.exports = { SCORE_CHART_SCHEMA, SCORE_CHART_SUFFIX, SCORE_CHART_DEFAULTS, normalizeScoreChartSettings, readScores, histogram, distributionHtml };
