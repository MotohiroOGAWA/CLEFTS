const fs = require('fs');
const path = require('path');
const readline = require('readline');

async function readScores(root) {
  const datasets = [];
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
    datasets.push({ label: directory || 'Dataset', scores, skipped, error: column < 0 ? 'assignment_score column is missing.' : '' });
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

function distributionClient(datasets, histogram) {
  const section = document.getElementById('scoreDistribution');
  const get = id => section.querySelector('#' + id);
  if (!datasets.length) { get('scoreStatus').textContent = 'No assignment_scores.tsv found. Generate assignment scores and refresh the result.'; return; }
  get('scoreControls').hidden = false;
  datasets.forEach((data, index) => { const option = document.createElement('option'); option.value = index; option.textContent = data.label; get('scoreDataset').appendChild(option); });
  const canvas = get('scoreCanvas'), ctx = canvas.getContext('2d');
  let rows = [], selected;
  function draw() {
    selected = datasets[Number(get('scoreDataset').value)];
    const width = Number(get('scoreWidth').value);
    try { rows = histogram(selected.scores, width); }
    catch (error) { get('scoreStatus').textContent = error.message; get('scorePng').disabled = get('scoreTsv').disabled = true; return; }
    get('scoreStatus').textContent = selected.error || `${selected.scores.length} samples · ${rows.length} bins · ${selected.skipped} invalid/missing scores excluded. Bins include the lower bound; only the final bin includes its upper bound (1).`;
    get('scorePng').disabled = get('scoreTsv').disabled = !selected.scores.length || !!selected.error;
    canvas.hidden = false;
    const foreground = get('scoreText').value;
    ctx.fillStyle = get('scoreBackground').value; ctx.fillRect(0, 0, 1200, 640);
    ctx.fillStyle = foreground; ctx.font = 'bold 22px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Assignment score distribution — ' + selected.label, 600, 36);
    ctx.font = '16px sans-serif'; ctx.fillText(`n = ${selected.scores.length} · bin width = ${width}`, 600, 63);
    const left = 100, top = 110, w = 1000, h = 420, bottom = top + h;
    const max = Math.max(1, ...rows.map(row => row.count)), step = Math.max(1, Math.ceil(max / 5)), ceiling = step * 5;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = bottom - i * h / 5;
      ctx.strokeStyle = foreground; ctx.globalAlpha = 0.15; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + w, y); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = foreground; ctx.textAlign = 'right'; ctx.fillText(String(i * step), left - 12, y + 5);
      ctx.textAlign = 'left'; ctx.fillText(i * 20 + '%', left + w + 12, y + 5);
    }
    ctx.fillStyle = get('scoreBar').value;
    for (const row of rows) {
      const height = row.count / ceiling * h, binPixels = (row.upper - row.lower) * w;
      ctx.fillRect(left + row.lower * w + Math.min(1, binPixels / 10), bottom - height, binPixels - Math.min(2, binPixels / 5), height);
    }
    ctx.strokeStyle = get('scoreLine').value; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(left, bottom);
    for (const row of rows) ctx.lineTo(left + row.upper * w, bottom - row.percent / 100 * h);
    ctx.stroke(); ctx.fillStyle = foreground; ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) ctx.fillText((i / 10).toFixed(1), left + i * w / 10, bottom + 28);
    ctx.fillText('Assignment score', 600, 592);
    ctx.save(); ctx.translate(28, 320); ctx.rotate(-Math.PI / 2); ctx.fillText('Sample count', 0, 0); ctx.restore();
    ctx.save(); ctx.translate(1180, 320); ctx.rotate(Math.PI / 2); ctx.fillText('Cumulative percentage (%)', 0, 0); ctx.restore();
    ctx.fillStyle = get('scoreBar').value; ctx.fillRect(365, 79, 20, 12); ctx.fillStyle = foreground; ctx.textAlign = 'left'; ctx.fillText('Sample count', 392, 91);
    ctx.fillStyle = get('scoreLine').value; ctx.fillRect(600, 83, 24, 3); ctx.fillStyle = foreground; ctx.fillText('Cumulative percentage', 632, 91);
    get('scoreTable').textContent = '';
    rows.forEach(row => { const tr = document.createElement('tr'); [row.lower, row.upper, row.count, row.cumulative, row.percent.toFixed(6)].forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.appendChild(td); }); get('scoreTable').appendChild(tr); });
  }
  section.addEventListener('input', draw);
  get('scorePng').onclick = () => vscode.postMessage({ type: 'exportScoreDistribution', format: 'png', data: canvas.toDataURL('image/png').split(',')[1] });
  get('scoreTsv').onclick = () => vscode.postMessage({ type: 'exportScoreDistribution', format: 'tsv', data: 'bin_lower_inclusive\tbin_upper\tcount\tcumulative_count\tcumulative_percent\n' + rows.map(row => [row.lower, row.upper, row.count, row.cumulative, row.percent].join('\t')).join('\n') + '\n' });
  draw();
}

function distributionHtml(datasets) {
  return `<section id="scoreDistribution"><h2>Assignment Score Distribution</h2>
  <div id="scoreControls" hidden><div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
  <label>Dataset<select id="scoreDataset"></select></label><label>Bin width<input id="scoreWidth" type="number" min="0.001" max="1" step="any" value="0.1"></label>
  ${[['Bar', '#4285d4'], ['Line', '#e67722'], ['Background', '#ffffff'], ['Text', '#222222']].map(([name, color]) => `<label>${name} color<input id="score${name}" type="color" value="${color}"></label>`).join('')}
  <button id="scorePng">Save PNG</button><button id="scoreTsv">Save TSV</button></div></div>
  <p id="scoreStatus" role="status"></p><canvas id="scoreCanvas" width="1200" height="640" style="width:100%;max-width:1200px;height:auto" aria-label="Assignment score histogram and cumulative percentage" role="img" hidden></canvas>
  <details><summary>Distribution table</summary><p>Upper bounds are exclusive, except the last bin which includes 1.</p><div style="max-height:360px;overflow:auto"><table><thead><tr><th>Lower (inclusive)</th><th>Upper</th><th>Count</th><th>Cumulative count</th><th>Cumulative (%)</th></tr></thead><tbody id="scoreTable"></tbody></table></div></details></section>
  <script>(${distributionClient.toString()})(${JSON.stringify(datasets).replace(/</g, '\\u003c')}, ${histogram.toString()});</script>`;
}
module.exports = { readScores, histogram, distributionHtml };
