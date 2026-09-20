const assert = require('assert/strict');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { JSDOM } = require('jsdom');
const { readTrainingReport, reportCharts, html } = require('../src/features/training-report/view');

(async () => {
  const grouped = reportCharts([
    { name:'metrics/train_loss', points:[[10,3]] },
    { name:'metrics/validation_loss', points:[[10,2]] },
    { name:'distributions/validation/cosine_similarity@main_adduct:[M+H]+_q10', points:[[10,.3]] },
    { name:'distributions/validation/cosine_similarity@main_adduct:[M+H]+_median', points:[[10,.6]] },
    { name:'distributions/validation/cosine_similarity@main_adduct:[M+H]+_q90', points:[[10,.9]] },
  ]);
  assert.deepEqual(grouped.find(chart => chart.metric === 'loss').lanes.map(lane => lane.split), ['train','validation']);
  const facet = grouped.find(chart => chart.dimension === 'main_adduct');
  assert.equal(facet.lanes[0].category, '[M+H]+');
  assert.deepEqual(Object.keys(facet.lanes[0].stats), ['q10','median','q90']);

  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'clefts-training-report-'));
  try {
    await fs.mkdir(path.join(directory, 'spectrum_validation'));
    await fs.writeFile(path.join(directory, 'metrics.tsv'), 'epoch\tglobal_step\ttrain_loss\tvalidation_loss\n1\t10\t3\t2\n');
    await fs.writeFile(path.join(directory, 'metric_distributions.tsv'), 'global_step\tepoch\tsplit\tmetric\tvalue\n10\t1\tvalidation\tcosine_similarity_q10\t0.2\n');
    const spectrum = { main_adduct:'[M+H]+', collision_energy:20, generated_peaks:[{mz:100,intensity:1,precursor:true}], original_peaks:[{mz:100,intensity:1,precursor:true}] };
    await fs.writeFile(path.join(directory, 'spectrum_validation', 'epoch_1.json'), JSON.stringify({ summary:{samples:1}, spectra:[spectrum], representatives:{ cosine_similarity:{q10:{spectrum_index:0,value:1,target:1}} } }));
    const report = { schema:'clefts.training-report', schemaVersion:1, status:'completed', project:'p', run:'r', completedEpochs:1, globalStep:10, latestValidation:'spectrum_validation/epoch_1.json', datasets:{train:{samples:2},validation:{samples:1}}, artifacts:{validationDirectory:'spectrum_validation'} };
    const filename = path.join(directory, 'training.pft.json');
    await fs.writeFile(filename, JSON.stringify(report));
    const loaded = await readTrainingReport(filename);
    assert.equal(loaded.validation.representatives.cosine_similarity.q10.spectrum.main_adduct, '[M+H]+');
    assert.ok(loaded.charts.some(chart => chart.metric === 'loss'));
    const page = html(loaded, '');
    assert.match(page, /Fragment Tree Training Report/);
    assert.match(page, /Representative level/);
    assert.match(page, /#e76f51/);
    assert.match(page, /#3a86ff/);
    for (const match of page.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)) if (!match[0].includes('application/json')) new Function(match[1]);
    const dom = new JSDOM(page, { runScripts:'dangerously', beforeParse(window) { window.acquireVsCodeApi=()=>({getState:()=>({}),setState(){},postMessage(){}}); } });
    assert.equal(dom.window.document.querySelectorAll('#representativeSpectra article').length, 1);
    assert.equal(dom.window.document.querySelector('#representativeSpectra select').value, 'q10');
    dom.window.close();
    console.log('Training .pft.json report parsing and viewer syntax checks passed.');
  } finally { await fs.rm(directory, { recursive:true, force:true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
