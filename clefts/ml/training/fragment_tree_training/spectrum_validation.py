"""Free-running validation against all original peaks, including unassigned peaks."""
import json
from pathlib import Path
import numpy as np
import torch
from clefts.libs.mmkit.mmkit import Compound, Adduct


def matched_cosine(mz, intensity, observed, tolerance):
    """Greedy one-to-one tolerance matching; unmatched peaks retain norm mass."""
    y=np.asarray([p['intensity'] for p in observed],dtype=float)
    x=np.asarray(intensity,dtype=float)
    candidates=sorted(((float(x[i]*y[j]),i,j) for i,m in enumerate(mz) for j,p in enumerate(observed)
                       if tolerance.within(float(p['mz']),float(m))),reverse=True)
    used_x=set();used_y=set();dot=0.
    for value,i,j in candidates:
        if i in used_x or j in used_y:continue
        used_x.add(i);used_y.add(j);dot+=value
    norm=float(np.linalg.norm(x)*np.linalg.norm(y))
    return min(1.,max(0.,dot/norm)) if norm else 0.


@torch.no_grad()
def validate_spectra(generator,pieces,output_dir,label,*,global_step=None):
    was_training=generator.training
    generator.eval();records=[]
    try:
        for piece_index,data in enumerate(pieces):
            if len(data.sample_annotations)!=data.num_samples:
                raise ValueError('Spectrum validation requires original peak annotations; regenerate structures')
            sources=[Compound.from_smiles(data.source_smiles[int(t)]) for t in data.sample_tree_index]
            annotations=data.sample_annotations
            adducts=[Adduct.parse(a['adduct']) for a in annotations]
            energies=[a['collisionEnergy'] for a in annotations]
            for result in generator.predict_batches(sources,adducts,energies):
                for sample,input_index in enumerate(result.sample_input_index):
                    mask=result.downstream.formula_sample_index==sample
                    mz=result.downstream.formula_mz[mask].detach().cpu().tolist()
                    intensity=result.spectra.intensity[mask].detach().cpu().tolist()
                    if not np.isfinite(intensity).all():raise FloatingPointError('Non-finite generated spectrum')
                    observed=annotations[input_index]['peaks']
                    nodes=result.fragments.node_sample_index==sample
                    raw=result.selection.decoded.state_sample_index==sample
                    records.append(dict(piece_index=piece_index,sample_index=input_index,
                        cosine_similarity=matched_cosine(mz,intensity,observed,generator.fragmenter.mass_tolerance),
                        generated_peak_count=sum(v>0 for v in intensity),generated_fragment_nodes=int(raw.sum()),
                        unique_fragment_nodes=int(nodes.sum()),generated_mz=mz,generated_intensity=intensity,
                        original_peaks=[dict(mz=p['mz'],intensity=p['intensity']) for p in observed]))
    finally:
        generator.train(was_training)
    values=np.asarray([r['cosine_similarity'] for r in records])
    if not len(values):raise ValueError('No validation spectra')
    hist,edges=np.histogram(values,bins=np.linspace(0,1,21))
    summary=dict(samples=len(records),cosine_mean=float(values.mean()),cosine_std=float(values.std()),
        cosine_quantiles=dict(zip(('min','p10','p25','median','p75','p90','max'),np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist())),
        histogram_counts=hist.tolist(),histogram_edges=edges.tolist(),
        nonempty_spectrum_fraction=sum(r['generated_peak_count']>0 for r in records)/len(records))
    path=Path(output_dir)/'spectrum_validation';path.mkdir(exist_ok=True,parents=True)
    (path/f'{label}.json').write_text(json.dumps(dict(mode='free-running spectrum generation',summary=summary,spectra=records),indent=2))
    if global_step is not None:
        distribution=Path(output_dir)/'metric_distributions.tsv'
        fresh=not distribution.exists()
        split='validation' if label.startswith('epoch_') else 'intermediate_validation'
        quantiles=summary['cosine_quantiles']
        with distribution.open('a') as stream:
            if fresh:stream.write('global_step\tsplit\tmetric\tvalue\n')
            for name,value in (('min',quantiles['min']),('q1',quantiles['p25']),('median',quantiles['median']),('q3',quantiles['p75']),('max',quantiles['max']),('mean',summary['cosine_mean'])):
                stream.write(f'{global_step}\t{split}\tspectrum_cosine_similarity_{name}\t{value}\n')
    return dict(spectrum_cosine_similarity=summary['cosine_mean'],spectrum_cosine_std=summary['cosine_std'],
        spectrum_nonempty_fraction=summary['nonempty_spectrum_fraction'],
        generated_fragment_nodes=float(np.mean([r['generated_fragment_nodes'] for r in records])),
        unique_fragment_nodes=float(np.mean([r['unique_fragment_nodes'] for r in records])))
