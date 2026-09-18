"""Reproducible CUDA capacity check with persistent curves and prediction metrics.

Uses two synthetic spectra from one source. This is not a generalization benchmark.
"""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',required=True)
parser.add_argument('--epochs',type=int,default=120)
parser.add_argument('--evaluate-only',action='store_true',help='Re-evaluate an existing verification run without training')
args=parser.parse_args()
root=Path(args.output_dir).resolve()
if not args.evaluate_only:root.mkdir(parents=True,exist_ok=False)
app=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(app));os.chdir(app)
from tests.ml.action.TestActionTrainingRevision import fixture
from clefts.ml.training.fragment_tree_training.training import train_actions
from clefts.ml.specgen.predict_spectrum import load_generator
value,generator,source,adducts,data=fixture()
if args.evaluate_only:
    report=json.loads((root/'run'/'training_report.json').read_text())
else:
    encoder=root/'encoder.pt'
    torch.save({'mol_encoder_params':value['mol_encoder_params'],'mol_encoder_state_dict':generator.mol_encoder.state_dict()},encoder)
    value['mol_encoder_checkpoint']=str(encoder)
    for name in ('train','validation'):
        (root/name).mkdir();data.save(root/name/'source.preft.pt')
        (root/name/'action_statistics.json').write_text(json.dumps({'model_config':value}))
    with (root/'training.log').open('w') as log,contextlib.redirect_stdout(log):
        report=train_actions(model_config=value,train_dir=root/'train',val_dir=root/'validation',output_dir=root/'run',epochs=args.epochs,device='cuda',lr=.003,warmup_steps=0,early_stopping_patience=0,max_samples=1)
trained=load_generator(model_path=str(root/'run'/'best.pt'),params_path=None,device=torch.device('cuda'),strict=True)
outputs=list(trained.predict_batches([source,source],adducts,[20.,40.],max_samples=1))
cosines=[]
for output in outputs:
    sample=output.sample_input_index[0]
    observed=data.sample_annotations[sample]['peaks']
    mz=output.spectra.mz.cpu().tolist();intensity=output.spectra.intensity.cpu().tolist()
    truth=[max((peak['intensity'] for peak in observed if trained.fragmenter.mass_tolerance.within(peak['mz'],mass)),default=0.) for mass in mz]
    for peak in observed:
        if not any(trained.fragmenter.mass_tolerance.within(peak['mz'],mass) for mass in mz):
            truth.append(peak['intensity']);intensity.append(0.)
    x=torch.tensor(intensity);y=torch.tensor(truth)
    cosines.append(float(torch.nn.functional.cosine_similarity(x[None],y[None])))
summary={'scope':'Two synthetic spectra of CCCO; same train/validation records; frozen randomly initialized encoder. Capacity check only.',
    'epochs':args.epochs,'device':torch.cuda.get_device_name(0),'first_train_loss':report['history'][0]['train_loss'],'last_train_loss':report['history'][-1]['train_loss'],
    'last_validation_loss':report['history'][-1]['validation_loss'],'stored_graph_cosine':report['history'][-1]['validation/spectrum_cosine_similarity'],
    'filter_intensity_recall':report['history'][-1]['validation/intensity_recall_at_filter'],'free_running_cosines':cosines,'free_running_mean_cosine':sum(cosines)/len(cosines)}
(root/'verification_summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
history=report['history'];epochs=[row['epoch'] for row in history]
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
axes[0].plot(epochs,[row['train_loss'] for row in history],label='Train')
axes[0].plot(epochs,[row['validation_loss'] for row in history],label='Validation (same records)')
axes[0].set(xlabel='Epoch',ylabel='Total loss');axes[0].legend()
axes[1].plot(epochs,[row['validation/spectrum_cosine_similarity'] for row in history],label='Stored graph cosine')
axes[1].plot(epochs,[row['validation/intensity_recall_at_filter'] for row in history],label='Filter intensity recall')
axes[1].set(xlabel='Epoch',ylabel='Score',ylim=(0,1.05));axes[1].legend()
fig.suptitle('CUDA capacity check: one source, two synthetic spectra')
fig.savefig(root/'learning_curves.png',dpi=160);fig.savefig(root/'learning_curves.svg');plt.close(fig)
(root/'verification_report.md').write_text('# CUDA implementation verification\n\n'+summary['scope']+'\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\n![Learning curves](learning_curves.png)\n\nFull epoch metrics, checkpoints and TensorBoard events are in `run/`. Free-running evaluation uses RDKit after GPU action decoding and penalizes both extra and missing peaks. These measurements do not estimate performance on unseen compounds.\n')
print(json.dumps({'output_dir':str(root),**summary},indent=2))
