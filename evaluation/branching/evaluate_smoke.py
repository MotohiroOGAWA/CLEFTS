"""Reload the produced checkpoint and compare genuine inference before/after training."""
import json
from pathlib import Path
import torch
from clefts.ml.input.source_action_structure import SourceActionStructure
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from clefts.ml.specgen.predict_spectrum import load_generator
from clefts.ml.training.fragment_tree_training.spectrum_validation import validate_spectra
root=Path('data/train_preprocessing/test/fragment_tree_redesign_verification')
checkpoint=torch.load(root/'training/last.pt',map_location='cpu',weights_only=False)
c=checkpoint['model_config']
torch.manual_seed(checkpoint['training_settings']['seed'])
initial=create_spectrum_generator(c).cuda().eval()
initial.mol_encoder.load_state_dict(torch.load(c['mol_encoder_checkpoint'],map_location='cpu',weights_only=False)['mol_encoder_state_dict'])
trained=load_generator(model_path=str(root/'training/last.pt'),params_path=None,device=torch.device('cuda'),strict=True)
summary={}
for split in ('train','validation'):
    pieces=[SourceActionStructure.load(path) for path in sorted((root/f'prepared/{split}_structures').rglob('*.preft.pt'))]
    for name,model in [('initial',initial),('trained',trained)]:
        summary[f'{split}_{name}']=validate_spectra(model,pieces,root/'comparison',f'{split}_{name}')
(root/'comparison/summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
