"""Thin JSON adapter around CLEFTS dataset and configuration APIs."""
from __future__ import annotations
import contextlib
import json
import math
import sys
from pathlib import Path

DEFAULT_MAPPING={'smilesColumn':'SMILES','adductTypeColumn':'AdductType','collisionEnergyColumn':'CollisionEnergy','precursorMzColumn':'PrecursorMZ'}

def columns(payload: dict) -> dict:
    """Column-mapping validity only: no peak stats, adduct list or row preview.

    Costs only a dataset load (dominated by file I/O, not row/column count),
    so the Column Mapping indicators can update immediately after choosing a
    file, before the slower per-record value checks (preview/data/check) run.
    """
    from clefts.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset
    file=Path(payload['path']).resolve()
    dataset=load_spectrum_dataset(file)
    mapping={**DEFAULT_MAPPING,**payload.get('mapping',{})}
    dataset_columns=dataset.columns
    validation={key:{'column':name,'exists':name in dataset_columns,'valid':0,'total':len(dataset)} for key,name in mapping.items()}
    errors=[]
    for key,item in validation.items():
        if not item['exists']: errors.append('Missing '+key+': '+item['column'])
    return {'path':str(file),'format':file.suffix.lstrip('.').upper(),'size':file.stat().st_size,'records':len(dataset),
            'columns':dataset_columns,'validation':validation,'errors':errors}

def preview(payload: dict) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Descriptors,rdMolDescriptors
    from clefts.libs.mmkit.mmkit import Adduct
    from clefts.domain.mass.parse_ce import parse_ce_to_ev
    from clefts.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset
    file=Path(payload['path']).resolve()
    dataset=load_spectrum_dataset(file)
    mapping={**DEFAULT_MAPPING,**payload.get('mapping',{})}
    columns=dataset.columns
    validation={key:{'column':name,'exists':name in columns,'valid':0,'total':len(dataset)} for key,name in mapping.items()}
    errors=[]
    for key,item in validation.items():
        if not item['exists']: errors.append('Missing '+key+': '+item['column'])
    check_values=bool(payload.get('validateValues',False))
    fragmenter=None
    if check_values:
        from clefts.domain.fragment.fragmenter import Fragmenter
        from clefts.ml.specgen.config_options import PRESET
        params=payload.get('fragmenterParams') or payload.get('modelConfig',{}).get('fragmenter_params') or json.loads(PRESET.read_text())['fragmenter_params']
        fragmenter=Fragmenter.from_dict(params)
    inspection={}
    if check_values and not errors:
        from clefts.ml.data_preparation.fragment_tree.record_validation import inspect_records
        inspection=inspect_records(dataset,fragmenter,mapping)
        validation=inspection['validation']
        if not inspection['eligibleRecords']: errors.append('The dataset contains no valid records to prepare.')
    import numpy as np
    lengths=dataset.peaks.lengths
    eligible=set(inspection.get('validIndexes',range(len(dataset))))
    checked_peaks=dataset[inspection['validIndexes']].peaks if inspection else dataset.peaks
    if np.any(checked_peaks.lengths==0): errors.append(f'{int((checked_peaks.lengths==0).sum())} spectra have no peaks.')
    if not np.isfinite(checked_peaks.mz).all() or np.any(checked_peaks.mz<=0): errors.append('Peak m/z values must be finite and positive.')
    if not np.isfinite(checked_peaks.intensity).all() or np.any(checked_peaks.intensity<0): errors.append('Peak intensities must be finite and non-negative.')
    # MSDataset.metadata rebuilds a full-length view on every dataset[name]
    # access, so each mapped/id column is fetched exactly once here; reading
    # them inside the row loop below would otherwise cost O(limit * columns)
    # full-dataset rebuilds instead of O(columns) for a large dataset.
    mapped_series={key:dataset[name] if name in columns else None for key,name in mapping.items()}
    spec_id_series=dataset['SpecID'] if 'SpecID' in columns else None
    adducts=mapped_series['adductTypeColumn'].astype(str).value_counts().to_dict() if mapped_series['adductTypeColumn'] is not None else {}
    normalized_adducts=[]
    for value in adducts:
        try: normalized_adducts.append(str(Adduct.parse(value)))
        except Exception: pass
    smiles=mapped_series['smilesColumn'].astype(str) if mapped_series['smilesColumn'] is not None else None
    rows=[]
    limit=min(max(int(payload.get('limit',50)),1),100)
    for index in range(min(len(dataset),limit)):
        record=dataset[index]
        values={key:str(series.iloc[index])[:4096] if series is not None else '' for key,series in mapped_series.items()}
        spectrum=list(record.peaks)
        if index in eligible and any(not math.isfinite(float(p.mz)) or float(p.mz)<=0 or not math.isfinite(float(p.intensity)) or float(p.intensity)<0 for p in spectrum):
            errors.append(f'Invalid peak values in preview record {index}')
        rows.append({'index':index,'id':str(spec_id_series.iloc[index]) if spec_id_series is not None else str(index),**values,'numPeaks':int(lengths[index]),
                     'peaks':[{'mz':float(p.mz),'intensity':float(p.intensity)} for p in spectrum[:2000] if math.isfinite(float(p.mz)) and math.isfinite(float(p.intensity))]})
    structure=None
    if rows:
        mol=Chem.MolFromSmiles(rows[0]['smilesColumn'])
        if mol:
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer=rdMolDraw2D.MolDraw2DSVG(320,230);drawer.DrawMolecule(mol);drawer.FinishDrawing()
            structure={'svg':drawer.GetDrawingText(),'formula':rdMolDescriptors.CalcMolFormula(mol),'exactMass':Descriptors.ExactMolWt(mol)}
    if len(dataset)==0: errors.append('The dataset contains no spectra.')
    return {'path':str(file),'format':file.suffix.lstrip('.').upper(),'size':file.stat().st_size,'records':len(dataset),'columns':columns,
            'summary':{'uniqueSmiles':int(smiles.nunique()) if smiles is not None else None,'adducts':adducts,'normalizedAdducts':normalized_adducts,'averagePeaks':float(lengths.mean()) if len(lengths) else 0},
            'valuesChecked':check_values,**{key:value for key,value in inspection.items() if key!='validIndexes'},'validation':validation,'errors':list(dict.fromkeys(errors)),'rows':rows,'structure':structure,
            **({'_smiles':[smiles.iloc[index] for index in inspection.get('validIndexes',range(len(dataset)))]} if payload.get('_identifiers') and smiles is not None else {})}

def validate(payload: dict, *, check_datasets: bool = True) -> dict:
    from clefts.ml.data_preparation.fragment_tree.context import create_preparation_context,validate_limits
    config=payload.get('modelConfig') or {'fragmenter_params':payload['fragmenterParams'],'symbols':payload.get('symbols')}
    generator=create_preparation_context(config)
    validate_limits(payload.get('maxNode',-1),payload.get('maxEdge',-1))
    for key in ('numWorkers','chunkSize'):
        value=payload.get(key,1)
        if type(value) is not int or value<1: raise ValueError('Worker processes and chunk size must be positive integers.')
    minimum=float(payload.get('minimumRelativeIntensity',0))
    if not math.isfinite(minimum) or not 0<=minimum<=1: raise ValueError('Minimum relative intensity must be between 0 and 1.')
    if not payload.get('validationInput'):
        ratio=float(payload.get('validationRatio',0.1))
        if not math.isfinite(ratio) or not 0<ratio<1: raise ValueError('Validation ratio must be between 0 and 1.')
    if check_datasets and payload.get('input'):
        train_mapping={key:payload.get(key,default) for key,default in DEFAULT_MAPPING.items()}
        # Falls back to the training column for anything not explicitly overridden,
        # matching buildArgs()'s --validation-*-column behavior on the CLI side.
        validation_mapping={key:payload.get('validation'+key[0].upper()+key[1:]) or value for key,value in train_mapping.items()}
        sources=[]
        for file,mapping in ((payload['input'],train_mapping),(payload.get('validationInput'),validation_mapping)):
            if not file: continue
            result=preview({'path':file,'mapping':mapping,'limit':1,'_identifiers':True,'validateValues':True,'fragmenterParams':generator.fragmenter.to_dict()})
            sources.append(set(result.pop('_smiles',[])))
            if result['errors']: raise ValueError('; '.join(result['errors']))
            if not payload.get('validationInput') and len(sources[-1])<2:
                raise ValueError('Automatic validation splitting requires at least two unique SMILES.')
        if len(sources)==2 and sources[0] & sources[1]: raise ValueError('Training and validation datasets share SMILES. Choose molecule-disjoint datasets.')
    return {'valid':True,'architecture':generator.architecture}

def model(payload: dict) -> dict:
    import torch
    checkpoint=torch.load(payload['path'],map_location='cpu')
    if checkpoint.get('fragmentation_schema')!='source-anchored-action-autoregressive-v1': raise ValueError('Choose a compatible Source-anchored action training checkpoint.')
    config=checkpoint.get('model_config')
    if not isinstance(config,dict): raise ValueError('Checkpoint does not contain a model configuration.')
    return {'modelConfig':config.get('params',config),'path':payload['path']}

def structure_manifest(payload):
    import torch
    from clefts.ml.data_preparation.fragment_tree.manifest_summary import structure_manifest_fields
    directory=Path(payload['directory'])
    rows=[]
    files=sorted(set(directory.glob('*.preft.pt'))|set((directory/'data').glob('*.preft.pt')))
    for file in files:
        saved=torch.load(file,map_location='cpu',weights_only=False)
        metadata=saved.get('metadata',{})
        indexes=metadata.get('record_indexes',[])
        rows.append(dict(file=str(file.relative_to(directory)),smiles=metadata.get('smiles',''),
            record_indexes=json.dumps(indexes),num_input_records=len(indexes),num_valid_samples=int(saved['structure'].num_samples),rejected_sample_count=max(0,len(indexes)-int(saved['structure'].num_samples)),rejection_log='',**structure_manifest_fields(saved['structure']),status='completed'))
    return rows

def training_checkpoint(payload):
    import torch
    checkpoint = torch.load(payload['path'], map_location='cpu', weights_only=False)
    if checkpoint.get('fragmentation_schema') != 'source-anchored-action-autoregressive-v1':
        raise ValueError('Checkpoint architecture is incompatible with training.')
    config = checkpoint.get('model_config')
    if not config:
        raise ValueError('Checkpoint does not contain model configuration.')
    group = checkpoint['optimizer_state_dict']['param_groups'][0]
    fields={'absolute_weight':'absoluteWeight','next_weight':'nextWeight','negative_weight':'negativeWeight','intensity_weight':'intensityWeight','absolute_intensity_weight':'absoluteIntensityWeight','train_mol_encoder':'trainMolEncoder','validation_interval_steps':'validationIntervalSteps','validation_fraction':'validationFraction'}
    restored={field:checkpoint.get('training_settings',{})[key] for key,field in fields.items() if key in checkpoint.get('training_settings',{})}
    return {'modelConfig': config.get('params', config), 'lr': group['lr'], 'weightDecay': group.get('weight_decay', 0), 'trainingSettings':restored}

def training_sources(payload):
    import torch
    from clefts.ml.training.fragment_tree_training.sources import inherit_model_config, dataset_sources
    saved = None
    path = payload.get('resume') or payload.get('fineTuneCheckpoint')
    if path:
        saved = torch.load(path, map_location='cpu', weights_only=False)['model_config']
        if not payload.get('resume'):
            inherited = dataset_sources(payload['trainDir'], payload['valDir'])
            saved = {**saved.get('params', saved), 'fragmenter_params': inherited['fragmenter_params'],
                     'adduct_type_strs': inherited['adduct_type_strs']}
    config = inherit_model_config({}, payload['trainDir'], payload['valDir'],
        encoder_checkpoint=payload.get('molEncoderCheckpoint'), saved_model=saved)
    return {'modelConfig': config}

def mol_smiles(payload):
    from rdkit import Chem, rdBase
    from rdkit.Chem.Draw import rdMolDraw2D
    count = checked = valid = 0
    symbols, unique, preview = set(), set(), []
    with rdBase.BlockLogs():
        for filename in payload['paths']:
            with open(filename, encoding='utf-8-sig') as stream:
                for line in stream:
                    smiles = line.strip()
                    if not smiles:
                        continue
                    count += 1
                    if checked >= 5000:
                        continue
                    checked += 1
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    valid += 1
                    canonical = Chem.MolToSmiles(mol)
                    unique.add(canonical)
                    symbols.update(atom.GetSymbol() for atom in mol.GetAtoms())
                    if len(preview) < 5:
                        draw = rdMolDraw2D.MolDraw2DSVG(180, 120)
                        rdMolDraw2D.PrepareAndDrawMolecule(draw, mol)
                        draw.FinishDrawing()
                        preview.append({'smiles': canonical, 'svg': draw.GetDrawingText()})
    return {'count': count, 'checked': checked, 'valid': valid, 'unique': len(unique),
            'symbols': sorted(symbols), 'preview': preview}


def mol_preflight(payload):
    import argparse
    from rdkit import Chem
    from clefts.ml.training.mol_training.training_model import build_arg_parser, mol_encoder_configs
    from clefts.domain.molecule.descriptors import compute_descriptor_values
    config = payload['config']
    argv = []
    for action in build_arg_parser()._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        parts = action.dest.split('_')
        key = parts[0] + ''.join(part.title() for part in parts[1:])
        value = config.get(key)
        if value is None or value == '':
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
        elif isinstance(value, list):
            argv.extend([action.option_strings[0], *map(str, value)])
        else:
            argv.extend([action.option_strings[0], str(value)])
    args = build_arg_parser().parse_args(argv)
    combinations = mol_encoder_configs(args)
    table = Chem.GetPeriodicTable()
    supported_symbols = {table.GetElementSymbol(number) for number in range(1, 119)}
    for symbol in config['symbols'].split(','):
        if symbol.strip() not in supported_symbols:
            raise ValueError('Unknown element: ' + symbol)
    if config.get('descriptorNames') and not config.get('disableGraphDescriptors'):
        compute_descriptor_values(Chem.MolFromSmiles('CCO'), tuple(name.strip() for name in config['descriptorNames'].split(',')))
    if config['device'].startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in the selected Python environment')
        torch.empty(0, device=config['device'])
    if config['device'] == 'mps':
        import torch
        if not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable in the selected Python environment')
    for filename in config['trainSmiles'] + config['valSmiles']:
        with open(filename, encoding='utf-8-sig') as stream:
            if not any(line.strip() for line in stream):
                raise ValueError('Empty SMILES file: ' + filename)
    return {'combinations': len(combinations)}

def main():
    request=json.loads(sys.stdin.read())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result={'mol-smiles':mol_smiles,'mol-preflight':mol_preflight,'training-sources':training_sources,'training-checkpoint':training_checkpoint,'structure-manifest':structure_manifest,'columns':columns,'preview':preview,'validate':validate,'validate-config':lambda payload:validate(payload,check_datasets=False),'model':model}[request['command']](request['payload'])
        print(json.dumps({'ok':True,'result':result},allow_nan=False))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'ok':False,'error':str(error)}))
if __name__=='__main__': main()
