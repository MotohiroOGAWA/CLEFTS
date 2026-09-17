"""Thin JSON adapter around CLEFTS dataset and configuration APIs."""
from __future__ import annotations
import contextlib
import json
import math
import sys
from pathlib import Path

DEFAULT_MAPPING={'smilesColumn':'SMILES','adductTypeColumn':'AdductType','collisionEnergyColumn':'CollisionEnergy','precursorMzColumn':'PrecursorMZ'}

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
    adducts=dataset[mapping['adductTypeColumn']].astype(str).value_counts().to_dict() if mapping['adductTypeColumn'] in columns else {}
    normalized_adducts=[]
    for value in adducts:
        try: normalized_adducts.append(str(Adduct.parse(value)))
        except Exception: pass
    smiles=dataset[mapping['smilesColumn']].astype(str) if mapping['smilesColumn'] in columns else None
    rows=[]
    limit=min(max(int(payload.get('limit',50)),1),100)
    for index in range(min(len(dataset),limit)):
        record=dataset[index]
        values={key:str(dataset[name].iloc[index])[:4096] if name in columns else '' for key,name in mapping.items()}
        spectrum=list(record.peaks)
        if index in eligible and any(not math.isfinite(float(p.mz)) or float(p.mz)<=0 or not math.isfinite(float(p.intensity)) or float(p.intensity)<0 for p in spectrum):
            errors.append(f'Invalid peak values in preview record {index}')
        rows.append({'index':index,'id':str(dataset['SpecID'].iloc[index]) if 'SpecID' in columns else str(index),**values,'numPeaks':int(lengths[index]),
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
        mapping={key:payload.get(key,default) for key,default in DEFAULT_MAPPING.items()}
        sources=[]
        for file in (payload['input'],payload.get('validationInput')):
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
    directory=Path(payload['directory'])
    rows=[]
    files=sorted(set(directory.glob('*.preft.pt'))|set((directory/'data').glob('*.preft.pt')))
    for file in files:
        saved=torch.load(file,map_location='cpu')
        metadata=saved.get('metadata',{})
        indexes=metadata.get('record_indexes',[])
        rows.append(dict(file=str(file.relative_to(directory)),smiles=metadata.get('smiles',''),
            record_indexes=json.dumps(indexes),num_input_records=len(indexes),num_valid_samples=len(indexes),status='completed'))
    return rows

def main():
    request=json.loads(sys.stdin.read())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result={'structure-manifest':structure_manifest,'preview':preview,'validate':validate,'validate-config':lambda payload:validate(payload,check_datasets=False),'model':model}[request['command']](request['payload'])
        print(json.dumps({'ok':True,'result':result},allow_nan=False))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'ok':False,'error':str(error)}))
if __name__=='__main__': main()
