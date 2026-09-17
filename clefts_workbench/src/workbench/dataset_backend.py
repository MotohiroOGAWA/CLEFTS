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
    cache={}
    def valid_smiles(value):
        text=str(value)
        if text not in cache: cache[text]=bool(text.strip()) and Chem.MolFromSmiles(text) is not None
        return cache[text]
    for key,item in validation.items():
        if not item['exists'] or not check_values: continue
        values=dataset[item['column']].tolist()
        for index,value in enumerate(values):
            try:
                if key=='smilesColumn': valid=valid_smiles(value)
                elif key=='adductTypeColumn': fragmenter.get_index_by_adduct_type(Adduct.parse(str(value)));valid=True
                elif key=='precursorMzColumn': valid=math.isfinite(float(value)) and float(value)>0
                else:
                    mz_name=mapping['precursorMzColumn']
                    mz=float(dataset[mz_name].iloc[index]) if mz_name in columns else 0.0
                    energy=parse_ce_to_ev(value,mz)
                    valid=energy is not None and math.isfinite(float(energy)) and float(energy)>=0
                item['valid']+=int(valid)
            except Exception: pass
        if item['valid']!=len(dataset):
            reason='invalid adducts or main adducts not registered in the fragmenter' if key=='adductTypeColumn' else 'values that cannot be parsed to finite non-negative eV' if key=='collisionEnergyColumn' else 'values that cannot be parsed to finite positive numbers' if key=='precursorMzColumn' else 'invalid RDKit SMILES'
            errors.append(f"{item['column']}: {len(dataset)-item['valid']} {reason}")
    import numpy as np
    lengths=dataset.peaks.lengths
    if np.any(lengths==0): errors.append(f'{int((lengths==0).sum())} spectra have no peaks.')
    if not np.isfinite(dataset.peaks.mz).all() or np.any(dataset.peaks.mz<=0): errors.append('Peak m/z values must be finite and positive.')
    if not np.isfinite(dataset.peaks.intensity).all() or np.any(dataset.peaks.intensity<0): errors.append('Peak intensities must be finite and non-negative.')
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
        if any(not math.isfinite(float(p.mz)) or float(p.mz)<=0 or not math.isfinite(float(p.intensity)) or float(p.intensity)<0 for p in spectrum):
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
            'valuesChecked':check_values,'validation':validation,'errors':list(dict.fromkeys(errors)),'rows':rows,'structure':structure,
            **({'_smiles':smiles.tolist()} if payload.get('_identifiers') and smiles is not None else {})}

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
            if not payload.get('validationInput') and result['summary']['uniqueSmiles']<2:
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

def main():
    request=json.loads(sys.stdin.read())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result={'preview':preview,'validate':validate,'validate-config':lambda payload:validate(payload,check_datasets=False),'model':model}[request['command']](request['payload'])
        print(json.dumps({'ok':True,'result':result},allow_nan=False))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'ok':False,'error':str(error)}))
if __name__=='__main__': main()
