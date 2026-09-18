"""Model inputs inherited from prepared datasets and pretrained checkpoints."""
from copy import deepcopy
import json
from pathlib import Path


def dataset_model_config(directory):
    directory = Path(directory)
    if directory.name == 'data':
        directory = directory.parent
    for filename in ('action_statistics.json', 'preparation_config.json', 'fragment-tree.pft.json'):
        file = directory / filename
        if not file.is_file():
            continue
        saved = json.loads(file.read_text())
        config = saved.get('model_config') or saved.get('modelConfig') or saved
        config = config.get('params', config)
        fragmenter = config.get('fragmenter_params') or config.get('fragmenterParams')
        if not fragmenter:
            continue
        return {**config, 'fragmenter_params': fragmenter,
                'symbols': config.get('symbols') or config.get('mol_encoder_params', {}).get('symbols')}
    raise ValueError(f'Dataset preparation configuration was not found in {directory}')


def dataset_sources(train_dir, val_dir):
    train = dataset_model_config(train_dir)
    validation = dataset_model_config(val_dir)
    if train['fragmenter_params'] != validation['fragmenter_params']:
        raise ValueError('Training and validation datasets have different Fragmenter parameters')
    if train.get('adduct_type_strs') != validation.get('adduct_type_strs'):
        raise ValueError('Training and validation datasets have different adduct ordering')
    if train.get('symbols') != validation.get('symbols'):
        raise ValueError('Training and validation datasets have different element ordering')
    adducts = train.get('adduct_type_strs')
    if not adducts:
        from clefts.domain.fragment.fragmenter import Fragmenter
        adducts = [str(value) for value in Fragmenter.from_dict(train['fragmenter_params']).adduct_types]
    return {'fragmenter_params': deepcopy(train['fragmenter_params']),
            'adduct_type_strs': list(adducts), 'symbols': train.get('symbols')}


def inherit_model_config(model_config, train_dir, val_dir, *, encoder_checkpoint=None, saved_model=None):
    import torch
    config = deepcopy(model_config.get('params', model_config))
    sources = dataset_sources(train_dir, val_dir)
    if saved_model is not None:
        saved = saved_model.get('params', saved_model)
        for key in ('fragmenter_params', 'adduct_type_strs'):
            if saved.get(key) != sources[key]:
                raise ValueError(f'Dataset {key} differs from the training checkpoint')
        encoder_params = saved['mol_encoder_params']
    elif encoder_checkpoint:
        checkpoint = torch.load(encoder_checkpoint, map_location='cpu', weights_only=False)
        encoder_params = checkpoint.get('mol_encoder_params')
        if not encoder_params or 'mol_encoder_state_dict' not in checkpoint:
            raise ValueError('Mol encoder checkpoint must contain parameters and pretrained weights')
        config['mol_encoder_checkpoint'] = str(encoder_checkpoint)
    else:
        raise ValueError('Specify a pretrained mol encoder checkpoint')
    if sources['symbols'] is not None and list(encoder_params['symbols']) != list(sources['symbols']):
        raise ValueError('Dataset elements and their ordering differ from the molecular encoder checkpoint')
    config.update(fragmenter_params=sources['fragmenter_params'], adduct_type_strs=sources['adduct_type_strs'],
                  mol_encoder_params=deepcopy(encoder_params))
    return config
