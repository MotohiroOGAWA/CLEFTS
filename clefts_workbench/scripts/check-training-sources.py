"""Check inherited inputs independently of model training and dataset tensor loading."""
import importlib.util
import json
from pathlib import Path
import tempfile
import torch

source = Path(__file__).resolve().parents[2] / 'clefts/ml/training/fragment_tree_training/sources.py'
spec = importlib.util.spec_from_file_location('training_sources', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, validation = root / 'train', root / 'validation'
    model = {'fragmenter_params': {'mass_tolerance': '0.01Da'},
             'adduct_type_strs': ['[M+H]+', '[M+Na]+'],
             'mol_encoder_params': {'symbols': ['C', 'O'], 'node_dim': 32}}
    for split in (train, validation):
        (split / 'data').mkdir(parents=True)
        (split / 'action_statistics.json').write_text(json.dumps({'model_config': model}))
    checkpoint = root / 'encoder.pt'
    encoder = {'symbols': ('C', 'O'), 'node_dim': 64, 'graph_dim': 256}
    torch.save({'mol_encoder_params': encoder, 'mol_encoder_state_dict': {'weight': torch.zeros(1)}}, checkpoint)
    result = module.inherit_model_config({'fragmenter_params': {'ignored': True},
        'mol_encoder_params': {'node_dim': 999}, 'adduct_type_strs': ['ignored'],
        'action_model_params': {'hidden_dim': 128}}, train / 'data', validation, encoder_checkpoint=checkpoint)
    assert result['fragmenter_params'] == model['fragmenter_params']
    assert result['adduct_type_strs'] == model['adduct_type_strs']
    assert result['mol_encoder_params'] == encoder
    assert result['action_model_params'] == {'hidden_dim': 128}
    assert module.inherit_model_config(result, train, validation, saved_model=result) == result
    def rejects(operation):
        try:
            operation()
        except ValueError:
            return
        raise AssertionError('Incompatible source settings were accepted')
    rejects(lambda: module.inherit_model_config({}, train, validation))
    model['fragmenter_params']['mass_tolerance'] = '0.1Da'
    (validation / 'action_statistics.json').write_text(json.dumps({'model_config': model}))
    rejects(lambda: module.dataset_sources(train, validation))
    model['fragmenter_params']['mass_tolerance'] = '0.01Da'
    model['adduct_type_strs'].reverse()
    (validation / 'action_statistics.json').write_text(json.dumps({'model_config': model}))
    rejects(lambda: module.dataset_sources(train, validation))
    model['adduct_type_strs'].reverse()
    (validation / 'action_statistics.json').write_text(json.dumps({'model_config': model}))
    encoder['symbols'] = ('O', 'C')
    torch.save({'mol_encoder_params': encoder, 'mol_encoder_state_dict': {}}, checkpoint)
    rejects(lambda: module.inherit_model_config({}, train, validation, encoder_checkpoint=checkpoint))
print('Dataset inheritance, encoder checkpoint metadata, resume compatibility and mismatch checks passed.')
