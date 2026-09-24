"""Model inputs inherited from prepared datasets and pretrained checkpoints."""
from copy import deepcopy
import json
from pathlib import Path


def canonical_symbols(symbols):
    """Return the feature order used by AtomFeatureLayer."""
    return tuple(sorted(set(symbols))) if symbols is not None else None


def _metadata_config(saved):
    """Extract normalized model inputs from any supported preparation file."""
    containers = [saved.get('model_config'), saved.get('modelConfig')]
    workbench = saved.get('workbench_config')
    if isinstance(workbench, dict):
        containers.extend((workbench.get('modelConfig'), workbench))
    containers.append(saved)
    for config in containers:
        if not isinstance(config, dict):
            continue
        config = config.get('params', config)
        fragmenter = config.get('fragmenter_params') or config.get('fragmenterParams')
        if not fragmenter:
            continue
        result = deepcopy(config)
        result['fragmenter_params'] = deepcopy(fragmenter)
        result['symbols'] = deepcopy(config.get('symbols') or config.get('mol_encoder_params', {}).get('symbols'))
        if config.get('adduct_type_strs') is not None:
            result['adduct_type_strs'] = list(config['adduct_type_strs'])
        return result
    return None


def _metadata_equal(key, left, right):
    if key == 'symbols':
        return canonical_symbols(left) == canonical_symbols(right)
    if key == 'adduct_type_strs':
        return tuple(left) == tuple(right)
    return left == right


def dataset_model_config(directory):
    directory = Path(directory)
    if directory.name == 'data':
        directory = directory.parent
    candidates = []
    files = [(directory / 'preparation_config.json', 'preparation_config.json')]
    if directory.name in ('train_structures', 'validation_structures'):
        files.append((directory.parent / 'preparation_config.json', '../preparation_config.json'))
    # fragment-tree.pft.json is the pre-rename name; datasets prepared before
    # that change only have it, and its data is otherwise identical.
    files.extend((directory / filename, filename) for filename in
                 ('fragment-tree.pft', 'fragment-tree.pft.json', 'action_statistics.json'))
    for file, filename in files:
        if not file.is_file():
            continue
        saved = json.loads(file.read_text())
        config = _metadata_config(saved)
        if config is not None:
            candidates.append((filename, config))
    if not candidates:
        raise ValueError(f'Dataset preparation configuration was not found in {directory}')
    for key in ('fragmenter_params', 'symbols', 'adduct_type_strs'):
        defined = [(filename, config.get(key)) for filename, config in candidates if config.get(key) is not None]
        if len(defined) > 1:
            first_file, first_value = defined[0]
            for filename, value in defined[1:]:
                if not _metadata_equal(key, first_value, value):
                    raise ValueError(
                        f'Dataset metadata conflict for {key}: {first_file} differs from {filename} in {directory}. '
                        'Regenerate this dataset before training.')
    result = deepcopy(candidates[0][1])
    for key in ('fragmenter_params', 'symbols', 'adduct_type_strs'):
        if result.get(key) is None:
            for _, config in candidates[1:]:
                if config.get(key) is not None:
                    result[key] = deepcopy(config[key])
                    break
    return result


def dataset_sources(train_dir, val_dir):
    train = dataset_model_config(train_dir)
    validation = dataset_model_config(val_dir)
    if train['fragmenter_params'] != validation['fragmenter_params']:
        raise ValueError('Training Fragmenter does not match Validation Fragmenter. Regenerate or reselect compatible datasets.')
    if train.get('adduct_type_strs') != validation.get('adduct_type_strs'):
        raise ValueError('Training adduct ordering does not match Validation adduct ordering. Regenerate or reselect compatible datasets.')
    train_symbols = canonical_symbols(train.get('symbols'))
    validation_symbols = canonical_symbols(validation.get('symbols'))
    if train_symbols != validation_symbols:
        raise ValueError('Training and validation datasets use different elements: Training symbols do not match Validation symbols. Regenerate or reselect compatible datasets.')
    adducts = train.get('adduct_type_strs')
    if not adducts:
        from clefts.domain.fragment.fragmenter import Fragmenter
        adducts = [str(value) for value in Fragmenter.from_dict(train['fragmenter_params']).adduct_types]
    return {'fragmenter_params': deepcopy(train['fragmenter_params']),
            'adduct_type_strs': list(adducts), 'symbols': train_symbols}


def prepared_max_action_role_count(train_dir, val_dir) -> int | None:
    """Largest action_source_atom role count across every prepared .preft.pt file.

    Sizes the shared SMARTS-query role embedding: only the actual prepared
    action universe (not a user guess) can say how large that ever gets, so
    this is not a user-configurable model parameter. Returns None if neither
    directory has a prepared file yet, so the caller can fall back to a
    pattern-derived estimate instead of failing outright.
    """
    from clefts.ml.input.source_action_structure import SourceActionStructure
    total = None
    for directory in (train_dir, val_dir):
        for file in sorted(Path(directory).rglob('*.preft.pt')):
            count = SourceActionStructure.load(file).max_action_role_count
            total = count if total is None else max(total, count)
    return total


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
    if sources['symbols'] is not None and canonical_symbols(encoder_params.get('symbols')) != sources['symbols']:
        raise ValueError('Dataset elements differ from the molecular encoder checkpoint: Dataset symbols do not match the MolEncoder / applicable checkpoint symbols')
    config.update(fragmenter_params=sources['fragmenter_params'], adduct_type_strs=sources['adduct_type_strs'],
                  mol_encoder_params=deepcopy(encoder_params))
    max_roles = prepared_max_action_role_count(train_dir, val_dir)
    if max_roles is not None:
        config.setdefault('action_model_params', {})['max_roles'] = max_roles
    return config
