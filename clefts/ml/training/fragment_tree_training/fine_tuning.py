"""Prepare and validate frozen-base fine-tuning for an expanded cleavage set."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re

import torch

from clefts.domain.fragment import Fragmenter
from clefts.domain.fragment.cleavage import CleavagePatternSet


def category_mapping(old_set, new_set):
    """Match chemical definitions, not positional IDs that may be sorted on import."""
    def categories(pattern_set):
        patterns, reactions, products = {}, {}, {}
        ids = set()
        for pattern in pattern_set.patterns:
            if pattern.pattern_id < 0 or pattern.pattern_id in ids or pattern.key in patterns:
                raise ValueError('Pattern IDs and chemical definitions must be unique.')
            ids.add(pattern.pattern_id)
            patterns[pattern.key] = int(pattern.pattern_id)
            for reaction in pattern.cleavage_reactions:
                key = (pattern.key, int(reaction.id))
                reactions[key] = len(reactions)
                for product_id, _ in enumerate(reaction.prod_idx_to_maps):
                    products[(*key, product_id)] = len(products)
        if ids != set(range(len(ids))):
            raise ValueError('Pattern IDs must be contiguous from zero.')
        return patterns, reactions, products

    old = categories(old_set)
    new = categories(new_set)
    if not old[0] or len(new[0]) <= len(old[0]):
        raise ValueError('Select a set containing every old pattern and at least one new pattern.')
    mapping = {}
    for name, before, after in zip(('pattern_embedding', 'reaction_embedding', 'product_embedding'), old, new):
        if not before.keys() <= after.keys():
            raise ValueError('The new set removes or changes an existing cleavage definition.')
        mapping[name] = [[index, after[key]] for key, index in before.items()]
    return mapping


def prepare_model_config(checkpoint_path, pattern_set_path, preprocessing, width=8):
    if width < 1:
        raise ValueError('Adapter width must be positive.')
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model_state_dict'), dict):
        raise ValueError('A fragment-tree training checkpoint with model_state_dict is required.')
    config = deepcopy(checkpoint.get('model_config', {}))
    config = config.get('params', config)
    if 'probability_model_params' not in config:
        raise ValueError('Checkpoint is missing probability_model_params.')
    if config.get('fine_tuning'):
        raise ValueError('This command starts from a base checkpoint. Resume an existing fine-tuning run with --ckpt-id.')
    old_fragmenter = Fragmenter.from_dict(config['probability_model_params']['fragmenter_params'])
    new_fragmenter = Fragmenter.from_dict(preprocessing['fragmenter_params'])
    requested = json.loads(Path(pattern_set_path).read_text())
    requested = CleavagePatternSet.from_dict(requested.get('cleavage_pattern_set', requested))
    if sorted(p.key for p in requested.patterns) != sorted(p.key for p in new_fragmenter.cleavage_pattern_set.patterns):
        raise ValueError('Selected cleavage set does not match the prepared data. Rebuild both splits with the new set first.')
    old_params, new_params = old_fragmenter.to_dict(), new_fragmenter.to_dict()
    for params in (old_params, new_params):
        params['fragment_ion_tree_builder'].pop('cleavage_pattern_set')
    if old_params != new_params:
        raise ValueError('Only the cleavage pattern set may change; keep adducts, depth and other Fragmenter settings unchanged.')
    mapping = category_mapping(old_fragmenter.cleavage_pattern_set, new_fragmenter.cleavage_pattern_set)
    config['probability_model_params']['fragmenter_params'] = new_fragmenter.to_dict()
    # Full checkpoints already contain MolEncoder weights; no external file needed.
    config['mol_encoder_checkpoint'] = None
    config['freeze_mol_encoder'] = True
    config['fine_tuning'] = dict(version=1, width=int(width), source_checkpoint=str(checkpoint_path), category_mapping=mapping)
    return config


def initialize_from_base(model, model_config):
    """Copy every old tensor; retain only new rows/adapter initialization and new lookup tables."""
    config = model_config['fine_tuning']
    source = torch.load(config['source_checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
    target = model.state_dict()
    used = set()
    for key, value in target.items():
        if re.search(r'\.parametrizations\.[^.]+\.0\.', key):
            continue  # adapter/row values and new-row index buffer
        original_key = re.sub(r'\.parametrizations\.([^.]+)\.original$', r'.\1', key)
        if original_key not in source:
            raise ValueError(f'Base checkpoint is missing {original_key}')
        old_value = source[original_key]
        used.add(original_key)
        if key.endswith(('.reaction_lookup_table', '.product_lookup_table')):
            continue  # constructed using the new definitions
        category = next((name for name in config['category_mapping']
                         if key.endswith(f'.{name}.parametrizations.weight.original')), None)
        if category:
            pairs = config['category_mapping'][category]
            if old_value.ndim != 2 or old_value.shape[1] != value.shape[1] or len(pairs) != old_value.shape[0]:
                raise ValueError(f'Incompatible category weights: {key}')
            for old_row, new_row in pairs:
                value[new_row].copy_(old_value[old_row])
        else:
            if old_value.shape != value.shape:
                raise ValueError(f'Base checkpoint shape mismatch for {original_key}')
            value.copy_(old_value)
    unexpected = set(source) - used
    if unexpected:
        raise ValueError(f'Unexpected base checkpoint tensors: {sorted(unexpected)[:5]}')
    model.load_state_dict(target, strict=True)


def parameter_report(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[name] = parameter.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(groups.values())
    return dict(total_parameters=total, frozen_parameters=total-trainable,
                trainable_parameters=trainable, trainable_tensors=groups)
