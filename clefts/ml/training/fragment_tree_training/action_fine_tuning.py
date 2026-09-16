"""Frozen-base fine-tuning config prep for the Source-anchored action architecture."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import torch
from clefts.domain.fragment import Fragmenter
from clefts.domain.fragment.cleavage import CleavagePatternSet
from clefts.ml.specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from .fine_tuning import category_mapping, initialize_from_base, parameter_report

__all__ = ["prepare_action_model_config", "initialize_from_base", "parameter_report"]


def _local_category_sizes(pattern_set: CleavagePatternSet) -> tuple[int, int]:
    """Match SourceAnchoredFragmentSpectrumGenerator's category_sizes exactly."""
    reaction_size = max((len(pattern.cleavage_reactions) for pattern in pattern_set.patterns), default=0)
    product_size = max((len(reaction.prod_temp) for pattern in pattern_set.patterns
                        for reaction in pattern.cleavage_reactions), default=1)
    return reaction_size, product_size


def _action_category_mapping(old_set: CleavagePatternSet, new_set: CleavagePatternSet) -> dict:
    """Map old_set's rows into new_set's, for ActionEncoder.categories.

    categories[0] (pattern) is keyed by the globally unique pattern identity,
    exactly like the edge-based architecture's pattern_embedding, so it reuses
    the same chemistry-identity matching. categories[1]/[2] (reaction/product)
    are instead *local* position indices shared additively by every pattern
    (row k always means "the k-th reaction of whichever pattern produced this
    action"), so old positions never move: expanding the pattern set can only
    ever append new local rows at the end, never reassign existing ones.
    """
    pattern_mapping = category_mapping(old_set, new_set)['pattern_embedding']
    old_reaction_size, old_product_size = _local_category_sizes(old_set)
    new_reaction_size, new_product_size = _local_category_sizes(new_set)
    if new_reaction_size < old_reaction_size or new_product_size < old_product_size:
        raise ValueError('The new pattern set has fewer reactions/products per pattern than the base; cannot fine-tune.')
    return {
        'categories.0': pattern_mapping,
        'categories.1': [[index, index] for index in range(old_reaction_size)],
        'categories.2': [[index, index] for index in range(old_product_size)],
    }


def prepare_action_model_config(*, checkpoint_path: str | Path, pattern_set_path: str | Path,
                                new_params_path: str | Path, width: int = 8) -> dict:
    """Build a fine-tuning model_config from a base action checkpoint.

    new_params_path must already carry the complete, expanded fragmenter_params
    (produced the same way as any other action model config); the training
    data itself must be regenerated with that same config before fine-tuning,
    since resolve_precursor_actions and the action universe both depend on it.
    """
    if width < 1:
        raise ValueError('Adapter width must be positive.')
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model_state_dict'), dict):
        raise ValueError('An action training checkpoint with model_state_dict is required.')
    if checkpoint.get('fragmentation_schema') != SourceAnchoredFragmentSpectrumGenerator.architecture:
        raise ValueError('This command fine-tunes a Source-anchored action checkpoint.')
    base_config = deepcopy(checkpoint.get('model_config', {}))
    base_config = base_config.get('params', base_config)
    if 'fragmenter_params' not in base_config:
        raise ValueError('Checkpoint is missing fragmenter_params.')
    if base_config.get('fine_tuning'):
        raise ValueError('This command starts from a base checkpoint. Resume an existing fine-tuning run with --resume.')
    new_config = json.loads(Path(new_params_path).read_text())
    new_config = new_config.get('params', new_config)
    if 'fragmenter_params' not in new_config:
        raise ValueError('new_params_path is missing fragmenter_params.')
    old_fragmenter = Fragmenter.from_dict(base_config['fragmenter_params'])
    new_fragmenter = Fragmenter.from_dict(new_config['fragmenter_params'])
    requested = json.loads(Path(pattern_set_path).read_text())
    requested = CleavagePatternSet.from_dict(requested.get('cleavage_pattern_set', requested))
    if sorted(p.key for p in requested.patterns) != sorted(p.key for p in new_fragmenter.cleavage_pattern_set.patterns):
        raise ValueError('Selected cleavage set does not match new_params_path. Regenerate the training data with the new set first.')
    old_params, new_params = old_fragmenter.to_dict(), new_fragmenter.to_dict()
    for params in (old_params, new_params):
        params['fragment_ion_tree_builder'].pop('cleavage_pattern_set')
    if old_params != new_params:
        raise ValueError('Only the cleavage pattern set may change; keep adducts, depth and other Fragmenter settings unchanged.')
    mapping = _action_category_mapping(old_fragmenter.cleavage_pattern_set, new_fragmenter.cleavage_pattern_set)
    config = deepcopy(new_config)
    config['fragmenter_params'] = new_fragmenter.to_dict()
    config['fine_tuning'] = dict(version=1, width=int(width), source_checkpoint=str(checkpoint_path), category_mapping=mapping)
    return config
