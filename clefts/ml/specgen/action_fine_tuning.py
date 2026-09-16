"""Frozen-base, low-rank expansion for the Source-anchored action architecture."""
from __future__ import annotations
from torch import nn
from torch.nn.utils import parametrize
from .fine_tuning import LowRankExpansion, NewCategoryRows

# ActionEncoder.categories is an nn.ModuleList indexed
# (pattern_id, reaction_id, product_molecule_id); category_mapping() in
# training/fragment_tree_training/fine_tuning.py returns semantic names, so
# callers translate them to this positional attribute path before calling in.
CATEGORY_ATTRIBUTE_NAMES = ("categories.0", "categories.1", "categories.2")


def install_action_expansion(generator, config: dict) -> None:
    """Called by the generator constructor, before any optimizer is created."""
    if config.get('version') != 1 or int(config.get('width', 0)) < 1:
        raise ValueError('Unsupported fine-tuning version or nonpositive adapter width.')
    width = int(config['width'])
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    mol_modules = {id(module) for module in generator.mol_encoder.modules()}
    # Snapshot first: newly registered low-rank parameters must not be expanded again.
    for _, module in list(generator.named_modules()):
        if id(module) in mol_modules:
            continue
        names = ['weight'] if isinstance(module, nn.Linear) else []
        if isinstance(module, nn.MultiheadAttention):
            names = [name for name in ('in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight')
                     if getattr(module, name, None) is not None]
        for name in names:
            parametrize.register_parametrization(module, name, LowRankExpansion(getattr(module, name), width))
    categories = generator.feature_model.action_encoder.categories
    for name, mapping in config['category_mapping'].items():
        embedding = categories[int(name.rsplit('.', 1)[1])]
        parametrize.register_parametrization(
            embedding, 'weight', NewCategoryRows(embedding.weight, [pair[1] for pair in mapping]))
    generator.feature_model.freeze_mol_encoder()
