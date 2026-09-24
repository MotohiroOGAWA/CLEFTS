"""Fine-tuning compatibility for ActionEncoder's hop-wise neighborhood projections.

A base checkpoint saved before this feature has no neighborhood_projections
weights at all; initialize_from_base() must tolerate exactly that gap (they
stay at their fresh zero initialization) while still treating every other
missing base parameter as a hard error.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

from clefts.ml.specgen.fine_tuning import LowRankExpansion
from clefts.ml.training.fragment_tree_training.fine_tuning import initialize_from_base


def _toy_generator():
    generator = nn.Module()
    generator.feature_model = nn.Module()
    generator.feature_model.action_encoder = nn.Module()
    action_encoder = generator.feature_model.action_encoder
    action_encoder.atom_projection = nn.Linear(4, 4)
    action_encoder.neighborhood_projections = nn.ModuleList(nn.Linear(4, 4, bias=False) for _ in range(3))
    for layer in action_encoder.neighborhood_projections:
        nn.init.zeros_(layer.weight)
    return generator


def _expand(generator, width=2):
    for module in generator.modules():
        if isinstance(module, nn.Linear):
            parametrize.register_parametrization(module, 'weight', LowRankExpansion(module.weight, width))


class TestNeighborhoodFineTuningCompatibility(unittest.TestCase):
    def test_missing_neighborhood_projections_in_base_checkpoint_is_tolerated(self):
        old = _toy_generator()
        old_state = old.state_dict()
        legacy_state = {key: value for key, value in old_state.items() if 'neighborhood_projections' not in key}
        self.assertLess(len(legacy_state), len(old_state))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'base.pt'
            torch.save({'model_state_dict': legacy_state}, checkpoint_path)
            new = _toy_generator()
            _expand(new)
            initialize_from_base(new, {'fine_tuning': {'source_checkpoint': str(checkpoint_path), 'category_mapping': {}}})
            # atom_projection had a base counterpart and is restored from it.
            self.assertTrue(torch.equal(new.feature_model.action_encoder.atom_projection.weight,
                                         old.feature_model.action_encoder.atom_projection.weight))
            self.assertTrue(torch.equal(new.feature_model.action_encoder.atom_projection.bias,
                                         old.feature_model.action_encoder.atom_projection.bias))
            # neighborhood projections had no base counterpart: left at fresh zero init.
            for layer in new.feature_model.action_encoder.neighborhood_projections:
                self.assertTrue(torch.equal(layer.weight, torch.zeros_like(layer.weight)))

    def test_other_missing_base_parameters_still_raise(self):
        old = _toy_generator()
        old_state = old.state_dict()
        broken_state = {key: value for key, value in old_state.items()
                        if key != 'feature_model.action_encoder.atom_projection.bias'}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'base.pt'
            torch.save({'model_state_dict': broken_state}, checkpoint_path)
            new = _toy_generator()
            _expand(new)
            with self.assertRaisesRegex(ValueError, 'Base checkpoint is missing'):
                initialize_from_base(new, {'fine_tuning': {'source_checkpoint': str(checkpoint_path), 'category_mapping': {}}})


if __name__ == '__main__':
    unittest.main()
