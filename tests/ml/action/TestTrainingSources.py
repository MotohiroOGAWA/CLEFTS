import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from clefts.ml.training.fragment_tree_training.sources import dataset_sources, inherit_model_config
from clefts.ml.training.fragment_tree_training import training


class TestTrainingSources(unittest.TestCase):
    def test_training_progress_bar_has_fixed_layout(self):
        with patch.object(training, 'tqdm') as tqdm:
            training.progress_bar(total=12, description='Iteration', position=1, leave=False)
        kwargs = tqdm.call_args.kwargs
        self.assertEqual(kwargs['ncols'], 100)
        self.assertFalse(kwargs['dynamic_ncols'])
        self.assertEqual(kwargs['position'], 1)
        self.assertIn('{bar:36}', kwargs['bar_format'])

    def _dataset(self, root: Path, name: str, symbols: list[str]) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / 'action_statistics.json').write_text(json.dumps({
            'model_config': {
                'fragmenter_params': {'test': True},
                'adduct_type_strs': ['[M+H]+'],
                'mol_encoder_params': {'symbols': symbols},
            }
        }))
        return directory

    def test_symbol_order_is_canonicalized_like_atom_feature_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, 'train', ['C', 'N', 'O', 'Cl', 'Br'])
            validation = self._dataset(root, 'validation', ['Br', 'C', 'Cl', 'N', 'O'])
            checkpoint = root / 'encoder.pt'
            torch.save({
                'mol_encoder_params': {'symbols': ('Br', 'C', 'Cl', 'N', 'O')},
                'mol_encoder_state_dict': {},
            }, checkpoint)

            sources = dataset_sources(train, validation)
            self.assertEqual(sources['symbols'], ('Br', 'C', 'Cl', 'N', 'O'))
            inherited = inherit_model_config({}, train, validation, encoder_checkpoint=checkpoint)
            self.assertEqual(inherited['mol_encoder_params']['symbols'], ('Br', 'C', 'Cl', 'N', 'O'))

    def test_different_symbol_sets_remain_incompatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, 'train', ['C', 'N', 'O'])
            validation = self._dataset(root, 'validation', ['C', 'N', 'S'])
            with self.assertRaisesRegex(ValueError, 'different elements'):
                dataset_sources(train, validation)


if __name__ == '__main__':
    unittest.main()
