import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from clefts.libs.mmkit.mmkit import Adduct, Compound, Formula
from clefts.ml.data_preparation.fragment_tree.context import create_preparation_context
from clefts.ml.input.action_batching import select_samples
from clefts.ml.input.structure_builder import ActionStructureBuilder
from clefts.ml.training.fragment_tree_training.sources import dataset_model_config, dataset_sources, inherit_model_config
from clefts.ml.training.fragment_tree_training import training
from .TestActionPipeline import config


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

    def test_metadata_priority_and_stale_file_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'train'
            directory.mkdir()
            fragmenter = {'mass_tolerance': '0.01Da'}
            model = {'fragmenter_params': fragmenter, 'symbols': ['C', 'O'],
                     'adduct_type_strs': ['[M+H]+'], 'metadata_source': 'preparation'}
            (directory / 'preparation_config.json').write_text(json.dumps({'model_config': model}))
            (directory / 'fragment-tree.pft.json').write_text(json.dumps({
                'fragmenterParams': fragmenter, 'symbols': ['O', 'C']}))
            (directory / 'action_statistics.json').write_text(json.dumps({'model_config': {
                'fragmenter_params': fragmenter, 'mol_encoder_params': {'symbols': ['C', 'O']},
                'adduct_type_strs': ['[M+H]+']}}))
            loaded = dataset_model_config(directory)
            self.assertEqual(loaded['metadata_source'], 'preparation')
            self.assertEqual(loaded['adduct_type_strs'], ['[M+H]+'])
            stale = {'fragmenter_params': {'mass_tolerance': '0.5Da'},
                     'mol_encoder_params': {'symbols': ['C', 'N']}, 'adduct_type_strs': ['[M+Na]+']}
            (directory / 'action_statistics.json').write_text(json.dumps({'model_config': stale}))
            with self.assertRaisesRegex(ValueError, 'Regenerate this dataset'):
                dataset_model_config(directory)

    def test_molecular_encoder_symbols_must_match_prepared_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, 'train', ['C', 'N', 'O'])
            validation = self._dataset(root, 'validation', ['O', 'N', 'C'])
            checkpoint = root / 'encoder.pt'
            torch.save({
                'mol_encoder_params': {'symbols': ('C', 'N', 'S')},
                'mol_encoder_state_dict': {},
            }, checkpoint)
            with self.assertRaisesRegex(ValueError, 'molecular encoder checkpoint'):
                inherit_model_config({}, train, validation, encoder_checkpoint=checkpoint)

    def test_sample_meets_assignment_score_treats_missing_scores_as_passing(self):
        self.assertTrue(training._sample_meets_assignment_score({'assignmentScore': None, 'assignmentScoreWithoutPrecursor': None}, 0.9, 0.9))
        self.assertTrue(training._sample_meets_assignment_score({'assignmentScore': 0.9, 'assignmentScoreWithoutPrecursor': None}, 0.5, 0.9))
        self.assertFalse(training._sample_meets_assignment_score({'assignmentScore': 0.4, 'assignmentScoreWithoutPrecursor': None}, 0.5, 0))
        self.assertFalse(training._sample_meets_assignment_score({'assignmentScore': 1.0, 'assignmentScoreWithoutPrecursor': 0.2}, 0, 0.5))

    def test_assignment_score_filter_drops_low_scoring_samples_via_select_samples(self):
        # Data preparation always keeps every sample; training-time loading is
        # where a below-threshold sample now gets dropped, via select_samples.
        model = config(); model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count'] = 1
        builder = ActionStructureBuilder(create_preparation_context(model))
        source = Compound.from_smiles('CCO'); adduct = Adduct.parse('[M+H]+')
        precursor_mz = Formula.parse('C2H7O+').exact_mass
        # Sample 0 is a lone matched precursor peak (assignmentScore 1.0). Sample 1 adds
        # a large unmatched peak alongside the matched precursor, dragging its score down.
        data, kept = builder.build(source, [adduct, adduct], [20., 20.],
            [[precursor_mz], [precursor_mz, 1000.]], [[1.], [1., 100.]])
        self.assertEqual(len(data.sample_annotations), 2)
        full_scores = [sample['assignmentScore'] for sample in data.sample_annotations]
        self.assertAlmostEqual(full_scores[0], 1.0)
        self.assertLess(full_scores[1], 0.5)
        surviving = [index for index, sample in enumerate(data.sample_annotations)
                     if training._sample_meets_assignment_score(sample, 0.5, 0)]
        self.assertEqual(surviving, [0])
        filtered = select_samples(data, surviving)
        self.assertEqual(filtered.num_samples, 1)
        self.assertAlmostEqual(filtered.sample_annotations[0]['assignmentScore'], 1.0)


if __name__ == '__main__':
    unittest.main()
