from pathlib import Path

from clefts.ml.training.mol_training.pretraining_model import (
    nodes_in_open_closed_radius,
)
from clefts.ml.training.mol_training.training_model import (
    MolEncoderConfig,
    pretraining_stage_output_dir,
)


def _config() -> MolEncoderConfig:
    return MolEncoderConfig(
        node_dim=64,
        graph_dim=128,
        num_layers=4,
        num_heads=8,
        max_degree=16,
        max_spatial_dist=5,
        max_edge_dist=5,
        dropout=0.1,
    )


def test_context_radius_excludes_inner_boundary():
    distances = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert nodes_in_open_closed_radius(distances, 1, 4) == [2, 3, 4]


def test_single_config_writes_directly_to_output_dir():
    output_dir = Path("output")
    assert pretraining_stage_output_dir(output_dir, _config(), 1) == output_dir


def test_config_search_uses_runs_and_config_id():
    output_dir = Path("output")
    config = _config()
    assert pretraining_stage_output_dir(output_dir, config, 2) == (
        output_dir / "runs" / config.id
    )


import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
import torch
from clefts.ml.training.mol_training.dataset import MolPretrainingDataset, DescriptorNormalizer
from clefts.ml.training.mol_training.trainer import make_loader, train_epochs
from clefts.ml.training.mol_training.training_model import build_arg_parser, make_pretraining_model, mol_encoder_configs


class TestMolTrainingStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self):
        args = build_arg_parser().parse_args(['--symbols', 'C,N,O', '--node-dim', '16',
            '--graph-dim', '64', '--num-layers', '2', '--num-heads', '8', '--ecfp-n-bits', '32'])
        config = mol_encoder_configs(args)[0]
        dataset = MolPretrainingDataset(['CCOCC', 'CCCCC', 'CCNCC', 'CCCOC'], symbols=('C','N','O'), ecfp_n_bits=32)
        dataset.descriptor_normalizer = DescriptorNormalizer.fit(dataset.descriptor_matrix())
        model = make_pretraining_model(args, symbols=('C','N','O'), config=config,
            descriptor_dim=len(dataset.descriptor_names), descriptor_names=dataset.descriptor_names)
        loader = make_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
        return model, loader

    def test_training_starts_without_evaluating_entire_dataset(self):
        model, loader = self.fixture()
        before = {key:value.clone() for key,value in model.state_dict().items()}
        modes = []
        hook = model.register_forward_pre_hook(lambda module, inputs: modes.append(module.training))
        logs = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(logs):
            result = train_epochs(model, loader, loader, device=torch.device('cpu'), epochs=1,
                lr=.001, node_mask_ratio=.5, edge_mask_ratio=.5, output_dir=Path(temporary), stage_name='pretraining')
            self.assertEqual(modes, [True,True,False,False])
            self.assertTrue(any(not torch.equal(value,model.state_dict()[key]) for key,value in before.items()))
            self.assertTrue(math.isfinite(result['val_loss']))
            self.assertEqual(result['epoch'],1)
            rows = list(csv.DictReader(io.StringIO((Path(temporary)/'pretraining_metrics.csv').read_text())))
            self.assertEqual([int(row['epoch']) for row in rows], [1])
            self.assertTrue((Path(temporary)/'pretraining_best.pt').is_file())
            self.assertTrue((Path(temporary)/'pretraining_last.pt').is_file())
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            accumulator=EventAccumulator(str(Path(temporary)/'tensorboard'));accumulator.Reload()
            self.assertEqual([event.step for event in accumulator.Scalars('batch/train_loss')],[1,2])
        hook.remove()
        events = [json.loads(line) for line in logs.getvalue().splitlines() if line.startswith('{')]
        self.assertEqual(events[0]['event'],'training_started')
        self.assertEqual(events[0]['initial_evaluation_batches'],0)
        self.assertIn('batch_start',[event['event'] for event in events])
        self.assertIn('batch_end',[event['event'] for event in events])
        self.assertNotIn('Initial training subset',logs.getvalue())

    def test_optional_initial_evaluation_is_bounded(self):
        model, loader = self.fixture()
        modes = []
        hook = model.register_forward_pre_hook(lambda module, inputs: modes.append(module.training))
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            result = train_epochs(model, loader, loader, device=torch.device('cpu'), epochs=1,
                lr=.001, node_mask_ratio=.5, edge_mask_ratio=.5, output_dir=Path(temporary),
                stage_name='pretraining', initial_evaluation_batches=1)
            self.assertEqual(modes,[False,False,True,True,False,False])
            rows = list(csv.DictReader(io.StringIO((Path(temporary)/'pretraining_metrics.csv').read_text())))
            self.assertEqual([int(row['epoch']) for row in rows],[0,1])
            self.assertEqual(result['epoch'],1)
        hook.remove()
        args=build_arg_parser().parse_args(['--initial-evaluation-batches','-1'])
        with self.assertRaisesRegex(ValueError,'non-negative'):mol_encoder_configs(args)
