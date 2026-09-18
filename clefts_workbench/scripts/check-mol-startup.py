"""Ensure run artifacts exist before preprocessing, and are enriched afterwards."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clefts.ml.training.mol_training import training_model as training

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, validation = root / 'train.smi', root / 'validation.smi'
    train.write_text('CCO\n')
    validation.write_text('CC\n')
    for dimensions in ('64', '64,128'):
        output = root / dimensions.replace(',', '-') / 'nested'
        argv = ['--train-smiles', str(train), '--val-smiles', str(validation),
                '--output-dir', str(output), '--symbols', 'C,O', '--node-dim', dimensions, '--graph-dim', '128']
        def before_preprocessing(**kwargs):
            assert output.is_dir()
            args = json.loads((output / 'training_args.json').read_text())
            assert args['train_smiles'] == [str(train)]
            initial = json.loads((output / 'pretraining_config.json').read_text())
            assert initial['args'] == args
            assert initial['symbols'] == ['C', 'O']
            assert initial['descriptor_names']
            assert initial['input_manifest']['train_files'][0]['size'] == train.stat().st_size
            assert (output / 'input_manifest.json').is_file()
            assert 'descriptor_mean' not in initial
            configs = training.mol_encoder_configs(training.parse_args(argv))
            for config in configs:
                stage = training.pretraining_stage_output_dir(output, config, len(configs))
                saved = json.loads((stage / 'mol_encoder_config.json').read_text())
                assert saved['mol_encoder_params']['node_dim'] == config.node_dim
                assert saved['mol_encoder_params']['symbols'] == ['C', 'O']
            raise RuntimeError('Stop before canonicalization')
        with patch.object(training, 'load_or_prepare_smiles_split', side_effect=before_preprocessing):
            try:
                training.main(argv)
            except RuntimeError as error:
                assert str(error) == 'Stop before canonicalization'
            else:
                raise AssertionError('Preprocessing was not reached')
        assert (output / 'pretraining_config.json').is_file()

    output = root / 'graph-failure'
    argv = ['--train-smiles', str(train), '--val-smiles', str(validation), '--output-dir', str(output),
            '--symbols', 'C,O', '--node-dim', '64', '--graph-dim', '128']
    def before_graph_preprocessing(**kwargs):
        record = json.loads((output / 'pretraining_config.json').read_text())
        assert record['smiles_split_summary'] == {'canonicalization_complete': True}
        assert 'descriptor_mean' not in record
        raise RuntimeError('Stop before graph preprocessing')
    with patch.object(training, 'load_or_prepare_smiles_split', return_value=(['CCO'], ['CC'], {'canonicalization_complete': True})), \
         patch.object(training, 'build_or_load_preprocessed_datasets', side_effect=before_graph_preprocessing):
        try:
            training.main(argv)
        except RuntimeError as error:
            assert str(error) == 'Stop before graph preprocessing'
        else:
            raise AssertionError('Graph preprocessing was not reached')

    output = root / 'completed'
    argv = ['--train-smiles', str(train), '--val-smiles', str(validation), '--output-dir', str(output),
            '--symbols', 'C,O', '--node-dim', '64', '--graph-dim', '128']
    def stage(args, **kwargs):
        record = json.loads((output / 'pretraining_config.json').read_text())
        assert record['num_train_molecules'] == 1
        assert record['descriptor_mean'] == [0.0]
        assert 'input_manifest' in record
        return {**asdict(kwargs['config']), 'selection_score': 1.0}
    with patch.multiple(training,
        load_or_prepare_smiles_split=lambda **kwargs: (['CCO'], ['CC'], {'tested': True}),
        build_or_load_preprocessed_datasets=lambda **kwargs: ([1], [2], SimpleNamespace(mean=torch.zeros(1), std=torch.ones(1)), {}),
        write_feature_target_summary=lambda **kwargs: {},
        build_feature_record_index=lambda *args, **kwargs: {},
        build_descriptor_sampling_index=lambda *args, **kwargs: ({}, {}),
        run_pretraining_stage=stage,
        save_selected_checkpoint=lambda *args: None):
        assert training.main(argv) == 0
print('Mol Training artifacts precede canonicalization, survive preprocessing failure and retain final summaries.')
