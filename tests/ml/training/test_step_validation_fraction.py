import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from clefts.libs.msentity.msentity.core.MSDataset import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.ml.training.fragment_tree_training import training_model as training


class StructureFiles(Dataset):
    def __init__(self, names):
        self.files = [Path(name) for name in names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index].name


def fixture(tmp_path, count=10):
    score_file = tmp_path / 'assignment_scores.tsv'
    rows = []
    for i in range(count):
        # Different compound sizes and both score partitions for each compound.
        for j in range(i % 3 + 2):
            rows.append(dict(index=len(rows), SpecID=f's{len(rows)}', smiles='C' * (i + 1),
                             structure_file=f'compound_{i}.preft.pt', assignment_score=0.9 if j == 0 else 0.1))
    with score_file.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    metadata = pd.DataFrame([{'SMILES': r['smiles'], 'SpecID': r['SpecID'],
                             '__fragment_tree_original_index': r['index']} for r in rows])
    dataset = MSDataset(spectrum_metadata=metadata, peak_series=PeakSeries(
        data=np.empty((0, 2)), offsets=np.zeros(len(rows) + 1, dtype=np.int64)))
    high, low = training.split_validation_records_by_assignment_score(dataset, score_file, 0.8)
    loader = DataLoader(StructureFiles(sorted({r['structure_file'] for r in rows})), batch_size=2)
    args = dict(val_loader=loader, val_excluded_loader=loader, validation_dataset=high,
                validation_excluded_dataset=low, score_file=score_file)
    return args, dataset


@pytest.mark.parametrize('fraction,count', [(1.0, 10), (0.1, 1), (0.25, 3), (0.001, 1)])
def test_compound_subset_keeps_all_measurements_in_both_partitions(tmp_path, fraction, count):
    args, _ = fixture(tmp_path)
    inputs, report = training.make_step_validation_subset(**args, fraction=fraction)
    again, report_again = training.make_step_validation_subset(**args, fraction=fraction)
    assert report == report_again
    if fraction == 1:
        assert all(a is b for a, b in zip(inputs, list(args.values())[:4]))
    for loader, records, full_records in zip(inputs[:2], inputs[2:], list(args.values())[2:4]):
        assert len(loader.dataset) == count
        selected = set(records.metadata['SMILES'])
        assert len(selected) == count
        expected = full_records.metadata[full_records.metadata.SMILES.isin(selected)]
        assert list(records.metadata.SpecID) == list(expected.SpecID)
    assert set(inputs[2].metadata.SMILES) == set(inputs[3].metadata.SMILES)
    assert len(args['val_loader'].dataset) == 10


def test_empty_partition_is_allowed(tmp_path):
    args, _ = fixture(tmp_path)
    args['val_excluded_loader'] = None
    args['validation_excluded_dataset'] = None
    inputs, _ = training.make_step_validation_subset(**args, fraction=0.1)
    assert inputs[1] is None and inputs[3] is None


@pytest.mark.parametrize('fraction', [0, -1, 1.01, float('nan'), float('inf')])
def test_invalid_fraction_is_rejected(tmp_path, fraction):
    with pytest.raises(ValueError, match='step_validation_fraction'):
        training.build_train_config(project_dir=tmp_path, step_validation_fraction=fraction)


def test_config_default_and_round_trip(tmp_path):
    assert training.build_train_config(project_dir=tmp_path)['step_validation_fraction'] == 1.0
    config = training.build_train_config(project_dir=tmp_path, step_validation_fraction=0.1)
    prepared = training.prepare_train_from_config(tmp_path, config)
    assert prepared[9]['step_validation_fraction'] == 0.1


def test_both_cli_entry_points_accept_fraction(tmp_path):
    with patch('sys.argv', ['train', '--train-dir', str(tmp_path), '--val-dir', str(tmp_path),
                           '--output-dir', str(tmp_path), '--mol-encoder-checkpoint', 'model.pt',
                           '--step-validation-fraction', '0.1']):
        args = training.parse_args()
        assert args.step_validation_fraction == 0.1
        assert training._workbench_training_config(args)['stepValidationFraction'] == 0.1
    from clefts.cli.train.commands.fragment_tree import FragmentTreeTrainCommand
    parser = argparse.ArgumentParser()
    FragmentTreeTrainCommand().configure(parser)
    args = parser.parse_args([str(tmp_path), '--mol-encoder-checkpoint', 'model.pt',
                              '--step-validation-fraction', '0.1'])
    assert args.step_validation_fraction == 0.1


@pytest.mark.parametrize('validate_at_start,patience,epochs', [(False, None, 2), (True, 1, 3)])
def test_step_is_small_but_each_epoch_end_is_full(tmp_path, validate_at_start, patience, epochs):
    args, records = fixture(tmp_path)
    valid_file = tmp_path / 'valid.msds'; valid_file.touch()
    manager = MagicMock()
    model = MagicMock()
    model.training_phase = torch.tensor(1)
    model.update_training_phase.return_value = False
    optimizer = SimpleNamespace(param_groups=[{'lr': 1e-5}])
    state = training.TrainState(model, optimizer, None, 1, 0, float('inf'))
    loss_calls, spectrum_calls = [], []
    metrics = training.EpochLossMetrics(1., 0.5, 0.5, 2, steps=1)

    def epoch(**kwargs):
        if kwargs.get('optimizer') is not None:
            step = kwargs['start_global_step'] + 1
            kwargs['on_validation_step'](step, metrics, metrics)
        else:
            loss_calls.append((kwargs['desc'], len(kwargs['loader'].dataset)))
        return metrics

    def cosine(**kwargs):
        spectrum_calls.append((kwargs['output_dir'], len(set(kwargs['dataset'].metadata.SMILES))))
        return 0.5

    from contextlib import ExitStack
    with ExitStack() as stack:
        for name, replacement in {
            'CheckPointManager': MagicMock(return_value=manager),
            'load_or_initialize_state': MagicMock(return_value=state),
            'save_managed_checkpoint': MagicMock(),
            'run_epoch': epoch,
            'evaluate_validation_cosine': cosine,
            'write_combined_validation_cosine_summary': MagicMock(return_value={}),
            'write_combined_peak_selection_summary': MagicMock(return_value={}),
            'FragmentTreeStructureFileDataset': lambda *a, **kw: StructureFiles(kw['included_samples_by_file']),
        }.items():
            stack.enter_context(patch.object(training, name, replacement))
        scheduler = stack.enter_context(patch.object(training, 'step_scheduler'))
        stack.enter_context(patch.object(training.MSDataset, 'load', return_value=records))
        training.main(model_config={}, experiment_dir=tmp_path, ckpt_id=None,
            device=torch.device('cpu'), epoch=epochs, save_interval=1, save_interval_steps=None,
            batch_size=2, train_loader=args['val_loader'], val_loader=args['val_loader'],
            optimizer_info={}, early_stopping_info={'patience': patience}, run_dir=tmp_path,
            extra_data={'validation_valid_records_file': str(valid_file),
                        'validation_assignment_score_file': str(args['score_file']),
                        'validation_structure_dir': str(tmp_path), 'assignment_score_threshold': 0.8,
                        'step_validation_fraction': 0.1},
            validation_interval_steps=1, validate_at_start=validate_at_start)
    assert len([desc for desc, _ in loss_calls if desc.startswith('ValEpochEnd')]) == 4
    assert all(n == (1 if desc.startswith('ValStep') else 10) for desc, n in loss_calls)
    assert all(n == (1 if 'step' in path.parts else 10) for path, n in spectrum_calls)
    assert sum(desc.startswith('ValStart') for desc, _ in loss_calls) == (2 if validate_at_start else 0)
    assert scheduler.call_count == 2  # Only full epoch results, including early stopping.
    assert manager.update_topk.call_count == 1
