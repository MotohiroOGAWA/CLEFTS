from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from clefts.domain.fragment import Fragmenter
from clefts.ml.training.fragment_tree_training.training_model import build_training_model
from clefts.ml.training.fragment_tree_training.fine_tuning import prepare_model_config, initialize_from_base, parameter_report
from clefts.ml.specgen.predict_spectrum import load_generator


@pytest.fixture()
def prepared(tmp_path):
    config = json.loads(Path('clefts/presets/spectrum_generator_params/single_bond_pos_model_config.json').read_text())
    config['mol_encoder_checkpoint'] = None
    params = config['probability_model_params']
    params['dropout'] = 0.0
    params['tree_encoder_params'] = dict(hidden_dim=16, num_layers=1, num_heads=4, max_degree=16)
    params['mol_encoder_params'].update(node_dim=16, graph_dim=16, num_layers=1, num_heads=4)
    params['fragment_edge_encoder_params'].update(feature_dim=16, category_dim=4, num_heads=4)
    base = build_training_model(config, torch.device('cpu'))
    checkpoint = tmp_path / 'base.pt'
    torch.save(dict(model_config=config, model_state_dict=base.state_dict()), checkpoint)
    fragmenter = Fragmenter.from_dict(params['fragmenter_params']).to_dict()
    pattern_set = fragmenter['fragment_ion_tree_builder']['cleavage_pattern_set']
    pattern_set['patterns'][0]['pattern_id'] = 1
    pattern_set['patterns'].insert(0, dict(pattern_id=0, name='carbonyl', reactant_smarts='[#6:1]=[#8:2]',
                                        products=[dict(name='carbon', smarts='[#6:1]')]))
    pattern_file = tmp_path / 'new.clevageset.json'
    pattern_file.write_text(json.dumps({'cleavage_pattern_set': pattern_set}))
    preprocessing = dict(symbols=params['mol_encoder_params']['symbols'], fragmenter_params=fragmenter)
    cfg_dir = tmp_path / 'data/config'; cfg_dir.mkdir(parents=True)
    (cfg_dir / 'preprocessing_config.json').write_text(json.dumps(preprocessing))
    for name in ('train_structures', 'validation_structures'):
        folder = tmp_path / 'data' / name; (folder / 'data').mkdir(parents=True)
        (folder / 'fragmenter.json').write_text(json.dumps(fragmenter))
    from clefts.ml.specgen.predict_spectrum import direct_input_dataset
    direct_input_dataset(smiles_values=['CC=O'], collision_energy='20 eV', adduct_type='[M+H]+').save(str(tmp_path / 'data/validation_structures/valid_records.msds'))
    new_config = prepare_model_config(checkpoint, pattern_file, preprocessing, width=2)
    model = build_training_model(new_config, torch.device('cpu'))
    initialize_from_base(model, new_config)
    return tmp_path, base, model, new_config, preprocessing, pattern_file


def test_frozen_parameters_new_rows_and_reload(prepared):
    tmp, base, model, config, _, _ = prepared
    fm = model.candidate_selector.feature_model
    edge = fm.fragment_edge_encoder
    before = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    assert before and not any(p.requires_grad for p in fm.mol_encoder.parameters())
    assert all('parametrizations.' in name for name, p in model.named_parameters() if p.requires_grad)
    old_embedding = base.candidate_selector.feature_model.fragment_edge_encoder.pattern_embedding.weight.detach()
    assert torch.equal(edge.pattern_embedding.weight[1], old_embedding[0])
    base_linear = base.candidate_selector.feature_model.fragment_edge_encoder.atom_projection
    x = torch.randn(4, base_linear.in_features)
    assert torch.equal(edge.atom_projection(x), base_linear(x))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    old_new_row = edge.pattern_embedding.weight[0].detach().clone()
    for _ in range(2):
        model.train(); optimizer.zero_grad()
        assert not fm.mol_encoder.training
        loss = edge.atom_projection(x).square().mean() + edge.pattern_embedding(torch.tensor([0, 1])).square().mean()
        q = torch.randn(1, 3, edge.feature_dim)
        loss = loss + edge.cross_attention(q, q, q)[0].square().mean()
        loss.backward(); optimizer.step()
    for name, value in model.named_parameters():
        if name in before:
            assert torch.equal(value, before[name]), name
    assert not torch.equal(edge.pattern_embedding.weight[0], old_new_row)
    assert torch.equal(edge.pattern_embedding.weight[1], old_embedding[0])
    assert not torch.equal(edge.atom_projection(x), base_linear(x))
    assert edge.cross_attention.parametrizations.in_proj_weight[0].up.abs().sum() > 0
    assert edge.cross_attention.out_proj.parametrizations.weight[0].up.abs().sum() > 0
    checkpoint = tmp / 'fine.pt'
    torch.save(dict(model_config=config, model_state_dict=model.state_dict()), checkpoint)
    (tmp / 'base.pt').unlink()
    loaded = load_generator(model_path=str(checkpoint), params_path=None, device=torch.device('cpu'), strict=True)
    model.eval()
    assert torch.equal(loaded.feature_model.fragment_edge_encoder.atom_projection(x), edge.atom_projection(x))
    assert torch.equal(loaded.feature_model.fragment_edge_encoder.pattern_embedding.weight, edge.pattern_embedding.weight)
    assert not any(p.requires_grad for p in loaded.feature_model.mol_encoder.parameters())
    restored = build_training_model(config, torch.device('cpu'))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert parameter_report(restored) == parameter_report(model)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01, weight_decay=0.1)
    restored_optimizer.load_state_dict(optimizer.state_dict())


def test_cli_dry_run_and_invalid_set(prepared):
    tmp, _, model, _, preprocessing, pattern_file = prepared
    args = [sys.executable, '-m', 'clefts.cli', 'train', 'fragment-tree-finetune',
            '--checkpoint', str(tmp / 'base.pt'), '--cleavage-pattern-set', str(pattern_file),
            '--train-dir', str(tmp / 'data/train_structures/data'),
            '--val-dir', str(tmp / 'data/validation_structures/data'),
            '--output-dir', str(tmp / 'output'), '--adapter-width', '2', '--dry-run']
    process = subprocess.run(args, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout[process.stdout.index('{'):])
    assert report['trainable_parameters'] == parameter_report(model)['trainable_parameters']
    assert report['frozen_parameters'] > report['trainable_parameters'] > 0
    assert not (tmp / 'output').exists()
    changed = deepcopy(preprocessing)
    changed['fragmenter_params']['fragment_ion_tree_builder']['max_depth'] += 1
    with pytest.raises(ValueError, match='Only the cleavage pattern set'):
        prepare_model_config(tmp / 'base.pt', pattern_file, changed)
    data = json.loads(pattern_file.read_text())
    data['cleavage_pattern_set']['patterns'].pop()
    pattern_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='does not match'):
        prepare_model_config(tmp / 'base.pt', pattern_file, preprocessing)


def test_real_fragment_tree_training_cli(prepared):
    import numpy as np
    from clefts.libs.msentity.msentity import MSDataset, PeakSeries
    from clefts.ml.specgen.predict_spectrum import direct_input_dataset
    from clefts.ml.input.training_fragment_tree_structure_builder import TrainingFragmentTreeStructureBuilder
    tmp, _, model, _, _, pattern_file = prepared
    metadata = direct_input_dataset(smiles_values=['CC=O'], collision_energy='20 eV', adduct_type='[M+H]+').metadata
    peaks = PeakSeries(data=np.array([[45.033491, 100.], [43.017841, 60.], [29.038576, 30.], [15.022926, 20.]]), offsets=np.array([0, 4]))
    dataset = MSDataset(spectrum_metadata=metadata, peak_series=peaks)
    builder = TrainingFragmentTreeStructureBuilder(model.candidate_selector.feature_model)
    ids = builder.add_training_sample(dataset, precursor_mz_column='PrecursorMZ', adduct_type_column='AdductType',
                                      collision_energy_column='CollisionEnergy', require_precursor_path_targets=False)
    assert ids.tolist() == [0]
    structure = builder.to_structure()
    for split in ('train_structures', 'validation_structures'):
        folder = tmp / 'data' / split
        torch.save(structure, folder / 'data/example.preft.pt')
        (folder / 'assignment_scores.tsv').write_text('structure_file\tassignment_score\tindex\nexample.preft.pt\t1.0\t0\n')
    dataset.save(str(tmp / 'data/validation_structures/valid_records.msds'))
    args = [sys.executable, '-m', 'clefts.cli', 'train', 'fragment-tree-finetune',
            '--checkpoint', str(tmp / 'base.pt'), '--cleavage-pattern-set', str(pattern_file),
            '--train-dir', str(tmp / 'data/train_structures'),
            '--val-dir', str(tmp / 'data/validation_structures'),
            '--output-dir', str(tmp / 'output'), '--adapter-width', '2', '--epochs', '1', '--max-samples', '1']
    process = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-4000:]
    assert list((tmp / 'output').rglob('fine_tuning_parameters.json'))
    checkpoints = list((tmp / 'output').rglob('model.pt'))
    assert checkpoints
    latest = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
    assert latest['iter'] > 0

    state = latest['model_state_dict']
    frozen = {name: p for name, p in model.named_parameters() if not p.requires_grad}
    for name, parameter in frozen.items():
        if any(name.endswith(f'.{category}.parametrizations.weight.original')
               for category in ('pattern_embedding', 'reaction_embedding', 'product_embedding')):
            category = name.split('.')[-4]
            for _, new_row in model.get_params()['fine_tuning']['category_mapping'][category]:
                assert torch.equal(state[name][new_row], parameter[new_row]), name
        else:
            assert torch.equal(state[name], parameter), name
    assert any(value.abs().sum() > 0 for key, value in state.items() if key.endswith('.0.up'))

    latest_path = max(checkpoints, key=lambda path: int(path.parent.name))
    resumed = subprocess.run(args + ['--ckpt-id', latest_path.parent.name], capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout[-4000:] + resumed.stderr[-4000:]
    resumed_path = max((tmp / 'output').rglob('model.pt'), key=lambda path: int(path.parent.name))
    resumed_checkpoint = torch.load(resumed_path, map_location='cpu', weights_only=False)
    assert resumed_checkpoint['iter'] > latest['iter']
    assert resumed_checkpoint['epoch'] == 2
