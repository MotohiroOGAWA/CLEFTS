"""Check explicit fragment training options without starting training."""
import contextlib
import io
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clefts.ml.training.fragment_tree_training.training import build_arg_parser, training_model_config, main, MODEL_OPTIONS
parser = build_arg_parser()
base = ['--train-dir', 'train', '--val-dir', 'val', '--output-dir', 'out']
for flag in ['--params', '--params-json', '--set', '--fragmenter-params-json', '--mol-encoder-params-json', '--action-model-params-json', '--post-model-params-json', '--adduct-types-json', '--precursor-max-action-count', '--max-action-count', '--mass-tolerance', '--fine-tune-pattern-set']:
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(base + [flag, '{}'])
        except SystemExit as error:
            assert error.code == 2, flag
        else:
            raise AssertionError(flag)
argv = base + ['--mol-encoder-checkpoint', 'encoder.pt']
for flag, (_, _, kind, _) in MODEL_OPTIONS.items():
    argv += ['--' + flag, str(0.25 if kind is float else 64)]
config = training_model_config(parser.parse_args(argv))
assert config['mol_encoder_checkpoint'] == 'encoder.pt'
assert 'fragmenter_params' not in config and 'mol_encoder_params' not in config
for _, (section, key, kind, _) in MODEL_OPTIONS.items():
    assert config[section][key] == (0.25 if kind is float else 64)
assert config['action_model_params']['action_neighborhood_mode'] == 'hop_pooling'
none_config = training_model_config(parser.parse_args(argv + ['--action-neighborhood-mode', 'none']))
assert none_config['action_model_params']['action_neighborhood_mode'] == 'none'
with contextlib.redirect_stderr(io.StringIO()):
    try:
        parser.parse_args(argv + ['--action-neighborhood-mode', 'gnn'])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('--action-neighborhood-mode gnn')
try:
    with tempfile.TemporaryDirectory() as output_dir:
        main(['--train-dir', 'train', '--val-dir', 'val', '--output-dir', output_dir])
except SystemExit as error:
    assert '--mol-encoder-checkpoint' in str(error)
else:
    raise AssertionError('New training must require an encoder checkpoint')
assert parser.parse_args(base + ['--resume', 'last.pt']).mol_encoder_checkpoint is None
print('Explicit fragment CLI options, removed JSON options and checkpoint requirement passed.')
