import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clefts.ml.data_preparation.fragment_tree import create_fragment_tree_training_data as module


PARAMS = Path(__file__).resolve().parents[3] / 'clefts/domain/fragment/presets/fragmenter_single_bond_pos.json'


def arguments(tmp_path, *extra):
    return module.parse_args([
        '--train-input', 'train input.msds', '--validation-input', 'validation.msds',
        '--output-dir', str(tmp_path), '--symbols', 'C', 'H', 'N', 'O', 'P', 'S',
        *extra,
    ])


def test_inline_params_match_file_context(tmp_path):
    inline = arguments(tmp_path, '--params-json', PARAMS.read_text())
    file = arguments(tmp_path, '--params', str(PARAMS))
    assert module.load_preprocessing_context(inline).to_dict() == module.load_preprocessing_context(file).to_dict()


def test_run_config_preserves_io_and_embeds_params(tmp_path):
    args = arguments(tmp_path, '--params-json', PARAMS.read_text(), '--train-valid-output', 'valid.msds')
    with patch.object(module, 'load_preprocessing_context', side_effect=RuntimeError('stop before processing')):
        with pytest.raises(RuntimeError, match='stop before processing'):
            module.main(args)
    config = json.loads((tmp_path / 'fragment-tree.pft.json').read_text())
    assert config['trainInput'] == 'train input.msds'
    assert config['validationInput'] == 'validation.msds'
    assert config['outputDir'] == str(tmp_path)
    assert config['trainValidOutput'] == 'valid.msds'
    assert config['fragmenterParams'] == json.loads(PARAMS.read_text())
    assert 'params' not in config


def test_inline_params_require_object(tmp_path):
    with pytest.raises(ValueError, match='JSON object'):
        module.resolve_fragmenter_params(arguments(tmp_path, '--params-json', '[]'))
