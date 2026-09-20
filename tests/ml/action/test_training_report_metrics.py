import json

import pytest

from clefts.domain.mass.tolerance import DaTolerance
from clefts.ml.training.fragment_tree_training.report import REPORT_NAME, write_training_report
from clefts.ml.training.fragment_tree_training.spectrum_validation import assignment_score, distribution, matched_cosine


def test_assignment_and_cosine_without_intensity_prediction():
    tolerance = DaTolerance(0.01)
    observed = [{'mz': 100., 'intensity': 9.}, {'mz': 200., 'intensity': 1.}]
    assert assignment_score([100.], observed, tolerance) == pytest.approx(.9)
    assert matched_cosine([100.], [1.], observed, tolerance) == pytest.approx(9 / (82 ** .5))


def test_report_quantiles_and_pft_manifest(tmp_path):
    stats = distribution(range(11))
    assert stats['q10'] == 1
    assert stats['q25'] == 2.5
    assert stats['median'] == 5
    assert stats['q75'] == 7.5
    assert stats['q90'] == 9
    target = write_training_report(tmp_path / 'project' / 'runs' / 'run-1', status='running',
                                   completed_epochs=2, global_step=40,
                                   latest_validation='spectrum_validation/epoch_2.json')
    assert target.name == REPORT_NAME == 'training.pft.json'
    payload = json.loads(target.read_text())
    assert payload['schema'] == 'clefts.training-report'
    assert payload['kind'] == 'training-result'
    assert payload['completedEpochs'] == 2
