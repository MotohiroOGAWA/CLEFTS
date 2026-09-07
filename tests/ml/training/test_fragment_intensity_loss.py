from types import SimpleNamespace
import torch
from clefts.ml.training.fragment_tree_training.model import FragmentTreeIntensityTrainingLoss


def fixture(precursor=1000.0):
    return SimpleNamespace(formula_peak_index=torch.tensor([0, 1, 2]),
        peak_sample_index=torch.tensor([0, 0, 0]), sample_peak_mz=torch.tensor([100., 30., 50.]),
        sample_peak_intensity=torch.tensor([precursor, 3., 1.]),
        target_formula=torch.tensor([[10., 1.], [3., 1.], [5., 1.]]))


def loss(prediction, target=None, formulas=None, mz=None):
    target = fixture() if target is None else target
    formulas = target.target_formula if formulas is None else formulas
    mz = torch.tensor([100., 30., 50.]) if mz is None else mz
    return FragmentTreeIntensityTrainingLoss().fragment_loss(prediction, formulas, mz, target, 0, [100.])


def test_fragment_loss_ignores_precursor_and_rescales_fragments():
    good = torch.tensor([1000., .3, .1], requires_grad=True)
    assert loss(good).abs() < 1e-6
    bad = torch.tensor([1000., .1, .3], requires_grad=True)
    assert loss(bad) > .4
    assert torch.allclose(loss(bad), loss(bad, fixture(1e8)))
    loss(bad).backward()
    assert bad.grad[0] == 0
    assert bad.grad[1] < 0 and bad.grad[2] > 0
    assert torch.allclose(loss(torch.tensor([1., 10., 30.])), loss(bad))


def test_missing_formula_and_false_positive_are_penalized():
    target = fixture()
    missing = loss(torch.tensor([1000., 1.]), target, target.target_formula[:2], torch.tensor([100., 30.]))
    assert missing > .05
    incorrect = loss(torch.tensor([1000., 3., 1., 2.]), target,
                     torch.cat([target.target_formula, torch.tensor([[8., 1.]])]), torch.tensor([100., 30., 50., 80.]))
    assert incorrect > .05


def test_precursor_tolerance_empty_fragments_and_zero_predictions():
    prediction = torch.tensor([4., 3., 1.], requires_grad=True)
    assert loss(prediction, mz=torch.tensor([100.005, 30., 50.])).abs() < 1e-6
    target = fixture()
    target.sample_peak_intensity = torch.tensor([1000., 0., 0.])
    result = loss(prediction, target)
    assert result == 0 and torch.isfinite(result)
    result.backward()
    assert torch.equal(prediction.grad, torch.zeros(3))
    assert torch.isfinite(loss(torch.zeros(3)))
    assert torch.isfinite(loss(torch.empty(0), formulas=torch.empty(0, 2), mz=torch.empty(0)))


def test_auxiliary_loss_is_added_to_existing_objective():
    target = fixture()
    prediction = SimpleNamespace(sample_index=torch.tensor([0, 0, 0]),
        formula_tensor=target.target_formula, logit=torch.tensor([.999, .00025, .00075], requires_grad=True))
    objective = FragmentTreeIntensityTrainingLoss()
    base = objective(prediction, target)
    enhanced = objective(prediction, target, precursor_mz={0: [100.]}, predicted_mz=target.sample_peak_mz)
    assert enhanced > base
    assert objective.last_fragment_loss > 0
    enhanced.backward()
    assert torch.isfinite(prediction.logit.grad).all()


def test_sample_with_no_predicted_candidates_is_not_reported_as_perfect():
    prediction = SimpleNamespace(sample_index=torch.empty(0, dtype=torch.long),
        formula_tensor=torch.empty(0, 2), logit=torch.empty(0, requires_grad=True))
    objective = FragmentTreeIntensityTrainingLoss()
    result = objective(prediction, fixture(), precursor_mz={0: [100.]}, predicted_mz=torch.empty(0))
    assert torch.isfinite(result) and result >= 1
    result.backward()
