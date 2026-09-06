from types import SimpleNamespace
import json
from pathlib import Path

import torch
import torch.nn as nn

from clefts.ml.specgen.components.fragment_edge import ConditionEdgeScorer
from clefts.ml.specgen.components.fragment_edge.conditioned_fragment_edge_encoder import (
    StructuralEdgeEncoder,
)
from clefts.ml.specgen.fragment_tree_candidate_selector import FragmentTreeCandidateSelector
from clefts.ml.specgen.fragment_tree_feature_model import FragmentTreeFeatureModel
from clefts.ml.training.fragment_tree_training.model import (
    FragmentEdgeAbsoluteRankerTrainingLoss,
    FragmentTreeSelectionTrainingLoss,
    FragmentTreeTrainingModel,
    PairwiseEdgeIntensityRankingLoss,
)


def test_structural_edge_embedding_is_shared_across_conditions():
    edge_h = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scorer = ConditionEdgeScorer(edge_dim=2, condition_dim=2, interaction_dim=2)
    scorer(edge_h, torch.eye(2), torch.tensor([0, 0]), torch.tensor([0, 1]))
    assert torch.equal(edge_h, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_same_edge_can_receive_different_condition_scores():
    scorer = ConditionEdgeScorer(edge_dim=2, condition_dim=2, interaction_dim=2)
    with torch.no_grad():
        scorer.edge_projection.weight.copy_(torch.eye(2))
        scorer.condition_projection.weight.copy_(torch.eye(2))
    scores = scorer(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
    )
    assert scores[0] != scores[1]


def test_tree_isolation_scores_only_explicit_sample_edge_pairs():
    scorer = ConditionEdgeScorer(edge_dim=2, condition_dim=2, interaction_dim=2)
    scores = scorer(
        torch.randn(4, 2),
        torch.randn(2, 2),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 0, 1, 1]),
    )
    assert scores.shape == (4,)


def _ranking_fixture(intensities):
    batch = SimpleNamespace(
        edge_id_global=torch.tensor([0, 1]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        batch=torch.tensor([0, 0, 0]),
        kept_sample_ids=torch.tensor([0]),
    )
    output = SimpleNamespace(
        edge_absolute_logit=torch.tensor([1.0, -1.0]), sample_tree_batch=batch
    )
    target = SimpleNamespace(
        target_edge_index=torch.tensor([[0, 0], [0, 1]]),
        target_edge_group_index=torch.tensor([0, 1]),
        formula_peak_index=torch.tensor([0, 1]),
        sample_peak_intensity=torch.tensor(intensities),
    )
    return output, target


def test_ranking_threshold_ignores_nearly_equal_intensities():
    output, target = _ranking_fixture([1.0, 0.99])
    loss_fn = PairwiseEdgeIntensityRankingLoss(intensity_threshold=0.1)
    better, worse = loss_fn.build_pairs(output, target)
    assert better.numel() == worse.numel() == 0


def _three_group_ranking_fixture(logits):
    """One sample, three intensity-ordered groups (rank1 > rank2 > rank3),
    each explained by exactly one edge, and no unassigned background edges."""
    batch = SimpleNamespace(
        edge_id_global=torch.tensor([10, 11, 12]),
        edge_index=torch.tensor([[0, 0, 0], [1, 2, 3]]),
        batch=torch.tensor([0, 0, 0, 0]),
        kept_sample_ids=torch.tensor([0]),
    )
    output = SimpleNamespace(
        edge_absolute_logit=torch.tensor(logits), sample_tree_batch=batch
    )
    target = SimpleNamespace(
        target_edge_index=torch.tensor([[0, 0, 0], [10, 11, 12]]),
        target_edge_group_index=torch.tensor([0, 1, 2]),
        formula_peak_index=torch.tensor([0, 1, 2]),
        sample_peak_intensity=torch.tensor([3.0, 2.0, 1.0]),
    )
    return output, target


def test_ranking_weight_favors_higher_intensity_rank_over_raw_intensity():
    """A rank-1-vs-rank-2 inversion must outweigh an equally-sized
    rank-2-vs-rank-3 inversion, since the loss is now weighted by
    (1/rank)/sum(1/rank) rather than by the raw intensity value."""
    loss_fn = PairwiseEdgeIntensityRankingLoss()

    # rank1 loses to rank2 (bad); rank2 clearly beats rank3 (fine).
    high_rank_inversion, target_a = _three_group_ranking_fixture([-1.0, 1.0, -5.0])
    # rank1 clearly beats rank2 (fine); rank2 loses to rank3 (bad), same gap.
    low_rank_inversion, target_b = _three_group_ranking_fixture([5.0, -1.0, 1.0])

    loss_high_rank = loss_fn(high_rank_inversion, target_a)
    loss_low_rank = loss_fn(low_rank_inversion, target_b)

    assert loss_high_rank > loss_low_rank


class _Selector(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.ranking_loss_weight = 0.0
        self.feature_model = SimpleNamespace(main_adduct_types={})

    def forward(self, batch):
        return self.output


class _ConstantLoss(nn.Module):
    def forward(self, output, target):
        return output.edge_absolute_logit.sum() * 0.0 + 2.0


def test_zero_ranking_weight_removes_ranking_from_total_loss():
    output, target = _ranking_fixture([1.0, 0.0])
    model = FragmentTreeTrainingModel(_Selector(output), loss_fn=_ConstantLoss())
    result = model(target)
    assert torch.allclose(result["edge_total_loss"], result["edge_retain_loss"])


def test_edge_retain_metrics_break_down_by_adduct():
    batch = SimpleNamespace(
        edge_id_global=torch.tensor([100, 101, 102, 103]),
        edge_index=torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]]),
        batch=torch.tensor([0, 0, 1, 1]),
        kept_sample_ids=torch.tensor([0, 1]),
    )
    # sample 0: neither edge is correctly predicted (miss + false positive).
    # sample 1: both edges are correctly predicted.
    output = SimpleNamespace(
        edge_absolute_logit=torch.tensor([-1.0, 1.0, 1.0, 1.0]), sample_tree_batch=batch
    )
    target = SimpleNamespace(
        target_edge_index=torch.tensor([[0, 1, 1], [100, 102, 103]]),
        sample_adduct_type_index=torch.tensor([0, 1]),
        sample_ce_value=torch.tensor([10.0, 20.0]),
    )
    selector = _Selector(output)
    selector.feature_model.main_adduct_types = {0: "[M+H]+", 1: "[M+Na]+"}
    model = FragmentTreeTrainingModel(selector, loss_fn=_ConstantLoss())

    metrics = model._edge_retain_metrics(output, target)

    assert metrics["edge_retain_recall"] == 2 / 3
    assert metrics["by_adduct/[M+H]+/edge_retain_recall"] == 0.0
    assert metrics["by_adduct/[M+Na]+/edge_retain_recall"] == 1.0
    assert metrics["by_adduct/[M+H]+/edge_retain_precision"] == 0.0
    assert metrics["by_adduct/[M+Na]+/edge_retain_precision"] == 1.0
    assert metrics["by_adduct/[M+H]+/edge_retain_accuracy"] == 0.0
    assert metrics["by_adduct/[M+Na]+/edge_retain_accuracy"] == 1.0


def test_tree_edge_budget_metrics_reports_recall_and_sharing():
    # Two trees, one sample each (edge ids are disjoint per tree, matching
    # how from_structures collates them): tree0 = sample0 -> edges{0,1},
    # tree1 = sample1 -> edges{2,3}. Both trees exceed the budget of 1.
    output = SimpleNamespace(selected_edge_index=torch.tensor([0, 1, 3]))
    target = SimpleNamespace(
        num_edges=4,
        num_samples=2,
        tree_sample_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
        target_edge_index=torch.tensor([[1], [2]]),
        sample_edge_index=torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]]),
    )
    selector = SimpleNamespace(max_edges_per_tree=1)
    model = FragmentTreeTrainingModel.__new__(FragmentTreeTrainingModel)
    nn.Module.__init__(model)
    model.candidate_selector = selector

    metrics = model._tree_edge_budget_metrics(output, target)

    assert metrics["tree_edge_budget/selected_edge_count"] == 3.0
    assert metrics["tree_edge_budget/available_edge_count"] == 4.0
    assert metrics["tree_edge_budget/target_edge_recall"] == 0.0
    assert metrics["tree_edge_budget/trees_over_budget_fraction"] == 1.0
    assert metrics["tree_edge_budget/mean_samples_per_selected_edge"] == 1.0


def test_cleave_targets_come_from_stored_expand_paths():
    batch = SimpleNamespace(
        node_is_precursor_root=torch.tensor([True, False, False]),
        kept_sample_ids=torch.tensor([0]),
        batch=torch.tensor([0, 0, 0]),
        node_id_global=torch.tensor([0, 1, 2]),
    )
    output = SimpleNamespace(
        cleave_logit=torch.zeros(3),
        sample_tree_batch=batch,
    )
    target = SimpleNamespace(
        target_expand_node_index=torch.tensor([0, 1]),
        terminal_expand_ptr=torch.tensor([0, 2]),
        target_sample_index=torch.tensor([0]),
    )
    values, mask = FragmentTreeSelectionTrainingLoss()._build_cleave_targets(
        output, target, device=torch.device("cpu")
    )
    assert torch.equal(values, torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(mask, torch.tensor([False, True, True]))


def test_next_stage_limits_nodes_not_their_outgoing_edges():
    selector = FragmentTreeCandidateSelector.__new__(FragmentTreeCandidateSelector)
    nn.Module.__init__(selector)
    selector.max_next_cleavage_candidates = 3
    features = SimpleNamespace(
        structure=SimpleNamespace(
            # Node 1 has three children; selecting node 1 must not select only
            # a subset of those edges later.
            edge_index=torch.tensor([[1, 1, 1, 2, 3, 4], [6, 7, 8, 9, 10, 11]])
        )
    )
    batch = SimpleNamespace(
        node_is_precursor_root=torch.tensor([True, False, False, False, False, False]),
        kept_sample_ids=torch.tensor([0]),
        batch=torch.zeros(6, dtype=torch.long),
        node_id_global=torch.arange(6),
    )
    selected = selector._select_next_cleavage_candidates(
        features=features,
        sample_tree_batch=batch,
        cleave_logit=torch.tensor([0.0, 5.0, 4.0, 3.0, 2.0, 100.0]),
    )
    assert [item.global_node_id for item in selected] == [1, 2, 3]


def test_tree_bounded_edge_selection_respects_tree_boundaries_and_targets():
    """Two trees, budget=2/tree: each tree's own top-2 by importance is kept,
    a low-scoring but targeted edge still survives via the bonus, and no
    edge from one tree is ever selected for the other."""
    selector = FragmentTreeCandidateSelector.__new__(FragmentTreeCandidateSelector)
    nn.Module.__init__(selector)
    selector.max_edges_per_tree = 2

    class _FakeEdgeEncoder:
        def encode_base(self, features):
            return torch.zeros((6, 1)), None

    class _FakeConditionEncoder:
        def __call__(self, adduct_index, ce_value):
            return torch.zeros((adduct_index.numel(), 1))

    class _FakeFeatureModel:
        _condition_encoder = _FakeConditionEncoder()

        def build_node_features(self, structure):
            return SimpleNamespace()

    class _FakeConditionEdgeScorer:
        def __call__(self, edge_h_base, condition_h, edge_ids, sample_ids):
            # Deterministic: score equals the edge id, so higher edge id
            # within a tree wins on raw score alone (before any target bonus).
            return edge_ids.float()

    selector.feature_model = _FakeFeatureModel()
    selector.fragment_edge_encoder = _FakeEdgeEncoder()
    selector.condition_edge_scorer = _FakeConditionEdgeScorer()

    # tree 0 = sample 0, referencing edges 0,1,2 (three candidates for a
    # budget of two: edge 0 is a low-raw-score target that must survive via
    # the bonus, edge 1 is the lowest-scoring non-target and must be
    # excluded, edge 2 is the highest-scoring non-target and survives on
    # score alone). tree 1 = sample 1, referencing edges 3,4 (both fit under
    # budget, no exclusion needed). Edge 5 exists in edge_index but is never
    # referenced by any sample, so it is never even a candidate.
    structure = SimpleNamespace(
        edge_index=torch.zeros((2, 6), dtype=torch.long),
        tree_sample_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
        num_edges=6,
        num_samples=2,
        sample_adduct_type_index=torch.tensor([0, 0]),
        sample_ce_value=torch.tensor([10.0, 10.0]),
        sample_edge_index=torch.tensor([[0, 0, 0, 1, 1], [0, 1, 2, 3, 4]]),
        target_edge_index=torch.tensor([[0], [0]]),
    )

    selected = selector._tree_bounded_edge_selection(structure)

    assert sorted(selected.tolist()) == [0, 2, 3, 4]


def test_depth_budget_uses_the_matching_reaction_depth():
    assert FragmentTreeCandidateSelector._depth_budget((128, 64, 32), 1) == 64
    assert FragmentTreeCandidateSelector._depth_budget((128, 64, 32), 2) == 32


def test_edge_depth_budget_count_must_match_fragmenter_depth():
    preset = Path("clefts/presets/spectrum_generator_params/single_bond_pos_model_config.json")
    params = json.loads(preset.read_text())["probability_model_params"]
    params["fragment_edge_encoder_params"]["max_edges_per_depth"] = [128, 64]
    try:
        FragmentTreeFeatureModel(**params)
    except ValueError as exc:
        assert "fragmenter.tree_max_depth=3" in str(exc)
    else:
        raise AssertionError("Expected max_edges_per_depth length validation.")


def test_training_edge_sampler_keeps_one_alternative_and_zero_edges():
    encoder = StructuralEdgeEncoder.__new__(StructuralEdgeEncoder)
    nn.Module.__init__(encoder)
    encoder.training_edges_per_sample = 4
    encoder.training_zero_edge_fraction = 0.5
    structure = SimpleNamespace(
        num_samples=1,
        target_edge_index=torch.tensor([[0, 0], [0, 1]]),
        target_edge_group_index=torch.tensor([0, 0]),
        formula_peak_index=torch.tensor([0]),
        sample_peak_intensity=torch.tensor([1.0]),
        sample_edge_index=torch.tensor([[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]]),
    )
    torch.manual_seed(0)
    selected = encoder._sample_training_edges(
        structure, torch.tensor([0.0, 0.0, 5.0, 1.0, -1.0])
    )
    values = set(selected.tolist())
    assert len(values & {0, 1}) == 1
    assert 2 in values  # highest-scoring zero-intensity hard negative
    assert len(values) <= 4


def test_alternative_edge_group_uses_smooth_or_not_all_positive():
    loss_fn = FragmentEdgeAbsoluteRankerTrainingLoss()
    loss_fn.metrics = lambda edge_output, target: {}
    target = SimpleNamespace(
        target_edge_index=torch.tensor([[0, 0], [0, 1]]),
        target_edge_group_index=torch.tensor([0, 0]),
    )
    one_high = SimpleNamespace(absolute_score_logit=torch.tensor([5.0, -5.0, -5.0]))
    both_low = SimpleNamespace(absolute_score_logit=torch.tensor([-5.0, -5.0, -5.0]))
    assert loss_fn(one_high, target)[0] < loss_fn(both_low, target)[0]
