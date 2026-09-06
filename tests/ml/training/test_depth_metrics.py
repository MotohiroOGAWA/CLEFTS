from types import SimpleNamespace as NS

import pytest
import torch

from clefts.ml.training.fragment_tree_training.depth_metrics import DepthEvaluation
from clefts.ml.training.fragment_tree_training.metric_labels import tensorboard_metric_card
from clefts.ml.specgen.fragment_tree_candidate_selector import FragmentTreeCandidateSelector


def target():
    # depth 1: 0->1 (true), 0->2 (false); depth 2: 1->3 (true), 1->4 (false), 2->5 (true).
    return NS(
        num_nodes=6, edge_index=torch.tensor([[0, 0, 1, 1, 2], [1, 2, 3, 4, 5]]),
        target_edge_index=torch.tensor([[0, 0, 0], [0, 2, 4]]),
        target_edge_group_index=torch.tensor([0, 1, 1]),
        target_sample_index=torch.tensor([0, 0]),
        target_terminal_node_index=torch.tensor([3, 5]),
        target_expand_node_index=torch.tensor([0, 1, 0, 2]),
        terminal_expand_ptr=torch.tensor([0, 2, 4]),
        target_path_edge_index=torch.tensor([0, 2, 1, 4]),
        terminal_path_ptr=torch.tensor([0, 2, 4]),
    )


def output(edge_ids, scores, chosen=(1,), sample=0):
    data = target()
    pairs = torch.tensor([[sample] * len(edge_ids), edge_ids], dtype=torch.long)
    return NS(
        sample_tree_batch=NS(
            edge_index=data.edge_index[:, edge_ids], edge_id_global=torch.tensor(edge_ids),
            kept_sample_ids=torch.tensor([sample]), batch=torch.zeros(6, dtype=torch.long),
            node_id_global=torch.arange(6), node_is_precursor_root=torch.tensor([True, False, False, False, False, False]),
        ),
        edge_absolute_logit=torch.tensor(scores, dtype=torch.float),
        kept_candidates=[NS(sample_id=sample, global_node_id=3)],
        next_cleavage_candidates=[NS(sample_id=sample, global_node_id=n, score=float(10-i)) for i, n in enumerate(chosen)],
        features=NS(structure=NS(sample_edge_index=pairs)),
    )


def test_depth_edges_include_pruned_targets_and_alternative_groups():
    metrics = DepthEvaluation(target()).evaluate(output([0, 1, 2, 3], [1, 1, 1, -1]))
    assert metrics['edge_selection/depth_1/precision'] == 0.5
    assert metrics['edge_selection/depth_2/recall'] == 0.5  # missing edge 4 is not silently removed
    assert metrics['edge_selection/depth_2/conditional_recall'] == 1
    assert metrics['edge_selection/depth_2/target_group_recall'] == 1  # 2 OR 4 explains the group
    assert metrics['edge_selection/depth_2/false_negative_count'] == 1
    assert metrics['path_coverage/depth_2/complete_path_recall'] == 0.5
    assert metrics['fragment_selection/depth_2/precision'] == 1
    assert metrics['fragment_selection/depth_2/recall'] == 0.5
    assert metrics['next_cleavage/depth_1/precision'] == 1
    assert metrics['next_cleavage/depth_1/recall'] == 0.5
    assert metrics['next_cleavage/depth_1/top1_accuracy'] == 1


def test_expanded_edges_exclude_old_edges_but_count_budget_drops():
    evaluator = DepthEvaluation(target())
    before = output([0, 1], [1, 1])
    after = output([0, 1, 3], [1, 1, 1])  # selected parent 1; budget lost correct child edge 2
    metrics = evaluator.after_expansion(before, after)
    assert metrics['expanded_edge_selection/depth_2/precision'] == 0
    assert metrics['expanded_edge_selection/depth_2/recall'] == 0
    assert metrics['expanded_edge_selection/depth_2/target_count'] == 1  # excludes edge 4 from unchosen parent 2
    assert metrics['expanded_edge_selection/depth_2/false_positive_count'] == 1
    assert evaluator.after_expansion(before, before)['expanded_edge_selection/depth_2/false_negative_count'] == 1


def test_rollout_next_targets_only_include_pending_paths():
    evaluator = DepthEvaluation(target())
    after = output([0, 1, 2, 3], [1, 1, 1, -1], chosen=(1,))
    metrics = evaluator.evaluate(after, pending_only=True)
    assert metrics['next_cleavage/depth_1/target_count'] == 1  # node 2 still needs expansion
    assert metrics['next_cleavage/depth_1/precision'] == 0  # node 1 has already completed its path
    assert metrics['next_cleavage/depth_1/top1_accuracy'] == 0


def test_samples_are_not_conflated_and_empty_depth_has_no_perfect_recall():
    metrics = DepthEvaluation(target()).evaluate(output([0, 1], [1, -1], sample=1))
    assert metrics['edge_selection/depth_1/precision'] == 0
    assert metrics['edge_selection/depth_1/recall'] == 0
    assert metrics['edge_selection/depth_2/recall'] == 0
    assert metrics['edge_selection/depth_2/precision'] != metrics['edge_selection/depth_2/precision']  # NaN


@pytest.mark.parametrize('expanded', [True, False])
def test_progressive_observer_records_actual_outputs_including_stalled_step(expanded):
    first = output([0, 1], [1, 1])
    second = output([0, 1, 2], [1, 1, 1])
    fake = NS(
        feature_model=NS(build_features=lambda *a, **k: first.features),
        _initial_depth_features=lambda f: f,
        _forward_progressive=lambda f, **k: first if f is first.features else second,
        _expand_features_for_next_cleavage=lambda *a, **k: second.features if expanded else None,
    )
    seen = []
    result = FragmentTreeCandidateSelector.generate_depth_limited_candidates(
        fake, NS(), max_depth=1, on_step=lambda step, out: seen.append((step, out)),
    )
    assert seen == [(0, first), (1, second if expanded else first)]
    assert result is seen[-1][1]


def test_depth_cards_keep_stage_step_and_depth_separate():
    assert tensorboard_metric_card('metrics', 'rollout_step_1/expanded_edge_selection/depth_2/recall') == (
        'expanded_edge_selection/rollout_step_1/depth_2/recall/overall', '',
    )
    assert tensorboard_metric_card('metrics', 'next_cleavage/depth_1/precision')[0] == 'next_cleavage/depth_1/precision/overall'
