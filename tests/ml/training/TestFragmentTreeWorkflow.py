from clefts.ml.training.fragment_tree_training.workflow import (
    balanced_sample_chunks,
    pack_compound_chunks,
)
from clefts.ml.training.fragment_tree_training.model import (
    PairwiseEdgeIntensityRankingLoss,
)
from clefts.ml.common.progress import (
    TQDM_NCOLS,
    advance_edge_progress,
    fixed_tqdm,
    iteration_edge_progress,
)
from types import SimpleNamespace
from io import StringIO
import torch
import torch.nn as nn
from clefts.ml.training.fragment_tree_training.training_model import run_epoch
from clefts.ml.training.fragment_tree_training.performance_profile import (
    run_training_performance_profile,
)


def test_balanced_sample_chunks_avoids_tiny_tail():
    assert balanced_sample_chunks(201, 100) == [(0, 67), (67, 134), (134, 201)]


def test_pack_compound_chunks_respects_limit_and_preserves_samples():
    batches = pack_compound_chunks([30, 100, 201], 100)
    assert all(sum(stop - start for _, start, stop in batch) <= 100 for batch in batches)
    assert sum(stop - start for batch in batches for _, start, stop in batch) == 331


def test_pairwise_edge_loss_prefers_the_more_intense_peak():
    batch = SimpleNamespace(
        edge_id_global=torch.tensor([10, 11]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        batch=torch.tensor([0, 0, 0]),
        kept_sample_ids=torch.tensor([0]),
    )
    target = SimpleNamespace(
        target_edge_index=torch.tensor([[0, 0], [10, 11]]),
        target_edge_group_index=torch.tensor([0, 1]),
        formula_peak_index=torch.tensor([0, 1]),
        sample_peak_intensity=torch.tensor([1.0, 0.1]),
    )
    good = SimpleNamespace(
        edge_absolute_logit=torch.tensor([2.0, -1.0]), sample_tree_batch=batch
    )
    bad = SimpleNamespace(
        edge_absolute_logit=torch.tensor([-1.0, 2.0]), sample_tree_batch=batch
    )
    loss_fn = PairwiseEdgeIntensityRankingLoss()
    assert loss_fn(good, target) < loss_fn(bad, target)


def test_progress_bars_have_fixed_width_and_nested_edge_updates():
    bar = fixed_tqdm(range(1), position=0, leave=False, file=StringIO())
    assert bar.ncols == TQDM_NCOLS
    assert bar.dynamic_ncols is False
    bar.close()
    with iteration_edge_progress(2, desc="edges"):
        advance_edge_progress()
        advance_edge_progress()


def test_train_metrics_callback_runs_every_fifty_successful_steps():
    class Structure:
        num_samples = 1
        edge_index = torch.empty((2, 0), dtype=torch.long)
        sample_edge_index = torch.empty((2, 0), dtype=torch.long)

        def to(self, device):
            return self

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.candidate_selector = SimpleNamespace(max_edges_per_step=1)

        def forward(self, structure):
            loss = self.weight.square()
            return {
                "loss": loss,
                "selection_loss": loss,
                "intensity_loss": loss * 0,
                "absolute_ranker_metrics": {"pairwise_ranking_accuracy": 1.0},
            }

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    logged = []
    metrics = run_epoch(
        model=model,
        loader=[{"structure": Structure()} for _ in range(108)],
        device=torch.device("cpu"),
        optimizer=optimizer,
        desc="test",
        train_log_interval_steps=50,
        on_train_log_step=lambda step, cumulative, window: logged.append(
            (
                step,
                cumulative.absolute_ranker_summary[
                    "pairwise_ranking_accuracy_mean"
                ],
                window.absolute_ranker_summary[
                    "pairwise_ranking_accuracy_mean"
                ],
            )
        ),
    )
    assert metrics.steps == 108
    assert logged == [(50, 1.0, 1.0), (100, 1.0, 1.0)]


def test_performance_profile_writes_total_module_and_operator_reports(tmp_path):
    class Structure:
        num_samples = 2
        num_nodes = 3
        edge_index = torch.tensor([[0, 1], [1, 2]])
        sample_edge_index = torch.tensor([[0, 1], [0, 1]])

        def to(self, device):
            return self

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, structure):
            return {"loss": self.weight.square()}

    report = run_training_performance_profile(
        model=Model(),
        loader=[{"structure": Structure()}],
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    assert report["status"] == "ok"
    assert report["batch_shape"]["nodes"] == 3
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "modules.json").exists()
    assert (tmp_path / "operators.json").exists()
    assert (tmp_path / "operator_table.txt").exists()
    assert (tmp_path / "trace.json").exists()
