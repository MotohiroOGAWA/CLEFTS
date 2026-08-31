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
