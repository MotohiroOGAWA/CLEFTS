from pathlib import Path

from clefts.ml.training.mol_training.pretraining_model import (
    nodes_in_open_closed_radius,
)
from clefts.ml.training.mol_training.training_model import (
    MolEncoderConfig,
    pretraining_stage_output_dir,
)


def _config() -> MolEncoderConfig:
    return MolEncoderConfig(
        node_dim=64,
        graph_dim=128,
        num_layers=4,
        num_heads=8,
        max_degree=16,
        max_spatial_dist=5,
        max_edge_dist=5,
        dropout=0.1,
    )


def test_context_radius_excludes_inner_boundary():
    distances = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert nodes_in_open_closed_radius(distances, 1, 4) == [2, 3, 4]


def test_single_config_writes_directly_to_output_dir():
    output_dir = Path("output")
    assert pretraining_stage_output_dir(output_dir, _config(), 1) == output_dir


def test_config_search_uses_runs_and_config_id():
    output_dir = Path("output")
    config = _config()
    assert pretraining_stage_output_dir(output_dir, config, 2) == (
        output_dir / "runs" / config.id
    )
