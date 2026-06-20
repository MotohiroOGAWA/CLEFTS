from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .fragment_tree_structure import FragmentTreeStructure


@dataclass(frozen=True)
class TrainingFragmentTreeStructure(FragmentTreeStructure):
    """FragmentTreeStructure with supervised targets for generation training."""

    target_node_keep: Tensor
    # [N] 1 when the fragment node should remain as an emitted candidate.

    target_node_expand: Tensor
    # [N] 1 when the node should be expanded in the next cleavage step.

    target_ion_index: Tensor
    # [T] Flat ion target indexes aligned with target_node_index.

    target_unsaturation_index: Tensor
    # [T] Flat unsaturation target indexes aligned with target_node_index.

    target_radical_index: Tensor
    # [T] Flat radical target indexes aligned with target_node_index.

    target_node_index: Tensor
    # [T] Node indexes for ion/unsaturation/radical targets.

    target_formula: Tensor
    # [T, F] Target formula tensors used for formula-group coverage losses.

    target_intensity: Tensor
    # [T] Optional normalized target intensity per target formula row.
