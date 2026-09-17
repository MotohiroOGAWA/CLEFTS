from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FragmentTreeFormulaIntensityPredictor(nn.Module):
    """Predict a merged-formula intensity from its candidate ions' scores."""

    def __init__(self, formula_dim: int, *, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.formula_dim = int(formula_dim)
        self.hidden_dim = int(hidden_dim or 128)
        self.net = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, formula_tensor: Tensor, group_score: Tensor, group_count: Optional[Tensor] = None) -> Tensor:
        if formula_tensor.dim() != 2:
            raise ValueError("formula_tensor must be 2D.")
        if group_score.dim() == 1:
            group_score = group_score[:, None]
        if group_count is None:
            group_count = torch.ones_like(group_score)
        elif group_count.dim() == 1:
            group_count = group_count[:, None]
        # Keep formula_tensor in the public signature for compatibility, but
        # never expose formula identity/composition to the intensity network.
        x = torch.cat([group_score.float(), group_count.float()], dim=-1)
        return F.softplus(self.net(x).squeeze(-1))
