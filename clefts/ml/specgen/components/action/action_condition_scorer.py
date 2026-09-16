"""Condition-specific scores without recomputing static action embeddings."""
from __future__ import annotations
from torch import Tensor, nn


class ActionConditionScorer(nn.Module):
    def __init__(self, action_dim: int, condition_dim: int, interaction_dim: int) -> None:
        super().__init__()
        self.base = nn.Linear(action_dim, 1)
        self.action_q = nn.Linear(action_dim, interaction_dim, bias=False)
        self.condition_q = nn.Linear(condition_dim, interaction_dim, bias=False)
        self.scale = interaction_dim ** -0.5

    def forward(self, action_h: Tensor, condition_h: Tensor) -> Tensor:
        return self.base(action_h).squeeze(-1)[None, :] + self.condition_q(condition_h) @ self.action_q(action_h).T * self.scale
