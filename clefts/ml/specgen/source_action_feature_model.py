"""Selection features have no dependency on materialized target graphs."""
from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor, nn
from .components.condition.condition_encoder import MS2ConditionEncoder
from .components.action.action_encoder import ActionEncoder
from .components.action.action_condition_scorer import ActionConditionScorer
from .components.action.action_decoder import ActionPool, ActionSequenceDecoder, build_action_pool


@dataclass(frozen=True)
class SourceActionFeatures:
    action_h: Tensor
    condition_h: Tensor
    absolute_logits: Tensor
    pool: ActionPool


class SourceActionFeatureModel(nn.Module):
    def __init__(self, mol_encoder: nn.Module, category_sizes: tuple[int, int, int],
                 condition_feature_dim: int, hidden_dim: int = 128, condition_dim: int = 128,
                 max_action_count: int = 3, action_prefilter_top_k: int = 64,
                 action_prefilter_max_k: int = 128, action_prefilter_threshold_logit: float = 0.0,
                 beam_size: int = 32, num_heads: int = 4, max_decode_steps: int = 16,
                 condition_encoder: nn.Module | None = None, max_roles: int = 64) -> None:
        super().__init__()
        if not 1 <= action_prefilter_top_k <= action_prefilter_max_k <= 128:
            raise ValueError("Require 1 <= action_prefilter_top_k <= action_prefilter_max_k <= 128")
        self.mol_encoder = mol_encoder
        self.action_encoder = ActionEncoder(mol_encoder.node_dim, mol_encoder.graph_dim, hidden_dim, category_sizes, num_heads,max_roles)
        self.condition_encoder = condition_encoder or nn.Sequential(nn.Linear(condition_feature_dim, condition_dim), nn.GELU(), nn.Linear(condition_dim, condition_dim))
        self.scorer = ActionConditionScorer(hidden_dim, condition_dim, hidden_dim)
        self.decoder = ActionSequenceDecoder(hidden_dim, condition_dim, max_action_count, beam_size, num_heads, max_decode_steps)
        self.top_k, self.max_k, self.threshold = action_prefilter_top_k, action_prefilter_max_k, action_prefilter_threshold_logit
        self.max_action_count = max_action_count

    def freeze_mol_encoder(self) -> None:
        self._freeze_mol_encoder = True
        self.mol_encoder.eval()
        for parameter in self.mol_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_mol_encoder", False):
            self.mol_encoder.eval()
        return self

    def forward(self, data: object, training_pool: bool = False) -> SourceActionFeatures:
        if data.max_action_role_count > self.action_encoder.roles.num_embeddings:
            raise ValueError("Configure max_roles for the largest SMARTS query in the action universe")
        source = self.mol_encoder(data.source_graph)
        action = self.action_encoder(source_atom_h=source.x, source_mol_h=source.embeddings,
            action_type=data.action_type, action_tree_index=data.action_tree_index,
            action_source_atom_ptr=data.action_source_atom_ptr, action_source_atom_index=data.action_source_atom_index,
            action_static_features=data.action_static_features,action_source_atom_features=data.action_source_atom_features)
        condition = (self.condition_encoder(adduct_idx=data.condition_features[:, 0].long(), collision_energy=data.condition_features[:, 1])
                     if isinstance(self.condition_encoder, MS2ConditionEncoder) else self.condition_encoder(data.condition_features))
        absolute = self.scorer(action, condition)
        pool = build_action_pool(action, absolute, data, top_k=self.top_k, max_k=self.max_k,
                                 threshold=self.threshold, training=training_pool, max_action_count=self.max_action_count)
        return SourceActionFeatures(action, condition, absolute, pool)
