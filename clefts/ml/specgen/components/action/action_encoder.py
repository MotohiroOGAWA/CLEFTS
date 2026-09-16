"""Condition-independent action embeddings from Original Source only."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from ....common.layers.set_transformer import SetTransformer


class ActionEncoder(nn.Module):
    def __init__(self, atom_dim: int, mol_dim: int, hidden_dim: int,
                 category_sizes: tuple[int, int, int], num_heads: int = 4,
                 max_roles: int = 64) -> None:
        super().__init__()
        self.categories = nn.ModuleList(nn.Embedding(size, hidden_dim) for size in category_sizes)
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        # SMARTS query roles are chemical features, never action-list positions.
        self.role_status = nn.Linear(6,hidden_dim,bias=False)
        self.roles = nn.Embedding(max_roles, hidden_dim)
        self.atom_set = SetTransformer(hidden_dim, hidden_dim, hidden_dim, num_heads=num_heads, num_layers=1)
        self.context = nn.Linear(mol_dim + hidden_dim + 6, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, *, source_atom_h: Tensor, source_mol_h: Tensor, action_type: Tensor,
                action_tree_index: Tensor, action_source_atom_ptr: Tensor,
                action_source_atom_index: Tensor, action_static_features: Tensor,
                action_source_atom_features: Tensor | None = None) -> Tensor:
        a = action_type.shape[0]
        if a == 0:
            return source_atom_h.new_empty((0, self.output[0].out_features))
        counts = action_source_atom_ptr[1:] - action_source_atom_ptr[:-1]
        owner = torch.repeat_interleave(torch.arange(a, device=counts.device), counts)
        role = torch.arange(owner.numel(), device=counts.device) - action_source_atom_ptr[owner]
        # Query role identity is independent of the order of actions.
        selected = self.atom_projection(source_atom_h[action_source_atom_index]) + self.roles(role.clamp_max(self.roles.num_embeddings - 1))
        if action_source_atom_features is not None:
            selected = selected + self.role_status(action_source_atom_features)
        width = self.roles.num_embeddings  # bounded packing; preprocessing validates role count
        slot = torch.arange(width, device=counts.device)[None, :]
        index = action_source_atom_ptr[:-1, None] + slot
        valid = slot < counts[:, None]
        padded = selected[index.clamp_max(max(selected.shape[0] - 1, 0))].masked_fill(~valid[:, :, None], 0)
        atoms = self.atom_set(padded, key_padding_mask=~valid)
        category = sum(embedding(action_type[:, i]) for i, embedding in enumerate(self.categories))
        context = self.context(torch.cat((source_mol_h[action_tree_index], category, action_static_features), dim=1))
        return self.output(torch.cat((atoms, context), dim=1))
