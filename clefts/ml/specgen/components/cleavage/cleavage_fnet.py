from typing import Tuple, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from clefts.domain.fragment.cleavage import CleavagePattern
from ....common.torch_utils.nn_utils import build_fc_layers

@dataclass(frozen=True)
class CleavageFNetInput:
    """
    Input container for CleavageFNet.

    Shapes:
      reactant_mol:       [B, mol_dim]
      reactant_atom_feats:[B, num_react_atoms, node_dim]
      product_mol:        [B, mol_dim]
      product_atom_feats: [B, num_prod_atoms, node_dim]
    """
    reactant_mol: torch.Tensor
    reactant_atom_feats: torch.Tensor
    product_mol: torch.Tensor
    product_atom_feats: torch.Tensor

    def __post_init__(self):
        self.size()

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "CleavageFNetInput":
        """Move all tensors to device/dtype (like nn.Module.to)."""
        def _move(x: torch.Tensor) -> torch.Tensor:
            y = x
            if device is not None:
                y = y.to(device=device, non_blocking=non_blocking)
            if dtype is not None:
                y = y.to(dtype=dtype)
            return y

        return CleavageFNetInput(
            reactant_mol=_move(self.reactant_mol),
            reactant_atom_feats=_move(self.reactant_atom_feats),
            product_mol=_move(self.product_mol),
            product_atom_feats=_move(self.product_atom_feats),
        )

    @property
    def batch_size(self) -> int:
        return int(self.reactant_mol.shape[0])

    def size(self) -> int:
        """Basic consistency checks (batch sizes, dims)."""
        B = self.batch_size
        if self.product_mol.shape[0] != B:
            raise ValueError("Batch size mismatch: reactant_mol vs product_mol")
        if self.reactant_atom_feats.shape[0] != B:
            raise ValueError("Batch size mismatch: reactant_mol vs reactant_atom_feats")
        if self.product_atom_feats.shape[0] != B:
            raise ValueError("Batch size mismatch: reactant_mol vs product_atom_feats")
        return B
        
class CleavageFNet(nn.Module):
    """
    Encode cleavage events using both molecule- and atom-level embeddings.
    Supports batched input.

    Input tensors:
        reactant_mol: [B, mol_dim]
        reactant_atom_feats: [B, num_react_atoms, node_dim]
        product_mol: [B, mol_dim]
        product_atom_feats: [B, num_prod_atoms, node_dim]

    Output:
        cleavage_feat: [B, cleavage_dim]
    """

    def __init__(
        self,
        pattern_id: int,
        reaction_id: int,
        product_molecule_id: int,
        num_reactant_atoms: int,
        num_product_atoms: int,
        feature_dim: int,
        mol_dim: int,
        atom_dim: int,
        fc_dims: Tuple[int, ...],
        dropout: float,
    ):
        super(CleavageFNet, self).__init__()

        self._pattern_id = pattern_id
        self._reaction_id = reaction_id
        self._product_molecule_id = product_molecule_id
        self._num_reactant_atoms = num_reactant_atoms
        self._num_product_atoms = num_product_atoms

        self._feature_dim = feature_dim
        self._mol_dim = mol_dim
        self._atom_dim = atom_dim

        self.fnet = build_fc_layers(
            input_dim=self.input_dim,
            fc_dims=fc_dims,
            output_dim=feature_dim,
            dropout=dropout,
        )

    @property
    def pattern_id(self) -> int:
        return self._pattern_id
    @property
    def reaction_id(self) -> int:
        return self._reaction_id
    @property
    def product_molecule_id(self) -> int:
        return self._product_molecule_id
    @property
    def num_reactant_atoms(self) -> int:
        return self._num_reactant_atoms
    @property
    def num_product_atoms(self) -> int:
        return self._num_product_atoms
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    @property
    def input_dim(self) -> int:
        return (
            self._mol_dim * 2 +
            self._atom_dim * (self._num_reactant_atoms + self._num_product_atoms)
        )
        
    def forward(self, batch: "CleavageFNetInput") -> torch.Tensor:
        """
        Batched forward encoding of cleavage features.

        Args:
            batch: CleavageFNetInput

        Returns:
            torch.Tensor: [B, feature_dim]
        """
        batch.size()
        return self.encode_cleavage(
            reactant_mol_feats=batch.reactant_mol,
            reactant_atom_feats=batch.reactant_atom_feats,
            product_mol_feats=batch.product_mol,
            product_atom_feats=batch.product_atom_feats,
        )

    def encode_cleavage(
        self,
        reactant_mol_feats: torch.Tensor,
        reactant_atom_feats: torch.Tensor,
        product_mol_feats: torch.Tensor,
        product_atom_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode cleavage features from both reactant and product representations.

        Args:
            reactant_mol_feats (torch.Tensor): [B, mol_dim]
            reactant_atom_feats (torch.Tensor): [B, num_react_atoms, node_dim]
            product_mol_feats (torch.Tensor): [B, mol_dim]
            product_atom_feats (torch.Tensor): [B, num_prod_atoms, node_dim]

        Returns:
            torch.Tensor: [B, feature_dim]
        """
        # --- Basic shape validation ---
        assert reactant_atom_feats.shape[-2:] == (self.num_reactant_atoms, self._atom_dim), \
            f"Expected reactant feats shape (*, {self.num_reactant_atoms}, {self._atom_dim}), got {reactant_atom_feats.shape[-2:]}"
        assert product_atom_feats.shape[-2:] == (self.num_product_atoms, self._atom_dim), \
            f"Expected product feats shape (*, {self.num_product_atoms}, {self._atom_dim}), got {product_atom_feats.shape[-2:]}"

        B = reactant_atom_feats.shape[0]

        # --- Flatten atom features ---
        flat_reactant = reactant_atom_feats.reshape(B, -1)  # [B, num_react_atoms * node_dim]
        flat_product = product_atom_feats.reshape(B, -1)    # [B, num_prod_atoms * node_dim]

        # --- Concatenate all molecular and atomic features ---
        concat_input = torch.cat(
            [reactant_mol_feats, product_mol_feats, flat_reactant, flat_product],
            dim=-1
        )  # [B, input_dim]

        # --- Encode cleavage ---
        cleavage_feat = self.fnet(concat_input)  # [B, feature_dim]

        return cleavage_feat
    
