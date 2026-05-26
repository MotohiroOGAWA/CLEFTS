from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..FragmentNode import FragmentNode


@dataclass(frozen=True, init=False)
class _FragmentNodeStore:
    """Array-backed storage for fragment nodes."""

    _node_smiles: np.ndarray

    def __init__(self, node_smiles: np.ndarray):
        """Create node storage from a node SMILES array."""
        node_smiles = np.asarray(node_smiles, dtype=object)
        assert node_smiles.ndim == 1, "node_smiles must be a 1D array."
        object.__setattr__(self, "_node_smiles", node_smiles)

    @property
    def node_smiles(self) -> np.ndarray:
        """Node SMILES array. The array index is the node ID."""
        return self._node_smiles

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self._node_smiles)

    def get_node(self, node_id: int) -> FragmentNode:
        """Return a node object by ID."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID."
        return FragmentNode(id=int(node_id), smiles=str(self._node_smiles[node_id]))

    def copy(self) -> "_FragmentNodeStore":
        """Return copied node storage."""
        return _FragmentNodeStore(node_smiles=self.node_smiles.copy())
