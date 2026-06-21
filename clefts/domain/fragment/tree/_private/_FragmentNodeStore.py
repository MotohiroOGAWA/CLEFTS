from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..FragmentNode import FragmentNode


@dataclass(frozen=True)
class _FragmentNodeStore:
    """Array-backed storage for fragment nodes.

    node_ids:
        Database fragment IDs.

    node_smiles:
        Canonical SMILES strings.

    The array position is the local node index in FragmentTree.
    """

    node_ids: np.ndarray
    node_smiles: np.ndarray

    def __post_init__(self) -> None:
        node_ids = np.asarray(self.node_ids, dtype=np.int64)
        node_smiles = np.asarray(self.node_smiles, dtype=object)

        if node_ids.ndim != 1:
            raise ValueError("node_ids must be a 1D array.")

        if node_smiles.ndim != 1:
            raise ValueError("node_smiles must be a 1D array.")

        if len(node_ids) != len(node_smiles):
            raise ValueError(
                "node_ids and node_smiles must have the same length."
            )

        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "node_smiles", node_smiles)

    @property
    def num_nodes(self) -> int:
        return len(self.node_smiles)

    def get_node(self, index: int) -> FragmentNode:
        if not 0 <= index < self.num_nodes:
            raise IndexError(f"Invalid node index: {index}")

        return FragmentNode(
            index=int(index),
            id=int(self.node_ids[index]),
            smiles=str(self.node_smiles[index]),
        )
    
    def get_node_by_smiles(self, smiles: str) -> FragmentNode | None:
        try:
            index = np.where(self.node_smiles == smiles)[0][0]
            return self.get_node(index)
        except IndexError:
            return None

    def copy(self) -> "_FragmentNodeStore":
        return _FragmentNodeStore(
            node_ids=self.node_ids.copy(),
            node_smiles=self.node_smiles.copy(),
        )