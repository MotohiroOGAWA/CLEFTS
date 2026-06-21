from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch

from .fragment_tree_structure import FragmentTreeStructure


@dataclass(frozen=True)
class FragmentTreeFeatures:
    node_graphs: Batch
    edge_attr: Tensor # [E, A]
    structure: FragmentTreeStructure

    @property
    def mol_x(self) -> Tensor:
        return self.node_graphs.embeddings
    
    @property
    def device(self) -> torch.device:
        return self.structure.device

    @classmethod
    def from_structure(
        cls,
        structure: FragmentTreeStructure,
        node_graphs: Batch,
        edge_attr: Optional[Tensor] = None,
    ) -> FragmentTreeFeatures:
        E = structure.num_edges
        if edge_attr is None:
            edge_attr = torch.zeros((E, 0), dtype=torch.float32, device=structure.edge_index.device)
        return cls(
            node_graphs=node_graphs,
            edge_attr=edge_attr,
            structure=structure,
        )

    

