from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn.aggr import AttentionalAggregation

from ...libs.mmkit.mmkit import Compound
from .graph_builder import MolGraphBuilder

from ..common.layers.graphormer import GraphormerEncoder

class MolEncoder(nn.Module):
    """
    MolEncoder based on GraphormerEncoder.

    - GraphormerEncoder returns:
        node_h: [N_total, dim]
        graph_h: [num_graphs, dim] if add_virtual_node=True else None
    """

    def __init__(
        self,
        symbols,
        node_dim: int,
        graph_dim: int,
        num_layers: int,
        # Graphormer-specific
        num_heads: int,
        max_degree: int = 4,
        max_spatial_dist: int = 3,
        max_edge_dist: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- Pre-graph builder (no nn.Module needed) ---
        self.graph_builder = MolGraphBuilder(symbols=symbols)

        atom_dim = self.graph_builder.atom_dim
        bond_dim = self.graph_builder.bond_dim

        self._node_dim = node_dim
        self._graph_dim = graph_dim

        self.encoder = GraphormerEncoder(
            in_dim=atom_dim,
            dim=node_dim,
            graph_dim=graph_dim,
            edge_dim=bond_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_degree=max_degree,
            max_spatial_dist=max_spatial_dist,
            max_edge_dist=max_edge_dist,
            dropout=dropout,
            add_virtual_node=True,
            undirected_for_spd=True,
            undirected_for_path=True,
            start_cap=64,
            cap_growth=2.0,
        )
    
    @property
    def node_dim(self) -> int:
        return self._node_dim
    
    @property
    def graph_dim(self) -> int:
        return self._graph_dim

    def encode_components(self, compound: Compound) -> Data:
        """
        Convert one Compound into a PyG Data object (features only).
        Device is aligned to this model's parameters.
        """
        device = next(self.parameters()).device
        return self.graph_builder.build(compound, device=device)

    def forward(self, batch: Batch) -> Batch:
        """
        Forward for a PyG Batch.
        """
        node_h, graph_h = self.encoder(batch)

        batch.x = node_h
        if graph_h is not None:
            batch.embeddings = graph_h  # [G, graph_dim]

        return batch

    def _make_empty_data(self, compound: Compound, *, device: torch.device) -> Data:
        """Create an empty graph that will not affect training."""
        atom_dim = self.graph_builder.atom_dim
        bond_dim = self.graph_builder.bond_dim

        x = torch.empty((0, atom_dim), device=device, dtype=torch.float32)
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        edge_attr = torch.empty((0, bond_dim), device=device, dtype=torch.float32)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            compound=compound,
        )

    def encode(self, compound: Compound) -> Optional[Data]:
        """
        Encode a single compound into a Data object.
        """
        try:
            data = self.encode_components(compound)
            batch = Batch.from_data_list([data])
            batch = self.forward(batch)

            n = data.x.size(0)
            data.x = batch.x[:n]                  # [num_atoms, node_dim]
            data.embedding = batch.embeddings[0]  # [node_dim]
            data.compound = compound
            return data
        except Exception:
            return None

    def encode_batch(self, compounds: List[Compound]) -> Tuple[Optional[Batch], Optional[torch.Tensor]]:
        """
        Encode multiple compounds into a Batch object.
        """
        data_list: List[Data] = []
        valid_indices: List[int] = []

        device = next(self.parameters()).device

        for i, c in enumerate(compounds):
            try:
                data = self.graph_builder.build(c, device=device)
                data_list.append(data)
                valid_indices.append(i)
            except Exception:
                # If you want to keep alignment, append empty data here.
                # data_list.append(self._make_empty_data(c, device=device))
                pass

        if len(valid_indices) == 0:
            return None, None

        valid_indices_t = torch.tensor(valid_indices, dtype=torch.long, device=device)

        batch = Batch.from_data_list(data_list)
        batch = self.forward(batch)

        return batch, valid_indices_t

    @property
    def atom_dim(self) -> int:
        return self.graph_builder.atom_dim

    @property
    def bond_dim(self) -> int:
        return self.graph_builder.bond_dim
    
    @property
    def symbols(self) -> Tuple[str]:
        return self.graph_builder.symbols