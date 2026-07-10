from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

from clefts.domain.fragment.cleavage import CleavagePatternSet
from .cleavage_fnet import CleavageFNetInput, CleavageFNet
from ....input.fragment_tree_features import FragmentTreeFeatures
from ....input.fragment_tree_structure import FragmentTreeStructure


class CleavageEdgeFeatureNet(nn.Module):
    """
    Encode cleavage-event features for fragment-tree edges.

    This module groups cleavage events by
    (cleavage_pattern_id, reaction_id, product_molecule_id) and encodes each
    event with the corresponding CleavageFNet. ``encode_events`` returns one
    row per cleavage event so callers can distinguish multiple events on the
    same source-target fragment-tree edge. ``forward`` keeps a legacy
    edge-level view by deterministic mean pooling, but no learnable aggregation
    model is used.
    """

    def __init__(
        self,
        cleavage_pattern_set_params: Dict,
        feature_dim: int,
        mol_dim: int,
        atom_dim: int,
        fc_dims: Tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._feature_dim = int(feature_dim)
        self._mol_dim = int(mol_dim)
        self._atom_dim = int(atom_dim)
        self._fc_dims = tuple(int(v) for v in fc_dims)
        self._dropout = float(dropout)
        self._cleavage_pattern_set = CleavagePatternSet.from_dict(
            cleavage_pattern_set_params
        )

        cleavage_fnet_dict = nn.ModuleDict()

        event_type_by_module_key: Dict[str, Tuple[int, int, int]] = {}
        module_key_by_event_type: Dict[Tuple[int, int, int], str] = {}

        reactant_tuple_length_by_event_type: Dict[Tuple[int, int], int] = {}
        product_tuple_length_by_event_type: Dict[Tuple[int, int, int], int] = {}

        for pattern in self._cleavage_pattern_set.patterns:
            pattern_id = int(pattern.pattern_id)

            for reaction in pattern.cleavage_reactions:
                reaction_id = int(reaction.id)
                num_reactant_atoms = len(reaction.react_idx_to_map)

                reactant_key = (
                    pattern_id,
                    reaction_id,
                )

                if reactant_key in reactant_tuple_length_by_event_type:
                    prev = reactant_tuple_length_by_event_type[reactant_key]

                    if prev != num_reactant_atoms:
                        raise ValueError(
                            "Inconsistent reactant tuple length for "
                            f"(pattern_id, reaction_id)={reactant_key}: "
                            f"{prev} vs {num_reactant_atoms}."
                        )

                reactant_tuple_length_by_event_type[reactant_key] = (
                    num_reactant_atoms
                )

                for product_molecule_id, prod_idx_to_map in enumerate(
                    reaction.prod_idx_to_maps
                ):
                    product_molecule_id = int(product_molecule_id)
                    num_product_atoms = len(prod_idx_to_map)

                    event_type = (
                        pattern_id,
                        reaction_id,
                        product_molecule_id,
                    )

                    module_key = self._make_cleavage_fnet_key(
                        cleavage_pattern_id=pattern_id,
                        reaction_id=reaction_id,
                        product_molecule_id=product_molecule_id,
                    )

                    if module_key in cleavage_fnet_dict:
                        raise ValueError(
                            "Duplicated CleavageFNet module key: "
                            f"{module_key} for event_type={event_type}."
                        )

                    product_tuple_length_by_event_type[event_type] = (
                        num_product_atoms
                    )

                    fnet = CleavageFNet(
                        pattern_id=pattern_id,
                        reaction_id=reaction_id,
                        product_molecule_id=product_molecule_id,
                        num_reactant_atoms=num_reactant_atoms,
                        num_product_atoms=num_product_atoms,
                        feature_dim=feature_dim,
                        mol_dim=mol_dim,
                        atom_dim=atom_dim,
                        fc_dims=fc_dims,
                        dropout=dropout,
                    )

                    cleavage_fnet_dict[module_key] = fnet
                    event_type_by_module_key[module_key] = event_type
                    module_key_by_event_type[event_type] = module_key

        self.cleavage_fnet_dict = cleavage_fnet_dict

        self.event_type_by_module_key = event_type_by_module_key
        self.module_key_by_event_type = module_key_by_event_type

        self.reactant_tuple_length_by_event_type = (
            reactant_tuple_length_by_event_type
        )
        self.product_tuple_length_by_event_type = (
            product_tuple_length_by_event_type
        )


    @staticmethod
    def _make_cleavage_fnet_key(
        *,
        cleavage_pattern_id: int,
        reaction_id: int,
        product_molecule_id: int,
    ) -> str:
        """
        Make a safe string key for nn.ModuleDict.

        ModuleDict keys must be strings and should not contain ".".
        """
        return (
            f"p{int(cleavage_pattern_id)}"
            f"__r{int(reaction_id)}"
            f"__m{int(product_molecule_id)}"
        )

    @property
    def num_cleavage_fnets(self) -> int:
        return len(self.cleavage_fnet_dict)

    @property
    def num_cleavage_patterns(self) -> int:
        return len(
            {
                event_type[0]
                for event_type in self.module_key_by_event_type.keys()
            }
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def mol_dim(self) -> int:
        return self._mol_dim

    @property
    def atom_dim(self) -> int:
        return self._atom_dim

    @property
    def fc_dims(self) -> Tuple[int, ...]:
        return self._fc_dims

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def cleavage_pattern_set(self) -> CleavagePatternSet:
        return self._cleavage_pattern_set.copy()

    @property
    def cleavage_pattern_set_params(self) -> Dict:
        return self._cleavage_pattern_set.to_dict()

    def config_dict(self) -> Dict:
        return {
            "cleavage_pattern_set_params": self.cleavage_pattern_set_params,
            "feature_dim": self.feature_dim,
            "mol_dim": self.mol_dim,
            "atom_dim": self.atom_dim,
            "fc_dims": tuple(self.fc_dims),
            "dropout": self.dropout,
        }

    @property
    def event_types(self) -> Tuple[Tuple[int, int, int], ...]:
        return tuple(self.module_key_by_event_type.keys())

    def encode_events(
        self,
        ft_features: FragmentTreeFeatures,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode cleavage events without merging events on the same edge.

        Returns
        -------
        event_row_id:
            [M] row indexes into ``structure.cleavage_event``.

        event_edge_id:
            [M] fragment-tree edge index for each event.

        event_attr:
            [M, feature_dim] event-level cleavage embeddings.
        """
        structure = ft_features.structure
        all_event_row_ids = []
        all_event_edge_ids = []
        all_event_feats = []

        for event_row_id, event_edge_id, event_type, cleavage_fnet_input in (
            self._iter_cleavage_fnet_batches(ft_features)
        ):
            if event_edge_id.size(0) != cleavage_fnet_input.batch_size:
                raise ValueError(
                    "event_edge_id and cleavage_fnet_input size mismatch: "
                    f"event_edge_id.size(0)={event_edge_id.size(0)}, "
                    f"cleavage_fnet_input.batch_size="
                    f"{cleavage_fnet_input.batch_size}."
                )

            if event_edge_id.size(0) == 0:
                continue

            module_key = self._get_cleavage_fnet_key(event_type)
            event_feat = self.cleavage_fnet_dict[module_key](
                cleavage_fnet_input
            )
            # [M_g, feature_dim]

            all_event_row_ids.append(event_row_id.long())
            all_event_edge_ids.append(event_edge_id.long())
            all_event_feats.append(event_feat)

        if len(all_event_feats) == 0:
            empty_ids = torch.empty(
                (0,),
                dtype=torch.long,
                device=structure.edge_index.device,
            )
            empty_attr = torch.empty(
                (0, self.feature_dim),
                dtype=ft_features.mol_x.dtype,
                device=structure.edge_index.device,
            )
            return empty_ids, empty_ids, empty_attr

        event_row_id = torch.cat(all_event_row_ids, dim=0).long()
        event_edge_id = torch.cat(all_event_edge_ids, dim=0).long()
        event_attr = torch.cat(all_event_feats, dim=0)

        order = torch.argsort(event_row_id)
        event_row_id = event_row_id[order]
        event_edge_id = event_edge_id[order]
        event_attr = event_attr[order]

        self._validate_event_edge_id(
            event_edge_id=event_edge_id,
            num_edges=int(structure.edge_index.size(1)),
        )
        return event_row_id, event_edge_id, event_attr

    def forward(self, ft_features: FragmentTreeFeatures) -> torch.Tensor:
        """Return a legacy edge-level view by mean-pooling event features.

        New event-aware training code should call ``encode_events`` directly.
        This method exists so current spectrum-generation code can continue to
        consume one feature row per fragment-tree edge while the learnable
        aggregation model is removed.
        """
        structure = ft_features.structure
        num_edges = int(structure.edge_index.size(1))
        _, event_edge_id, event_attr = self.encode_events(ft_features)
        if event_attr.size(0) == 0:
            return self._empty_edge_attr(
                num_edges=num_edges,
                device=structure.edge_index.device,
                dtype=ft_features.mol_x.dtype,
            )
        return self._mean_events_to_edges(
            event_attr=event_attr,
            event_edge_id=event_edge_id,
            num_edges=num_edges,
        )

    def _get_cleavage_fnet_key(
        self,
        event_type: Tuple[int, int, int],
    ) -> str:
        event_type = (
            int(event_type[0]),
            int(event_type[1]),
            int(event_type[2]),
        )

        if event_type not in self.module_key_by_event_type:
            raise KeyError(
                f"Unknown cleavage event type: {event_type}."
            )

        return self.module_key_by_event_type[event_type]

    def _empty_edge_attr(
        self,
        *,
        num_edges: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create zero edge features when no cleavage events are available.
        """
        return torch.zeros(
            (int(num_edges), self.feature_dim),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _validate_event_edge_id(
        *,
        event_edge_id: torch.Tensor,
        num_edges: int,
    ) -> None:
        """Validate that event_edge_id references valid edge indices."""
        if event_edge_id.dim() != 1:
            raise ValueError(
                "event_edge_id must be 1D, "
                f"got shape {tuple(event_edge_id.shape)}."
            )

        if event_edge_id.numel() == 0:
            return

        min_edge_id = int(event_edge_id.min().item())
        max_edge_id = int(event_edge_id.max().item())

        if min_edge_id < 0:
            raise ValueError(
                f"event_edge_id contains negative edge index: {min_edge_id}."
            )

        if max_edge_id >= int(num_edges):
            raise ValueError(
                "event_edge_id contains out-of-range edge index: "
                f"max={max_edge_id}, num_edges={num_edges}."
            )

    def _iter_cleavage_fnet_batches(
        self,
        ft_features: FragmentTreeFeatures,
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int], CleavageFNetInput]]:
        """
        Iterate over CleavageFNet mini-batches grouped by event type.

        Groups are defined by:
            (cleavage_pattern_id, reaction_id, product_molecule_id)

        Yields
        ------
        event_row_id:
            [M_g] Row indices into structure.cleavage_event.

        event_edge_id:
            [M_g] Edge indices corresponding to cleavage events in this group.

        event_type:
            Tuple[int, int, int]
            (cleavage_pattern_id, reaction_id, product_molecule_id)

        cleavage_fnet_input:
            CleavageFNetInput for this event group.
        """
        structure = ft_features.structure

        if structure.num_cleavage_events == 0:
            return

        node_mol_feats = ft_features.mol_x
        # [N, mol_dim]

        atom_feats_global = ft_features.node_graphs.x
        # [A, atom_dim]

        event_edge_id_all = structure.cleavage_event_edge_index
        # [M]

        event_src_node_all = structure.edge_index[0, event_edge_id_all]
        # [M]

        event_dst_node_all = structure.edge_index[1, event_edge_id_all]
        # [M]

        event_cleavage_pattern_id_all = structure.cleavage_pattern_ids
        # [M]

        event_reaction_id_all = structure.cleavage_reaction_ids
        # [M]

        event_product_molecule_id_all = (
            structure.cleavage_product_molecule_ids
        )
        # [M]

        event_reactant_row_index_all = (
            structure.cleavage_reactant_row_indices
        )
        # [M]

        event_product_row_index_all = (
            structure.cleavage_product_row_indices
        )
        # [M]

        for event_type in self.module_key_by_event_type.keys():
            cleavage_pattern_id, reaction_id, product_molecule_id = event_type

            mask = (
                (event_cleavage_pattern_id_all == cleavage_pattern_id)
                & (event_reaction_id_all == reaction_id)
                & (event_product_molecule_id_all == product_molecule_id)
            )
            # [M]

            if not mask.any():
                continue

            event_idx = torch.nonzero(
                mask,
                as_tuple=False,
            ).flatten()
            # [M_g]

            event_edge_id = event_edge_id_all[event_idx]
            # [M_g]

            src_node = event_src_node_all[event_idx]
            # [M_g]

            dst_node = event_dst_node_all[event_idx]
            # [M_g]

            reactant_row_index = event_reactant_row_index_all[event_idx]
            # [M_g]

            product_row_index = event_product_row_index_all[event_idx]
            # [M_g]

            num_reactant_atoms = self._get_reactant_tuple_length(
                cleavage_pattern_id=cleavage_pattern_id,
                reaction_id=reaction_id,
            )

            num_product_atoms = self._get_product_tuple_length(
                cleavage_pattern_id=cleavage_pattern_id,
                reaction_id=reaction_id,
                product_molecule_id=product_molecule_id,
            )

            reactant_atom_index_table = self._get_atom_index_table(
                structure=structure,
                tuple_length=num_reactant_atoms,
                table_name="reactant_atom_index_table",
            )
            # [A_len, num_reactant_atoms]

            product_atom_index_table = self._get_atom_index_table(
                structure=structure,
                tuple_length=num_product_atoms,
                table_name="product_atom_index_table",
            )
            # [A_len, num_product_atoms]

            cleavage_fnet_input = self._build_cleavage_fnet_input_for_events(
                src_node=src_node,
                dst_node=dst_node,
                reactant_row_index=reactant_row_index,
                product_row_index=product_row_index,
                node_mol_feats=node_mol_feats,
                atom_feats_global=atom_feats_global,
                node_graph_offset=structure.node_graph_offset,
                reactant_atom_index_table=reactant_atom_index_table,
                product_atom_index_table=product_atom_index_table,
            )

            yield (
                event_idx,
                event_edge_id,
                event_type,
                cleavage_fnet_input,
            )

    def _get_reactant_tuple_length(
        self,
        *,
        cleavage_pattern_id: int,
        reaction_id: int,
    ) -> int:
        key = (
            int(cleavage_pattern_id),
            int(reaction_id),
        )

        if key not in self.reactant_tuple_length_by_event_type:
            raise KeyError(
                "Unknown reactant event type: "
                f"(cleavage_pattern_id, reaction_id)={key}."
            )

        return self.reactant_tuple_length_by_event_type[key]

    def _get_product_tuple_length(
        self,
        *,
        cleavage_pattern_id: int,
        reaction_id: int,
        product_molecule_id: int,
    ) -> int:
        key = (
            int(cleavage_pattern_id),
            int(reaction_id),
            int(product_molecule_id),
        )

        if key not in self.product_tuple_length_by_event_type:
            raise KeyError(
                "Unknown product event type: "
                "(cleavage_pattern_id, reaction_id, product_molecule_id)="
                f"{key}."
            )

        return self.product_tuple_length_by_event_type[key]

    @staticmethod
    def _get_atom_index_table(
        *,
        structure: FragmentTreeStructure,
        tuple_length: int,
        table_name: str,
    ) -> torch.Tensor:
        """
        Get a shared atom-index tuple table from FragmentTreeStructure.

        Parameters
        ----------
        structure:
            FragmentTreeStructure.

        tuple_length:
            Atom tuple length.

        table_name:
            Name used only for error messages.

        Returns
        -------
        atom_index_table:
            [A_len, tuple_length]
        """
        tuple_length = int(tuple_length)

        if tuple_length not in structure.cleavage_atom_idxs:
            raise KeyError(
                f"{table_name} requires tuple_length={tuple_length}, "
                "but structure.cleavage_atom_idxs does not contain that key. "
                f"Available tuple lengths: "
                f"{sorted(structure.cleavage_atom_idxs.keys())}."
            )

        atom_index_table = structure.cleavage_atom_idxs[tuple_length]

        if atom_index_table.dim() != 2:
            raise ValueError(
                f"{table_name} must be 2D, "
                f"got shape {tuple(atom_index_table.shape)}."
            )

        if atom_index_table.size(1) != tuple_length:
            raise ValueError(
                f"{table_name} has invalid tuple length: "
                f"got {atom_index_table.size(1)}, expected {tuple_length}."
            )

        return atom_index_table

    def _build_cleavage_fnet_input_for_events(
        self,
        *,
        src_node: torch.Tensor,
        dst_node: torch.Tensor,
        reactant_row_index: torch.Tensor,
        product_row_index: torch.Tensor,
        node_mol_feats: torch.Tensor,
        atom_feats_global: torch.Tensor,
        node_graph_offset: torch.Tensor,
        reactant_atom_index_table: torch.Tensor,
        product_atom_index_table: torch.Tensor,
    ) -> CleavageFNetInput:
        """
        Build CleavageFNetInput for one grouped cleavage event batch.

        Parameters
        ----------
        src_node:
            [B] Source fragment node indices.

        dst_node:
            [B] Destination fragment node indices.

        reactant_row_index:
            [B] Row indices into reactant_atom_index_table.

        product_row_index:
            [B] Row indices into product_atom_index_table.

        node_mol_feats:
            [N, mol_dim] Fragment node-level molecular features.

        atom_feats_global:
            [A, atom_dim] Global atom-level features.

        reactant_atom_index_table:
            [A_len, num_reactant_atoms]

        product_atom_index_table:
            [A_len, num_product_atoms]

        Returns
        -------
        CleavageFNetInput
        """
        src_node = src_node.long()
        dst_node = dst_node.long()
        reactant_row_index = reactant_row_index.long()
        product_row_index = product_row_index.long()

        self._validate_row_index(
            row_index=reactant_row_index,
            table_size=reactant_atom_index_table.size(0),
            name="reactant_row_index",
        )

        self._validate_row_index(
            row_index=product_row_index,
            table_size=product_atom_index_table.size(0),
            name="product_row_index",
        )

        reactant_local_atom_indices = reactant_atom_index_table[
            reactant_row_index
        ]
        # [B, num_reactant_atoms]

        product_local_atom_indices = product_atom_index_table[
            product_row_index
        ]
        # [B, num_product_atoms]

        reactant_atom_indices = (
            reactant_local_atom_indices
            + node_graph_offset[src_node].view(-1, 1)
        )
        # [B, num_reactant_atoms]

        product_atom_indices = (
            product_local_atom_indices
            + node_graph_offset[dst_node].view(-1, 1)
        )
        # [B, num_product_atoms]

        reactant_atom_indices = reactant_atom_indices.long()
        product_atom_indices = product_atom_indices.long()

        self._validate_atom_indices(
            atom_indices=reactant_atom_indices,
            num_atoms=atom_feats_global.size(0),
            name="reactant_atom_indices",
        )

        self._validate_atom_indices(
            atom_indices=product_atom_indices,
            num_atoms=atom_feats_global.size(0),
            name="product_atom_indices",
        )

        reactant_mol = node_mol_feats[src_node]
        # [B, mol_dim]

        product_mol = node_mol_feats[dst_node]
        # [B, mol_dim]

        reactant_atom_feats = atom_feats_global[reactant_atom_indices]
        # [B, num_reactant_atoms, atom_dim]

        product_atom_feats = atom_feats_global[product_atom_indices]
        # [B, num_product_atoms, atom_dim]

        return CleavageFNetInput(
            reactant_mol=reactant_mol,
            reactant_atom_feats=reactant_atom_feats,
            product_mol=product_mol,
            product_atom_feats=product_atom_feats,
        )

    @staticmethod
    def _validate_row_index(
        *,
        row_index: torch.Tensor,
        table_size: int,
        name: str,
    ) -> None:
        if row_index.numel() == 0:
            return

        min_row = int(row_index.min().item())
        max_row = int(row_index.max().item())

        if min_row < 0:
            raise IndexError(
                f"{name} contains negative row index: {min_row}."
            )

        if max_row >= int(table_size):
            raise IndexError(
                f"{name} out of range: "
                f"max={max_row}, table_size={table_size}."
            )

    @staticmethod
    def _validate_atom_indices(
        *,
        atom_indices: torch.Tensor,
        num_atoms: int,
        name: str,
    ) -> None:
        if atom_indices.numel() == 0:
            return

        min_atom = int(atom_indices.min().item())
        max_atom = int(atom_indices.max().item())

        if min_atom < 0:
            raise IndexError(
                f"{name} contains negative atom index: {min_atom}."
            )

        if max_atom >= int(num_atoms):
            raise IndexError(
                f"{name} out of range: "
                f"max={max_atom}, num_atoms={num_atoms}."
            )

    def _mean_events_to_edges(
        self,
        *,
        event_attr: torch.Tensor,
        event_edge_id: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        """Deterministically mean-pool event features for legacy edge users."""
        edge_attr = torch.zeros(
            (int(num_edges), self.feature_dim),
            dtype=event_attr.dtype,
            device=event_attr.device,
        )
        counts = torch.zeros(
            (int(num_edges), 1),
            dtype=event_attr.dtype,
            device=event_attr.device,
        )
        if event_attr.numel() == 0:
            return edge_attr
        edge_attr.index_add_(0, event_edge_id.long(), event_attr)
        ones = torch.ones(
            (event_attr.size(0), 1),
            dtype=event_attr.dtype,
            device=event_attr.device,
        )
        counts.index_add_(0, event_edge_id.long(), ones)
        return edge_attr / counts.clamp_min(1.0)
