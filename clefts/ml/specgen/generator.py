import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Tuple, Dict, List, Union, Optional

from ...libs.mmkit.mmkit import Adduct
from ...domain.fragment import Fragmenter
from ...domain.mass import parse_ce_to_ev
from ..common.torch_utils.model_base import ModelBase
from ..input.fragment_tree_structure import FragmentTreeStructure
from ..input.fragment_tree_features import FragmentTreeFeatures

from ..mol import MolEncoder
from .components.condition.condition_encoder import MS2ConditionEncoder
from .components.cleavage.cleavage_edge_feature_net import CleavageEdgeFeatureNet

class CleftsSpecGen(ModelBase):
    def __init__(self,
                 mol_encoder_params:Dict,
                 condition_encoder_params:Dict,
                 cleavage_edge_fnet_params:Dict,
                 fragmenter_params:Dict,
                 dropout: float,
                 ):
        super(CleftsSpecGen, self).__init__(
            ignore_config_keys=[],
            **{k: v for k, v in locals().items() if k != 'self'}
        )

        mol_encoder_params = mol_encoder_params.copy()
        mol_encoder_params['dropout'] = dropout
        self._mol_encoder = MolEncoder(**mol_encoder_params)
        self._fragmenter = Fragmenter.from_dict(fragmenter_params)

        condition_encoder_params = condition_encoder_params.copy()
        condition_encoder_params['adduct_type_strs'] = tuple(str(ad) for ad in self._fragmenter.adduct_types)
        self._condition_encoder = MS2ConditionEncoder(**condition_encoder_params)
        
        
        cleavage_edge_fnet_params['cleavage_pattern_set_params'] = self._fragmenter.cleavage_pattern_set.to_dict()
        cleavage_edge_fnet_params['mol_dim'] = self.mol_encoder.graph_dim
        cleavage_edge_fnet_params['atom_dim'] = self.mol_encoder.node_dim
        cleavage_edge_fnet_params['dropout'] = dropout
        self.cleavage_edge_fnet = CleavageEdgeFeatureNet(**cleavage_edge_fnet_params)


        pass
    

    @property
    def mol_atom_dim(self) -> int:
        return self._mol_encoder._node_dim
    
    @property
    def mol_graph_dim(self) -> int:
        return self._mol_encoder._graph_dim

    @property
    def tree_max_depth(self) -> int:
        return self._fragmenter.tree_max_depth

    @property
    def precursor_candidate_max_depth(self) -> int:
        return self._fragmenter.precursor_candidate_max_depth

    @property
    def mol_encoder(self) -> MolEncoder:
        return self._mol_encoder
    
    @property
    def fragmenter(self) -> Fragmenter:
        return self._fragmenter

    def get_index_by_adduct_type(self, adduct_type: Adduct) -> int:
        return self._fragmenter.get_index_by_adduct_type(adduct_type)

    @staticmethod
    def parse_ce_to_ev(ce: str, precursor_mz: float, instrument: str = None) -> Optional[float]:
        ev = parse_ce_to_ev(ce, precursor_mz, instrument)
        return ev
    
    def forward(self, data: Union[FragmentTreeStructure, FragmentTreeFeatures]) -> FragmentTreeFeatures:
        if isinstance(data, FragmentTreeStructure):
            mol_graph = self._mol_encoder(data.node_graph)
            self._validate_mol_encoder_output(mol_graph, data)
            ft_features = FragmentTreeFeatures.from_structure(data, mol_graph)

            edge_attr = self.cleavage_edge_fnet(ft_features)
            if edge_attr.size(0) != data.edge_index.size(1):
                raise ValueError(f"CleavageEdgeFeatureNet output has {edge_attr.size(0)} edges, but FragmentTreeStructure has {data.edge_index.size(1)} edges. These must match.")
            ft_features = FragmentTreeFeatures.from_structure(data, node_graphs=mol_graph, edge_attr=edge_attr)
        elif isinstance(data, FragmentTreeFeatures):
            ft_features = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        tree_pyg_features, exp_feat, kept_sample_ids = self._build_tree_pyg_data(ft_batch)

        return ft_features


    def _validate_mol_encoder_output(self, mol_graph: Batch, structure: FragmentTreeStructure):
        if mol_graph.num_graphs != structure.num_nodes:
            raise ValueError(f"MolEncoder output has num_graphs={mol_graph.num_graphs}, but FragmentTreeStructure has n_nodes={structure.n_nodes}. These must match.")
        if mol_graph.x.size(0) != structure.node_graph.num_nodes:
            raise ValueError(f"MolEncoder output has x.size(0)={mol_graph.x.size(0)}, but FragmentTreeStructure's node_graph have num_nodes={structure.node_graph.num_nodes}. These must match.")
        if mol_graph.edge_index.size(1) != structure.node_graph.num_edges:
            raise ValueError(f"MolEncoder output has edge_index.size(1)={mol_graph.edge_index.size(1)}, but FragmentTreeStructure's node_graph have num_edges={structure.node_graph.num_edges}. These must match.")


    def _build_tree_pyg_data(
        self,
        ft_features: "FragmentTreeFeatures",
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
        """
        Build per-sample FragmentTree PyG graphs from FragmentTreeFeatures.

        Current FragmentTreeStructure specification
        -------------------------------------------
        This implementation assumes:

            st.edge_index:
                [2, E]

            st.sample_edge_index:
                [2, L]
                row 0: sample index
                row 1: edge index

            st.sample_adduct_type_index:
                [S]

            st.sample_ce_value:
                [S]

            st.sample_precursor_edge_index_path:
                [P, D]
                Each row is a padded sequence of edge ids.
                Padding value is -1.

            st.sample_precursor_path_index:
                [P]
                sample index for each precursor edge path.

        Returns
        -------
        pyg_batch:
            PyG Batch of per-sample fragment-tree graphs.

        graph_repr:
            [G, tree_dim]
            Per-graph experimental representation aligned with pyg_batch graph order.

        kept_sample_ids:
            [G]
            Original sample ids used in pyg_batch.
        """

        st = ft_features.structure

        tree_node_x = ft_features.mol_x
        # [N, node_dim]

        all_edge_attr = ft_features.edge_attr
        # [E, edge_dim]

        device = tree_node_x.device

        if st.edge_index.dim() != 2 or st.edge_index.size(0) != 2:
            raise ValueError(
                "st.edge_index must have shape [2, E], "
                f"got shape {tuple(st.edge_index.shape)}."
            )

        E = int(st.edge_index.size(1))
        N = int(tree_node_x.size(0))
        S = int(st.num_samples)

        if S <= 0:
            raise ValueError("FragmentTreeStructure must contain at least one sample.")

        if all_edge_attr.dim() != 2 or all_edge_attr.size(0) != E:
            raise ValueError(
                "ft_features.edge_attr must have shape [E, edge_dim]. "
                f"Got shape {tuple(all_edge_attr.shape)}, E={E}."
            )

        sample_edge_index = st.sample_edge_index.to(device).long()
        # [2, L]

        if sample_edge_index.dim() != 2 or sample_edge_index.size(0) != 2:
            raise ValueError(
                "st.sample_edge_index must have shape [2, L], "
                f"got shape {tuple(sample_edge_index.shape)}."
            )

        sample_precursor_edge_index_path = (
            st.sample_precursor_edge_index_path.to(device).long()
        )
        # [P, D]

        sample_precursor_path_index = (
            st.sample_precursor_path_index.to(device).long()
        )
        # [P]

        if sample_precursor_edge_index_path.dim() != 2:
            raise ValueError(
                "st.sample_precursor_edge_index_path must be 2D, "
                f"got shape {tuple(sample_precursor_edge_index_path.shape)}."
            )

        if sample_precursor_path_index.dim() != 1:
            raise ValueError(
                "st.sample_precursor_path_index must be 1D, "
                f"got shape {tuple(sample_precursor_path_index.shape)}."
            )

        if sample_precursor_edge_index_path.size(0) != sample_precursor_path_index.size(0):
            raise ValueError(
                "sample_precursor_edge_index_path and sample_precursor_path_index "
                "must have the same number of rows. "
                f"Got {sample_precursor_edge_index_path.size(0)} and "
                f"{sample_precursor_path_index.size(0)}."
            )

        global_edge_u = st.edge_index[0].to(device).long()
        # [E]

        global_edge_v = st.edge_index[1].to(device).long()
        # [E]

        # -------------------------
        # Per-sample experimental features
        # -------------------------
        exp_feat_all = self.experimental_fnet(
            st.sample_adduct_type_index.to(device).long(),
            st.sample_ce_value.to(device),
        )
        # [S, exp_dim]

        exp_feat_all = self.tree_graphrepr_proj(exp_feat_all)
        # [S, tree_dim]

        if exp_feat_all.size(0) != S:
            raise ValueError(
                "experimental_fnet output has invalid sample dimension: "
                f"got {exp_feat_all.size(0)}, expected {S}."
            )

        if exp_feat_all.size(1) != self.tree_encoder.dim:
            raise ValueError(
                "tree_graphrepr_proj output has invalid feature dimension: "
                f"got {exp_feat_all.size(1)}, expected {self.tree_encoder.dim}."
            )

        data_list: List[Data] = []
        kept_sample_ids: List[int] = []

        seq_list: List[Tensor] = []
        seq_ptr_list: List[int] = [0]
        edge_ptr_list: List[int] = [0]

        path_width = int(sample_precursor_edge_index_path.size(1))

        for sample_id in range(S):
            # -------------------------
            # 1) Edges assigned to this sample
            # -------------------------
            sample_edge_mask = sample_edge_index[0] == sample_id
            edge_ids_from_sample = sample_edge_index[1][sample_edge_mask].long()
            edge_ids_from_sample = edge_ids_from_sample[
                edge_ids_from_sample >= 0
            ]

            if edge_ids_from_sample.numel() > 0:
                edge_ids_from_sample = torch.unique(
                    edge_ids_from_sample,
                    sorted=True,
                )

            # -------------------------
            # 2) Precursor edge paths assigned to this sample
            # -------------------------
            path_mask = sample_precursor_path_index == sample_id
            precursor_edge_paths = sample_precursor_edge_index_path[path_mask]
            # [P_s, D]

            if precursor_edge_paths.numel() > 0:
                precursor_path_edge_ids = precursor_edge_paths.reshape(-1)
                precursor_path_edge_ids = precursor_path_edge_ids[
                    precursor_path_edge_ids >= 0
                ]

                if precursor_path_edge_ids.numel() > 0:
                    precursor_path_edge_ids = torch.unique(
                        precursor_path_edge_ids,
                        sorted=True,
                    )
            else:
                precursor_path_edge_ids = torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=device,
                )

            # Use both sample edges and precursor-path edges.
            # This prevents precursor path edges from being missing in the local graph.
            edge_ids = torch.cat(
                [
                    edge_ids_from_sample,
                    precursor_path_edge_ids,
                ],
                dim=0,
            )

            if edge_ids.numel() > 0:
                edge_ids = torch.unique(
                    edge_ids,
                    sorted=True,
                )

            if edge_ids.numel() == 0:
                raise ValueError(
                    f"No edges found for sample {sample_id}. "
                    "Each sample must have at least one edge in sample_edge_index "
                    "or sample_precursor_edge_index_path."
                )

            if edge_ids.min().item() < 0 or edge_ids.max().item() >= E:
                raise IndexError(
                    f"edge_ids out of range for sample {sample_id}: "
                    f"min={edge_ids.min().item()}, max={edge_ids.max().item()}, E={E}."
                )

            # -------------------------
            # 3) Nodes for this sample
            # -------------------------
            u = global_edge_u[edge_ids]
            v = global_edge_v[edge_ids]

            node_ids = torch.unique(
                torch.cat(
                    [
                        u,
                        v,
                    ],
                    dim=0,
                ),
                sorted=True,
            )
            # [N_s]

            if node_ids.numel() == 0:
                raise ValueError(
                    f"No nodes found for sample {sample_id}."
                )

            if node_ids.min().item() < 0 or node_ids.max().item() >= N:
                raise IndexError(
                    f"node_ids out of range for sample {sample_id}: "
                    f"min={node_ids.min().item()}, max={node_ids.max().item()}, N={N}."
                )

            num_nodes_in_sample = int(node_ids.numel())

            # -------------------------
            # 4) Global node id -> local node id
            # -------------------------
            node_old_to_new = torch.full(
                (N,),
                -1,
                dtype=torch.long,
                device=device,
            )

            node_old_to_new[node_ids] = torch.arange(
                num_nodes_in_sample,
                dtype=torch.long,
                device=device,
            )

            # -------------------------
            # 5) Slice node features
            # -------------------------
            x = tree_node_x[node_ids]
            # [N_s, node_dim]

            # -------------------------
            # 6) Node type features
            # -------------------------
            node_type = torch.zeros(
                (num_nodes_in_sample, 3),
                dtype=torch.float32,
                device=device,
            )
            node_type[:, 2] = 1.0
            # columns:
            #   0: precursor_path_node
            #   1: precursor_root
            #   2: normal

            precursor_root_global_nodes = self._infer_precursor_root_nodes_from_edge_paths(
                edge_paths=precursor_edge_paths,
                global_edge_u=global_edge_u,
            )
            # [R_s]

            if precursor_root_global_nodes.numel() > 0:
                precursor_root_local_nodes = node_old_to_new[
                    precursor_root_global_nodes
                ]
                precursor_root_local_nodes = precursor_root_local_nodes[
                    precursor_root_local_nodes >= 0
                ]

                if precursor_root_local_nodes.numel() > 0:
                    precursor_root_local_nodes = torch.unique(
                        precursor_root_local_nodes,
                        sorted=True,
                    )

                    node_type[precursor_root_local_nodes] = 0.0
                    node_type[precursor_root_local_nodes, 1] = 1.0

            precursor_path_global_nodes = self._infer_nodes_from_edge_paths(
                edge_paths=precursor_edge_paths,
                global_edge_u=global_edge_u,
                global_edge_v=global_edge_v,
            )
            # [Q_s]

            if precursor_path_global_nodes.numel() > 0:
                precursor_path_local_nodes = node_old_to_new[
                    precursor_path_global_nodes
                ]
                precursor_path_local_nodes = precursor_path_local_nodes[
                    precursor_path_local_nodes >= 0
                ]

                if precursor_path_local_nodes.numel() > 0:
                    precursor_path_local_nodes = torch.unique(
                        precursor_path_local_nodes,
                        sorted=True,
                    )

                    # Do not overwrite precursor roots.
                    if precursor_root_global_nodes.numel() > 0:
                        root_local = node_old_to_new[precursor_root_global_nodes]
                        root_local = root_local[root_local >= 0]
                        if root_local.numel() > 0:
                            precursor_path_local_nodes = precursor_path_local_nodes[
                                ~torch.isin(
                                    precursor_path_local_nodes,
                                    torch.unique(root_local),
                                )
                            ]

                    if precursor_path_local_nodes.numel() > 0:
                        node_type[precursor_path_local_nodes] = 0.0
                        node_type[precursor_path_local_nodes, 0] = 1.0

            # -------------------------
            # 7) Optional adduct / neutral-HS node features
            # -------------------------
            extra_node_features: List[Tensor] = [
                node_type,
            ]

            if hasattr(self, "_adduct_type_idx_to_one_hot_vec"):
                adduct_one_hot_table = self._adduct_type_idx_to_one_hot_vec.to(device)
                adduct_dim = int(adduct_one_hot_table.size(1))

                ion_hot_node = torch.zeros(
                    (num_nodes_in_sample, adduct_dim),
                    dtype=torch.float32,
                    device=device,
                )

                sample_adduct_idx = int(
                    st.sample_adduct_type_index[sample_id].item()
                )

                if 0 <= sample_adduct_idx < adduct_one_hot_table.size(0):
                    if precursor_root_global_nodes.numel() > 0:
                        precursor_root_local_nodes = node_old_to_new[
                            precursor_root_global_nodes
                        ]
                        precursor_root_local_nodes = precursor_root_local_nodes[
                            precursor_root_local_nodes >= 0
                        ]

                        if precursor_root_local_nodes.numel() > 0:
                            ion_hot_node[precursor_root_local_nodes] = (
                                adduct_one_hot_table[sample_adduct_idx]
                            )

                extra_node_features.append(ion_hot_node)

            if hasattr(self, "_neutral_hs_adduct_idx_to_one_hot_vec"):
                neutral_one_hot_table = self._neutral_hs_adduct_idx_to_one_hot_vec.to(
                    device
                )
                neutral_dim = int(neutral_one_hot_table.size(1))

                neutral_hot_node = torch.zeros(
                    (num_nodes_in_sample, neutral_dim),
                    dtype=torch.float32,
                    device=device,
                )

                extra_node_features.append(neutral_hot_node)

            x = torch.cat(
                [
                    x,
                    *extra_node_features,
                ],
                dim=1,
            )

            # -------------------------
            # 8) Slice/remap edges
            # -------------------------
            edge_old_to_new = torch.full(
                (E,),
                -1,
                dtype=torch.long,
                device=device,
            )

            edge_old_to_new[edge_ids] = torch.arange(
                edge_ids.numel(),
                dtype=torch.long,
                device=device,
            )

            u_new = node_old_to_new[global_edge_u[edge_ids]]
            v_new = node_old_to_new[global_edge_v[edge_ids]]

            if (u_new < 0).any() or (v_new < 0).any():
                raise ValueError(
                    f"Some edges for sample {sample_id} have endpoints "
                    "that do not map to local node ids."
                )

            edge_index = torch.stack(
                [
                    u_new,
                    v_new,
                ],
                dim=0,
            )
            # [2, E_s]

            edge_attr = all_edge_attr[edge_ids]
            # [E_s, edge_dim]

            edge_id_global = edge_ids
            # [E_s]

            # -------------------------
            # 9) Build local precursor pathway sequence
            # -------------------------
            precursor_pathway_seq_local = self._edge_paths_to_local_node_edge_seq(
                edge_paths=precursor_edge_paths,
                global_edge_u=global_edge_u,
                global_edge_v=global_edge_v,
                node_old_to_new=node_old_to_new,
                edge_old_to_new=edge_old_to_new,
                max_path_width=path_width,
                device=device,
            )
            # [P_s, 2 * D + 1]

            if precursor_pathway_seq_local.numel() > 0:
                node_values = precursor_pathway_seq_local[:, 0::2].reshape(-1)
                node_values = node_values[node_values != -1]

                if node_values.numel() > 0:
                    if node_values.min().item() < 0:
                        raise IndexError(
                            f"Local node ids in precursor path are negative "
                            f"for sample {sample_id}."
                        )

                    if node_values.max().item() >= num_nodes_in_sample:
                        raise IndexError(
                            f"Local node ids in precursor path are out of range "
                            f"for sample {sample_id}."
                        )

                edge_values = precursor_pathway_seq_local[:, 1::2].reshape(-1)
                edge_values = edge_values[edge_values != -1]

                if edge_values.numel() > 0:
                    if edge_values.min().item() < 0:
                        raise IndexError(
                            f"Local edge ids in precursor path are negative "
                            f"for sample {sample_id}."
                        )

                    if edge_values.max().item() >= edge_ids.numel():
                        raise IndexError(
                            f"Local edge ids in precursor path are out of range "
                            f"for sample {sample_id}."
                        )

            seq_list.append(precursor_pathway_seq_local)
            seq_ptr_list.append(
                seq_ptr_list[-1] + int(precursor_pathway_seq_local.size(0))
            )
            edge_ptr_list.append(
                edge_ptr_list[-1] + int(edge_index.size(1))
            )

            # -------------------------
            # 10) Build Data
            # -------------------------
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            data.node_id_global = node_ids
            data.edge_id_global = edge_id_global
            data.sample_id = torch.tensor(
                sample_id,
                dtype=torch.long,
                device=device,
            )

            # Backward-compatible alias if other code still expects member_id.
            data.member_id = data.sample_id

            data_list.append(data)
            kept_sample_ids.append(sample_id)

        if len(data_list) == 0:
            raise ValueError("No sample graphs were created.")

        pyg_batch = Batch.from_data_list(data_list)

        kept_sample_ids_t = torch.tensor(
            kept_sample_ids,
            dtype=torch.long,
            device=device,
        )

        graph_repr = exp_feat_all[kept_sample_ids_t]
        # [G, tree_dim]

        # -------------------------
        # Concatenate precursor pathway sequences
        # -------------------------
        if len(seq_list) == 0:
            precursor_pathway_seq_batch = torch.empty(
                (0, 2 * path_width + 1),
                dtype=torch.long,
                device=device,
            )

            precursor_pathway_ptr_batch = torch.zeros(
                (len(data_list) + 1,),
                dtype=torch.long,
                device=device,
            )
        else:
            precursor_pathway_seq_batch = torch.cat(
                seq_list,
                dim=0,
            )

            precursor_pathway_ptr_batch = torch.tensor(
                seq_ptr_list,
                dtype=torch.long,
                device=device,
            )

        edge_ptr = torch.tensor(
            edge_ptr_list,
            dtype=torch.long,
            device=device,
        )

        precursor_pathway_seq_batch = pathway_seq_local_to_batch_global(
            seq_local=precursor_pathway_seq_batch,
            pathway_ptr=precursor_pathway_ptr_batch,
            node_ptr=pyg_batch.ptr,
            edge_ptr=edge_ptr,
        )

        pyg_batch.precursor_pathway_seq = precursor_pathway_seq_batch
        pyg_batch.precursor_pathway_ptr = precursor_pathway_ptr_batch
        pyg_batch.edge_ptr = edge_ptr

        # Backward-compatible aliases.
        pyg_batch.kept_sample_ids = kept_sample_ids_t
        pyg_batch.kept_member_ids = kept_sample_ids_t

        return pyg_batch, graph_repr, kept_sample_ids_t


    @staticmethod
    def _infer_precursor_root_nodes_from_edge_paths(
        *,
        edge_paths: torch.Tensor,
        global_edge_u: torch.Tensor,
    ) -> torch.Tensor:
        """Infer precursor root nodes from the first valid edge in each path."""

        device = global_edge_u.device

        if edge_paths.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        roots: List[int] = []

        for row in edge_paths:
            valid_edges = row[row >= 0]

            if valid_edges.numel() == 0:
                continue

            first_edge_id = int(valid_edges[0].item())
            roots.append(
                int(global_edge_u[first_edge_id].item())
            )

        if len(roots) == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        return torch.unique(
            torch.tensor(
                roots,
                dtype=torch.long,
                device=device,
            ),
            sorted=True,
        )


    @staticmethod
    def _infer_nodes_from_edge_paths(
        *,
        edge_paths: torch.Tensor,
        global_edge_u: torch.Tensor,
        global_edge_v: torch.Tensor,
    ) -> torch.Tensor:
        """Infer all node ids appearing in edge paths."""

        device = global_edge_u.device

        if edge_paths.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        valid_edges = edge_paths.reshape(-1)
        valid_edges = valid_edges[valid_edges >= 0]

        if valid_edges.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        nodes = torch.cat(
            [
                global_edge_u[valid_edges],
                global_edge_v[valid_edges],
            ],
            dim=0,
        )

        return torch.unique(
            nodes,
            sorted=True,
        )


    @staticmethod
    def _edge_paths_to_local_node_edge_seq(
        *,
        edge_paths: torch.Tensor,
        global_edge_u: torch.Tensor,
        global_edge_v: torch.Tensor,
        node_old_to_new: torch.Tensor,
        edge_old_to_new: torch.Tensor,
        max_path_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert edge-only precursor paths to local node-edge-node sequences.

        Parameters
        ----------
        edge_paths:
            [P_s, D]
            Global edge ids padded by -1.

        Returns
        -------
        seq:
            [P_s, 2 * D + 1]

            columns:
                0: node id
                1: edge id
                2: node id
                3: edge id
                ...
        """

        path_count = int(edge_paths.size(0))
        seq_width = 2 * int(max_path_width) + 1

        if path_count == 0:
            return torch.empty(
                (0, seq_width),
                dtype=torch.long,
                device=device,
            )

        seq = torch.full(
            (path_count, seq_width),
            -1,
            dtype=torch.long,
            device=device,
        )

        for path_index in range(path_count):
            row = edge_paths[path_index]
            valid_edges = row[row >= 0]

            if valid_edges.numel() == 0:
                continue

            first_edge_id = int(valid_edges[0].item())
            first_src_node = int(global_edge_u[first_edge_id].item())

            first_src_node_local = int(node_old_to_new[first_src_node].item())

            if first_src_node_local < 0:
                raise ValueError(
                    f"First source node {first_src_node} does not map to "
                    "a local node id."
                )

            seq[path_index, 0] = first_src_node_local

            for step_index, edge_id_tensor in enumerate(valid_edges):
                edge_id = int(edge_id_tensor.item())

                local_edge_id = int(edge_old_to_new[edge_id].item())
                if local_edge_id < 0:
                    raise ValueError(
                        f"Edge {edge_id} does not map to a local edge id."
                    )

                dst_node = int(global_edge_v[edge_id].item())
                local_dst_node = int(node_old_to_new[dst_node].item())
                if local_dst_node < 0:
                    raise ValueError(
                        f"Destination node {dst_node} does not map to "
                        "a local node id."
                    )

                edge_col = 2 * step_index + 1
                node_col = 2 * step_index + 2

                if edge_col >= seq_width or node_col >= seq_width:
                    raise IndexError(
                        "Precursor path is longer than max_path_width."
                    )

                seq[path_index, edge_col] = local_edge_id
                seq[path_index, node_col] = local_dst_node

        return seq