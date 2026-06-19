import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from dataclasses import dataclass
from typing import Tuple, Dict, List, Union, Optional
from bidict import bidict
import numpy as np

from ...libs.mmkit.mmkit import Adduct
from ...domain.fragment import Fragmenter
from ...domain.mass import parse_ce_to_ev
from ..common.layers.graphormer import GraphormerEncoder
from ..input.fragment_tree_structure import FragmentTreeStructure
from ..input.fragment_tree_features import FragmentTreeFeatures

from ..mol import MolEncoder
from .components.condition.condition_encoder import MS2ConditionEncoder
from .components.cleavage.cleavage_edge_feature_net import CleavageEdgeFeatureNet

from dataclasses import dataclass
from torch import Tensor
from torch_geometric.data import Batch


@dataclass
class FlatAdductNodeChoiceProbabilities:
    """Flat node choice probabilities with adduct-dependent valid candidates."""

    logit: Tensor
    # [N_tree, C_flat]
    # Invalid columns are filled with -inf.

    prob: Tensor
    # [N_tree, C_flat]
    # Invalid columns are 0.

    valid_mask: Tensor
    # [N_tree, C_flat]
    # True if the column is valid for the node's main adduct type.


@dataclass
class FragmentGenerationProbabilities:
    """Probabilities used for fragment-tree generation."""

    precursor_logit: Tensor
    # [P]

    p_precursor: Tensor
    # [P]
    # Sum is 1 within each sample graph.

    precursor_expand_logit: Tensor
    # [P]

    p_precursor_expand: Tensor
    # [P]

    p_precursor_stop: Tensor
    # [P]

    edge_logit: Tensor
    # [E_tree]

    p_edge: Tensor
    # [E_tree]

    expand_logit: Tensor
    # [N_tree]

    p_expand: Tensor
    # [N_tree]

    p_stop: Tensor
    # [N_tree]

    ion: FlatAdductNodeChoiceProbabilities
    # ion.prob: [N_tree, C_ion_flat]

    unsaturation: FlatAdductNodeChoiceProbabilities
    # unsaturation.prob: [N_tree, C_unsaturation_flat]

    radical: FlatAdductNodeChoiceProbabilities
    # radical.prob: [N_tree, C_radical_flat]


@dataclass
class FragmentTreeProbabilityOutput:
    """Output of FragmentTreeProbabilityModel."""

    ft_features: "FragmentTreeFeatures"

    sample_tree_batch: Batch
    # sample_tree_batch.x:
    #   [N_tree, tree_dim]
    #
    # sample_tree_batch.tree_graph_repr:
    #   [G, tree_graph_repr_dim]
    #
    # sample_tree_batch.condition_tree_repr:
    #   [G, tree_graph_repr_dim]
    #
    # sample_tree_batch.kept_sample_ids:
    #   [G]

    probabilities: FragmentGenerationProbabilities

@dataclass
class BestFlatAdductCombination:
    """Best ion / unsaturation / radical combination for each node."""

    ion_index: Tensor
    # [N_tree]

    unsaturation_index: Tensor
    # [N_tree]

    radical_index: Tensor
    # [N_tree]

    probability: Tensor
    # [N_tree]
    # max over ion * unsaturation * radical


@dataclass
class NodeGenerationFlowProbabilities:
    """Graph-wise node generation flow probabilities."""

    p_precursor_init: Tensor
    # [N_tree]
    # Initial probability mass at precursor candidate nodes.
    # Sum is 1 within each sample graph.

    p_reach: Tensor
    # [N_tree]
    # Total probability of reaching each node.

    p_emit: Tensor
    # [N_tree]
    # p_reach * p_stop

    p_emit_best_adduct: Tensor
    # [N_tree]
    # p_reach * p_stop * best_adduct_probability

    p_expand_node: Tensor
    # [N_tree]
    # p_reach * p_expand

    p_expand_edge: Tensor
    # [E_tree]
    # p_reach[src] * p_expand[src] * p_edge

    best_adduct: BestFlatAdductCombination

class FragmentTreeProbabilityModel(nn.Module):
    def __init__(self,
                 mol_encoder_params:Dict,
                 condition_encoder_params:Dict,
                 cleavage_edge_fnet_params:Dict,
                 tree_encoder_params: Dict,
                 fragmenter_params:Dict,
                 dropout: float,
                 ):
        super(FragmentTreeProbabilityModel, self).__init__()

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


        # -------------------------
        # Tree encoder
        # -------------------------
        tree_encoder_params = tree_encoder_params.copy()

        tree_encoder_in_dim = (
            self.mol_encoder.graph_dim
            + 3
        )
        # Current _build_tree_pyg_data concatenates:
        #   ft_features.mol_x     [N, mol_graph_dim]
        #   node_type             [N, 3]
        #
        # Therefore:
        #   tree node feature dim = mol_graph_dim + 3

        tree_encoder_params["in_dim"] = tree_encoder_in_dim
        tree_encoder_params["edge_dim"] = self.cleavage_edge_fnet.feature_dim
        tree_encoder_params["max_spatial_dist"] = self.fragmenter.tree_max_depth + 1
        tree_encoder_params["max_edge_dist"] = self.fragmenter.tree_max_depth + 1
        tree_encoder_params["dropout"] = dropout
        tree_encoder_params["add_virtual_node"] = True
        tree_encoder_params["undirected_for_spd"] = False
        tree_encoder_params["undirected_for_path"] = False
        tree_encoder_params["freeze_vnode"] = True

        self.tree_encoder = GraphormerEncoder(**tree_encoder_params)

        # -------------------------
        # MS2 condition -> tree graph representation
        # -------------------------
        self.ms2_condition_to_tree_proj = nn.Sequential(
            nn.Linear(self._condition_encoder.feature_dim, self.tree_encoder.dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -------------------------
        # Probability heads
        # -------------------------
        tree_dim = self.tree_encoder.dim
        edge_dim = self.cleavage_edge_fnet.feature_dim

        self.precursor_select_head = nn.Sequential(
            nn.Linear(tree_dim, tree_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tree_dim, 1),
        )

        self.precursor_expand_head = nn.Sequential(
            nn.Linear(tree_dim, tree_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tree_dim, 1),
        )

        self.edge_select_head = nn.Sequential(
            nn.Linear(
                tree_dim * 2 + edge_dim,
                tree_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                tree_dim,
                1,
            ),
        )

        self.node_expand_head = nn.Sequential(
            nn.Linear(
                tree_dim,
                tree_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                tree_dim,
                1,
            ),
        )

        self.main_adduct_types = bidict(
            {
                idx: adduct
                for idx, adduct in enumerate(self._fragmenter.adduct_types)
            }
        )

        self.ion_candidates_by_adduct: Dict[Adduct, np.ndarray] = {
            adduct: np.asarray(
                self._fragmenter.get_ion_shift_adducts_by_adduct_type(adduct),
                dtype=object,
            )
            for adduct in self._fragmenter.adduct_types
        }

        self.unsaturation_candidates_by_adduct: Dict[Adduct, np.ndarray] = {
            adduct: np.asarray(
                self._fragmenter.get_unsaturation_adduct_candidates_by_adduct_type(
                    adduct
                ),
                dtype=object,
            )
            for adduct in self._fragmenter.adduct_types
        }

        self.radical_candidates_by_adduct: Dict[Adduct, np.ndarray] = {
            adduct: np.asarray(
                self._fragmenter.get_radical_adduct_candidates_by_adduct_type(
                    adduct
                ),
                dtype=object,
            )
            for adduct in self._fragmenter.adduct_types
        }

        self.ion_flat_candidates, self.ion_candidate_slice_by_adduct = self._build_flat_adduct_candidate_table(self.ion_candidates_by_adduct)
        self.unsaturation_flat_candidates, self.unsaturation_candidate_slice_by_adduct = self._build_flat_adduct_candidate_table(self.unsaturation_candidates_by_adduct)
        self.radical_flat_candidates, self.radical_candidate_slice_by_adduct = self._build_flat_adduct_candidate_table(self.radical_candidates_by_adduct)

        self.node_ion_heads_by_adduct_str = nn.ModuleDict()
        self.node_unsaturation_heads_by_adduct_str = nn.ModuleDict()
        self.node_radical_heads_by_adduct_str = nn.ModuleDict()

        for adduct_type in self._fragmenter.adduct_types:
            adduct_key = str(adduct_type)

            ion_candidates = self.ion_candidates_by_adduct[adduct_type]
            unsaturation_candidates = self.unsaturation_candidates_by_adduct[
                adduct_type
            ]
            radical_candidates = self.radical_candidates_by_adduct[adduct_type]

            if len(ion_candidates) == 0:
                raise ValueError(
                    f"No ion candidates found for adduct type: {adduct_type}"
                )

            if len(unsaturation_candidates) == 0:
                raise ValueError(
                    f"No unsaturation candidates found for adduct type: {adduct_type}"
                )

            if len(radical_candidates) == 0:
                raise ValueError(
                    f"No radical candidates found for adduct type: {adduct_type}"
                )

            self.node_ion_heads_by_adduct_str[adduct_key] = (
                self._make_node_choice_head(
                    tree_dim=tree_dim,
                    num_candidates=len(ion_candidates),
                    dropout=dropout,
                )
            )

            self.node_unsaturation_heads_by_adduct_str[adduct_key] = (
                self._make_node_choice_head(
                    tree_dim=tree_dim,
                    num_candidates=len(unsaturation_candidates),
                    dropout=dropout,
                )
            )

            self.node_radical_heads_by_adduct_str[adduct_key] = (
                self._make_node_choice_head(
                    tree_dim=tree_dim,
                    num_candidates=len(radical_candidates),
                    dropout=dropout,
                )
            )
    

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
        ft_features = self._build_fragment_tree_features(data)

        sample_tree_batch = self._encode_sample_tree(ft_features)

        probabilities = self._compute_fragment_generation_probabilities(
            ft_features=ft_features,
            sample_tree_batch=sample_tree_batch,
        )

        return FragmentTreeProbabilityOutput(
            ft_features=ft_features,
            sample_tree_batch=sample_tree_batch,
            probabilities=probabilities,
        )

    def _build_flat_adduct_candidate_table(
        self,
        candidates_by_adduct: Dict[Adduct, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[Adduct, slice]]:
        """
        Build a flat candidate table.

        Returns
        -------
        flat_candidates:
            np.ndarray with shape [C_flat, 2]

            flat_candidates[:, 0]:
                main adduct types

            flat_candidates[:, 1]:
                candidate adducts

        candidate_slice_by_adduct:
            Dict[Adduct, slice]
            Column slice in flat_candidates for each main adduct type.
        """

        rows: List[Tuple[Adduct, Adduct]] = []
        candidate_slice_by_adduct: Dict[Adduct, slice] = {}

        cursor = 0

        for _, main_adduct_type in self.main_adduct_types.items():
            candidates = candidates_by_adduct[main_adduct_type]

            if candidates.ndim != 1:
                raise ValueError(
                    "Each candidates_by_adduct value must be 1D np.ndarray. "
                    f"Got shape {candidates.shape} for {main_adduct_type}."
                )

            if len(candidates) == 0:
                raise ValueError(
                    f"No candidates found for adduct type: {main_adduct_type}"
                )

            start = cursor

            for candidate_adduct in candidates:
                rows.append(
                    (
                        main_adduct_type,
                        candidate_adduct,
                    )
                )
                cursor += 1

            end = cursor
            candidate_slice_by_adduct[main_adduct_type] = slice(start, end)

        flat_candidates = np.asarray(
            rows,
            dtype=object,
        )

        if flat_candidates.size == 0:
            flat_candidates = np.empty(
                (0, 2),
                dtype=object,
            )
        else:
            flat_candidates = flat_candidates.reshape(-1, 2)

        return flat_candidates, candidate_slice_by_adduct

    def _build_fragment_tree_features(
        self,
        data: "FragmentTreeStructure | FragmentTreeFeatures",
    ) -> "FragmentTreeFeatures":
        """Build FragmentTreeFeatures from structure or return given features."""

        if isinstance(data, FragmentTreeFeatures):
            return data

        if not isinstance(data, FragmentTreeStructure):
            raise TypeError(
                f"Unsupported data type: {type(data)}"
            )

        mol_graph = self._mol_encoder(data.node_graph)
        self._validate_mol_encoder_output(mol_graph, data)

        ft_features = FragmentTreeFeatures.from_structure(
            data,
            node_graphs=mol_graph,
        )

        edge_attr = self.cleavage_edge_fnet(ft_features)

        if edge_attr.size(0) != data.edge_index.size(1):
            raise ValueError(
                "CleavageEdgeFeatureNet output has invalid edge dimension: "
                f"got {edge_attr.size(0)}, "
                f"expected {data.edge_index.size(1)}."
            )

        return FragmentTreeFeatures.from_structure(
            data,
            node_graphs=mol_graph,
            edge_attr=edge_attr,
        )

    def _encode_sample_tree(
        self,
        ft_features: "FragmentTreeFeatures",
    ) -> Batch:
        """
        Build sample-wise tree graphs and encode them.

        Returns
        -------
        Batch
            PyG Batch with encoded tree node embeddings attached to x.
        """

        sample_tree_batch, condition_tree_repr, kept_sample_ids = (
            self._build_sample_tree_pyg_batch(ft_features)
        )

        tree_node_emb, tree_graph_repr = self.tree_encoder(
            sample_tree_batch,
            graph_repr=condition_tree_repr,
        )

        # Store encoded features in the batch itself.
        sample_tree_batch.x = tree_node_emb
        sample_tree_batch.tree_graph_repr = tree_graph_repr
        sample_tree_batch.condition_tree_repr = condition_tree_repr
        sample_tree_batch.kept_sample_ids = kept_sample_ids

        return sample_tree_batch

    def _validate_mol_encoder_output(self, mol_graph: Batch, structure: FragmentTreeStructure):
        if mol_graph.num_graphs != structure.num_nodes:
            raise ValueError(f"MolEncoder output has num_graphs={mol_graph.num_graphs}, but FragmentTreeStructure has n_nodes={structure.n_nodes}. These must match.")
        if mol_graph.x.size(0) != structure.node_graph.num_nodes:
            raise ValueError(f"MolEncoder output has x.size(0)={mol_graph.x.size(0)}, but FragmentTreeStructure's node_graph have num_nodes={structure.node_graph.num_nodes}. These must match.")
        if mol_graph.edge_index.size(1) != structure.node_graph.num_edges:
            raise ValueError(f"MolEncoder output has edge_index.size(1)={mol_graph.edge_index.size(1)}, but FragmentTreeStructure's node_graph have num_edges={structure.node_graph.num_edges}. These must match.")

    def _build_sample_tree_pyg_batch(
        self,
        ft_features: "FragmentTreeFeatures",
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
        """
        Build per-sample PyG fragment-tree graphs.

        This function converts a global FragmentTreeStructure into a PyG Batch
        where each graph corresponds to one MS/MS sample.

        Current FragmentTreeStructure specification
        -------------------------------------------
        st.edge_index:
            [2, E]
            Global fragment-tree edges.

        st.sample_edge_index:
            [2, L]
            row 0: sample index
            row 1: edge index

        st.sample_adduct_type_index:
            [S]
            Adduct type index for each sample.

        st.sample_ce_value:
            [S]
            Collision energy value for each sample.

        st.sample_precursor_edge_index_path:
            [P, D]
            Padded precursor edge paths.
            Padding value is -1.

        st.sample_precursor_path_index:
            [P]
            sample index for each precursor path.

        Returns
        -------
        sample_tree_batch:
            PyG Batch of per-sample fragment-tree graphs.

        condition_tree_repr:
            [G, tree_graph_repr_dim]
            MS2 condition representation aligned with sample_tree_batch graph order.

        kept_sample_ids:
            [G]
            Original sample ids used in sample_tree_batch.
        """

        structure = ft_features.structure

        global_node_features = ft_features.mol_x
        # [N, mol_graph_dim]

        global_edge_features = ft_features.edge_attr
        # [E, edge_dim]

        device = global_node_features.device

        if structure.edge_index.dim() != 2 or structure.edge_index.size(0) != 2:
            raise ValueError(
                "structure.edge_index must have shape [2, E], "
                f"got shape {tuple(structure.edge_index.shape)}."
            )

        num_global_edges = int(structure.edge_index.size(1))
        num_global_nodes = int(global_node_features.size(0))
        num_samples = int(structure.num_samples)

        if num_samples <= 0:
            raise ValueError(
                "FragmentTreeStructure must contain at least one sample."
            )

        if (
            global_edge_features.dim() != 2
            or global_edge_features.size(0) != num_global_edges
        ):
            raise ValueError(
                "ft_features.edge_attr must have shape [E, edge_dim]. "
                f"Got shape {tuple(global_edge_features.shape)}, "
                f"E={num_global_edges}."
            )

        sample_edge_index = structure.sample_edge_index.to(device).long()
        # [2, L]

        if sample_edge_index.dim() != 2 or sample_edge_index.size(0) != 2:
            raise ValueError(
                "structure.sample_edge_index must have shape [2, L], "
                f"got shape {tuple(sample_edge_index.shape)}."
            )

        sample_precursor_edge_paths = (
            structure.sample_precursor_edge_index_path.to(device).long()
        )
        # [P, D]

        sample_precursor_path_sample_index = (
            structure.sample_precursor_path_index.to(device).long()
        )
        # [P]

        if sample_precursor_edge_paths.dim() != 2:
            raise ValueError(
                "structure.sample_precursor_edge_index_path must be 2D, "
                f"got shape {tuple(sample_precursor_edge_paths.shape)}."
            )

        if sample_precursor_path_sample_index.dim() != 1:
            raise ValueError(
                "structure.sample_precursor_path_index must be 1D, "
                f"got shape {tuple(sample_precursor_path_sample_index.shape)}."
            )

        if (
            sample_precursor_edge_paths.size(0)
            != sample_precursor_path_sample_index.size(0)
        ):
            raise ValueError(
                "sample_precursor_edge_index_path and sample_precursor_path_index "
                "must have the same number of rows. "
                f"Got {sample_precursor_edge_paths.size(0)} and "
                f"{sample_precursor_path_sample_index.size(0)}."
            )

        global_edge_src = structure.edge_index[0].to(device).long()
        # [E]

        global_edge_dst = structure.edge_index[1].to(device).long()
        # [E]

        # -------------------------
        # Sample condition -> tree graph representation
        # -------------------------
        sample_condition_features = self._condition_encoder(
            structure.sample_adduct_type_index.to(device).long(),
            structure.sample_ce_value.to(device),
        )
        # [S, condition_dim]

        sample_condition_tree_repr = self.ms2_condition_to_tree_proj(
            sample_condition_features
        )
        # [S, tree_graph_repr_dim]

        if sample_condition_tree_repr.size(0) != num_samples:
            raise ValueError(
                "MS2ConditionEncoder output has invalid sample dimension: "
                f"got {sample_condition_tree_repr.size(0)}, "
                f"expected {num_samples}."
            )

        if sample_condition_tree_repr.size(1) != self.tree_encoder.graph_repr_dim:
            raise ValueError(
                "ms2_condition_to_tree_proj output has invalid feature dimension: "
                f"got {sample_condition_tree_repr.size(1)}, "
                f"expected {self.tree_encoder.graph_repr_dim}."
            )

        sample_data_list: List[Data] = []
        kept_sample_ids: List[int] = []

        local_precursor_sequence_parts: List[torch.Tensor] = []
        precursor_sequence_ptr: List[int] = [0]
        edge_ptr: List[int] = [0]

        max_precursor_path_width = int(sample_precursor_edge_paths.size(1))

        for sample_id in range(num_samples):
            # -------------------------
            # 1) Collect edges assigned to this sample
            # -------------------------
            sample_edge_mask = sample_edge_index[0] == sample_id

            sample_edge_ids = sample_edge_index[1][sample_edge_mask].long()
            sample_edge_ids = sample_edge_ids[sample_edge_ids >= 0]

            if sample_edge_ids.numel() > 0:
                sample_edge_ids = torch.unique(
                    sample_edge_ids,
                    sorted=True,
                )

            # -------------------------
            # 2) Collect precursor paths assigned to this sample
            # -------------------------
            precursor_path_mask = sample_precursor_path_sample_index == sample_id

            sample_precursor_paths = sample_precursor_edge_paths[
                precursor_path_mask
            ]
            # [P_s, D]

            if sample_precursor_paths.numel() > 0:
                precursor_edge_ids = sample_precursor_paths.reshape(-1)
                precursor_edge_ids = precursor_edge_ids[precursor_edge_ids >= 0]

                if precursor_edge_ids.numel() > 0:
                    precursor_edge_ids = torch.unique(
                        precursor_edge_ids,
                        sorted=True,
                    )
            else:
                precursor_edge_ids = torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=device,
                )

            # The local sample graph must contain both:
            #   - edges assigned to the sample
            #   - edges appearing in precursor paths
            global_edge_ids = torch.cat(
                [
                    sample_edge_ids,
                    precursor_edge_ids,
                ],
                dim=0,
            )

            if global_edge_ids.numel() > 0:
                global_edge_ids = torch.unique(
                    global_edge_ids,
                    sorted=True,
                )

            if global_edge_ids.numel() == 0:
                raise ValueError(
                    f"No edges found for sample {sample_id}. "
                    "Each sample must have at least one edge in sample_edge_index "
                    "or sample_precursor_edge_index_path."
                )

            if (
                global_edge_ids.min().item() < 0
                or global_edge_ids.max().item() >= num_global_edges
            ):
                raise IndexError(
                    f"global_edge_ids out of range for sample {sample_id}: "
                    f"min={global_edge_ids.min().item()}, "
                    f"max={global_edge_ids.max().item()}, "
                    f"E={num_global_edges}."
                )

            # -------------------------
            # 3) Collect nodes used by this sample
            # -------------------------
            global_src_nodes = global_edge_src[global_edge_ids]
            global_dst_nodes = global_edge_dst[global_edge_ids]

            global_node_ids = torch.unique(
                torch.cat(
                    [
                        global_src_nodes,
                        global_dst_nodes,
                    ],
                    dim=0,
                ),
                sorted=True,
            )
            # [N_s]

            if global_node_ids.numel() == 0:
                raise ValueError(
                    f"No nodes found for sample {sample_id}."
                )

            if (
                global_node_ids.min().item() < 0
                or global_node_ids.max().item() >= num_global_nodes
            ):
                raise IndexError(
                    f"global_node_ids out of range for sample {sample_id}: "
                    f"min={global_node_ids.min().item()}, "
                    f"max={global_node_ids.max().item()}, "
                    f"N={num_global_nodes}."
                )

            num_sample_nodes = int(global_node_ids.numel())

            # -------------------------
            # 4) Global node id -> local node id
            # -------------------------
            local_node_id_by_global_node_id = torch.full(
                (num_global_nodes,),
                -1,
                dtype=torch.long,
                device=device,
            )

            local_node_id_by_global_node_id[global_node_ids] = torch.arange(
                num_sample_nodes,
                dtype=torch.long,
                device=device,
            )

            # -------------------------
            # 5) Slice node features
            # -------------------------
            sample_node_features = global_node_features[global_node_ids]
            # [N_s, mol_graph_dim]

            # -------------------------
            # 6) Add node role features
            # -------------------------
            node_role_one_hot = torch.zeros(
                (num_sample_nodes, 3),
                dtype=torch.float32,
                device=device,
            )
            node_role_one_hot[:, 2] = 1.0
            # columns:
            #   0: precursor_path_node
            #   1: precursor_root_node
            #   2: normal_node

            precursor_root_global_node_ids = (
                self._get_precursor_root_node_ids_from_edge_paths(
                    edge_paths=sample_precursor_paths,
                    global_edge_src=global_edge_src,
                )
            )
            # [R_s]

            if precursor_root_global_node_ids.numel() > 0:
                precursor_root_local_node_ids = local_node_id_by_global_node_id[
                    precursor_root_global_node_ids
                ]

                precursor_root_local_node_ids = precursor_root_local_node_ids[
                    precursor_root_local_node_ids >= 0
                ]

                if precursor_root_local_node_ids.numel() > 0:
                    precursor_root_local_node_ids = torch.unique(
                        precursor_root_local_node_ids,
                        sorted=True,
                    )

                    node_role_one_hot[precursor_root_local_node_ids] = 0.0
                    node_role_one_hot[precursor_root_local_node_ids, 1] = 1.0

            precursor_path_global_node_ids = (
                self._get_node_ids_from_edge_paths(
                    edge_paths=sample_precursor_paths,
                    global_edge_src=global_edge_src,
                    global_edge_dst=global_edge_dst,
                )
            )
            # [Q_s]

            if precursor_path_global_node_ids.numel() > 0:
                precursor_path_local_node_ids = local_node_id_by_global_node_id[
                    precursor_path_global_node_ids
                ]

                precursor_path_local_node_ids = precursor_path_local_node_ids[
                    precursor_path_local_node_ids >= 0
                ]

                if precursor_path_local_node_ids.numel() > 0:
                    precursor_path_local_node_ids = torch.unique(
                        precursor_path_local_node_ids,
                        sorted=True,
                    )

                    # Do not overwrite precursor root nodes.
                    if precursor_root_global_node_ids.numel() > 0:
                        root_local_node_ids = local_node_id_by_global_node_id[
                            precursor_root_global_node_ids
                        ]
                        root_local_node_ids = root_local_node_ids[
                            root_local_node_ids >= 0
                        ]

                        if root_local_node_ids.numel() > 0:
                            precursor_path_local_node_ids = (
                                precursor_path_local_node_ids[
                                    ~torch.isin(
                                        precursor_path_local_node_ids,
                                        torch.unique(root_local_node_ids),
                                    )
                                ]
                            )

                    if precursor_path_local_node_ids.numel() > 0:
                        node_role_one_hot[precursor_path_local_node_ids] = 0.0
                        node_role_one_hot[precursor_path_local_node_ids, 0] = 1.0

            sample_node_features = torch.cat(
                [
                    sample_node_features,
                    node_role_one_hot,
                ],
                dim=1,
            )
            # [N_s, mol_graph_dim + 3]

            # -------------------------
            # 7) Slice and remap edges
            # -------------------------
            local_edge_id_by_global_edge_id = torch.full(
                (num_global_edges,),
                -1,
                dtype=torch.long,
                device=device,
            )

            local_edge_id_by_global_edge_id[global_edge_ids] = torch.arange(
                global_edge_ids.numel(),
                dtype=torch.long,
                device=device,
            )

            local_src_nodes = local_node_id_by_global_node_id[
                global_edge_src[global_edge_ids]
            ]

            local_dst_nodes = local_node_id_by_global_node_id[
                global_edge_dst[global_edge_ids]
            ]

            if (local_src_nodes < 0).any() or (local_dst_nodes < 0).any():
                raise ValueError(
                    f"Some edges for sample {sample_id} have endpoints "
                    "that do not map to local node ids."
                )

            sample_edge_index_local = torch.stack(
                [
                    local_src_nodes,
                    local_dst_nodes,
                ],
                dim=0,
            )
            # [2, E_s]

            sample_edge_features = global_edge_features[global_edge_ids]
            # [E_s, edge_dim]

            # -------------------------
            # 8) Build local precursor pathway sequence
            # -------------------------
            local_precursor_sequence = (
                self._convert_edge_paths_to_local_node_edge_sequences(
                    edge_paths=sample_precursor_paths,
                    global_edge_src=global_edge_src,
                    global_edge_dst=global_edge_dst,
                    local_node_id_by_global_node_id=(
                        local_node_id_by_global_node_id
                    ),
                    local_edge_id_by_global_edge_id=(
                        local_edge_id_by_global_edge_id
                    ),
                    max_path_width=max_precursor_path_width,
                    device=device,
                )
            )
            # [P_s, 2 * D + 1]

            self._validate_local_precursor_sequences(
                local_precursor_sequence=local_precursor_sequence,
                num_sample_nodes=num_sample_nodes,
                num_sample_edges=int(global_edge_ids.numel()),
                sample_id=sample_id,
            )

            local_precursor_sequence_parts.append(local_precursor_sequence)

            precursor_sequence_ptr.append(
                precursor_sequence_ptr[-1] + int(local_precursor_sequence.size(0))
            )

            edge_ptr.append(
                edge_ptr[-1] + int(sample_edge_index_local.size(1))
            )

            # -------------------------
            # 9) Build sample Data
            # -------------------------
            sample_main_adduct_type_index = structure.sample_adduct_type_index.to(device).long()[sample_id]
            sample_node_main_adduct_type_index = torch.full((num_sample_nodes,), int(sample_main_adduct_type_index.item()), dtype=torch.long, device=device)

            sample_data = Data(
                x=sample_node_features,
                edge_index=sample_edge_index_local,
                edge_attr=sample_edge_features,
            )

            sample_data.node_id_global = global_node_ids
            sample_data.edge_id_global = global_edge_ids
            sample_data.node_main_adduct_type_index = sample_node_main_adduct_type_index
            sample_data.sample_id = torch.tensor(sample_id, dtype=torch.long, device=device)

            sample_data_list.append(sample_data)
            kept_sample_ids.append(sample_id)

        if len(sample_data_list) == 0:
            raise ValueError("No sample graphs were created.")

        sample_tree_batch = Batch.from_data_list(sample_data_list)

        kept_sample_ids_tensor = torch.tensor(
            kept_sample_ids,
            dtype=torch.long,
            device=device,
        )

        condition_tree_repr = sample_condition_tree_repr[
            kept_sample_ids_tensor
        ]
        # [G, tree_graph_repr_dim]

        # -------------------------
        # Concatenate precursor pathway sequences
        # -------------------------
        if len(local_precursor_sequence_parts) == 0:
            precursor_pathway_seq = torch.empty(
                (0, 2 * max_precursor_path_width + 1),
                dtype=torch.long,
                device=device,
            )

            precursor_pathway_ptr = torch.zeros(
                (len(sample_data_list) + 1,),
                dtype=torch.long,
                device=device,
            )
        else:
            precursor_pathway_seq = torch.cat(
                local_precursor_sequence_parts,
                dim=0,
            )
            precursor_pathway_seq[precursor_pathway_seq < 0] = 0
            if torch.any(precursor_pathway_seq[:, 0] != 0):
                raise ValueError(
                    "All precursor pathway sequences have invalid node ids. "
                    "This indicates a bug in the local sequence conversion."
                )

            precursor_pathway_ptr = torch.tensor(
                precursor_sequence_ptr,
                dtype=torch.long,
                device=device,
            )

        edge_ptr_tensor = torch.tensor(
            edge_ptr,
            dtype=torch.long,
            device=device,
        )

        precursor_pathway_seq = self._convert_local_pathway_sequences_to_batch_indices(
            seq_local=precursor_pathway_seq,
            pathway_ptr=precursor_pathway_ptr,
            node_ptr=sample_tree_batch.ptr,
            edge_ptr=edge_ptr_tensor,
        )

        sample_tree_batch.precursor_pathway_seq = precursor_pathway_seq
        sample_tree_batch.precursor_pathway_ptr = precursor_pathway_ptr
        sample_tree_batch.edge_ptr = edge_ptr_tensor
        sample_tree_batch.kept_sample_ids = kept_sample_ids_tensor

        return sample_tree_batch, condition_tree_repr, kept_sample_ids_tensor

    @staticmethod
    def _get_last_valid_nodes_from_node_edge_sequences(
        node_edge_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the last valid node id from each node-edge-node sequence.

        Parameters
        ----------
        node_edge_sequences:
            [P, L]
            Node-edge-node sequences.
            Node ids are placed at columns 0, 2, 4, ...

        Returns
        -------
        Tensor
            [P]
            Last valid node id for each sequence.
        """

        if node_edge_sequences.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=node_edge_sequences.device)

        node_columns = node_edge_sequences[:, 0::2]
        # [P, num_node_positions]

        last_nodes: List[int] = []

        for row in node_columns:
            valid_nodes = row[row >= 0]

            if valid_nodes.numel() == 0:
                raise ValueError("Each precursor pathway must contain at least one valid node.")

            last_nodes.append(int(valid_nodes[-1].item()))

        return torch.tensor(last_nodes, dtype=torch.long, device=node_edge_sequences.device)

    @staticmethod
    def _get_precursor_root_node_ids_from_edge_paths(
        *,
        edge_paths: torch.Tensor,
        global_edge_src: torch.Tensor,
    ) -> torch.Tensor:
        """Get precursor root node ids from the first valid edge in each path."""

        device = global_edge_src.device

        if edge_paths.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        root_node_ids: List[int] = []

        for edge_path in edge_paths:
            valid_edge_ids = edge_path[edge_path >= 0]

            if valid_edge_ids.numel() == 0:
                continue

            first_edge_id = int(valid_edge_ids[0].item())
            root_node_ids.append(
                int(global_edge_src[first_edge_id].item())
            )

        if len(root_node_ids) == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        return torch.unique(
            torch.tensor(
                root_node_ids,
                dtype=torch.long,
                device=device,
            ),
            sorted=True,
        )


    @staticmethod
    def _get_node_ids_from_edge_paths(
        *,
        edge_paths: torch.Tensor,
        global_edge_src: torch.Tensor,
        global_edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Get all node ids appearing in edge paths."""

        device = global_edge_src.device

        if edge_paths.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        valid_edge_ids = edge_paths.reshape(-1)
        valid_edge_ids = valid_edge_ids[valid_edge_ids >= 0]

        if valid_edge_ids.numel() == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        node_ids = torch.cat(
            [
                global_edge_src[valid_edge_ids],
                global_edge_dst[valid_edge_ids],
            ],
            dim=0,
        )

        return torch.unique(
            node_ids,
            sorted=True,
        )


    @staticmethod
    def _convert_edge_paths_to_local_node_edge_sequences(
        *,
        edge_paths: torch.Tensor,
        global_edge_src: torch.Tensor,
        global_edge_dst: torch.Tensor,
        local_node_id_by_global_node_id: torch.Tensor,
        local_edge_id_by_global_edge_id: torch.Tensor,
        max_path_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert global edge paths to local node-edge-node sequences.

        Parameters
        ----------
        edge_paths:
            [P_s, D]
            Global edge ids padded by -1.

        Returns
        -------
        local_sequences:
            [P_s, 2 * D + 1]

            columns:
                0: local node id
                1: local edge id
                2: local node id
                3: local edge id
                ...
        """

        path_count = int(edge_paths.size(0))
        sequence_width = 2 * int(max_path_width) + 1

        if path_count == 0:
            return torch.empty(
                (0, sequence_width),
                dtype=torch.long,
                device=device,
            )

        local_sequences = torch.full(
            (path_count, sequence_width),
            -1,
            dtype=torch.long,
            device=device,
        )

        for path_index in range(path_count):
            edge_path = edge_paths[path_index]
            valid_edge_ids = edge_path[edge_path >= 0]

            if valid_edge_ids.numel() == 0:
                continue

            first_edge_id = int(valid_edge_ids[0].item())
            first_global_src_node = int(global_edge_src[first_edge_id].item())

            first_local_src_node = int(
                local_node_id_by_global_node_id[first_global_src_node].item()
            )

            if first_local_src_node < 0:
                raise ValueError(
                    f"First source node {first_global_src_node} does not map "
                    "to a local node id."
                )

            local_sequences[path_index, 0] = first_local_src_node

            for step_index, edge_id_tensor in enumerate(valid_edge_ids):
                global_edge_id = int(edge_id_tensor.item())

                local_edge_id = int(
                    local_edge_id_by_global_edge_id[global_edge_id].item()
                )

                if local_edge_id < 0:
                    raise ValueError(
                        f"Edge {global_edge_id} does not map to a local edge id."
                    )

                global_dst_node = int(global_edge_dst[global_edge_id].item())

                local_dst_node = int(
                    local_node_id_by_global_node_id[global_dst_node].item()
                )

                if local_dst_node < 0:
                    raise ValueError(
                        f"Destination node {global_dst_node} does not map "
                        "to a local node id."
                    )

                edge_column = 2 * step_index + 1
                node_column = 2 * step_index + 2

                if edge_column >= sequence_width or node_column >= sequence_width:
                    raise IndexError(
                        "Precursor path is longer than max_path_width."
                    )

                local_sequences[path_index, edge_column] = local_edge_id
                local_sequences[path_index, node_column] = local_dst_node

        return local_sequences


    @staticmethod
    def _validate_local_precursor_sequences(
        *,
        local_precursor_sequence: torch.Tensor,
        num_sample_nodes: int,
        num_sample_edges: int,
        sample_id: int,
    ) -> None:
        """Validate local node and edge ids in precursor pathway sequences."""

        if local_precursor_sequence.numel() == 0:
            return

        node_values = local_precursor_sequence[:, 0::2].reshape(-1)
        node_values = node_values[node_values != -1]

        if node_values.numel() > 0:
            if node_values.min().item() < 0:
                raise IndexError(
                    f"Local node ids in precursor path are negative "
                    f"for sample {sample_id}."
                )

            if node_values.max().item() >= num_sample_nodes:
                raise IndexError(
                    f"Local node ids in precursor path are out of range "
                    f"for sample {sample_id}."
                )

        edge_values = local_precursor_sequence[:, 1::2].reshape(-1)
        edge_values = edge_values[edge_values != -1]

        if edge_values.numel() > 0:
            if edge_values.min().item() < 0:
                raise IndexError(
                    f"Local edge ids in precursor path are negative "
                    f"for sample {sample_id}."
                )

            if edge_values.max().item() >= num_sample_edges:
                raise IndexError(
                    f"Local edge ids in precursor path are out of range "
                    f"for sample {sample_id}."
                )

    def convert_local_pathway_sequences_to_batch_indices(
        seq_local: torch.Tensor,
        pathway_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        edge_ptr: torch.Tensor,
        *,
        edge_positions: str = "odd0",
    ) -> torch.Tensor:
        """
        Convert local node/edge ids in pathway sequences to PyG-batch-global ids.

        seq_local:
            [P, L]
            Local node-edge-node sequences.

        pathway_ptr:
            [G + 1]
            CSR pointer that groups pathway rows by sample graph.

        node_ptr:
            [G + 1]
            PyG Batch node pointer.

        edge_ptr:
            [G + 1]
            PyG Batch edge pointer.
        """

        device = seq_local.device

        seq = seq_local.long().clone()
        pathway_ptr = pathway_ptr.long().to(device)
        node_ptr = node_ptr.long().to(device)
        edge_ptr = edge_ptr.long().to(device)

        num_sequences, sequence_width = seq.shape
        num_graphs = int(pathway_ptr.numel() - 1)

        if node_ptr.numel() != num_graphs + 1:
            raise ValueError(
                "node_ptr must have shape [G + 1] aligned with pathway_ptr."
            )

        if edge_ptr.numel() != num_graphs + 1:
            raise ValueError(
                "edge_ptr must have shape [G + 1] aligned with pathway_ptr."
            )

        sequence_counts = (
            pathway_ptr[1:] - pathway_ptr[:-1]
        ).clamp_min(0)

        graph_id_by_sequence = torch.repeat_interleave(
            torch.arange(
                num_graphs,
                device=device,
                dtype=torch.long,
            ),
            sequence_counts,
        )
        # [P]

        if graph_id_by_sequence.numel() != num_sequences:
            raise ValueError(
                "pathway_ptr does not match seq_local rows."
            )

        node_offset_by_sequence = node_ptr[graph_id_by_sequence]
        edge_offset_by_sequence = edge_ptr[graph_id_by_sequence]

        positions = torch.arange(
            sequence_width,
            device=device,
        )

        if edge_positions == "odd0":
            is_edge_position = positions % 2 == 1
        elif edge_positions == "even0":
            is_edge_position = positions % 2 == 0
        else:
            raise ValueError(
                'edge_positions must be "odd0" or "even0".'
            )

        is_edge = is_edge_position.view(1, sequence_width).expand(
            num_sequences,
            sequence_width,
        )

        is_node = ~is_edge
        is_valid = seq >= 0

        node_valid = is_valid & is_node
        edge_valid = is_valid & is_edge

        if node_valid.any():
            node_offsets = node_offset_by_sequence.view(
                num_sequences,
                1,
            ).expand(
                num_sequences,
                sequence_width,
            )

            seq[node_valid] = seq[node_valid] + node_offsets[node_valid]

        if edge_valid.any():
            edge_offsets = edge_offset_by_sequence.view(
                num_sequences,
                1,
            ).expand(
                num_sequences,
                sequence_width,
            )

            seq[edge_valid] = seq[edge_valid] + edge_offsets[edge_valid]

        return seq

    def _compute_fragment_generation_probabilities(
        self,
        *,
        ft_features: FragmentTreeFeatures,
        sample_tree_batch: Batch,
    ) -> FragmentGenerationProbabilities:
        """
        Compute fragment generation probabilities from encoded sample tree batch.
        """

        tree_node_emb = sample_tree_batch.x
        # [N_tree, tree_dim]

        (
            precursor_logit,
            p_precursor,
            precursor_expand_logit,
            p_precursor_expand,
            p_precursor_stop,
        ) = self._compute_precursor_probabilities(
            sample_tree_batch=sample_tree_batch,
            tree_node_emb=tree_node_emb,
        )

        edge_src, edge_dst = sample_tree_batch.edge_index
        # [E_tree], [E_tree]

        edge_input = torch.cat(
            [
                tree_node_emb[edge_src],
                tree_node_emb[edge_dst],
                sample_tree_batch.edge_attr,
            ],
            dim=-1,
        )
        # [E_tree, tree_dim * 2 + edge_dim]

        edge_logit = self.edge_select_head(edge_input).squeeze(-1)
        # [E_tree]

        p_edge = _softmax_by_group(
            logits=edge_logit,
            group=edge_src,
            num_groups=int(tree_node_emb.size(0)),
        )
        # [E_tree]

        expand_logit = self.node_expand_head(tree_node_emb).squeeze(-1)
        # [N_tree]

        p_expand = torch.sigmoid(expand_logit)
        # [N_tree]

        p_stop = 1.0 - p_expand
        # [N_tree]

        ion = self._compute_flat_adduct_node_choice_probabilities(
            node_emb=tree_node_emb,
            node_main_adduct_type_index=sample_tree_batch.node_main_adduct_type_index,
            head_by_adduct_str=self.node_ion_heads_by_adduct_str,
            flat_candidates=self.ion_flat_candidates,
            candidate_slice_by_adduct=self.ion_candidate_slice_by_adduct,
        )

        unsaturation = self._compute_flat_adduct_node_choice_probabilities(
            node_emb=tree_node_emb,
            node_main_adduct_type_index=sample_tree_batch.node_main_adduct_type_index,
            head_by_adduct_str=self.node_unsaturation_heads_by_adduct_str,
            flat_candidates=self.unsaturation_flat_candidates,
            candidate_slice_by_adduct=self.unsaturation_candidate_slice_by_adduct,
        )

        radical = self._compute_flat_adduct_node_choice_probabilities(
            node_emb=tree_node_emb,
            node_main_adduct_type_index=sample_tree_batch.node_main_adduct_type_index,
            head_by_adduct_str=self.node_radical_heads_by_adduct_str,
            flat_candidates=self.radical_flat_candidates,
            candidate_slice_by_adduct=self.radical_candidate_slice_by_adduct,
        )

        structure = ft_features.structure

        node_graph_index = sample_tree_batch.batch
        # [N_tree]

        node_sample_ids = sample_tree_batch.kept_sample_ids[node_graph_index]
        # [N_tree]

        node_main_adduct_type_index = structure.sample_adduct_type_index.to(
            tree_node_emb.device
        )[node_sample_ids]
        # [N_tree]

        return FragmentGenerationProbabilities(
            precursor_logit=precursor_logit,
            p_precursor=p_precursor,
            precursor_expand_logit=precursor_expand_logit,
            p_precursor_expand=p_precursor_expand,
            p_precursor_stop=p_precursor_stop,
            edge_logit=edge_logit,
            p_edge=p_edge,
            expand_logit=expand_logit,
            p_expand=p_expand,
            p_stop=p_stop,
            ion=ion,
            unsaturation=unsaturation,
            radical=radical,
        )

    def _compute_precursor_probabilities(
        self,
        *,
        sample_tree_batch: Batch,
        tree_node_emb: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute precursor selection and precursor expand / stop probabilities.

        Returns
        -------
        precursor_logit:
            [P]

        p_precursor:
            [P]
            Sum is 1 within each sample graph.

        precursor_expand_logit:
            [P]

        p_precursor_expand:
            [P]

        p_precursor_stop:
            [P]
        """

        if not hasattr(sample_tree_batch, "precursor_pathway_seq"):
            raise ValueError("sample_tree_batch must have precursor_pathway_seq.")

        if not hasattr(sample_tree_batch, "precursor_pathway_ptr"):
            raise ValueError("sample_tree_batch must have precursor_pathway_ptr.")

        device = tree_node_emb.device

        precursor_pathway_seq = sample_tree_batch.precursor_pathway_seq.to(device).long()
        # [P, L]

        precursor_pathway_ptr = sample_tree_batch.precursor_pathway_ptr.to(device).long()
        # [G + 1]

        if precursor_pathway_seq.numel() == 0:
            raise ValueError("precursor_pathway_seq must not be empty.")

        row_counts = precursor_pathway_ptr[1:] - precursor_pathway_ptr[:-1]
        # [G]

        if (row_counts <= 0).any():
            raise ValueError(
                "Each sample graph must have at least one precursor pathway. "
                f"row_counts={row_counts.detach().cpu().tolist()}"
            )

        precursor_node_index = self._get_last_valid_nodes_from_node_edge_sequences(
            precursor_pathway_seq
        )
        # [P]

        if precursor_node_index.numel() != precursor_pathway_seq.size(0):
            raise ValueError(
                "Failed to get one precursor node for each precursor pathway."
            )

        precursor_node_emb = tree_node_emb[precursor_node_index]
        # [P, tree_dim]

        precursor_logit = self.precursor_select_head(precursor_node_emb).squeeze(-1)
        # [P]

        precursor_expand_logit = self.precursor_expand_head(precursor_node_emb).squeeze(-1)
        # [P]

        graph_index_by_precursor = torch.repeat_interleave(
            torch.arange(row_counts.numel(), dtype=torch.long, device=device),
            row_counts,
        )
        # [P]

        if graph_index_by_precursor.numel() != precursor_logit.numel():
            raise ValueError(
                "precursor_pathway_ptr does not match precursor_pathway_seq rows."
            )

        p_precursor = _softmax_by_group(
            logits=precursor_logit,
            group=graph_index_by_precursor,
            num_groups=int(row_counts.numel()),
        )
        # [P]

        p_precursor_expand = torch.sigmoid(precursor_expand_logit)
        # [P]

        p_precursor_stop = 1.0 - p_precursor_expand
        # [P]

        return (
            precursor_logit,
            p_precursor,
            precursor_expand_logit,
            p_precursor_expand,
            p_precursor_stop,
        )

    def _compute_flat_adduct_node_choice_probabilities(
        self,
        *,
        node_emb: Tensor,
        node_main_adduct_type_index: Tensor,
        head_by_adduct_str: nn.ModuleDict,
        flat_candidates: np.ndarray,
        candidate_slice_by_adduct: Dict[Adduct, slice],
    ) -> FlatAdductNodeChoiceProbabilities:
        """
        Compute flat node-wise choice probabilities.

        Invalid columns are assigned:
            logit = -inf
            prob = 0

        Parameters
        ----------
        node_emb:
            [N_tree, tree_dim]

        node_main_adduct_type_index:
            [N_tree]

        head_by_adduct_str:
            ModuleDict for adduct-specific heads.

        flat_candidates:
            np.ndarray with shape [C_flat, 2]

            flat_candidates[:, 0]:
                main adduct type

            flat_candidates[:, 1]:
                candidate adduct

        candidate_slice_by_adduct:
            Dict from main adduct type to flat column slice.

        Returns
        -------
        FlatAdductNodeChoiceProbabilities
        """

        device = node_emb.device
        dtype = node_emb.dtype

        num_nodes = int(node_emb.size(0))
        num_choices = int(flat_candidates.shape[0])

        flat_logit = torch.full(
            (num_nodes, num_choices),
            -float("inf"),
            dtype=dtype,
            device=device,
        )

        flat_prob = torch.zeros(
            (num_nodes, num_choices),
            dtype=dtype,
            device=device,
        )

        valid_mask = torch.zeros(
            (num_nodes, num_choices),
            dtype=torch.bool,
            device=device,
        )

        for adduct_index, main_adduct_type in self.main_adduct_types.items():
            adduct_key = str(main_adduct_type)

            if adduct_key not in head_by_adduct_str:
                continue

            node_index = (
                node_main_adduct_type_index == adduct_index
            ).nonzero(
                as_tuple=False,
            ).view(-1)
            # [N_adduct]

            if node_index.numel() == 0:
                continue

            if main_adduct_type not in candidate_slice_by_adduct:
                raise KeyError(
                    f"No candidate slice found for adduct type: {main_adduct_type}"
                )

            candidate_slice = candidate_slice_by_adduct[main_adduct_type]

            candidate_index = torch.arange(
                candidate_slice.start,
                candidate_slice.stop,
                dtype=torch.long,
                device=device,
            )
            # [C_adduct]

            node_emb_for_adduct = node_emb[node_index]
            # [N_adduct, tree_dim]

            local_logit = head_by_adduct_str[adduct_key](
                node_emb_for_adduct
            )
            # [N_adduct, C_adduct]

            if local_logit.size(1) != candidate_index.numel():
                raise ValueError(
                    "Head output size and candidate slice size mismatch. "
                    f"adduct_type={main_adduct_type}, "
                    f"local_logit.size(1)={local_logit.size(1)}, "
                    f"candidate_count={candidate_index.numel()}."
                )

            local_prob = torch.softmax(
                local_logit,
                dim=-1,
            )
            # [N_adduct, C_adduct]

            flat_logit[
                node_index[:, None],
                candidate_index[None, :],
            ] = local_logit

            flat_prob[
                node_index[:, None],
                candidate_index[None, :],
            ] = local_prob

            valid_mask[
                node_index[:, None],
                candidate_index[None, :],
            ] = True

        return FlatAdductNodeChoiceProbabilities(
            logit=flat_logit,
            prob=flat_prob,
            valid_mask=valid_mask,
        )

    @staticmethod
    def _make_node_choice_head(
        *,
        tree_dim: int,
        num_candidates: int,
        dropout: float,
    ) -> nn.Module:
        """Create a node-wise candidate selection head."""

        if num_candidates <= 0:
            raise ValueError(
                "num_candidates must be positive."
            )

        return nn.Sequential(
            nn.Linear(
                tree_dim,
                tree_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                tree_dim,
                num_candidates,
            ),
        )

    @staticmethod
    def _convert_local_pathway_sequences_to_batch_indices(
        *,
        seq_local: torch.Tensor,
        pathway_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        edge_ptr: torch.Tensor,
        edge_positions: str = "odd0",
        require_root: bool = True,
    ) -> torch.Tensor:
        """
        Convert local node/edge ids in pathway sequences to PyG-batch-global ids.

        Parameters
        ----------
        seq_local:
            [P, L]
            Local node-edge-node sequences.

            Example:
                [local_node, local_edge, local_node]

        pathway_ptr:
            [G + 1]
            Pointer that groups pathway rows by graph.

        node_ptr:
            [G + 1]
            PyG Batch node pointer.

        edge_ptr:
            [G + 1]
            Edge pointer for concatenated edge_attr / edge ids.

        edge_positions:
            "odd0":
                Edge ids are at positions 1, 3, 5, ...
                This matches [node, edge, node, edge, node].

            "even0":
                Edge ids are at positions 0, 2, 4, ...

        require_root:
            If True, seq_local[:, 0] must be a valid local root node id.

        Returns
        -------
        torch.Tensor
            [P, L]
            Pathway sequences converted to batch-global node/edge ids.
        """

        device = seq_local.device

        seq = seq_local.long().clone()
        pathway_ptr = pathway_ptr.long().to(device)
        node_ptr = node_ptr.long().to(device)
        edge_ptr = edge_ptr.long().to(device)

        if seq.dim() != 2:
            raise ValueError(f"seq_local must be 2D, got shape {tuple(seq.shape)}.")

        if pathway_ptr.dim() != 1:
            raise ValueError(f"pathway_ptr must be 1D, got shape {tuple(pathway_ptr.shape)}.")

        if node_ptr.dim() != 1:
            raise ValueError(f"node_ptr must be 1D, got shape {tuple(node_ptr.shape)}.")

        if edge_ptr.dim() != 1:
            raise ValueError(f"edge_ptr must be 1D, got shape {tuple(edge_ptr.shape)}.")

        num_sequences, sequence_width = seq.shape
        num_graphs = int(pathway_ptr.numel() - 1)

        if num_graphs < 0:
            raise ValueError("pathway_ptr must have at least one element.")

        if pathway_ptr.numel() != num_graphs + 1:
            raise ValueError(
                "pathway_ptr must have shape [G + 1]. "
                f"Got pathway_ptr={tuple(pathway_ptr.shape)}."
            )

        if node_ptr.numel() != num_graphs + 1:
            raise ValueError(
                "node_ptr must have shape [G + 1] aligned with pathway_ptr. "
                f"Got node_ptr={tuple(node_ptr.shape)}, pathway_ptr={tuple(pathway_ptr.shape)}."
            )

        if edge_ptr.numel() != num_graphs + 1:
            raise ValueError(
                "edge_ptr must have shape [G + 1] aligned with pathway_ptr. "
                f"Got edge_ptr={tuple(edge_ptr.shape)}, pathway_ptr={tuple(pathway_ptr.shape)}."
            )

        if num_sequences == 0:
            return seq

        if require_root and (seq[:, 0] < 0).any():
            bad_rows = (seq[:, 0] < 0).nonzero(as_tuple=False).view(-1)
            raise ValueError(
                "Each pathway sequence must start with a valid local root node. "
                f"bad_rows={bad_rows.detach().cpu().tolist()}."
            )

        sequence_counts = pathway_ptr[1:] - pathway_ptr[:-1]
        # [G]

        if (sequence_counts < 0).any():
            raise ValueError("pathway_ptr must be non-decreasing.")

        graph_id_by_sequence = torch.repeat_interleave(
            torch.arange(num_graphs, dtype=torch.long, device=device),
            sequence_counts,
        )
        # [P]

        if graph_id_by_sequence.numel() != num_sequences:
            raise ValueError(
                "pathway_ptr does not match seq_local rows. "
                f"Expected {num_sequences} rows, but pathway_ptr represents {graph_id_by_sequence.numel()} rows."
            )

        positions = torch.arange(sequence_width, dtype=torch.long, device=device)

        if edge_positions == "odd0":
            is_edge_position = positions % 2 == 1
        elif edge_positions == "even0":
            is_edge_position = positions % 2 == 0
        else:
            raise ValueError('edge_positions must be "odd0" or "even0".')

        is_edge = is_edge_position.view(1, sequence_width).expand(num_sequences, sequence_width)
        is_node = ~is_edge
        is_valid = seq >= 0

        node_offset_by_sequence = node_ptr[graph_id_by_sequence]
        edge_offset_by_sequence = edge_ptr[graph_id_by_sequence]

        node_valid = is_valid & is_node
        edge_valid = is_valid & is_edge

        if node_valid.any():
            node_offsets = node_offset_by_sequence.view(num_sequences, 1).expand(num_sequences, sequence_width)
            seq[node_valid] = seq[node_valid] + node_offsets[node_valid]

        if edge_valid.any():
            edge_offsets = edge_offset_by_sequence.view(num_sequences, 1).expand(num_sequences, sequence_width)
            seq[edge_valid] = seq[edge_valid] + edge_offsets[edge_valid]

        return seq

def _softmax_by_group(
    logits: Tensor,
    group: Tensor,
    num_groups: int,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute softmax over values that belong to the same group.

    Parameters
    ----------
    logits:
        [N]

    group:
        [N]
        group[i] is the group id of logits[i].

    num_groups:
        Number of groups.

    Returns
    -------
    Tensor
        [N]
        Softmax-normalized values within each group.
    """

    if logits.numel() == 0:
        return logits

    if logits.dim() != 1:
        raise ValueError(
            f"logits must be 1D, got shape {tuple(logits.shape)}."
        )

    if group.dim() != 1:
        raise ValueError(
            f"group must be 1D, got shape {tuple(group.shape)}."
        )

    if logits.size(0) != group.size(0):
        raise ValueError(
            "logits and group must have the same length. "
            f"Got logits={logits.size(0)}, group={group.size(0)}."
        )

    if group.min().item() < 0 or group.max().item() >= num_groups:
        raise IndexError(
            "group index out of range. "
            f"min={group.min().item()}, "
            f"max={group.max().item()}, "
            f"num_groups={num_groups}."
        )

    max_per_group = torch.full(
        (num_groups,),
        -float("inf"),
        dtype=logits.dtype,
        device=logits.device,
    )

    max_per_group.scatter_reduce_(
        dim=0,
        index=group,
        src=logits,
        reduce="amax",
        include_self=True,
    )

    shifted_logits = logits - max_per_group[group]
    exp_logits = torch.exp(shifted_logits)

    denom = torch.zeros(
        (num_groups,),
        dtype=logits.dtype,
        device=logits.device,
    )

    denom.scatter_add_(
        dim=0,
        index=group,
        src=exp_logits,
    )

    return exp_logits / (denom[group] + eps)