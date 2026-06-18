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
            fp_features = FragmentTreeFeatures.from_structure(data, node_graphs=mol_graph, edge_attr=edge_attr)
        elif isinstance(data, FragmentTreeFeatures):
            fp_features = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        return fp_features


    def _validate_mol_encoder_output(self, mol_graph: Batch, structure: FragmentTreeStructure):
        if mol_graph.num_graphs != structure.num_nodes:
            raise ValueError(f"MolEncoder output has num_graphs={mol_graph.num_graphs}, but FragmentTreeStructure has n_nodes={structure.n_nodes}. These must match.")
        if mol_graph.x.size(0) != structure.node_graph.num_nodes:
            raise ValueError(f"MolEncoder output has x.size(0)={mol_graph.x.size(0)}, but FragmentTreeStructure's node_graph have num_nodes={structure.node_graph.num_nodes}. These must match.")
        if mol_graph.edge_index.size(1) != structure.node_graph.num_edges:
            raise ValueError(f"MolEncoder output has edge_index.size(1)={mol_graph.edge_index.size(1)}, but FragmentTreeStructure's node_graph have num_edges={structure.node_graph.num_edges}. These must match.")