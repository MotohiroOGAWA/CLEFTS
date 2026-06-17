import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Union, Optional

from ...libs.mmkit.mmkit import Adduct
from ...domain.fragment import Fragmenter
from ...domain.mass import parse_ce_to_ev
from ..common.torch_utils.model_base import ModelBase
from ..input.fragment_tree_structure import FragmentTreeStructure

from ..mol import MolEncoder

class CleftsSpecGen(ModelBase):
    def __init__(self,
                 mol_encoder_params:Dict,
                 fragmenter_params:Dict,
                 dropout: float,
                 ):
        super(CleftsSpecGen, self).__init__(
            ignore_config_keys=[],
            **{k: v for k, v in locals().items() if k != 'self'}
        )

        mol_encoder_params = mol_encoder_params.copy()
        mol_encoder_params['dropout'] = dropout
        self.mol_encoder = MolEncoder(**mol_encoder_params)
        self.fragmenter = Fragmenter.from_dict(fragmenter_params)

        pass

    def get_index_by_adduct_type(self, adduct_type: Adduct) -> int:
        return self.fragmenter.get_index_by_adduct_type(adduct_type)

    @staticmethod
    def parse_ce_to_ev(ce: str, precursor_mz: float, instrument: str = None) -> Optional[float]:
        ev = parse_ce_to_ev(ce, precursor_mz, instrument)
        return ev
    
    def forward(self, structure: FragmentTreeStructure):
        if isinstance(structure, FragmentTreeStructure):
            mol_graph = self.mol_encoder(structure.node_graph)
            pass