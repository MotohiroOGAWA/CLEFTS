"""Selection features have no dependency on materialized target graphs."""
from __future__ import annotations
import torch
from dataclasses import dataclass, replace
from torch import Tensor, nn
from .components.condition.adduct_embedding import AdductEmbeddingLayer
from .components.condition.collision_energy_feature import CollisionEnergyFeatureLayer
from .components.action.action_encoder import ActionEncoder
from .components.action.action_decoder import ActionPool, BranchingCleavageDecoder, build_action_pool
from ..input.source_action_structure import MAX_NEIGHBORHOOD_HOP


@dataclass(frozen=True)
class SourceActionFeatures:
    action_h: Tensor
    branch_main_adduct_h: Tensor
    pool: ActionPool
    source_embeddings: Tensor | None = None
    sample_main_adduct_h: Tensor | None = None
    collision_energy_h: Tensor | None = None


class SourceActionFeatureModel(nn.Module):
    def __init__(self, mol_encoder: nn.Module, category_sizes: tuple[int, int, int],
                 hidden_dim: int = 128, branch_main_adduct_dim: int = 128,
                 max_action_count: int = 3, num_heads: int = 4,
                 state_num_layers: int = 2, state_dropout: float = 0.0,
                 adduct_encoder: nn.Module | None = None,
                 collision_energy_encoder: nn.Module | None = None,
                 max_roles: int = 64,
                 branch_path_threshold: float = 0.0, max_fragment_nodes: int = 100,
                 intensity_main_adduct_dim: int | None = None,
                 action_neighborhood_mode: str = "hop_pooling",
                 action_neighborhood_max_hop: int = 3,
                 ) -> None:
        super().__init__()
        if not 1 <= action_neighborhood_max_hop <= MAX_NEIGHBORHOOD_HOP:
            raise ValueError(f"action_neighborhood_max_hop must be between 1 and {MAX_NEIGHBORHOOD_HOP} "
                              "(prepared datasets only carry hop neighbors up to that distance)")
        self.mol_encoder = mol_encoder
        self.action_encoder = ActionEncoder(mol_encoder.node_dim, mol_encoder.graph_dim, hidden_dim, category_sizes, num_heads,
            max_roles, action_neighborhood_mode=action_neighborhood_mode, action_neighborhood_max_hop=action_neighborhood_max_hop)
        if not isinstance(adduct_encoder,AdductEmbeddingLayer):
            raise TypeError("Branching requires an explicit main-adduct encoder")
        if not isinstance(collision_energy_encoder,CollisionEnergyFeatureLayer):
            raise TypeError("Intensity prediction requires a separate collision-energy encoder")
        self.adduct_encoder=adduct_encoder
        self.collision_energy_encoder=collision_energy_encoder
        self.main_adduct_projection=nn.Sequential(nn.Linear(adduct_encoder.feature_dim,branch_main_adduct_dim),nn.GELU(),nn.LayerNorm(branch_main_adduct_dim))
        intensity_main_adduct_dim=intensity_main_adduct_dim or branch_main_adduct_dim
        self.intensity_main_adduct_projection=nn.Sequential(nn.Linear(adduct_encoder.feature_dim,intensity_main_adduct_dim),nn.GELU(),nn.LayerNorm(intensity_main_adduct_dim))
        self.decoder = BranchingCleavageDecoder(hidden_dim,branch_main_adduct_dim,max_action_count,num_heads,
            state_num_layers,state_dropout,branch_path_threshold,max_fragment_nodes)
        self.max_action_count = max_action_count

    def freeze_mol_encoder(self) -> None:
        self._freeze_mol_encoder = True
        self.mol_encoder.eval()
        for parameter in self.mol_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_mol_encoder", False):
            self.mol_encoder.eval()
        return self

    def encode_static(self,data,max_graphs=128):
        """Bound source encoding, then share action tokens across prediction chunks."""
        from torch_geometric.data import Batch
        device=next(self.parameters()).device
        graphs=data.source_graph.to_data_list()
        atoms=[];molecules=[]
        for start in range(0,len(graphs),max_graphs):
            encoded=self.mol_encoder(Batch.from_data_list(graphs[start:start+max_graphs]).to(device))
            atoms.append(encoded.x);molecules.append(encoded.embeddings)
        source_x=torch.cat(atoms);source_h=torch.cat(molecules)
        kwargs={name:getattr(data,name).to(device) for name in ('action_type','action_tree_index','action_source_atom_ptr','action_source_atom_index','action_static_features','action_source_atom_features')}
        kwargs['action_source_atom_hops']=self._hop_tensors(data,device)
        actions=self.action_encoder(source_atom_h=source_x,source_mol_h=source_h,**kwargs)
        return actions,source_h

    def _hop_tensors(self,data,device=None):
        hops=[(getattr(data,f'action_source_atom_hop{hop}_ptr'),getattr(data,f'action_source_atom_hop{hop}_index'))
              for hop in range(1,self.action_encoder.action_neighborhood_max_hop+1)]
        return [(ptr.to(device),index.to(device)) for ptr,index in hops] if device is not None else hops

    def forward(self, data: object, training_pool: bool = False, static_features: tuple[Tensor, Tensor] | None = None) -> SourceActionFeatures:
        if data.max_action_role_count > self.action_encoder.roles.num_embeddings:
            raise ValueError("This data's max_action_role_count exceeds the model's role embedding size; "
                              "max_roles is derived automatically from prepared training data, so retrain "
                              "against a dataset that matches this checkpoint's action universe")
        source = self.mol_encoder(data.source_graph) if static_features is None else None
        action = self.action_encoder(source_atom_h=source.x, source_mol_h=source.embeddings,
            action_type=data.action_type, action_tree_index=data.action_tree_index,
            action_source_atom_ptr=data.action_source_atom_ptr, action_source_atom_index=data.action_source_atom_index,
            action_static_features=data.action_static_features,action_source_atom_features=data.action_source_atom_features,
            action_source_atom_hops=self._hop_tensors(data)) if static_features is None else static_features[0]
        source_embeddings = source.embeddings if source is not None else static_features[1]
        sample_adduct=self.intensity_main_adduct_projection(self.adduct_encoder(data.condition_features[:,0].long()))
        collision_energy=self.collision_energy_encoder(data.condition_features[:,1])
        group_adduct_index=(data.branch_group_adduct_index if data.branch_group_adduct_index.numel()
                            else data.condition_features[:,0].long())
        condition=self.main_adduct_projection(self.adduct_encoder(group_adduct_index))
        # Static action tokens are shared by all CE/adduct conditions.
        pool=build_action_pool(action,data)
        return SourceActionFeatures(action,condition,pool,source_embeddings,sample_adduct,collision_energy)
