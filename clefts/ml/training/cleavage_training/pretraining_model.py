from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor

from ...input.fragment_tree_features import FragmentTreeFeatures
from ...input.fragment_tree_structure import FragmentTreeStructure
from ...mol.mol_encoder import MolEncoder
from ...specgen.components.cleavage.cleavage_edge_feature_net import CleavageEdgeFeatureNet
from .dataset import IGNORE_INDEX, CleavageEdgeTargets, build_cleavage_edge_targets


@dataclass(frozen=True)
class CleavagePretrainingOutput:
    loss: Tensor
    metrics: Dict[str, float]
    event_attr: Tensor


class CleavagePretrainingModel(nn.Module):
    """Pretrain cleavage-edge features on fragment-tree structures.

    The model reuses the same design boundary as specgen:
    ``MolEncoder`` creates source/target fragment node and atom embeddings,
    ``CleavageEdgeFeatureNet.encode_events`` creates one embedding per cleavage
    event, and lightweight heads predict cleavage type plus cleavage location.
    """

    def __init__(
        self,
        *,
        mol_encoder: MolEncoder,
        cleavage_edge_fnet: CleavageEdgeFeatureNet,
        num_patterns: int,
        num_reactions: int,
        num_product_molecules: int,
        hidden_dim: int = 256,
        observed_edge_loss_weight: float = 1.0,
        pattern_loss_weight: float = 1.0,
        reaction_loss_weight: float = 1.0,
        product_loss_weight: float = 1.0,
        compound_identity_loss_weight: float = 1.0,
        compound_identity_temperature: float = 0.1,
        atom_location_loss_weight: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.cleavage_edge_fnet = cleavage_edge_fnet
        self.num_patterns = int(num_patterns)
        self.num_reactions = int(num_reactions)
        self.num_product_molecules = int(num_product_molecules)
        self.observed_edge_loss_weight = float(observed_edge_loss_weight)
        self.pattern_loss_weight = float(pattern_loss_weight)
        self.reaction_loss_weight = float(reaction_loss_weight)
        self.product_loss_weight = float(product_loss_weight)
        self.compound_identity_loss_weight = float(compound_identity_loss_weight)
        self.compound_identity_temperature = float(compound_identity_temperature)
        self.atom_location_loss_weight = float(atom_location_loss_weight)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self._freeze_mol_encoder = False

        edge_dim = int(cleavage_edge_fnet.feature_dim)
        atom_dim = int(mol_encoder.node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_body = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.observed_edge_head = nn.Linear(hidden_dim, 1)
        self.pattern_head = nn.Linear(hidden_dim, self.num_patterns) if self.num_patterns > 0 else None
        self.reaction_head = nn.Linear(hidden_dim, self.num_reactions) if self.num_reactions > 0 else None
        self.product_head = nn.Linear(hidden_dim, self.num_product_molecules) if self.num_product_molecules > 0 else None
        self.edge_atom_query = nn.Linear(hidden_dim, hidden_dim)
        self.atom_key = nn.Linear(atom_dim, hidden_dim)


    def freeze_mol_encoder(self) -> None:
        self._freeze_mol_encoder = True
        self.mol_encoder.eval()
        for param in self.mol_encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_mol_encoder", False):
            self.mol_encoder.eval()
        return self

    def training_config_dict(self) -> Dict[str, float | int]:
        return {
            "num_patterns": self.num_patterns,
            "num_reactions": self.num_reactions,
            "num_product_molecules": self.num_product_molecules,
            "hidden_dim": self.hidden_dim,
            "observed_edge_loss_weight": self.observed_edge_loss_weight,
            "pattern_loss_weight": self.pattern_loss_weight,
            "reaction_loss_weight": self.reaction_loss_weight,
            "product_loss_weight": self.product_loss_weight,
            "compound_identity_loss_weight": self.compound_identity_loss_weight,
            "compound_identity_temperature": self.compound_identity_temperature,
            "atom_location_loss_weight": self.atom_location_loss_weight,
            "dropout": self.dropout,
        }

    def forward(self, structure: FragmentTreeStructure) -> CleavagePretrainingOutput:
        mol_graph = self.mol_encoder(structure.node_graph)
        ft_features = FragmentTreeFeatures.from_structure(structure, node_graphs=mol_graph)
        return self.forward_features(ft_features)

    def forward_features(self, ft_features: FragmentTreeFeatures) -> CleavagePretrainingOutput:
        structure = ft_features.structure
        targets = build_cleavage_edge_targets(structure)
        event_row_id, event_edge_id, event_attr = self.cleavage_edge_fnet.encode_events(ft_features)
        event_h = self.edge_body(self.edge_norm(event_attr))

        losses = []
        metrics: Dict[str, float] = {}

        if event_attr.size(0) > 0:
            observed_target = targets.observed_edge[event_edge_id]
            observed_logits = self.observed_edge_head(event_h).squeeze(-1)
            observed_loss = self._observed_edge_loss(observed_logits, observed_target)
            losses.append(self.observed_edge_loss_weight * observed_loss)
            metrics.update(self._binary_metrics("observed_event_edge", observed_logits, observed_target, observed_loss))

            event_labels = structure.cleavage_event[event_row_id.long()]
            for name, labels, head, weight in (
                ("pattern", event_labels[:, 0].long(), self.pattern_head, self.pattern_loss_weight),
                ("reaction", event_labels[:, 1].long(), self.reaction_head, self.reaction_loss_weight),
                ("product_molecule", event_labels[:, 2].long(), self.product_head, self.product_loss_weight),
            ):
                if head is None:
                    continue
                loss, task_metrics = self._classification_loss_and_metrics(name, head(event_h), labels)
                if loss is not None:
                    losses.append(float(weight) * loss)
                    metrics.update(task_metrics)

            identity_loss, identity_metrics = self._compound_identity_loss_and_metrics(
                structure=structure,
                event_h=event_h,
                event_row_id=event_row_id,
                event_edge_id=event_edge_id,
            )
            if identity_loss is not None:
                losses.append(self.compound_identity_loss_weight * identity_loss)
                metrics.update(identity_metrics)

            atom_loss, atom_metrics = self._atom_location_loss_and_metrics(
                structure=structure,
                atom_h=ft_features.node_graphs.x,
                event_h=event_h,
                event_row_id=event_row_id,
                event_edge_id=event_edge_id,
            )
            if atom_loss is not None:
                losses.append(self.atom_location_loss_weight * atom_loss)
                metrics.update(atom_metrics)

        if losses:
            loss = torch.stack([item if item.dim() == 0 else item.mean() for item in losses]).sum()
        else:
            loss = event_attr.sum() * 0.0
        metrics["loss"] = float(loss.detach().cpu())
        return CleavagePretrainingOutput(loss=loss, metrics=metrics, event_attr=event_attr)

    @staticmethod
    def _observed_edge_loss(logits: Tensor, target: Tensor) -> Tensor:
        positives = target.sum()
        negatives = target.numel() - positives
        pos_weight = None
        if positives.detach().item() > 0 and negatives.detach().item() > 0:
            pos_weight = (negatives / positives).clamp_min(1.0).detach()
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    @staticmethod
    def _binary_metrics(prefix: str, logits: Tensor, target: Tensor, loss: Tensor) -> Dict[str, float]:
        pred = torch.sigmoid(logits) >= 0.5
        target_bool = target >= 0.5
        metrics = {
            f"{prefix}_loss": float(loss.detach().cpu()),
            f"{prefix}_acc": float((pred == target_bool).float().mean().detach().cpu()),
            f"{prefix}_count": float(target.numel()),
        }
        pos = target_bool
        neg = ~target_bool
        if bool(pos.any()):
            metrics[f"{prefix}_pos_acc"] = float((pred[pos] == target_bool[pos]).float().mean().detach().cpu())
            metrics[f"{prefix}_pos_count"] = float(pos.sum().detach().cpu())
        if bool(neg.any()):
            metrics[f"{prefix}_neg_acc"] = float((pred[neg] == target_bool[neg]).float().mean().detach().cpu())
            metrics[f"{prefix}_neg_count"] = float(neg.sum().detach().cpu())
        return metrics

    @staticmethod
    def _classification_loss_and_metrics(prefix: str, logits: Tensor, labels: Tensor) -> Tuple[Optional[Tensor], Dict[str, float]]:
        valid = labels != IGNORE_INDEX
        if not bool(valid.any()):
            return None, {}
        loss = F.cross_entropy(logits[valid], labels[valid])
        pred = logits[valid].argmax(dim=-1)
        target = labels[valid]
        return loss, {
            f"{prefix}_loss": float(loss.detach().cpu()),
            f"{prefix}_acc": float((pred == target).float().mean().detach().cpu()),
            f"{prefix}_count": float(target.numel()),
        }


    def _compound_identity_loss_and_metrics(
        self,
        *,
        structure: FragmentTreeStructure,
        event_h: Tensor,
        event_row_id: Tensor,
        event_edge_id: Tensor,
    ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        labels = self._compound_identity_labels_from_structure(
            structure=structure,
            event_row_id=event_row_id,
            event_edge_id=event_edge_id,
        )
        valid = labels != IGNORE_INDEX
        if not bool(valid.any()):
            return None, {}

        z = F.normalize(event_h[valid], dim=-1)
        labels = labels[valid]
        n_events = int(labels.numel())
        if n_events < 2:
            return None, {}

        same = labels.view(-1, 1).eq(labels.view(1, -1))
        eye = torch.eye(n_events, dtype=torch.bool, device=labels.device)
        positive_mask = same & ~eye
        anchor_mask = positive_mask.any(dim=1)
        if not bool(anchor_mask.any()):
            return None, {
                "compound_identity_anchor_count": 0.0,
                "compound_identity_pair_count": 0.0,
            }

        temperature = max(self.compound_identity_temperature, 1e-6)
        logits = z @ z.t() / temperature
        logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        loss = -(positive_log_prob / positive_count)[anchor_mask].mean()

        with torch.no_grad():
            nearest = (z @ z.t()).masked_fill(eye, -2.0).argmax(dim=1)
            nearest_acc = (labels[nearest] == labels).float().mean()

        return loss, {
            "compound_identity_loss": float(loss.detach().cpu()),
            "compound_identity_nearest_acc": float(nearest_acc.detach().cpu()),
            "compound_identity_anchor_count": float(anchor_mask.sum().detach().cpu()),
            "compound_identity_pair_count": float(positive_mask.sum().detach().cpu()),
        }

    def _compound_identity_labels_from_structure(
        self,
        *,
        structure: FragmentTreeStructure,
        event_row_id: Tensor,
        event_edge_id: Tensor,
    ) -> Tensor:
        labels = torch.full(
            (event_row_id.size(0),),
            IGNORE_INDEX,
            dtype=torch.long,
            device=event_row_id.device,
        )
        identity_id_by_key: Dict[str, int] = {}

        for row_pos in range(int(event_row_id.numel())):
            event_idx = int(event_row_id[row_pos].item())
            edge_idx = int(event_edge_id[row_pos].item())
            dst_node = int(structure.edge_index[1, edge_idx].item())
            event = structure.cleavage_event[event_idx]
            pattern_id = int(event[0].item())
            reaction_id = int(event[1].item())
            product_molecule_id = int(event[2].item())
            product_atoms = self._event_atom_tuple(
                structure,
                tuple_length=self._lookup_product_tuple_length(
                    structure,
                    pattern_id,
                    reaction_id,
                    product_molecule_id,
                ),
                row_index=int(event[4].item()),
            )
            key = canonical_fragment_smiles(
                smiles=str(structure.node_smiles[dst_node]),
                atom_indices=tuple(int(i) for i in product_atoms.detach().cpu().tolist()),
            )
            if key is None:
                continue
            if key not in identity_id_by_key:
                identity_id_by_key[key] = len(identity_id_by_key)
            labels[row_pos] = identity_id_by_key[key]

        return labels

    def _atom_location_loss_and_metrics(
        self,
        *,
        structure: FragmentTreeStructure,
        atom_h: Tensor,
        event_h: Tensor,
        event_row_id: Tensor,
        event_edge_id: Tensor,
    ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        losses = []
        correct = 0.0
        count = 0.0
        for row_pos in range(int(event_row_id.numel())):
            event_idx = int(event_row_id[row_pos].item())
            edge_idx = int(event_edge_id[row_pos].item())
            src_node = int(structure.edge_index[0, edge_idx].item())
            dst_node = int(structure.edge_index[1, edge_idx].item())
            event = structure.cleavage_event[event_idx]
            pattern_id = int(event[0].item())
            reaction_id = int(event[1].item())
            product_molecule_id = int(event[2].item())

            reactant_atoms = self._event_atom_tuple(
                structure,
                tuple_length=self._lookup_reactant_tuple_length(structure, pattern_id, reaction_id),
                row_index=int(event[3].item()),
            )
            product_atoms = self._event_atom_tuple(
                structure,
                tuple_length=self._lookup_product_tuple_length(structure, pattern_id, reaction_id, product_molecule_id),
                row_index=int(event[4].item()),
            )
            for node_idx, local_atoms in ((src_node, reactant_atoms), (dst_node, product_atoms)):
                node_loss, node_correct, node_count = self._node_atom_location_loss(
                    structure=structure,
                    atom_h=atom_h,
                    edge_h=event_h[row_pos],
                    node_idx=node_idx,
                    positive_local_atoms=local_atoms,
                )
                if node_loss is not None:
                    losses.append(node_loss)
                    correct += node_correct
                    count += node_count
        if not losses:
            return None, {}
        loss = torch.stack(losses).mean()
        return loss, {
            "atom_location_loss": float(loss.detach().cpu()),
            "atom_location_acc": float(correct / max(count, 1.0)),
            "atom_location_count": float(count),
        }

    def _node_atom_location_loss(
        self,
        *,
        structure: FragmentTreeStructure,
        atom_h: Tensor,
        edge_h: Tensor,
        node_idx: int,
        positive_local_atoms: Tensor,
    ) -> Tuple[Optional[Tensor], float, float]:
        start = int(structure.node_graph_offset[int(node_idx)].item())
        stop = int(structure.node_graph_offset[int(node_idx) + 1].item())
        if stop <= start:
            return None, 0.0, 0.0
        node_atoms = atom_h[start:stop]
        query = self.edge_atom_query(edge_h).view(1, -1)
        keys = self.atom_key(node_atoms)
        logits = (keys * query).sum(dim=-1) / math.sqrt(max(keys.size(-1), 1))
        target = torch.zeros((stop - start,), dtype=logits.dtype, device=logits.device)
        valid_atoms = positive_local_atoms[(positive_local_atoms >= 0) & (positive_local_atoms < target.numel())].long()
        if valid_atoms.numel() == 0:
            return None, 0.0, 0.0
        target[valid_atoms] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits) >= 0.5
        correct = float((pred == target.bool()).float().sum().detach().cpu())
        return loss, correct, float(target.numel())

    @staticmethod
    def _event_atom_tuple(structure: FragmentTreeStructure, *, tuple_length: int, row_index: int) -> Tensor:
        table = structure.cleavage_atom_idxs[int(tuple_length)]
        return table[int(row_index)].long()

    @staticmethod
    def _lookup_reactant_tuple_length(structure: FragmentTreeStructure, pattern_id: int, reaction_id: int) -> int:
        table = structure.reactant_tuple_length_table.long()
        mask = (table[:, 0] == int(pattern_id)) & (table[:, 1] == int(reaction_id))
        if not bool(mask.any()):
            raise KeyError(f"Missing reactant tuple length for pattern={pattern_id}, reaction={reaction_id}.")
        return int(table[mask][0, 2].item())

    @staticmethod
    def _lookup_product_tuple_length(structure: FragmentTreeStructure, pattern_id: int, reaction_id: int, product_molecule_id: int) -> int:
        table = structure.product_tuple_length_table.long()
        mask = (table[:, 0] == int(pattern_id)) & (table[:, 1] == int(reaction_id)) & (table[:, 2] == int(product_molecule_id))
        if not bool(mask.any()):
            raise KeyError(
                "Missing product tuple length for "
                f"pattern={pattern_id}, reaction={reaction_id}, product_molecule={product_molecule_id}."
            )
        return int(table[mask][0, 3].item())


def canonical_fragment_smiles(smiles: str, atom_indices: Tuple[int, ...]) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    atoms = tuple(sorted({int(i) for i in atom_indices if int(i) >= 0}))
    if not atoms or atoms[-1] >= mol.GetNumAtoms():
        return None
    atom_set = set(atoms)
    bonds = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
    ]
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=list(atoms),
        bondsToUse=bonds,
        canonical=True,
        isomericSmiles=True,
    )
