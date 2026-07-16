from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor

from ...input.fragment_tree_features import FragmentTreeFeatures
from ...input.fragment_tree_structure import FragmentTreeStructure
from ...mol.atom_feature import AtomFeatureLayer
from ...mol.bond_feature import BondFeatureLayer
from ...mol.mol_encoder import MolEncoder
from ...specgen.components.cleavage.cleavage_edge_feature_net import CleavageEdgeFeatureNet
from .dataset import (
    IGNORE_INDEX,
    first_stage_edge_mask,
)


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
        pattern_loss_weight: float = 1.0,
        reaction_loss_weight: float = 1.0,
        product_loss_weight: float = 1.0,
        reactant_structure_loss_weight: float = 1.0,
        surrounding_structure_loss_weight: float = 1.0,
        surrounding_structure_radius: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.cleavage_edge_fnet = cleavage_edge_fnet
        self.num_patterns = int(num_patterns)
        self.num_reactions = int(num_reactions)
        self.num_product_molecules = int(num_product_molecules)
        self.pattern_loss_weight = float(pattern_loss_weight)
        self.reaction_loss_weight = float(reaction_loss_weight)
        self.product_loss_weight = float(product_loss_weight)
        self.reactant_structure_loss_weight = float(reactant_structure_loss_weight)
        self.surrounding_structure_loss_weight = float(surrounding_structure_loss_weight)
        self.surrounding_structure_radius = int(surrounding_structure_radius)
        if self.surrounding_structure_radius < 1:
            raise ValueError("surrounding_structure_radius must be at least 1.")
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self._freeze_mol_encoder = False

        edge_dim = int(cleavage_edge_fnet.feature_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_body = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pattern_head = nn.Linear(hidden_dim, self.num_patterns) if self.num_patterns > 0 else None
        self.reaction_head = nn.Linear(hidden_dim, self.num_reactions) if self.num_reactions > 0 else None
        self.product_head = nn.Linear(hidden_dim, self.num_product_molecules) if self.num_product_molecules > 0 else None
        self.reactant_pair_head = nn.Linear(hidden_dim * 2, 1)
        self.surrounding_position = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.surrounding_message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.surrounding_bond_body = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.atom_feature_layer = AtomFeatureLayer(symbols=mol_encoder.symbols)
        self.bond_feature_layer = BondFeatureLayer()
        self.surrounding_atom_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, len(values))
                for name, values in self.atom_feature_layer.feature_sets.items()
            }
        )
        self.surrounding_bond_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, len(values))
                for name, values in self.bond_feature_layer.feature_sets.items()
            }
        )


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
            "pattern_loss_weight": self.pattern_loss_weight,
            "reaction_loss_weight": self.reaction_loss_weight,
            "product_loss_weight": self.product_loss_weight,
            "reactant_structure_loss_weight": self.reactant_structure_loss_weight,
            "surrounding_structure_loss_weight": self.surrounding_structure_loss_weight,
            "surrounding_structure_radius": self.surrounding_structure_radius,
            "dropout": self.dropout,
        }

    def forward(
        self,
        structure: FragmentTreeStructure,
        *,
        selected_event_rows: Optional[Tensor] = None,
    ) -> CleavagePretrainingOutput:
        mol_graph = self.mol_encoder(structure.node_graph)
        ft_features = FragmentTreeFeatures.from_structure(structure, node_graphs=mol_graph)
        return self.forward_features(
            ft_features, selected_event_rows=selected_event_rows
        )

    def forward_features(
        self,
        ft_features: FragmentTreeFeatures,
        *,
        selected_event_rows: Optional[Tensor] = None,
    ) -> CleavagePretrainingOutput:
        structure = ft_features.structure
        event_row_id, event_edge_id, event_attr = self.cleavage_edge_fnet.encode_events(ft_features)
        stage_mask = first_stage_edge_mask(structure)[event_edge_id.long()]
        if selected_event_rows is not None:
            selected_event_rows = selected_event_rows.to(
                device=event_row_id.device, dtype=event_row_id.dtype
            )
            stage_mask &= torch.isin(event_row_id, selected_event_rows)
        event_row_id = event_row_id[stage_mask]
        event_edge_id = event_edge_id[stage_mask]
        event_attr = event_attr[stage_mask]
        event_h = self.edge_body(self.edge_norm(event_attr))

        losses = []
        metrics: Dict[str, float] = {}

        if event_attr.size(0) > 0:
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

            structure_loss, structure_metrics = self._reactant_structure_loss_and_metrics(
                structure=structure,
                event_h=event_h,
                event_row_id=event_row_id,
                event_edge_id=event_edge_id,
            )
            if structure_loss is not None:
                losses.append(self.reactant_structure_loss_weight * structure_loss)
                metrics.update(structure_metrics)

            surrounding_loss, surrounding_metrics = self._surrounding_structure_loss_and_metrics(
                structure=structure,
                event_h=event_h,
                event_row_id=event_row_id,
                event_edge_id=event_edge_id,
            )
            if surrounding_loss is not None:
                losses.append(self.surrounding_structure_loss_weight * surrounding_loss)
                metrics.update(surrounding_metrics)

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
        metrics = {
            f"{prefix}_loss": float(loss.detach().cpu()),
            f"{prefix}_acc": float((pred == target).float().mean().detach().cpu()),
            f"{prefix}_count": float(target.numel()),
        }
        for class_id in target.unique(sorted=True).tolist():
            class_mask = target == int(class_id)
            metrics[f"{prefix}_class_{int(class_id)}_acc"] = float(
                (pred[class_mask] == target[class_mask]).float().mean().detach().cpu()
            )
            metrics[f"{prefix}_class_{int(class_id)}_count"] = float(
                class_mask.sum().detach().cpu()
            )
        return loss, metrics


    def _reactant_structure_loss_and_metrics(
        self,
        *,
        structure: FragmentTreeStructure,
        event_h: Tensor,
        event_row_id: Tensor,
        event_edge_id: Tensor,
    ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        labels = self._reactant_structure_labels_from_structure(
            structure=structure, event_row_id=event_row_id, event_edge_id=event_edge_id
        )
        valid = labels != IGNORE_INDEX
        if int(valid.sum().item()) < 2:
            return None, {}

        z = event_h[valid]
        labels = labels[valid]
        pair_i, pair_j = torch.triu_indices(
            labels.numel(), labels.numel(), offset=1, device=labels.device
        )
        pair_features = torch.cat(
            (torch.abs(z[pair_i] - z[pair_j]), z[pair_i] * z[pair_j]), dim=-1
        )
        target_bool = labels[pair_i].eq(labels[pair_j])
        positive_pairs = target_bool.nonzero(as_tuple=False).flatten()
        negative_pairs = (~target_bool).nonzero(as_tuple=False).flatten()
        if positive_pairs.numel() == 0 or negative_pairs.numel() == 0:
            return None, {}

        # Equalize pair contributions. Merely placing both identities in a
        # batch does not balance the O(N^2) pair combinations and previously
        # allowed one side to dominate the gradient.
        pair_count = min(positive_pairs.numel(), negative_pairs.numel())
        if self.training:
            positive_pairs = positive_pairs[
                torch.randperm(positive_pairs.numel(), device=labels.device)[:pair_count]
            ]
            negative_pairs = negative_pairs[
                torch.randperm(negative_pairs.numel(), device=labels.device)[:pair_count]
            ]
        else:
            positive_pairs = positive_pairs[:pair_count]
            negative_pairs = negative_pairs[:pair_count]
        selected_pairs = torch.cat((positive_pairs, negative_pairs))
        logits = self.reactant_pair_head(pair_features[selected_pairs]).squeeze(-1)
        target = target_bool[selected_pairs].to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss, self._binary_metrics("reactant_structure", logits, target, loss)

    def _reactant_structure_labels_from_structure(
        self,
        *,
        structure: FragmentTreeStructure,
        event_row_id: Tensor,
        event_edge_id: Tensor,
    ) -> Tensor:
        labels = torch.full(
            (event_row_id.size(0),), IGNORE_INDEX, dtype=torch.long, device=event_row_id.device
        )
        identity_id_by_key: Dict[str, int] = {}
        for row_pos in range(int(event_row_id.numel())):
            event_idx = int(event_row_id[row_pos].item())
            edge_idx = int(event_edge_id[row_pos].item())
            src_node = int(structure.edge_index[0, edge_idx].item())
            event = structure.cleavage_event[event_idx]
            pattern_id = int(event[0].item())
            reaction_id = int(event[1].item())
            reactant_atoms = self._event_atom_tuple(
                structure,
                tuple_length=self._lookup_reactant_tuple_length(
                    structure, pattern_id, reaction_id
                ),
                row_index=int(event[3].item()),
            )
            key = canonical_fragment_smiles(
                smiles=str(structure.node_smiles[src_node]),
                atom_indices=tuple(int(i) for i in reactant_atoms.detach().cpu().tolist()),
            )
            if key is None:
                continue
            if key not in identity_id_by_key:
                identity_id_by_key[key] = len(identity_id_by_key)
            labels[row_pos] = identity_id_by_key[key]
        return labels

    def _surrounding_structure_loss_and_metrics(
        self,
        *,
        structure: FragmentTreeStructure,
        event_h: Tensor,
        event_row_id: Tensor,
        event_edge_id: Tensor,
    ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        losses = []
        graph_correct = graph_count = 0.0
        atom_correct = atom_count = 0.0
        bond_correct = bond_count = 0.0
        group_correct: Dict[str, float] = {}
        group_count: Dict[str, float] = {}
        for row_pos in range(int(event_row_id.numel())):
            event_idx = int(event_row_id[row_pos].item())
            edge_idx = int(event_edge_id[row_pos].item())
            src_node = int(structure.edge_index[0, edge_idx].item())
            event = structure.cleavage_event[event_idx]
            reactant_atoms = self._event_atom_tuple(
                structure,
                tuple_length=self._lookup_reactant_tuple_length(
                    structure, int(event[0].item()), int(event[1].item())
                ),
                row_index=int(event[3].item()),
            )
            mol = Chem.MolFromSmiles(str(structure.node_smiles[src_node]))
            if mol is None:
                continue
            decoded = self._decode_local_skeleton(
                mol=mol,
                reactant_atoms=tuple(int(value) for value in reactant_atoms.tolist()),
                event_h=event_h[row_pos],
            )
            if decoded is None:
                continue
            item_loss, item_metrics = decoded
            losses.append(item_loss)
            graph_correct += item_metrics["graph_correct"]
            graph_count += 1.0
            atom_correct += item_metrics["atom_correct"]
            atom_count += item_metrics["atom_count"]
            bond_correct += item_metrics["bond_correct"]
            bond_count += item_metrics["bond_count"]
            for key, value in item_metrics.items():
                if key.endswith("_group_correct"):
                    group_correct[key[:-len("_group_correct")]] = (
                        group_correct.get(key[:-len("_group_correct")], 0.0) + value
                    )
                elif key.endswith("_group_count"):
                    group_count[key[:-len("_group_count")]] = (
                        group_count.get(key[:-len("_group_count")], 0.0) + value
                    )
        if not losses:
            return None, {}
        loss = torch.stack(losses).mean()
        metrics = {
            "surrounding_structure_loss": float(loss.detach().cpu()),
            # Exact reconstruction: every atom and bond label group in the
            # supplied unlabeled local skeleton must be correct.
            "surrounding_structure_acc": graph_correct / max(graph_count, 1.0),
            "surrounding_structure_count": graph_count,
            "surrounding_structure_atom_acc": atom_correct / max(atom_count, 1.0),
            "surrounding_structure_atom_count": atom_count,
            "surrounding_structure_bond_acc": bond_correct / max(bond_count, 1.0),
            "surrounding_structure_bond_count": bond_count,
        }
        for key in sorted(group_count):
            count = group_count[key]
            metrics[f"surrounding_structure_{key}_acc"] = (
                group_correct.get(key, 0.0) / max(count, 1.0)
            )
            metrics[f"surrounding_structure_{key}_count"] = count
        return loss, metrics

    def _decode_local_skeleton(
        self,
        *,
        mol: Chem.Mol,
        reactant_atoms: Tuple[int, ...],
        event_h: Tensor,
    ) -> Optional[Tuple[Tensor, Dict[str, float]]]:
        valid_reactant = tuple(
            atom for atom in reactant_atoms if 0 <= atom < mol.GetNumAtoms()
        )
        if not valid_reactant:
            return None
        distance_matrix = Chem.GetDistanceMatrix(mol)
        local_atoms = sorted(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if min(float(distance_matrix[center, atom.GetIdx()]) for center in valid_reactant)
            <= self.surrounding_structure_radius
        )
        if not local_atoms:
            return None
        local_set = set(local_atoms)
        atom_to_local = {atom: index for index, atom in enumerate(local_atoms)}
        local_bonds = [
            bond for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() in local_set and bond.GetEndAtomIdx() in local_set
        ]

        position_rows = []
        reactant_slot = {atom: slot for slot, atom in enumerate(valid_reactant)}
        slot_scale = max(len(valid_reactant) - 1, 1)
        radius_scale = max(self.surrounding_structure_radius, 1)
        for atom_index in local_atoms:
            nearest_slot, nearest_distance = min(
                (
                    slot,
                    float(distance_matrix[center, atom_index]),
                )
                for slot, center in enumerate(valid_reactant)
            )
            degree = sum(
                1
                for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
                if neighbor.GetIdx() in local_set
            )
            position_rows.append(
                [
                    1.0 if atom_index in reactant_slot else 0.0,
                    float(reactant_slot.get(atom_index, nearest_slot)) / slot_scale,
                    nearest_distance / radius_scale,
                    float(degree) / 4.0,
                ]
            )
        positions = torch.tensor(
            position_rows, dtype=event_h.dtype, device=event_h.device
        )
        node_h = event_h.view(1, -1) + self.surrounding_position(positions)

        if local_bonds:
            undirected_edges = [
                (
                    atom_to_local[bond.GetBeginAtomIdx()],
                    atom_to_local[bond.GetEndAtomIdx()],
                )
                for bond in local_bonds
            ]
            for _ in range(2):
                source = torch.tensor(
                    [edge[0] for edge in undirected_edges] + [edge[1] for edge in undirected_edges],
                    dtype=torch.long,
                    device=node_h.device,
                )
                target = torch.tensor(
                    [edge[1] for edge in undirected_edges] + [edge[0] for edge in undirected_edges],
                    dtype=torch.long,
                    device=node_h.device,
                )
                aggregated = torch.zeros_like(node_h).index_add(0, target, node_h[source])
                degree = torch.zeros(
                    (node_h.size(0), 1), dtype=node_h.dtype, device=node_h.device
                ).index_add(
                    0,
                    target,
                    torch.ones((target.numel(), 1), dtype=node_h.dtype, device=node_h.device),
                )
                message = self.surrounding_message(
                    torch.cat((node_h, aggregated / degree.clamp_min(1.0)), dim=-1)
                )
                node_h = node_h + message

        losses = []
        all_correct = True
        atom_correct = atom_count = 0.0
        group_metrics: Dict[str, float] = {}
        atom_targets = torch.stack(
            [self.atom_feature_layer.encode(mol.GetAtomWithIdx(index)) for index in local_atoms]
        ).to(device=event_h.device)
        offset = 0
        for name, values in self.atom_feature_layer.feature_sets.items():
            width = len(values)
            target = atom_targets[:, offset : offset + width].argmax(dim=-1)
            logits = self.surrounding_atom_heads[name](node_h)
            losses.append(F.cross_entropy(logits, target))
            correct = logits.argmax(dim=-1).eq(target)
            atom_correct += float(correct.sum().detach().cpu())
            atom_count += float(target.numel())
            group_metrics[f"atom_{name}_group_correct"] = float(
                correct.sum().detach().cpu()
            )
            group_metrics[f"atom_{name}_group_count"] = float(target.numel())
            all_correct = all_correct and bool(correct.all())
            offset += width

        bond_correct = bond_count = 0.0
        if local_bonds:
            left = torch.tensor(
                [atom_to_local[b.GetBeginAtomIdx()] for b in local_bonds],
                dtype=torch.long,
                device=event_h.device,
            )
            right = torch.tensor(
                [atom_to_local[b.GetEndAtomIdx()] for b in local_bonds],
                dtype=torch.long,
                device=event_h.device,
            )
            bond_h = self.surrounding_bond_body(
                torch.cat((torch.abs(node_h[left] - node_h[right]), node_h[left] * node_h[right]), dim=-1)
            )
            bond_targets = torch.stack(
                [self.bond_feature_layer.encode(bond) for bond in local_bonds]
            ).to(device=event_h.device)
            offset = 0
            for name, values in self.bond_feature_layer.feature_sets.items():
                width = len(values)
                target = bond_targets[:, offset : offset + width].argmax(dim=-1)
                logits = self.surrounding_bond_heads[name](bond_h)
                losses.append(F.cross_entropy(logits, target))
                correct = logits.argmax(dim=-1).eq(target)
                bond_correct += float(correct.sum().detach().cpu())
                bond_count += float(target.numel())
                group_metrics[f"bond_{name}_group_correct"] = float(
                    correct.sum().detach().cpu()
                )
                group_metrics[f"bond_{name}_group_count"] = float(target.numel())
                all_correct = all_correct and bool(correct.all())
                offset += width

        return torch.stack(losses).mean(), {
            "graph_correct": float(all_correct),
            "atom_correct": atom_correct,
            "atom_count": atom_count,
            "bond_correct": bond_correct,
            "bond_count": bond_count,
            **group_metrics,
        }

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

def atom_neighborhood_indices(
    smiles: str, *, center_atom: int, radius: int
) -> Optional[Tuple[int, ...]]:
    """Return atoms one to ``radius`` bonds from a local center atom."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or center_atom < 0 or center_atom >= mol.GetNumAtoms() or radius < 1:
        return None
    distances = Chem.GetDistanceMatrix(mol)[int(center_atom)]
    return tuple(
        int(idx) for idx, distance in enumerate(distances)
        if 0 < float(distance) <= int(radius)
    )

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
