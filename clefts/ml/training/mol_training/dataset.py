from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ....libs.mmkit.mmkit import Compound
from ...mol.graph_builder import MolGraphBuilder


DEFAULT_DESCRIPTOR_NAMES = (
    "MolWt",
    # "ExactMolWt",  # Mostly redundant with MolWt for this pretraining target.
    "TPSA",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    # "HeavyAtomCount",  # Largely a molecule-size target and easy to infer from graph size.
)


def load_smiles_file(path: str | Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_descriptors(mol: Chem.Mol, names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES) -> torch.Tensor:
    values = []
    for name in names:
        if name == "MolWt":
            value = Descriptors.MolWt(mol)
        elif name == "ExactMolWt":
            value = Descriptors.ExactMolWt(mol)
        elif name == "TPSA":
            value = rdMolDescriptors.CalcTPSA(mol)
        elif name == "MolLogP":
            value = Descriptors.MolLogP(mol)
        elif name == "NumHAcceptors":
            value = rdMolDescriptors.CalcNumHBA(mol)
        elif name == "NumHDonors":
            value = rdMolDescriptors.CalcNumHBD(mol)
        elif name == "NumRotatableBonds":
            value = rdMolDescriptors.CalcNumRotatableBonds(mol)
        elif name == "RingCount":
            value = rdMolDescriptors.CalcNumRings(mol)
        elif name == "FractionCSP3":
            value = rdMolDescriptors.CalcFractionCSP3(mol)
        elif name == "HeavyAtomCount":
            value = mol.GetNumHeavyAtoms()
        else:
            raise ValueError(f"Unsupported descriptor: {name}")
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class DescriptorNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "DescriptorNormalizer":
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.device)) / self.std.to(values.device)


class MolPretrainingDataset(Dataset):
    def __init__(
        self,
        smiles_values: Sequence[str],
        *,
        symbols: Sequence[str],
        descriptor_names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES,
        descriptor_normalizer: Optional[DescriptorNormalizer] = None,
        progress_desc: str = "Building molecule dataset",
    ) -> None:
        self.graph_builder = MolGraphBuilder(symbols=tuple(symbols))
        self.descriptor_names = tuple(descriptor_names)
        self.descriptor_normalizer = descriptor_normalizer

        self.items: List[Data] = []
        for smiles in tqdm(smiles_values, desc=progress_desc, unit="mol"):
            try:
                compound = Compound.from_smiles(smiles)
                data = self.graph_builder.build(compound)
                data.smiles = compound.smiles
                data.descriptors = compute_descriptors(compound.mol, self.descriptor_names)
                self.items.append(data)
            except Exception:
                continue

        if not self.items:
            raise ValueError("No valid molecules were loaded.")

    @classmethod
    def from_items(
        cls,
        items: Sequence[Data],
        *,
        symbols: Sequence[str],
        descriptor_names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES,
        descriptor_normalizer: Optional[DescriptorNormalizer] = None,
    ) -> "MolPretrainingDataset":
        obj = cls.__new__(cls)
        obj.graph_builder = MolGraphBuilder(symbols=tuple(symbols))
        obj.descriptor_names = tuple(descriptor_names)
        obj.descriptor_normalizer = descriptor_normalizer
        obj.items = list(items)
        if not obj.items:
            raise ValueError("No valid molecules were loaded.")
        return obj

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Data:
        data = self.items[index].clone()
        if self.descriptor_normalizer is not None:
            data.descriptors = self.descriptor_normalizer.transform(data.descriptors)
        return data

    def descriptor_matrix(self) -> torch.Tensor:
        return torch.stack([data.descriptors for data in self.items], dim=0)

    def feature_target_counts(self, attr_name: str, groups) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = {}
        for group in groups:
            counts[group.name] = {label: 0 for label in group.labels}

        for data in self.items:
            features = getattr(data, attr_name)
            if features.numel() == 0:
                continue
            if features.dim() == 1:
                features = features.view(1, -1)
            for group in groups:
                target_slice = features[:, group.start : group.stop]
                valid = target_slice.sum(dim=-1) > 0
                if not bool(valid.any()):
                    continue
                target = target_slice[valid].argmax(dim=-1)
                for class_idx, label in enumerate(group.labels):
                    counts[group.name][label] += int((target == class_idx).sum().item())
        return counts

    def feature_record_index(self, attr_name: str, groups) -> Dict[str, Dict[str, List[int]]]:
        index: Dict[str, Dict[str, List[int]]] = {}
        for group in groups:
            index[group.name] = {label: [] for label in group.labels}

        for item_index, data in enumerate(self.items):
            features = getattr(data, attr_name)
            if features.numel() == 0:
                continue
            if features.dim() == 1:
                features = features.view(1, -1)
            for group in groups:
                target_slice = features[:, group.start : group.stop]
                valid = target_slice.sum(dim=-1) > 0
                if not bool(valid.any()):
                    continue
                target = target_slice[valid].argmax(dim=-1)
                present = set(target.detach().cpu().tolist())
                for class_idx, label in enumerate(group.labels):
                    if class_idx in present:
                        index[group.name][label].append(item_index)
        return index


def collate_mol_graphs(items: Sequence[Data]) -> Batch:
    return Batch.from_data_list(list(items))
