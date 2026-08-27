from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ....libs.mmkit.mmkit import Compound
from ....domain.molecule.descriptors import (
    DEFAULT_DESCRIPTOR_NAMES,
    compute_descriptor_values,
)
from ...mol.graph_builder import MolGraphBuilder
from .descriptor_coverage import DEFAULT_DESCRIPTOR_BIN_SPECS, build_descriptor_record_index


def load_smiles_file(path: str | Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_descriptors(mol: Chem.Mol, names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES) -> torch.Tensor:
    return torch.tensor(compute_descriptor_values(mol, names), dtype=torch.float32)


def compute_ecfp(mol: Chem.Mol, *, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    return torch.tensor([float(bit) for bit in fingerprint.ToBitString()], dtype=torch.float32)


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
        ecfp_radius: int = 2,
        ecfp_n_bits: int = 2048,
        progress_desc: str = "Building molecule dataset",
    ) -> None:
        self.graph_builder = MolGraphBuilder(symbols=tuple(symbols))
        self.descriptor_names = tuple(descriptor_names)
        self.descriptor_normalizer = descriptor_normalizer
        self.ecfp_radius = int(ecfp_radius)
        self.ecfp_n_bits = int(ecfp_n_bits)

        self.items: List[Data] = []
        for smiles in tqdm(smiles_values, desc=progress_desc, unit="mol"):
            try:
                compound = Compound.from_smiles(smiles)
                data = self.graph_builder.build(compound)
                data.smiles = compound.smiles
                data.descriptors = compute_descriptors(compound.mol, self.descriptor_names)
                data.ecfp = compute_ecfp(compound.mol, radius=self.ecfp_radius, n_bits=self.ecfp_n_bits)
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
        ecfp_radius: int = 2,
        ecfp_n_bits: int = 2048,
    ) -> "MolPretrainingDataset":
        obj = cls.__new__(cls)
        obj.graph_builder = MolGraphBuilder(symbols=tuple(symbols))
        obj.descriptor_names = tuple(descriptor_names)
        obj.descriptor_normalizer = descriptor_normalizer
        obj.ecfp_radius = int(ecfp_radius)
        obj.ecfp_n_bits = int(ecfp_n_bits)
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

    def descriptor_record_index(
        self,
        *,
        min_record_count: int = 10,
        descriptor_bin_specs=DEFAULT_DESCRIPTOR_BIN_SPECS,
    ) -> Dict[str, Dict[str, List[int]]]:
        descriptor_rows = [data.descriptors.detach().cpu().tolist() for data in self.items]
        index, _ = build_descriptor_record_index(
            descriptor_rows,
            self.descriptor_names,
            min_record_count=min_record_count,
            specs_by_name={spec.name: spec for spec in descriptor_bin_specs},
        )
        return index


def collate_mol_graphs(items: Sequence[Data]) -> Batch:
    return Batch.from_data_list(list(items))
