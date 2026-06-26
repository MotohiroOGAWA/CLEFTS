from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ....libs.mmkit.mmkit import Compound
from ....libs.msentity.msentity import MSDataset
from ...mol.graph_builder import MolGraphBuilder


DEFAULT_SMILES_COLUMN = "SMILES"
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


def load_smiles_from_msds(path: str | Path, smiles_column: str = DEFAULT_SMILES_COLUMN) -> List[str]:
    dataset = MSDataset.load(str(path), load_peak_metadata=False)
    metadata = dataset.metadata
    if smiles_column not in metadata.columns:
        raise KeyError(
            f"SMILES column '{smiles_column}' was not found in {path}. "
            f"Available columns: {list(metadata.columns)}"
        )
    smiles = metadata[smiles_column].dropna().astype(str).unique().tolist()
    return smiles

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
        std = values.std(dim=0).clamp_min(1e-6)
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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Data:
        data = self.items[index].clone()
        if self.descriptor_normalizer is not None:
            data.descriptors = self.descriptor_normalizer.transform(data.descriptors)
        return data

    def descriptor_matrix(self) -> torch.Tensor:
        return torch.stack([data.descriptors for data in self.items], dim=0)


def collate_mol_graphs(items: Sequence[Data]) -> Batch:
    return Batch.from_data_list(list(items))
