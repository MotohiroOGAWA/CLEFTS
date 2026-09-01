from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def mol_from_smiles(smiles: object) -> Chem.Mol | None:
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None
    return mol


def morgan_fingerprint(
    mol: Chem.Mol | None,
    *,
    radius: int,
    n_bits: int,
) -> ExplicitBitVect | None:
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def fingerprint_to_hex(fp: ExplicitBitVect) -> str:
    return DataStructs.BitVectToBinaryText(fp).hex()
