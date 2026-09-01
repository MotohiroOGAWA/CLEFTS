"""RDKit backend for the CLEFTS Workbench cleavage-pattern builder."""
from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, ".")

from rdkit import Chem
from rdkit.Chem import rdDepictor

from clefts.domain.fragment.cleavage._CleavagePattern import _CleavagePattern, ProductRule


def molecule(payload: dict[str, Any]) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(payload.get("smiles", "")))
    if mol is None:
        raise ValueError("Invalid SMILES.")
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        point = conf.GetAtomPosition(atom.GetIdx())
        atoms.append({"index": atom.GetIdx(), "symbol": atom.GetSymbol(), "x": point.x, "y": -point.y})
    bonds = []
    for bond in mol.GetBonds():
        bonds.append({
            "index": bond.GetIdx(), "begin": bond.GetBeginAtomIdx(), "end": bond.GetEndAtomIdx(),
            "order": 1.5 if bond.GetIsAromatic() else float(bond.GetBondTypeAsDouble()),
        })
    return {"canonicalSmiles": Chem.MolToSmiles(mol), "atoms": atoms, "bonds": bonds}


def _source(payload: dict[str, Any]) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(payload.get("smiles", "")))
    if mol is None:
        raise ValueError("Invalid SMILES.")
    return mol


def reactant(payload: dict[str, Any]) -> dict[str, Any]:
    mol = _source(payload)
    atom_ids = sorted({int(value) for value in payload.get("atoms", [])})
    bond_ids = sorted({int(value) for value in payload.get("bonds", [])})
    if not atom_ids:
        raise ValueError("Select at least one atom.")
    selected = set(atom_ids)
    if any(mol.GetBondWithIdx(i).GetBeginAtomIdx() not in selected or mol.GetBondWithIdx(i).GetEndAtomIdx() not in selected for i in bond_ids):
        raise ValueError("Every selected bond must connect selected atoms.")
    for map_number, atom_id in enumerate(atom_ids, 1):
        mol.GetAtomWithIdx(atom_id).SetAtomMapNum(map_number)
    smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=atom_ids, bondsToUse=bond_ids, isomericSmarts=False)
    constraints = payload.get("constraints", {})
    for map_number, atom_id in enumerate(atom_ids, 1):
        mode = constraints.get(str(atom_id), "exact")
        atom = mol.GetAtomWithIdx(atom_id)
        exact = f"#{atom.GetAtomicNum()}"
        replacement = exact
        if mode == "any-heavy": replacement = "!#1"
        elif mode == "C,N": replacement = "#6,#7"
        elif mode == "C,N,O": replacement = "#6,#7,#8"
        import re
        smarts, count = re.subn(rf"\[[^\]]*:{map_number}\]", f"[{replacement}:{map_number}]", smarts, count=1)
        if count != 1:
            raise ValueError(f"Could not apply the constraint for atom map {map_number}.")
    _CleavagePattern.from_rules(name=str(payload.get("name", "")), reactant_smarts=smarts, products=())
    return {"smarts": smarts, "atomMapBySource": {str(atom_id): i + 1 for i, atom_id in enumerate(atom_ids)}}


def product(payload: dict[str, Any]) -> dict[str, Any]:
    mol = _source(payload)
    atom_ids = sorted({int(value) for value in payload.get("atoms", [])})
    kept = {int(value) for value in payload.get("keptAtoms", atom_ids)}
    map_by_source = {int(k): int(v) for k, v in payload.get("atomMapBySource", {}).items()}
    if not kept:
        raise ValueError("A product must retain at least one atom.")
    editable = Chem.RWMol(mol)
    for source_id, map_number in map_by_source.items():
        editable.GetAtomWithIdx(source_id).SetAtomMapNum(map_number)
    overrides = payload.get("bondOverrides", {})
    bond_types = {"1": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE, "3": Chem.BondType.TRIPLE, "1.5": Chem.BondType.AROMATIC}
    for key, value in overrides.items():
        bond = editable.GetBondWithIdx(int(key))
        if value == "remove": editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        elif str(value) in bond_types: bond.SetBondType(bond_types[str(value)])
    for atom_id in sorted(set(range(mol.GetNumAtoms())) - kept, reverse=True):
        editable.RemoveAtom(atom_id)
    result = editable.GetMol()
    smarts = Chem.MolToSmarts(result, isomericSmiles=False)
    return {"smarts": smarts}


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    products = tuple(ProductRule(name=str(item.get("name", "")), smarts=str(item.get("smarts", ""))) for item in payload.get("products", []))
    pattern = _CleavagePattern.from_rules(name=str(payload.get("name", "")), reactant_smarts=str(payload.get("reactant_smarts", "")), products=products)
    return {"valid": True, "pattern": pattern.to_dict()}


COMMANDS = {"molecule": molecule, "reactant": reactant, "product": product, "validate": validate}


def main() -> int:
    try:
        request = json.load(sys.stdin)
        command = str(request.get("command", ""))
        if command not in COMMANDS:
            raise ValueError(f"Unknown command: {command}")
        print(json.dumps({"ok": True, "result": COMMANDS[command](request.get("payload", {}))}))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
