"""RDKit backend for the CLEFTS Workbench cleavage-pattern builder."""
from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, ".")

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from clefts.domain.fragment.cleavage._CleavagePattern import _CleavagePattern, ProductRule
from clefts.libs.mmkit.mmkit import Adduct, Formula


def molecule(payload: dict[str, Any]) -> dict[str, Any]:
    is_smarts = bool(payload.get("smarts"))
    mol = Chem.MolFromSmarts(str(payload.get("smarts", ""))) if is_smarts else Chem.MolFromSmiles(str(payload.get("smiles", "")))
    if mol is None:
        raise ValueError("Invalid SMARTS." if is_smarts else "Invalid SMILES.")
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        point = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol() if atom.GetAtomicNum() else "*"
        atoms.append({
            "index": atom.GetIdx(), "symbol": symbol, "smarts": atom.GetSmarts(),
            "x": point.x, "y": -point.y, "mapNumber": atom.GetAtomMapNum(),
        })
    bonds = []
    for bond in mol.GetBonds():
        bonds.append({
            "index": bond.GetIdx(), "begin": bond.GetBeginAtomIdx(), "end": bond.GetEndAtomIdx(),
            "order": 1.5 if bond.GetIsAromatic() else float(bond.GetBondTypeAsDouble()),
            "aromatic": bond.GetIsAromatic(), "inRing": bond.IsInRing(),
        })
    return {"canonicalSmiles": "" if is_smarts else Chem.MolToSmiles(mol), "sourceType": "smarts" if is_smarts else "smiles", "atoms": atoms, "bonds": bonds}


def depict(payload: dict[str, Any]) -> dict[str, Any]:
    """Return RDKit's own SVG depiction instead of approximating bonds in JS."""
    mol = Chem.MolFromSmiles(str(payload.get("smiles", "")))
    if mol is None:
        raise ValueError("Invalid SMILES.")
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(640, 400)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    neutral_formula = Formula.from_mol(mol).plain
    used_adduct = str(payload.get("usedAdduct", ""))
    state_modified = bool(payload.get("stateModified"))
    precursor_mol = Chem.MolFromSmiles(str(payload.get("precursorSmiles", "")))
    precursor_mass = Formula.from_mol(precursor_mol).plain.exact_mass if precursor_mol is not None else None
    adduct_rows = []
    for text in dict.fromkeys(str(value) for value in payload.get("adducts", []) if value):
        try:
            adduct = Adduct.parse(text)
            ion_formula = adduct.apply_to_formula(neutral_formula)
            if not ion_formula.is_nonnegative:
                raise ValueError("resulting formula contains a negative element count")
            adduct_rows.append({
                "adduct": str(adduct),
                "formula": str(ion_formula),
                "mz": adduct.apply_to_mz(neutral_formula.exact_mass),
                "neutralLoss": None if precursor_mass is None else adduct.apply_to_mz(precursor_mass) - adduct.apply_to_mz(neutral_formula.exact_mass),
                "used": str(adduct) == used_adduct,
                "stateModified": str(adduct) == used_adduct and state_modified,
            })
        except Exception as exc:
            adduct_rows.append({"adduct": text, "error": str(exc)})
    return {
        "canonicalSmiles": Chem.MolToSmiles(mol),
        "svg": svg,
        "formula": str(neutral_formula),
        "exactMass": neutral_formula.exact_mass,
        "adducts": adduct_rows,
    }


def _source(payload: dict[str, Any]) -> Chem.Mol:
    source_type = str(payload.get("sourceType", "smiles"))
    mol = Chem.MolFromSmarts(str(payload.get("smarts", ""))) if source_type == "smarts" else Chem.MolFromSmiles(str(payload.get("smiles", "")))
    if mol is None:
        raise ValueError("Invalid SMARTS." if source_type == "smarts" else "Invalid SMILES.")
    return mol


def _atom_query_body(value: str) -> str:
    """Validate an atom SMARTS expression and return it without brackets/map."""
    query = value.strip()
    if not query:
        raise ValueError("Atom SMARTS must not be empty.")
    if not query.startswith("["):
        query = f"[{query}]"
    atom = Chem.AtomFromSmarts(query)
    if atom is None:
        raise ValueError(f"Invalid atom SMARTS: {value!r}")
    atom.SetAtomMapNum(0)
    canonical = atom.GetSmarts()
    return canonical[1:-1] if canonical.startswith("[") and canonical.endswith("]") else canonical


def reactant(payload: dict[str, Any]) -> dict[str, Any]:
    mol = _source(payload)
    atom_ids = sorted({int(value) for value in payload.get("atoms", [])})
    bond_ids = sorted({int(value) for value in payload.get("bonds", [])})
    if not atom_ids:
        raise ValueError("Select at least one atom.")
    selected = set(atom_ids)
    if any(mol.GetBondWithIdx(i).GetBeginAtomIdx() not in selected or mol.GetBondWithIdx(i).GetEndAtomIdx() not in selected for i in bond_ids):
        raise ValueError("Every selected bond must connect selected atoms.")
    requested_maps = {
        int(key): int(value)
        for key, value in payload.get("atomMapBySource", {}).items()
        if int(key) in selected
    }
    map_by_atom = {
        atom_id: requested_maps.get(atom_id, index + 1)
        for index, atom_id in enumerate(atom_ids)
    }
    map_numbers = list(map_by_atom.values())
    if any(number <= 0 for number in map_numbers) or len(set(map_numbers)) != len(map_numbers):
        raise ValueError("Reactant atom mapping numbers must be unique positive integers.")
    for atom_id in atom_ids:
        map_number = map_by_atom[atom_id]
        mol.GetAtomWithIdx(atom_id).SetAtomMapNum(map_number)
    bond_constraints = payload.get("bondConstraints", {})
    editable = Chem.RWMol(mol)
    for bond_id in bond_ids:
        constraint = bond_constraints.get(str(bond_id), {})
        requested_types = constraint.get("types")
        bond_type = str(constraint.get("type", "preserve"))
        if requested_types is None and bond_type == "preserve" and not constraint.get("ring"):
            continue
        bond = mol.GetBondWithIdx(bond_id)
        preserved = ":" if bond.GetIsAromatic() else {1.0: "-", 2.0: "=", 3.0: "#"}.get(bond.GetBondTypeAsDouble(), "~")
        symbols = {"any": "~", "single": "-", "double": "=", "triple": "#", "aromatic": ":"}
        if requested_types is not None:
            tokens = [symbols[value] for value in requested_types if value in symbols]
            if not tokens: raise ValueError(f"Choose at least one bond type for bond {bond_id}.")
            token = ",".join(tokens)
        else:
            token = symbols.get(bond_type, preserved)
        query = "@" if constraint.get("ring") and token == "~" else (f"{token};@" if constraint.get("ring") else token)
        editable.ReplaceBond(bond_id, Chem.BondFromSmarts(query))
    mol = editable.GetMol()
    constraints = payload.get("constraints", {})
    atom_editable = Chem.RWMol(mol)
    for atom_id in atom_ids:
        map_number = map_by_atom[atom_id]
        constraint = constraints.get(str(atom_id), "exact")
        atom = mol.GetAtomWithIdx(atom_id)
        exact = f"#{atom.GetAtomicNum()}"
        replacement = exact
        if isinstance(constraint, dict):
            mode = constraint.get("mode", "elements")
            elements = constraint.get("elements", [atom.GetSymbol()])
            if mode == "custom": replacement = _atom_query_body(str(constraint.get("smarts", "")))
            elif mode == "any": replacement = "*"
            elif mode == "any-heavy": replacement = "!#1"
            else:
                numbers = []
                for symbol in elements:
                    number = Chem.GetPeriodicTable().GetAtomicNumber(str(symbol))
                    if number and number not in numbers: numbers.append(number)
                if not numbers: raise ValueError(f"Choose at least one element for atom {atom_id}.")
                replacement = ",".join(f"#{number}" for number in numbers)
        else:
            mode = constraint
            if mode == "any": replacement = "*"
            elif mode == "any-heavy": replacement = "!#1"
            elif mode == "C,N": replacement = "#6,#7"
            elif mode == "C,N,O": replacement = "#6,#7,#8"
        query_atom = Chem.AtomFromSmarts(f"[{replacement}]")
        if query_atom is None:
            raise ValueError(f"Invalid atom SMARTS for atom map {map_number}: {replacement!r}")
        query_atom.SetAtomMapNum(map_number)
        atom_editable.ReplaceAtom(atom_id, query_atom)
    mol = atom_editable.GetMol()
    smarts = Chem.MolFragmentToSmarts(
        mol, atomsToUse=atom_ids, bondsToUse=bond_ids, isomericSmarts=False
    )
    _CleavagePattern.from_rules(name=str(payload.get("name", "")), reactant_smarts=smarts, products=())
    return {"smarts": smarts, "atomMapBySource": {str(atom_id): map_by_atom[atom_id] for atom_id in atom_ids}}


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
    bond_smarts = {"1": "-", "2": "=", "3": "#", "1.5": ":"}
    for key, value in overrides.items():
        bond = editable.GetBondWithIdx(int(key))
        if value == "remove": editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        elif str(value) in bond_smarts:
            editable.ReplaceBond(int(key), Chem.BondFromSmarts(bond_smarts[str(value)]))
    for atom_id in sorted(set(range(mol.GetNumAtoms())) - kept, reverse=True):
        editable.RemoveAtom(atom_id)
    result = editable.GetMol()
    smarts = Chem.MolToSmarts(result, isomericSmiles=False)
    return {"smarts": smarts}


def product_from_structure(payload: dict[str, Any]) -> dict[str, Any]:
    """Build mapped product SMARTS from an independently drawn product skeleton."""
    mol = _source(payload)
    reactant = Chem.MolFromSmarts(str(payload.get("reactantSmarts", "")))
    if reactant is None:
        raise ValueError("Invalid reactant SMARTS.")
    reactant_atoms = {
        atom.GetAtomMapNum(): atom.GetSmarts()
        for atom in reactant.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    map_by_atom = {int(key): int(value) for key, value in payload.get("atomMaps", {}).items()}
    all_atoms = set(range(mol.GetNumAtoms()))
    deleted_atoms = {int(value) for value in payload.get("deletedAtoms", [])}
    if not deleted_atoms <= all_atoms:
        raise ValueError(f"Unknown product atoms selected for deletion: {sorted(deleted_atoms - all_atoms)}")
    expected = all_atoms - deleted_atoms
    if not expected:
        raise ValueError("A product must retain at least one atom.")
    if not expected <= set(map_by_atom):
        missing = sorted(expected - set(map_by_atom))
        raise ValueError(f"Assign a mapping number to every product atom; missing atoms: {missing}")
    map_numbers = [map_by_atom[atom_id] for atom_id in expected]
    if any(number <= 0 for number in map_numbers):
        raise ValueError("Product atom mapping numbers must be positive integers.")
    if len(set(map_numbers)) != len(map_numbers):
        raise ValueError("Product atom mapping numbers must be unique.")

    editable = Chem.RWMol(mol)
    atom_overrides = {int(key): str(value).strip() for key, value in payload.get("atomOverrides", {}).items()}
    for atom_id in sorted(expected):
        map_number = map_by_atom[atom_id]
        query = atom_overrides.get(atom_id, reactant_atoms.get(map_number, ""))
        if not query:
            raise ValueError(
                f"Product mapping number {map_number} does not exist in the reactant. "
                "Enable atom type changes and specify Atom SMARTS to create it explicitly."
            )
        if query.startswith(("#", "!")):
            query = f"[{query}]"
        replacement = Chem.AtomFromSmarts(query)
        if replacement is None:
            raise ValueError(f"Invalid atom SMARTS for product atom {atom_id}: {query!r}")
        replacement.SetAtomMapNum(map_number)
        editable.ReplaceAtom(atom_id, replacement)

    bond_smarts = {"1": "-", "2": "=", "3": "#", "1.5": ":"}
    overrides = {int(key): str(value) for key, value in payload.get("bondOverrides", {}).items()}
    removals: list[tuple[int, int]] = []
    for bond_id, value in overrides.items():
        bond = mol.GetBondWithIdx(bond_id)
        if value == "remove":
            removals.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        elif str(value) in bond_smarts:
            editable.ReplaceBond(bond_id, Chem.BondFromSmarts(bond_smarts[str(value)]))
        else:
            raise ValueError(f"Unsupported product bond type: {value}")
    for begin, end in removals:
        editable.RemoveBond(begin, end)

    active_pairs = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in editable.GetBonds()
    }
    for item in payload.get("addedBonds", []):
        begin, end = int(item.get("begin", -1)), int(item.get("end", -1))
        order = str(item.get("order", "1"))
        if begin == end or begin not in expected or end not in expected:
            raise ValueError(f"New bonds must connect two different, non-deleted product atoms: {begin}, {end}")
        pair = tuple(sorted((begin, end)))
        if pair in active_pairs:
            raise ValueError(f"A product bond already exists between atoms {begin} and {end}.")
        if order not in bond_smarts:
            raise ValueError(f"Unsupported new product bond type: {order}")
        editable.AddBond(begin, end, Chem.BondType.SINGLE)
        editable.ReplaceBond(editable.GetNumBonds() - 1, Chem.BondFromSmarts(bond_smarts[order]))
        active_pairs.add(pair)

    for atom_id in sorted(deleted_atoms, reverse=True):
        editable.RemoveAtom(atom_id)

    return {"smarts": Chem.MolToSmarts(editable.GetMol(), isomericSmiles=False)}


def product_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Recover visual-editor state from an existing reactant/product SMARTS pair."""
    reactant = Chem.MolFromSmarts(str(payload.get("reactantSmarts", "")))
    product_mol = Chem.MolFromSmarts(str(payload.get("productSmarts", "")))
    if reactant is None:
        raise ValueError("Invalid reactant SMARTS.")
    if product_mol is None:
        raise ValueError("Invalid product SMARTS.")

    reactant_by_map = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in reactant.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    product_by_map = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in product_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    if not reactant_by_map:
        raise ValueError("Reactant SMARTS does not contain mapped atoms.")
    unknown = sorted(set(product_by_map) - set(reactant_by_map))
    if unknown:
        raise ValueError(f"Product contains atom maps absent from the reactant: {unknown}")

    kept_atoms = sorted(reactant_by_map[number] for number in product_by_map)
    atom_map_by_source = {str(source): number for number, source in reactant_by_map.items()}
    product_bonds: dict[tuple[int, int], float] = {}
    for bond in product_mol.GetBonds():
        begin = product_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        end = product_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        if begin > 0 and end > 0:
            order = 1.5 if bond.GetIsAromatic() else float(bond.GetBondTypeAsDouble())
            product_bonds[tuple(sorted((begin, end)))] = order

    overrides: dict[str, str] = {}
    kept = set(kept_atoms)
    for bond in reactant.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin_idx not in kept or end_idx not in kept:
            continue
        begin_map = reactant.GetAtomWithIdx(begin_idx).GetAtomMapNum()
        end_map = reactant.GetAtomWithIdx(end_idx).GetAtomMapNum()
        product_order = product_bonds.get(tuple(sorted((begin_map, end_map))))
        if product_order is None:
            overrides[str(bond.GetIdx())] = "remove"
            continue
        reactant_order = 1.5 if bond.GetIsAromatic() else float(bond.GetBondTypeAsDouble())
        if product_order != reactant_order:
            overrides[str(bond.GetIdx())] = f"{product_order:g}"

    return {
        "keptAtoms": kept_atoms,
        "atomMapBySource": atom_map_by_source,
        "bondOverrides": overrides,
    }


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    products = tuple(ProductRule(name=str(item.get("name", "")), smarts=str(item.get("smarts", ""))) for item in payload.get("products", []))
    pattern = _CleavagePattern.from_rules(name=str(payload.get("name", "")), reactant_smarts=str(payload.get("reactant_smarts", "")), products=products)
    return {"valid": True, "pattern": pattern.to_dict()}


COMMANDS = {
    "molecule": molecule, "depict": depict, "reactant": reactant,
    "product": product, "productFromStructure": product_from_structure,
    "productState": product_state, "validate": validate,
}


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
