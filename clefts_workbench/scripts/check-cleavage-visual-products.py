from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


backend_path = Path(__file__).parents[1] / "src/features/cleavage-pattern-set/backend.py"
sys.path.insert(0, str(Path(__file__).parents[2]))
spec = importlib.util.spec_from_file_location("cleavage_pattern_backend", backend_path)
assert spec is not None and spec.loader is not None
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

reactant = "[#6:1]-[#8:2]-[#7:3]"
state = backend.product_state({
    "reactantSmarts": reactant,
    "productSmarts": "[#6:1]=[#8:2]",
})
assert state["keptAtoms"] == [0, 1]
assert state["atomMapBySource"] == {"0": 1, "1": 2, "2": 3}
assert state["bondOverrides"] == {"0": "2"}

round_trip = backend.product({
    "sourceType": "smarts", "smarts": reactant, "atoms": [0, 1, 2],
    "keptAtoms": state["keptAtoms"], "atomMapBySource": state["atomMapBySource"],
    "bondOverrides": state["bondOverrides"],
})
assert round_trip["smarts"] == "[#6:1]=[#8:2]"

disconnected = backend.product_state({
    "reactantSmarts": "[#6:1]-[#8:2]",
    "productSmarts": "[#6:1].[#8:2]",
})
assert disconnected["keptAtoms"] == [0, 1]
assert disconnected["bondOverrides"] == {"0": "remove"}

ring = backend.product_from_structure({
    "sourceType": "smiles", "smiles": "C1CCCCC1O",
    "reactantSmarts": "[#6:1]-[#6:2]-[#6:3]-[#6:4]-[#6:5]-[#6:6]-[#8:7]",
    "atomMaps": {str(index): index + 1 for index in range(7)},
    "atomOverrides": {"0": "#7"},
    "bondOverrides": {"0": "2", "6": "remove"},
})
ring_mol = backend.Chem.MolFromSmarts(ring["smarts"])
assert ring_mol is not None
assert [atom.GetAtomMapNum() for atom in ring_mol.GetAtoms()] == list(range(1, 8))
assert ring_mol.GetAtomWithIdx(0).GetAtomicNum() == 7
assert ring_mol.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == 2
assert ring_mol.GetBondBetweenAtoms(5, 0) is None
assert backend.validate({
    "name": "ring_rearrangement",
    "reactant_smarts": "[#6:1]-[#6:2]-[#6:3]-[#6:4]-[#6:5]-[#6:6]-[#8:7]",
    "products": [{"name": "changed_ring", "smarts": ring["smarts"]}],
})["valid"]

rewired = backend.product_from_structure({
    "sourceType": "smiles", "smiles": "CCCCO",
    "reactantSmarts": "[#6:1]-[#6:2]-[#6:3]-[#6:4]-[#8:5]",
    "atomMaps": {str(index): index + 1 for index in range(5)},
    "deletedAtoms": [4],
    "addedBonds": [{"begin": 0, "end": 3, "order": "1"}],
})
rewired_mol = backend.Chem.MolFromSmarts(rewired["smarts"])
assert rewired_mol is not None and rewired_mol.GetNumAtoms() == 4
assert rewired_mol.GetBondBetweenAtoms(0, 3) is not None
assert sorted(atom.GetAtomMapNum() for atom in rewired_mol.GetAtoms()) == [1, 2, 3, 4]

# Preserve complex atom queries and non-sequential atom maps when an existing
# SMARTS pattern is opened visually and applied without simplifying it.
complex_smarts = (
    r"[#8:1]=[#6:2]1-[#6;!$([#6]-[OX2H1]):3]-[#6:4](-\[#6:5]2:"
    r"[#6:6]:[#6:7]:[#6:8]:[#6:9]:[#6:10]:2)-[#8:11]-[#6:13]2:"
    r"[#6:12]-1:[#6:17]:[#6:16]:[#6:15]:[#6:14]:2"
)
complex_graph = backend.molecule({"sourceType": "smarts", "smarts": complex_smarts})
assert len(complex_graph["atoms"]) == 17
assert len(complex_graph["bonds"]) == 19

complex_constraints = {}
complex_maps = {}
for atom in complex_graph["atoms"]:
    query_atom = backend.Chem.AtomFromSmarts(atom["smarts"])
    assert query_atom is not None
    complex_maps[str(atom["index"])] = atom["mapNumber"]
    query_atom.SetAtomMapNum(0)
    complex_constraints[str(atom["index"])] = {
        "mode": "custom",
        "smarts": query_atom.GetSmarts(),
    }

complex_bond_constraints = {}
for bond in complex_graph["bonds"]:
    bond_type = {
        1.0: "single", 1.5: "aromatic", 2.0: "double", 3.0: "triple",
    }[bond["order"]]
    complex_bond_constraints[str(bond["index"])] = {
        "types": [bond_type],
        "ring": False,
    }

complex_round_trip = backend.reactant({
    "sourceType": "smarts",
    "smarts": complex_smarts,
    "name": "conditional_flavonoid",
    "atoms": [atom["index"] for atom in complex_graph["atoms"]],
    "bonds": [bond["index"] for bond in complex_graph["bonds"]],
    "constraints": complex_constraints,
    # editVisualPattern exposes every existing bond, so submit the same state
    # that the webview sends when Apply Reactant is clicked without edits.
    "bondConstraints": complex_bond_constraints,
    "atomMapBySource": complex_maps,
})
complex_round_trip_mol = backend.Chem.MolFromSmarts(complex_round_trip["smarts"])
assert complex_round_trip_mol is not None
assert [atom.GetAtomMapNum() for atom in complex_round_trip_mol.GetAtoms()] == [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 12, 17, 16, 15, 14,
]
assert "$(" in complex_round_trip["smarts"]
assert "!$(" in complex_round_trip["smarts"]
assert "X2" in complex_round_trip["smarts"]
assert "H1" in complex_round_trip["smarts"]
assert backend.validate({
    "name": "conditional_flavonoid",
    "reactant_smarts": complex_round_trip["smarts"],
    "products": [],
})["valid"]

unconstrained_product = backend.product_from_structure({
    "sourceType": "smarts",
    "smarts": complex_smarts,
    "reactantSmarts": complex_round_trip["smarts"],
    "atomMaps": complex_maps,
})
unconstrained_product_mol = backend.Chem.MolFromSmarts(unconstrained_product["smarts"])
assert unconstrained_product_mol is not None
assert "$(" not in unconstrained_product["smarts"]
assert "X2" not in unconstrained_product["smarts"]
assert "H1" not in unconstrained_product["smarts"]
assert unconstrained_product_mol.GetAtomWithIdx(2).GetSmarts() == "[#6:3]"

aromatic_complex_smarts = (
    r"[#8:1]=[#6:2]1:[#6;!$([#6]-[OX2H1]):3]:[#6:4](-\[#6:5]2:"
    r"\[#6:6]:\[#6:7]:\[#6:8]:\[#6:9]:\[#6:10]:2):[#8:11]:[#6:12]2:"
    r"[#6:13]:1:[#6:14]:[#6:15]:[#6:16]:[#6:17]:2"
)
aromatic_graph = backend.molecule({
    "sourceType": "smarts",
    "smarts": aromatic_complex_smarts,
})
assert len(aromatic_graph["atoms"]) == 17
assert len(aromatic_graph["bonds"]) == 19
assert aromatic_graph["atoms"][2]["smarts"].startswith("[#6&!$(")
assert backend.validate({
    "name": "aromatic_conditional_flavonoid",
    "reactant_smarts": aromatic_complex_smarts,
    "products": [],
})["valid"]

extension = (Path(__file__).parents[1] / "src/extension.js").read_text(encoding="utf-8")
assert 'id="builderProductSelect"' in extension
assert "chemistry('productFromStructure'" in extension
assert "atom.mapNumber||index+1" in extension
assert 'id="allowProductAtomTypes"' in extension
assert "Delete Atom" in extension
assert "Delete Selected" in extension
assert "productSelectedAtoms" in extension
assert "productCanvas.onpointerdown" in extension
assert "data-delete-selected-product-atoms" in extension
assert ".selection-lasso{stroke:#fff}" in extension
assert "Start New Bond" in extension
assert "Update Product" in extension
assert "Custom SMARTS" in extension
assert "atomMapBySource" in extension
assert 'id="builderSourceType"' in extension
assert "Source SMILES / SMARTS" in extension
print("Complex reactant and visual product round-trip checks passed.")
