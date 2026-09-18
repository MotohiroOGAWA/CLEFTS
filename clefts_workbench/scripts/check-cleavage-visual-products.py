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
assert "stroke:var(--accent,#168bdf)" in extension
assert "Start New Bond" in extension
assert "Update Product" in extension
assert "Custom SMARTS" in extension
assert "atomMapBySource" in extension
assert 'id="builderSourceType"' in extension
assert "Source SMILES / SMARTS" in extension
print("Complex reactant and visual product round-trip checks passed.")

# Query semantics are checked by matching real molecules, rather than string spelling.
def query(atom=None, bond=None):
    return backend.Chem.MolFromSmarts(backend.reactant({
        "sourceType": "smiles", "smiles": "CC", "atoms": [0, 1], "bonds": [0],
        "constraints": {"0": atom or {"mode": "any"}, "1": {"mode": "any"}},
        "bondConstraints": {"0": bond or {"mode": "any"}},
    })["smarts"])

def matches(q, smiles):
    return backend.Chem.MolFromSmiles(smiles).HasSubstructMatch(q)

assert matches(query({"mode": "any", "elements": ["O"], "nonHydrogen": True}), "NN")
q = query({"mode": "elements", "elements": ["N", "O"], "nonHydrogen": True})
assert matches(q, "NC") and matches(q, "OC") and not matches(q, "CC")
assert matches(query({"mode": "custom", "smarts": "[#7,#8]"}), "NC")
hydrogen = backend.Chem.MolFromSmiles("[H][H]")
assert hydrogen.HasSubstructMatch(query({"mode": "any", "nonHydrogen": False}))
assert not hydrogen.HasSubstructMatch(query({"mode": "any", "nonHydrogen": True}))
assert not hydrogen.HasSubstructMatch(query({"mode": "elements", "elements": ["H", "C"], "nonHydrogen": True}))
q = query(bond={"mode": "any", "types": ["single"], "ringStatus": "notInRing"})
assert matches(q, "C=C") and matches(q, "C1CC1")
q = query(bond={"mode": "custom", "types": ["single", "double"], "ringStatus": "any"})
assert matches(q, "CC") and matches(q, "C=C") and not matches(q, "C#C")
q = query(bond={"mode": "custom", "types": ["single"], "ringStatus": "inRing"})
assert matches(q, "C1CC1") and not matches(q, "CC")
q = query(bond={"mode": "custom", "types": ["single"], "ringStatus": "notInRing"})
assert matches(q, "CC") and not matches(q, "C1CC1")
from rdkit.Chem import rdChemReactions
reaction = rdChemReactions.ReactionFromSmarts(reactant + ">>" + round_trip["smarts"])
assert reaction is not None and reaction.GetNumProductTemplates() == 1
print("Any/custom, element OR, Non-H, bond OR, ring status and mapped reaction serialization checks passed.")

# Product graph edits and query constraints use the same mapped serializer.
changed = backend.product_from_structure({
    "sourceType": "smiles", "smiles": "CCO", "reactantSmarts": "[#6:1]-[#6:2]-[#8:3]",
    "atomMaps": {"0": 1, "1": 2, "2": 3}, "bondOverrides": {"0": "2"},
    "atomOverrides": {"2": "[!#1]"},
    "bondConstraints": {"0": {"mode": "custom", "types": ["double"], "ringStatus": "any"}},
})
changed_mol = backend.Chem.MolFromSmarts(changed["smarts"])
assert changed_mol.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == 2
assert backend.Chem.MolFromSmiles("C=CN").HasSubstructMatch(changed_mol)
assert not backend.Chem.MolFromSmiles("CCN").HasSubstructMatch(changed_mol)
new_bond = backend.product_from_structure({
    "sourceType": "smiles", "smiles": "CCC", "reactantSmarts": "[#6:1]-[#6:2]-[#6:3]",
    "atomMaps": {"0": 1, "1": 2, "2": 3}, "bondOverrides": {"0": "remove"},
    "addedBonds": [{"begin": 0, "end": 2, "order": "2", "constraint": {
        "mode": "custom", "types": ["double"], "ringStatus": "any"}}],
})
new_mol = backend.Chem.MolFromSmarts(new_bond["smarts"])
mapped = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()}
assert new_mol.GetBondBetweenAtoms(mapped[1], mapped[3]).GetBondTypeAsDouble() == 2
assert new_mol.GetBondBetweenAtoms(mapped[1], mapped[2]) is None
print("Any + Non-H, product type edits and added-bond query serialization checks passed.")

# Selection-based products inherit the entire mapped reactant query by default.
base = '[#6,#7;!#1:1]-[#6:2]=[#8:3]'
def selected_product(atoms=(0, 1, 2), bonds=(0, 1), **edits):
    return backend.product_from_selection({
        'reactantSmarts': base, 'atoms': list(atoms), 'bonds': list(bonds), **edits,
    })['smarts']

unchanged = selected_product()
source_query = backend.Chem.MolFromSmarts(base)
unchanged_query = backend.Chem.MolFromSmarts(unchanged)
assert [a.GetSmarts() for a in unchanged_query.GetAtoms()] == [a.GetSmarts() for a in source_query.GetAtoms()]
assert [b.GetSmarts() for b in unchanged_query.GetBonds()] == [b.GetSmarts() for b in source_query.GetBonds()]
split = selected_product(bonds=(1,))
assert split.count('.') == 1
assert selected_product(bonds=()).count('.') == 2
split_state = backend.product_state({'reactantSmarts': base, 'productSmarts': split})
assert split_state['keptAtoms'] == [0, 1, 2]
assert split_state['keptBonds'] == [1]
assert backend.validate({'name': 'selected_fragments', 'reactant_smarts': base,
                         'products': [{'name': 'fragments', 'smarts': split}]})['valid']
# Excluding an atom also excludes any selected bonds that end at that atom.
subset = backend.Chem.MolFromSmarts(selected_product(atoms=(0, 1)))
assert subset.GetNumAtoms() == 2 and subset.GetNumBonds() == 1
assert {a.GetAtomMapNum() for a in subset.GetAtoms()} == {1, 2}
changed_selection = selected_product(atomOverrides={'2': '[#7]'}, bondOverrides={'1': '1'})
changed_query = backend.Chem.MolFromSmarts(changed_selection)
assert changed_query.GetAtomWithIdx(2).GetAtomicNum() == 7
assert changed_query.GetBondWithIdx(1).GetBondTypeAsDouble() == 1
recovered = backend.product_state({'reactantSmarts': base, 'productSmarts': changed_selection})
assert recovered['atomOverrides'] == {'2': '[#7]'}
assert selected_product(atomOverrides=recovered['atomOverrides'], bondOverrides=recovered['bondOverrides'],
                        bondQueries=recovered['bondQueries']) == changed_selection
for included in (True, False):
    formed = selected_product(addedBonds=[{'id': 1, 'begin': 0, 'end': 2, 'order': '1', 'selected': included}])
    formed_query = backend.Chem.MolFromSmarts(formed)
    assert formed_query.GetNumBonds() == (3 if included else 2)
    if included:
        saved = backend.product_state({'reactantSmarts': base, 'productSmarts': formed})
        assert len(saved['addedBonds']) == 1
        rebuilt = selected_product(addedBonds=saved['addedBonds'])
        assert backend.Chem.MolFromSmarts(rebuilt).GetNumBonds() == 3
try:
    selected_product(atoms=())
except ValueError as error:
    assert 'Select at least one product atom' in str(error)
else:
    raise AssertionError('Empty products must not be generated')
print('Selection-based products: unchanged queries, dot-separated fragments, subset maps, explicit edits, new-bond selection and recovery passed.')

assert backend.reactant({'sourceType': 'smiles', 'smiles': 'CC', 'atoms': [0, 1], 'bonds': []})['smarts'].count('.') == 1
