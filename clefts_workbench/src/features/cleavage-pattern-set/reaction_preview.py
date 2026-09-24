"""Direct RDKit reaction inspection, with source-indexed matches (no tree)."""
from __future__ import annotations
from typing import Any
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors, rdChemReactions, rdqueries
from functools import lru_cache
import json
from rdkit.Chem.Draw import rdMolDraw2D

from clefts.domain.fragment.cleavage import CleavageActionSequence, CleavagePatternSet
from clefts.domain.fragment.cleavage.CleavageActionGenerator import create_cleavage_actions
from clefts.domain.fragment.tree import FragmentTreeBuilder
from clefts.libs.mmkit.mmkit import Compound

LIMIT = 512


def drawing(mol: Chem.Mol, width=760, height=500) -> dict[str, Any]:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    options.fixedBondLength = 45
    options.bondLineWidth = 2
    options.setAtomPalette({-1: (.84,.89,.94), 0: (.84,.89,.94), 6: (.84,.89,.94),
                            7: (.35,.7,1), 8: (1,.42,.48), 9: (.4,.85,.57),
                            16: (.95,.8,.3), 17: (.4,.85,.57)})
    drawer.DrawMolecule(mol)
    positions = [{"index": i, "x": drawer.GetDrawCoords(i).x, "y": drawer.GetDrawCoords(i).y}
                 for i in range(mol.GetNumAtoms())]
    drawer.FinishDrawing()
    return {"svg": drawer.GetDrawingText().replace("svg:", "").replace("#D6E2EF", "var(--vscode-editor-foreground,#d6e2ef)"), "positions": positions,
            "bonds": [{"index": b.GetIdx(), "begin": b.GetBeginAtomIdx(), "end": b.GetEndAtomIdx()}
                      for b in mol.GetBonds()]}


def source_molecule(payload):
    mol = Chem.MolFromSmiles(str(payload.get("smiles", "")).strip())
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Enter a valid, non-empty SMILES.")
    return mol


@lru_cache(maxsize=256)
def reactant_query(smarts):
    query = Chem.MolFromSmarts(smarts)
    if query is None or not query.GetNumAtoms():
        raise ValueError(f"Invalid reactant SMARTS: {smarts}")
    maps = [a.GetAtomMapNum() for a in query.GetAtoms()]
    if any(number <= 0 for number in maps) or len(set(maps)) != len(maps):
        raise ValueError("Reactant atoms must have unique positive atom maps.")
    return query


def matched_sites(mol, query, index=0):
    assignments = mol.GetSubstructMatches(query, uniquify=False, maxMatches=LIMIT + 1)
    sites = {}
    for assignment in assignments[:LIMIT]:
        bonds = sorted(mol.GetBondBetweenAtoms(assignment[b.GetBeginAtomIdx()], assignment[b.GetEndAtomIdx()]).GetIdx()
                       for b in query.GetBonds())
        key = (tuple(sorted(assignment)), tuple(bonds))
        if key not in sites:
            sites[key] = {"id": f"{index}:{len(sites)}", "atoms": list(key[0]), "bonds": bonds,
                          "products": [], "errors": [], "assignments": 0, "loaded": False,
                          "bindings": []}
        sites[key]["assignments"] += 1
        sites[key]["bindings"].append(assignment)
    return list(sites.values()), len(assignments) > LIMIT


def reaction_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Match reactant queries only; do not compile/run reactions or draw products."""
    mol = source_molecule(payload)
    definitions = payload.get("patterns", [])
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("Select at least one Cleavage Pattern.")
    result = {"smiles": Chem.MolToSmiles(mol), "drawing": drawing(mol), "patterns": [], "truncated": False}
    for index, definition in enumerate(definitions):
        name = str(definition.get("name", "")) or f"Pattern {index + 1}"
        row = {"index": index, "name": name, "reactantSmarts": definition.get("reactant_smarts", ""),
               "productCount": len(definition.get("products", [])), "matches": [], "errors": []}
        result["patterns"].append(row)
        try:
            query = reactant_query(str(definition.get("reactant_smarts", "")))
            sites, truncated = matched_sites(mol, query, index)
            for site in sites:
                del site["bindings"]
            row["matches"] = sites
            result["truncated"] |= truncated
        except Exception as exc:
            row["errors"].append(str(exc))
    return result


@lru_cache(maxsize=128)
def compiled_pattern(definition_json):
    definition = json.loads(definition_json)
    reactant = str(definition.get("reactant_smarts", ""))
    reactant_query(reactant)
    products = {str(p.get("smarts", "")): p for p in definition.get("products", [])}
    reactions = []
    for smarts, product in sorted(products.items()):
        smirks = f"{reactant}>>{smarts}"
        rxn = rdChemReactions.ReactionFromSmarts(smirks)
        if rxn is None or rxn.GetNumReactantTemplates() != 1 or not rxn.GetNumProductTemplates():
            raise ValueError(f"Invalid single-reactant SMIRKS: {smirks}")
        maps = [atom.GetAtomMapNum() for template in rxn.GetProducts() for atom in template.GetAtoms()]
        if any(number <= 0 for number in maps) or len(set(maps)) != len(maps):
            raise ValueError("Product atoms must have unique positive atom maps.")
        reactions.append((str(product.get("name", "")) or "Product", smarts, smirks, rxn))
    return tuple(reactions)


@lru_cache(maxsize=512)
def product_drawing(smiles):
    return drawing(Chem.MolFromSmiles(smiles), 360, 220)


def reaction_preview_products(payload: dict[str, Any]) -> dict[str, Any]:
    """Run SMIRKS only at the selected physical site, including symmetric roles."""
    mol = source_molecule(payload)
    definition = payload.get("pattern", {})
    query = reactant_query(str(definition.get("reactant_smarts", "")))
    sites, truncated = matched_sites(mol, query)
    atoms, bonds = sorted(payload.get("atoms", [])), sorted(payload.get("bonds", []))
    site = next((s for s in sites if s["atoms"] == atoms and s["bonds"] == bonds), None)
    if site is None:
        raise ValueError("The selected site no longer matches the Reactant SMARTS.")
    result = {"products": [], "errors": [], "truncated": truncated}
    try:
        reactions = compiled_pattern(json.dumps(definition, sort_keys=True))
    except Exception as exc:
        result["errors"].append(str(exc))
        return result
    for atom in mol.GetAtoms():
        atom.SetIntProp("_clefts_preview_index", atom.GetIdx())
    seen = set()
    for name, smarts, smirks, template in reactions:
        for binding in site["bindings"]:
            # Bind each query role to its selected input atom. RunReactants then
            # yields this site's products, without enumerating every other site.
            rxn = rdChemReactions.ChemicalReaction(template)
            for atom in rxn.GetReactantTemplate(0).GetAtoms():
                atom.ExpandQuery(rdqueries.HasIntPropWithValueQueryAtom("_clefts_preview_index", binding[atom.GetIdx()]))
            rxn.Initialize(silent=True)
            try:
                groups = rxn.RunReactants((mol,), maxProducts=LIMIT + 1)
            except Exception as exc:
                result["errors"].append(f"{name}: {exc}")
                continue
            result["truncated"] |= len(groups) > LIMIT
            for group in groups[:LIMIT]:
                try:
                    components = []
                    for generated in group:
                        product = Chem.Mol(generated)
                        for atom in product.GetAtoms():
                            atom.SetAtomMapNum(0)
                        Chem.SanitizeMol(product)
                        for fragment in Chem.GetMolFrags(product, asMols=True):
                            components.append({"smiles": Chem.MolToSmiles(fragment),
                                               "formula": rdMolDescriptors.CalcMolFormula(fragment),
                                               "exactMass": rdMolDescriptors.CalcExactMolWt(fragment),
                                               "sourceAtoms": sorted({a.GetIntProp("react_atom_idx") for a in fragment.GetAtoms()
                                                                      if a.HasProp("react_atom_idx")})})
                    key = (smarts, tuple(sorted(c["smiles"] for c in components)))
                    if key in seen:
                        continue
                    for component in components:
                        component["drawing"] = product_drawing(component["smiles"])
                    seen.add(key)
                    result["products"].append({"name": name or "Product",
                                               "smarts": smarts, "smirks": smirks,
                                               "smiles": ".".join(c["smiles"] for c in components), "molecules": components})
                except Exception as exc:
                    message = f"{name or 'Product'}: {exc}"
                    if message not in result["errors"]:
                        result["errors"].append(message)
    return result


def _pattern_set(payload: dict[str, Any]) -> CleavagePatternSet:
    definitions = payload.get("patterns", [])
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("Select at least one Cleavage Pattern.")
    return CleavagePatternSet.from_dict({"name": str(payload.get("name", "Preview")),
                                         "patterns": definitions})


def _molecule_summary(compound: Compound) -> dict[str, Any]:
    mol = compound.mol
    smiles = compound.smiles
    return {"smiles": smiles, "formula": rdMolDescriptors.CalcMolFormula(mol),
            "exactMass": rdMolDescriptors.CalcExactMolWt(mol),
            "drawing": product_drawing(smiles)}


def _action_summary(action, action_id: int, patterns: CleavagePatternSet,
                    map_to_index: dict[int, int]) -> dict[str, Any]:
    pattern = patterns.get_pattern(action.cleavage_pattern_id)
    reaction = next(r for r in pattern.cleavage_reactions if r.id == action.reaction_id)
    name = reaction.source_rule.name or f"Product {action.product_molecule_id + 1}"
    return {
        "id": action_id, "patternIndex": action.cleavage_pattern_id,
        "patternName": pattern.name, "name": name,
        "productMoleculeId": action.product_molecule_id,
        # Keep query-role order. Symmetric matches at the same physical site can
        # produce different actions, so sorting would erase the visible distinction.
        "atoms": [map_to_index[m] for m in action.source_atom_maps],
        # Filled below after Source-map edges are converted to SVG bond indexes.
        "bonds": [],
        "sourceAtomMaps": list(action.source_atom_maps),
        "retainedAtomMaps": sorted(action.retained_atom_maps),
        "changedBondMaps": [list(edge) for edge in sorted(action.changed_bond_maps)],
        "reactionBondMaps": [list(edge) for edge in sorted(
            action.cut_bond_maps | frozenset((u, v) for u, v, _ in action.bond_updates))],
    }


def _run_sequence(source: Compound, sequence: CleavageActionSequence) -> Compound | None:
    try:
        products = sequence.compile(source).run(source)
        return Compound(products[0]) if len(products) == 1 else None
    except (Chem.rdchem.MolSanitizeException, ValueError):
        return None


def cleavage_explore(payload: dict[str, Any]) -> dict[str, Any]:
    """Explore Source-anchored primitive actions and unordered action sets."""
    source = Compound.from_smiles(str(payload.get("smiles", "")).strip())
    patterns = _pattern_set(payload)
    raw_maximum = payload.get("maxActionCount", 1)
    if isinstance(raw_maximum, bool) or not isinstance(raw_maximum, int):
        raise ValueError("Max actions must be a positive integer.")
    maximum = raw_maximum
    if maximum < 1:
        raise ValueError("Max actions must be a positive integer.")
    actions = create_cleavage_actions(source, patterns)
    mapped = source.mapped_mol
    map_to_index = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mapped.GetAtoms()}
    edge_to_bond = {tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(),
                                  bond.GetEndAtom().GetAtomMapNum()))): bond.GetIdx()
                    for bond in mapped.GetBonds()}
    summaries = []
    for action_id, action in enumerate(actions):
        item = _action_summary(action, action_id, patterns, map_to_index)
        item["bonds"] = sorted(edge_to_bond[edge] for edge in action.matched_bond_maps)
        item["reactionBonds"] = sorted(edge_to_bond[tuple(edge)]
                                       for edge in item["reactionBondMaps"])
        summaries.append(item)

    requested_ids = tuple(int(value) for value in payload.get("selectedActionIds", []))
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("The same action cannot be selected more than once.")
    chosen_ids = requested_ids
    if any(value < 0 or value >= len(actions) for value in chosen_ids):
        raise ValueError("A selected action no longer exists. Run the viewer again.")
    selected = tuple(actions[value] for value in chosen_ids)
    if len(selected) > maximum:
        raise ValueError("The selected actions exceed Max actions.")
    selected_sequence = CleavageActionSequence(selected) if selected else None
    if selected_sequence is not None and len(selected_sequence.actions) != len(selected):
        raise ValueError("The selected action is redundant with the current action set.")

    available = []
    candidate_sequences = {}
    if len(selected) < maximum:
        for action_id, action in enumerate(actions):
            if action_id in chosen_ids:
                continue
            try:
                sequence = CleavageActionSequence((*selected, action))
                if (len(sequence.actions) == len(selected) + 1 and sequence.retained_atom_maps
                        and len(sequence.actions) <= maximum):
                    if _run_sequence(source, sequence) is not None:
                        available.append(action_id)
                        candidate_sequences[action_id] = sequence
            except ValueError:
                pass

    selected_product = None
    if selected_sequence is not None:
        product = _run_sequence(source, selected_sequence)
        if product is None:
            raise ValueError("The selected action set did not produce a valid fragment.")
        selected_product = _molecule_summary(product)

    candidate_product = None
    inspect_id = payload.get("inspectActionId")
    if inspect_id is not None:
        inspect_id = int(inspect_id)
        if inspect_id not in candidate_sequences:
            raise ValueError("The inspected action is not valid for the current action set.")
        candidate_product = _run_sequence(source, candidate_sequences[inspect_id])
        candidate_product = _molecule_summary(candidate_product) if candidate_product else None

    result = {"smiles": source.smiles, "drawing": drawing(source.mol),
              "actions": summaries, "availableActionIds": available,
              "selectedActionIds": list(chosen_ids), "selectedProduct": selected_product,
              "candidateProduct": candidate_product, "inspectedActionId": inspect_id,
              "maxActionCount": maximum, "results": [], "truncated": False}
    if str(payload.get("mode", "stepwise")) == "exhaustive":
        builder = FragmentTreeBuilder(maximum, patterns, only_add_min_action_count=False)
        search, candidates = builder._search(source, seed_action_sequences=None,
                                             max_action_count=maximum)
        action_ids = {action: index for index, action in enumerate(actions)}
        for _, item in builder._materialize_iter(source, search, candidates):
            ids = [action_ids[action] for action in item.action_sequence.actions]
            result["results"].append({"actionIds": ids, "actionCount": len(ids),
                                      "smirks": item.smirks,
                                      "molecule": _molecule_summary(item.compound)})
    return result
