"""Backend for predicting a single MS/MS spectrum from SMILES + CE + adduct."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from clefts.libs.mmkit.mmkit import Adduct, Formula
from clefts.ml.specgen.predict_spectrum import (
    build_structure_from_dataset,
    direct_input_dataset,
    load_generator,
)


def _node_formula(structure, node_id: int, elements: tuple[str, ...]) -> Formula:
    row = structure.node_formula[node_id].detach().cpu().tolist()
    charge = int(row[-1]) if row else 0
    return Formula({name: int(value) for name, value in zip(elements, row[:-1]) if value}, charge=charge)


def _build_tree(structure, spectrum) -> dict[str, Any]:
    """Fragment-tree nodes/edges explored for this prediction, annotated with
    per-node neutral formula/exact mass plus any predicted-peak ion states."""
    elements = tuple(getattr(structure, "formula_element_order", ()))
    num_nodes = int(structure.num_nodes)
    edge_pairs = structure.edge_index.detach().cpu().tolist()
    edges = [
        {"id": i, "source": int(edge_pairs[0][i]), "target": int(edge_pairs[1][i])}
        for i in range(len(edge_pairs[0]))
    ]

    indegree = [0] * num_nodes
    for edge in edges:
        indegree[edge["target"]] += 1
    depth = [0 if indegree[i] == 0 else -1 for i in range(num_nodes)]
    for _ in range(num_nodes):
        changed = False
        for edge in edges:
            if depth[edge["source"]] < 0:
                continue
            candidate = depth[edge["source"]] + 1
            if depth[edge["target"]] < 0 or candidate < depth[edge["target"]]:
                depth[edge["target"]] = candidate
                changed = True
        if not changed:
            break

    annotations_by_node: dict[int, list[dict[str, Any]]] = {}
    for peak in spectrum.peaks:
        for annotation in peak.fragment_ion_annotations or []:
            annotations_by_node.setdefault(annotation.global_node_id, []).append(
                {
                    "peakMz": float(peak.mz),
                    "peakIntensity": float(peak.intensity),
                    "ionFormula": annotation.formula,
                    "adduct": str(annotation.adduct),
                    "probability": float(annotation.probability),
                }
            )

    nodes = []
    for node_id in range(num_nodes):
        neutral_formula = _node_formula(structure, node_id, elements)
        nodes.append(
            {
                "id": node_id,
                "smiles": str(structure.node_smiles[node_id]),
                "depth": depth[node_id],
                "formula": str(neutral_formula),
                "exactMass": neutral_formula.exact_mass,
                "annotations": annotations_by_node.get(node_id, []),
            }
        )

    return {"nodes": nodes, "edges": edges}


def list_adducts(request: dict[str, Any]) -> dict[str, Any]:
    """Main adduct types (the measurement-condition adducts, not per-fragment
    ion-shift states) that this checkpoint was trained to condition on."""
    model_path = str(request.get("modelPath", "")).strip()
    if not model_path:
        raise ValueError("Model checkpoint path is required.")
    if not Path(model_path).exists():
        raise ValueError(f"Model checkpoint was not found: {model_path}")

    device = torch.device(str(request.get("device") or "cpu"))
    generator = load_generator(
        model_path=model_path, params_path=None, device=device, strict=False
    )
    adducts = [
        str(adduct)
        for _, adduct in sorted(generator.probability_model.main_adduct_types.items())
    ]
    return {"adducts": adducts}


def predict(request: dict[str, Any]) -> dict[str, Any]:
    smiles = str(request.get("smiles", "")).strip()
    if not smiles:
        raise ValueError("SMILES is required.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")

    adduct_type = str(request.get("adductType", "")).strip()
    if not adduct_type:
        raise ValueError("Adduct type is required.")
    adduct = Adduct.parse(adduct_type)

    ce = str(request.get("ce", "")).strip()
    if not ce:
        raise ValueError("Collision energy is required.")

    model_path = str(request.get("modelPath", "")).strip()
    if not model_path:
        raise ValueError("Model checkpoint path is required.")
    if not Path(model_path).exists():
        raise ValueError(f"Model checkpoint was not found: {model_path}")

    device = torch.device(str(request.get("device") or "cpu"))

    generator = load_generator(
        model_path=model_path, params_path=None, device=device, strict=False
    )
    dataset = direct_input_dataset(
        smiles_values=[smiles], collision_energy=ce, adduct_type=adduct_type
    )
    structure, _metadata = build_structure_from_dataset(
        dataset=dataset,
        generator=generator,
        smiles_values=[smiles],
        smiles_column="SMILES",
        precursor_mz_column="PrecursorMZ",
        adduct_type_column="AdductType",
        collision_energy_column="CollisionEnergy",
        instrument_column=None,
        device=device,
    )

    with torch.no_grad():
        output = generator(
            structure,
            include_formula_annotation=True,
            include_fragment_ion_annotation=True,
        )
    if not output.spectra:
        raise ValueError("The model did not produce any peaks for this input.")
    spectrum = output.spectra[0]
    peaks = sorted(
        (
            {"mz": float(peak.mz), "intensity": float(peak.intensity), "formula": peak.formula}
            for peak in spectrum.peaks
        ),
        key=lambda peak: peak["mz"],
    )
    tree_structure = output.candidate_output.features.structure
    tree = _build_tree(tree_structure, spectrum)

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")

    neutral_formula = Formula.from_mol(mol).plain
    precursor_mz = adduct.apply_to_mz(neutral_formula.exact_mass)

    return {
        "canonicalSmiles": Chem.MolToSmiles(mol),
        "svg": svg,
        "formula": str(neutral_formula),
        "exactMass": neutral_formula.exact_mass,
        "adduct": str(adduct),
        "precursorMz": precursor_mz,
        "peaks": peaks,
        "tree": tree,
    }


COMMANDS = {"predict": predict, "listAdducts": list_adducts}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        command = str(request.get("command", "predict"))
        if command not in COMMANDS:
            raise ValueError(f"Unknown command: {command}")
        # load_generator/build_structure_from_dataset print progress lines;
        # redirect them to stderr so stdout stays a single JSON line.
        with contextlib.redirect_stdout(sys.stderr):
            result = COMMANDS[command](request.get("payload", {}))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
