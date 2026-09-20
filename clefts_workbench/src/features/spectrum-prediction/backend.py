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

from clefts.libs.mmkit.mmkit import Adduct, Compound, Formula
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.ml.specgen.predict_spectrum import load_generator


def _build_tree(result, sample: int, tensorizer, threshold: float) -> dict[str, Any]:
    """Fragment-tree nodes/edges explored for this prediction, annotated with
    per-node neutral formula/exact mass plus any predicted-peak ion states."""
    fragments = result.fragments
    node_offsets = [i for i, s in enumerate(fragments.node_sample_index.tolist()) if s == sample]
    local_index = {global_index: local for local, global_index in enumerate(node_offsets)}
    num_nodes = len(node_offsets)

    edge_pairs = fragments.edge_index.detach().cpu().tolist()
    edges = [
        {"id": i, "source": local_index[src], "target": local_index[dst]}
        for i, (src, dst) in enumerate(zip(*edge_pairs))
        if src in local_index and dst in local_index
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

    downstream = result.downstream
    spectra = result.spectra
    ion_node_index = downstream.ion_node_index.detach().cpu().tolist()
    ion_formula_index = downstream.ion_formula_index.detach().cpu().tolist()
    ion_probability = spectra.ion_probability.detach().cpu().tolist()
    ion_adduct = downstream.ion_adduct
    formula_sample_index = downstream.formula_sample_index.detach().cpu().tolist()
    formula_mz = downstream.formula_mz.detach().cpu().tolist()
    formula_tensor = downstream.formula_tensor
    intensity = spectra.intensity.detach().cpu().tolist()

    # Candidate ions below the trained confidence threshold are the same
    # unrelated adduct/hydrogen-shift guesses excluded from the peak list
    # below; keep the node inspector consistent with what the spectrum shows.
    annotations_by_node: dict[int, list[dict[str, Any]]] = {}
    for entry, (node, node_ion_formula) in enumerate(zip(ion_node_index, ion_formula_index)):
        if node not in local_index or formula_sample_index[node_ion_formula] != sample:
            continue
        probability = float(ion_probability[entry])
        if probability < threshold:
            continue
        annotations_by_node.setdefault(local_index[node], []).append(
            {
                "peakMz": float(formula_mz[node_ion_formula]),
                "peakIntensity": float(intensity[node_ion_formula]),
                "adduct": ion_adduct[entry] if entry < len(ion_adduct) else "",
                "ionFormula": str(tensorizer.tensor_to_formula(formula_tensor[node_ion_formula])),
                "probability": probability,
            }
        )

    nodes = []
    for global_index in node_offsets:
        compound = fragments.compounds[global_index]
        neutral_formula = compound.formula
        local = local_index[global_index]
        nodes.append(
            {
                "id": local,
                "smiles": compound.smiles,
                "depth": depth[local],
                "precursor": depth[local] == 0,
                "formula": str(neutral_formula),
                "exactMass": neutral_formula.exact_mass,
                "annotations": annotations_by_node.get(local, []),
            }
        )

    return {"nodes": nodes, "edges": edges}


def list_adducts(request: dict[str, Any]) -> dict[str, Any]:
    """Main adduct types (the measurement-condition adducts) that this
    checkpoint was trained to condition on."""
    model_path = str(request.get("modelPath", "")).strip()
    if not model_path:
        raise ValueError("Model checkpoint path is required.")
    if not Path(model_path).exists():
        raise ValueError(f"Model checkpoint was not found: {model_path}")

    device = torch.device(str(request.get("device") or "cpu"))
    generator = load_generator(
        model_path=model_path, params_path=None, device=device, strict=False
    )
    return {"adducts": list(generator.adduct_type_strs)}


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
    source = Compound.from_smiles(smiles)
    neutral_formula = Formula.from_mol(mol).plain
    precursor_mz = adduct.apply_to_mz(neutral_formula.exact_mass)
    ce_ev = parse_ce_to_ev(ce, precursor_mz)
    if ce_ev is None:
        raise ValueError(f"Could not interpret collision energy: {ce!r}")

    with torch.no_grad():
        result = generator.predict([source], [adduct], [ce_ev])

    if not result.spectra.mz.numel():
        raise ValueError("The model did not produce any peaks for this input.")
    threshold = generator.post_model.ion_prediction_threshold
    peaks = sorted(
        (
            {
                "mz": float(mz),
                "intensity": float(intensity),
                "formula": str(generator.tensorizer.tensor_to_formula(formula_row)),
                "confidence": float(confidence),
            }
            for mz, intensity, formula_row, confidence in zip(
                result.spectra.mz.detach().cpu().tolist(),
                result.spectra.intensity.detach().cpu().tolist(),
                result.spectra.formula_tensor,
                result.spectra.confidence.detach().cpu().tolist(),
            )
            # Below the trained ion confidence threshold: the same unrelated
            # adduct/hydrogen-shift candidate this model now learns to push
            # toward zero, kept out of the predicted spectrum shown here.
            if confidence >= threshold
        ),
        key=lambda peak: peak["mz"],
    )
    if not peaks:
        raise ValueError("Every candidate peak was below the model's ion confidence threshold.")
    tree = _build_tree(result, sample=0, tensorizer=generator.tensorizer, threshold=threshold)

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")

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
        # load_generator prints progress lines; redirect them to stderr so
        # stdout stays a single JSON line.
        with contextlib.redirect_stdout(sys.stderr):
            result = COMMANDS[command](request.get("payload", {}))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
