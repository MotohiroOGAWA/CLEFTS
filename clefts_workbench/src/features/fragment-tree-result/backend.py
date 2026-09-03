"""Backend for inspecting generated fragment-tree structures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from clefts.libs.mmkit.mmkit import Formula


def _formula(row, elements):
    values = row.detach().cpu().tolist()
    charge = int(values[-1]) if values else 0
    return str(Formula({name: int(value) for name, value in zip(elements, values[:-1]) if value}, charge=charge))


def inspect_structure(file_path: Path) -> dict:
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    structure = payload["structure"] if isinstance(payload, dict) and "structure" in payload else payload
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload, dict) else {}
    required = ("sample_peak_ptr", "peak_formula_ptr", "formula_assignment_ptr")
    if not all(hasattr(structure, name) for name in required):
        raise TypeError("This .preft.pt file is not a training fragment-tree structure.")

    elements = tuple(getattr(structure, "formula_element_order", ()))
    peak_depth = getattr(structure, "target_peak_depth", torch.full_like(structure.sample_peak_mz, -1, dtype=torch.long))
    path_edges = getattr(structure, "target_path_edge_index", torch.empty(0, dtype=torch.long))
    path_ptr = getattr(structure, "terminal_path_ptr", torch.zeros(structure.target_terminal_node_index.numel() + 1, dtype=torch.long))
    expand_nodes = getattr(structure, "target_expand_node_index", torch.empty(0, dtype=torch.long))
    expand_ptr = getattr(structure, "terminal_expand_ptr", torch.zeros(structure.target_terminal_node_index.numel() + 1, dtype=torch.long))
    samples = []
    valid_depths = peak_depth[peak_depth >= 0].long()
    assigned_peaks = int(valid_depths.numel())
    emitted_peaks = 0
    total_peaks = int(structure.sample_peak_mz.numel())
    max_samples, max_peaks = 250, 5000
    all_targets = set(int(value) for value in structure.edge_index[1].tolist())
    structural_roots = [index for index in range(int(structure.num_nodes)) if index not in all_targets]
    for sample_id in range(min(int(structure.num_samples), max_samples)):
        start, end = int(structure.sample_peak_ptr[sample_id]), int(structure.sample_peak_ptr[sample_id + 1])
        peaks = []
        for peak_id in range(start, end):
            if emitted_peaks >= max_peaks:
                break
            emitted_peaks += 1
            depth = int(peak_depth[peak_id])
            formulas = []
            f0, f1 = int(structure.peak_formula_ptr[peak_id]), int(structure.peak_formula_ptr[peak_id + 1])
            for formula_id in range(f0, f1):
                assignments = []
                a0, a1 = int(structure.formula_assignment_ptr[formula_id]), int(structure.formula_assignment_ptr[formula_id + 1])
                for assignment_id in range(a0, a1):
                    p0, p1 = int(path_ptr[assignment_id]), int(path_ptr[assignment_id + 1])
                    x0, x1 = int(expand_ptr[assignment_id]), int(expand_ptr[assignment_id + 1])
                    assignments.append({"terminalNode": int(structure.target_terminal_node_index[assignment_id]), "ion": int(structure.target_ion_index[assignment_id]), "unsaturation": int(structure.target_unsaturation_index[assignment_id]), "radical": int(structure.target_radical_index[assignment_id]), "pathEdges": [int(v) for v in path_edges[p0:p1].tolist()], "expandNodes": [int(v) for v in expand_nodes[x0:x1].tolist()]})
                formulas.append({"id": formula_id, "formula": _formula(structure.target_formula[formula_id], elements), "assignments": assignments})
            peaks.append({"index": peak_id - start, "mz": float(structure.sample_peak_mz[peak_id]), "intensity": float(structure.sample_peak_intensity[peak_id]), "depth": depth, "formulas": formulas})
        total_intensity = float(structure.sample_peak_intensity[start:end].sum())
        assigned_local = [peak_id for peak_id in range(start, end) if int(peak_depth[peak_id]) >= 0]
        assigned_intensity = sum(float(structure.sample_peak_intensity[peak_id]) for peak_id in assigned_local)
        precursor_nodes = set()
        precursor_mask = structure.precursor_sample_index == sample_id
        for row in structure.precursor_edge_index_path[precursor_mask]:
            valid = row[row >= 0]
            if valid.numel():
                precursor_nodes.add(int(structure.edge_index[1, int(valid[-1])]))
        if not precursor_nodes:
            precursor_nodes.update(structural_roots)
        raw_ce = metadata.get("sample_collision_energy_raw", [])
        raw_adduct = metadata.get("sample_adduct_types", [])
        target_path_edges = sorted({edge for peak in peaks for formula in peak["formulas"] for assignment in formula["assignments"] for edge in assignment["pathEdges"]})
        active_edge_ids = set(target_path_edges)
        active_node_ids = set(precursor_nodes)
        for edge_id in active_edge_ids:
            active_node_ids.update(int(value) for value in structure.edge_index[:, edge_id].tolist())
        active_graph = {
            "nodes": [{"id": index, "smiles": str(structure.node_smiles[index])} for index in sorted(active_node_ids)],
            "edges": [{"id": edge_id, "source": int(structure.edge_index[0, edge_id]), "target": int(structure.edge_index[1, edge_id]), "targetPath": True} for edge_id in sorted(active_edge_ids)],
            "truncated": False,
        }
        samples.append({"id": sample_id, "adductIndex": int(structure.sample_adduct_type_index[sample_id]), "adduct": raw_adduct[sample_id] if sample_id < len(raw_adduct) else "?", "collisionEnergy": float(structure.sample_ce_value[sample_id]), "collisionEnergyRaw": raw_ce[sample_id] if sample_id < len(raw_ce) else "?", "peakCount": end - start, "assignedIntensity": assigned_intensity, "totalIntensity": total_intensity, "assignmentScore": assigned_intensity / total_intensity if total_intensity > 0 else 0.0, "precursorNodes": sorted(precursor_nodes), "targetPathEdges": target_path_edges, "activeGraph": active_graph, "peaks": peaks})
    graph_node_count = min(int(structure.num_nodes), 500)
    graph_edges = [
        {"id": edge_id, "source": int(source), "target": int(target)}
        for edge_id, (source, target) in enumerate(structure.edge_index.detach().cpu().t().tolist())
        if int(source) < graph_node_count and int(target) < graph_node_count
    ]
    highlighted = set(int(value) for value in path_edges.tolist())
    for edge in graph_edges:
        edge["targetPath"] = edge["id"] in highlighted
    indegree = [0] * graph_node_count
    for edge in graph_edges:
        indegree[edge["target"]] += 1
    node_depth = [0 if indegree[index] == 0 else -1 for index in range(graph_node_count)]
    for _ in range(graph_node_count):
        changed = False
        for edge in graph_edges:
            if node_depth[edge["source"]] < 0:
                continue
            candidate = node_depth[edge["source"]] + 1
            current = node_depth[edge["target"]]
            if current < 0 or candidate < current:
                node_depth[edge["target"]] = candidate
                changed = True
        if not changed:
            break
    depth_counts = {str(depth): node_depth.count(depth) for depth in sorted(set(node_depth)) if depth >= 0}
    graph = {
        "nodes": [{"id": index, "smiles": str(structure.node_smiles[index])} for index in range(graph_node_count)],
        "edges": graph_edges,
        "truncated": graph_node_count < int(structure.num_nodes),
    }
    return {"file": str(file_path), "metadata": metadata, "summary": {"samples": int(structure.num_samples), "nodes": int(structure.num_nodes), "edges": int(structure.num_edges), "peaks": total_peaks, "assignedPeaks": assigned_peaks, "depthCounts": depth_counts}, "graph": graph, "samples": samples, "truncated": int(structure.num_samples) > max_samples or total_peaks > max_peaks}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(json.dumps({"ok": True, "result": inspect_structure(Path(request["path"]).resolve())}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
