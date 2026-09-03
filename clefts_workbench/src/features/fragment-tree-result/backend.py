"""Backend for inspecting generated fragment-tree structures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from clefts.libs.mmkit.mmkit import Formula
from clefts.ml.input.fragment_tree_preprocessing_context import FragmentTreePreprocessingContext


def _formula(row, elements):
    values = row.detach().cpu().tolist()
    charge = int(values[-1]) if values else 0
    return str(Formula({name: int(value) for name, value in zip(elements, values[:-1]) if value}, charge=charge))


def _preprocessing_context(file_path: Path):
    for parent in file_path.parents:
        config_file = parent / "config" / "preprocessing_config.json"
        if not config_file.exists():
            continue
        try:
            config = json.loads(config_file.read_text())
            return FragmentTreePreprocessingContext(**config)
        except (KeyError, TypeError, ValueError, OSError):
            return None
        break
    return None


def _node_depths(structure) -> list[int]:
    count = int(structure.num_nodes)
    indegree = [0] * count
    edges = [(int(source), int(target)) for source, target in structure.edge_index.detach().cpu().t().tolist()]
    for _, target in edges:
        indegree[target] += 1
    depths = [0 if indegree[index] == 0 else -1 for index in range(count)]
    for _ in range(count):
        changed = False
        for source, target in edges:
            if depths[source] < 0:
                continue
            candidate = depths[source] + 1
            if depths[target] < 0 or candidate < depths[target]:
                depths[target] = candidate
                changed = True
        if not changed:
            break
    return depths


def inspect_structure(file_path: Path) -> dict:
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    structure = payload["structure"] if isinstance(payload, dict) and "structure" in payload else payload
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload, dict) else {}
    require_precursor_path_targets = bool(metadata.get("require_precursor_path_targets", True))
    context = _preprocessing_context(file_path)
    node_depths = _node_depths(structure)
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
    assigned_peaks = 0
    emitted_peaks = 0
    total_peaks = int(structure.sample_peak_mz.numel())
    max_samples, max_peaks = 250, 5000
    all_targets = set(int(value) for value in structure.edge_index[1].tolist())
    structural_roots = [index for index in range(int(structure.num_nodes)) if index not in all_targets]
    for sample_id in range(min(int(structure.num_samples), max_samples)):
        start, end = int(structure.sample_peak_ptr[sample_id]), int(structure.sample_peak_ptr[sample_id + 1])
        precursor_nodes = set()
        precursor_mask = structure.precursor_sample_index == sample_id
        for row in structure.precursor_edge_index_path[precursor_mask]:
            valid = row[row >= 0]
            if valid.numel():
                precursor_nodes.add(int(structure.edge_index[1, int(valid[-1])]))
        if not precursor_nodes:
            precursor_nodes.update(structural_roots)
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
                    edge_ids = [int(v) for v in path_edges[p0:p1].tolist()]
                    terminal_node = int(structure.target_terminal_node_index[assignment_id])
                    passes_precursor = not require_precursor_path_targets or (
                        terminal_node in precursor_nodes if not edge_ids
                        else int(structure.edge_index[0, edge_ids[0]]) in precursor_nodes
                    )
                    if not passes_precursor:
                        continue
                    ion_index = int(structure.target_ion_index[assignment_id])
                    unsaturation_index = int(structure.target_unsaturation_index[assignment_id])
                    radical_index = int(structure.target_radical_index[assignment_id])
                    labels = None if context is None else {
                        "ion": str(context.ion_flat_candidates[ion_index, 1]),
                        "unsaturation": str(context.unsaturation_flat_candidates[unsaturation_index, 1]),
                        "radical": str(context.radical_flat_candidates[radical_index, 1]),
                    }
                    assignments.append({"terminalNode": terminal_node, "ion": ion_index, "unsaturation": unsaturation_index, "radical": radical_index, "stateLabels": labels, "stateModified": bool(labels and (labels["unsaturation"] != "[M]" or labels["radical"] != "[M]")), "pathEdges": edge_ids, "pathSteps": [{"edge": edge_id, "source": int(structure.edge_index[0, edge_id]), "target": int(structure.edge_index[1, edge_id])} for edge_id in edge_ids], "expandNodes": [int(v) for v in expand_nodes[x0:x1].tolist()]})
                if assignments:
                    formulas.append({"id": formula_id, "formula": _formula(structure.target_formula[formula_id], elements), "assignments": assignments})
            effective_depth = depth if formulas else -1
            if effective_depth >= 0:
                assigned_peaks += 1
            peaks.append({"index": peak_id - start, "mz": float(structure.sample_peak_mz[peak_id]), "intensity": float(structure.sample_peak_intensity[peak_id]), "depth": effective_depth, "formulas": formulas})
        total_intensity = float(structure.sample_peak_intensity[start:end].sum())
        assigned_local = [start + peak["index"] for peak in peaks if peak["depth"] >= 0]
        assigned_intensity = sum(float(structure.sample_peak_intensity[peak_id]) for peak_id in assigned_local)
        raw_ce = metadata.get("sample_collision_energy_raw", [])
        raw_adduct = metadata.get("sample_adduct_types", [])
        target_path_edges = sorted({edge for peak in peaks for formula in peak["formulas"] for assignment in formula["assignments"] for edge in assignment["pathEdges"]})
        target_terminal_nodes = sorted({assignment["terminalNode"] for peak in peaks for formula in peak["formulas"] for assignment in formula["assignments"]})
        target_terminal_edges = sorted({assignment["pathEdges"][-1] for peak in peaks for formula in peak["formulas"] for assignment in formula["assignments"] if assignment["pathEdges"]})
        active_edge_ids = set(target_path_edges)
        active_node_ids = set(precursor_nodes)
        for edge_id in active_edge_ids:
            active_node_ids.update(int(value) for value in structure.edge_index[:, edge_id].tolist())
        active_graph = {
            "nodes": [{"id": index, "smiles": str(structure.node_smiles[index]), "depth": node_depths[index]} for index in sorted(active_node_ids)],
            "edges": [{"id": edge_id, "source": int(structure.edge_index[0, edge_id]), "target": int(structure.edge_index[1, edge_id]), "targetPath": True} for edge_id in sorted(active_edge_ids)],
            "truncated": False,
        }
        samples.append({"id": sample_id, "adductIndex": int(structure.sample_adduct_type_index[sample_id]), "adduct": raw_adduct[sample_id] if sample_id < len(raw_adduct) else "?", "collisionEnergy": float(structure.sample_ce_value[sample_id]), "collisionEnergyRaw": raw_ce[sample_id] if sample_id < len(raw_ce) else "?", "peakCount": end - start, "assignedIntensity": assigned_intensity, "totalIntensity": total_intensity, "assignmentScore": assigned_intensity / total_intensity if total_intensity > 0 else 0.0, "precursorNodes": sorted(precursor_nodes), "targetPathEdges": target_path_edges, "targetTerminalNodes": target_terminal_nodes, "targetTerminalEdges": target_terminal_edges, "activeGraph": active_graph, "peaks": peaks})
    graph_node_count = min(int(structure.num_nodes), 500)
    graph_edges = [
        {"id": edge_id, "source": int(source), "target": int(target)}
        for edge_id, (source, target) in enumerate(structure.edge_index.detach().cpu().t().tolist())
        if int(source) < graph_node_count and int(target) < graph_node_count
    ]
    highlighted = set(int(value) for value in path_edges.tolist())
    for edge in graph_edges:
        edge["targetPath"] = edge["id"] in highlighted
    depth_counts = {str(depth): node_depths.count(depth) for depth in sorted(set(node_depths)) if depth >= 0}
    graph = {
        "nodes": [{"id": index, "smiles": str(structure.node_smiles[index]), "depth": node_depths[index]} for index in range(graph_node_count)],
        "edges": graph_edges,
        "truncated": graph_node_count < int(structure.num_nodes),
    }
    target_summary = {
        "schemaVersion": metadata.get("target_schema_version", "legacy/unknown"),
        "depthPolicy": metadata.get("target_depth_policy", "unknown"),
        "formulas": int(structure.target_formula.size(0)),
        "assignments": int(structure.target_terminal_node_index.numel()),
        "pathEdges": int(structure.target_path_edge_index.numel()),
        "expandNodes": int(structure.target_expand_node_index.numel()),
        "requirePrecursorPath": require_precursor_path_targets,
    }
    registered_adducts = [] if context is None else [str(value) for _, value in context.main_adduct_types.items()]
    return {"file": str(file_path), "metadata": metadata, "registeredAdducts": registered_adducts, "summary": {"samples": int(structure.num_samples), "nodes": int(structure.num_nodes), "edges": int(structure.num_edges), "peaks": total_peaks, "assignedPeaks": assigned_peaks, "depthCounts": depth_counts, "targets": target_summary}, "graph": graph, "samples": samples, "truncated": int(structure.num_samples) > max_samples or total_peaks > max_peaks}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(json.dumps({"ok": True, "result": inspect_structure(Path(request["path"]).resolve())}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
