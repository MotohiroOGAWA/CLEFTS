"""Backend for inspecting generated schema-v5 action training structures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def _teacher_states(structure) -> list[dict]:
    """One row per (sample, normalized action-set) teacher state, with its
    valid next actions/EOS and, if materialized, the resulting fragment SMILES."""
    state_action_ptr = structure.teacher_state_action_ptr.tolist()
    state_action_index = structure.teacher_state_action_index.tolist()
    positive_ptr = structure.teacher_positive_action_ptr.tolist()
    positive_index = structure.teacher_positive_action_index.tolist()
    eos = structure.teacher_positive_eos.tolist()
    sample_index = structure.teacher_state_sample_index.tolist()
    fragment_node_index = structure.state_fragment_node_index.tolist()
    compounds = structure.downstream.decoded.compounds if structure.downstream is not None else None

    rows = []
    for row in range(len(sample_index)):
        actions = state_action_index[state_action_ptr[row]:state_action_ptr[row + 1]]
        next_actions = positive_index[positive_ptr[row]:positive_ptr[row + 1]]
        node = fragment_node_index[row]
        rows.append({
            "sample": sample_index[row],
            "actions": actions,
            "eos": bool(eos[row]),
            "fragmentSmiles": str(compounds[node].smiles) if compounds is not None and node >= 0 else None,
            "positiveNextActions": next_actions,
        })
    return rows


def _transitions(structure) -> list[dict]:
    parents = structure.transition_parent_state_index.tolist()
    children = structure.transition_child_state_index.tolist()
    added = structure.transition_added_action_index.tolist()
    return [
        {"parent": parent, "child": child, "addedAction": action}
        for parent, child, action in zip(parents, children, added)
    ]


def _precursor_rows(structure) -> list[list[list[int]]]:
    """Per sample, every alternative precursor action set (empty = Source
    itself is the precursor for that sample)."""
    sample_ptr = structure.sample_precursor_row_ptr.tolist()
    row_action_ptr = structure.precursor_row_action_ptr.tolist()
    row_action_index = structure.precursor_row_action_index.tolist()
    rows = []
    for sample in range(len(sample_ptr) - 1):
        alternatives = []
        for row in range(sample_ptr[sample], sample_ptr[sample + 1]):
            alternatives.append(row_action_index[row_action_ptr[row]:row_action_ptr[row + 1]])
        rows.append(alternatives)
    return rows


def inspect_structure(file_path: Path) -> dict:
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "structure" not in payload:
        raise TypeError("This .preft.pt file has no saved structure.")
    structure = payload["structure"]
    metadata = dict(payload.get("metadata") or {})
    required = ("teacher_state_action_ptr", "sample_precursor_row_ptr", "action_type")
    if not all(hasattr(structure, name) for name in required):
        raise TypeError("This .preft.pt file is not a schema-v5 Source/action training structure.")

    source_smiles = str(metadata.get("smiles", ""))
    conditions = structure.condition_features.tolist()
    states = _teacher_states(structure)
    transitions = _transitions(structure)
    precursor_rows = _precursor_rows(structure)

    states_by_sample: dict[int, list[dict]] = {}
    for local_id, state in enumerate(states):
        state = {**state, "id": local_id}
        states_by_sample.setdefault(state["sample"], []).append(state)
    transitions_by_sample: dict[int, list[dict]] = {}
    for transition in transitions:
        sample = states[transition["parent"]]["sample"] if transition["parent"] < len(states) else None
        if sample is not None:
            transitions_by_sample.setdefault(sample, []).append(transition)

    num_samples = int(structure.num_samples)
    samples = []
    for sample_id in range(num_samples):
        adduct_index, collision_energy = (conditions[sample_id] if sample_id < len(conditions) else (None, None))
        samples.append({
            "id": sample_id,
            "adductIndex": adduct_index,
            "collisionEnergy": collision_energy,
            "precursorAlternatives": precursor_rows[sample_id] if sample_id < len(precursor_rows) else [],
            "states": states_by_sample.get(sample_id, []),
            "transitions": transitions_by_sample.get(sample_id, []),
        })

    return {
        "file": str(file_path),
        "metadata": metadata,
        "sourceSmiles": source_smiles,
        "summary": {
            "samples": num_samples,
            "primitiveActions": int(structure.action_type.shape[0]),
            "teacherStates": len(states),
            "transitions": len(transitions),
        },
        "samples": samples,
    }


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(json.dumps({"ok": True, "result": inspect_structure(Path(request["path"]).resolve())}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
