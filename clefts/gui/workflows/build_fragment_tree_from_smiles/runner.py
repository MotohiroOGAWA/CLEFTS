from __future__ import annotations

from typing import Any
from uuid import uuid4

import gradio as gr

from clefts.domain.fragment.fragment_tree.FragmentTreeBuilder import FragmentTreeBuilder
from clefts.gui.components.cleavage_patterns_input_panel import json_to_pattern_set
from clefts.libs.mmkit.mmkit import Compound


MAX_PREVIEW_ROWS = 80
_RESULT_STORE: dict[str, str] = {}


def save_result(result: str) -> str:
    result_id = uuid4().hex
    _RESULT_STORE[result_id] = result
    return result_id


def get_result(result_id: str | None) -> str | None:
    if not result_id:
        return None
    return _RESULT_STORE.get(str(result_id))


def build_fragment_tree_and_store_result_url(*args: Any) -> str:
    result = build_fragment_tree_result(*args)
    result_id = save_result(result)
    return f"/build_fragment_tree_from_smiles/result/?result_id={result_id}"


def build_fragment_tree_and_store_result(*args: Any) -> str:
    result_url = build_fragment_tree_and_store_result_url(*args)
    return (
        '<p><a class="clefts-back-link" '
        f'href="{result_url}">Open result page</a></p>'
    )


def load_result_from_request(request: gr.Request | None = None) -> str:
    query_params = getattr(request, "query_params", {}) or {}
    result_id = query_params.get("result_id") if hasattr(query_params, "get") else None
    result = get_result(result_id)
    if result is None:
        return _error_result("Result was not found. Please return to the input page and run the workflow again.")
    return result



def build_fragment_tree_result(
    smiles: str,
    max_depth: int | float,
    only_add_min_depth: bool,
    min_depth_only_from: int | float,
    max_node: int | float,
    max_edge: int | float,
    cleavage_patterns_json: str,
) -> str:
    try:
        if not str(smiles or "").strip():
            return _error_result("SMILES is required.")

        compound = Compound.from_smiles(str(smiles).strip())
        cleavage_pattern_set = json_to_pattern_set(cleavage_patterns_json)
        builder = FragmentTreeBuilder(
            max_depth=int(max_depth),
            cleavage_pattern_set=cleavage_pattern_set,
            only_add_min_depth=bool(only_add_min_depth),
            min_depth_only_from=int(min_depth_only_from),
        )
        tree = builder.build(
            compound=compound,
            max_node=int(max_node),
            max_edge=int(max_edge),
        )
        
        return summarize_fragment_tree(
            tree=tree,
            max_depth=int(max_depth),
            only_add_min_depth=bool(only_add_min_depth),
            min_depth_only_from=int(min_depth_only_from),
            max_node=int(max_node),
            max_edge=int(max_edge),
            cleavage_pattern_count=len(cleavage_pattern_set.patterns),
        )
    except Exception as exc:
        return _error_result(str(exc))


def demo_result(smiles: str, *builder_values_and_patterns: Any) -> str:
    if not builder_values_and_patterns:
        from clefts.gui.workflows.build_fragment_tree_from_smiles.metadata import DEFAULT_CLEAVAGE_PATTERNS_PATH
        from clefts.gui.components.cleavage_patterns_input_panel import load_pattern_set_json_file, pattern_set_to_json

        return build_fragment_tree_result(
            smiles,
            2,
            True,
            0,
            -1,
            -1,
            pattern_set_to_json(load_pattern_set_json_file(DEFAULT_CLEAVAGE_PATTERNS_PATH)),
        )
    return build_fragment_tree_result(smiles, *builder_values_and_patterns)


def summarize_fragment_tree(
    *,
    tree,
    max_depth: int,
    only_add_min_depth: bool,
    min_depth_only_from: int,
    max_node: int,
    max_edge: int,
    cleavage_pattern_count: int,
) -> str:
    lines = [
        "# Fragment Tree Result",
        "",
        "## Summary",
        f"- Root SMILES: `{tree.smiles}`",
        f"- Nodes: {tree.num_nodes}",
        f"- Edges: {tree.num_edges}",
        f"- Cleavage patterns: {cleavage_pattern_count}",
        f"- max_depth: {max_depth}",
        f"- only_add_min_depth: {only_add_min_depth}",
        f"- min_depth_only_from: {min_depth_only_from}",
        f"- max_node: {_format_limit(max_node)}",
        f"- max_edge: {_format_limit(max_edge)}",
        "",
        "## Nodes",
        "| ID | Depth | SMILES |",
        "|---:|---:|---|",
    ]

    depths = tree.node_depths
    for node_id, node_smiles in enumerate(tree.node_smiles[:MAX_PREVIEW_ROWS]):
        lines.append(f"| {node_id} | {int(depths[node_id])} | `{node_smiles}` |")
    if tree.num_nodes > MAX_PREVIEW_ROWS:
        lines.append(f"| ... | ... | {tree.num_nodes - MAX_PREVIEW_ROWS} more nodes |")

    lines.extend([
        "",
        "## Edges",
        "| ID | Source | Target | Events |",
        "|---:|---:|---:|---:|",
    ])
    edge_count = min(tree.num_edges, MAX_PREVIEW_ROWS)
    for edge_id in range(edge_count):
        edge = tree.get_edge(edge_id)
        lines.append(
            f"| {edge.id} | {edge.source_id} | {edge.target_id} | {len(edge.events)} |"
        )
    if tree.num_edges > MAX_PREVIEW_ROWS:
        lines.append(f"| ... | ... | ... | {tree.num_edges - MAX_PREVIEW_ROWS} more edges |")

    return "\n".join(lines)


def _format_limit(value: int) -> str:
    return "unlimited" if int(value) < 0 else str(int(value))


def _error_result(message: str) -> str:
    return "\n".join([
        "# Fragment Tree Result",
        "",
        "## Error",
        message,
    ])
