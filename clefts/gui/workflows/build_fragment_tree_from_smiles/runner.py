from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import gradio as gr
import pandas as pd

from clefts.domain.fragment.fragment_tree.FragmentTreeBuilder import FragmentTreeBuilder
from clefts.gui.components.cleavage_patterns_input_panel import json_to_pattern_set
from clefts.gui.components.paginated_dataframe import make_page_html
from clefts.libs.mmkit.mmkit import Compound


NODE_HEADERS = ["ID", "Depth", "SMILES"]
EDGE_HEADERS = ["ID", "Source", "Target", "Events"]


@dataclass(frozen=True)
class FragmentTreeResult:
    summary_markdown: str
    node_dataframe: pd.DataFrame
    edge_dataframe: pd.DataFrame


_RESULT_STORE: dict[str, FragmentTreeResult] = {}


def save_result(result: FragmentTreeResult) -> str:
    result_id = uuid4().hex
    _RESULT_STORE[result_id] = result
    return result_id


def get_result(result_id: str | None) -> FragmentTreeResult | None:
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


def load_result_from_request(request: gr.Request | None = None):
    query_params = getattr(request, "query_params", {}) or {}
    result_id = query_params.get("result_id") if hasattr(query_params, "get") else None
    result = get_result(result_id)
    if result is None:
        result_id = ""
        result = _error_result("Result was not found. Please return to the input page and run the workflow again.")
    return (str(result_id or ""), result.summary_markdown, *_node_page_updates(result, 1, 20), *_edge_page_updates(result, 1, 20))


def build_fragment_tree_result(
    smiles: str,
    max_depth: int | float,
    only_add_min_depth: bool,
    min_depth_only_from: int | float,
    max_node: int | float,
    max_edge: int | float,
    cleavage_patterns_json: str,
) -> FragmentTreeResult:
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


def demo_result(smiles: str, *builder_values_and_patterns: Any) -> FragmentTreeResult:
    if not builder_values_and_patterns:
        from clefts.gui.workflows.build_fragment_tree_from_smiles.metadata import DEFAULT_CLEAVAGE_PATTERNS_PATH

        return build_fragment_tree_result(
            smiles,
            2,
            True,
            0,
            -1,
            -1,
            DEFAULT_CLEAVAGE_PATTERNS_PATH.read_text(encoding="utf-8"),
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
) -> FragmentTreeResult:
    summary_markdown = "\n".join([
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
    ])

    depths = tree.node_depths
    node_dataframe = pd.DataFrame(
        {
            "ID": range(tree.num_nodes),
            "Depth": [int(depths[node_id]) for node_id in range(tree.num_nodes)],
            "SMILES": tree.node_smiles.tolist(),
        },
        columns=NODE_HEADERS,
    )
    edge_rows = []
    for edge_id in range(tree.num_edges):
        edge = tree.get_edge(edge_id)
        edge_rows.append(
            {
                "ID": edge.id,
                "Source": edge.source_id,
                "Target": edge.target_id,
                "Events": len(edge.events),
            }
        )
    edge_dataframe = pd.DataFrame(edge_rows, columns=EDGE_HEADERS)

    return FragmentTreeResult(
        summary_markdown=summary_markdown,
        node_dataframe=node_dataframe,
        edge_dataframe=edge_dataframe,
    )


def move_node_page(result_id: str, page: str, rows_per_page: int, delta: int):
    result = get_result(result_id)
    if result is None:
        return _empty_page(NODE_HEADERS)
    return _node_page_updates(result, _int_or_default(page, 1) + delta, rows_per_page)


def set_node_page(result_id: str, page: str, rows_per_page: int):
    result = get_result(result_id)
    if result is None:
        return _empty_page(NODE_HEADERS)
    return _node_page_updates(result, page, rows_per_page)


def move_edge_page(result_id: str, page: str, rows_per_page: int, delta: int):
    result = get_result(result_id)
    if result is None:
        return _empty_page(EDGE_HEADERS)
    return _edge_page_updates(result, _int_or_default(page, 1) + delta, rows_per_page)


def set_edge_page(result_id: str, page: str, rows_per_page: int):
    result = get_result(result_id)
    if result is None:
        return _empty_page(EDGE_HEADERS)
    return _edge_page_updates(result, page, rows_per_page)


def _node_page_updates(result: FragmentTreeResult, page: Any, rows_per_page: Any):
    return make_page_html(
        result.node_dataframe,
        page=page,
        rows_per_page=rows_per_page,
    )


def _edge_page_updates(result: FragmentTreeResult, page: Any, rows_per_page: Any):
    return make_page_html(
        result.edge_dataframe,
        page=page,
        rows_per_page=rows_per_page,
    )


def _empty_page(headers: list[str]):
    return make_page_html(pd.DataFrame(columns=headers), page=1, rows_per_page=20)


def _format_limit(value: int) -> str:
    return "unlimited" if int(value) < 0 else str(int(value))


def _error_result(message: str) -> FragmentTreeResult:
    return FragmentTreeResult(
        summary_markdown="\n".join(["## Error", message]),
        node_dataframe=pd.DataFrame(columns=NODE_HEADERS),
        edge_dataframe=pd.DataFrame(columns=EDGE_HEADERS),
    )


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
