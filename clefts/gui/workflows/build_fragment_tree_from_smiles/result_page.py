from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gradio as gr
import pandas as pd

from ...components.common.paginated_dataframe import PaginatedDataframe, render_paginated_dataframe
from .navigation import render_result_nav_html
from .runner import (
    EDGE_HEADERS,
    NODE_HEADERS,
    FragmentTreeResult,
    get_result,
)


@dataclass(frozen=True)
class BuildFragmentTreeResultPage:
    result_id: gr.Textbox
    summary: gr.Markdown
    nodes: PaginatedDataframe
    edges: PaginatedDataframe
    load_result_from_request: Callable

    @property
    def load_outputs(self) -> list[gr.Component]:
        return [self.result_id, self.summary, *self.nodes.data_outputs, *self.edges.data_outputs]


def render_result_page() -> BuildFragmentTreeResultPage:
    gr.HTML(render_result_nav_html())
    gr.HTML(
        """
        <section class="clefts-page-heading">
            <h1>Results</h1>
            <p>Review the output generated from the submitted workflow input.</p>
        </section>
        """
    )
    empty_nodes_df = pd.DataFrame(
        columns=NODE_HEADERS,
    )

    empty_edges_df = pd.DataFrame(
        columns=EDGE_HEADERS,
    )

    result_id = gr.Textbox(visible=False)
    summary = gr.Markdown(label="Summary")

    with gr.Tabs():
        with gr.Tab("Nodes"):
            nodes = render_paginated_dataframe(
                empty_nodes_df,
                headers=NODE_HEADERS,
                rows_per_page=20,
                height=None,
            )

        with gr.Tab("Edges"):
            edges = render_paginated_dataframe(
                empty_edges_df,
                headers=EDGE_HEADERS,
                rows_per_page=20,
                height=None,
            )


    def load_result_from_request(request: gr.Request | None = None):
        query_params = getattr(request, "query_params", {}) or {}
        loaded_result_id = query_params.get("result_id") if hasattr(query_params, "get") else None
        result = get_result(loaded_result_id)
        if result is None:
            loaded_result_id = ""
            result = FragmentTreeResult(
                summary_markdown="\n".join([
                    "## Error",
                    "Result was not found. Please return to the input page and run the workflow again.",
                ]),
                node_dataframe=pd.DataFrame(columns=NODE_HEADERS),
                edge_dataframe=pd.DataFrame(columns=EDGE_HEADERS),
            )

        return (
            str(loaded_result_id or ""),
            result.summary_markdown,
            *nodes.build_outputs(result.node_dataframe),
            *edges.build_outputs(result.edge_dataframe),
        )

    return BuildFragmentTreeResultPage(
        result_id=result_id,
        summary=summary,
        nodes=nodes,
        edges=edges,
        load_result_from_request=load_result_from_request,
    )
