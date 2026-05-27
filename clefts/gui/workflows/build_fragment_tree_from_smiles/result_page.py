from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from clefts.gui.components.paginated_dataframe import PaginatedDataframe, render_paginated_dataframe
from clefts.gui.workflows.build_fragment_tree_from_smiles.navigation import render_result_nav_html
from clefts.gui.workflows.build_fragment_tree_from_smiles.runner import (
    move_edge_page,
    move_node_page,
    set_edge_page,
    set_node_page,
)


@dataclass(frozen=True)
class BuildFragmentTreeResultPage:
    result_id: gr.Textbox
    summary: gr.Markdown
    nodes: PaginatedDataframe
    edges: PaginatedDataframe

    @property
    def load_outputs(self) -> list[gr.Component]:
        return [self.result_id, self.summary, *self.nodes.page_outputs, *self.edges.page_outputs]


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
    result_id = gr.Textbox(visible=False)
    summary = gr.Markdown(label="Summary")
    with gr.Tabs():
        with gr.Tab("Nodes"):
            nodes = render_paginated_dataframe(
                [],
                headers=["ID", "Depth", "SMILES"],
                rows_per_page=20,
                datatype=["number", "number", "str"],
                interactive=False,
                show_row_numbers=False,
                max_height=None,
                elem_classes="clefts-result-dataframe",
                bind_events=False,
                show_sort_controls=False,
            )
        with gr.Tab("Edges"):
            edges = render_paginated_dataframe(
                [],
                headers=["ID", "Source", "Target", "Events"],
                rows_per_page=20,
                datatype=["number", "number", "number", "number"],
                interactive=False,
                show_row_numbers=False,
                max_height=None,
                elem_classes="clefts-result-dataframe",
                bind_events=False,
                show_sort_controls=False,
            )

    nodes.previous_button.click(
        fn=lambda stored_result_id, page, per_page: move_node_page(stored_result_id, page, per_page, -1),
        inputs=[result_id, nodes.page_number, nodes.rows_per_page],
        outputs=nodes.page_outputs,
        show_progress="hidden",
    )
    nodes.next_button.click(
        fn=lambda stored_result_id, page, per_page: move_node_page(stored_result_id, page, per_page, 1),
        inputs=[result_id, nodes.page_number, nodes.rows_per_page],
        outputs=nodes.page_outputs,
        show_progress="hidden",
    )
    nodes.page_number.submit(
        fn=set_node_page,
        inputs=[result_id, nodes.page_number, nodes.rows_per_page],
        outputs=nodes.page_outputs,
        show_progress="hidden",
    )
    nodes.rows_per_page.change(
        fn=set_node_page,
        inputs=[result_id, nodes.page_number, nodes.rows_per_page],
        outputs=nodes.page_outputs,
        show_progress="hidden",
    )

    edges.previous_button.click(
        fn=lambda stored_result_id, page, per_page: move_edge_page(stored_result_id, page, per_page, -1),
        inputs=[result_id, edges.page_number, edges.rows_per_page],
        outputs=edges.page_outputs,
        show_progress="hidden",
    )
    edges.next_button.click(
        fn=lambda stored_result_id, page, per_page: move_edge_page(stored_result_id, page, per_page, 1),
        inputs=[result_id, edges.page_number, edges.rows_per_page],
        outputs=edges.page_outputs,
        show_progress="hidden",
    )
    edges.page_number.submit(
        fn=set_edge_page,
        inputs=[result_id, edges.page_number, edges.rows_per_page],
        outputs=edges.page_outputs,
        show_progress="hidden",
    )
    edges.rows_per_page.change(
        fn=set_edge_page,
        inputs=[result_id, edges.page_number, edges.rows_per_page],
        outputs=edges.page_outputs,
        show_progress="hidden",
    )

    return BuildFragmentTreeResultPage(result_id=result_id, summary=summary, nodes=nodes, edges=edges)
