from __future__ import annotations

import gradio as gr

from clefts.gui.workflows.build_fragment_tree_from_smiles.navigation import render_result_nav_html


def render_result_page() -> gr.Code:
    gr.HTML(render_result_nav_html())
    gr.HTML(
        """
        <section class="clefts-page-heading">
            <h1>Results</h1>
            <p>Review the output generated from the submitted workflow input.</p>
        </section>
        """
    )
    return gr.Code(label="Summary", language="markdown")
