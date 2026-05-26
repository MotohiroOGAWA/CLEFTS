from __future__ import annotations

import gradio as gr

from clefts.gui.workflows.build_fragment_tree_from_smiles.page import render_workflow_item_html as render_build_fragment_tree_workflow_item_html


def create_app() -> gr.Blocks:
    with gr.Blocks(title="CLEFTS") as app:
        with gr.Group(elem_classes="clefts-page clefts-home"):
            gr.HTML(
                """
                <section class="clefts-hero">
                    <div class="clefts-title-block">
                        <div class="clefts-kicker">CLEFTS Workflow Portal</div>
                        <h1>CLEFTS</h1>
                        <p>
                            CLEFTS provides tools for cleavage-aware molecular fragmentation analysis,
                            helping you build fragment trees and inspect fragmentation workflows from
                            molecular inputs.
                        </p>
                    </div>
                </section>
                """
            )
            gr.HTML(
                """
                <section class="clefts-section">
                    <h2>Available Workflows</h2>
                    <div class="clefts-rule"></div>
                </section>
                """
            )
            gr.HTML(render_build_fragment_tree_workflow_item_html())
    return app
