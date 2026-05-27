from __future__ import annotations

import gradio as gr

from clefts.gui.workflows.build_fragment_tree_from_smiles.input_page import render_input_page
from clefts.gui.workflows.build_fragment_tree_from_smiles.result_page import render_result_page
from clefts.gui.workflows.build_fragment_tree_from_smiles.runner import build_fragment_tree_and_store_result_url, load_result_from_request


def create_input_app() -> gr.Blocks:
    with gr.Blocks(title="Build Fragment Tree from SMILES") as app:
        with gr.Group(elem_classes="clefts-page"):
            input_page = render_input_page()
            result_url = gr.Textbox(visible=False)
            input_page.run_button.click(
                fn=build_fragment_tree_and_store_result_url,
                inputs=input_page.run_inputs,
                outputs=[result_url],
            ).then(
                fn=lambda url: url,
                inputs=[result_url],
                outputs=[result_url],
                js="""(url) => {
                    if (url) {
                        window.location.href = url;
                    }
                    return url;
                }""",
            )
    return app


def create_result_app() -> gr.Blocks:
    with gr.Blocks(title="Build Fragment Tree from SMILES Results") as app:
        with gr.Group(elem_classes="clefts-page"):
            result_page = render_result_page()
            app.load(
                fn=load_result_from_request,
                outputs=result_page.load_outputs,
            )
    return app


def create_app() -> gr.Blocks:
    return create_input_app()
