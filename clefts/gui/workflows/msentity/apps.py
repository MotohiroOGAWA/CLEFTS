from __future__ import annotations

import gradio as gr

from .metadata import WORKFLOW_TITLE
from .page import render_page


def create_app() -> gr.Blocks:
    with gr.Blocks(title=WORKFLOW_TITLE) as app:
        with gr.Group(elem_classes="clefts-page"):
            page = render_page()
    return app
