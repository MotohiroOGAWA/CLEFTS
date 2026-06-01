from __future__ import annotations

import gradio as gr

from .workflow_item import render_workflow_item_html
from .metadata import WORKFLOW_BASE_PATH, WORKFLOW_TITLE
from .navigation import render_back_to_main_html

from ...components.msentity.msdataset_upload_panel import render_msdataset_upload_panel
from ...components.msentity.msdataset_table import render_msdataset_table, set_msdataset_table_dataset

def render_page() -> gr.Blocks:
    with gr.Blocks(title=WORKFLOW_TITLE) as page:
        gr.HTML(render_back_to_main_html())

        gr.Markdown(
            """
            # msentity GUI

            msentity-based tools are provided as one of the CLEFTS GUI features.
            Select a tool below.
            """
        )
        
        msdataset_upload_panel = render_msdataset_upload_panel()

        msdataset_table = render_msdataset_table()

        msdataset_upload_panel.modal.load_button.click(
            fn=set_msdataset_table_dataset,
            inputs=[
                msdataset_upload_panel.dataset_state,
            ],
            outputs=[
                msdataset_table.source_dataset_state,
                msdataset_table.dataset_state,
                msdataset_table.dataframe,
                msdataset_table.page_number,
                msdataset_table.page_count,
            ],
        )
                

    return page

def create_app() -> gr.Blocks:
    return render_page()


def get_routes() -> dict[str, gr.Blocks]:
    return {
        WORKFLOW_BASE_PATH: create_app(),
    }