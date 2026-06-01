from __future__ import annotations

import html
from typing import Any, Sequence

import gradio as gr

from .models import MSDatasetTable
from .table import render_table, render_dataset_html, MSDATASET_TABLE_STYLE
from .paging import render_paging, render_page_count_html, update_paging, clamp_page, go_next_page, go_previous_page, MSDATASET_PAGING_STYLE
from .download import render_download, export_dataset_to_hdf5_base64, MSDATASET_HDF5_DOWNLOAD_JS

from .....libs.msentity.msentity import MSDataset

MSDATASET_STYLE = "\n\n".join([
    MSDATASET_TABLE_STYLE,
    MSDATASET_PAGING_STYLE,
])

def set_msdataset_table_dataset(dataset: MSDataset | None, height: int | None = None, show_index: bool = True):
    return (
        dataset,  # table.source_dataset_state
        dataset,  # table.dataset_state
        render_dataset_html(dataset, page=1, rows_per_page=20, height=height, show_index=show_index), # table.dataframe
        1,        # table.page_number
        render_page_count_html(dataset, rows_per_page=20), # table.page_count
    )

def render_msdataset_table(
    dataset: MSDataset | None = None,
    *,
    rows_per_page: int = 20,
    rows_per_page_choices: Sequence[int] = (10, 20, 50, 100),
    height: int | None = None,
    show_index: bool = True,
) -> MSDatasetTable:
    rows_per_page_choices = sorted(
        {
            int(choice)
            for choice in rows_per_page_choices
        }
        | {int(rows_per_page)}
    )

    download_button, exported_hdf5_base64, error_message = render_download()

    source_dataset_state, dataset_state, dataframe = render_table(
        dataset=dataset,
        page=1,
        rows_per_page=rows_per_page,
        height=height,
        show_index=show_index,
    )

    previous_button, page_number, page_count, next_button, rows_per_page_dropdown = render_paging()

    dataset_state.change(
        fn=update_paging,
        inputs=[
            dataset_state,
            rows_per_page_dropdown,
        ],
        outputs=[
            page_number,
            page_count,
        ],
    )

    previous_button.click(
        fn=go_previous_page,
        inputs=[
            page_number,
            dataset_state,
            rows_per_page_dropdown,
        ],
        outputs=[
            page_number,
        ],
    ).then(
        fn=render_dataset_html,
        inputs=[
            dataset_state,
            page_number,
            rows_per_page_dropdown,
        ],
        outputs=[
            dataframe,
        ],
    )

    next_button.click(
        fn=go_next_page,
        inputs=[
            page_number,
            dataset_state,
            rows_per_page_dropdown,
        ],
        outputs=[
            page_number,
        ],
    ).then(
        fn=render_dataset_html,
        inputs=[
            dataset_state,
            page_number,
            rows_per_page_dropdown,
        ],
        outputs=[
            dataframe,
        ],
    )

    page_number.submit(
        fn=clamp_page,
        inputs=[
            page_number,
            dataset_state,
            rows_per_page_dropdown,
        ],
        outputs=[
            page_number,
        ],
    ).then(
        fn=render_dataset_html,
        inputs=[
            dataset_state,
            page_number,
            rows_per_page_dropdown,
        ],
        outputs=[
            dataframe,
        ],
    )

    rows_per_page_dropdown.change(
        fn=update_paging,
        inputs=[
            dataset_state,
            rows_per_page_dropdown,
        ],
        outputs=[
            page_number,
            page_count,
        ],
    ).then(
        fn=render_dataset_html,
        inputs=[
            dataset_state,
            page_number,
            rows_per_page_dropdown,
        ],
        outputs=[
            dataframe,
        ],
    )

    download_button.click(
        fn=export_dataset_to_hdf5_base64,
        inputs=[
            dataset_state,
        ],
        outputs=[
            exported_hdf5_base64,
        ],
        show_progress="hidden",
    ).then(
        fn=lambda value: "",
        inputs=[
            exported_hdf5_base64,
        ],
        outputs=[
            error_message,
        ],
        js=MSDATASET_HDF5_DOWNLOAD_JS,
        show_progress="hidden",
    )

    return MSDatasetTable(
        source_dataset_state=source_dataset_state,
        dataset_state=dataset_state,
        dataframe=dataframe,
        previous_button=previous_button,
        page_number=page_number,
        page_count=page_count,
        next_button=next_button,
        rows_per_page=rows_per_page_dropdown,
        download_button=download_button,
    )