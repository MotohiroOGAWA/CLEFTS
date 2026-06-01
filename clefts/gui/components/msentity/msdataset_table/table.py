from __future__ import annotations

from typing import Tuple
import gradio as gr

from .....libs.msentity.msentity import MSDataset

MSDATASET_TABLE_STYLE = """
.msdataset-table-container {
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    overflow-y: auto;
}

.msdataset-table {
    width: max-content;
    min-width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

/* Header */
.msdataset-table thead th {
    background-color: #4f81bd;
    color: white;
    font-weight: 600;
    text-align: left;
    padding: 8px 12px;
    border: 1px solid #d0d7de;
    position: sticky;
    top: 0;
    z-index: 1;
}

/* Cells */
.msdataset-table td {
    padding: 6px 12px;
    border: 1px solid #e5e7eb;
    white-space: nowrap;
}

/* Zebra stripes */
.msdataset-table tbody tr:nth-child(odd) {
    background-color: #ffffff;
}

.msdataset-table tbody tr:nth-child(even) {
    background-color: #f3f4f6;
}

/* Hover */
.msdataset-table tbody tr:hover {
    background-color: #dbeafe;
}
"""

def render_dataset_html(
    dataset: MSDataset | None,
    page: int,
    rows_per_page: int,
    height: int | None = None,
    show_index: bool = True,
) -> str:
    if dataset is None:
        return ""

    start = (page - 1) * rows_per_page
    end = start + rows_per_page

    page_dataset = dataset[start:end]
    metadata = page_dataset.metadata.copy()

    metadata.insert(
        0,
        "Spectrum",
        [
            (
                f'<button '
                f'class="msdataset-spectrum-button" '
                f'data-spectrum-index="{start + i}">'
                f'View'
                f'</button>'
            )
            for i in range(len(metadata))
        ],
    )

    if show_index:
        metadata.index = range(start, start + len(metadata))

    table_html = metadata.to_html(
        index=show_index,
        escape=False,
        classes="msdataset-table",
    )

    height_style = "" if height is None else f"height:{height}px;"

    return f"""
    <div
        class="msdataset-table-container"
        style="
            {height_style}
            overflow-x: auto;
            overflow-y: auto;
            max-width: 100%;
        "
    >
        {table_html}
    </div>
    """

def render_table(
    dataset: MSDataset | None = None,
    *,
    page: int = 1,
    rows_per_page: int = 20,
    height: int | None = None,
    show_index: bool = False,
) -> Tuple[
    gr.State,
    gr.State,
    gr.HTML,
]:
    source_dataset_state = gr.State(dataset)
    dataset_state = gr.State(dataset)

    dataframe = gr.HTML(
        value=render_dataset_html(
            dataset,
            page=page,
            rows_per_page=rows_per_page,
            height=height,
            show_index=show_index,
        )
    )

    return (
        source_dataset_state,
        dataset_state,
        dataframe,
    )