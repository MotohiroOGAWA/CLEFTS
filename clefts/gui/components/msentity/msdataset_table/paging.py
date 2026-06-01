from __future__ import annotations

from typing import Tuple, Sequence
import math
import gradio as gr

from .....libs.msentity.msentity import MSDataset

MSDATASET_PAGING_STYLE = """
.msdataset-paging-row {
    align-items: center;
    gap: 6px;
}

.msdataset-paging-left {
    flex-grow: 0 !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 6px;
}

.msdataset-page-button {
    width: 28px !important;
    min-width: 28px !important;
    height: 28px !important;
    padding: 0 !important;
}

.msdataset-page-number input::-webkit-outer-spin-button,
.msdataset-page-number input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

.msdataset-page-number input[type="number"] {
    -moz-appearance: textfield;
}

.msdataset-page-count {
    width: 28px !important;
    min-width: 28px !important;
    padding: 0 !important;
}

.msdataset-rows-per-page {
    width: 72px !important;
    min-width: 72px !important;
}

/* Dropdown の外枠を弱くする / 消す */
.msdataset-rows-per-page input,
.msdataset-rows-per-page .wrap,
.msdataset-rows-per-page .container {
    border: none !important;
    box-shadow: none !important;
}
"""
def calculate_page_count(
    dataset: MSDataset | None,
    *,
    rows_per_page: int,
) -> int:
    if dataset is None:
        return 0

    rows_per_page = max(1, int(rows_per_page))

    return max(
        1,
        math.ceil(len(dataset) / rows_per_page),
    )


def clamp_page(
    page: int | float | None,
    dataset: MSDataset | None,
    rows_per_page: int,
) -> int:
    page_count = calculate_page_count(
        dataset,
        rows_per_page=rows_per_page,
    )

    if page_count <= 0:
        return 1

    try:
        page_int = int(page or 1)
    except (TypeError, ValueError):
        page_int = 1

    return min(
        max(1, page_int),
        page_count,
    )

def page_number_min_width(page_count: int) -> int:
    """
    Calculate a suitable width for page_number based on the
    number of digits in page_count.
    """
    digits = max(1, len(str(page_count)))

    return max(
        60,
        digits * 14 + 20,
    )

def update_paging(
    dataset: MSDataset | None,
    rows_per_page: int,
) -> tuple[
    gr.Number,
    gr.HTML,
]:
    page_count = calculate_page_count(
        dataset,
        rows_per_page=rows_per_page,
    )

    return (
        gr.Number(
            value=1,
            precision=0,
            min_width=page_number_min_width(page_count),
        ),
        render_page_count_html(
            dataset,
            rows_per_page=rows_per_page,
        ),
    )

def update_page_number(
    page: int | float | None,
    dataset: MSDataset | None,
    rows_per_page: int,
) -> gr.Number:
    page_count = calculate_page_count(
        dataset,
        rows_per_page=rows_per_page,
    )

    page = clamp_page(
        page,
        dataset,
        rows_per_page=rows_per_page,
    )

    return gr.Number(
        value=page,
        precision=0,
        min_width=page_number_min_width(page_count),
    )


def go_previous_page(
    page: int | float | None,
    dataset: MSDataset | None,
    rows_per_page: int,
) -> gr.Number:
    try:
        page = int(page or 1)
    except (TypeError, ValueError):
        page = 1

    return update_page_number(
        page - 1,
        dataset,
        rows_per_page,
    )


def go_next_page(
    page: int | float | None,
    dataset: MSDataset | None,
    rows_per_page: int,
) -> gr.Number:
    try:
        page = int(page or 1)
    except (TypeError, ValueError):
        page = 1

    return update_page_number(
        page + 1,
        dataset,
        rows_per_page,
    )

def render_page_count_html(
    dataset: MSDataset | None,
    *,
    rows_per_page: int,
) -> str:
    page_count = calculate_page_count(
        dataset,
        rows_per_page=rows_per_page,
    )

    return f"/ {page_count}"


def render_paging(
    rows_per_page: int = 20,
    rows_per_page_choices: Sequence[int] = (10, 20, 50, 100),
) -> Tuple[
    gr.Button,
    gr.Number,
    gr.HTML,
    gr.Button,
]:
    with gr.Row(elem_classes="msdataset-paging-row"):
        with gr.Row(
            scale=0,
            elem_classes="msdataset-paging-left",
        ):
            previous_button = gr.Button(
                value="◀",
                size="sm",
                scale=0,
                min_width=28,
                elem_classes="msdataset-page-button",
            )

            page_number = gr.Number(
                value=1,
                precision=0,
                minimum=None,
                maximum=None,
                scale=0,
                min_width=64,
                container=False,
                interactive=True,
                elem_classes="msdataset-page-number",
            )

            page_count = gr.HTML(
                value="/ 1",
                scale=0,
                min_width=36,
                elem_classes="msdataset-page-count",
            )

            next_button = gr.Button(
                value="▶",
                size="sm",
                scale=0,
                min_width=28,
                elem_classes="msdataset-page-button",
            )

        gr.HTML("", scale=1)

        rows_per_page_dropdown = gr.Dropdown(
            choices=rows_per_page_choices,
            value=rows_per_page,
            interactive=True,
            scale=0,
            min_width=72,
            show_label=False,
            container=False,
            elem_classes="msdataset-rows-per-page",
        )

    return (
        previous_button,
        page_number,
        page_count,
        next_button,
        rows_per_page_dropdown,
    )