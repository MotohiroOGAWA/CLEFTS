from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Sequence

import gradio as gr
import pandas as pd

from .styles import PAGINATED_DATAFRAME_CSS


@dataclass(frozen=True)
class PaginatedDataframe:
    source_dataframe: gr.State
    sorted_dataframe: gr.State
    dataframe: gr.HTML

    previous_button: gr.Button
    page_number: gr.Textbox
    page_count: gr.HTML
    next_button: gr.Button

    rows_per_page: gr.Dropdown
    sort_column: gr.Dropdown
    sort_order: gr.Dropdown


def render_paginated_dataframe(
    dataframe: Any,
    *,
    headers: Sequence[str] | None = None,
    rows_per_page: int = 10,
    rows_per_page_choices: Sequence[int] = (10, 20, 50, 100),
    label: str | None = None,
    height: int | None = None,
    show_index: bool = False,
    show_sort_controls: bool = True,
) -> PaginatedDataframe:
    source_df = to_dataframe(dataframe, headers=headers)
    sorted_df = source_df.copy()

    rows_per_page = _coerce_choice(rows_per_page, rows_per_page_choices)

    page_html, initial_page, initial_page_count = make_page_html(
        sorted_df,
        page=1,
        rows_per_page=rows_per_page,
        height=height,
        show_index=show_index,
    )

    dataframe_headers = list(source_df.columns)
    page_number_width = page_number_min_width(initial_page_count)

    source_state = gr.State(source_df)
    sorted_state = gr.State(sorted_df)

    with gr.Column():
        if label:
            gr.HTML(f'<div class="clefts-table-label">{html.escape(label)}</div>')

        html_table = gr.HTML(
            value=page_html,
            elem_classes="clefts-html-table-wrap",
        )

        with gr.Row(elem_classes="clefts-pager-row"):
            previous_button = gr.Button(
                "‹",
                elem_classes="clefts-page-button",
                min_width=22,
                scale=0,
            )

            page_number = gr.Textbox(
                value=initial_page,
                show_label=False,
                container=False,
                elem_classes="clefts-page-number",
                min_width=page_number_width,
                scale=0,
            )

            page_count = gr.HTML(
                f'<span class="clefts-page-count">{initial_page_count}</span>'
            )

            next_button = gr.Button(
                "›",
                elem_classes="clefts-page-button",
                min_width=22,
                scale=0,
            )

            # gr.HTML(
            #     '<div style="width:12px;"></div>'
            # )

            rows_per_page_input = gr.Dropdown(
                choices=[int(v) for v in rows_per_page_choices],
                value=rows_per_page,
                show_label=False,
                container=False,
                min_width=70,
                scale=0,
            )

        with gr.Row(elem_classes="clefts-sort-row"):
            if show_sort_controls:
                sort_column = gr.Dropdown(
                    choices=["None", *dataframe_headers],
                    value="None",
                    show_label=False,
                    container=False,
                    min_width=120,
                    scale=0,
                )

                sort_order = gr.Dropdown(
                    choices=["Ascending", "Descending"],
                    value="Ascending",
                    show_label=False,
                    container=False,
                    min_width=110,
                    scale=0,
                )
            else:
                sort_column = gr.Dropdown(choices=["None"], value="None", visible=False)
                sort_order = gr.Dropdown(choices=["Ascending"], value="Ascending", visible=False)

    def update_page(
        sorted_dataframe: pd.DataFrame,
        page_value: str,
        rows_per_page_value: int,
    ):
        page_html_value, next_page, page_count_value = make_page_html(
            sorted_dataframe,
            page=page_value,
            rows_per_page=rows_per_page_value,
            height=height,
            show_index=show_index,
        )

        return (
            gr.update(value=page_html_value),
            gr.update(value=next_page, min_width=page_number_min_width(page_count_value)),
            page_count_value,
        )

    def move_page(
        sorted_dataframe: pd.DataFrame,
        page_value: str,
        rows_per_page_value: int,
        delta: int,
    ):
        next_page_value = _int_or_default(page_value, 1) + delta
        return update_page(sorted_dataframe, str(next_page_value), rows_per_page_value)

    def update_sort(
        source_dataframe: pd.DataFrame,
        rows_per_page_value: int,
        sort_column_value: str,
        sort_order_value: str,
    ):
        new_sorted_df = sort_dataframe(
            source_dataframe,
            sort_column_value,
            sort_order_value,
        )

        page_html_value, next_page, page_count_value = make_page_html(
            new_sorted_df,
            page=1,
            rows_per_page=rows_per_page_value,
            height=height,
            show_index=show_index,
        )

        return (
            new_sorted_df,
            gr.update(value=page_html_value),
            gr.update(value=next_page, min_width=page_number_min_width(page_count_value)),
            page_count_value,
        )

    previous_button.click(
        fn=lambda sorted_df_value, page, per_page: move_page(
            sorted_df_value,
            page,
            per_page,
            -1,
        ),
        inputs=[sorted_state, page_number, rows_per_page_input],
        outputs=[html_table, page_number, page_count],
        show_progress="hidden",
    )

    next_button.click(
        fn=lambda sorted_df_value, page, per_page: move_page(
            sorted_df_value,
            page,
            per_page,
            1,
        ),
        inputs=[sorted_state, page_number, rows_per_page_input],
        outputs=[html_table, page_number, page_count],
        show_progress="hidden",
    )

    page_number.submit(
        fn=update_page,
        inputs=[sorted_state, page_number, rows_per_page_input],
        outputs=[html_table, page_number, page_count],
        show_progress="hidden",
    )

    rows_per_page_input.change(
        fn=update_page,
        inputs=[sorted_state, page_number, rows_per_page_input],
        outputs=[html_table, page_number, page_count],
        show_progress="hidden",
    )

    sort_column.change(
        fn=update_sort,
        inputs=[source_state, rows_per_page_input, sort_column, sort_order],
        outputs=[sorted_state, html_table, page_number, page_count],
        show_progress="hidden",
    )

    sort_order.change(
        fn=update_sort,
        inputs=[source_state, rows_per_page_input, sort_column, sort_order],
        outputs=[sorted_state, html_table, page_number, page_count],
        show_progress="hidden",
    )

    return PaginatedDataframe(
        source_dataframe=source_state,
        sorted_dataframe=sorted_state,
        dataframe=html_table,
        previous_button=previous_button,
        page_number=page_number,
        page_count=page_count,
        next_button=next_button,
        rows_per_page=rows_per_page_input,
        sort_column=sort_column,
        sort_order=sort_order,
    )


def make_page_html(
    dataframe: pd.DataFrame,
    *,
    page: Any,
    rows_per_page: Any,
    height: int | None = None,
    show_index: bool = False,
) -> tuple[str, str, str]:
    current_page, total_pages, per_page = page_bounds(
        row_count=len(dataframe),
        page=page,
        rows_per_page=rows_per_page,
    )

    start = (current_page - 1) * per_page
    end = start + per_page

    page_dataframe = dataframe.iloc[start:end].copy()

    html_value = dataframe_to_html(
        page_dataframe,
        height=height,
        show_index=show_index,
    )

    return html_value, str(current_page), f"/ {total_pages}"


def dataframe_to_html(
    dataframe: pd.DataFrame,
    *,
    height: int | None = None,
    show_index: bool = False,
) -> str:
    table_html = dataframe.to_html(
        index=show_index,
        escape=True,
        classes="clefts-html-table",
    )

    if height is None:
        container_style = "margin:0;padding:0;"
    else:
        container_style = (
            f"margin:0;padding:0;max-height:{height}px;overflow-y:auto;"
        )

    return f'<div style="{container_style}">{table_html}</div>'


def page_number_min_width(page_count_text: str) -> int:
    digits = "".join(ch for ch in page_count_text if ch.isdigit())
    digit_count = max(1, len(digits))
    return max(28, 14 + digit_count * 8)


def to_dataframe(
    dataframe: Any,
    headers: Sequence[str] | None = None,
) -> pd.DataFrame:
    if dataframe is None:
        result = pd.DataFrame(columns=list(headers or []))
    elif isinstance(dataframe, pd.DataFrame):
        result = dataframe.copy()
    elif isinstance(dataframe, dict):
        result = pd.DataFrame(dataframe)
    else:
        rows = list(dataframe)
        result = pd.DataFrame(rows, columns=list(headers)) if headers else pd.DataFrame(rows)

    if headers:
        result = result.reindex(columns=list(headers), fill_value="")

    return result.reset_index(drop=True)


def sort_dataframe(
    dataframe: pd.DataFrame,
    sort_column: str | None,
    sort_order: str,
) -> pd.DataFrame:
    if not sort_column or sort_column == "None" or sort_column not in dataframe.columns:
        return dataframe.copy()

    return dataframe.sort_values(
        by=sort_column,
        ascending=sort_order != "Descending",
        kind="mergesort",
        ignore_index=True,
    )


def page_bounds(
    row_count: int,
    page: Any,
    rows_per_page: Any,
) -> tuple[int, int, int]:
    per_page = max(1, _int_or_default(rows_per_page, 10))
    total_pages = max(1, math.ceil(row_count / per_page))
    current_page = min(max(1, _int_or_default(page, 1)), total_pages)
    return current_page, total_pages, per_page


def _coerce_choice(value: Any, choices: Sequence[int]) -> int:
    normalized_choices = [int(choice) for choice in choices] or [10]
    requested = _int_or_default(value, normalized_choices[0])
    return requested if requested in normalized_choices else normalized_choices[0]


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def create_demo_app() -> gr.Blocks:
    import random
    rows = pd.DataFrame(
        {
            "ID": range(1, 201),
            "Name": [f"row-{index}" for index in range(1, 201)],
            "Value": [random.random() * 100 for index in range(1, 201)],
        }
    ).astype(
        {
            "ID": "int",
            "Name": "str",
            "Value": "float",
        }
    )

    with gr.Blocks(
        title="Paginated HTML Dataframe Demo"
    ) as app:
        gr.Markdown(
            "# Paginated HTML Dataframe Demo"
        )

        render_paginated_dataframe(
            rows,
            headers=["ID", "Name", "Value"],
            rows_per_page=10,
            label="Demo Table",
            height=None,
        )

    return app


if __name__ == "__main__":
    create_demo_app().launch(
        css=PAGINATED_DATAFRAME_CSS
    )