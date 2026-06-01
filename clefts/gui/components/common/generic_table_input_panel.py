from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import pandas as pd

from .error_message import render_error_html

ICON_DIR = Path(__file__).resolve().parent.parent / "icons"


@dataclass(frozen=True)
class GenericTableInputPanel:
    table_json: gr.Textbox
    table_html: gr.HTML
    copy_button: gr.Button
    download_button: gr.Button
    open_add_button: gr.Button
    add_inputs: dict[str, gr.Textbox]
    add_button: gr.Button
    cancel_add_button: gr.Button
    load_file: gr.UploadButton
    delete_index: gr.Textbox
    error_message: gr.HTML


@dataclass(frozen=True)
class GenericTableColumn:
    key: str
    label: str
    required: bool = False
    code: bool = False
    placeholder: str = ""
    choices: list[str] | None = None


def rows_to_json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, ensure_ascii=False, indent=2)


def json_to_rows(value: str) -> list[dict[str, Any]]:
    if not value:
        return []

    data = json.loads(value)

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    raise ValueError("JSON must be a list of row objects or {'rows': [...]}.")


def table_json_to_dataframe(value: str) -> pd.DataFrame:
    return pd.DataFrame(json_to_rows(value))


def normalize_rows(
    rows: Any,
    columns: list[GenericTableColumn],
) -> list[dict[str, str]]:
    if rows is None:
        return []

    if isinstance(rows, str):
        rows = json_to_rows(rows)

    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")

    if not isinstance(rows, list):
        raise ValueError("Rows must be list[dict], JSON string, or pandas.DataFrame.")

    normalized_rows: list[dict[str, str]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        normalized_row = {
            column.key: str(row.get(column.key, "")).strip()
            for column in columns
        }

        if any(normalized_row.values()):
            normalized_rows.append(normalized_row)

    return normalized_rows

def export_table_json(
    table_json: str,
    export_json_fn: Callable[[str], str] | None,
) -> str:
    if export_json_fn is None:
        return table_json or "[]"
    return export_json_fn(table_json or "[]")

def load_json_file(
    path: str | Path,
    columns: list[GenericTableColumn],
    import_json_fn: Callable[[str], list[dict[str, str]]] | None = None,
) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if import_json_fn is not None:
        return normalize_rows(import_json_fn(text), columns)

    return normalize_rows(text, columns)


def render_table_html(
    rows: list[dict[str, str]],
    columns: list[GenericTableColumn],
    *,
    empty_message: str,
    delete_input_elem_id: str,
) -> str:
    if not rows:
        body = (
            f'<tr><td colspan="{len(columns) + 1}" '
            f'class="clefts-generic-table-empty-cell">'
            f"{html.escape(empty_message)}"
            "</td></tr>"
        )
    else:
        body = "\n".join(
            render_row_html(
                row=row,
                index=index,
                columns=columns,
                delete_input_elem_id=delete_input_elem_id,
            )
            for index, row in enumerate(rows)
        )

    header_cells = "".join(
        f"<th>{html.escape(column.label)}</th>"
        for column in columns
    )

    return f"""
<table class="clefts-generic-table">
    <thead>
        <tr>{header_cells}<th></th></tr>
    </thead>
    <tbody>{body}</tbody>
</table>
"""


def render_row_html(
    row: dict[str, str],
    index: int,
    columns: list[GenericTableColumn],
    delete_input_elem_id: str,
) -> str:
    cells = []

    for column in columns:
        value = html.escape(str(row.get(column.key, "")))
        if column.code:
            value = f"<code>{value}</code>"
        cells.append(f"<td>{value}</td>")

    delete_button = f"""
<td class="clefts-generic-table-delete-cell">
    <button
        type="button"
        class="clefts-generic-table-delete-button"
        data-index="{index}"
        aria-label="Delete row"
        onclick="
            const root = document.getElementById('{delete_input_elem_id}');
            const input = root ? root.querySelector('textarea, input') : null;
            if (input) {{
                const valueSetter =
                    Object.getOwnPropertyDescriptor(
                        input instanceof HTMLTextAreaElement
                            ? HTMLTextAreaElement.prototype
                            : HTMLInputElement.prototype,
                        'value'
                    ).set;
                valueSetter.call(input, '{index}');
                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
        "
    >
        &times;
    </button>
</td>
"""

    return "<tr>" + "".join(cells) + delete_button + "</tr>"


def open_add_panel():
    return gr.update(visible=True)


def close_add_panel(columns: list[GenericTableColumn]):
    return (gr.update(visible=False), *["" for _ in columns])


def delete_row(
    table_json: str,
    delete_index: str | int | float | None,
    columns: list[GenericTableColumn],
    empty_message: str,
    delete_input_elem_id: str,
):
    try:
        if delete_index is None:
            return gr.update(), gr.update(), "", "-1"

        index = int(delete_index)

        if index < 0:
            return gr.update(), gr.update(), "", "-1"

        rows = normalize_rows(table_json, columns)

        if index >= len(rows):
            return (
                gr.update(),
                gr.update(),
                render_error_html("Row was not found."),
                "-1",
            )

        del rows[index]

        return (
            render_table_html(
                rows,
                columns,
                empty_message=empty_message,
                delete_input_elem_id=delete_input_elem_id,
            ),
            rows_to_json(rows),
            "",
            "-1",
        )

    except Exception as exc:
        return (
            gr.update(),
            gr.update(),
            render_error_html(f"Delete error: {exc}"),
            "-1",
        )


def add_row(
    table_json: str,
    columns: list[GenericTableColumn],
    empty_message: str,
    delete_input_elem_id: str,
    *values: str,
):
    try:
        new_row = {
            column.key: str(value or "").strip()
            for column, value in zip(columns, values, strict=True)
        }

        missing_columns = [
            column.label
            for column in columns
            if column.required and not new_row[column.key]
        ]

        if missing_columns:
            return (
                gr.update(),
                gr.update(),
                render_error_html(f"Required: {', '.join(missing_columns)}"),
                gr.update(visible=True),
                *values,
            )

        rows = normalize_rows(table_json, columns)
        rows.append(new_row)

        return (
            render_table_html(
                rows,
                columns,
                empty_message=empty_message,
                delete_input_elem_id=delete_input_elem_id,
            ),
            rows_to_json(rows),
            "",
            gr.update(visible=False),
            *["" for _ in columns],
        )

    except Exception as exc:
        return (
            gr.update(),
            gr.update(),
            render_error_html(f"Add error: {exc}"),
            gr.update(visible=True),
            *values,
        )


def apply_json_file(
    file_value: Any,
    columns: list[GenericTableColumn],
    empty_message: str,
    delete_input_elem_id: str,
    import_json_fn: Callable[[str], list[dict[str, str]]] | None = None,
):
    try:
        if file_value is None:
            return gr.update(), gr.update(), render_error_html("Select a JSON file first.")

        path = getattr(file_value, "name", file_value)
        rows = load_json_file(path, columns, import_json_fn=import_json_fn)

        return (
            render_table_html(
                rows,
                columns,
                empty_message=empty_message,
                delete_input_elem_id=delete_input_elem_id,
            ),
            rows_to_json(rows),
            "",
        )

    except Exception as exc:
        return gr.update(), gr.update(), render_error_html(f"File error: {exc}")


def render_generic_table_input_panel(
    *,
    title: str,
    columns: list[GenericTableColumn],
    default_rows: Any | None = None,
    panel_elem_id: str = "clefts-generic-table-panel",
    add_panel_elem_id: str = "clefts-generic-table-add-panel",
    delete_input_elem_id: str = "clefts-generic-table-delete-index",
    empty_message: str = "No rows.",
    download_filename: str = "table.json",
    export_json_fn: Callable[[str], str] | None = None,
    import_json_fn: Callable[[str], list[dict[str, str]]] | None = None,
) -> GenericTableInputPanel:
    rows = normalize_rows(default_rows or [], columns)

    table_json = gr.Textbox(
        value=rows_to_json(rows),
        visible=False,
    )
    exported_table_json = gr.Textbox(
        value=export_table_json(
            table_json.value,
            export_json_fn,
        ),
        visible=False,
    )
    table_json.change(
        fn=lambda value: export_table_json(
            value,
            export_json_fn,
        ),
        inputs=[table_json],
        outputs=[exported_table_json],
        show_progress="hidden",
    )

    with gr.Group(
        elem_id=panel_elem_id,
        elem_classes="clefts-generic-table-panel",
    ):
        with gr.Row(elem_classes="clefts-generic-table-toolbar"):
            with gr.Column(
                scale=1,
                min_width=0,
                elem_classes="clefts-generic-table-title-column",
            ):
                gr.HTML(
                    f'<div class="clefts-generic-table-heading">'
                    f"{html.escape(title)}"
                    f"</div>"
                )

            with gr.Column(
                scale=0,
                min_width=148,
                elem_classes="clefts-generic-table-actions-column",
            ):
                with gr.Row(elem_classes="clefts-generic-table-actions"):
                    copy_button = gr.Button(
                        "",
                        size="sm",
                        icon=str(ICON_DIR / "copy.svg"),
                        elem_id=f"{panel_elem_id}-copy",
                        elem_classes="clefts-generic-table-icon-button",
                        min_width=30,
                        scale=0,
                    )

                    download_button = gr.Button(
                        "",
                        size="sm",
                        icon=str(ICON_DIR / "download.svg"),
                        elem_classes="clefts-generic-table-icon-button",
                        min_width=30,
                        scale=0,
                    )

                    load_file = gr.UploadButton(
                        "",
                        size="sm",
                        icon=str(ICON_DIR / "upload.svg"),
                        type="filepath",
                        elem_classes="clefts-generic-table-icon-button",
                        min_width=30,
                        scale=0,
                    )

                    open_add_button = gr.Button(
                        "",
                        variant="secondary",
                        size="sm",
                        icon=str(ICON_DIR / "plus.svg"),
                        elem_classes="clefts-generic-table-icon-button",
                        min_width=30,
                        scale=0,
                    )

        table_html = gr.HTML(
            render_table_html(
                rows,
                columns,
                empty_message=empty_message,
                delete_input_elem_id=delete_input_elem_id,
            )
        )

    with gr.Column(
        visible=False,
        elem_id=add_panel_elem_id,
        elem_classes="clefts-generic-table-modal-backdrop",
    ) as add_panel:
        with gr.Group(elem_classes="clefts-generic-table-modal-dialog"):
            gr.HTML(
                f'<div class="clefts-generic-table-modal-title">'
                f"Add {html.escape(title)}"
                f"</div>"
            )

            add_inputs = {}

            for column in columns:

                component: gr.Component

                if column.choices is not None:
                    component = gr.Dropdown(
                        choices=column.choices,
                        label=(
                            column.label
                            if column.required
                            else f"{column.label} (Optional)"
                        ),
                        info=None if column.required else "Optional",
                        allow_custom_value=True,
                    )

                else:
                    component = gr.Textbox(
                        label=column.label,
                        info=None if column.required else "Optional",
                        placeholder=column.placeholder,
                    )

                add_inputs[column.key] = component

            with gr.Row(elem_classes="clefts-generic-table-modal-actions"):
                cancel_add_button = gr.Button("Cancel", variant="secondary", size="sm")
                add_button = gr.Button("Add", variant="primary", size="sm")

    delete_index = gr.Textbox(
        value="-1",
        visible="hidden",
        elem_id=delete_input_elem_id,
    )

    error_message = gr.HTML(render_error_html())

    copy_button.click(
        fn=lambda value: "",
        inputs=[exported_table_json],
        outputs=[error_message],
        js=f"""(value) => {{
            navigator.clipboard.writeText(value || '[]');

            const root = document.getElementById('{panel_elem_id}-copy');
            const img = root ? root.querySelector('img') : null;

            if (img) {{
                if (!img.dataset.copySrc && !img.src.startsWith('data:')) {{
                    img.dataset.copySrc = img.src;
                }}

                const copySrc = img.dataset.copySrc || img.src;
                const checkSrc = 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2216%22 height=%2216%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22%23174f78%22 stroke-width=%222%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cpath d=%22M20 6 9 17l-5-5%22/%3E%3C/svg%3E';

                img.src = checkSrc;

                if (img.dataset.copyTimer) {{
                    window.clearTimeout(Number(img.dataset.copyTimer));
                }}

                img.dataset.copyTimer = String(window.setTimeout(() => {{
                    img.src = copySrc;
                    delete img.dataset.copyTimer;
                }}, 1000));
            }}

            return '';
        }}""",
        show_progress="hidden",
    )

    download_button.click(
        fn=lambda value: "",
        inputs=[exported_table_json],
        outputs=[error_message],
        js=f"""(value) => {{
            const blob = new Blob([value || '[]'], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');

            link.href = url;
            link.download = '{download_filename}';

            document.body.appendChild(link);
            link.click();
            link.remove();

            URL.revokeObjectURL(url);
            return '';
        }}""",
        show_progress="hidden",
    )

    open_add_button.click(
        fn=open_add_panel,
        outputs=[add_panel],
        show_progress="hidden",
    )

    cancel_add_button.click(
        fn=lambda: close_add_panel(columns),
        outputs=[add_panel, *add_inputs.values()],
        show_progress="hidden",
    )

    add_button.click(
        fn=lambda table_json_value, *values: add_row(
            table_json_value,
            columns,
            empty_message,
            delete_input_elem_id,
            *values,
        ),
        inputs=[table_json, *add_inputs.values()],
        outputs=[
            table_html,
            table_json,
            error_message,
            add_panel,
            *add_inputs.values(),
        ],
    )

    load_file.upload(
        fn=lambda file_value: apply_json_file(
            file_value,
            columns,
            empty_message,
            delete_input_elem_id,
            import_json_fn=import_json_fn,
        ),
        inputs=[load_file],
        outputs=[table_html, table_json, error_message],
    )

    delete_index.change(
        fn=lambda table_json_value, index: delete_row(
            table_json_value,
            index,
            columns,
            empty_message,
            delete_input_elem_id,
        ),
        inputs=[table_json, delete_index],
        outputs=[table_html, table_json, error_message, delete_index],
        show_progress="hidden",
    )

    return GenericTableInputPanel(
        table_json=table_json,
        table_html=table_html,
        copy_button=copy_button,
        download_button=download_button,
        open_add_button=open_add_button,
        add_inputs=add_inputs,
        add_button=add_button,
        cancel_add_button=cancel_add_button,
        load_file=load_file,
        delete_index=delete_index,
        error_message=error_message,
    )

# python -m clefts.gui.components.generic_table_input_panel
if __name__ == "__main__":
    from .styles import STYLES
    demo_columns = [
        GenericTableColumn(
            key="name",
            label="Name",
            required=True,
            placeholder="Alice Smith",
        ),
        GenericTableColumn(
            key="age",
            label="Age",
            required=True,
            placeholder="32",
        ),
        GenericTableColumn(
            key="gender",
            label="Gender",
            required=True,
            choices=[
                "Male",
                "Female",
                "Other",
            ],
        )
    ]

    demo_rows = [
        {"name": "Alice Johnson", "age": "29", "gender": "Female"},
        {"name": "Michael Brown", "age": "41", "gender": "Male"},
        {"name": "Sophia Williams", "age": "35", "gender": "Female"},
        {"name": "Daniel Miller", "age": "24", "gender": "Male"},
        {"name": "Emma Davis", "age": "31", "gender": "Female"},
    ]

    def export_people_json(table_json: str) -> str:
        return json.dumps(
            {
                "people": {
                    "rows": json.loads(table_json or "[]"),
                }
            },
            ensure_ascii=False,
            indent=2,
        )


    def import_people_json(value: str) -> list[dict[str, str]]:
        data = json.loads(value or "{}")

        people = data.get("people", {})
        rows = people.get("rows", [])

        if not isinstance(rows, list):
            raise ValueError("people.rows must be a list.")

        return rows

    with gr.Blocks(css=STYLES) as demo:
        gr.Markdown("## Generic Table Input Panel Demo")

        panel = render_generic_table_input_panel(
            title="People Table",
            columns=demo_columns,
            default_rows=demo_rows,
            panel_elem_id="people-table-panel",
            add_panel_elem_id="people-table-add-panel",
            delete_input_elem_id="people-table-delete-index",
            empty_message="No people.",
            download_filename="people_table.json",
            export_json_fn=export_people_json,
            import_json_fn=import_people_json,
        )

        dataframe_output = gr.Dataframe(
            label="Current table as pandas.DataFrame",
            interactive=False,
        )

        show_dataframe_button = gr.Button(
            "Convert current table to pandas.DataFrame",
            variant="primary",
        )

        show_dataframe_button.click(
            fn=table_json_to_dataframe,
            inputs=[panel.table_json],
            outputs=[dataframe_output],
        )

    demo.launch()