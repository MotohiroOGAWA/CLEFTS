from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import gradio as gr

from ...domain.fragment.cleavage.CleavagePattern import CleavagePattern
from ...domain.fragment.cleavage.CleavagePatternSet import CleavagePatternSet
from .common.generic_table_input_panel import (
    GenericTableColumn,
    GenericTableInputPanel,
    json_to_rows,
    normalize_rows,
    render_generic_table_input_panel,
    table_json_to_dataframe,
)


DEFAULT_CHARGE_MODE = "any"


@dataclass(frozen=True)
class CleavagePatternsInputPanel:
    table_json: gr.Textbox
    pattern_set_json: gr.Textbox
    patterns_html: gr.HTML
    copy_button: gr.Button
    download_button: gr.Button
    open_add_button: gr.Button
    add_name: gr.Component
    add_smirks: gr.Component
    add_button: gr.Button
    cancel_add_button: gr.Button
    load_file: gr.UploadButton
    delete_index: gr.Textbox
    error_message: gr.HTML


def get_cleavage_pattern_columns() -> list[GenericTableColumn]:
    return [
        GenericTableColumn(
            key="name",
            label="Name",
            required=False,
            code=False,
            placeholder="optional",
        ),
        GenericTableColumn(
            key="smirks",
            label="SMIRKS",
            required=True,
            code=True,
            placeholder="[!#1:1]-!@[!#1:2]>>[!#1:1][H]",
        ),
    ]


def pattern_set_to_rows(
    pattern_set: CleavagePatternSet,
) -> list[dict[str, str]]:
    return [
        {
            "name": str(pattern.name or ""),
            "smirks": str(pattern.smirks or ""),
        }
        for pattern in pattern_set.patterns
    ]


def rows_to_pattern_set(
    rows: Any,
    *,
    name: str = "",
) -> CleavagePatternSet:
    columns = get_cleavage_pattern_columns()
    normalized_rows = normalize_rows(rows, columns)

    patterns: list[CleavagePattern] = []

    for row in normalized_rows:
        smirks = str(row.get("smirks", "")).strip()
        if not smirks:
            continue

        patterns.append(
            CleavagePattern(
                name=str(row.get("name", "")).strip(),
                smirks=smirks,
                charge_mode=DEFAULT_CHARGE_MODE,
            )
        )

    return CleavagePatternSet(
        patterns=patterns,
        name=name,
    )


def pattern_set_to_json(
    pattern_set: CleavagePatternSet,
) -> str:
    return json.dumps(
        {
            "name": str(getattr(pattern_set, "name", "") or ""),
            "patterns": pattern_set_to_rows(pattern_set),
        },
        ensure_ascii=False,
        indent=2,
    )


def json_to_pattern_set(
    value: str,
) -> CleavagePatternSet:
    data = json.loads(value or "{}")

    if isinstance(data, list):
        return rows_to_pattern_set(data)

    if not isinstance(data, dict):
        raise ValueError(
            "JSON must be a CleavagePatternSet object or a pattern list."
        )

    if "patterns" in data:
        return rows_to_pattern_set(
            data.get("patterns", []),
            name=str(data.get("name", "") or ""),
        )

    return rows_to_pattern_set([data])


def export_cleavage_patterns_json(
    table_json: str,
) -> str:
    rows = json_to_rows(table_json or "[]")
    pattern_set = rows_to_pattern_set(rows)
    return pattern_set_to_json(pattern_set)


def import_cleavage_patterns_json(
    pattern_set_json: str,
) -> list[dict[str, str]]:
    pattern_set = json_to_pattern_set(pattern_set_json)
    return pattern_set_to_rows(pattern_set)


def render_cleavage_patterns_input_panel(
    default_pattern_set: CleavagePatternSet,
) -> CleavagePatternsInputPanel:
    columns = get_cleavage_pattern_columns()
    default_rows = pattern_set_to_rows(default_pattern_set)

    generic_panel: GenericTableInputPanel = render_generic_table_input_panel(
        title="Cleavage Patterns",
        columns=columns,
        default_rows=default_rows,
        panel_elem_id="clefts-cleavage-patterns-table-panel",
        add_panel_elem_id="clefts-cleavage-patterns-table-add-panel",
        delete_input_elem_id="clefts-cleavage-patterns-table-delete-index",
        empty_message="No cleavage patterns.",
        download_filename="cleavage_patterns.json",
        export_json_fn=export_cleavage_patterns_json,
        import_json_fn=import_cleavage_patterns_json,
    )

    pattern_set_json = gr.Textbox(
        value=pattern_set_to_json(default_pattern_set),
        visible=False,
    )

    generic_panel.table_json.change(
        fn=export_cleavage_patterns_json,
        inputs=[generic_panel.table_json],
        outputs=[pattern_set_json],
        show_progress="hidden",
    )

    return CleavagePatternsInputPanel(
        table_json=generic_panel.table_json,
        pattern_set_json=pattern_set_json,
        patterns_html=generic_panel.table_html,
        copy_button=generic_panel.copy_button,
        download_button=generic_panel.download_button,
        open_add_button=generic_panel.open_add_button,
        add_name=generic_panel.add_inputs["name"],
        add_smirks=generic_panel.add_inputs["smirks"],
        add_button=generic_panel.add_button,
        cancel_add_button=generic_panel.cancel_add_button,
        load_file=generic_panel.load_file,
        delete_index=generic_panel.delete_index,
        error_message=generic_panel.error_message,
    )


# python -m clefts.gui.components.cleavage_patterns_input_panel
if __name__ == "__main__":

    demo_pattern_set = CleavagePatternSet(
        patterns=[
            CleavagePattern(
                name="amide cleavage",
                smirks="[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH]",
                charge_mode="any",
            ),
            CleavagePattern(
                name="ester cleavage",
                smirks="[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH]",
                charge_mode="any",
            ),
            CleavagePattern(
                name="c-c bond cleavage",
                smirks="[C:1]-[C:2]>>[C:1]",
                charge_mode="any",
            ),
            CleavagePattern(
                name="ring opening",
                smirks="[C:1]1[C:2][C:3]1>>[C:1]",
                charge_mode="any",
            ),
        ],
        name="Demo Cleavage Patterns",
    )

    with gr.Blocks() as demo:
        gr.Markdown("# Cleavage Patterns Input Panel Demo")

        panel = render_cleavage_patterns_input_panel(
            default_pattern_set=demo_pattern_set,
        )

        with gr.Row():
            pattern_set_json_output = gr.Textbox(
                label="Pattern Set JSON",
                lines=20,
                interactive=False,
            )

            dataframe_output = gr.Dataframe(
                label="Current Table as pandas.DataFrame",
                interactive=False,
            )

        refresh_button = gr.Button(
            "Refresh Outputs",
            variant="primary",
        )

        refresh_button.click(
            fn=lambda table_json: (
                export_cleavage_patterns_json(table_json),
                table_json_to_dataframe(table_json),
            ),
            inputs=[panel.table_json],
            outputs=[
                pattern_set_json_output,
                dataframe_output,
            ],
        )

    demo.launch()