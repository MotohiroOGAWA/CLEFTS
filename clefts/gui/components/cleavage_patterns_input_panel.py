from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from clefts.domain.fragment.cleavage.CleavagePattern import CleavagePattern
from clefts.domain.fragment.cleavage.CleavagePatternSet import CleavagePatternSet
from clefts.gui.components.error_message import render_error_html


DEFAULT_CHARGE_MODE = "any"
ICON_DIR = Path(__file__).resolve().parent / "icons"


@dataclass(frozen=True)
class CleavagePatternsInputPanel:
    pattern_set_json: gr.Textbox
    patterns_html: gr.HTML
    copy_button: gr.Button
    download_button: gr.Button
    open_add_button: gr.Button
    add_name: gr.Textbox
    add_smirks: gr.Textbox
    add_button: gr.Button
    cancel_add_button: gr.Button
    load_file: gr.UploadButton
    delete_index: gr.Textbox
    error_message: gr.HTML


def pattern_set_to_json(pattern_set: CleavagePatternSet) -> str:
    data = {
        "name": "",
        "patterns": [
            {"name": pattern.name, "smirks": pattern.smirks}
            for pattern in pattern_set.patterns
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def pattern_set_to_rows(pattern_set: CleavagePatternSet) -> list[dict[str, str]]:
    return [{"name": pattern.name, "smirks": pattern.smirks} for pattern in pattern_set.patterns]


def rows_to_pattern_set(rows: Any, name: str = "") -> CleavagePatternSet:
    patterns = []
    for row in _normalize_rows(rows):
        smirks = str(row.get("smirks", "")).strip()
        if not smirks:
            continue
        patterns.append(
            CleavagePattern(
                smirks=smirks,
                name=str(row.get("name", "")).strip(),
                charge_mode=DEFAULT_CHARGE_MODE,
            )
        )
    return CleavagePatternSet(patterns, name=name)


def json_to_pattern_set(value: str) -> CleavagePatternSet:
    data = json.loads(value)
    if isinstance(data, list):
        return rows_to_pattern_set(data)
    if not isinstance(data, dict):
        raise ValueError("JSON must be a CleavagePatternSet object or a pattern list.")
    if "patterns" in data:
        cleaned = {
            "name": str(data.get("name", "")),
            "patterns": [
                {
                    "name": str(item.get("name", "")),
                    "smirks": str(item["smirks"]),
                    "charge_mode": DEFAULT_CHARGE_MODE,
                }
                for item in data.get("patterns", [])
                if isinstance(item, dict) and str(item.get("smirks", "")).strip()
            ],
        }
        return CleavagePatternSet.from_dict(cleaned)
    return rows_to_pattern_set([data])


def load_pattern_set_json_file(path: str | Path) -> CleavagePatternSet:
    with open(path, "r", encoding="utf-8") as f:
        return json_to_pattern_set(f.read())


def render_patterns_table_html(pattern_set: CleavagePatternSet) -> str:
    rows = pattern_set_to_rows(pattern_set)
    if not rows:
        body = '<tr><td colspan="3" class="clefts-empty-cell">No cleavage patterns.</td></tr>'
    else:
        body = "\n".join(
            "<tr>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td><code>{html.escape(row['smirks'])}</code></td>"
            "<td class=\"clefts-pattern-delete-cell\">"
            f"<button type=\"button\" class=\"clefts-pattern-delete-button\" data-index=\"{index}\" aria-label=\"Delete pattern\" "
            f"onclick=\"const input=document.querySelector('#clefts-pattern-delete-index textarea, #clefts-pattern-delete-index input'); if(input){{input.value='{index}'; input.dispatchEvent(new Event('input', {{bubbles:true}})); input.dispatchEvent(new Event('change', {{bubbles:true}}));}}\">&times;</button>"
            "</td>"
            "</tr>"
            for index, row in enumerate(rows)
        )
    return f"""
    <table class="clefts-pattern-table">
        <thead>
            <tr><th>Name</th><th>SMIRKS</th><th></th></tr>
        </thead>
        <tbody>{body}</tbody>
    </table>
    """


def _normalize_rows(value: Any) -> list[dict[str, str]]:
    rows = []
    for row in value or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        smirks = str(row.get("smirks", "")).strip()
        if name or smirks:
            rows.append({"name": name, "smirks": smirks})
    return rows


def open_add_panel():
    return gr.update(visible=True)


def close_add_panel():
    return gr.update(visible=False), "", ""


def delete_pattern(pattern_set_json: str, delete_index: str | int | float | None):
    try:
        if delete_index is None or int(delete_index) < 0:
            return gr.update(), gr.update(), "", "-1"
        pattern_set = json_to_pattern_set(pattern_set_json)
        rows = pattern_set_to_rows(pattern_set)
        index = int(delete_index)
        if index >= len(rows):
            return gr.update(), gr.update(), render_error_html("Pattern was not found."), "-1"
        del rows[index]
        next_pattern_set = rows_to_pattern_set(rows, name=pattern_set.name)
        return render_patterns_table_html(next_pattern_set), pattern_set_to_json(next_pattern_set), "", "-1"
    except Exception as exc:
        return gr.update(), gr.update(), render_error_html(f"Delete error: {exc}"), "-1"


def add_pattern(pattern_set_json: str, name: str, smirks: str):
    smirks = str(smirks or "").strip()
    if not smirks:
        return gr.update(), gr.update(), render_error_html("SMIRKS is required."), gr.update(visible=True), name, smirks

    pattern_set = json_to_pattern_set(pattern_set_json)
    rows = pattern_set_to_rows(pattern_set)
    rows.append({"name": str(name or "").strip(), "smirks": smirks})
    next_pattern_set = rows_to_pattern_set(rows, name=pattern_set.name)
    return (
        render_patterns_table_html(next_pattern_set),
        pattern_set_to_json(next_pattern_set),
        "",
        gr.update(visible=False),
        "",
        "",
    )


def apply_json_file(file_value):
    try:
        if file_value is None:
            return gr.update(), gr.update(), render_error_html("Select a JSON file first.")
        path = getattr(file_value, "name", file_value)
        pattern_set = load_pattern_set_json_file(path)
        return render_patterns_table_html(pattern_set), pattern_set_to_json(pattern_set), ""
    except Exception as exc:
        return gr.update(), gr.update(), render_error_html(f"File error: {exc}")


def render_cleavage_patterns_input_panel(
    default_pattern_set: CleavagePatternSet,
) -> CleavagePatternsInputPanel:
    pattern_set_json = gr.Textbox(
        value=pattern_set_to_json(default_pattern_set),
        visible=False,
    )
    with gr.Group(elem_id="clefts-pattern-panel", elem_classes="clefts-pattern-panel"):
        with gr.Row(elem_classes="clefts-pattern-toolbar"):
            with gr.Column(scale=1, min_width=0, elem_classes="clefts-pattern-title-column"):
                gr.HTML('<div class="clefts-pattern-heading">Cleavage Patterns</div>')
            with gr.Column(scale=0, min_width=148, elem_classes="clefts-pattern-actions-column"):
                with gr.Row(elem_classes="clefts-pattern-actions"):
                    copy_button = gr.Button(
                        "",
                        size="sm",
                        icon=str(ICON_DIR / "copy.svg"),
                        elem_id="clefts-copy-patterns-json",
                        elem_classes="clefts-icon-button",
                        min_width=30,
                        scale=0,
                    )
                    download_button = gr.Button(
                        "",
                        size="sm",
                        icon=str(ICON_DIR / "download.svg"),
                        elem_classes="clefts-icon-button",
                        min_width=30,
                        scale=0,
                    )
                    load_file = gr.UploadButton(
                        "",
                        size="sm",
                        icon=str(ICON_DIR / "file.svg"),
                        type="filepath",
                        elem_classes="clefts-icon-button",
                        min_width=30,
                        scale=0,
                    )
                    open_add_button = gr.Button(
                        "",
                        variant="secondary",
                        size="sm",
                        icon=str(ICON_DIR / "plus.svg"),
                        elem_classes="clefts-icon-button",
                        min_width=30,
                        scale=0,
                    )

        patterns_html = gr.HTML(render_patterns_table_html(default_pattern_set))

    with gr.Column(visible=False, elem_id="clefts-add-pattern-panel", elem_classes="clefts-modal-backdrop") as add_panel:
        with gr.Group(elem_classes="clefts-modal-dialog"):
            gr.HTML('<div class="clefts-modal-title">Add cleavage pattern</div>')
            add_name = gr.Textbox(label="name", placeholder="optional")
            add_smirks = gr.Textbox(label="SMIRKS", placeholder="[!#1:1]-!@[!#1:2]>>[!#1:1][H]")
            with gr.Row(elem_classes="clefts-modal-actions"):
                cancel_add_button = gr.Button("Cancel", variant="secondary", size="sm")
                add_button = gr.Button("Add", variant="primary", size="sm")

    delete_index = gr.Textbox(value="-1", visible="hidden", elem_id="clefts-pattern-delete-index")
    error_message = gr.HTML(render_error_html())

    copy_button.click(
        fn=lambda value: "",
        inputs=[pattern_set_json],
        outputs=[error_message],
        js="""(value) => {
            navigator.clipboard.writeText(value || '');
            const root = document.getElementById('clefts-copy-patterns-json');
            const img = root ? root.querySelector('img') : null;
            if (img) {
                if (!img.dataset.copySrc && !img.src.startsWith('data:')) {
                    img.dataset.copySrc = img.src;
                }
                const copySrc = img.dataset.copySrc || img.src;
                const checkSrc = 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2216%22 height=%2216%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22%23174f78%22 stroke-width=%222%22 stroke-linecap=%22round%22 stroke-linejoin=%22round%22%3E%3Cpath d=%22M20 6 9 17l-5-5%22/%3E%3C/svg%3E';
                img.src = checkSrc;
                if (img.dataset.copyTimer) {
                    window.clearTimeout(Number(img.dataset.copyTimer));
                }
                img.dataset.copyTimer = String(window.setTimeout(() => {
                    img.src = copySrc;
                    delete img.dataset.copyTimer;
                }, 1000));
            }
            return '';
        }""",
        show_progress="hidden",
    )
    download_button.click(
        fn=lambda value: "",
        inputs=[pattern_set_json],
        outputs=[error_message],
        js="""(value) => {
            const blob = new Blob([value || ''], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'cleavage_patterns.json';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            return '';
        }""",
        show_progress="hidden",
    )
    open_add_button.click(
        fn=open_add_panel,
        outputs=[add_panel],
        js="""() => {
            const panel = document.getElementById('clefts-add-pattern-panel');
            if (panel) { panel.style.display = ''; }
            return [];
        }""",
        show_progress="hidden",
    )
    cancel_add_button.click(
        fn=close_add_panel,
        outputs=[add_panel, add_name, add_smirks],
        js="""() => {
            const panel = document.getElementById('clefts-add-pattern-panel');
            if (panel) { panel.style.display = 'none'; }
            return [];
        }""",
        show_progress="hidden",
    )
    add_button.click(
        fn=add_pattern,
        inputs=[pattern_set_json, add_name, add_smirks],
        outputs=[patterns_html, pattern_set_json, error_message, add_panel, add_name, add_smirks],
        js="""(patternSetJson, name, smirks) => {
            if (String(smirks || '').trim()) {
                const panel = document.getElementById('clefts-add-pattern-panel');
                if (panel) { panel.style.display = 'none'; }
            }
            return [patternSetJson, name, smirks];
        }""",
    )
    load_file.upload(
        fn=apply_json_file,
        inputs=[load_file],
        outputs=[patterns_html, pattern_set_json, error_message],
    )
    delete_index.change(
        fn=delete_pattern,
        inputs=[pattern_set_json, delete_index],
        outputs=[patterns_html, pattern_set_json, error_message, delete_index],
        show_progress="hidden",
    )

    return CleavagePatternsInputPanel(
        pattern_set_json=pattern_set_json,
        patterns_html=patterns_html,
        copy_button=copy_button,
        download_button=download_button,
        open_add_button=open_add_button,
        add_name=add_name,
        add_smirks=add_smirks,
        add_button=add_button,
        cancel_add_button=cancel_add_button,
        load_file=load_file,
        delete_index=delete_index,
        error_message=error_message,
    )
