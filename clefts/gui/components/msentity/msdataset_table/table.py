from __future__ import annotations

from typing import Tuple
import gradio as gr
from pathlib import Path

from .....libs.msentity.msentity import MSDataset

from .spectrum_window import render_spectrum_window, open_spectrum_window

ICON_DIR = Path(__file__).resolve().parent.parent.parent / "icons"
MS_ICON_SVG = (ICON_DIR / "ms.svg").read_text(encoding="utf-8")

SPECTRUM_BUTTON_SCRIPT = """
<script>
document.addEventListener("click", function(event) {
    const button = event.target.closest(".msdataset-spectrum-button");

    if (!button) {
        return;
    }

    const spectrumIndex = button.getAttribute("data-spectrum-index");

    const wrapper = document.querySelector("#selected-spectrum-index");

    if (!wrapper) {
        console.warn("selected-spectrum-index wrapper was not found.");
        return;
    }

    const textbox =
        wrapper.querySelector("textarea") ||
        wrapper.querySelector("input");

    if (!textbox) {
        console.warn("textbox input was not found.");
        return;
    }

    textbox.value = spectrumIndex;

    textbox.dispatchEvent(
        new Event("input", {
            bubbles: true,
        })
    );

    textbox.dispatchEvent(
        new Event("change", {
            bubbles: true,
        })
    );
});
</script>
"""

def render_dataset_html(
    dataset: MSDataset | None,
    page: int,
    rows_per_page: int,
    height: int | None = None,
    show_index: bool = True,
    spectrum_input_elem_id: str = "selected-spectrum-index",
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
                f'type="button" '
                f'class="msdataset-spectrum-button" '
                f'data-spectrum-index="{start + i}" '
                f'title="View spectrum" '
                f'aria-label="View spectrum" '
                f'onclick="'
                f"const root = document.getElementById('{spectrum_input_elem_id}');"
                f"const input = root ? root.querySelector('textarea, input') : null;"
                f"if (input) {{"
                f"const valueSetter = "
                f"Object.getOwnPropertyDescriptor("
                f"input instanceof HTMLTextAreaElement "
                f"? HTMLTextAreaElement.prototype "
                f": HTMLInputElement.prototype, "
                f"'value'"
                f").set;"
                f"valueSetter.call(input, '{start + i}');"
                f"input.dispatchEvent(new Event('input', {{ bubbles: true }}));"
                f"input.dispatchEvent(new Event('change', {{ bubbles: true }}));"
                f"}}"
                f'">'
                f'{MS_ICON_SVG}'
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
    spectrum_input_elem_id: str = "selected-spectrum-index",
) -> Tuple[
    gr.State,
    gr.State,
    gr.HTML,
    gr.Textbox,
]:
    source_dataset_state = gr.State(dataset)
    dataset_state = gr.State(dataset)

    selected_spectrum_index = gr.Textbox(
        value="-1",
        visible="hidden",
        elem_id=spectrum_input_elem_id,
    )

    dataframe = gr.HTML(
        value=render_dataset_html(
            dataset,
            page=page,
            rows_per_page=rows_per_page,
            height=height,
            show_index=show_index,
            spectrum_input_elem_id=spectrum_input_elem_id,
        )
    )

    spectrum_window = render_spectrum_window()
    selected_spectrum_index.change(
        fn=open_spectrum_window,
        inputs=[
            selected_spectrum_index,
            dataset_state,
        ],
        outputs=[
            spectrum_window.window.container,
            spectrum_window.spectrum_record,
        ],
    )

    return (
        source_dataset_state,
        dataset_state,
        dataframe,
        selected_spectrum_index,
    )