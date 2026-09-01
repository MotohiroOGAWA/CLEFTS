from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from ...common.file_upload_modal import FileUploadModal, render_file_upload_modal
from .....libs.msentity.msentity import *


@dataclass(frozen=True)
class MSDataUploadPanel:
    modal: FileUploadModal
    file_path: gr.State
    dataset_state: gr.State
    summary: gr.Markdown


def select_file(file_path: str | None) -> tuple[str, str]:
    """Store only the selected file path. Do not load the data yet."""
    if not file_path:
        return "", "No file selected."

    return (
        file_path,
        f"Selected `{Path(file_path).name}`. Click **Load data** to read the file.",
    )


def load_selected_file(file_path: str | None) -> tuple[Any, str, gr.update]:
    """Load the selected file as MSDataset and close the modal."""
    if not file_path:
        return (
            None,
            "No file selected.",
            gr.update(visible=False),
        )

    dataset = load_ms_data(file_path)

    return (
        dataset,
        f"Loaded {len(dataset)} records from `{Path(file_path).name}`.",
        gr.update(visible=False),
    )


def close_without_loading() -> tuple[None, str, gr.update]:
    """Close the modal and return empty dataset state."""
    return (
        None,
        "No file loaded.",
        gr.update(visible=False),
    )


def render_msdataset_upload_panel() -> MSDataUploadPanel:
    file_path = gr.State(value="")
    dataset_state = gr.State(value=None)

    summary = gr.Markdown("No file loaded.")

    modal = render_file_upload_modal(
        button_label="Select MS data",
        modal_title="Select MS data file",
        description="""
<div>
  Select a mass spectrometry data file to use in this application.

  <br><br>
  Supported formats:
  <ul>
    <li><b>.msp</b>: MSP spectral library file</li>
    <li><b>.mgf</b>: Mascot Generic Format file</li>
    <li><b>.hdf5 / .h5</b>: saved MSDataset file</li>
  </ul>

  The file is only selected at first. Click <b>Load data</b> to read it.
</div>
""",
        file_section_title="Select file",
        file_label="Drag and drop MS data file here",
        file_types=[".msp", ".mgf", ".hdf5", ".h5"],
        type="filepath",
        show_load_button=True,
        load_button_label="Load data",
    )

    modal.file.change(
        fn=select_file,
        inputs=[modal.file],
        outputs=[
            file_path,
            summary,
        ],
        show_progress="hidden",
    )

    if modal.load_button is None:
        raise RuntimeError("Load button was not created.")

    modal.load_button.click(
        fn=load_selected_file,
        inputs=[file_path],
        outputs=[
            dataset_state,
            summary,
            modal.modal,
        ],
    )

    modal.close_button.click(
        fn=close_without_loading,
        inputs=[],
        outputs=[
            dataset_state,
            summary,
            modal.modal,
        ],
        show_progress="hidden",
    )

    return MSDataUploadPanel(
        modal=modal,
        file_path=file_path,
        dataset_state=dataset_state,
        summary=summary,
    )