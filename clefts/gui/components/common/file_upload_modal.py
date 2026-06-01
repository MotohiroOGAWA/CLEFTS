from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gradio as gr


@dataclass(frozen=True)
class FileUploadModal:
    open_button: gr.Button
    modal: gr.Column
    file: gr.File
    close_button: gr.Button
    load_button: gr.Button | None = None


def render_file_upload_modal(
    *,
    button_label: str = "Upload File",
    modal_title: str = "Upload File",
    description: str = "Upload your file here.",
    file_section_title: str = "Upload file",
    file_label: str = "Drag and drop file here",
    file_types: Sequence[str] | None = None,
    type: str = "filepath",
    elem_id: str = "clefts-file-upload-modal",
    show_load_button: bool = False,
    load_button_label: str = "Load",
    close_button_label: str = "Cancel",
) -> FileUploadModal:
    open_button = gr.Button(
        button_label,
        size="sm",
        elem_classes=["clefts-file-upload-modal-open-button"],
    )

    with gr.Column(
        visible=False,
        elem_id=elem_id,
        elem_classes=["clefts-file-upload-modal-backdrop"],
    ) as modal:
        with gr.Group(elem_classes=["clefts-file-upload-modal-dialog"]):
            with gr.Row(elem_classes=["clefts-file-upload-modal-header"]):
                gr.HTML(
                    f'<div class="clefts-file-upload-modal-title">{modal_title}</div>'
                )

            with gr.Column(elem_classes=["clefts-file-upload-modal-body"]):
                gr.HTML(
                    f'<div class="clefts-file-upload-modal-description">'
                    f"{description}"
                    f"</div>"
                )

                gr.HTML(
                    f'<div class="clefts-file-upload-modal-section-title">'
                    f"{file_section_title}"
                    f"</div>"
                )

                file = gr.File(
                    label=file_label,
                    file_types=list(file_types) if file_types is not None else None,
                    type=type,
                    elem_classes=["clefts-file-upload-modal-file"],
                )

            with gr.Row(elem_classes=["clefts-file-upload-modal-footer"]):
                close_button = gr.Button(
                    close_button_label,
                    size="sm",
                    variant="secondary",
                    elem_classes=["clefts-file-upload-modal-close-button"],
                )

                load_button = None
                if show_load_button:
                    load_button = gr.Button(
                        load_button_label,
                        size="sm",
                        variant="primary",
                        elem_classes=["clefts-file-upload-modal-load-button"],
                    )

    open_button.click(
        fn=lambda: gr.update(visible=True),
        outputs=[modal],
        show_progress="hidden",
    )

    close_button.click(
        fn=lambda: gr.update(visible=False),
        outputs=[modal],
        show_progress="hidden",
    )

    return FileUploadModal(
        open_button=open_button,
        modal=modal,
        file=file,
        close_button=close_button,
        load_button=load_button,
    )