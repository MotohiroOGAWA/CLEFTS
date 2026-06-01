from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gradio as gr

ICON_DIR = Path(__file__).resolve().parent.parent / "icons"


@dataclass(frozen=True)
class ActionButtons:
    copy_button: gr.Button | None = None
    download_button: gr.DownloadButton | None = None
    upload_button: gr.UploadButton | None = None


def render_action_buttons(
    *,
    show_copy: bool = True,
    show_download: bool = True,
    show_upload: bool = True,
    elem_id_prefix: str = "clefts-action",
    elem_classes: str = "clefts-generic-table-icon-button",
    upload_file_types: list[str] | None = None,
) -> ActionButtons:
    copy_button: gr.Button | None = None
    download_button: gr.DownloadButton | None = None
    upload_button: gr.UploadButton | None = None

    with gr.Row(elem_classes="clefts-action-buttons"):
        if show_copy:
            copy_button = gr.Button(
                "",
                size="sm",
                icon=str(ICON_DIR / "copy.svg"),
                elem_id=f"{elem_id_prefix}-copy",
                elem_classes=elem_classes,
                min_width=30,
                scale=0,
            )

        if show_download:
            download_button = gr.DownloadButton(
                "",
                size="sm",
                icon=str(ICON_DIR / "download.svg"),
                elem_id=f"{elem_id_prefix}-download",
                elem_classes=elem_classes,
                min_width=30,
                scale=0,
            )

        if show_upload:
            upload_button = gr.UploadButton(
                "",
                size="sm",
                icon=str(ICON_DIR / "upload.svg"),
                type="filepath",
                file_types=upload_file_types,
                elem_id=f"{elem_id_prefix}-upload",
                elem_classes=elem_classes,
                min_width=30,
                scale=0,
            )

    return ActionButtons(
        copy_button=copy_button,
        download_button=download_button,
        upload_button=upload_button,
    )