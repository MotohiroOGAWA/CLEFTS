from __future__ import annotations

import gradio as gr
from pathlib import Path
import tempfile


from ...common.action_buttons import render_action_buttons
from .....libs.msentity.msentity import MSDataset

def export_dataset_to_hdf5(
    dataset: MSDataset | None,
) -> str | None:
    if dataset is None:
        return None

    filepath = Path(tempfile.mkdtemp()) / "msdataset.hdf5"

    dataset.to_hdf5(filepath)

    return str(filepath)

def cleanup_download(
    download_data: gr.DownloadData,
) -> None:
    if download_data is None:
        return

    filepath = Path(download_data.file.path)
    print(f"Cleaning up downloaded file: {filepath}")

    if filepath.exists():
        print(f"Deleting file: {filepath}")
        filepath.unlink()

    parent_dir = filepath.parent

    try:
        parent_dir.rmdir()
    except OSError:
        pass

def render_download() -> gr.DownloadButton:
    buttons = render_action_buttons(
        show_copy=False,
        show_download=True,
        show_upload=False,
        elem_id_prefix="msdataset-table",
    )
    download_button = buttons.download_button

    return download_button