from __future__ import annotations

import base64
import io
from pathlib import Path
import tempfile

import gradio as gr

from ...common.action_buttons import render_action_buttons
from .....libs.msentity.msentity import MSDataset

MSDATASET_HDF5_DOWNLOAD_JS = """
(base64Value) => {
    console.log("base64 length:", base64Value ? base64Value.length : 0);

    if (!base64Value) {
        return "Dataset is empty.";
    }

    const binaryString = atob(base64Value);
    const bytes = new Uint8Array(binaryString.length);

    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob(
        [bytes],
        { type: "application/octet-stream" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "msdataset.hdf5";
    link.style.display = "none";

    document.body.appendChild(link);
    link.click();

    setTimeout(() => {
        link.remove();
        URL.revokeObjectURL(url);
    }, 1000);

    return "";
}
"""

def export_dataset_to_hdf5_base64(
    dataset: MSDataset | None,
) -> str:
    if dataset is None:
        return ""
    buffer = io.BytesIO()

    try:
        dataset.to_hdf5(buffer)
        buffer.seek(0)
        data = buffer.read()

    except Exception as e:
        print(f"Failed to export dataset to HDF5 in-memory. Falling back to temporary file. Error: {e}")
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "msdataset.hdf5"
            dataset.to_hdf5(filepath)
            data = filepath.read_bytes()

    print(f"Exported dataset to HDF5, size: {len(data)} bytes")
    return base64.b64encode(data).decode("ascii")


def render_download() -> tuple[gr.Button, gr.State, gr.Textbox]:
    buttons = render_action_buttons(
        show_copy=False,
        show_download=True,
        show_upload=False,
        elem_id_prefix="msdataset-table",
    )

    download_button = buttons.download_button
    exported_hdf5_base64 = gr.Textbox(visible=False)
    error_message = gr.Textbox(visible=False)
    # exported_hdf5_base64 = gr.State("")
    # error_message = gr.Textbox(visible=False)

    return download_button, exported_hdf5_base64, error_message