from __future__ import annotations
from operator import index

from dataclasses import dataclass
import gradio as gr

from .....libs.msentity.msentity import MSDataset

from ...common.floating_window import render_floating_window, FloatingWindow
from ...widgets.mass_spectrum_viewer import render_mass_spectrum_view, make_mass_spectrum_outputs_from_record, MassSpectrumView

@dataclass(frozen=True)
class SpectrumWindow:
    window: FloatingWindow
    spectrum_record: gr.State

def open_spectrum_window(
    index_text: str,
    dataset: MSDataset | None,
):
    print("=" * 80)
    print("open_spectrum_window called")
    print("index_text =", repr(index_text))

    if dataset is None:
        print("dataset is None")
        return

    try:
        index = int(index_text)
    except ValueError:
        print("invalid index")
        return

    if index < 0:
        print("index is negative")
        return

    spectrum = dataset[index]

    return (
        gr.update(visible=True),
        spectrum,
        # "Spectrum: {0}".format(index),
    )



def render_spectrum_window() -> SpectrumWindow:
    spectrum_record = gr.State(None)

    with render_floating_window(
        label="MS/MS Spectrum Viewer",
        visible=False,
    ) as window:

        spectrum_view = render_mass_spectrum_view()

    spectrum_record.change(
        fn=make_mass_spectrum_outputs_from_record,
        inputs=[spectrum_record],
        outputs=[
            spectrum_view.plot,
            spectrum_view.table,
            spectrum_view.peaks_state,
            spectrum_view.selected_peak_id,
            spectrum_view.hovered_peak_id,
        ],
    )

    return SpectrumWindow(
        window=window,
        spectrum_record=spectrum_record,
    )