from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Sequence

import gradio as gr
from gradio.context import Context
from gradio_interactiveplot import InteractivePlot
from gradio_interactiveplot.events import PlotClickData, PlotHoverData, PlotRelayoutData
import pandas as pd

from .....libs.msentity.msentity import SpectrumRecord

from .peak_table import (
    extract_peak_id_from_table_event,
    make_peak_dataframe,
    make_peak_table,
)
from .spectrum_plot import (
    extract_peak_id_from_plot_event,
    make_spectrum_figure,
)
from .styles import MASS_SPECTRUM_VIEW_CSS


@dataclass(frozen=True)
class MassSpectrumView:
    plot: InteractivePlot
    table: gr.DataFrame
    peaks_state: gr.State
    selected_peak_id: gr.State
    hovered_peak_id: gr.State

MassSpectrumOutputs = Tuple[
    Any,                 # InteractivePlot value
    Any,                 # gr.DataFrame value
    pd.DataFrame,        # peaks_state
    int | None,          # selected_peak_id
    int | None,          # hovered_peak_id
]


def make_mass_spectrum_outputs_from_peaks(
    peaks: pd.DataFrame,
    *,
    selected_peak_id: int | None = None,
    hovered_peak_id: int | None = None,
) -> MassSpectrumOutputs:
    return (
        make_spectrum_figure(
            peaks,
            selected_peak_id=selected_peak_id,
            hovered_peak_id=hovered_peak_id,
        ),
        make_peak_table(
            peaks,
            selected_peak_id=selected_peak_id,
            hovered_peak_id=hovered_peak_id,
        ),
        peaks,
        selected_peak_id,
        hovered_peak_id,
    )


def make_mass_spectrum_outputs_from_mz_intensity(
    mz: Sequence[float],
    intensity: Sequence[float],
    *,
    selected_peak_id: int | None = None,
    hovered_peak_id: int | None = None,
) -> MassSpectrumOutputs:
    peaks = make_peak_dataframe(
        list(mz),
        list(intensity),
    )

    return make_mass_spectrum_outputs_from_peaks(
        peaks,
        selected_peak_id=selected_peak_id,
        hovered_peak_id=hovered_peak_id,
    )


def make_mass_spectrum_outputs_from_record(
    record: SpectrumRecord | None,
    *,
    selected_peak_id: int | None = None,
    hovered_peak_id: int | None = None,
) -> MassSpectrumOutputs:
    if record is None:
        return make_empty_mass_spectrum_outputs()

    return make_mass_spectrum_outputs_from_mz_intensity(
        mz=record.spectrum.mz.tolist(),
        intensity=record.spectrum.intensity.tolist(),
        selected_peak_id=selected_peak_id,
        hovered_peak_id=hovered_peak_id,
    )


def make_empty_mass_spectrum_outputs() -> MassSpectrumOutputs:
    return make_mass_spectrum_outputs_from_mz_intensity(
        mz=[],
        intensity=[],
        selected_peak_id=None,
        hovered_peak_id=None,
    )

def register_styles(*styles: str) -> None:
    root_block = Context.root_block

    if root_block is None:
        raise RuntimeError(
            "Styles must be registered inside gr.Blocks()."
        )

    registered_styles = getattr(
        root_block,
        "_clefts_registered_styles",
        set(),
    )

    for style in styles:
        if style and style not in registered_styles:
            root_block.css = "\n\n".join(
                filter(None, [root_block.css, style])
            )
            registered_styles.add(style)

    root_block._clefts_registered_styles = registered_styles


def empty_peak_dataframe():
    return make_peak_dataframe([], [])


def record_to_peak_dataframe(record: SpectrumRecord):
    return make_peak_dataframe(
        mz=record.spectrum.mz.tolist(),
        intensity=record.spectrum.intensity.tolist(),
    )


def render_mass_spectrum_view() -> MassSpectrumView:
    register_styles(MASS_SPECTRUM_VIEW_CSS)

    initial_peaks = empty_peak_dataframe()

    peaks_state = gr.State(initial_peaks)
    selected_peak_id = gr.State(None)
    hovered_peak_id = gr.State(None)

    with gr.Row(elem_classes=["clefts-spectrum-view"]):
        with gr.Column(
            scale=3,
            min_width=520,
            elem_classes=["clefts-spectrum-plot-column"],
        ):
            spectrum_plot = InteractivePlot(
                value=make_spectrum_figure(initial_peaks),
                elem_classes=["clefts-spectrum-plot"],
            )

        with gr.Column(
            scale=1,
            min_width=300,
            elem_classes=["clefts-spectrum-table-column"],
        ):
            peak_table = gr.DataFrame(
                value=make_peak_table(initial_peaks),
                interactive=False,
                label="Peaks",
                elem_classes=["clefts-spectrum-table"],
            )

    def on_plot_click(
        evt: PlotClickData,
        peaks,
        selected: int | None,
        hovered: int | None,
    ):
        peak_id = extract_peak_id_from_plot_event(evt)

        return make_mass_spectrum_outputs_from_peaks(
            peaks,
            selected_peak_id=peak_id,
            hovered_peak_id=hovered,
        )

    def on_plot_hover(
        evt: PlotHoverData,
        peaks,
        selected: int | None,
        hovered: int | None,
    ):
        peak_id = extract_peak_id_from_plot_event(evt)

        return make_mass_spectrum_outputs_from_peaks(
            peaks,
            selected_peak_id=selected,
            hovered_peak_id=peak_id,
        )

    def on_table_select(
        evt: gr.SelectData,
        peaks,
        selected: int | None,
        hovered: int | None,
    ):
        peak_id = extract_peak_id_from_table_event(peaks, evt)

        return make_mass_spectrum_outputs_from_peaks(
            peaks,
            selected_peak_id=peak_id,
            hovered_peak_id=hovered,
        )

    def on_relayout(
        evt: PlotRelayoutData,
        peaks,
        selected_peak_id,
        hovered_peak_id,
    ):
        if "xaxis.range[0]" in evt.payload and "xaxis.range[1]" in evt.payload:
            xmin = float(evt.payload["xaxis.range[0]"])
            xmax = float(evt.payload["xaxis.range[1]"])
        elif 'xaxis.showspikes' in evt.payload and evt.payload['xaxis.showspikes'] == False:
            xmin = 0.0
            xmax = float(peaks["mz"].max()) * 1.05 if len(peaks) else 1.0
        else:
            return gr.update()

        visible_peaks = peaks[
            (peaks["mz"] >= xmin)
            & (peaks["mz"] <= xmax)
        ]

        if len(visible_peaks):
            y_max = (
                float(
                    visible_peaks["intensity"].max()
                )
                * 1.05
            )
        else:
            y_max = 1.0

        return (
            make_spectrum_figure(
                peaks,
                selected_peak_id=selected_peak_id,
                hovered_peak_id=hovered_peak_id,
                x_min=xmin,
                x_max=xmax,
                y_min=0.0,
                y_max=y_max,
            ),
            make_peak_table(
                peaks,
                selected_peak_id=selected_peak_id,
                hovered_peak_id=hovered_peak_id,
            ),
            peaks,
            selected_peak_id,
            hovered_peak_id,
        )

    spectrum_plot.click(
        fn=on_plot_click,
        inputs=[
            peaks_state,
            selected_peak_id,
            hovered_peak_id,
        ],
        outputs=[
            spectrum_plot,
            peak_table,
            peaks_state,
            selected_peak_id,
            hovered_peak_id,
        ],
    )

    peak_table.select(
        fn=on_table_select,
        inputs=[
            peaks_state,
            selected_peak_id,
            hovered_peak_id,
        ],
        outputs=[
            spectrum_plot,
            peak_table,
            peaks_state,
            selected_peak_id,
            hovered_peak_id,
        ],
    )

    spectrum_plot.relayout(
        fn=on_relayout,
        inputs=[
            peaks_state,
            selected_peak_id,
            hovered_peak_id,
        ],
        outputs=[
            spectrum_plot,
            peak_table,
            peaks_state,
            selected_peak_id,
            hovered_peak_id,
        ],
    )

    return MassSpectrumView(
        plot=spectrum_plot,
        table=peak_table,
        peaks_state=peaks_state,
        selected_peak_id=selected_peak_id,
        hovered_peak_id=hovered_peak_id,
    )

# python -m clefts.gui.components.widgets.mass_spectrum_viewer.main
if __name__ == "__main__":
    demo_mz = [
        55.0542,
        69.0698,
        83.0855,
        97.0648,
        111.0804,
        125.0961,
        139.0754,
        153.0910,
        167.1067,
        181.0860,
        195.1016,
        209.1173,
        223.0966,
        237.1122,
        251.1279,
    ]

    demo_intensity = [
        0.08,
        0.18,
        0.52,
        0.21,
        0.87,
        0.35,
        0.15,
        1.00,
        0.42,
        0.19,
        0.68,
        0.31,
        0.11,
        0.25,
        0.09,
    ]

    def load_demo_spectrum():
        peaks = make_peak_dataframe(
            demo_mz,
            demo_intensity,
        )

        return make_mass_spectrum_outputs_from_peaks(peaks)

    with gr.Blocks(
        fill_height=True,
        css="""
        .demo-spectrum-parent {
            height: 1000px;
            border: 1px solid red;
        }
        """,
    ) as demo:

        load_button = gr.Button(
            "Load Demo Spectrum"
        )

        with gr.Column(
            elem_classes=["demo-spectrum-parent"],
        ):
            spectrum_view = render_mass_spectrum_view()

        load_button.click(
            fn=load_demo_spectrum,
            outputs=[
                spectrum_view.plot,
                spectrum_view.table,
                spectrum_view.peaks_state,
                spectrum_view.selected_peak_id,
                spectrum_view.hovered_peak_id,
            ],
        )

    demo.launch()