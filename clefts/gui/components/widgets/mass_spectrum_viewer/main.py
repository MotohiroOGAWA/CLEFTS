from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
from gradio.context import Context
from gradio_interactiveplot import InteractivePlot
from gradio_interactiveplot.events import PlotClickData, PlotHoverData

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
    selected_peak_id: gr.State
    hovered_peak_id: gr.State


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


def render_mass_spectrum_view(
    mz: list[float],
    intensity: list[float],
    *,
    label: str = "Mass spectrum",
) -> MassSpectrumView:
    register_styles(MASS_SPECTRUM_VIEW_CSS)

    peaks = make_peak_dataframe(mz, intensity)

    selected_peak_id = gr.State(None)
    hovered_peak_id = gr.State(None)

    with gr.Row(elem_classes=["clefts-spectrum-view"]):
        with gr.Column(
            scale=3, 
            min_width=520,
            elem_classes=["clefts-spectrum-plot-column"],
            ):
            spectrum_plot = InteractivePlot(
                value=make_spectrum_figure(peaks),
                label=label,
                elem_classes=["clefts-spectrum-plot"],
            )

        with gr.Column(
            scale=1, 
            min_width=300,
            elem_classes=["clefts-spectrum-table-column"],
            ):
            peak_table = gr.DataFrame(
                value=make_peak_table(peaks),
                interactive=False,
                label="Peaks",
                elem_classes=["clefts-spectrum-table"],
            )

    def update_view(
        selected: int | None,
        hovered: int | None,
    ):
        return (
            make_spectrum_figure(
                peaks,
                selected_peak_id=selected,
                hovered_peak_id=hovered,
            ),
            make_peak_table(
                peaks,
                selected_peak_id=selected,
                hovered_peak_id=hovered,
            ),
            selected,
            hovered,
        )

    def on_plot_click(
        evt: PlotClickData,
        selected: int | None,
        hovered: int | None,
    ):
        peak_id = extract_peak_id_from_plot_event(evt)
        return update_view(peak_id, hovered)

    def on_plot_hover(
        evt: PlotHoverData,
        selected: int | None,
        hovered: int | None,
    ):
        peak_id = extract_peak_id_from_plot_event(evt)
        return update_view(selected, peak_id)

    def on_table_select(
        evt: gr.SelectData,
        selected: int | None,
        hovered: int | None,
    ):
        peak_id = extract_peak_id_from_table_event(peaks, evt)
        return update_view(peak_id, hovered)

    spectrum_plot.click(
        fn=on_plot_click,
        inputs=[selected_peak_id, hovered_peak_id],
        outputs=[
            spectrum_plot,
            peak_table,
            selected_peak_id,
            hovered_peak_id,
        ],
    )

    peak_table.select(
        fn=on_table_select,
        inputs=[selected_peak_id, hovered_peak_id],
        outputs=[
            spectrum_plot,
            peak_table,
            selected_peak_id,
            hovered_peak_id,
        ],
    )

    return MassSpectrumView(
        plot=spectrum_plot,
        table=peak_table,
        selected_peak_id=selected_peak_id,
        hovered_peak_id=hovered_peak_id,
    )


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

    with gr.Blocks(
        fill_height=True,
        css="""
        .demo-spectrum-parent {
            height: 1000px;
            border: 1px solid red;
        }
        """
    ) as demo:

        with gr.Column(
            elem_classes=["demo-spectrum-parent"],
        ):

            render_mass_spectrum_view(
                demo_mz,
                demo_intensity,
                label="MS/MS spectrum",
            )

    demo.launch()