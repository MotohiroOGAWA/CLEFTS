from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_spectrum_figure(
    peaks: pd.DataFrame,
    *,
    selected_peak_id: int | None = None,
    hovered_peak_id: int | None = None,
) -> go.Figure:
    fig = go.Figure()

    for _, row in peaks.iterrows():
        peak_id = int(row["peak_id"])
        mz = float(row["mz"])
        intensity = float(row["intensity"])

        is_selected = selected_peak_id == peak_id
        is_hovered = hovered_peak_id == peak_id

        if is_selected:
            color = "red"
            width = 4
        elif is_hovered:
            color = "orange"
            width = 4
        else:
            color = "black"
            width = 2

        fig.add_trace(
            go.Scatter(
                x=[mz, mz],
                y=[0, intensity],
                mode="lines",
                line={
                    "color": color,
                    "width": width,
                },
                customdata=[
                    [peak_id, mz, intensity],
                    [peak_id, mz, intensity],
                ],
                hovertemplate=(
                    "Peak %{customdata[0]}<br>"
                    "<i>m/z</i>: %{customdata[1]:.5f}<br>"
                    "Intensity: %{customdata[2]:.5f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[mz],
                y=[intensity],
                mode="markers",
                marker={
                    "size": 14,
                    "opacity": 0,
                    "color": "rgba(0,0,0,0)",
                },
                customdata=[[peak_id, mz, intensity]],
                hovertemplate=(
                    "Peak %{customdata[0]}<br>"
                    "<i>m/z</i>: %{customdata[1]:.5f}<br>"
                    "Intensity: %{customdata[2]:.5f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    max_mz = float(peaks["mz"].max()) if len(peaks) else 1.0
    max_intensity = float(peaks["intensity"].max()) if len(peaks) else 1.0

    fig.update_layout(
        autosize=True,
        margin={"l": 50, "r": 20, "t": 30, "b": 55},
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={
            "title": {"text": "<i>m/z</i>"},
            "range": [0, max_mz * 1.05],
            "showline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": True,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": "Intensity"},
            "range": [0, max_intensity * 1.05],
            "showline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": True,
            "zeroline": False,
        },
    )

    return fig


def extract_peak_id_from_plot_event(evt) -> int | None:
    point = getattr(evt, "point", None)

    if point is None:
        return None

    customdata = getattr(point, "customdata", None)

    if customdata is None:
        return None

    if isinstance(customdata, dict):
        return int(customdata["peak_id"])

    return int(customdata[0])