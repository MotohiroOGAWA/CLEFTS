import gradio as gr
import plotly.graph_objects as go

from gradio_interactiveplot import InteractivePlot
from gradio_interactiveplot.events import (
    PlotClickData,
    PlotDoubleClickData,
    PlotHoverData,
    PlotRelayoutData,
    PlotSelectData,
)


def make_scatter():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[100, 150, 200, 250],
        y=[0.2, 0.8, 0.5, 0.9],
        mode="markers+lines",
        name="Peaks",
        customdata=[
            {"peak_id": 0, "label": "peak_100"},
            {"peak_id": 1, "label": "peak_150"},
            {"peak_id": 2, "label": "peak_200"},
            {"peak_id": 3, "label": "peak_250"},
        ],
    ))
    fig.update_layout(title="Scatter", dragmode="select")
    return fig


def make_bar():
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["A", "B", "C", "D"],
        y=[10, 25, 15, 30],
        name="Bar",
        customdata=[
            {"group": "A"},
            {"group": "B"},
            {"group": "C"},
            {"group": "D"},
        ],
    ))
    fig.update_layout(title="Bar Plot")
    return fig


def make_line():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[2, 3, 2.5, 4, 3.5],
        mode="lines+markers",
        name="Line",
    ))
    fig.update_layout(title="Line Plot")
    return fig


def make_heatmap():
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[
            [1, 20, 30],
            [20, 1, 60],
            [30, 60, 1],
        ],
        x=["A", "B", "C"],
        y=["X", "Y", "Z"],
        name="Heatmap",
    ))
    fig.update_layout(title="Heatmap")
    return fig


def make_box():
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[1, 2, 2, 3, 4, 8],
        name="Box A",
    ))
    fig.add_trace(go.Box(
        y=[2, 3, 3, 4, 5, 9],
        name="Box B",
    ))
    fig.update_layout(title="Box Plot")
    return fig


def make_3d_scatter():
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[1, 2, 3, 4],
        y=[2, 1, 4, 3],
        z=[3, 5, 2, 6],
        mode="markers",
        name="3D points",
    ))
    fig.update_layout(title="3D Scatter")
    return fig


def event_to_dict(evt):
    return {
        "event_type": evt.type,
        "points": [point.to_dict() for point in getattr(evt, "points", [])],
        "point": None if getattr(evt, "point", None) is None else evt.point.to_dict(),
        "range": getattr(evt, "range", None),
        "lasso_points": getattr(evt, "lasso_points", None),
        "payload": getattr(evt, "payload", None),
    }


def on_click(evt: PlotClickData):
    return event_to_dict(evt)


def on_select(evt: PlotSelectData):
    return event_to_dict(evt)


def on_hover(evt: PlotHoverData):
    return event_to_dict(evt)


def on_relayout(evt: PlotRelayoutData):
    return event_to_dict(evt)


def on_double_click(evt: PlotDoubleClickData):
    return event_to_dict(evt)


def add_plot_tab(name: str, figure):
    with gr.Tab(name):
        plot = InteractivePlot(
            value=figure,
            label=name,
        )

        with gr.Row():
            clicked_json = gr.JSON(label="Click")
            selected_json = gr.JSON(label="Select")

        with gr.Row():
            hover_json = gr.JSON(label="Hover")
            relayout_json = gr.JSON(label="Relayout")

        double_click_json = gr.JSON(label="Double click")

        plot.click(fn=on_click, inputs=None, outputs=clicked_json)
        plot.select(fn=on_select, inputs=None, outputs=selected_json)
        plot.hover(fn=on_hover, inputs=None, outputs=hover_json)
        plot.relayout(fn=on_relayout, inputs=None, outputs=relayout_json)
        plot.double_click(fn=on_double_click, inputs=None, outputs=double_click_json)


with gr.Blocks() as demo:
    gr.Markdown("# InteractivePlot Test Gallery")

    with gr.Tabs():
        add_plot_tab("Scatter", make_scatter())
        add_plot_tab("Bar", make_bar())
        add_plot_tab("Line", make_line())
        add_plot_tab("Heatmap", make_heatmap())
        add_plot_tab("Box", make_box())
        add_plot_tab("3D Scatter", make_3d_scatter())


if __name__ == "__main__":
    demo.launch()