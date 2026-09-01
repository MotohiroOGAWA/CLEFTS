from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

def _positive_limit_value(value: int) -> int:
    return max(1, int(value))


def _effective_limit(enabled: bool, value: int | float) -> int:
    if not enabled:
        return -1
    return _positive_limit_value(int(value))


def _update_limit_enabled(enabled: bool, value: int | float):
    return gr.update(interactive=enabled), _effective_limit(enabled, value)


def _update_limit_value(enabled: bool, value: int | float) -> int:
    return _effective_limit(enabled, value)


@dataclass(frozen=True)
class FragmentTreeBuilderInputPanel:
    max_depth: gr.components.Component
    only_add_min_depth: gr.components.Component
    min_depth_only_from: gr.components.Component
    max_node: gr.components.Component
    max_edge: gr.components.Component

    @property
    def components(self) -> list[gr.components.Component]:
        return [
            self.max_depth,
            self.only_add_min_depth,
            self.min_depth_only_from,
            self.max_node,
            self.max_edge,
        ]


def render_fragment_tree_builder_input_panel(
    *,
    default_max_depth: int = 2,
    default_only_add_min_depth: bool = True,
    default_min_depth_only_from: int = 0,
    default_max_node: int = -1,
    default_max_edge: int = -1,
    show_max_depth: bool = True,
    show_only_add_min_depth: bool = True,
    show_min_depth_only_from: bool = False,
    show_max_node: bool = True,
    show_max_edge: bool = True,
) -> FragmentTreeBuilderInputPanel:
    if show_max_depth:
        max_depth = gr.Number(
            label="max_depth",
            value=int(default_max_depth),
            precision=0,
            minimum=1,
        )
    else:
        max_depth = gr.State(int(default_max_depth))

    if show_only_add_min_depth:
        only_add_min_depth = gr.Checkbox(
            label="only_add_min_depth",
            value=bool(default_only_add_min_depth),
        )
    else:
        only_add_min_depth = gr.State(bool(default_only_add_min_depth))

    if show_min_depth_only_from:
        min_depth_only_from = gr.Number(
            label="min_depth_only_from",
            value=int(default_min_depth_only_from),
            precision=0,
            minimum=0,
        )
    else:
        min_depth_only_from = gr.State(int(default_min_depth_only_from))

    max_node = _render_optional_positive_limit(
        label="max_node",
        default_value=int(default_max_node),
        visible=show_max_node,
    )
    max_edge = _render_optional_positive_limit(
        label="max_edge",
        default_value=int(default_max_edge),
        visible=show_max_edge,
    )

    return FragmentTreeBuilderInputPanel(
        max_depth=max_depth,
        only_add_min_depth=only_add_min_depth,
        min_depth_only_from=min_depth_only_from,
        max_node=max_node,
        max_edge=max_edge,
    )


def _render_optional_positive_limit(
    *,
    label: str,
    default_value: int,
    visible: bool,
) -> gr.components.Component:
    if not visible:
        return gr.State(default_value if default_value > 0 else -1)

    enabled_default = default_value > 0
    number_default = _positive_limit_value(default_value if enabled_default else 1)
    effective_value = default_value if enabled_default else -1

    with gr.Row(elem_classes="clefts-limit-row"):
        number = gr.Number(
            label=label,
            value=number_default,
            precision=0,
            minimum=1,
            interactive=enabled_default,
            scale=4,
        )
        enabled = gr.Checkbox(
            label="limit",
            value=enabled_default,
            scale=1,
            elem_classes="clefts-limit-toggle",
        )

    state = gr.State(effective_value)
    enabled.change(
        fn=_update_limit_enabled,
        inputs=[enabled, number],
        outputs=[number, state],
    )
    number.change(
        fn=_update_limit_value,
        inputs=[enabled, number],
        outputs=[state],
    )
    return state
