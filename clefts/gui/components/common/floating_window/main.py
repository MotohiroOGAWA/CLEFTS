from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
import gradio as gr

from ...style_registry import register_styles
from .styles import FLOATING_WINDOW_CSS

FLOATING_WINDOW_SCRIPT = """
<script>
document.addEventListener("mousedown", function(event) {
    const handle = event.target.closest(".clefts-floating-window-header");
    if (!handle) return;

    if (event.target.closest(".clefts-floating-window-close")) {
        return;
    }

    const windowElement = handle.closest(".clefts-floating-window");
    if (!windowElement) return;

    const rect = windowElement.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    const offsetY = event.clientY - rect.top;

    function onMouseMove(moveEvent) {
        windowElement.style.left = `${moveEvent.clientX - offsetX}px`;
        windowElement.style.top = `${moveEvent.clientY - offsetY}px`;
        windowElement.style.right = "auto";
        windowElement.style.bottom = "auto";
    }

    function onMouseUp() {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
    }

    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
});
</script>
"""

@dataclass(frozen=True)
class FloatingWindow:
    container: gr.Group
    body: gr.Group
    close_button: gr.Button

    def __enter__(self):
        self.body.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.body.__exit__(exc_type, exc_value, traceback)


def render_floating_window(
    *,
    label: str,
    visible: bool = False,
    elem_classes: list[str] | None = None,
) -> FloatingWindow:
    register_styles(FLOATING_WINDOW_CSS)

    classes = ["clefts-floating-window"]
    if elem_classes:
        classes.extend(elem_classes)

    with gr.Group(visible=visible, elem_classes=classes) as container:
        with gr.Row(elem_classes=["clefts-floating-window-header"]):
            gr.HTML(
                f'<div class="clefts-floating-window-title">{label}</div>'
            )
            close_button = gr.Button(
                "×",
                size="sm",
                elem_classes=["clefts-floating-window-close"],
            )

        with gr.Group(elem_classes=["clefts-floating-window-body"]) as body:
            pass

    close_button.click(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=container,
    )

    return FloatingWindow(
        container=container,
        body=body,
        close_button=close_button,
    )

# python -m clefts.gui.components.common.floating_window.main
if __name__ == "__main__":
    with gr.Blocks(head=FLOATING_WINDOW_SCRIPT) as demo:
        open_button = gr.Button("Open Window")

        with render_floating_window(
            label="Spectrum Viewer",
            visible=False,
        ) as window:
            gr.Markdown("This is a floating window.")
            gr.Textbox(label="Example")

        open_button.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=window.container,
        )

    demo.launch()