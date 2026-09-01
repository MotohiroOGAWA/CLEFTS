from __future__ import annotations

from gradio.context import Context


def register_styles(*styles: str) -> None:
    """Attach component styles to the root Blocks once per rendered app."""
    root_block = Context.root_block
    if root_block is None:
        raise RuntimeError("Component styles must be registered inside gr.Blocks.")

    registered_styles = getattr(root_block, "_clefts_registered_styles", set())
    for style in styles:
        if style and style not in registered_styles:
            root_block.css = "\n\n".join(filter(None, [root_block.css, style]))
            registered_styles.add(style)

    root_block._clefts_registered_styles = registered_styles
