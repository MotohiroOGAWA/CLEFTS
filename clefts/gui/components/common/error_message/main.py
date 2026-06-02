from __future__ import annotations

import html


def render_error_html(
    message: str = "",
    *,
    duration_seconds: float | int | None = 3,
) -> str:
    if not message:
        return ""

    duration_style = ""

    if duration_seconds is not None:
        duration_style = (
            f'--clefts-error-duration:{float(duration_seconds)}s;'
        )

    return f"""
<div
    class="clefts-generic-error-toast"
    style="{duration_style}"
>
    {html.escape(message)}
</div>
"""