from __future__ import annotations

import html


def render_error_html(message: str = "") -> str:
    if not message:
        return ""
    return f'<div class="clefts-error-message">{html.escape(message)}</div>'
