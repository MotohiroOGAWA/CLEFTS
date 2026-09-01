from __future__ import annotations

from .metadata import WORKFLOW_BASE_PATH, WORKFLOW_TITLE


def render_workflow_item_html() -> str:
    return f"""
    <div class="clefts-tool-grid">
        <article class="clefts-tool-item">
            <h3>
                <a href="{WORKFLOW_BASE_PATH}">
                    {WORKFLOW_TITLE}
                </a>
            </h3>
            <ul>
                <li>Load MS datasets with msentity</li>
                <li>Inspect spectrum metadata and peak data</li>
                <li>Filter, sort, and export spectral records</li>
            </ul>
        </article>
    </div>
    """
