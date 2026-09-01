from __future__ import annotations

from .metadata import INPUT_PATH, WORKFLOW_TITLE


def render_workflow_item_html() -> str:
    return f"""
    <div class="clefts-tool-grid">
        <article class="clefts-tool-item">
            <h3>
                <a href="{INPUT_PATH}">
                    {WORKFLOW_TITLE}
                </a>
            </h3>
            <ul>
                <li>Start from a single molecular SMILES</li>
                <li>Configure fragmentation constraints</li>
                <li>Generate and inspect a fragment tree</li>
            </ul>
        </article>
    </div>
    """
