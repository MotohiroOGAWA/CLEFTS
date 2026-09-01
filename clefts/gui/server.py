from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse


APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

GUI_ROOT = Path(__file__).resolve().parent

from .app import create_app as create_home_app
from .workflows.msentity.apps import create_app as create_msentity_app
from .workflows.build_fragment_tree_from_smiles.page import (
    create_input_app as create_build_fragment_tree_from_smiles_input_app,
    create_result_app as create_build_fragment_tree_from_smiles_result_app,
)

APP_LAYOUT_STYLES = """
footer {
    display: none !important;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    background: #ffffff !important;
    color: #1f2933;
    font-family: Arial, Helvetica, sans-serif;
}

.clefts-page {
    box-sizing: border-box;
    width: 100%;
    padding: 24px clamp(18px, 4vw, 58px) 46px;
}

.clefts-hero {
    border-top: 5px solid #1f5f8b;
    background: linear-gradient(#f7fbfd, #ffffff);
    padding: 30px 4px 22px;
    border-bottom: 1px solid #d8e4eb;
}

.clefts-kicker {
    color: #c46a1a;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.clefts-title-block h1 {
    color: #174f78;
    font-size: 42px;
    margin: 0 0 8px;
    font-weight: 700;
}

.clefts-title-block p {
    max-width: 980px;
    color: #3e4c59;
    font-size: 16px;
    line-height: 1.55;
    margin: 0;
}

.clefts-section {
    padding-top: 28px;
}

.clefts-section h2,
.clefts-page-heading h1 {
    color: #174f78;
    font-size: 28px;
    margin: 0 0 8px;
    font-weight: 700;
}

.clefts-page-heading {
    margin: 14px 0 22px;
    border-bottom: 1px solid #d8e4eb;
    padding-bottom: 14px;
}

.clefts-page-heading p {
    color: #52616d;
    margin: 0;
}

.clefts-rule {
    height: 1px;
    background: #d8e4eb;
    margin: 10px 0 18px;
}

.clefts-tool-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 22px 42px;
    width: 100%;
}

.clefts-tool-item {
    border-left: 4px solid #e6a23c;
    padding: 2px 0 4px 14px;
}

.clefts-tool-item h3 {
    color: #174f78;
    font-size: 18px;
    margin: 0 0 8px;
}

.clefts-tool-item ul {
    margin: 0 0 10px 18px;
    padding: 0;
    color: #3e4c59;
    line-height: 1.45;
}

.clefts-link-button button {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: #0b67a3 !important;
    cursor: pointer !important;
    font-weight: 700 !important;
    justify-content: flex-start !important;
    padding: 0 !important;
    text-align: left !important;
    width: auto !important;
}

.clefts-link-button button:hover {
    color: #084f7d !important;
    text-decoration: underline !important;
}

.clefts-workflow-title button {
    font-size: 18px !important;
    line-height: 1.25 !important;
}

.clefts-back-link-button {
    margin-bottom: 14px;
}

.clefts-back-link-button button {
    font-size: 14px !important;
}

.clefts-tool-item a,
.clefts-back-link,
.clefts-link-nav a {
    color: #0b67a3;
    text-decoration: none;
    font-weight: 700;
}

.clefts-tool-item a:hover,
.clefts-back-link:hover,
.clefts-link-nav a:hover {
    text-decoration: underline;
}

.clefts-back-link {
    display: inline-block;
    margin-bottom: 14px;
    font-size: 14px;
}

.clefts-link-nav {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 14px;
    font-size: 14px;
}

.clefts-link-nav span {
    color: #9aa5b1;
}

.clefts-form-panel {
    border: 1px solid #d8e4eb;
    background: #fbfdfe;
    padding: 18px;
}

.clefts-form-panel button.primary {
    background: #1f5f8b !important;
    border-color: #1f5f8b !important;
}
"""

def _with_app_layout(blocks: gr.Blocks) -> gr.Blocks:
    blocks.css = "\n\n".join(filter(None, [APP_LAYOUT_STYLES, blocks.css]))
    blocks.config = blocks.get_config_file()
    return blocks

def create_server() -> FastAPI:
    app = FastAPI()
    @app.get("/build_fragment_tree_from_smiles")
    @app.get("/build_fragment_tree_from_smiles/")
    def redirect_build_fragment_tree_from_smiles():
        return RedirectResponse(url="/build_fragment_tree_from_smiles/input/")

    gr.mount_gradio_app(
        app,
        _with_app_layout(create_msentity_app()),
        path="/msentity",
        allowed_paths=[str(GUI_ROOT)],
    )

    gr.mount_gradio_app(
        app,
        _with_app_layout(create_build_fragment_tree_from_smiles_input_app()),
        path="/build_fragment_tree_from_smiles/input",
        allowed_paths=[str(GUI_ROOT)],
    )
    gr.mount_gradio_app(
        app,
        _with_app_layout(create_build_fragment_tree_from_smiles_result_app()),
        path="/build_fragment_tree_from_smiles/result",
        allowed_paths=[str(GUI_ROOT)],
    )
    gr.mount_gradio_app(
        app,
        _with_app_layout(create_home_app()),
        path="/",
        allowed_paths=[str(GUI_ROOT)],
    )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CLEFTS GUI app.")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(create_server(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
