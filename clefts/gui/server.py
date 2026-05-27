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

from clefts.gui.app import create_app as create_home_app
from clefts.gui.workflows.build_fragment_tree_from_smiles.page import (
    create_input_app as create_build_fragment_tree_from_smiles_input_app,
    create_result_app as create_build_fragment_tree_from_smiles_result_app,
)
from .components.styles import STYLES as COMPONENT_STYLES

APP_CSS = """
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
.clefts-pattern-panel {
    position: relative;
}
.clefts-pattern-panel > div:not(.clefts-pattern-dropzone) {
    position: relative;
    z-index: 1;
}
.clefts-pattern-toolbar {
    align-items: center !important;
    display: flex;
    gap: 8px;
    justify-content: flex-end;
    margin-bottom: 0;
    min-height: 30px;
    position: relative;
    width: 100%;
    z-index: 3;
}
.clefts-pattern-title-column,
.clefts-pattern-actions-column {
    align-self: center !important;
    min-height: 30px !important;
}
.clefts-pattern-title-column > div,
.clefts-pattern-actions-column > div {
    min-height: 30px !important;
}
.clefts-pattern-heading {
    align-items: center;
    color: #174f78;
    display: flex;
    font-size: 16px;
    font-weight: 700;
    height: 30px;
    line-height: 1;
}
.clefts-pattern-title-column .prose,
.clefts-pattern-title-column p {
    margin: 0 !important;
}
.clefts-pattern-actions {
    align-items: center !important;
    display: flex;
    gap: 8px;
    height: 30px;
    justify-content: flex-end;
    min-height: 30px !important;
}
.clefts-pattern-actions > div {
    align-self: center !important;
}
.clefts-pattern-panel .clefts-pattern-toolbar,
.clefts-pattern-panel .clefts-pattern-toolbar > div {
    padding-bottom: 0 !important;
}
.clefts-pattern-panel .clefts-pattern-toolbar + div {
    margin-top: 0 !important;
}
.clefts-pattern-toolbar .clefts-icon-button {
    align-self: center !important;
    flex: 0 0 30px !important;
    height: 30px !important;
    max-height: 30px !important;
    max-width: 30px !important;
    min-height: 30px !important;
    min-width: 30px !important;
    width: 30px !important;
}
.clefts-pattern-toolbar .clefts-icon-button > div,
.clefts-pattern-toolbar .clefts-icon-button .wrap,
.clefts-pattern-toolbar .clefts-icon-button label {
    max-width: 30px !important;
    min-width: 30px !important;
    width: 30px !important;
}
.clefts-error-message {
    background: #fff3f3;
    border-left: 4px solid #c0392b;
    color: #8a1f16;
    font-size: 13px;
    margin-top: 10px;
    padding: 8px 10px;
}

.clefts-icon-button button {
    align-items: center !important;
    display: flex !important;
    border-color: #c8d8e2 !important;
    color: #174f78 !important;
    gap: 5px !important;
    height: 30px !important;
    justify-content: center !important;
    max-width: 30px !important;
    min-height: 30px !important;
    min-width: 30px !important;
    padding: 0 !important;
    width: 30px !important;
}
.clefts-icon-button img {
    display: block !important;
    height: 16px !important;
    margin: auto !important;
    object-fit: contain !important;
    width: 16px !important;
}
.clefts-pattern-dropzone {
    background: transparent !important;
    border: 0 !important;
    height: 100% !important;
    inset: 0;
    margin: 0 !important;
    opacity: 0;
    overflow: hidden;
    position: absolute !important;
    width: 100% !important;
    z-index: 2;
}
.clefts-pattern-dropzone > div,
.clefts-pattern-dropzone label,
.clefts-pattern-dropzone .wrap,
.clefts-pattern-dropzone input {
    cursor: default !important;
    height: 100% !important;
    min-height: 100% !important;
    width: 100% !important;
}
.clefts-toolbar-json {
    max-width: 260px;
}
.clefts-toolbar-json textarea,
.clefts-toolbar-json .cm-editor {
    max-height: 42px !important;
    overflow: hidden !important;
}
.clefts-toolbar-file {
    max-width: 190px;
    opacity: .86;
}
.clefts-pattern-table {
    border-collapse: collapse;
    width: 100%;
    background: #ffffff;
    border: 1px solid #d8e4eb;
}
.clefts-pattern-table th {
    background: #f1f7fb;
    color: #174f78;
    font-weight: 700;
    text-align: left;
    border-bottom: 1px solid #d8e4eb;
    padding: 9px 11px;
}
.clefts-pattern-table td {
    border-bottom: 1px solid #e7eef3;
    color: #2f3b45;
    padding: 9px 11px;
    vertical-align: top;
}
.clefts-pattern-table tbody tr:last-child td {
    border-bottom: 1px solid #d8e4eb;
}
.clefts-pattern-table th:last-child,
.clefts-pattern-delete-cell {
    text-align: center;
    width: 34px;
}
.clefts-pattern-delete-button {
    align-items: center;
    background: transparent;
    border: 0;
    color: #52616d;
    cursor: pointer;
    display: inline-flex;
    font-size: 18px;
    height: 24px;
    justify-content: center;
    line-height: 1;
    padding: 0;
    width: 24px;
}
.clefts-pattern-delete-button:hover {
    color: #c0392b;
}
.clefts-pattern-table code {
    color: #263238;
    white-space: normal;
    overflow-wrap: anywhere;
}
.clefts-empty-cell {
    color: #7b8794 !important;
    font-style: italic;
}
.clefts-modal-backdrop {
    align-items: flex-start;
    background: rgba(15, 36, 50, .28);
    inset: 0;
    padding-top: 92px;
    position: fixed !important;
    z-index: 80;
}
.clefts-modal-backdrop > div {
    margin: 0 auto !important;
    width: min(520px, calc(100vw - 36px)) !important;
}
.clefts-modal-dialog {
    background: #ffffff;
    border: 1px solid #c8d8e2;
    box-shadow: 0 22px 58px rgba(31, 95, 139, .28);
    padding: 18px;
    width: 100%;
}
.clefts-modal-title {
    color: #174f78;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 12px;
}
.clefts-modal-actions {
    justify-content: flex-end;
    gap: 8px;
    margin-top: 4px;
}
.clefts-modal-actions > div {
    flex: 0 0 auto !important;
}

.clefts-patterns-layout {
    align-items: start;
}
.clefts-patterns-sidecar {
    border-left: 1px solid #d8e4eb;
    padding-left: 16px;
}
.clefts-add-pattern-panel {
    border: 1px solid #d8e4eb;
    background: #ffffff;
    margin-top: 12px;
    padding: 12px 14px;
}
.clefts-add-pattern-panel .prose {
    margin-bottom: 6px !important;
}
.clefts-add-pattern-panel p {
    color: #174f78;
    font-weight: 700;
    margin: 0 !important;
}
.clefts-compact-file {
    opacity: .86;
}
.clefts-compact-file label {
    color: #52616d !important;
    font-size: 12px !important;
    font-weight: 400 !important;
}

.clefts-subtle-loader {
    align-items: end;
    opacity: .86;
}
.clefts-subtle-loader label,
.clefts-copy-json label {
    color: #52616d !important;
    font-size: 12px !important;
    font-weight: 400 !important;
}
.clefts-copy-json {
    opacity: .82;
}

.clefts-limit-row {
    align-items: end;
}
.clefts-limit-toggle {
    opacity: 0.72;
}
.clefts-limit-toggle label {
    font-size: 12px !important;
    color: #52616d !important;
    font-weight: 400 !important;
}
.clefts-limit-toggle input {
    transform: scale(.85);
}

.clefts-dataframe-pager {
    align-items: center !important;
    display: grid !important;
    grid-template-columns: 1fr auto 1fr !important;
    gap: 10px !important;
    margin-top: 8px !important;
    width: 100% !important;
}
.clefts-dataframe-page-controls {
    align-items: center !important;
    display: inline-flex !important;
    gap: 8px !important;
    grid-column: 2 !important;
    justify-content: center !important;
}
.clefts-dataframe-page-controls > div {
    flex: 0 0 auto !important;
}
.clefts-dataframe-rows-control {
    align-items: center !important;
    display: inline-flex !important;
    gap: 6px !important;
    grid-column: 3 !important;
    justify-self: end !important;
}
.clefts-dataframe-page-number input {
    text-align: center !important;
}
.clefts-dataframe-page-count {
    color: #52616d;
    display: inline-flex;
    min-width: 44px;
}
.clefts-dataframe-rows-label {
    color: #52616d;
    font-size: 13px;
    white-space: nowrap;
}
.clefts-dataframe-rows-per-page {
    min-width: 88px !important;
}
.clefts-dataframe-sort-column {
    min-width: 132px !important;
}
.clefts-dataframe-sort-order {
    min-width: 116px !important;
}

.clefts-form-panel button.primary {
    background: #1f5f8b !important;
    border-color: #1f5f8b !important;
}
"""

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

APP_CSS = '\n\n'.join([APP_LAYOUT_STYLES, COMPONENT_STYLES])

def create_server() -> FastAPI:
    app = FastAPI()
    @app.get("/build_fragment_tree_from_smiles")
    @app.get("/build_fragment_tree_from_smiles/")
    def redirect_build_fragment_tree_from_smiles():
        return RedirectResponse(url="/build_fragment_tree_from_smiles/input/")

    gr.mount_gradio_app(
        app,
        create_build_fragment_tree_from_smiles_input_app(),
        path="/build_fragment_tree_from_smiles/input",
        css=APP_CSS,
        allowed_paths=[str(GUI_ROOT)],
    )
    gr.mount_gradio_app(
        app,
        create_build_fragment_tree_from_smiles_result_app(),
        path="/build_fragment_tree_from_smiles/result",
        css=APP_CSS,
        allowed_paths=[str(GUI_ROOT)],
    )
    gr.mount_gradio_app(
        app,
        create_home_app(),
        path="/",
        css=APP_CSS,
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
