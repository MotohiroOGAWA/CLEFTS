FLOATING_WINDOW_CSS = """
.clefts-floating-window {
    position: fixed;
    right: 24px;
    bottom: 24px;

    width: fit-content;
    height: fit-content;
    min-width: 320px;
    max-width: 95vw;
    max-height: 90vh;

    resize: both;
    overflow: hidden;

    z-index: 1000;

    background: #ffffff !important;
    border: 1px solid #b7cff7 !important;
    border-radius: 8px !important;

    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}

.clefts-floating-window-header {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;

    height: 36px !important;
    min-height: 36px !important;
    padding: 4px 8px 4px 12px !important;

    background: #e8f1ff !important;
    border-bottom: 1px solid #b7cff7 !important;

    cursor: move;
    user-select: none;
}

/* Gradio が HTML/Button を wrapper div で包むため、その wrapper を無効化 */
.clefts-floating-window-header > div {
    display: contents !important;
}

.clefts-floating-window-title {
    flex: 1 1 auto !important;
    min-width: 0 !important;

    margin: 0 !important;

    font-size: 14px;
    font-weight: 600;
    line-height: 28px;
    color: #1f2937;

    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.clefts-floating-window-close {
    flex: 0 0 28px !important;

    width: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;

    height: 28px !important;
    min-height: 28px !important;
    max-height: 28px !important;

    padding: 0 !important;
    margin: 0 !important;

    background: #ffe5e5 !important;
    border: 1px solid #f1a4a4 !important;
    border-radius: 4px !important;

    color: #b91c1c !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    line-height: 1 !important;

    cursor: pointer !important;
}

.clefts-floating-window-close:hover {
    background: #ffd0d0 !important;
}

.clefts-floating-window-body {
    background: #ffffff !important;
    padding: 12px !important;
    overflow: auto;
}
"""