FILE_UPLOAD_MODAL_CSS = """
.clefts-file-upload-modal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 1000;

    background: rgba(0, 0, 0, 0.35);

    display: flex;
    align-items: center;
    justify-content: center;
}

/* Important: force hidden Gradio components to disappear completely */
.clefts-file-upload-modal-backdrop.hide,
.clefts-file-upload-modal-backdrop[hidden],
.clefts-file-upload-modal-backdrop[style*="display: none"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

.clefts-file-upload-modal-dialog {
    width: min(720px, 92vw);
    max-height: 86vh;
    overflow-y: auto;

    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;

    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
}

.clefts-file-upload-modal-header {
    padding: 16px 22px 12px 22px;
    border-bottom: 1px solid #eeeeee;
}

.clefts-file-upload-modal-title {
    font-size: 21px;
    font-weight: 600;
    color: #1f2937;
}

.clefts-file-upload-modal-body {
    padding: 16px 22px;
    gap: 10px;
}

.clefts-file-upload-modal-description {
    font-size: 14px;
    line-height: 1.55;
    color: #374151;
}

.clefts-file-upload-modal-description ul {
    margin: 6px 0 0 18px;
    padding: 0;
}

.clefts-file-upload-modal-description li {
    margin: 2px 0;
}

.clefts-file-upload-modal-section-title {
    margin-top: 6px;
    font-size: 15px;
    font-weight: 600;
    color: #111827;
}

.clefts-file-upload-modal-file {
    border: 1px dashed #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 10px !important;
    background: #f9fafb !important;
}

.clefts-file-upload-modal-footer {
    padding: 12px 22px 16px 22px;
    border-top: 1px solid #eeeeee;

    display: flex;
    justify-content: flex-end;
    gap: 10px;
}

.clefts-file-upload-modal-footer button {
    width: auto !important;
    min-width: 96px;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.clefts-file-upload-modal-close-button {
    background: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
}

.clefts-file-upload-modal-load-button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
}
"""
