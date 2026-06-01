ERROR_MESSAGE_CSS = """
.clefts-generic-error-toast {
    position: fixed;

    top: 18px;
    right: 18px;

    z-index: 9999;

    background: #fff3f3;
    border-left: 4px solid #c0392b;

    color: #8a1f16;

    padding: 12px 14px;

    border-radius: 6px;

    box-shadow: 0 8px 24px rgba(0,0,0,.12);

    min-width: 240px;
    max-width: 420px;

    font-size: 14px;
    line-height: 1.4;

    animation:
        clefts-error-toast-in .16s ease-out,
        clefts-error-toast-out .18s ease-in
            var(--clefts-error-duration, 3s)
            forwards;
}

@keyframes clefts-error-toast-in {
    from {
        opacity: 0;
        transform: translateY(-8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes clefts-error-toast-out {
    to {
        opacity: 0;
        transform: translateY(-8px);
        visibility: hidden;
    }
}
"""

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

GENERIC_TABLE_INPUT_CSS = """
.clefts-generic-table-panel {
    position: relative;
}

.clefts-generic-table-toolbar {
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

.clefts-generic-table-title-column,
.clefts-generic-table-actions-column {
    align-self: center !important;
    min-height: 30px !important;
}

.clefts-generic-table-heading {
    align-items: center;
    color: #174f78;
    display: flex;
    font-size: 16px;
    font-weight: 700;
    height: 30px;
    line-height: 1;
}

.clefts-generic-table-actions {
    align-items: center !important;
    display: flex;
    gap: 8px;
    height: 30px;
    justify-content: flex-end;
    min-height: 30px !important;
}

.clefts-generic-table-icon-button {
    align-self: center !important;
    flex: 0 0 30px !important;
    height: 30px !important;
    max-height: 30px !important;
    max-width: 30px !important;
    min-height: 30px !important;
    min-width: 30px !important;
    width: 30px !important;
}

.clefts-generic-table-icon-button button {
    align-items: center !important;
    border-color: #c8d8e2 !important;
    color: #174f78 !important;
    display: flex !important;
    height: 30px !important;
    justify-content: center !important;
    max-width: 30px !important;
    min-height: 30px !important;
    min-width: 30px !important;
    padding: 0 !important;
    width: 30px !important;
}

.clefts-generic-table-icon-button img {
    display: block !important;
    height: 16px !important;
    margin: auto !important;
    object-fit: contain !important;
    width: 16px !important;
}

.clefts-generic-table {
    background: #ffffff;
    border: 1px solid #d8e4eb;
    border-collapse: collapse;
    width: 100%;
}

.clefts-generic-table th {
    background: #f1f7fb;
    border-bottom: 1px solid #d8e4eb;
    color: #174f78;
    font-weight: 700;
    padding: 9px 11px;
    text-align: left;
}

.clefts-generic-table td {
    border-bottom: 1px solid #e7eef3;
    color: #2f3b45;
    padding: 9px 11px;
    vertical-align: top;
}

.clefts-generic-table tbody tr:last-child td {
    border-bottom: 1px solid #d8e4eb;
}

.clefts-generic-table th:last-child,
.clefts-generic-table-delete-cell {
    text-align: center;
    width: 34px;
}

.clefts-generic-table-delete-button {
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

.clefts-generic-table-delete-button:hover {
    color: #c0392b;
}

.clefts-generic-table code {
    color: #263238;
    overflow-wrap: anywhere;
    white-space: normal;
}

.clefts-generic-table-empty-cell {
    color: #7b8794 !important;
    font-style: italic;
}

.clefts-generic-table-modal-backdrop {
    align-items: flex-start;
    background: rgba(15, 36, 50, .28);
    inset: 0;
    padding-top: 92px;
    position: fixed !important;
    z-index: 80;
}

.clefts-generic-table-modal-backdrop > div {
    margin: 0 auto !important;
    width: min(520px, calc(100vw - 36px)) !important;
}

.clefts-generic-table-modal-dialog {
    background: #ffffff;
    border: 1px solid #c8d8e2;
    box-shadow: 0 22px 58px rgba(31, 95, 139, .28);
    padding: 18px;
    width: 100%;
}

.clefts-generic-table-modal-title {
    color: #174f78;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 12px;
}

.clefts-generic-table-modal-actions {
    gap: 8px;
    justify-content: flex-end;
    margin-top: 4px;
}

.clefts-generic-table-modal-actions > div {
    flex: 0 0 auto !important;
}
"""

PAGINATED_DATAFRAME_CSS = """
.clefts-table-label {
    font-size: 14px;
    font-weight: 600;
    margin: 0 0 2px 0;
}

/* Remove outer HTML/component spacing as much as possible */
.clefts-html-table-wrap {
    margin: 0 !important;
    padding: 0 !important;
}

.clefts-html-table-wrap > div {
    margin: 0 !important;
    padding: 0 !important;
}

/* HTML table */
.clefts-html-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 0 !important;
}

.clefts-html-table th,
.clefts-html-table td {
    border: 1px solid #d8d8d8;
    padding: 4px 6px;
    text-align: left;
}

.clefts-html-table th {
    background: #f5f5f5;
    font-weight: 600;
}

.clefts-html-table tr:nth-child(even) {
    background: #fafafa;
}

.clefts-html-table tr:hover {
    background: #f0f7ff;
}

/* Pager */
.clefts-pager-row,
.clefts-sort-row {
    display: flex !important;
    align-items: center !important;
    gap: 3px !important;
    margin: 2px 0 0 0 !important;
    padding: 0 !important;
    flex-wrap: wrap !important;
}

.clefts-small-text {
    font-size: 12px;
    color: #52616d;
    white-space: nowrap;
    margin-left: 8px;
}

.clefts-page-count {
    font-size: 12px;
    color: #52616d;
    white-space: nowrap;
}

/* Page number */
.clefts-page-number {
    margin: 0 !important;
    padding: 0 !important;
}

.clefts-page-number input,
.clefts-page-number textarea {
    text-align: center !important;
    min-height: 22px !important;
    height: 22px !important;
    padding: 0 3px !important;
    font-size: 12px !important;
    line-height: 22px !important;
}

/* Previous / next button: no background, large text */
.clefts-page-button {
    min-width: 22px !important;
    width: 22px !important;
    height: 22px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 20px !important;
    line-height: 20px !important;
    color: #333 !important;
}

.clefts-page-button:hover {
    background: transparent !important;
    color: #000 !important;
}

/* Reduce Gradio default spacing */
.clefts-pager-row .wrap,
.clefts-sort-row .wrap {
    min-height: 22px !important;
    padding: 0 !important;
    margin: 0 !important;
}

.clefts-pager-row .container,
.clefts-sort-row .container {
    padding: 0 !important;
    margin: 0 !important;
}

.clefts-pager-row label,
.clefts-sort-row label {
    margin: 0 !important;
}

/* Dropdown compact */
.clefts-pager-row input,
.clefts-sort-row input {
    min-height: 22px !important;
    height: 22px !important;
    font-size: 12px !important;
}
"""

STYLES = "\n\n".join([ERROR_MESSAGE_CSS, FILE_UPLOAD_MODAL_CSS, GENERIC_TABLE_INPUT_CSS, PAGINATED_DATAFRAME_CSS])