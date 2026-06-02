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
