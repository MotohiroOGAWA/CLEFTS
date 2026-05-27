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

STYLES = "\n\n".join([PAGINATED_DATAFRAME_CSS])