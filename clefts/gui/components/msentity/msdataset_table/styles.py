MSDATASET_TABLE_STYLE = """
.msdataset-table-container {
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    overflow-y: auto;
}

.msdataset-table {
    width: max-content;
    min-width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

/* Header */
.msdataset-table thead th {
    background-color: #4f81bd;
    color: white;
    font-weight: 600;
    text-align: left;
    padding: 8px 12px;
    border: 1px solid #d0d7de;
    position: sticky;
    top: 0;
    z-index: 1;
}

/* Cells */
.msdataset-table td {
    padding: 6px 12px;
    border: 1px solid #e5e7eb;
    white-space: nowrap;
}

/* Zebra stripes */
.msdataset-table tbody tr:nth-child(odd) {
    background-color: #ffffff;
}

.msdataset-table tbody tr:nth-child(even) {
    background-color: #f3f4f6;
}

/* Hover */
.msdataset-table tbody tr:hover {
    background-color: #dbeafe;
}
"""

MSDATASET_PAGING_STYLE = """
.msdataset-paging-row {
    align-items: center;
    gap: 6px;
}

.msdataset-paging-left {
    flex-grow: 0 !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 6px;
}

.msdataset-page-button {
    width: 28px !important;
    min-width: 28px !important;
    height: 28px !important;
    padding: 0 !important;
}

.msdataset-page-number input::-webkit-outer-spin-button,
.msdataset-page-number input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

.msdataset-page-number input[type="number"] {
    -moz-appearance: textfield;
}

.msdataset-page-count {
    width: 28px !important;
    min-width: 28px !important;
    padding: 0 !important;
}

.msdataset-rows-per-page {
    width: 72px !important;
    min-width: 72px !important;
}

.msdataset-rows-per-page input,
.msdataset-rows-per-page .wrap,
.msdataset-rows-per-page .container {
    border: none !important;
    box-shadow: none !important;
}
"""

MSDATASET_STYLE = f"""
{MSDATASET_TABLE_STYLE}

{MSDATASET_PAGING_STYLE}
"""
