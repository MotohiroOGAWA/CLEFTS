from __future__ import annotations

MASS_SPECTRUM_VIEW_CSS = """
.clefts-spectrum-view {
    width: 100%;
    align-items: stretch;
}

.clefts-spectrum-plot-column,
.clefts-spectrum-table-column {
    align-self: stretch;
}

.clefts-spectrum-plot,
.clefts-spectrum-table {
    height: 100%;
}

.clefts-spectrum-plot > .wrap,
.clefts-spectrum-table > .wrap {
    height: 100%;
}

.clefts-spectrum-plot .plot-container,
.clefts-spectrum-plot .js-plotly-plot,
.clefts-spectrum-plot .plotly,
.clefts-spectrum-plot .svg-container {
    height: 100% !important;
}

.clefts-spectrum-table [data-testid="dataframe"] {
    height: 100% !important;
}

.clefts-spectrum-table .table-wrap,
.clefts-spectrum-table .dataframe {
    height: 100% !important;
}

.clefts-spectrum-view .form {
    height: 100%;
}
"""