from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass(frozen=True)
class MSDatasetTable:
    source_dataset_state: gr.State
    dataset_state: gr.State

    dataframe: gr.HTML

    previous_button: gr.Button
    page_number: gr.Number
    page_count: gr.HTML
    next_button: gr.Button

    rows_per_page: gr.Dropdown

    download_button: gr.DownloadButton

    @property
    def data_outputs(self) -> list[gr.Component]:
        return [
            self.source_dataset_state,
            self.dataset_state,
            self.dataframe,
            self.page_number,
            self.page_count,
        ]