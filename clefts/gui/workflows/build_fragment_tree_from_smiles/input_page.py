from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from clefts.gui.components.cleavage_patterns_input_panel import load_pattern_set_json_file, render_cleavage_patterns_input_panel
from clefts.gui.components.fragment_tree_builder_input_panel import render_fragment_tree_builder_input_panel
from clefts.gui.workflows.build_fragment_tree_from_smiles.metadata import DEFAULT_CLEAVAGE_PATTERNS_PATH, DEMO_SMILES, WORKFLOW_DESCRIPTION, WORKFLOW_TITLE
from clefts.gui.workflows.build_fragment_tree_from_smiles.navigation import render_back_to_main_html


@dataclass(frozen=True)
class BuildFragmentTreeFromSmilesInputPage:
    smiles: gr.Textbox
    builder_components: list[gr.components.Component]
    cleavage_patterns_json: gr.Code
    run_button: gr.Button

    @property
    def run_inputs(self) -> list[gr.components.Component]:
        return [self.smiles, *self.builder_components, self.cleavage_patterns_json]


def render_input_page() -> BuildFragmentTreeFromSmilesInputPage:
    gr.HTML(render_back_to_main_html())
    gr.HTML(
        f"""
        <section class="clefts-page-heading">
            <h1>{WORKFLOW_TITLE}</h1>
            <p>{WORKFLOW_DESCRIPTION}</p>
        </section>
        """
    )
    with gr.Group(elem_classes="clefts-form-panel"):
        smiles = gr.Textbox(label="SMILES", value=DEMO_SMILES)
        builder_panel = render_fragment_tree_builder_input_panel()
        cleavage_patterns_panel = render_cleavage_patterns_input_panel(
            default_pattern_set=load_pattern_set_json_file(DEFAULT_CLEAVAGE_PATTERNS_PATH)
        )
        run_button = gr.Button("Run", variant="primary")

    return BuildFragmentTreeFromSmilesInputPage(
        smiles=smiles,
        builder_components=builder_panel.components,
        cleavage_patterns_json=cleavage_patterns_panel.pattern_set_json,
        run_button=run_button,
    )
