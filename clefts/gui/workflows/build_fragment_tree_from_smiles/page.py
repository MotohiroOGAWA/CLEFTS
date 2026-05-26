from clefts.gui.workflows.build_fragment_tree_from_smiles.apps import (
    create_app,
    create_input_app,
    create_result_app,
)
from clefts.gui.workflows.build_fragment_tree_from_smiles.input_page import (
    BuildFragmentTreeFromSmilesInputPage,
    render_input_page,
)
from clefts.gui.workflows.build_fragment_tree_from_smiles.metadata import (
    DEMO_SMILES,
    INPUT_PATH,
    RESULT_PATH,
    WORKFLOW_BASE_PATH,
    WORKFLOW_DESCRIPTION,
    WORKFLOW_TITLE,
)
from clefts.gui.workflows.build_fragment_tree_from_smiles.result_page import render_result_page
from clefts.gui.workflows.build_fragment_tree_from_smiles.runner import (
    build_fragment_tree_and_store_result,
    build_fragment_tree_and_store_result_url,
    build_fragment_tree_result,
    get_result,
    load_result_from_request,
    save_result,
)
from clefts.gui.workflows.build_fragment_tree_from_smiles.workflow_item import render_workflow_item_html
