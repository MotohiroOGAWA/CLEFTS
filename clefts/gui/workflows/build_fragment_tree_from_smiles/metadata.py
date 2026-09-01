from __future__ import annotations

WORKFLOW_TITLE = "Build Fragment Tree from SMILES"
WORKFLOW_DESCRIPTION = "Enter a SMILES string and configure fragment-tree builder settings."
WORKFLOW_BASE_PATH = "/build_fragment_tree_from_smiles"
INPUT_PATH = f"{WORKFLOW_BASE_PATH}/input/"
RESULT_PATH = f"{WORKFLOW_BASE_PATH}/result/"
DEMO_SMILES = "CCCOCNCSCF"

from pathlib import Path

DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"
DEFAULT_CLEAVAGE_PATTERNS_PATH = DEFAULTS_DIR / "cleavage_patterns.json"
