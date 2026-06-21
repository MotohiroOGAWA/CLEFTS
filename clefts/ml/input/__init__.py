from .fragment_tree_structure import FragmentTreeStructure
from .fragment_tree_features import FragmentTreeFeatures
from .training_fragment_tree_structure import TrainingFragmentTreeStructure

__all__ = [
    "FragmentTreeStructure",
    "FragmentTreeFeatures",
    "TrainingFragmentTreeStructure",
    "SupervisedFragmentTreeStructureBuilder",
    "FragmentTreeStructureFileDataset",
    "FragmentTreeStructureFileItem",
    "build_fragment_tree_structure_files",
    "collate_fragment_tree_structure_items",
    "make_fragment_tree_structure_dataloader",
]


def __getattr__(name: str):
    if name == "SupervisedFragmentTreeStructureBuilder":
        from .supervised_fragment_tree_structure_builder import (
            SupervisedFragmentTreeStructureBuilder,
        )

        return SupervisedFragmentTreeStructureBuilder

    if name in {
        "FragmentTreeStructureFileDataset",
        "FragmentTreeStructureFileItem",
        "build_fragment_tree_structure_files",
        "collate_fragment_tree_structure_items",
        "make_fragment_tree_structure_dataloader",
    }:
        from . import fragment_tree_training_data

        return getattr(fragment_tree_training_data, name)

    raise AttributeError(name)
