from .fragment_tree_structure import FragmentTreeStructure
from .fragment_tree_features import FragmentTreeFeatures
from .training_fragment_tree_structure import TrainingFragmentTreeStructure

__all__ = [
    "FragmentTreeStructure",
    "FragmentTreeFeatures",
    "TrainingFragmentTreeStructure",
    "SupervisedFragmentTreeStructureBuilder",
]


def __getattr__(name: str):
    if name == "SupervisedFragmentTreeStructureBuilder":
        from .supervised_fragment_tree_structure_builder import (
            SupervisedFragmentTreeStructureBuilder,
        )

        return SupervisedFragmentTreeStructureBuilder
    raise AttributeError(name)
