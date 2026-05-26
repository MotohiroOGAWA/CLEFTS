from __future__ import annotations
from typing import Tuple

class CleavageProduct:
    """
    Represents a single product obtained from a cleavage reaction.
    Stores both the SMILES and index mapping between
    reactant and product atoms.
    """

    def __init__(
        self,
        smiles: str,
        reactant_indices: Tuple[int, ...],
        product_indices: Tuple[int, ...],
    ):
        self._smiles = smiles
        self._reactant_indices = tuple(reactant_indices)
        self._product_indices = tuple(product_indices)

    @property
    def smiles(self) -> str:
        """SMILES of this cleavage product."""
        return self._smiles

    @property
    def reactant_indices(self) -> Tuple[int, ...]:
        """Indices of corresponding atoms in the original molecule."""
        return self._reactant_indices

    @property
    def product_indices(self) -> Tuple[int, ...]:
        """Indices of atoms in this cleavage product molecule."""
        return self._product_indices

    def __repr__(self) -> str:
        return (
            f"CleavageProduct("
            f"smiles='{self._smiles}', "
            f"reactant_indices={self._reactant_indices}, "
            f"product_indices={self._product_indices})"
        )