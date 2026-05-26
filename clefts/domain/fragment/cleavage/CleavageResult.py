from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING
if TYPE_CHECKING:
    from .CleavagePattern import CleavagePattern
    from .CleavageProduct import CleavageProduct

class CleavageResult:
    """
    Represents the result of a single cleavage pattern applied to one molecule.
    Contains the reaction SMIRKS, the original (reactant) SMILES,
    and a list of product cleavage information.
    """

    def __init__(
        self,
        cleavage: CleavagePattern,
        reactant_smiles: str,
        products: Tuple['CleavageProduct', ...],
    ):
        self._cleavage = cleavage
        self._reactant_smiles = reactant_smiles
        self._products = tuple(products)

    @property
    def cleavage(self) -> CleavagePattern:
        return self._cleavage

    @property
    def reactant_smiles(self) -> str:
        """SMILES of the original molecule before cleavage."""
        return self._reactant_smiles

    @property
    def products(self) -> Tuple['CleavageProduct', ...]:
        """Tuple of product cleavage information."""
        return self._products

    def __repr__(self) -> str:
        return (
            f"CleavageResult("
            f"cleavage='{self._cleavage}', "
            f"reactant='{self._reactant_smiles}', "
            f"n_products={len(self._products)})"
        )

