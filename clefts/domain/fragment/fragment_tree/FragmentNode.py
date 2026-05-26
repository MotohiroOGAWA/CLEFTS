from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class FragmentNode:
    """Node in a generated fragmentation tree.

    A ``FragmentNode`` stores the identifier and canonical SMILES string for one
    molecular fragment. The identifier corresponds to the position of the node
    in :attr:`FragmentTree.node_smiles`.
    """

    _id: int
    _smiles: str

    def __init__(self, id: int, smiles: str):
        """Create a fragment node.

        Parameters
        ----------
        id : int
            Node identifier in the fragment tree.
        smiles : str
            Canonical SMILES string for the fragment.
        """
        object.__setattr__(self, "_id", int(id))
        object.__setattr__(self, "_smiles", str(smiles))

    @property
    def id(self) -> int:
        """Node identifier."""
        return self._id

    @property
    def smiles(self) -> str:
        """Canonical SMILES string for the fragment."""
        return self._smiles

    def __repr__(self):
        return "FragmentNode" + self.__str__()

    def __str__(self):
        return f"(id={self.id};{self.smiles})"

    def copy(self) -> "FragmentNode":
        """Return a copy of this fragment node."""
        return FragmentNode(id=self.id, smiles=self.smiles)
