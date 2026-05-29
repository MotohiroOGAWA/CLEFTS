from ....libs.mmkit.mmkit import Compound
from .repository import FragmentGraphRepository
from ..fragment_tree import FragmentTreeBuilder

class FragmentDatabaseBuilder:
    def __init__(
        self,
        tree_builder: FragmentTreeBuilder,
        repository: FragmentGraphRepository,
    ) -> None:
        self.tree_builder = tree_builder
        self.repository = repository

    def expand_from_compound(
        self,
        compound: Compound,
        max_depth: int,
    ) -> int:
        root_fragment_id = self.repository.get_or_create_fragment(
            smiles=compound.smiles,
        )

        next_fragment_ids = {root_fragment_id}
        processed_fragment_ids: set[int] = set()

        for depth in range(1, max_depth + 1):
            new_fragment_ids: set[int] = set()

            for source_fragment_id in next_fragment_ids:
                source_smiles = self.repository.get_fragment_smiles(
                    fragment_id=source_fragment_id,
                )
                source_compound = Compound.from_smiles(source_smiles)

                for cleavage_pattern in self.tree_builder.cleavage_patterns:
                    fragment_result = self.tree_builder.cleave_by_pattern(
                        compound=source_compound,
                        cleavage_pattern=cleavage_pattern,
                    )

                    if fragment_result is None:
                        continue

                    for product in fragment_result.products:
                        target_fragment_id = self.repository.get_or_create_fragment(
                            smiles=product.smiles,
                        )

                        self.repository.register_fragment_edge(
                            source_smiles=source_smiles,
                            target_smiles=product.smiles,
                            cleavage_pattern=cleavage_pattern,
                            react_indices=str(product.reactant_indices),
                            prod_indices=str(product.product_indices),
                        )

                        if target_fragment_id not in processed_fragment_ids:
                            new_fragment_ids.add(target_fragment_id)

            processed_fragment_ids.update(next_fragment_ids)
            next_fragment_ids = new_fragment_ids - processed_fragment_ids

        return root_fragment_id