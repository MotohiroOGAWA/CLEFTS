import time

from ....libs.mmkit.mmkit import Compound
from .repository import FragmentGraphRepository as repository
from .database import FragmentGraphDatabase
from ...fragment.fragment_tree.FragmentTreeBuilder import FragmentTree, FragmentTreeBuilder, FragmentTreeBuildState

class FragmentDatabaseBuilder:
    def __init__(
        self,
        tree_builder: FragmentTreeBuilder,
        database: FragmentGraphDatabase,
    ) -> None:
        self.tree_builder = tree_builder
        self.database = database

    def build(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentTree:
        assert isinstance(compound, Compound)
        assert isinstance(max_node, int) and (max_node == -1 or max_node > 0)
        assert isinstance(max_edge, int) and max_edge >= -1

        start_time = time.time()

        with self.database.session() as session:
            def create_node_id(smiles: str, depth: int) -> int:
                node_id = repository.get_or_create_fragment(
                    session=session,
                    compound=Compound.from_smiles(smiles),
                    depth=depth,
                )
                return node_id
            
            root_compound = compound.copy()
            repository.get_or_create_compound(
                session=session,
                compound=root_compound,
            )

            state = FragmentTreeBuildState(
                root_smiles=root_compound.smiles,
                max_node=max_node,
                max_edge=max_edge,
                only_add_min_depth=self._only_add_min_depth,
                min_depth_only_from=self._min_depth_only_from,
                create_node_id_func=create_node_id,
            )

            for depth in range(1, self._max_depth + 1):
                if len(state.next_node_ids) == 0:
                    break

                new_node_ids: set[int] = set()

                for node_id in sorted(state.next_node_ids):
                    source_smiles = state.get_node_smiles(node_id)
                    source_compound = Compound.from_smiles(source_smiles)
                    repository.get_or_create_fragment(
                        session=session,
                        compound=source_compound,
                    )

                    frag_group = self.cleave_all(source_compound)

                    for frag_result in frag_group:
                        cleavage_id = self._cleavage_pattern_set.get_id(
                            frag_result.cleavage
                        )

                        for frag_product in frag_result.products:
                            target_exists = state.node_exists(frag_product.smiles)

                            if (
                                not target_exists
                                and (
                                    not state.can_add_node()
                                    or not state.can_add_edge()
                                )
                            ):
                                continue

                            target_node_id = state.get_or_create_node_id(
                                smiles=frag_product.smiles,
                                depth=depth,
                            )

                            if target_node_id is None:
                                continue

                            edge_id = state.add_fragment_edge(
                                source_node_id=node_id,
                                target_node_id=target_node_id,
                                cleavage_pattern_id=cleavage_id,
                                react_indices=frag_product.reactant_indices,
                                prod_indices=frag_product.product_indices,
                                depth=depth,
                            )

                            if edge_id is not None:
                                new_node_ids.add(target_node_id)

                    state.mark_processed(node_id)

                state.move_to_next_depth(new_node_ids)

                if print_info:
                    elapsed = time.time() - start_time
                    print(
                        f"Depth {depth} completed. "
                        f"New nodes: {len(new_node_ids)}. "
                        f"Total nodes: {len(state.nodes)}. "
                        f"Time elapsed: {elapsed:.2f} seconds."
                    )

        return state.to_fragment_tree()