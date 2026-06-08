from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ....libs.mmkit.mmkit import Compound
from ..cleavage.CleavagePattern import _CleavagePattern
from ..cleavage.CleavagePatternSet import CleavagePatternSet, CleavagePattern, CleavageResult
from ..cleavage.CleavagePattern import _CleavageResult
from .CleavageEvent import CleavageEvent
from .FragmentEdge import FragmentEdge
from .FragmentNode import FragmentNode
from .FragmentTree import FragmentTree


@dataclass(frozen=True)
class FragmentTreeBuilder:
    """Build a FragmentTree from cleavage patterns.

    Notes
    -----
    This class only builds FragmentTree.

    Optional DB registration is delegated to fragment_tree.db.
    The builder does not know SQLite table details.
    """

    max_depth: int
    cleavage_pattern_set: CleavagePatternSet
    only_add_min_depth: bool = True
    min_depth_only_from: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.max_depth, int) or self.max_depth <= 0:
            raise ValueError("max_depth must be a positive integer.")

        if not isinstance(self.cleavage_pattern_set, CleavagePatternSet):
            raise TypeError(
                "cleavage_pattern_set must be a CleavagePatternSet."
            )

        if not isinstance(self.only_add_min_depth, bool):
            raise TypeError("only_add_min_depth must be a bool.")

        if (
            not isinstance(self.min_depth_only_from, int)
            or self.min_depth_only_from < 0
        ):
            raise ValueError(
                "min_depth_only_from must be a non-negative integer."
            )

    @property
    def cleavage_patterns(self) -> tuple[_CleavagePattern, ...]:
        return self.cleavage_pattern_set.patterns

    @property
    def name(self) -> str:
        return self.cleavage_pattern_set.name

    def cleave_by_pattern(
        self,
        compound: Compound,
        cleavage_pattern: _CleavagePattern,
    ) -> _CleavageResult | None:
        if not isinstance(compound, Compound):
            raise TypeError("compound must be a Compound.")

        if not isinstance(cleavage_pattern, _CleavagePattern):
            raise TypeError("cleavage_pattern must be a _CleavagePattern.")

        result = cleavage_pattern.fragment(compound)

        if result is None:
            return None

        if len(result.products) == 0:
            return None

        return result

    def cleave_by_pattern_id(
        self,
        compound: Compound,
        cleavage_pattern_id: int,
    ) -> CleavageResult | None:
        cleavage_pattern = self.cleavage_pattern_set.patterns[
            cleavage_pattern_id
        ]

        return self.cleave_by_pattern(
            compound=compound,
            cleavage_pattern=cleavage_pattern,
        )

    def cleave_all(
        self,
        compound: Compound,
    ) -> tuple[CleavageResult, ...]:
        results: list[CleavageResult] = []

        for cleavage_pattern in self.cleavage_pattern_set.patterns:
            result = self.cleave_by_pattern(
                compound=compound,
                cleavage_pattern=cleavage_pattern,
            )

            if result is not None:
                results.append(result)

        return tuple(results)

    def build(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentTree:
        if not isinstance(compound, Compound):
            raise TypeError("compound must be a Compound.")

        if not isinstance(max_node, int) or not (
            max_node == -1 or max_node > 0
        ):
            raise ValueError("max_node must be -1 or a positive integer.")

        if not isinstance(max_edge, int) or max_edge < -1:
            raise ValueError("max_edge must be -1 or a non-negative integer.")

        start_time = time.time()
        root_compound = compound.copy()

        state = _FragmentTreeBuildState(
            root_smiles=root_compound.smiles,
            max_node=max_node,
            max_edge=max_edge,
            only_add_min_depth=self.only_add_min_depth,
            min_depth_only_from=self.min_depth_only_from,
        )

        for depth in range(1, self.max_depth + 1):
            if not state.next_node_indices:
                break

            new_node_indices: set[int] = set()

            for source_index in sorted(state.next_node_indices):
                source_smiles = state.get_node_smiles(source_index)
                source_compound = Compound.from_smiles(source_smiles)

                cleavage_results = self.cleave_all(source_compound)

                for cleavage_result in cleavage_results:
                    cleavage_pattern_id = cleavage_result.pattern_id

                    for cleavage_product in cleavage_result.products:
                        reaction_id = state.create_reaction_id()

                        for product_molecule in cleavage_product.product_molecules:
                            target_exists = state.node_exists(product_molecule.smiles)

                            if (
                                not target_exists
                                and (
                                    not state.can_add_node()
                                    or not state.can_add_edge()
                                )
                            ):
                                continue

                            target_index = state.get_or_create_node_index(
                                smiles=product_molecule.smiles,
                                depth=depth,
                            )

                            if target_index is None:
                                continue

                            edge_index = state.add_fragment_edge(
                                source_index=source_index,
                                target_index=target_index,
                                cleavage_pattern_id=cleavage_pattern_id,
                                reaction_id=reaction_id,
                                product_molecule_id=product_molecule.id,
                                reactant_indices=cleavage_product.reactant_indices,
                                product_indices=product_molecule.product_indices,
                                depth=depth,
                            )

                            if edge_index is not None:
                                new_node_indices.add(target_index)

                state.mark_processed(source_index)

            state.move_to_next_depth(new_node_indices)

            if print_info:
                elapsed = time.time() - start_time
                print(
                    f"Depth {depth} completed. "
                    f"New nodes: {len(new_node_indices)}. "
                    f"Total nodes: {len(state.nodes)}. "
                    f"Total edges: {len(state.edges)}. "
                    f"Time elapsed: {elapsed:.2f} seconds."
                )

        return state.to_fragment_tree()

    def copy(self) -> FragmentTreeBuilder:
        return FragmentTreeBuilder(
            max_depth=self.max_depth,
            cleavage_pattern_set=self.cleavage_pattern_set.copy(),
            only_add_min_depth=self.only_add_min_depth,
            min_depth_only_from=self.min_depth_only_from,
        )


class _FragmentTreeBuildState:
    """Mutable build state used only by FragmentTreeBuilder."""

    def __init__(
        self,
        root_smiles: str,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        only_add_min_depth: bool = True,
        min_depth_only_from: int = 0,
        create_node_id_func: Callable[[str, int, int], int] | None = None,
        create_edge_id_func: Callable[[int, int, int], int] | None = None,
    ) -> None:
        self.root_smiles = root_smiles
        self.max_node = max_node
        self.max_edge = max_edge
        self.only_add_min_depth = only_add_min_depth
        self.min_depth_only_from = min_depth_only_from

        self.nodes: dict[int, FragmentNode] = {}
        self.edges: dict[tuple[int, int], FragmentEdge] = {}

        self.smiles_to_node_index: dict[str, int] = {}
        self.processed_node_indices: set[int] = set()
        self.node_depths: dict[int, int] = {}

        self._next_reaction_id = 0
        self._next_event_id = 0

        self.create_node_id_func = (
            create_node_id_func
            if create_node_id_func is not None
            else lambda smiles, depth, index: index
        )

        self.create_edge_id_func = (
            create_edge_id_func
            if create_edge_id_func is not None
            else lambda source_index, target_index, index: index
        )

        root_index = self.get_or_create_node_index(
            smiles=root_smiles,
            depth=0,
        )

        if root_index is None:
            raise ValueError("max_node must allow at least the root node.")

        self.root_index = root_index
        self.next_node_indices: set[int] = {root_index}

    def can_add_node(self) -> bool:
        return self.max_node < 0 or len(self.nodes) < self.max_node

    def can_add_edge(self) -> bool:
        return self.max_edge < 0 or len(self.edges) < self.max_edge

    def create_reaction_id(self) -> int:
        reaction_id = self._next_reaction_id
        self._next_reaction_id += 1
        return reaction_id

    def create_event_id(self) -> int:
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id

    def node_exists(self, smiles: str) -> bool:
        return smiles in self.smiles_to_node_index

    def get_node_smiles(self, node_index: int) -> str:
        return self.nodes[node_index].smiles

    def get_or_create_node_index(
        self,
        smiles: str,
        depth: int,
    ) -> int | None:
        if smiles in self.smiles_to_node_index:
            return self.smiles_to_node_index[smiles]

        if not self.can_add_node():
            return None

        node_index = len(self.nodes)

        node_id = self.create_node_id_func(
            smiles,
            depth,
            node_index,
        )

        self.nodes[node_index] = FragmentNode(
            index=node_index,
            id=node_id,
            smiles=smiles,
        )

        self.smiles_to_node_index[smiles] = node_index
        self.node_depths[node_index] = depth

        return node_index

    def should_skip_edge(
        self,
        target_index: int,
        depth: int,
    ) -> bool:
        return (
            self.only_add_min_depth
            and depth > self.min_depth_only_from + 1
            and depth > self.node_depths[target_index]
        )

    def add_fragment_edge(
        self,
        source_index: int,
        target_index: int,
        cleavage_pattern_id: int,
        reaction_id: int,
        product_molecule_id: int,
        reactant_indices: tuple[int, ...],
        product_indices: tuple[int, ...],
        depth: int,
    ) -> int | None:
        if self.should_skip_edge(
            target_index=target_index,
            depth=depth,
        ):
            return None

        edge_key = (source_index, target_index)
        reaction_id = self.create_reaction_id()

        if edge_key in self.edges:
            edge = self.edges[edge_key]

            event = CleavageEvent(
                index=len(edge.events),
                cleavage_pattern_id=cleavage_pattern_id,
                reaction_id=reaction_id,
                product_molecule_id=product_molecule_id,
                event_id=self.create_event_id(),
                reactant_indices=reactant_indices,
                product_indices=product_indices,
            )

            self.edges[edge_key] = edge.with_event(event)

            return edge.index

        if not self.can_add_edge():
            return None

        edge_index = len(self.edges)

        edge_id = self.create_edge_id_func(
            source_index,
            target_index,
            edge_index,
        )

        event = CleavageEvent(
            index=0,
            cleavage_pattern_id=cleavage_pattern_id,
            reaction_id=reaction_id,
            product_molecule_id=0,
            event_id=self.create_event_id(),
            reactant_indices=reactant_indices,
            product_indices=product_indices,
        )

        self.edges[edge_key] = FragmentEdge(
            index=edge_index,
            id=edge_id,
            source_index=source_index,
            target_index=target_index,
            source_id=self.nodes[source_index].id,
            target_id=self.nodes[target_index].id,
            events=(event,),
        )

        return edge_index

    def mark_processed(self, node_index: int) -> None:
        self.processed_node_indices.add(node_index)

    def move_to_next_depth(
        self,
        new_node_indices: set[int],
    ) -> None:
        self.next_node_indices = (
            new_node_indices - self.processed_node_indices
        )

    def to_fragment_tree(self) -> FragmentTree:
        return FragmentTree.from_nodes_and_edges(
            smiles=self.root_smiles,
            nodes=tuple(
                self.nodes[node_index]
                for node_index in sorted(self.nodes)
            ),
            edges=tuple(
                sorted(
                    self.edges.values(),
                    key=lambda edge: edge.index,
                )
            ),
        )