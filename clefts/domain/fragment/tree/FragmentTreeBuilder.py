from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Any, Dict, Tuple

from ....libs.mmkit.mmkit import Compound
from ..cleavage._CleavagePattern import _CleavagePattern
from ..cleavage.CleavagePatternSet import CleavagePatternSet, CleavagePattern, CleavageResult
from ..cleavage._CleavagePattern import _CleavageResult
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
    def cleavage_patterns(self) -> Tuple[_CleavagePattern, ...]:
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
        result = self._build_result(
            compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info
        )
        return result['fragment_tree']

    def _build_result(
        self,
        compound: Compound,
        *,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(compound, Compound):
            raise TypeError("compound must be a Compound.")

        if not isinstance(max_node, int) or not (
            max_node == -1 or max_node > 0
        ):
            raise ValueError("max_node must be -1 or a positive integer.")

        if not isinstance(max_edge, int) or max_edge < -1:
            raise ValueError("max_edge must be -1 or a non-negative integer.")

        start_time = time.time()
        fragment_compound_by_index: Dict[int, Compound] = {}
        root_compound = compound.copy()
        fragment_compound_by_index[0] = root_compound

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
                if source_index not in fragment_compound_by_index:
                    source_compound = Compound.from_smiles(source_smiles)
                    fragment_compound_by_index[source_index] = source_compound
                else:
                    source_compound = fragment_compound_by_index[source_index]

                cleavage_results = self.cleave_all(source_compound)

                for cleavage_result in cleavage_results:
                    cleavage_pattern_id = cleavage_result.pattern_id

                    for cleavage_product in cleavage_result.products:
                        reaction_id = cleavage_product.id

                        for product_molecule in cleavage_product.product_molecules:
                            target_exists = state.node_exists(product_molecule.compound.smiles)

                            if (
                                not target_exists
                                and (
                                    not state.can_add_node()
                                    or not state.can_add_edge()
                                )
                            ):
                                continue

                            target_index = state.get_or_create_node_index(
                                smiles=product_molecule.compound.smiles,
                                depth=depth,
                            )
                            fragment_compound_by_index[target_index] = product_molecule.compound


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

        result = {
            "fragment_tree": state.to_fragment_tree(),
            "fragment_compound_by_index": fragment_compound_by_index,
        }
        return result
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "cleavage_pattern_set": self.cleavage_pattern_set.to_dict(),
            "only_add_min_depth": self.only_add_min_depth,
            "min_depth_only_from": self.min_depth_only_from,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> FragmentTreeBuilder:
        return cls(
            max_depth=data["max_depth"],
            cleavage_pattern_set=CleavagePatternSet.from_dict(data["cleavage_pattern_set"]),
            only_add_min_depth=bool(data["only_add_min_depth"]),
            min_depth_only_from=int(data["min_depth_only_from"]),
        )

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

        self.create_node_id_func = (
            create_node_id_func
            if create_node_id_func is not None
            else lambda smiles, depth, index: -1
        )

        self.create_edge_id_func = (
            create_edge_id_func
            if create_edge_id_func is not None
            else lambda source_index, target_index, index: -1
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

        if edge_key in self.edges:
            edge = self.edges[edge_key]

            event_id = self._find_or_create_local_event_id(
                edge=edge,
                cleavage_pattern_id=cleavage_pattern_id,
                reaction_id=reaction_id,
                product_molecule_id=product_molecule_id,
                reactant_indices=reactant_indices,
                product_indices=product_indices,
            )

            if event_id is None:
                return edge.index

            event = CleavageEvent(
                index=-1,
                cleavage_pattern_id=cleavage_pattern_id,
                reaction_id=reaction_id,
                product_molecule_id=product_molecule_id,
                event_id=event_id,
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
            index=-1,
            cleavage_pattern_id=cleavage_pattern_id,
            reaction_id=reaction_id,
            product_molecule_id=product_molecule_id,
            event_id=0,
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

    def _find_or_create_local_event_id(
        self,
        *,
        edge: FragmentEdge,
        cleavage_pattern_id: int,
        reaction_id: int,
        product_molecule_id: int,
        reactant_indices: tuple[int, ...],
        product_indices: tuple[int, ...],
    ) -> int | None:
        """Return local event_id for one event group inside an edge.

        Returns
        -------
        int | None
            - Existing event_id if the same event already exists.
            - New local event_id if the event is new.
            - None if the exactly same event already exists and should not be added.
        """

        same_group_events = [
            event
            for event in edge.events
            if (
                event.cleavage_pattern_id == cleavage_pattern_id
                and event.reaction_id == reaction_id
                and event.product_molecule_id == product_molecule_id
            )
        ]

        for event in same_group_events:
            if (
                event.reactant_indices == reactant_indices
                and event.product_indices == product_indices
            ):
                return None

        if not same_group_events:
            return 0

        return max(event.event_id for event in same_group_events) + 1

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