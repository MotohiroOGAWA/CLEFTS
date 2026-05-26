import os
import time
from typing import Callable, Dict, Optional, Tuple

import yaml

from ....libs.mmkit.mmkit import Compound
from ..cleavage.CleavagePattern import CleavagePattern
from ..cleavage.CleavagePatternSet import CleavagePatternSet
from ..cleavage.CleavageResult import CleavageResult
from .CleavageEvent import CleavageEvent
from .FragmentEdge import FragmentEdge
from .FragmentNode import FragmentNode
from .FragmentTree import FragmentTree


class FragmentTreeBuilder:
    """Build a fragmentation tree from user-defined cleavage patterns.

    ``FragmentTreeBuilder`` repeatedly applies a :class:`CleavagePatternSet` to
    a root compound and to generated product fragments. Each unique product
    SMILES becomes a :class:`FragmentNode`, and each cleavage relationship is
    stored as a :class:`FragmentEdge`.

    The builder is the main API for converting a ``Compound`` into a
    :class:`FragmentTree`. Use :meth:`build` when you already have a configured
    builder, or :meth:`from_compound` for one-shot construction.
    """

    def __init__(
        self,
        max_depth: int,
        cleavage_pattern_set: CleavagePatternSet,
        *,
        only_add_min_depth: bool = True,
        min_depth_only_from: int = 0,
    ):
        """Create a fragment tree builder.

        Parameters
        ----------
        max_depth : int
            Maximum number of recursive cleavage rounds from the root compound.
        cleavage_pattern_set : CleavagePatternSet
            Cleavage patterns used to generate fragment products.
        only_add_min_depth : bool, optional
            If ``True``, suppress additional edges to nodes that were already
            discovered at a shallower depth after ``min_depth_only_from``.
        min_depth_only_from : int, optional
            Depth threshold used with ``only_add_min_depth``.
        """
        assert isinstance(max_depth, int) and max_depth > 0, "max_depth must be a positive integer."
        assert isinstance(cleavage_pattern_set, CleavagePatternSet), "cleavage_pattern_set must be an instance of CleavagePatternSet."
        assert isinstance(only_add_min_depth, bool), "only_add_min_depth must be a boolean."
        assert isinstance(min_depth_only_from, int) and min_depth_only_from >= 0, "min_depth_only_from must be a non-negative integer."

        self.max_depth = max_depth
        self.cleavage_pattern_set = cleavage_pattern_set
        self.only_add_min_depth = only_add_min_depth
        self.min_depth_only_from = min_depth_only_from

    @property
    def cleavage_patterns(self) -> Tuple[CleavagePattern, ...]:
        """Cleavage patterns used by this builder in stable ID order."""
        return self.cleavage_pattern_set.patterns

    @property
    def name(self) -> str:
        """Human-readable name of the underlying cleavage pattern set."""
        return self.cleavage_pattern_set.name

    def to_dict(self) -> Dict[str, object]:
        """Serialize this builder configuration to a dictionary."""
        return {
            "max_depth": self.max_depth,
            "cleavage_pattern_set": self.cleavage_pattern_set.to_dict(),
            "only_add_min_depth": self.only_add_min_depth,
            "min_depth_only_from": self.min_depth_only_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FragmentTreeBuilder":
        """Create a builder from a serialized configuration dictionary."""
        max_depth = int(data.get("max_depth", 1))
        cleavage_pattern_set = CleavagePatternSet.from_dict(
            data.get("cleavage_pattern_set", {})
        )
        only_add_min_depth = bool(data.get("only_add_min_depth", True))
        min_depth_only_from = int(data.get("min_depth_only_from", 0))
        return cls(
            max_depth=max_depth,
            cleavage_pattern_set=cleavage_pattern_set,
            only_add_min_depth=only_add_min_depth,
            min_depth_only_from=min_depth_only_from,
        )

    def to_yaml(self, path: str) -> None:
        """Save this builder configuration to a YAML file.

        Parameters
        ----------
        path : str
            Output YAML path. Parent directories are created when needed.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )

    @classmethod
    def from_yaml(cls, path: str) -> "FragmentTreeBuilder":
        """Load a builder configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @staticmethod
    def from_compound(
        compound: Compound,
        cleavage_pattern_set: CleavagePatternSet,
        max_depth: int,
        *,
        only_add_min_depth: bool = True,
        min_depth_only_from: int = 0,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentTree:
        """Build a fragment tree without explicitly creating a builder.

        Parameters
        ----------
        compound : Compound
            Root molecule used as the starting point of the fragmentation tree.
        cleavage_pattern_set : CleavagePatternSet
            Cleavage patterns to apply recursively.
        max_depth : int
            Maximum number of recursive cleavage rounds.
        only_add_min_depth : bool, optional
            Whether to suppress deeper duplicate edges after the configured
            threshold.
        min_depth_only_from : int, optional
            Depth threshold used with ``only_add_min_depth``.
        max_node : int, optional
            Maximum number of nodes to keep. Use ``-1`` for no limit.
        max_edge : int, optional
            Maximum number of edges to keep. Use ``-1`` for no limit.
        print_info : bool, optional
            If ``True``, print progress after each depth.

        Returns
        -------
        FragmentTree
            Fragmentation tree generated from ``compound``.
        """
        builder = FragmentTreeBuilder(
            max_depth=max_depth,
            cleavage_pattern_set=cleavage_pattern_set,
            only_add_min_depth=only_add_min_depth,
            min_depth_only_from=min_depth_only_from,
        )
        return builder.build(
            compound=compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info,
        )

    def cleave_all(
        self,
        compound: Compound,
    ) -> Tuple[CleavageResult, ...]:
        """Apply all matching cleavage patterns to a compound.

        Parameters
        ----------
        compound : Compound
            Molecule to cleave.

        Returns
        -------
        tuple of CleavageResult
            Non-empty cleavage results in stable pattern ID order.
        """
        fragment_group = []
        for pattern in self.cleavage_pattern_set.patterns:
            fragment_result = pattern.fragment(compound)
            if fragment_result is not None and len(fragment_result.products) > 0:
                fragment_group.append(fragment_result)
        return tuple(fragment_group)

    def build(
        self,
        compound: Compound,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentTree:
        """Build a :class:`FragmentTree` from a root compound.

        Parameters
        ----------
        compound : Compound
            Root molecule used as the starting point of the fragmentation tree.
        max_node : int, optional
            Maximum number of nodes to keep. Use ``-1`` for no limit.
        max_edge : int, optional
            Maximum number of edges to keep. Use ``-1`` for no limit.
        print_info : bool, optional
            If ``True``, print progress after each depth.

        Returns
        -------
        FragmentTree
            Fragmentation tree whose root node is ``compound``.

        Raises
        ------
        """
        assert isinstance(compound, Compound), "compound must be a Compound instance."
        assert isinstance(max_node, int) and (max_node == -1 or max_node > 0), "max_node must be -1 or a positive integer."
        assert isinstance(max_edge, int) and max_edge >= -1, "max_edge must be -1 or a non-negative integer."
        start_time = time.time()

        def can_add_node() -> bool:
            return max_node < 0 or len(nodes) < max_node

        def can_add_edge() -> bool:
            return max_edge < 0 or len(edges) < max_edge

        nodes: Dict[int, FragmentNode] = {}
        edges: Dict[Tuple[int, int], FragmentEdge] = {}
        smi_to_node_id: Dict[str, int] = {}
        processed_node_ids = set()
        node_depths: Dict[int, int] = {}

        def get_or_create_node_id(smiles: str, depth: int) -> Optional[int]:
            if smiles in smi_to_node_id:
                return smi_to_node_id[smiles]
            if not can_add_node():
                return None

            node_id = len(nodes)
            nodes[node_id] = FragmentNode(node_id, smiles)
            smi_to_node_id[smiles] = node_id
            node_depths[node_id] = depth
            return node_id

        def should_skip_edge(target_node_id: int, depth: int) -> bool:
            return (
                self.only_add_min_depth
                and depth > self.min_depth_only_from + 1
                and depth > node_depths[target_node_id]
            )

        def add_fragment_edge(
            source_node_id: int,
            target_node_id: int,
            cleavage_pattern_id: int,
            react_indices: Tuple[int, ...],
            prod_indices: Tuple[int, ...],
            depth: int,
        ) -> Optional[int]:
            if should_skip_edge(target_node_id, depth):
                return None

            edge_key = (source_node_id, target_node_id)
            if edge_key in edges:
                edge = edges[edge_key]
                event_id = len(edge.events)
                edges[edge_key] = edge.with_event(
                    CleavageEvent(
                        event_id=event_id,
                        cleavage_pattern_id=cleavage_pattern_id,
                        react_indices=react_indices,
                        prod_indices=prod_indices,
                    )
                )
                return edge.id

            if not can_add_edge():
                return None

            edge_id = len(edges)
            edges[edge_key] = FragmentEdge(
                id=edge_id,
                source_id=source_node_id,
                target_id=target_node_id,
                events=(
                    CleavageEvent(
                        event_id=0,
                        cleavage_pattern_id=cleavage_pattern_id,
                        react_indices=react_indices,
                        prod_indices=prod_indices,
                    ),
                ),
            )
            return edge_id

        root_compound = compound.copy()
        root_node_id = get_or_create_node_id(root_compound.smiles, depth=0)
        if root_node_id is None:
            raise ValueError("max_node must allow at least the root node.")

        next_node_ids = {root_node_id}
        for depth in range(1, self.max_depth + 1):
            if len(next_node_ids) == 0:
                break

            new_node_ids = set()
            for node_id in sorted(next_node_ids):
                source_smiles = nodes[node_id].smiles
                source_compound = Compound.from_smiles(source_smiles)
                frag_group = self.cleave_all(source_compound)

                for frag_result in frag_group:
                    cleavage_id = self.cleavage_pattern_set.get_id(frag_result.cleavage)
                    for frag_product in frag_result.products:
                        target_exists = frag_product.smiles in smi_to_node_id
                        if not target_exists and (not can_add_node() or not can_add_edge()):
                            continue
                        target_node_id = get_or_create_node_id(frag_product.smiles, depth=depth)
                        if target_node_id is None:
                            continue
                        edge_id = add_fragment_edge(
                            source_node_id=node_id,
                            target_node_id=target_node_id,
                            cleavage_pattern_id=cleavage_id,
                            react_indices=frag_product.reactant_indices,
                            prod_indices=frag_product.product_indices,
                            depth=depth,
                        )
                        if edge_id is not None:
                            new_node_ids.add(target_node_id)

                processed_node_ids.add(node_id)

            next_node_ids = new_node_ids - processed_node_ids

            if print_info:
                elapsed = time.time() - start_time
                print(
                    f"Depth {depth} completed. "
                    f"New nodes: {len(new_node_ids)}. "
                    f"Total nodes: {len(nodes)}. "
                    f"Time elapsed: {elapsed:.2f} seconds."
                )

        return FragmentTree.from_nodes_and_edges(
            smiles=root_compound.smiles,
            nodes=tuple(nodes[node_id] for node_id in sorted(nodes)),
            edges=tuple(sorted(edges.values(), key=lambda edge: edge.id)),
        )

    def create_fragment_tree(
        self,
        compound: Compound,
        max_node: int = -1,
        max_edge: int = -1,
        print_info: bool = False,
    ) -> FragmentTree:
        """Build a fragment tree.

        This method is kept as a compatibility alias for :meth:`build`.
        """
        return self.build(
            compound=compound,
            max_node=max_node,
            max_edge=max_edge,
            print_info=print_info,
        )


    def copy(self) -> "FragmentTreeBuilder":
        """Create a copy of this builder configuration."""
        return FragmentTreeBuilder(
            max_depth=self.max_depth,
            cleavage_pattern_set=self.cleavage_pattern_set.copy(),
            only_add_min_depth=self.only_add_min_depth,
            min_depth_only_from=self.min_depth_only_from,
        )
