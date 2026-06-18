from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Set, Sequence

import torch
from torch_geometric.data import Batch, Data
import numpy as np

from ...libs.mmkit.mmkit import Compound, Adduct
from ...libs.msentity.msentity import SpectrumRecord
from ...domain.fragment.cleavage import CleavageResult
from ...domain.fragment.ion_tree import FragmentIonTree
from ...domain.fragment.pathway import (
    FragmentPathway,
    FragmentPathwayGroup,
    FragmentPathwayNode,
    FragmentPathwayEdge,
    CleavageStep,
)
from ..specgen import CleftsSpecGen
from .fragment_tree_structure import FragmentTreeStructure


@dataclass
class FragmentTreeSample:
    """One sample that references a shared fragment tree."""

    adduct_type_index: int
    # Index of the adduct type used for this sample.

    ce_value: float
    # Collision energy value for this sample.

    edge_indexes: Set[int] = field(default_factory=set)
    # Set of edge_index values used by this sample.
    #
    # This includes all fragment-tree edges used by this sample,
    # including precursor pathways and peak-assigned pathways.
    #
    # Each edge_index refers to:
    #     edge_src[edge_index]
    #     edge_dst[edge_index]
    # in the shared FragmentTreeStructureBuilder.

    precursor_edge_indexes: Set[Tuple[int, ...]] = field(default_factory=set)
    # Set of precursor pathway edge-index tuples.
    #
    # Each tuple represents one complete precursor fragment pathway:
    #     (edge_index_0, edge_index_1, ..., edge_index_k)
    #
    # Each edge_index in the tuple refers to:
    #     edge_src[edge_index]
    #     edge_dst[edge_index]
    # in the shared FragmentTreeStructureBuilder.
    #
    # This field preserves pathway-level grouping.
    # In contrast, edge_indexes stores the flattened set of all edges
    # used by this sample.


@dataclass
class SingleFragmentTreeStructureBuilder:
    _model: CleftsSpecGen = field(repr=False, compare=False)
    # CleftsSpecGen model used to define model-derived IDs and settings.
    #
    # This builder depends on the model for:
    #     adduct type definitions
    #     cleavage_id definitions
    #     reaction_id definitions
    #     product_molecule_id definitions
    #     cleavage_id -> reactant/product tuple length mapping

    # -------------------------
    # Nodes
    # -------------------------
    node_smiles: List[str] = field(default_factory=list)
    # [N] Canonical SMILES strings for fragment nodes.
    # node_index = index in this list.
    # node_smiles[node_index] gives the SMILES of the corresponding node.

    node_graph: List[Data] = field(default_factory=list)
    # [N] Molecular graphs aligned with node_smiles.
    # node_graph[node_index] corresponds to node_smiles[node_index].

    _node_graph_offset: List[int] = field(default_factory=lambda: [0])
    # [N + 1] Prefix-sum offsets for converting local atom indices
    # in each node graph to global atom indices in the batched graph.
    #
    # For a node with node_index:
    #     global_atom_index = local_atom_index + _node_graph_offset[node_index]
    #
    # The atom index range of node_index is:
    #     [_node_graph_offset[node_index], _node_graph_offset[node_index + 1])

    # -------------------------
    # Edges
    # -------------------------
    edge_src: List[int] = field(default_factory=list)
    # [E] Source node_index for each fragmentation edge.
    # edge_index = index in this list.
    # edge_src[edge_index] gives the parent/source node_index.

    edge_dst: List[int] = field(default_factory=list)
    # [E] Destination node_index for each fragmentation edge.
    # edge_dst[edge_index] gives the child/destination node_index.
    # edge_src[edge_index] and edge_dst[edge_index] form one directed edge.

    # -------------------------
    # Cleavage events
    # -------------------------
    cleavage_event_edge_index: List[int] = field(default_factory=list)
    # [M] Edge indexes associated with cleavage events.
    # cleavage_event_edge_index[event_index] gives the edge_index
    # to which the cleavage event belongs.
    #
    # edge_index refers to:
    #     edge_src[edge_index]
    #     edge_dst[edge_index]

    cleavage_event: List[Tuple[int, int, int, int, int]] = field(default_factory=list)
    # [M] Cleavage event references.
    # event_index = index in this list.
    #
    # cleavage_event[event_index] is paired with:
    #     cleavage_event_edge_index[event_index]
    #
    # Each tuple represents:
    # (
    #     cleavage_id,
    #     reaction_id,
    #     product_molecule_id,
    #     reactant_atom_row_index,
    #     product_atom_row_index,
    # )
    #
    # cleavage_id, reaction_id, and product_molecule_id are identifiers
    # defined by the cleavage/reaction model or rule set.
    #
    # reactant_atom_row_index and product_atom_row_index both refer to:
    #     cleavage_atom_idxs[tuple_length][row_index]
    #
    # The tuple_length is not stored in cleavage_event.
    # It is inferred outside this builder from:
    #     reactant: (cleavage_id, reaction_id)
    #     product:  (cleavage_id, reaction_id, product_molecule_id)

    # -------------------------
    # Cleavage atom-index buckets
    # tuple_length -> rows of unique atom-index tuples
    # -------------------------
    cleavage_atom_idxs: Dict[int, List[Tuple[int, ...]]] = field(default_factory=dict)
    # Dict[tuple_length, List[atom_index_tuple]]
    #
    # Stores unique global atom-index tuples grouped by tuple length.
    #
    # key:
    #     tuple_length, i.e. len(atom_index_tuple)
    #
    # value:
    #     list of unique global atom-index tuples with that length.
    #
    # cleavage_atom_idxs[tuple_length][row_index]
    # gives one atom-index tuple.
    #
    # Both reactant-side and product-side atom tuples are stored here.

    samples: Dict[int, FragmentTreeSample] = field(default_factory=dict)
    # sample_index -> sample-specific information.
    #
    # Each sample stores:
    #     adduct_type_index
    #     ce_value
    #     edge_indexes
    #
    # The fragment tree nodes, edges, and cleavage events are shared across samples.

    # -------------------------
    # Private lookup tables
    # -------------------------
    _node_index_by_smiles: Dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # canonical_smiles -> node_index.
    #
    # Used to avoid adding duplicate nodes for the same canonical SMILES.

    _edge_index_by_node_indexes: Dict[Tuple[int, int], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # (src_node_index, dst_node_index) -> edge_index.
    #
    # Used to avoid adding duplicate directed edges between the same source
    # and destination nodes.

    _cleavage_event_index_by_key: Dict[
        Tuple[int, int, int, int, int, int],
        int,
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # (
    #     edge_index,
    #     cleavage_id,
    #     reaction_id,
    #     product_molecule_id,
    #     reactant_atom_row_index,
    #     product_atom_row_index,
    # ) -> cleavage_event_index.
    #
    # Used to avoid adding duplicate cleavage events.
    #
    # edge_index and row indexes are internal indexes in this builder.
    # cleavage_id, reaction_id, and product_molecule_id are external/model-defined IDs.

    _atom_row_index_by_atom_idxs: Dict[Tuple[int, ...], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # atom_index_tuple -> row_index.
    #
    # The tuple length is obtained by:
    #     tuple_length = len(atom_index_tuple)
    #
    # The row index refers to:
    #     cleavage_atom_idxs[tuple_length][row_index]
    #
    # Used to avoid storing duplicate atom-index tuples.

    def __post_init__(self) -> None:
        if not isinstance(self._model, CleftsSpecGen):
            raise TypeError(
                "_model must be an instance of CleftsSpecGen, "
                f"but got {type(self._model).__name__}."
            )

    def to_structure(self) -> FragmentTreeStructure:
        """Convert this builder into a torch-ready FragmentTreeStructure."""

        if len(self.node_graph) == 0:
            raise ValueError("Cannot build FragmentTreeStructure with no nodes.")

        # -------------------------
        # Node graphs
        # -------------------------
        node_smiles = np.asarray(
            self.node_smiles,
            dtype=object,
        )

        node_graph = Batch.from_data_list(self.node_graph)

        node_graph_offset = torch.tensor(
            self._node_graph_offset,
            dtype=torch.long,
        )

        # -------------------------
        # Fragment-tree edges
        # -------------------------
        if len(self.edge_src) > 0:
            edge_index = torch.tensor(
                [self.edge_src, self.edge_dst],
                dtype=torch.long,
            )
        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        # -------------------------
        # Cleavage events
        # -------------------------
        cleavage_event_edge_index = torch.tensor(
            self.cleavage_event_edge_index,
            dtype=torch.long,
        )

        if len(self.cleavage_event) > 0:
            cleavage_event = torch.tensor(
                self.cleavage_event,
                dtype=torch.long,
            )
        else:
            cleavage_event = torch.empty(
                (0, 5),
                dtype=torch.long,
            )

        cleavage_atom_idxs = {
            int(tuple_length): torch.tensor(rows, dtype=torch.long)
            for tuple_length, rows in self.cleavage_atom_idxs.items()
        }

        # -------------------------
        # Tuple-length tables
        # -------------------------
        reactant_tuple_length_table = self._tuple_length_dict_to_tensor(
            self._model.cleavage_edge_fnet.reactant_tuple_length_by_event_type,
            key_width=2,
        )
        # [R, 3]
        # columns:
        #   0: cleavage_id
        #   1: reaction_id
        #   2: reactant_tuple_length

        product_tuple_length_table = self._tuple_length_dict_to_tensor(
            self._model.cleavage_edge_fnet.product_tuple_length_by_event_type,
            key_width=3,
        )
        # [P, 4]
        # columns:
        #   0: cleavage_id
        #   1: reaction_id
        #   2: product_molecule_id
        #   3: product_tuple_length

        # -------------------------
        # Samples
        # -------------------------
        sample_indexes = sorted(self.samples.keys())

        sample_index_remap = {
            old_sample_index: new_sample_index
            for new_sample_index, old_sample_index in enumerate(sample_indexes)
        }

        sample_adduct_type_index = torch.tensor(
            [
                int(self.samples[sample_index].adduct_type_index)
                for sample_index in sample_indexes
            ],
            dtype=torch.long,
        )

        sample_ce_value = torch.tensor(
            [
                float(self.samples[sample_index].ce_value)
                for sample_index in sample_indexes
            ],
            dtype=torch.float32,
        )

        sample_edge_pairs: list[tuple[int, int]] = []

        for old_sample_index in sample_indexes:
            new_sample_index = sample_index_remap[old_sample_index]
            sample = self.samples[old_sample_index]

            for edge_index_value in sorted(sample.edge_indexes):
                if edge_index_value < 0:
                    continue

                sample_edge_pairs.append(
                    (
                        int(new_sample_index),
                        int(edge_index_value),
                    )
                )

        if len(sample_edge_pairs) > 0:
            sample_edge_index = torch.tensor(
                sample_edge_pairs,
                dtype=torch.long,
            ).t().contiguous()
        else:
            sample_edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        precursor_paths: list[tuple[int, ...]] = []
        precursor_path_sample_indexes: list[int] = []

        for old_sample_index in sample_indexes:
            new_sample_index = sample_index_remap[old_sample_index]
            sample = self.samples[old_sample_index]

            for edge_index_path in sorted(sample.precursor_edge_indexes):
                precursor_paths.append(
                    tuple(int(edge_index) for edge_index in edge_index_path)
                )
                precursor_path_sample_indexes.append(int(new_sample_index))

        if len(precursor_paths) > 0:
            path_lengths = {len(path) for path in precursor_paths}
            if len(path_lengths) != 1:
                raise ValueError(
                    "All precursor edge-index paths must have the same length. "
                    f"Got lengths: {sorted(path_lengths)}"
                )

            sample_precursor_edge_index_path = torch.tensor(
                precursor_paths,
                dtype=torch.long,
            )

            sample_precursor_path_index = torch.tensor(
                precursor_path_sample_indexes,
                dtype=torch.long,
            )
        else:
            sample_precursor_edge_index_path = torch.empty(
                (0, 0),
                dtype=torch.long,
            )
            sample_precursor_path_index = torch.empty(
                (0,),
                dtype=torch.long,
            )

        return FragmentTreeStructure(
            node_smiles=node_smiles,
            node_graph=node_graph,
            node_graph_offset=node_graph_offset,
            edge_index=edge_index,
            cleavage_event_edge_index=cleavage_event_edge_index,
            cleavage_event=cleavage_event,
            cleavage_atom_idxs=cleavage_atom_idxs,
            sample_adduct_type_index=sample_adduct_type_index,
            sample_ce_value=sample_ce_value,
            sample_edge_index=sample_edge_index,
            sample_precursor_edge_index_path=sample_precursor_edge_index_path,
            sample_precursor_path_index=sample_precursor_path_index,
            reactant_tuple_length_table=reactant_tuple_length_table,
            product_tuple_length_table=product_tuple_length_table,
        )

    def add_same_smiles_dataset_first_cleavage(
        self,
        dataset,
        *,
        precursor_mz_column: str = "PrecursorMZ",
        adduct_type_column: str = "AdductType",
        collision_energy_column: str = "CollisionEnergy",
        smiles_column: str = "SMILES",
        instrument_column: Optional[str] = None,
    ) -> np.ndarray:
        """
        Add multiple spectrum records with the same SMILES as samples.

        This method:
            - validates that all records have the same SMILES
            - builds FragmentIonTree only once
            - caches precursor FragmentPathwayGroup by adduct type
            - registers precursor / first-cleavage fragment nodes
            - registers first-cleavage edges and cleavage events
            - adds each record as one FragmentTreeSample

        Returns
        -------
        np.ndarray
            Added sample indexes.
        """
        if len(dataset) == 0:
            raise ValueError("dataset must not be empty.")

        smiles = dataset[smiles_column].unique()
        if len(smiles) != 1:
            raise ValueError(
                "All records in the dataset must have the same SMILES. "
                f"Got unique SMILES: {smiles}"
            )
        smiles = str(smiles[0])

        compound = Compound.from_smiles(smiles)

        # -------------------------
        # Build shared FragmentIonTree only once
        # -------------------------
        fragment_ion_tree = self._model._fragmenter.build_fragment_ion_tree(
            compound=compound,
            max_depth=self._model.precursor_candidate_max_depth,
            _include_fragment_compound_cache=True,
        )

        fragment_compound_by_index = (
            fragment_ion_tree._fragment_compound_by_index
            if hasattr(fragment_ion_tree, "_fragment_compound_by_index")
            else None
        )

        fragment_compound_by_smiles: Dict[str, Compound] = {}
        tree_node_index_by_smiles: Dict[str, int] = {}
        src_smiles_to_outgoing_fragment_edges: Dict[
            str,
            List[Tuple[str, FragmentPathwayEdge]],
        ] = {}
        cleavage_results_by_smiles: Dict[str, List[CleavageResult]] = {}

        if fragment_compound_by_index is not None:
            for node_index, compound in fragment_compound_by_index.items():
                fragment_compound_by_smiles[compound.smiles] = compound
                tree_node_index_by_smiles[compound.smiles] = node_index

        def get_compound_by_smiles(smiles: str) -> Compound:
            compound = fragment_compound_by_smiles.get(smiles)
            if compound is None:
                compound = Compound.from_smiles(smiles)
                fragment_compound_by_smiles[smiles] = compound
            return compound

        def get_tree_node_index_by_smiles(smiles: str) -> Optional[int]:
            tree_node_index = tree_node_index_by_smiles.get(smiles)
            if tree_node_index is None:
                tree_node = fragment_ion_tree.get_node_by_smiles(smiles)
                if tree_node is not None:
                    tree_node_index = int(tree_node.index)
                    tree_node_index_by_smiles[smiles] = tree_node_index
                else:
                    tree_node_index = None
            return tree_node_index

        def get_cleavage_results_by_smiles(smiles: str) -> List[CleavageResult]:
            cleavage_results = cleavage_results_by_smiles.get(smiles)
            if cleavage_results is None:
                compound = get_compound_by_smiles(smiles)
                cleavage_results = (
                    self._model.fragmenter.cleavage_pattern_set.fragment_all(
                        compound
                    )
                )
                cleavage_results_by_smiles[smiles] = cleavage_results
            return cleavage_results

        def get_outgoing_fragment_edges_by_src_smiles(
            src_smiles: str,
            fragment_ion_tree: FragmentIonTree,
        ) -> List[Tuple[str, FragmentPathwayEdge]]:
            dst_smiles_and_edge_list = src_smiles_to_outgoing_fragment_edges.get(
                src_smiles
            )

            if dst_smiles_and_edge_list is None:
                dst_smiles_and_edge_list = []

                src_node = fragment_ion_tree.get_node_by_smiles(src_smiles)
                if src_node is None:
                    raise ValueError(
                        f"Source SMILES {src_smiles!r} not found in "
                        "fragment ion tree."
                    )

                src_node_index = int(src_node.index)
                out_edges = fragment_ion_tree.get_out_edges(src_node_index)

                if len(out_edges) > 0:
                    for out_edge in out_edges:
                        dst_node_index = out_edge.target_index
                        dst_node = fragment_ion_tree.get_node(dst_node_index)
                        dst_smiles = dst_node.smiles
                        fragment_pathway_edge = (
                            FragmentPathwayEdge.from_fragment_edge(out_edge)
                        )
                        dst_smiles_and_edge_list.append(
                            (
                                dst_smiles,
                                fragment_pathway_edge,
                            )
                        )
                else:
                    dst_smiles_to_cleavage_steps: Dict[
                        str,
                        List[CleavageStep],
                    ] = {}

                    cleavage_results = get_cleavage_results_by_smiles(src_smiles)

                    for cleavage_result in cleavage_results:
                        for product in cleavage_result.products:
                            for product_molecule in product.product_molecules:
                                dst_smiles = product_molecule.compound.smiles

                                if dst_smiles not in dst_smiles_to_cleavage_steps:
                                    dst_smiles_to_cleavage_steps[dst_smiles] = []

                                cleavage_step = CleavageStep(
                                    cleavage_pattern_id=cleavage_result.pattern_id,
                                    reaction_id=product.id,
                                    product_molecule_id=product_molecule.id,
                                    reactant_indices=tuple(product.reactant_indices),
                                    product_indices=tuple(
                                        product_molecule.product_indices
                                    ),
                                )

                                dst_smiles_to_cleavage_steps[dst_smiles].append(
                                    cleavage_step
                                )

                    for dst_smiles, cleavage_steps in (
                        dst_smiles_to_cleavage_steps.items()
                    ):
                        dst_smiles_and_edge_list.append(
                            (
                                dst_smiles,
                                FragmentPathwayEdge(tuple(cleavage_steps)),
                            )
                        )

                src_smiles_to_outgoing_fragment_edges[src_smiles] = list(
                    dst_smiles_and_edge_list
                )

            return dst_smiles_and_edge_list

        # -------------------------
        # Cache precursor pathways by adduct type
        # -------------------------
        precursor_fragment_pathways_by_adduct: Dict[str, FragmentPathwayGroup] = {}

        sample_indexes: List[int] = []

        for record_index in range(len(dataset)):
            record = dataset[record_index]

            try:
                (
                    smiles,
                    precursor_mz,
                    adduct_type,
                    adduct_type_index,
                    ce_value,
                    instrument,
                ) = self._parse_record_info(
                    record,
                    precursor_mz_column=precursor_mz_column,
                    adduct_type_column=adduct_type_column,
                    collision_energy_column=collision_energy_column,
                    smiles_column=smiles_column,
                    instrument_column=instrument_column,
                )
            except Exception:
                sample_indexes.append(-1)
                continue

            sample_index = len(self.samples)

            adduct_type_key = str(adduct_type)

            if adduct_type_key not in precursor_fragment_pathways_by_adduct:
                precursor_fragment_pathways, _ = (
                    self._model.fragmenter.assign_fragment_pathways_to_peaks(
                        fragment_ion_tree=fragment_ion_tree,
                        precursor_type=adduct_type,
                        peaks_mz=[],
                    )
                )

                precursor_fragment_pathways_by_adduct[adduct_type_key] = (
                    precursor_fragment_pathways
                )

            precursor_fragment_pathways = (
                precursor_fragment_pathways_by_adduct[adduct_type_key]
            )

            sample = FragmentTreeSample(
                adduct_type_index=int(adduct_type_index),
                ce_value=float(ce_value),
            )

            precursor_edge_index_paths = self._fragment_pathway_group_to_edge_index_paths(
                fragment_pathway_group=precursor_fragment_pathways,
                padding_length=self._model.fragmenter.precursor_candidate_max_depth,
                fragment_compound_by_smiles=fragment_compound_by_smiles,
            )
            sample.precursor_edge_indexes.update(precursor_edge_index_paths)

            # -------------------------
            # Add first-cleavage edges for this sample
            # -------------------------
            for precursor_node in precursor_fragment_pathways.precursor_nodes:
                tree_precursor_node = fragment_ion_tree.get_node_by_smiles(
                    precursor_node.smiles
                )

                if tree_precursor_node is None:
                    raise ValueError(
                        f"Precursor node SMILES {precursor_node.smiles!r} "
                        "not found in fragment ion tree."
                    )

                for dst_smiles, fp_edge in get_outgoing_fragment_edges_by_src_smiles(
                    precursor_node.smiles,
                    fragment_ion_tree,
                ):
                    edge_index = self._ensure_get_edge_index(
                        precursor_node.smiles,
                        dst_smiles,
                        fp_edge,
                    )

                    if edge_index is None:
                        continue

                    edge_index = int(edge_index)

                    sample.edge_indexes.add(edge_index)

                self.samples[int(sample_index)] = sample
                sample_indexes.append(int(sample_index))

        return np.asarray(sample_indexes, dtype=int)

    def reset(self) -> None:
        """Reset all accumulated structure and sample information.

        Notes
        -----
        This method keeps self._model unchanged.
        """

        # -------------------------
        # Nodes
        # -------------------------
        self.node_smiles = []
        self.node_graph = []
        self._node_graph_offset = [0]

        # -------------------------
        # Edges
        # -------------------------
        self.edge_src = []
        self.edge_dst = []

        # -------------------------
        # Cleavage events
        # -------------------------
        self.cleavage_event_edge_index = []
        self.cleavage_event = []

        # -------------------------
        # Shared cleavage atom-index table
        # -------------------------
        self.cleavage_atom_idxs = {}

        # -------------------------
        # Samples
        # -------------------------
        self.samples = {}

        # -------------------------
        # Private lookup tables
        # -------------------------
        self._node_index_by_smiles = {}
        self._edge_index_by_node_indexes = {}
        self._cleavage_event_index_by_key = {}
        self._atom_row_index_by_atom_idxs = {}

    # -------------------------
    # internal helpers
    # -------------------------
    def _get_mol_graph(
        self,
        smiles: str,
        compound: Optional[Compound] = None,
    ) -> Data:
        builder = (
            self._model._mol_encoder.graph_builder
            if self._model is not None
            else None
        )

        if builder is None:
            raise ValueError(
                "mol_graph_builder is not set. "
                "Set model.mol_encoder.graph_builder to auto-add nodes."
            )

        if compound is None:
            compound = Compound.from_smiles(smiles)

        graph = builder.build(compound)

        if not isinstance(graph, Data):
            raise TypeError("mol_graph_builder must return torch_geometric.data.Data")

        return graph

    def _get_node_index(
        self,
        smiles: str,
        compound: Optional[Compound] = None,
    ) -> int:
        if smiles in self._node_index_by_smiles:
            return self._node_index_by_smiles[smiles]

        raise KeyError(f"Node with SMILES {smiles!r} not found.")

    def _ensure_get_node_index(
        self,
        smiles: Optional[str],
        compound: Optional[Compound] = None,
    ) -> Optional[int]:
        if smiles is None:
            return None

        if smiles in self._node_index_by_smiles:
            return self._node_index_by_smiles[smiles]

        node_index = len(self.node_smiles)
        graph = self._get_mol_graph(smiles, compound)

        self.node_smiles.append(smiles)
        self.node_graph.append(graph)

        prev = self._node_graph_offset[-1]
        num_atoms = graph.num_nodes
        self._node_graph_offset.append(prev + num_atoms)

        self._node_index_by_smiles[smiles] = node_index

        return node_index

    def _get_edge_index(
        self,
        src_smiles: str,
        dst_smiles: str,
    ) -> int:
        src_index = self._get_node_index(src_smiles)
        dst_index = self._get_node_index(dst_smiles)

        key = (
            src_index,
            dst_index,
        )

        if key in self._edge_index_by_node_indexes:
            return self._edge_index_by_node_indexes[key]

        raise KeyError(
            f"Edge from {src_smiles!r} to {dst_smiles!r} not found. "
            f"Source node index: {src_index}, destination node index: {dst_index}."
        )

    def _ensure_get_edge_index(
        self,
        src_smiles: str,
        dst_smiles: str,
        fragment_pathway_edge: FragmentPathwayEdge,
    ) -> Optional[int]:
        """Ensure edge and its cleavage events are registered.

        This method registers:
            - source node
            - destination node
            - directed edge
            - atom-index rows
            - cleavage events

        Returns
        -------
        Optional[int]
            edge_index.
        """

        src_index = self._ensure_get_node_index(src_smiles)
        dst_index = self._ensure_get_node_index(dst_smiles)

        if src_index is None or dst_index is None:
            return None

        key = (
            src_index,
            dst_index,
        )

        if key in self._edge_index_by_node_indexes:
            edge_index = self._edge_index_by_node_indexes[key]
        else:
            edge_index = len(self.edge_src)
            self._edge_index_by_node_indexes[key] = edge_index
            self.edge_src.append(src_index)
            self.edge_dst.append(dst_index)

        # -------------------------
        # Register cleavage events
        # -------------------------
        for cleavage_step in fragment_pathway_edge.steps:
            cleavage_id = int(cleavage_step.cleavage_pattern_id)
            reaction_id = int(cleavage_step.reaction_id)
            product_molecule_id = int(cleavage_step.product_molecule_id)

            reactant_atom_idxs = tuple(
                int(atom_index)
                for atom_index in cleavage_step.reactant_indices
            )
            # Local atom indices in the source fragment SMILES.

            product_atom_idxs = tuple(
                int(atom_index)
                for atom_index in cleavage_step.product_indices
            )
            # Local atom indices in the destination fragment SMILES.

            reactant_row_index = self._ensure_get_atom_row_index(
                reactant_atom_idxs,
            )

            product_row_index = self._ensure_get_atom_row_index(
                product_atom_idxs,
            )

            self._ensure_get_cleavage_event_index(
                edge_index=edge_index,
                cleavage_id=cleavage_id,
                reaction_id=reaction_id,
                product_molecule_id=product_molecule_id,
                reactant_row_index=reactant_row_index,
                product_row_index=product_row_index,
            )

        return edge_index

    def _get_node_edge_index(
        self,
        src_smiles: str,
        dst_smiles: str,
    ) -> Tuple[int, int, int]:
        src_index = self._get_node_index(src_smiles)
        dst_index = self._get_node_index(dst_smiles)
        edge_index = self._get_edge_index(src_smiles, dst_smiles)

        return src_index, dst_index, edge_index

    def _ensure_get_node_edge_index(
        self,
        src_smiles: str,
        dst_smiles: str,
        fragment_pathway_edge: FragmentPathwayEdge,
    ) -> Optional[Tuple[int, int, int]]:
        src_index = self._ensure_get_node_index(src_smiles)
        dst_index = self._ensure_get_node_index(dst_smiles)

        edge_index = self._ensure_get_edge_index(
            src_smiles,
            dst_smiles,
            fragment_pathway_edge,
        )

        if src_index is None or dst_index is None or edge_index is None:
            return None

        return src_index, dst_index, edge_index

    def _get_atom_row_index(
        self,
        atom_idxs: Sequence[int],
    ) -> int:
        atom_idx_tuple = tuple(int(i) for i in atom_idxs)

        if atom_idx_tuple in self._atom_row_index_by_atom_idxs:
            return self._atom_row_index_by_atom_idxs[atom_idx_tuple]

        raise KeyError(
            f"Atom index tuple {atom_idx_tuple} not found."
        )

    def _ensure_get_atom_row_index(
        self,
        atom_idxs: Sequence[int],
    ) -> int:
        """Ensure an atom-index tuple is registered.

        Parameters
        ----------
        atom_idxs:
            Global atom indexes.

        Returns
        -------
        int
            Row index in cleavage_atom_idxs[tuple_length].
        """

        atom_idx_tuple = tuple(int(i) for i in atom_idxs)
        tuple_length = len(atom_idx_tuple)

        if atom_idx_tuple in self._atom_row_index_by_atom_idxs:
            return self._atom_row_index_by_atom_idxs[atom_idx_tuple]

        if tuple_length not in self.cleavage_atom_idxs:
            self.cleavage_atom_idxs[tuple_length] = []

        row_index = len(self.cleavage_atom_idxs[tuple_length])
        self.cleavage_atom_idxs[tuple_length].append(atom_idx_tuple)

        self._atom_row_index_by_atom_idxs[atom_idx_tuple] = row_index

        return row_index

    def _get_cleavage_event_index(
        self,
        *,
        edge_index: int,
        cleavage_id: int,
        reaction_id: int,
        product_molecule_id: int,
        reactant_row_index: int,
        product_row_index: int,
    ) -> int:
        key = (
            int(edge_index),
            int(cleavage_id),
            int(reaction_id),
            int(product_molecule_id),
            int(reactant_row_index),
            int(product_row_index),
        )

        if key in self._cleavage_event_index_by_key:
            return self._cleavage_event_index_by_key[key]

        raise KeyError(
            f"Cleavage event not found for edge_index={edge_index}, "
            f"cleavage_id={cleavage_id}, reaction_id={reaction_id}, "
            f"product_molecule_id={product_molecule_id}, "
            f"reactant_row_index={reactant_row_index}, "
            f"product_row_index={product_row_index}."
        )

    def _ensure_get_cleavage_event_index(
        self,
        *,
        edge_index: int,
        cleavage_id: int,
        reaction_id: int,
        product_molecule_id: int,
        reactant_row_index: int,
        product_row_index: int,
    ) -> int:
        """Ensure a cleavage event is registered.

        Returns
        -------
        int
            cleavage_event_index.
        """

        key = (
            int(edge_index),
            int(cleavage_id),
            int(reaction_id),
            int(product_molecule_id),
            int(reactant_row_index),
            int(product_row_index),
        )

        if key in self._cleavage_event_index_by_key:
            return self._cleavage_event_index_by_key[key]

        cleavage_event_index = len(self.cleavage_event)

        self.cleavage_event_edge_index.append(int(edge_index))
        self.cleavage_event.append(
            (
                int(cleavage_id),
                int(reaction_id),
                int(product_molecule_id),
                int(reactant_row_index),
                int(product_row_index),
            )
        )

        self._cleavage_event_index_by_key[key] = cleavage_event_index

        return cleavage_event_index

    def _local_to_global_atom_idxs(
        self,
        node_index: int,
        atom_idxs: Sequence[int],
    ) -> Tuple[int, ...]:
        node_index = int(node_index)

        if node_index < 0 or node_index >= len(self.node_smiles):
            raise IndexError(
                f"node_index is out of range: {node_index}. "
                f"Expected 0 <= node_index < {len(self.node_smiles)}."
            )

        offset = self._node_graph_offset[node_index]

        return tuple(int(i) + offset for i in atom_idxs)

    def _add_fragment_pathway_edge_to_structure(
        self,
        *,
        fragment_pathway_src_node: Optional[FragmentPathwayNode],
        fragment_pathway_dst_node: FragmentPathwayNode,
        fragment_pathway_edge: Optional[FragmentPathwayEdge],
        fragment_compound_by_smiles: Optional[Dict[str, Compound]] = None,
    ) -> Optional[int]:
        """Add one FragmentPathwayEdge to the shared tree structure.

        This method registers:
            - node
            - edge
            - atom-index rows
            - cleavage events

        It does not update self.samples.
        """

        src_smiles = (
            fragment_pathway_src_node.smiles
            if fragment_pathway_src_node is not None
            else None
        )
        dst_smiles = fragment_pathway_dst_node.smiles

        if src_smiles is None or fragment_pathway_edge is None:
            return None

        edge_index = self._ensure_get_edge_index(
            src_smiles,
            dst_smiles,
            fragment_pathway_edge,
        )

        return edge_index

    def _fragment_pathway_to_edge_index_path(
        self,
        *,
        fragment_pathway: FragmentPathway,
        padding_length: int,
        fragment_compound_by_smiles: Optional[Dict[str, Compound]] = None,
    ) -> Tuple[int, ...]:
        """Convert one FragmentPathway into one edge_index tuple."""

        edge_indexes: List[int] = []

        for path_index in range(1, len(fragment_pathway)):
            fragment_pathway_src_node = fragment_pathway.get_node(path_index - 1)
            fragment_pathway_edge = fragment_pathway.get_edge(path_index - 1)
            fragment_pathway_dst_node = fragment_pathway.get_node(path_index)

            edge_index = self._add_fragment_pathway_edge_to_structure(
                fragment_pathway_src_node=fragment_pathway_src_node,
                fragment_pathway_dst_node=fragment_pathway_dst_node,
                fragment_pathway_edge=fragment_pathway_edge,
                fragment_compound_by_smiles=fragment_compound_by_smiles,
            )

            if edge_index is not None:
                edge_indexes.append(int(edge_index))

        edge_indexes.extend(
            -1
            for _ in range(padding_length - len(fragment_pathway) + 1)
        )

        return tuple(edge_indexes)

    def _fragment_pathway_group_to_edge_index_paths(
        self,
        *,
        fragment_pathway_group: FragmentPathwayGroup,
        padding_length: int,
        fragment_compound_by_smiles: Optional[Dict[str, Compound]] = None,
    ) -> Set[Tuple[int, ...]]:
        """Convert precursor FragmentPathwayGroup into edge_index paths.

        Returns
        -------
        Set[Tuple[int, ...]]
            Set of edge_index tuples.
            Each tuple represents one complete precursor fragment pathway.
        """

        edge_index_paths: Set[Tuple[int, ...]] = set()

        for fragment_pathway in fragment_pathway_group:
            edge_index_path = self._fragment_pathway_to_edge_index_path(
                fragment_pathway=fragment_pathway,
                padding_length=padding_length,
                fragment_compound_by_smiles=fragment_compound_by_smiles,
            )

            if len(edge_index_path) > 0:
                edge_index_paths.add(edge_index_path)

        return edge_index_paths

    @staticmethod
    def _tuple_length_dict_to_tensor(
        tuple_length_by_key: Dict[Tuple[int, ...], int],
        *,
        key_width: int,
    ) -> torch.Tensor:
        """Convert tuple-length lookup dict to a sorted tensor table.

        Parameters
        ----------
        tuple_length_by_key:
            Mapping from event-type key to tuple length.

            Reactant:
                (cleavage_id, reaction_id) -> reactant_tuple_length

            Product:
                (cleavage_id, reaction_id, product_molecule_id)
                -> product_tuple_length

        key_width:
            Number of elements in the event-type key.

            Reactant:
                key_width = 2

            Product:
                key_width = 3

        Returns
        -------
        Tensor
            Reactant:
                [R, 3]
                columns:
                    0: cleavage_id
                    1: reaction_id
                    2: reactant_tuple_length

            Product:
                [P, 4]
                columns:
                    0: cleavage_id
                    1: reaction_id
                    2: product_molecule_id
                    3: product_tuple_length
        """

        rows: list[tuple[int, ...]] = []

        for key, tuple_length in tuple_length_by_key.items():
            key = tuple(int(value) for value in key)

            if len(key) != key_width:
                raise ValueError(
                    f"Expected key width {key_width}, "
                    f"but got key={key}."
                )

            rows.append(
                key + (int(tuple_length),)
            )

        rows.sort()

        if len(rows) == 0:
            return torch.empty(
                (0, key_width + 1),
                dtype=torch.long,
            )

        return torch.tensor(
            rows,
            dtype=torch.long,
        )

    def _parse_record_info(
        self,
        record: SpectrumRecord,
        *,
        precursor_mz_column: str,
        adduct_type_column: str,
        collision_energy_column: str,
        smiles_column: str = "smiles",
        instrument_column: Optional[str] = None,
    ) -> Tuple[str, float, Adduct, int, float, Optional[str]]:
        """
        Parse commonly used metadata from one spectrum record.

        This method extracts:
            - precursor m/z
            - adduct type
            - adduct type index
            - collision energy in eV
            - SMILES
            - optional instrument
        """
        precursor_mz = float(record[precursor_mz_column])

        adduct_type_str = str(record[adduct_type_column])
        adduct_type = Adduct.parse(adduct_type_str)

        ce_value_raw = record[collision_energy_column]

        instrument = None
        if instrument_column is not None:
            instrument = record[instrument_column]

        adduct_type_index = self._model.get_index_by_adduct_type(adduct_type)

        ce_value = CleftsSpecGen.parse_ce_to_ev(
            ce_value_raw,
            precursor_mz=precursor_mz,
            instrument=instrument,
        )

        if ce_value is None:
            raise ValueError(
                f"Failed to parse collision energy: {ce_value_raw!r} "
                f"for SMILES={record[smiles_column]!r}, "
                f"precursor_mz={precursor_mz}, "
                f"adduct_type={adduct_type}."
            )

        smiles = str(record[smiles_column])

        return smiles, precursor_mz, adduct_type, adduct_type_index, ce_value, instrument