from __future__ import annotations

from typing import Tuple, List, Dict, Optional, Iterable

from ....libs.mmkit.mmkit import Adduct

from .CleavageStep import CleavageStep
from .FragmentPathway import (
    FragmentPathway,
    FragmentPathwayNode,
    FragmentPathwayEdge,
)
from ..tree.FragmentTree import FragmentTree

PathwayItem = FragmentPathwayNode | FragmentPathwayEdge

def build_pathway_items_for_node(
    fragment_tree: FragmentTree,
    target_node_index: int,
    precursor_adduct_types: Dict[int, Iterable[Adduct]],
    *,
    max_depth: int,
    precursor_candidate_max_depth: int,
) -> Tuple[Tuple[PathwayItem, ...], ...]:
    """Build pathway items for one target node.

    This function does not assign the final fragment ion adduct.
    It only builds the structural pathway items.

    If a precursor node has multiple possible adduct types, this function
    expands the pathway into multiple item sequences.
    """

    precursor_node_indices = set(precursor_adduct_types.keys())

    paths = fragment_tree.collect_global_shortest_node_paths_from_root_via(
        target_node_index,
        precursor_node_indices,
        max_depth=max_depth,
        max_via_depth=precursor_candidate_max_depth,
    )

    if len(paths) == 0:
        return tuple()

    all_pathway_items: List[Tuple[PathwayItem, ...]] = []

    for path in paths:
        pathway_item_candidates: List[List[PathwayItem]] = [[]]

        for i, path_index in enumerate(path):
            if i % 2 == 0:
                node_index = path_index
                node = fragment_tree.get_node(node_index)

                if node_index in precursor_adduct_types:
                    adduct_candidates = tuple(
                        precursor_adduct_types.get(node_index, ())
                    )

                    if not adduct_candidates:
                        adduct_candidates = (None,)

                    new_pathway_item_candidates: List[List[PathwayItem]] = []

                    for current_items in pathway_item_candidates:
                        for precursor_adduct_type in adduct_candidates:
                            pathway_node = FragmentPathwayNode(
                                node.smiles,
                                precursor_adduct_type=precursor_adduct_type,
                            )

                            new_pathway_item_candidates.append(
                                current_items + [pathway_node]
                            )

                    pathway_item_candidates = new_pathway_item_candidates

                else:
                    pathway_node = FragmentPathwayNode(node.smiles)

                    for current_items in pathway_item_candidates:
                        current_items.append(pathway_node)

            else:
                edge_index = path_index
                fragment_edge = fragment_tree.get_edge(edge_index)

                cleavage_steps = tuple(
                    CleavageStep(
                        cleavage_pattern_id=cleavage_event.cleavage_pattern_id,
                        reaction_id=cleavage_event.reaction_id,
                        product_molecule_id=cleavage_event.product_molecule_id,
                        reactant_indices=cleavage_event.reactant_indices,
                        product_indices=cleavage_event.product_indices,
                    )
                    for cleavage_event in fragment_edge.events
                )

                pathway_edge = FragmentPathwayEdge(cleavage_steps)

                for current_items in pathway_item_candidates:
                    current_items.append(pathway_edge)

        all_pathway_items.extend(
            tuple(pathway_items)
            for pathway_items in pathway_item_candidates
        )

    return tuple(all_pathway_items)