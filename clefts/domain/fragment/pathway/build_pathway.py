from __future__ import annotations

from typing import Tuple, List, Dict, Iterable, Iterator
from collections import deque

from ....libs.mmkit.mmkit import Adduct

from ..cleavage.CleavageActionSequence import CleavageActionSequence
from .CleavageStep import CleavageStep
from .FragmentPathway import (
    FragmentPathway,
    FragmentPathwayNode,
    FragmentPathwayEdge,
)
from ..tree.FragmentTree import FragmentTree

PathwayItem = FragmentPathwayNode | FragmentPathwayEdge


def _iter_precursor_containing_paths(
    fragment_tree: FragmentTree,
    target_node_index: int,
    precursor_adduct_types: Dict[int, Iterable[Adduct]],
    *,
    max_action_count: int,
    precursor_candidate_max_action_count: int,
) -> Iterator[Tuple[Tuple[int, ...], frozenset, CleavageActionSequence | None]]:
    """Yield every shortest Source history reaching target_node_index that
    passes through an already-selected precursor node.

    Shared by pathway-item presentation (build_pathway_items_for_node) and
    action-sequence resolution (resolve_action_sequences_for_node), so both
    consumers walk the exact same fragment_tree exactly once.
    """
    # Traverse compatible Source histories for presentation. Edge distance is
    # never used as an action budget, and no intermediate molecule is reacted.
    root_is_precursor = 0 in precursor_adduct_types
    queue = deque([(0, None, (0,), frozenset((0,)) if root_is_precursor else frozenset())])
    best_depth = {(0, None, root_is_precursor): 0}
    shortest = None
    while queue:
        node_index, sequence, path, precursor_positions = queue.popleft()
        depth = (len(path) - 1) // 2
        if shortest is not None and depth > shortest:
            break
        if node_index == target_node_index and precursor_positions:
            shortest = depth
            yield path, precursor_positions, sequence
            continue
        for edge in fragment_tree.get_out_edges(node_index):
            for transition in edge.transitions:
                if transition.parent_action_sequence != sequence:
                    continue
                child_sequence = transition.action_sequence
                count = len(child_sequence.actions)
                if count > max_action_count:
                    continue
                qualifies = (edge.target_index in precursor_adduct_types
                             and count <= precursor_candidate_max_action_count)
                positions = precursor_positions | (frozenset((len(path) + 1,)) if qualifies else frozenset())
                state = (edge.target_index, child_sequence, bool(positions))
                if best_depth.get(state, depth + 1) < depth + 1:
                    continue
                best_depth[state] = depth + 1
                queue.append((edge.target_index, child_sequence,
                              (*path, edge.index, edge.target_index), positions))


def resolve_action_sequences_for_node(
    fragment_tree: FragmentTree,
    target_node_index: int,
    precursor_adduct_types: Dict[int, Iterable[Adduct]],
    *,
    max_action_count: int,
    precursor_candidate_max_action_count: int,
) -> frozenset[CleavageActionSequence | None]:
    """The exact Source CleavageActionSequence(s) realizing every shortest,
    precursor-containing history to target_node_index. None means Source
    itself already satisfies the requirement (target_node_index == 0)."""
    return frozenset(sequence for _, _, sequence in _iter_precursor_containing_paths(
        fragment_tree, target_node_index, precursor_adduct_types,
        max_action_count=max_action_count,
        precursor_candidate_max_action_count=precursor_candidate_max_action_count))


def build_pathway_items_for_node(
    fragment_tree: FragmentTree,
    target_node_index: int,
    precursor_adduct_types: Dict[int, Iterable[Adduct]],
    *,
    max_action_count: int,
    precursor_candidate_max_action_count: int,
) -> Tuple[Tuple[PathwayItem, ...], ...]:
    """Build pathway items for one target node.

    This function does not assign the final fragment ion adduct.
    It only builds the structural pathway items.

    If a precursor node has multiple possible adduct types, this function
    expands the pathway into multiple item sequences.
    """
    paths = [(path, precursor_positions) for path, precursor_positions, _ in
              _iter_precursor_containing_paths(fragment_tree, target_node_index, precursor_adduct_types,
                  max_action_count=max_action_count,
                  precursor_candidate_max_action_count=precursor_candidate_max_action_count)]

    if not paths:
        return tuple()

    all_pathway_items: List[Tuple[PathwayItem, ...]] = []

    for path, precursor_positions in paths:
        pathway_item_candidates: List[List[PathwayItem]] = [[]]

        for i, path_index in enumerate(path):
            if i % 2 == 0:
                node_index = path_index
                node = fragment_tree.get_node(node_index)

                if i in precursor_positions:
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