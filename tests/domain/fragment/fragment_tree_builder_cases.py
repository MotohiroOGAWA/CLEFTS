from __future__ import annotations

from dataclasses import dataclass

from clefts.domain.fragment.cleavage.CleavagePattern import (
    ProductRule,
    _CleavagePattern,
)
from clefts.domain.fragment.cleavage.CleavagePatternSet import (
    CleavagePatternSet,
)
from clefts.domain.fragment.fragment_tree.FragmentTreeBuilder import (
    FragmentTreeBuilder,
)


@dataclass(frozen=True)
class ExpectedFragmentTreeEdge:
    """Expected edge in a generated FragmentTree."""

    source_smiles: str
    target_smiles: str
    min_event_count: int = 1


@dataclass(frozen=True)
class FragmentTreeBuilderCase:
    """Expected FragmentTree for one input compound."""

    name: str
    smiles: str

    expected_node_smiles_by_depth: dict[int, tuple[str, ...]]
    expected_edges: tuple[ExpectedFragmentTreeEdge, ...]

    expected_num_nodes: int
    expected_num_edges: int
    expected_min_events: int


@dataclass(frozen=True)
class FragmentTreeBuilderQuestion:
    """One test question for FragmentTreeBuilder.

    A question is a pair of:
    - FragmentTreeBuilder
    - FragmentTreeBuilderCase list

    This allows each question to define not only cleavage patterns,
    but also builder options such as max_depth and only_add_min_depth.
    """

    name: str
    builder: FragmentTreeBuilder
    cases: tuple[FragmentTreeBuilderCase, ...]


def make_fragment_tree_builder_questions(
) -> tuple[FragmentTreeBuilderQuestion, ...]:
    """Return all FragmentTreeBuilder questions.

    Future questions should be added here.
    """

    return (
        make_single_bond_cleavage_question(),
    )


def make_single_bond_cleavage_question(
) -> FragmentTreeBuilderQuestion:
    builder = FragmentTreeBuilder(
        max_depth=1,
        cleavage_pattern_set=_make_single_bond_cleavage_pattern_set(),
        only_add_min_depth=True,
        min_depth_only_from=0,
    )

    cases = (
        FragmentTreeBuilderCase(
            name="ethanol",
            smiles="CCO",
            expected_node_smiles_by_depth={
                0: ("CCO",),
                1: (
                    "C",
                    "CO",
                    "CC",
                    "O",
                ),
            },
            expected_edges=(
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="C",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="CO",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="CC",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="O",
                    min_event_count=1,
                ),
            ),
            expected_num_nodes=5,
            expected_num_edges=4,
            expected_min_events=4,
        ),
        FragmentTreeBuilderCase(
            name="1-propanol",
            smiles="CCCO",
            expected_node_smiles_by_depth={
                0: ("CCCO",),
                1: (
                    "C",
                    "CCO",
                    "CC",
                    "CO",
                    "CCC",
                    "O",
                ),
            },
            expected_edges=(
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="C",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CCO",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CC",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CO",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CCC",
                    min_event_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="O",
                    min_event_count=1,
                ),
            ),
            expected_num_nodes=7,
            expected_num_edges=6,
            expected_min_events=6,
        ),
        FragmentTreeBuilderCase(
            name="diethyl_ether",
            smiles="CCOCC",
            expected_node_smiles_by_depth={
                0: ("CCOCC",),
                1: (
                    "C",
                    "CCOC",
                    "CC",
                    "CCO",
                ),
            },
            expected_edges=(
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="C",
                    min_event_count=2,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="CCOC",
                    min_event_count=2,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="CC",
                    min_event_count=2,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="CCO",
                    min_event_count=2,
                ),
            ),
            expected_num_nodes=5,
            expected_num_edges=4,
            expected_min_events=8,
        ),
    )

    return FragmentTreeBuilderQuestion(
        name="non_hydrogen_single_bond_cleavage",
        builder=builder,
        cases=cases,
    )

def _make_single_bond_cleavage_pattern(
) -> _CleavagePattern:

    return _CleavagePattern.from_rules(
        reactant_smarts="[!#1:1]-[!#1:2]",
        products=(
            ProductRule(
                name="single_bond_cleavage",
                smarts="[!#1:1].[!#1:2]",
            ),
        ),
        name="single_bond_cleavage",
    )

def _make_single_bond_cleavage_pattern_set(
) -> CleavagePatternSet:
    pattern = _make_single_bond_cleavage_pattern()

    return CleavagePatternSet.from_patterns(
        name="single_bond_cleavage_pattern_set",
        patterns=(pattern,),
    )