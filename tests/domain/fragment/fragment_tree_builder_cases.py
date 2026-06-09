from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clefts.domain.fragment.cleavage._CleavagePattern import (
    ProductRule,
    _CleavagePattern,
)
from clefts.domain.fragment.cleavage.CleavagePatternSet import (
    CleavagePatternSet,
)
from clefts.domain.fragment.tree.FragmentTreeBuilder import (
    FragmentTreeBuilder,
)
from clefts.domain.fragment.ion_tree.FragmentIonAdductRuleSet import (
    FragmentIonAdductRuleSet,
)
from clefts.domain.fragment.ion_tree.FragmentIonTreeBuilder import (
    FragmentIonTreeBuilder,
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

    # Optional expectations for FragmentIonTreeBuilder.
    # Key is fragment SMILES, not node index.
    expected_ion_states_by_smiles: (
        dict[str, tuple[tuple[int, int], ...]] | None
    ) = None

    expected_shift_rule_mask_by_smiles: (
        dict[str, tuple[bool, ...]] | None
    ) = None


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

    fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet | None = None

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
    fragment_ion_adduct_rule_set = _make_hydrogen_rearrangement_fragment_ion_adduct_rule_set()

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

            expected_ion_states_by_smiles={
                "CCO": ((2, 1),),
                "C": ((2, 1),),
                "CO": ((2, 1),),
                "CC": ((2, 1),),
                "O": ((2, 1),),
            },
            expected_shift_rule_mask_by_smiles={
                "CCO": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
                "C": (
                    True, False, True, True, False,
                    True, False, True, False, True,
                ),
                "CO": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
                "CC": (
                    True, False, True, True, False,
                    True, False, True, False, True,
                ),
                "O": (
                    False, True, True, False, True,
                    True, True, False, True, False,
                ),
            },
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

            expected_ion_states_by_smiles={
                "CCCO": ((2, 1),),
                "C": ((2, 1),),
                "CCO": ((2, 1),),
                "CC": ((2, 1),),
                "CO": ((2, 1),),
                "CCC": ((2, 1),),
                "O": ((2, 1),),
            },
            expected_shift_rule_mask_by_smiles={
                "CCCO": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
                "C": (
                    True, False, True, True, False,
                    True, False, True, False, True,
                ),
                "CCO": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
                "CC": (
                    True, False, True, True, False,
                    True, False, True, False, True,
                ),
                "CO": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
                "CCC": (
                    True, False, True, True, False,
                    True, False, True, False, True,
                ),
                "O": (
                    False, True, True, False, True,
                    True, True, False, True, False,
                ),
            },
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

            expected_ion_states_by_smiles={
                "CCOCC": ((2, 1),),
                "C": ((2, 1),),
                "CCOC": ((2, 1),),
                "CC": ((2, 1),),
                "CCO": ((2, 1),),
            },
            expected_shift_rule_mask_by_smiles={
                "CCOCC": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
                "C": (
                    True, False, True, True, False,
                    True, False, True, False, True,
                ),
                "CCOC": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
                "CC": (
                    True, False, True, True, False,
                    True, False, True, False, True,
                ),
                "CCO": (
                    True, True, True, True, True,
                    True, True, True, True, True,
                ),
            },
        ),
    )

    return FragmentTreeBuilderQuestion(
        name="non_hydrogen_single_bond_cleavage",
        builder=builder,
        cases=cases,
        fragment_ion_adduct_rule_set=fragment_ion_adduct_rule_set,
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

def _make_hydrogen_rearrangement_fragment_ion_adduct_rule_set(
) -> FragmentIonAdductRuleSet:
    test_data_dir = Path(__file__).parent / "test_data"

    return FragmentIonAdductRuleSet.from_json(
        test_data_dir / "hydrogen_rearrangement_rule_set_pos.json",
        name="hydrogen_rearrangement_rule_set_pos",
    )