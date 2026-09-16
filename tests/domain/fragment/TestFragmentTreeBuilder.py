from __future__ import annotations

from itertools import combinations
import random
import unittest
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from clefts.domain.fragment.cleavage import (
    CleavageAction, CleavageActionSequence, CleavagePatternSet, CompositeCleavageReaction,
)
from clefts.domain.fragment.cleavage._CleavagePattern import _CleavagePattern, ProductRule
from clefts.domain.fragment.cleavage.CleavageActionSearch import CleavageActionSearch, _ActionRelations
from clefts.domain.fragment.tree import FragmentTreeBuilder
from clefts.libs.mmkit.mmkit import Compound


def pattern(reactant: str, product: str) -> _CleavagePattern:
    return _CleavagePattern.from_rules(reactant_smarts=reactant,
                                      products=(ProductRule('test', product),))


def builder(*patterns: _CleavagePattern, limit: int = 3) -> FragmentTreeBuilder:
    return FragmentTreeBuilder(limit, CleavagePatternSet.from_patterns(patterns),
                               only_add_min_depth=False)


def action(
    pattern_id: int,
    roles: tuple[int, ...],
    *,
    retained: tuple[int, ...] = (1, 2, 3, 4),
    matched: tuple[tuple[int, int], ...] = (),
    cuts: tuple[tuple[int, int], ...] = (),
    updates: tuple[tuple[int, int, Chem.BondType], ...] = (),
) -> CleavageAction:
    return CleavageAction(pattern_id, 0, roles, frozenset(retained),
                          frozenset((1, 2, 3, 4)) - frozenset(retained),
                          frozenset(matched), frozenset(cuts), frozenset(updates),
                          product_molecule_id=0)


class TestPrimitiveCleavageActions(unittest.TestCase):
    def test_all_matches_products_and_query_order_without_reactions(self) -> None:
        source = Compound.from_smiles('[CH3:10][CH2:20][OH:30]')
        build = builder(pattern('[C:7]-[C:2]', '[C:7].[C:2]'))
        calls = []
        original = Chem.Mol.GetSubstructMatches

        def matching(mol: Chem.Mol, query: Chem.Mol, **kwargs: object) -> tuple[tuple[int, ...], ...]:
            calls.append(query)
            return original(mol, query, **kwargs)

        with patch.object(Chem.Mol, 'GetSubstructMatches', matching), \
             patch.object(Chem.Mol, 'HasSubstructMatch', side_effect=AssertionError('per-action matching')), \
             patch.object(rdChemReactions.ChemicalReaction, 'RunReactants',
                          side_effect=AssertionError('primitive reaction')):
            actions = build.create_cleavage_actions(source)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(actions), 4)
        self.assertEqual({a.source_atom_maps for a in actions}, {(10, 20), (20, 10)})
        self.assertEqual({a.product_molecule_id for a in actions}, {0, 1})
        self.assertEqual(actions, tuple(sorted(actions, key=lambda a: a.key)))
        a = next(a for a in actions if a.source_atom_maps == (10, 20) and a.product_molecule_id == 1)
        self.assertEqual(a.retained_atom_maps, frozenset((20, 30)))
        self.assertEqual(a.discarded_atom_maps, frozenset((10,)))
        self.assertEqual(a.cut_bond_maps, frozenset(((10, 20),)))

    def test_template_inspection_once_for_all_matches(self) -> None:
        from clefts.domain.fragment.cleavage import CleavageActionGenerator as generator
        source = Compound.from_smiles('CCCCCC')
        build = builder(pattern('[C:1]-[C:2]', '[C:1]'))
        with patch.object(generator, '_action_templates', wraps=generator._action_templates) as parsed:
            actions = build.create_cleavage_actions(source)
        self.assertEqual(parsed.call_count, 1)
        self.assertEqual(len(actions), 10)

    def test_whole_reaction_cuts_and_product_order_update(self) -> None:
        source = Compound.from_smiles('[CH3:10][CH2:20][CH3:30]')
        build = builder(pattern('[C:7]-[C:2]-[C:9]', '[C:7]=[C:2].[C:9]'))
        actions = build.create_cleavage_actions(source)
        a = next(a for a in actions if a.source_atom_maps == (10, 20, 30)
                 and a.product_molecule_id == 0)
        self.assertEqual(a.retained_atom_maps, frozenset((10, 20)))
        self.assertEqual(a.cut_bond_maps, frozenset(((20, 30),)))
        self.assertEqual(a.bond_updates, frozenset(((10, 20, Chem.BondType.DOUBLE),)))
        self.assertEqual(a.changed_bond_maps, frozenset(((10, 20), (20, 30))))
        self.assertEqual(Compound(CleavageActionSequence((a,)).compile(source).run(source)[0]).smiles, 'C=C')

    def test_multiple_anchor_components_and_creation_are_unsupported(self) -> None:
        source = Compound.from_smiles('CC')
        for rhs in ('([C:1].[C:2])', '[C:1]-[C:2]-[O:3]'):
            with self.subTest(rhs=rhs):
                self.assertEqual(builder(pattern('[C:1]-[C:2]', rhs)).create_cleavage_actions(source), ())
        self.assertEqual(builder(pattern('[C:1]-[C:2]-[C:3]', '[C:1]1[C:2][C:3]1'))
                         .create_cleavage_actions(Compound.from_smiles('CCC')), ())

    def test_unspecified_atom_charge_is_preserved(self) -> None:
        source = Compound.from_smiles('[CH3:1][O-:2]')
        build = builder(pattern('[O-:1]-[C:2]', '[O:1]'))
        actions = build.create_cleavage_actions(source)
        self.assertEqual(len(actions), 1)
        self.assertEqual(build.cleave_all(source)[0].compound.charge, -1)
        self.assertEqual(builder(pattern('[O-:1]-[C:2]', '[O+0:1]'))
                         .create_cleavage_actions(source), ())
        self.assertEqual(builder(pattern('[O-:1]-[C:2]', '[N:1]'))
                         .create_cleavage_actions(source), ())

    def test_each_pattern_matches_once(self) -> None:
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'), pattern('[C:1]-[N:2]', '[C:1]'))
        original = Chem.Mol.GetSubstructMatches
        calls = []

        def matching(mol: Chem.Mol, query: Chem.Mol, **kwargs: object) -> tuple[tuple[int, ...], ...]:
            calls.append(query)
            return original(mol, query, **kwargs)

        with patch.object(Chem.Mol, 'GetSubstructMatches', matching):
            build.build(Compound.from_smiles('NC(O)C'))
        self.assertEqual(len(calls), 2)


class TestCleavageActionCombinationSearch(unittest.TestCase):
    def test_independent_three_actions_without_permutations_or_rdkit(self) -> None:
        actions = tuple(action(i, edge, matched=(edge,), cuts=(edge,))
                        for i, edge in enumerate(((1, 2), (2, 3), (3, 4))))
        search = CleavageActionSearch(actions, max_action_count=3)
        with patch.object(Chem.Mol, 'GetSubstructMatches', side_effect=AssertionError('matching')), \
             patch.object(rdChemReactions.ChemicalReaction, 'RunReactants', side_effect=AssertionError('reaction')):
            candidates = search.enumerate()
        sequences = {c.action_sequence for c in candidates}
        self.assertEqual(len(sequences), 7)
        self.assertEqual(sum(len(s.actions) == 3 for s in sequences), 1)
        self.assertEqual(CleavageActionSequence(actions), CleavageActionSequence(reversed(actions)))
        self.assertEqual(len(candidates), 7)

    def test_search_matches_small_exhaustive_combination_oracle(self) -> None:
        edges = tuple(combinations((1, 2, 3, 4), 2))
        rng = random.Random(0)
        for trial in range(12):
            actions = tuple(action(i, (1, 2, 3, 4),
                                   matched=tuple(edge for edge in edges if edge == cut or rng.random() < 0.25),
                                   cuts=(cut,)) for i, cut in enumerate(edges))
            relations = _ActionRelations.from_actions(actions)
            for limit in (1, 2, 3):
                for seeded in (False, True):
                    with self.subTest(trial=trial, limit=limit, seeded=seeded):
                        expected = set()
                        for count in range(1, limit + 1):
                            for indices in combinations(range(len(actions)), count):
                                if seeded and len(actions) - 1 not in indices:
                                    continue
                                if relations.rejection(indices) is None:
                                    expected.add(CleavageActionSequence(actions[i] for i in indices))
                        seeds = (CleavageActionSequence((actions[-1],)),) if seeded else None
                        search = CleavageActionSearch(actions, max_action_count=limit,
                                                      seed_action_sequences=seeds)
                        actual = {candidate.action_sequence for candidate in search.enumerate()}
                        self.assertEqual(actual, expected)

    def test_hard_conflict_pruned_before_compilation(self) -> None:
        a = action(0, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        b = action(1, (2, 1), matched=((2, 1),), updates=((2, 1, Chem.BondType.DOUBLE),))
        search = CleavageActionSearch((a, b), max_action_count=2)
        candidates = search.enumerate()
        self.assertTrue(all(len(c.action_sequence.actions) == 1 for c in candidates))
        self.assertEqual(search.stats.num_hard_conflict_pruned, 1)

    def test_reverse_ordering_is_allowed_with_precedence(self) -> None:
        a = action(0, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        b = action(1, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((2, 3),))
        relations = _ActionRelations.from_actions((a, b))
        self.assertEqual(relations.must_precede_mask, (0, 1))
        search = CleavageActionSearch((a, b), max_action_count=2)
        self.assertIn(CleavageActionSequence((a, b)), {c.action_sequence for c in search.enumerate()})

    def test_unchanged_context_overlap_is_allowed(self) -> None:
        a = action(0, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((1, 2),))
        b = action(1, (2, 3, 4), matched=((2, 3), (3, 4)), cuts=((3, 4),))
        search = CleavageActionSearch((a, b), max_action_count=2)
        self.assertIn(CleavageActionSequence((a, b)), {c.action_sequence for c in search.enumerate()})

    def test_two_and_three_action_precedence_cycles_are_pruned(self) -> None:
        a = action(0, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((1, 2),))
        b = action(1, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((2, 3),))
        search = CleavageActionSearch((a, b), max_action_count=2)
        self.assertTrue(all(len(c.action_sequence.actions) == 1 for c in search.enumerate()))
        self.assertEqual(search.stats.num_precedence_cycle_pruned, 1)
        c = action(2, (1, 2, 3, 4), matched=((3, 4), (1, 2)), cuts=((3, 4),))
        b = action(1, (2, 3, 4), matched=((2, 3), (3, 4)), cuts=((2, 3),))
        search = CleavageActionSearch((a, b, c), max_action_count=3)
        sequences = {x.action_sequence for x in search.enumerate()}
        self.assertEqual(sum(len(s.actions) == 2 for s in sequences), 3)
        self.assertFalse(any(len(s.actions) == 3 for s in sequences))
        self.assertGreater(search.stats.num_precedence_cycle_pruned, 0)

    def test_atom_deletion_precedence_and_cycle(self) -> None:
        a = action(0, (1, 2), retained=(1, 2, 3))
        b = action(1, (3, 4), matched=((3, 4),), cuts=((3, 4),))
        relations = _ActionRelations.from_actions((a, b))
        self.assertEqual(relations.must_precede_mask, (0, 1))
        # Give B a surviving cut, ensuring normalization retains its contribution.
        b = action(1, (2, 3, 4), matched=((2, 3),), cuts=((2, 3),))
        search = CleavageActionSearch((a, b), max_action_count=2)
        self.assertIn(CleavageActionSequence((a, b)), {c.action_sequence for c in search.enumerate()})
        b = action(1, (3, 4), retained=(2, 3, 4))
        relations = _ActionRelations.from_actions((a, b))
        self.assertEqual(relations.rejection((0, 1)), 'precedence_cycle')

    def test_redundancy_preserves_surviving_cut_and_order_updates(self) -> None:
        deletion = action(2, (3, 4), retained=(1, 2, 3))
        no_effect = action(0, (1, 2))
        self.assertEqual(CleavageActionSequence((deletion, no_effect)).actions, (deletion,))
        for edit in (action(0, (1, 2), matched=((1, 2),), cuts=((1, 2),)),
                     action(0, (1, 2), matched=((1, 2),), updates=((1, 2, Chem.BondType.DOUBLE),))):
            with self.subTest(edit=edit):
                self.assertEqual(len(CleavageActionSequence((deletion, edit)).actions), 2)
        search = CleavageActionSearch((deletion, no_effect), max_action_count=2)
        self.assertTrue(all(len(c.action_sequence.actions) <= 2 for c in search.enumerate()))
        self.assertGreater(search.stats.num_redundancy_pruned, 0)

    def test_limit_uses_normalized_action_count(self) -> None:
        a = action(0, (1,))
        b = action(1, (2,), retained=(1, 2, 3))
        c = action(2, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        search = CleavageActionSearch((a, b, c), max_action_count=2)
        sequences = {candidate.action_sequence for candidate in search.enumerate()}
        self.assertIn(CleavageActionSequence((a, b, c)), sequences)
        self.assertTrue(all(len(sequence.actions) <= 2 for sequence in sequences))

    def test_discarded_center_hint_keeps_a_contributing_action(self) -> None:
        a = action(0, (1, 2), retained=(1, 2, 3))
        b = action(1, (3,), retained=(3, 4))
        search = CleavageActionSearch((a, b), max_action_count=2)
        self.assertEqual(search.relations.redundancy_hint_mask[0], 1 << 1)
        combined = CleavageActionSequence((a, b))
        self.assertEqual(combined.retained_atom_maps, frozenset((3,)))
        self.assertIn(combined, {c.action_sequence for c in search.enumerate()})

    def test_effect_signature_ignores_discarded_bond_edits(self) -> None:
        a = action(0, (3, 4), retained=(1, 2), matched=((3, 4),), cuts=((3, 4),))
        b = action(1, (1, 2), retained=(1, 2))
        self.assertNotEqual(CleavageActionSequence((a,)).key, CleavageActionSequence((b,)).key)
        self.assertEqual(CleavageActionSequence((a,)).effect_key, CleavageActionSequence((b,)).effect_key)


class TestFragmentTreeBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Compound.from_smiles('[CH3:1][C:2]([OH:3])([NH2:4])[SH:5]')
        self.builder = builder(*(pattern(f'[C:1]-[{symbol}:2]', '[C:1]') for symbol in ('O', 'N', 'S')))

    def test_build_anchored_to_source_and_counter_evidence(self) -> None:
        run = CompositeCleavageReaction.run
        compile_sequence = CleavageActionSequence.compile
        executed = []
        compiled = []

        def compiling(sequence: CleavageActionSequence, source: Compound | Chem.Mol) -> CompositeCleavageReaction:
            self.assertIs(source, self.source)
            compiled.append(sequence.effect_key)
            return compile_sequence(sequence, source)

        def running(reaction: CompositeCleavageReaction, source: Compound | Chem.Mol) -> tuple[Chem.Mol, ...]:
            self.assertIs(source, self.source)
            executed.append(reaction.action_sequence.effect_key)
            return run(reaction, source)

        with patch.object(CleavageActionSequence, 'compile', compiling), \
             patch.object(CompositeCleavageReaction, 'run', running):
            result = self.builder._build_result(self.source)
        tree = result['fragment_tree']
        stats = result['search_stats']
        self.assertEqual(tree.get_node(0).smiles, self.source.smiles)
        self.assertEqual(tree.num_nodes, 8)
        self.assertEqual(tree.num_transitions, 12)
        self.assertEqual(tree.num_events, 0)
        self.assertEqual(stats['num_primitive_actions'], 3)
        self.assertEqual(stats['num_compiled_sequences'], 7)
        self.assertEqual(stats['num_rdkit_run_reactants'], 7)
        self.assertEqual(len(executed), len(set(executed)))
        self.assertEqual(compiled, executed)
        self.assertTrue(any(len(state.action_sequence.actions) == 3
                            for state in result['expansion_states'] if state.action_sequence))

    def test_none_and_empty_seeds_start_at_source(self) -> None:
        for seeds in (None, ()):
            result = self.builder._build_result(self.source, seed_action_sequences=seeds)
            self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 7)
            self.assertFalse(any(t.is_seed for i in range(result['fragment_tree'].num_edges)
                                 for t in result['fragment_tree'].get_edge(i).transitions))

    def test_single_seed_and_lower_canonical_index_addition(self) -> None:
        actions = self.builder.create_cleavage_actions(self.source)
        seed = CleavageActionSequence((actions[-1],))
        result = self.builder._build_result(self.source, seed_action_sequences=(seed,), max_action_count=2)
        tree = result['fragment_tree']
        self.assertEqual(len(tree.get_out_edges(0)), 1)
        transition = tree.get_out_edges(0)[0].transitions[0]
        self.assertTrue(transition.is_seed)
        self.assertEqual(transition.action_sequence, seed)
        sequences = [s.action_sequence for s in result['expansion_states'] if s.action_sequence]
        self.assertEqual(len(sequences), 3)
        self.assertTrue(all(len(s.actions) <= 2 for s in sequences))
        self.assertTrue(all(actions[-1] in s.actions for s in sequences))

    def test_multiple_seeds_keep_histories_and_execute_effect_once(self) -> None:
        actions = self.builder.create_cleavage_actions(self.source)
        seeds = tuple(CleavageActionSequence((a,)) for a in actions[:2])
        result = self.builder._build_result(self.source, seed_action_sequences=seeds)
        tree = result['fragment_tree']
        self.assertEqual(len(tree.get_out_edges(0)), 2)
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 6)
        triple = [s for s in result['expansion_states'] if s.action_sequence
                  and len(s.action_sequence.actions) == 3]
        self.assertEqual(len(triple), 1)
        self.assertEqual(len(tree.get_in_edges(triple[0].node_index)), 3)

    def test_same_smiles_seeds_have_distinct_states(self) -> None:
        source = Compound.from_smiles('[CH3:1][CH:2]([OH:3])[OH:4]')
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'), limit=2)
        actions = build.create_cleavage_actions(source)
        seeds = tuple(CleavageActionSequence((a,)) for a in actions)
        result = build._build_result(source, seed_action_sequences=seeds)
        tree = result['fragment_tree']
        self.assertEqual(tree.num_nodes, 3)
        self.assertEqual(len(tree.get_out_edges(0)), 1)
        self.assertEqual(len(tree.get_out_edges(0)[0].transitions), 2)
        states = [s for s in result['expansion_states'] if s.action_sequence
                  and len(s.action_sequence.actions) == 1]
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0].node_index, states[1].node_index)
        self.assertNotEqual(states[0].action_sequence, states[1].action_sequence)

    def test_different_histories_same_effect_use_cached_reaction(self) -> None:
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'),
                        pattern('[C:1](-[O:2])-[N:3]', '[C:1]-[N:3]'), limit=1)
        result = build._build_result(Compound.from_smiles('CC(O)N'))
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 1)
        self.assertEqual(result['search_stats']['num_duplicate_effect_pruned'], 1)
        self.assertEqual(len(result['expansion_states']), 3)
        self.assertEqual(len(result['fragment_tree'].get_edge(0).transitions), 2)
        self.assertEqual(result['fragment_tree'].copy().get_edge(0).transitions,
                         result['fragment_tree'].get_edge(0).transitions)

    def test_cleave_all_returns_separate_action_results(self) -> None:
        results = self.builder.cleave_all(self.source)
        self.assertEqual(len(results), 7)
        self.assertEqual(len({r.action_sequence for r in results}), 7)
        self.assertTrue(all(r.smirks for r in results))

    def test_cycle_and_conflict_candidates_never_run_rdkit(self) -> None:
        source = Compound.from_smiles('[CH3:1][CH2:2][CH2:3][CH3:4]')
        a = action(0, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((1, 2),))
        b = action(1, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((2, 3),))
        build = builder(limit=2)
        with patch.object(FragmentTreeBuilder, 'create_cleavage_actions', return_value=(a, b)):
            result = build._build_result(source)
        stats = result['search_stats']
        self.assertEqual(stats['num_raw_combinations'], 3)
        self.assertEqual(stats['num_precedence_cycle_pruned'], 1)
        self.assertEqual(stats['num_rdkit_run_reactants'], 2)
        b = action(1, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        with patch.object(FragmentTreeBuilder, 'create_cleavage_actions', return_value=(a, b)):
            result = build._build_result(source)
        stats = result['search_stats']
        self.assertEqual(stats['num_hard_conflict_pruned'], 1)
        self.assertEqual(stats['num_duplicate_effect_pruned'], 1)
        self.assertEqual(stats['num_rdkit_run_reactants'], 1)

    def test_seed_multiple_actions_count_towards_limit(self) -> None:
        actions = self.builder.create_cleavage_actions(self.source)
        seed = CleavageActionSequence(actions[:2])
        result = self.builder._build_result(self.source, seed_action_sequences=(seed,), max_action_count=2)
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 1)
        self.assertEqual(len(result['expansion_states']), 2)
        edge = result['fragment_tree'].get_edge(0)
        self.assertTrue(edge.transitions[0].is_seed)
        self.assertEqual(edge.transitions[0].action_sequence, seed)
        extended = self.builder._build_result(self.source, seed_action_sequences=(seed,), max_action_count=3)
        self.assertEqual(extended['search_stats']['num_rdkit_run_reactants'], 2)

    def test_valid_combined_target_does_not_require_invalid_primitive_target(self) -> None:
        source = Compound.from_smiles('[CH3:1][C:2]([CH3:3])([CH3:4])[OH:5]')
        build = builder(pattern('[C:1]-[C:2]', '[C:1]=[C:2]'),
                        pattern('[C:1]-[O:2]', '[C:1]'), limit=2)
        result = build._build_result(source)
        tree = result['fragment_tree']
        self.assertEqual(set(tree.node_smiles), {source.smiles, 'CC(C)C', 'C=C(C)C'})
        child = tree.get_node_by_smiles('C=C(C)C')
        parent = tree.get_node_by_smiles('CC(C)C')
        self.assertTrue(any(edge.source_index == parent.index for edge in tree.get_in_edges(child.index)))
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 4)

    def test_limits_seed_validation_and_removed_apis(self) -> None:
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                FragmentTreeBuilder(limit, self.builder.cleavage_pattern_set)
        for kwargs, message in (({'max_node': 1}, 'node limit'), ({'max_edge': 0}, 'edge limit')):
            with self.assertRaisesRegex(ValueError, message):
                self.builder.build(self.source, **kwargs)
        actions = self.builder.create_cleavage_actions(self.source)
        with self.assertRaisesRegex(ValueError, 'Seed exceeds'):
            self.builder.build(self.source, seed_action_sequences=(CleavageActionSequence(actions),),
                               max_action_count=2)
        with self.assertRaisesRegex(ValueError, 'universe'):
            self.builder.build(Compound.from_smiles('CC'),
                seed_action_sequences=(CleavageActionSequence((actions[0],)),))
        self.assertEqual(self.builder.build(self.source, max_action_count=0).num_nodes, 1)
        for removed in ('max_depth', 'min_depth_only_from', 'cleave_by_pattern', 'cleave_by_pattern_id'):
            self.assertFalse(hasattr(self.builder, removed))
        data = self.builder.to_dict()
        self.assertNotIn('max_depth', data)
        self.assertNotIn('min_depth_only_from', data)
        self.assertEqual(FragmentTreeBuilder.from_dict(data).to_dict(), data)
        self.assertEqual(self.builder.copy().to_dict(), data)
        self.assertIsNot(self.builder.copy().cleavage_pattern_set, self.builder.cleavage_pattern_set)


if __name__ == '__main__':
    unittest.main()
