import unittest
from dataclasses import replace
from rdkit import Chem
from clefts.domain.fragment.cleavage import CleavageAction, CleavageActionSequence
from clefts.domain.fragment.cleavage._CleavagePattern import _CleavagePattern, ProductRule
from clefts.libs.mmkit.mmkit import Compound


class TestCleavageAction(unittest.TestCase):
    def action(self, roles=(1, 2), retained=(1, 2), pattern_id=0, matched=None, cuts=(), universe=(1, 2, 3)):
        return CleavageAction(pattern_id, 0, roles, frozenset(retained),
                              frozenset(universe) - frozenset(retained),
                              frozenset(matched if matched is not None else [roles]), frozenset(cuts), product_molecule_id=0)

    def test_subset_order_duplicates_and_identity(self):
        a = self.action(retained=(1,2,3))
        b = self.action(retained=(1,2), pattern_id=1)
        seq = CleavageActionSequence((a, b, a))
        self.assertEqual(seq.actions, (b,))
        self.assertEqual(seq, CleavageActionSequence((b, a)))
        self.assertEqual(hash(seq), hash(CleavageActionSequence((b,))))

    def test_shared_bond_rejected_before_redundancy(self):
        a = self.action(cuts=((1, 2),))
        b = self.action(roles=(2, 1), retained=(1,), pattern_id=1)
        for actions in ((a, b), (b, a), (a, a, b)):
            with self.subTest(actions=actions), self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
                CleavageActionSequence(actions)
        self.assertEqual(CleavageActionSequence((a, a)).actions, (a,))

    def test_context_bond_and_three_actions(self):
        a = self.action(roles=(1, 2, 3), retained=(1, 2, 3), matched=((1, 2), (2, 3)), cuts=((1, 2),))
        b = self.action(roles=(2, 3), retained=(1, 2, 3), pattern_id=1, cuts=((2, 3),))
        c = self.action(roles=(1,), retained=(1, 2, 3), matched=(), pattern_id=2)
        self.assertTrue(CleavageActionSequence((c, a, b)).actions)

    def test_shared_atom_allowed_and_single_rdkit_execution(self):
        source = Chem.MolFromSmiles('[CH3:1][CH:2]([OH:3])[NH2:4]')
        universe = (1, 2, 3, 4)
        a = self.action(roles=(2, 3), retained=(1, 2, 4), universe=universe)
        b = self.action(roles=(2, 4), retained=(1, 2, 3), universe=universe, pattern_id=1)
        seq = CleavageActionSequence((a, b))
        self.assertEqual(len(seq.actions), 2)
        self.assertEqual(seq.retained_atom_maps, frozenset((1, 2)))
        compiled = seq.compile(source)
        self.assertEqual(compiled.smirks, CleavageActionSequence((b, a)).compile(source).smirks)
        products = compiled.run(source)
        self.assertEqual(len(products), 1)
        self.assertEqual(Chem.MolToSmiles(products[0]), '[CH3:1][CH3:2]')
        self.assertEqual(source.GetNumAtoms(), 4)
        self.assertEqual(len(compiled.rxn.RunReactants((source,))), 1)
        reordered = Chem.RenumberAtoms(source, (3, 2, 1, 0))
        self.assertEqual(Chem.MolToSmiles(compiled.run(reordered)[0]), Chem.MolToSmiles(products[0]))
        self.assertEqual(compiled.smirks, seq.compile(reordered).smirks)

    def test_surviving_cut_is_not_removed_by_subset(self):
        a = self.action(retained=(1, 2, 3), cuts=((1, 2),))
        b = self.action(roles=(2, 3), retained=(1, 2), pattern_id=1)
        seq = CleavageActionSequence((a, b))
        self.assertEqual(len(seq.actions), 2)
        result = seq.compile(Chem.MolFromSmiles('[CH3:1][CH2:2][CH3:3]')).run(
            Chem.MolFromSmiles('[CH3:1][CH2:2][CH3:3]'))[0]
        self.assertEqual(result.GetNumBonds(), 0)

    def test_equal_retention_does_not_remove_all_actions(self):
        a = self.action()
        b = self.action(roles=(2, 1), pattern_id=1)
        self.assertEqual(CleavageActionSequence((b, a)).actions, (a,))

    def test_existing_rules_manual_match_with_compound(self):
        source = Compound.from_smiles('CC(O)N')
        mol = source.mapped_mol
        center = next(a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and a.GetDegree() == 3)
        all_maps = frozenset(a.GetAtomMapNum() for a in mol.GetAtoms())
        actions = []
        for pattern_id, symbol in enumerate(('O', 'N')):
            terminal = next(a for a in mol.GetAtoms() if a.GetSymbol() == symbol)
            pattern = _CleavagePattern.from_rules(reactant_smarts=f'[C:1]-[{symbol}:2]',
                                                  products=(ProductRule('retain carbon', '[C:1]'),))
            actions.append(CleavageAction.from_match(
                source=source, product_molecule_id=0, cleavage_pattern=pattern, cleavage_pattern_id=pattern_id,
                source_atom_maps=(center.GetAtomMapNum(), terminal.GetAtomMapNum()),
                retained_atom_maps=all_maps - {terminal.GetAtomMapNum()}))
        result = CleavageActionSequence(actions).compile(source).run(source)[0]
        for atom in result.GetAtoms():
            atom.SetAtomMapNum(0)
        self.assertEqual(Chem.MolToSmiles(result), 'CC')

    def test_query_bonds_only_and_invalid_match(self):
        source = Chem.MolFromSmiles('[CH2:1]1[CH2:2][CH2:3]1')
        pattern = _CleavagePattern.from_rules(reactant_smarts='[C:1]-[C:2]-[C:3]',
                                              products=(ProductRule('keep', '[C:1]-[C:2]'),))
        a = CleavageAction.from_match(source=source, product_molecule_id=0, cleavage_pattern=pattern, cleavage_pattern_id=0,
                                     source_atom_maps=(1, 2, 3), retained_atom_maps=(1, 2))
        self.assertEqual(a.changed_bond_maps, frozenset(((1, 3), (2, 3))))
        self.assertEqual(a.matched_bond_maps, frozenset(((1, 2), (2, 3))))
        wrong = _CleavagePattern.from_rules(reactant_smarts='[C:1]-[O:2]',
                                            products=(ProductRule('keep', '[C:1]'),))
        with self.assertRaisesRegex(ValueError, 'do not match'):
            CleavageAction.from_match(source=source, product_molecule_id=0, cleavage_pattern=wrong, cleavage_pattern_id=0,
                                      source_atom_maps=(1, 2), retained_atom_maps=(1,))

    def test_product_selection_and_bond_order(self):
        source = Chem.MolFromSmiles('[CH3:10][CH2:20][CH3:30]')
        pattern = _CleavagePattern.from_rules(
            reactant_smarts='[C:7]-[C:2]-[C:9]',
            products=(ProductRule('split', '[C:7]=[C:2].[C:9]'),))
        a = CleavageAction.from_match(source=source, cleavage_pattern=pattern,
            cleavage_pattern_id=0, product_molecule_id=0,
            source_atom_maps=(10, 20, 30), retained_atom_maps=(10, 20))
        self.assertEqual(a.source_atom_maps, (10, 20, 30))
        self.assertEqual(a.bond_updates, frozenset(((10, 20, Chem.BondType.DOUBLE),)))
        self.assertEqual(a.changed_bond_maps, frozenset(((10, 20), (20, 30))))
        product = CleavageActionSequence((a,)).compile(source).run(source)[0]
        self.assertEqual(Chem.MolToSmiles(product), '[CH2:10]=[CH2:20]')
        b = CleavageAction.from_match(source=source, cleavage_pattern=pattern,
            cleavage_pattern_id=0, product_molecule_id=1,
            source_atom_maps=(10, 20, 30), retained_atom_maps=(30,))
        self.assertNotEqual(a.key, b.key)
        self.assertEqual(Chem.MolToSmiles(CleavageActionSequence((b,)).compile(source).run(source)[0]), '[CH4:30]')
        with self.assertRaisesRegex(ValueError, 'product_molecule_id'):
            CleavageAction.from_match(source=source, cleavage_pattern=pattern,
                cleavage_pattern_id=0, product_molecule_id=2,
                source_atom_maps=(10, 20, 30), retained_atom_maps=(30,))

    def test_order_change_overlap_rejected(self):
        a = replace(self.action(retained=(1, 2, 3)),
                    bond_updates=frozenset(((1, 2, Chem.BondType.DOUBLE),)))
        b = self.action(pattern_id=1, cuts=((1, 2),))
        with self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
            CleavageActionSequence((a, b))

    def test_all_bond_pairs_are_normalized(self):
        a = replace(self.action(roles=(1, 2, 3), retained=(1, 2, 3), matched=((2, 1), (3, 2)),
                                cuts=((3, 2),)),
                    bond_updates=frozenset(((2, 1, Chem.BondType.DOUBLE),)),
                    changed_bond_maps=frozenset(((3, 1),)))
        self.assertEqual(a.matched_bond_maps, frozenset(((1, 2), (2, 3))))
        self.assertEqual(a.cut_bond_maps, frozenset(((2, 3),)))
        self.assertEqual(a.bond_updates, frozenset(((1, 2, Chem.BondType.DOUBLE),)))
        self.assertEqual(a.changed_bond_maps, frozenset(((1, 2), (1, 3), (2, 3))))
        b = replace(a, matched_bond_maps=frozenset(((1, 2), (2, 3))),
                    cut_bond_maps=frozenset(((2, 3),)),
                    bond_updates=frozenset(((1, 2, Chem.BondType.DOUBLE),)))
        self.assertEqual(a, b)
        self.assertEqual(a.key, b.key)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(CleavageActionSequence((a, b)).actions, (a,))
        conflict = self.action(roles=(3, 2), retained=(1, 2, 3),
                               pattern_id=1, cuts=((3, 2),))
        with self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
            CleavageActionSequence((a, conflict))

    def test_validation(self):
        with self.assertRaises(ValueError):
            CleavageActionSequence(())
        with self.assertRaises(ValueError):
            replace(self.action(), cut_bond_maps=frozenset(((1, 3),)))
        with self.assertRaises(ValueError):
            CleavageActionSequence((self.action(), self.action(roles=(2, 3), universe=(1, 2, 3, 4))))
        with self.assertRaises(ValueError):
            CleavageActionSequence((self.action(),)).compile(Chem.MolFromSmiles('CCC'))
        compiled = CleavageActionSequence((self.action(),)).compile(Chem.MolFromSmiles('[CH3:1][CH2:2][CH3:3]'))
        with self.assertRaises(ValueError):
            compiled.run(Chem.MolFromSmiles('[CH3:1][CH2:2][OH:3]'))
