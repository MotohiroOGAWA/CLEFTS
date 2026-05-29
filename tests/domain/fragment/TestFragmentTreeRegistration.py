from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clefts.libs.mmkit.mmkit import Compound
from clefts.domain.fragment.cleavage.CleavagePatternSet import CleavagePatternSet, CleavagePattern
from clefts.domain.fragment.fragment_db.database import (
    FragmentGraphDatabase,
    FragmentGraphProjectPath,
)
from clefts.domain.fragment.fragment_db.repository import FragmentGraphRepository
from clefts.domain.fragment.fragment_db.tables import (
    CleavageEventTable,
    CleavagePatternTable,
    CompoundTable,
    FormulaTable,
    FragmentEdgeTable,
    FragmentTable,
)


class TestFragmentTreeBuilder(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

        self.database = FragmentGraphDatabase(
            FragmentGraphProjectPath(
                root_dir=Path(self.temp_dir.name),
                project_name="test_project",
                database_name="test_database",
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_database_file_is_created(self) -> None:
        self.assertTrue(self.database.project_path.database_path.exists())

    def test_get_formula_returns_none_when_missing(self) -> None:
        compound = Compound.from_smiles("CCO")

        with self.database.session() as session:
            formula = FragmentGraphRepository.get_formula(
                session=session,
                formula=compound.formula,
            )

            self.assertIsNone(formula)

    def test_get_or_create_formula(self) -> None:
        compound = Compound.from_smiles("CCO")

        with self.database.session() as session:
            formula_1 = FragmentGraphRepository.get_or_create_formula(
                session=session,
                formula=compound.formula,
            )
            formula_2 = FragmentGraphRepository.get_or_create_formula(
                session=session,
                formula=compound.formula,
            )

            self.assertEqual(formula_1.formula_id, formula_2.formula_id)
            self.assertEqual(formula_1.formula, compound.formula.value)
            self.assertAlmostEqual(formula_1.exact_mass, compound.formula.exact_mass)

            self.assertEqual(session.query(FormulaTable).count(), 1)

            session.commit()

    def test_get_fragment_returns_none_when_missing(self) -> None:
        compound = Compound.from_smiles("CCO")

        with self.database.session() as session:
            fragment = FragmentGraphRepository.get_fragment(
                session=session,
                compound=compound,
            )

            self.assertIsNone(fragment)

    def test_get_or_create_fragment(self) -> None:
        compound = Compound.from_smiles("CCO")

        with self.database.session() as session:
            fragment_1 = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=compound,
            )
            fragment_2 = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=compound,
            )

            self.assertEqual(fragment_1.fragment_id, fragment_2.fragment_id)
            self.assertEqual(fragment_1.smiles, compound.smiles)

            self.assertEqual(session.query(FragmentTable).count(), 1)
            self.assertEqual(session.query(FormulaTable).count(), 1)

            session.commit()

    def test_get_compound_returns_none_when_missing(self) -> None:
        compound = Compound.from_smiles("CCO")

        with self.database.session() as session:
            compound_row = FragmentGraphRepository.get_compound(
                session=session,
                compound=compound,
            )

            self.assertIsNone(compound_row)

    def test_get_or_create_compound(self) -> None:
        compound = Compound.from_smiles("CCO")

        with self.database.session() as session:
            compound_1 = FragmentGraphRepository.get_or_create_compound(
                session=session,
                compound=compound,
            )
            compound_2 = FragmentGraphRepository.get_or_create_compound(
                session=session,
                compound=compound,
            )

            self.assertEqual(compound_1.compound_id, compound_2.compound_id)
            self.assertEqual(compound_1.smiles, compound.smiles)

            self.assertEqual(session.query(CompoundTable).count(), 1)
            self.assertEqual(session.query(FragmentTable).count(), 1)
            self.assertEqual(session.query(FormulaTable).count(), 1)

            session.commit()

    def test_get_fragment_edge_returns_none_when_missing(self) -> None:
        source = Compound.from_smiles("CCO")
        target = Compound.from_smiles("CC")

        with self.database.session() as session:
            source_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=source,
            )
            target_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=target,
            )

            edge = FragmentGraphRepository.get_fragment_edge(
                session=session,
                source_id=source_fragment.fragment_id,
                target_id=target_fragment.fragment_id,
            )

            self.assertIsNone(edge)

            session.commit()

    def test_get_or_create_fragment_edge(self) -> None:
        source = Compound.from_smiles("CCO")
        target = Compound.from_smiles("CC")

        with self.database.session() as session:
            source_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=source,
            )
            target_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=target,
            )

            edge_1 = FragmentGraphRepository.get_or_create_fragment_edge(
                session=session,
                source_id=source_fragment.fragment_id,
                target_id=target_fragment.fragment_id,
            )
            edge_2 = FragmentGraphRepository.get_or_create_fragment_edge(
                session=session,
                source_id=source_fragment.fragment_id,
                target_id=target_fragment.fragment_id,
            )

            self.assertEqual(edge_1.edge_id, edge_2.edge_id)
            self.assertEqual(edge_1.source_id, source_fragment.fragment_id)
            self.assertEqual(edge_1.target_id, target_fragment.fragment_id)

            self.assertEqual(session.query(FragmentEdgeTable).count(), 1)

            session.commit()

    def test_get_cleavage_pattern_returns_none_when_missing(self) -> None:
        cleavage_pattern = CleavagePattern(
            smirks="[C:1]-[O:2]>>[C:1]",
            charge_mode="any",
            name="test_pattern",
        )

        with self.database.session() as session:
            pattern = FragmentGraphRepository.get_cleavage_pattern(
                session=session,
                cleavage_pattern=cleavage_pattern,
            )

            self.assertIsNone(pattern)

    def test_initialize_cleavage_patterns(self) -> None:
        cleavage_pattern = CleavagePattern(
            smirks="[C:1]-[O:2]>>[C:1]",
            charge_mode="any",
            name="test_pattern",
        )

        cleavage_pattern_set = CleavagePatternSet(
            name="test_pattern_set",
            patterns=(cleavage_pattern,),
        )

        with self.database.session() as session:
            FragmentGraphRepository.initialize_cleavage_patterns(
                session=session,
                cleavage_pattern_set=cleavage_pattern_set,
            )

            pattern = FragmentGraphRepository.get_cleavage_pattern(
                session=session,
                cleavage_pattern=cleavage_pattern,
            )

            self.assertIsNotNone(pattern)
            self.assertEqual(pattern.cleavage_pattern_id, 0)
            self.assertEqual(pattern.name, cleavage_pattern.name)
            self.assertEqual(pattern.smirks, cleavage_pattern.smirks)
            self.assertEqual(pattern.charge_mode, cleavage_pattern.charge_mode)

            self.assertEqual(session.query(CleavagePatternTable).count(), 1)

            session.commit()

    def test_get_cleavage_event_returns_none_when_missing(self) -> None:
        source = Compound.from_smiles("CCO")
        target = Compound.from_smiles("CC")

        with self.database.session() as session:
            source_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=source,
            )
            target_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=target,
            )
            edge = FragmentGraphRepository.get_or_create_fragment_edge(
                session=session,
                source_id=source_fragment.fragment_id,
                target_id=target_fragment.fragment_id,
            )

            event = FragmentGraphRepository.get_cleavage_event(
                session=session,
                edge_id=edge.edge_id,
                event_id=0,
            )

            self.assertIsNone(event)

            session.commit()

    def test_create_and_get_cleavage_event(self) -> None:
        source = Compound.from_smiles("CCO")
        target = Compound.from_smiles("CC")

        cleavage_pattern = CleavagePattern(
            smirks="[C:1]-[O:2]>>[C:1]",
            charge_mode="any",
            name="test" \
            "cleavage_pattern",
        )
        cleavage_pattern_set = CleavagePatternSet(
            name="test_pattern_set",
            patterns=(cleavage_pattern,),
        )

        with self.database.session() as session:
            FragmentGraphRepository.initialize_cleavage_patterns(
                session=session,
                cleavage_pattern_set=cleavage_pattern_set,
            )

            pattern = FragmentGraphRepository.get_cleavage_pattern(
                session=session,
                cleavage_pattern=cleavage_pattern,
            )

            source_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=source,
            )
            target_fragment = FragmentGraphRepository.get_or_create_fragment(
                session=session,
                compound=target,
            )

            edge = FragmentGraphRepository.get_or_create_fragment_edge(
                session=session,
                source_id=source_fragment.fragment_id,
                target_id=target_fragment.fragment_id,
            )

            pattern = FragmentGraphRepository.get_cleavage_pattern(
                session=session,
                cleavage_pattern=cleavage_pattern,
            )

            event = FragmentGraphRepository.create_cleavage_event(
                session=session,
                edge_id=edge.edge_id,
                event_id=0,
                cleavage_pattern_id=pattern.cleavage_pattern_id,
                react_indices="[0, 1]",
                prod_indices="[0]",
            )

            fetched = FragmentGraphRepository.get_cleavage_event(
                session=session,
                edge_id=edge.edge_id,
                event_id=0,
            )

            self.assertIsNotNone(fetched)
            self.assertEqual(event.edge_id, fetched.edge_id)
            self.assertEqual(event.event_id, fetched.event_id)
            self.assertEqual(event.cleavage_pattern_id, pattern.cleavage_pattern_id)
            self.assertEqual(event.react_indices, "[0, 1]")
            self.assertEqual(event.prod_indices, "[0]")

            self.assertEqual(session.query(CleavageEventTable).count(), 1)

            session.commit()


if __name__ == "__main__":
    unittest.main()