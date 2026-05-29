from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ....libs.mmkit.mmkit import *
from ..cleavage.CleavagePatternSet import CleavagePatternSet, CleavagePattern
from .tables import (
    Base,
    CompoundTable,
    FragmentTable,
    CleavagePatternTable,
    FragmentEdgeTable,
    CleavageEventTable,
    FormulaTable,
)
from .path import FragmentGraphProjectPath
from .registration import FragmentEdgeRegistration


class FragmentGraphRepository:
    """Operate fragment graph tables using an externally managed Session."""

    # ---------- Compound ----------

    @classmethod
    def get_compound(
        cls,
        session: Session,
        compound: Compound,
    ) -> CompoundTable | None:
        return session.scalar(
            select(CompoundTable).where(
                CompoundTable.smiles == compound.smiles,
            )
        )

    @classmethod
    def get_or_create_compound(
        cls,
        session: Session,
        compound: Compound,
    ) -> CompoundTable:
        compound_row = cls.get_compound(
            session=session,
            compound=compound,
        )

        if compound_row is not None:
            return compound_row

        fragment = cls.get_or_create_fragment(
            session=session,
            compound=compound,
        )

        session.add(
            CompoundTable(
                smiles=compound.smiles,
                fragment_id=fragment.fragment_id,
            )
        )
        session.flush()

        compound_row = cls.get_compound(
            session=session,
            compound=compound,
        )

        if compound_row is None:
            raise RuntimeError(
                f"Failed to get or create compound: {compound.smiles}"
            )

        return compound_row

    # ---------- Formula ----------

    @classmethod
    def get_formula(
        cls,
        session: Session,
        formula: Formula,
    ) -> FormulaTable | None:
        return session.scalar(
            select(FormulaTable).where(
                FormulaTable.formula == formula.value,
            )
        )

    @classmethod
    def get_or_create_formula(
        cls,
        session: Session,
        formula: Formula,
    ) -> FormulaTable:
        formula_row = cls.get_formula(
            session=session,
            formula=formula,
        )

        if formula_row is not None:
            return formula_row

        session.add(
            FormulaTable(
                formula=formula.value,
                exact_mass=formula.exact_mass,
            )
        )
        session.flush()

        formula_row = cls.get_formula(
            session=session,
            formula=formula,
        )

        if formula_row is None:
            raise RuntimeError(f"Failed to get or create formula: {formula}")

        return formula_row

    # ---------- Fragment ----------

    @classmethod
    def get_fragment(
        cls,
        session: Session,
        compound: Compound,
    ) -> FragmentTable | None:
        return session.scalar(
            select(FragmentTable).where(
                FragmentTable.smiles == compound.smiles,
            )
        )

    @classmethod
    def get_or_create_fragment(
        cls,
        session: Session,
        compound: Compound,
    ) -> FragmentTable:
        fragment = cls.get_fragment(
            session=session,
            compound=compound,
        )

        if fragment is not None:
            return fragment

        formula_row = cls.get_or_create_formula(
            session=session,
            formula=compound.formula,
        )
        if formula_row is None:
            raise RuntimeError(
                f"Failed to get or create formula for fragment: {compound.smiles}"
            )

        session.add(
            FragmentTable(
                smiles=compound.smiles,
                formula_id=formula_row.formula_id,
            )
        )
        session.flush()

        fragment = cls.get_fragment(
            session=session,
            compound=compound,
        )

        if fragment is None:
            raise RuntimeError(f"Failed to get or create fragment: {compound.smiles}")

        return fragment

    # ---------- CleavagePattern ----------

    @classmethod
    def get_cleavage_pattern(
        cls,
        session: Session,
        cleavage_pattern: CleavagePattern,
    ) -> CleavagePatternTable | None:
        cleavage_pattern_row = session.scalar(
            select(CleavagePatternTable).where(
                CleavagePatternTable.smirks == cleavage_pattern.smirks,
                CleavagePatternTable.charge_mode == cleavage_pattern.charge_mode,
            )
        )
        return cleavage_pattern_row

    @classmethod
    def initialize_cleavage_patterns(
        cls,
        session: Session,
        cleavage_pattern_set: CleavagePatternSet,
    ) -> None:
        for cleavage_pattern in cleavage_pattern_set.patterns:
            existing = session.scalar(
                select(CleavagePatternTable).where(
                    CleavagePatternTable.smirks == cleavage_pattern.smirks,
                    CleavagePatternTable.charge_mode == cleavage_pattern.charge_mode,
                )
            )

            if existing is not None:
                continue

            session.add(
                CleavagePatternTable(
                    cleavage_pattern_id=cleavage_pattern_set.get_id(
                        cleavage_pattern
                    ),
                    name=getattr(cleavage_pattern, "name", None),
                    smirks=cleavage_pattern.smirks,
                    charge_mode=cleavage_pattern.charge_mode,
                )
            )

        session.flush()

    # ---------- FragmentEdge ----------

    @classmethod
    def get_fragment_edge(
        cls,
        session: Session,
        source_id: int,
        target_id: int,
    ) -> FragmentEdgeTable | None:
        return session.scalar(
            select(FragmentEdgeTable).where(
                FragmentEdgeTable.source_id == source_id,
                FragmentEdgeTable.target_id == target_id,
            )
        )

    @classmethod
    def get_or_create_fragment_edge(
        cls,
        session: Session,
        source_id: int,
        target_id: int,
    ) -> FragmentEdgeTable:
        edge = cls.get_fragment_edge(
            session=session,
            source_id=source_id,
            target_id=target_id,
        )

        if edge is not None:
            return edge

        session.add(
            FragmentEdgeTable(
                source_id=source_id,
                target_id=target_id,
            )
        )
        session.flush()

        edge = cls.get_fragment_edge(
            session=session,
            source_id=source_id,
            target_id=target_id,
        )

        if edge is None:
            raise RuntimeError(
                f"Failed to get or create fragment edge: {source_id} -> {target_id}"
            )

        return edge
    
    # ---------- CleavageEvent ----------

    @classmethod
    def get_cleavage_event(
        cls,
        session: Session,
        edge_id: int,
        event_id: int,
    ) -> CleavageEventTable | None:
        return session.scalar(
            select(CleavageEventTable).where(
                CleavageEventTable.edge_id == edge_id,
                CleavageEventTable.event_id == event_id,
            )
        )


    @classmethod
    def create_cleavage_event(
        cls,
        session: Session,
        *,
        edge_id: int,
        event_id: int,
        cleavage_pattern_id: int,
        react_indices: str,
        prod_indices: str,
    ) -> CleavageEventTable:
        session.add(
            CleavageEventTable(
                edge_id=edge_id,
                event_id=event_id,
                cleavage_pattern_id=cleavage_pattern_id,
                react_indices=react_indices,
                prod_indices=prod_indices,
            )
        )
        session.flush()

        cleavage_event = cls.get_cleavage_event(
            session=session,
            edge_id=edge_id,
            event_id=event_id,
        )

        if cleavage_event is None:
            raise RuntimeError(
                f"Failed to create cleavage event: edge_id={edge_id}, event_id={event_id}"
            )

        return cleavage_event