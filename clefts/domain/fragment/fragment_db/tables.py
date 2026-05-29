from __future__ import annotations

from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Float,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CompoundTable(Base):
    __tablename__ = "compound"

    compound_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    fragment_id: Mapped[int] = mapped_column(
        ForeignKey("fragment.fragment_id"),
        nullable=False,
        unique=True,
    )
    smiles: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,
    )


class FragmentTable(Base):
    __tablename__ = "fragment"

    fragment_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    smiles: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,
    )
    formula_id: Mapped[int] = mapped_column(
        ForeignKey("formula.formula_id"),
        nullable=False,
    )


class CleavagePatternTable(Base):
    __tablename__ = "cleavage_pattern"

    cleavage_pattern_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    smirks: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    charge_mode: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "smirks",
            "charge_mode",
            name="uq_cleavage_pattern_smirks_charge_mode",
        ),
    )


class FragmentEdgeTable(Base):
    __tablename__ = "fragment_edge"

    edge_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    source_id: Mapped[int] = mapped_column(
        ForeignKey("fragment.fragment_id"),
        nullable=False,
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("fragment.fragment_id"),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "target_id",
            name="uq_fragment_edge_source_target",
        ),
    )


class CleavageEventTable(Base):
    __tablename__ = "cleavage_event"

    edge_id: Mapped[int] = mapped_column(
        ForeignKey("fragment_edge.edge_id"),
        primary_key=True,
    )
    event_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    cleavage_pattern_id: Mapped[int] = mapped_column(
        ForeignKey("cleavage_pattern.cleavage_pattern_id"),
        nullable=False,
    )
    react_indices: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    prod_indices: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

class FormulaTable(Base):
    __tablename__ = "formula"

    formula_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    formula: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,
    )
    exact_mass: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )