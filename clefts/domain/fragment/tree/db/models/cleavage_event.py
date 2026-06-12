from __future__ import annotations

from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CleavageEventTable(Base):
    """One product-molecule-level cleavage event for one fragment edge.

    This table corresponds to one ProductMolecule generated from one
    CleavageProduct and linked to one FragmentEdge.

    Notes
    -----
    edge_id:
        Parent fragment edge.
        The source fragment, target fragment, and depth are stored in
        FragmentEdgeTable.

    cleavage_pattern_id:
        CleavagePattern used for this event.

    reaction_id:
        Local CleavageReaction ID inside one CleavagePattern.

    product_molecule_id:
        Local product molecule index inside one reaction result.
        This corresponds to ProductMolecule.id.

    event_id:
        Index used to distinguish multiple matches that produce the same
        edge, reaction, and product molecule.

    reactant_indices:
        Serialized reactant atom indices.

    product_indices:
        Serialized product atom indices.
    """

    __tablename__ = "cleavage_event"

    edge_id: Mapped[int] = mapped_column(
        ForeignKey("fragment_edge.id"),
        primary_key=True,
    )

    cleavage_pattern_id: Mapped[int] = mapped_column(
        ForeignKey("cleavage_pattern.id"),
        primary_key=True,
    )

    reaction_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    product_molecule_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    event_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    reactant_indices: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    product_indices: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["cleavage_pattern_id", "reaction_id"],
            [
                "cleavage_reaction.cleavage_pattern_id",
                "cleavage_reaction.reaction_id",
            ],
            name="fk_cleavage_event_reaction",
        ),
    )