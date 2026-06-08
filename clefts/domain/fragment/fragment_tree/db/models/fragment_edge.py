from __future__ import annotations

from sqlalchemy import (
    ForeignKey,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class FragmentEdgeTable(Base):
    """One selected edge in the fragmentation graph.

    Notes
    -----
    id:
        Global edge ID.

    source_fragment_id:
        Parent/source fragment.

    target_fragment_id:
        Child/target fragment.

    depth:
        Fragmentation depth of this edge.

    This table stores the selected/adopted edge.
    Detailed cleavage information is stored in CleavageEventTable.
    """

    __tablename__ = "fragment_edge"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    source_fragment_id: Mapped[int] = mapped_column(
        ForeignKey("fragment.id"),
        nullable=False,
    )

    target_fragment_id: Mapped[int] = mapped_column(
        ForeignKey("fragment.id"),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "source_fragment_id",
            "target_fragment_id",
            name="uq_fragment_edge_source_target",
        ),
    )