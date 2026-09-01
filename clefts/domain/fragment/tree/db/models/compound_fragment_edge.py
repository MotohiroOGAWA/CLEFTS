from __future__ import annotations

from sqlalchemy import (
    ForeignKey,
    Integer,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CompoundFragmentEdgeTable(Base):
    """One edge adopted in the fragmentation graph of one compound.

    Notes
    -----
    compound_id:
        Original compound whose fragmentation graph contains this edge.

    edge_id:
        Universal fragment edge ID.

    depth:
        Depth at which this edge appears in the compound-specific
        fragmentation graph.

    This table represents compound-specific edge adoption.
    The source and target fragments are stored in FragmentEdgeTable.
    """

    __tablename__ = "compound_fragment_edge"

    compound_id: Mapped[int] = mapped_column(
        ForeignKey("compound.id"),
        primary_key=True,
    )

    edge_id: Mapped[int] = mapped_column(
        ForeignKey("fragment_edge.id"),
        primary_key=True,
    )

    depth: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )