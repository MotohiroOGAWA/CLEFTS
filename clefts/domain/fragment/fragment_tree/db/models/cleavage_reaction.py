from __future__ import annotations

from sqlalchemy import (
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CleavageReactionTable(Base):
    """One reaction belonging to one cleavage pattern.

    This corresponds to CleavageReaction.

    Notes
    -----
    cleavage_pattern_id:
        Parent CleavagePattern ID.

    reaction_id:
        Local ID inside one CleavagePattern.
        This corresponds to CleavageReaction.id.

    name:
        Original ProductRule.name.

    product_smarts:
        Original ProductRule.smarts.
    """

    __tablename__ = "cleavage_reaction"

    cleavage_pattern_id: Mapped[int] = mapped_column(
        ForeignKey("cleavage_pattern.id"),
        primary_key=True,
    )

    reaction_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    product_smarts: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "cleavage_pattern_id",
            "product_smarts",
            name="uq_cleavage_reaction_pattern_product_smarts",
        ),
    )