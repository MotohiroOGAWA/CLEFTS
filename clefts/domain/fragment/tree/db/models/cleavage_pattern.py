from __future__ import annotations

from sqlalchemy import Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CleavagePatternTable(Base):
    """One unique cleavage pattern.

    This corresponds to _CleavagePattern.

    Notes
    -----
    cleavage_pattern_id:
        Global database ID.

    pattern_key:
        Semantic identity of _CleavagePattern.
        Usually:
            reactant_smarts>>product_smarts_1|product_smarts_2|...

    name:
        Metadata only. It should not be used as the identity.
    """

    __tablename__ = "cleavage_pattern"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    reactant_smarts: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )