from __future__ import annotations

from sqlalchemy import (
    Integer,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .base import Base

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