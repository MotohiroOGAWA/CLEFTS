from __future__ import annotations

from sqlalchemy import (
    Integer,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .base import Base

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