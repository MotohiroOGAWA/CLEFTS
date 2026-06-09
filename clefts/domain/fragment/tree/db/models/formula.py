from __future__ import annotations

from sqlalchemy import (
    Integer,
    Float,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .base import Base

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