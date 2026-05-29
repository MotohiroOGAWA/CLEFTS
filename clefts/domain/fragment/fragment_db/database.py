from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .tables import Base
from .path import FragmentGraphProjectPath


@event.listens_for(Engine, "connect")
def enable_sqlite_settings(dbapi_connection: Any, connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class FragmentGraphDatabase:
    def __init__(
        self,
        project_path: FragmentGraphProjectPath,
        *,
        create_if_missing: bool = True,
    ) -> None:
        self.project_path = project_path

        if create_if_missing:
            self.project_path.database_dir.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{self.project_path.database_path}",
            future=True,
        )

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )

        if create_if_missing:
            Base.metadata.create_all(self.engine)

    @classmethod
    def from_names(
        cls,
        root_dir: str | Path,
        project_name: str,
        database_name: str,
        *,
        create_if_missing: bool = True,
    ) -> FragmentGraphDatabase:
        return cls(
            project_path=FragmentGraphProjectPath(
                root_dir=Path(root_dir),
                project_name=project_name,
                database_name=database_name,
            ),
            create_if_missing=create_if_missing,
        )

    def session(self) -> Session:
        return self.SessionLocal()