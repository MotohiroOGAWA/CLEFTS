from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATABASES_DIRNAME = "databases"

PROJECT_CONFIG_FILENAME = "project.yaml"
DATABASE_CONFIG_FILENAME = "database.yaml"

DATABASE_FILENAME = "fragment_graph.db"


@dataclass(frozen=True, slots=True)
class FragmentGraphProjectPath:
    root_dir: Path
    project_name: str
    database_name: str

    @property
    def project_dir(self) -> Path:
        return self.root_dir / self.project_name

    @property
    def databases_dir(self) -> Path:
        return self.project_dir / DATABASES_DIRNAME

    @property
    def database_dir(self) -> Path:
        return self.databases_dir / self.database_name

    @property
    def database_path(self) -> Path:
        return self.database_dir / DATABASE_FILENAME

    @property
    def project_config_path(self) -> Path:
        return self.project_dir / PROJECT_CONFIG_FILENAME

    @property
    def database_config_path(self) -> Path:
        return self.database_dir / DATABASE_CONFIG_FILENAME

    def ensure_dirs(self) -> None:
        self.database_dir.mkdir(parents=True, exist_ok=True)

