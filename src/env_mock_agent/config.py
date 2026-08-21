from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / ".specify").is_dir():
            return candidate
    raise RuntimeError(f"Environment Mock Agent project root not found from {current}")


class Settings(BaseModel):
    project_root: Path = Field(default_factory=find_project_root)
    runs_dir_name: str = Field(default_factory=lambda: os.environ.get("ENVMOCK_RUNS_ROOT", "runs"))
    default_quality_profile: str = "standard"
    local_process_trusted_only: bool = True
    claude_cli_path: str = Field(default_factory=lambda: os.environ.get("CLAUDE_BIN", "claude"))
    pi_bridge_command: tuple[str, ...] = (
        "node",
        "node/pi_bridge/dist/index.js",
    )

    @property
    def runs_root(self) -> Path:
        return self.project_root / self.runs_dir_name

    @property
    def checkpoint_path(self) -> Path:
        return self.runs_root / ".checkpoints.sqlite"


def get_settings(project_root: Path | None = None) -> Settings:
    if project_root is None:
        return Settings()
    return Settings(project_root=project_root.resolve())
