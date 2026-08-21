from __future__ import annotations

from pathlib import Path

from env_mock_agent.profiles.lh import LhProfile
from env_mock_agent.schemas import TaskSpec
from env_mock_agent.store import RunStore


class CcProfile(LhProfile):
    name = "cc"

    def export(self, store: RunStore, task: TaskSpec, destination: Path, *, overwrite: bool = False) -> Path:
        return super().export(store, task, destination, overwrite=overwrite)
