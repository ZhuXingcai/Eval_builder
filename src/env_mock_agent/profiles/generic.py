from __future__ import annotations

import json
from pathlib import Path

from env_mock_agent.profiles.base import PackageProfile
from env_mock_agent.schemas import TaskSpec
from env_mock_agent.store import RunStore


class GenericProfile(PackageProfile):
    name = "generic"

    def export(self, store: RunStore, task: TaskSpec, destination: Path, *, overwrite: bool = False) -> Path:
        self.assert_exportable(store)
        output = self.prepare_destination(destination, overwrite)
        (output / "task.json").write_text(
            json.dumps(task.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.copy_workspace(store, output)
        return output
