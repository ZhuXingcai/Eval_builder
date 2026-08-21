from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from env_mock_agent.schemas import ProblemLedger, RunStatus, TaskSpec
from env_mock_agent.store import RunStore


class PackageProfile(ABC):
    name: str

    @abstractmethod
    def export(self, store: RunStore, task: TaskSpec, destination: Path, *, overwrite: bool = False) -> Path:
        raise NotImplementedError

    @staticmethod
    def prepare_destination(destination: Path, overwrite: bool) -> Path:
        resolved = destination.expanduser().resolve()
        if resolved.exists():
            if not overwrite:
                raise FileExistsError(f"destination already exists: {resolved}")
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True)
        return resolved

    @staticmethod
    def assert_exportable(store: RunStore) -> None:
        record = store.read_record()
        if record.status not in {RunStatus.PACKAGED, RunStatus.EXPORTED}:
            raise RuntimeError(
                f"run is not exportable in status {record.status.value}; package through the release gate first"
            )
        ledger = store.read_model("validation/problem_ledger.json", ProblemLedger)
        if ledger.has_release_blocker:
            raise RuntimeError("run has open P0/P1 problems and cannot be exported")

    @staticmethod
    def copy_workspace(store: RunStore, destination: Path) -> None:
        source = store.path / "package"
        workspace = destination / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for item in source.iterdir():
                target = workspace / item.name
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
