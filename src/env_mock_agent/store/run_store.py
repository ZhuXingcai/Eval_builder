from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from env_mock_agent.schemas import (
    ProblemLedger,
    RunRecord,
    RunStatus,
    SourceEvidence,
    TaskSpec,
    WorldLedger,
)
from env_mock_agent.store.event_log import EventLog

ModelT = TypeVar("ModelT", bound=BaseModel)


class RunStore:
    DIRECTORIES = (
        "task",
        "evidence/downloads",
        "world",
        "staging",
        "validation",
        "operations",
        "package",
        "export",
        "transcripts/claude_cli",
        "transcripts/claude_sdk",
        "transcripts/pi",
    )

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root.expanduser().resolve()
        self.run_id = run_id
        self.path = self.root / run_id
        self.events = EventLog(self.path / "events.jsonl")
        self.evidence = EventLog(self.path / "evidence/sources.jsonl")

    def initialize(self, record: RunRecord, task: TaskSpec) -> None:
        self.path.mkdir(parents=True, exist_ok=False)
        for directory in self.DIRECTORIES:
            (self.path / directory).mkdir(parents=True, exist_ok=True)
        self.write_model("run.json", record)
        self.write_model("task/normalized_task.json", task)
        self.write_json(
            "task/dependency_graph.json",
            {"dependencies": [item.model_dump(mode="json") for item in task.dependencies]},
        )
        self.write_json(
            "task/forbidden_outputs.json",
            {"forbidden_outputs": [item.model_dump(mode="json") for item in task.forbidden_outputs]},
        )
        self.write_model("world/ledger.json", WorldLedger())
        self.write_model("validation/problem_ledger.json", ProblemLedger())

    def write_json(self, relative_path: str, value: object) -> Path:
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._atomic_write(target, serialized)
        return target

    def write_model(self, relative_path: str, value: BaseModel) -> Path:
        return self.write_json(relative_path, value.model_dump(mode="json"))

    def read_json(self, relative_path: str) -> object:
        return json.loads((self.path / relative_path).read_text(encoding="utf-8"))

    def read_model(self, relative_path: str, model_type: type[ModelT]) -> ModelT:
        return model_type.model_validate(self.read_json(relative_path))

    def read_record(self) -> RunRecord:
        return self.read_model("run.json", RunRecord)

    def update_status(
        self,
        status: RunStatus,
        *,
        current_node: str | None = None,
        error: str | None = None,
    ) -> RunRecord:
        record = self.read_record()
        record.status = status
        record.current_node = current_node
        record.error = error
        record.updated_at = datetime.now(UTC)
        self.write_model("run.json", record)
        return record

    def staging_path(self, artifact_id: str) -> Path:
        path = self.path / "staging" / artifact_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append_evidence(self, evidence: SourceEvidence) -> None:
        self.evidence.append(evidence.model_dump(mode="json"))

    def read_evidence(self) -> list[SourceEvidence]:
        return [SourceEvidence.model_validate(item) for item in self.evidence.read()]

    def operation_path(self, operation_key: str) -> Path:
        return self.path / "operations" / f"{operation_key}.json"

    def read_operation(self, operation_key: str) -> object | None:
        path = self.operation_path(operation_key)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def write_operation(self, operation_key: str, value: object) -> Path:
        return self.write_json(f"operations/{operation_key}.json", value)

    @staticmethod
    def _atomic_write(target: Path, text: str) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def open(cls, root: Path, run_id: str) -> RunStore:
        store = cls(root, run_id)
        if not (store.path / "run.json").exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        return store
