from __future__ import annotations

from pathlib import Path

from env_mock_agent.schemas import (
    RunRecord,
    RunStatus,
    SourceEvidence,
    SourceType,
    TaskSpec,
)
from env_mock_agent.store import RunStore


def make_task() -> TaskSpec:
    return TaskSpec(task_id="T-1", task_name="task", prompt="Build from these inputs.")


def test_run_store_initializes_and_updates_atomically(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run-1")
    store.initialize(RunRecord(run_id="run-1", task_id="T-1", profile="generic"), make_task())
    assert store.read_record().status == RunStatus.CREATED
    updated = store.update_status(RunStatus.PLANNING, current_node="plan")
    assert updated.current_node == "plan"
    assert RunStore.open(tmp_path, "run-1").read_record().status == RunStatus.PLANNING


def test_event_log_does_not_create_run_directory_early(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run-2")
    assert not store.path.exists()


def test_run_store_appends_and_reads_source_evidence(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run-3")
    store.initialize(
        RunRecord(run_id="run-3", task_id="T-1", profile="generic"),
        make_task(),
    )
    source = SourceEvidence(
        source_id="source-1",
        url="https://example.gov/source",
        source_type=SourceType.OFFICIAL,
        sha256="a" * 64,
    )
    store.append_evidence(source)
    assert store.read_evidence() == [source]
