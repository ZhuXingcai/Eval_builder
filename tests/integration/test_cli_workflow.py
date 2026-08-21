from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

from env_mock_agent.cli import app
from env_mock_agent.graph import EnvironmentMockWorkflow
from env_mock_agent.schemas import DependencySpec, RunRecord, RunStatus, TaskSpec
from env_mock_agent.store import RunStore, sqlite_checkpointer

runner = CliRunner()


def write_task(tmp_path: Path) -> Path:
    task = TaskSpec(
        task_id="CLI-1",
        task_name="CLI fixture",
        prompt="Use the provided input.",
        dependencies=[
            DependencySpec(
                dependency_id="D-001",
                path="input.txt",
                asset_type="txt",
                expected_content="Neutral input facts.",
            )
        ],
    )
    path = tmp_path / "task.json"
    path.write_text(task.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_generate_cli_runs_durable_workflow(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("ENVMOCK_RUNS_ROOT", str(runs))
    result = runner.invoke(
        app,
        [
            "generate",
            "--task",
            str(write_task(tmp_path)),
            "--run-id",
            "cli-generate",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == RunStatus.PACKAGED.value
    assert RunStore.open(runs, "cli-generate").read_record().status == RunStatus.PACKAGED


def test_resume_cli_continues_interrupted_checkpoint(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("ENVMOCK_RUNS_ROOT", str(runs))
    task = TaskSpec.model_validate_json(write_task(tmp_path).read_text(encoding="utf-8"))
    store = RunStore(runs, "cli-resume")
    store.initialize(
        RunRecord(run_id="cli-resume", task_id=task.task_id, profile="generic"),
        task,
    )
    with sqlite_checkpointer(runs / ".checkpoints.sqlite") as checkpointer:
        graph = cast(
            Any,
            EnvironmentMockWorkflow().compile(
                checkpointer,
                interrupt_after=["build_artifacts"],
            ),
        )
        graph.invoke(
            {
                "run_id": store.run_id,
                "runs_root": str(runs),
                "profile": "generic",
                "task": task.model_dump(mode="json"),
                "overwrite": False,
            },
            {"configurable": {"thread_id": store.run_id}},
        )
    result = runner.invoke(app, ["resume", "cli-resume"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == RunStatus.PACKAGED.value


def test_cancel_cli_persists_status_and_export_cannot_bypass_gate(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("ENVMOCK_RUNS_ROOT", str(runs))
    task_path = write_task(tmp_path)
    create = runner.invoke(
        app,
        [
            "create-run",
            "--task",
            str(task_path),
            "--run-id",
            "cli-cancel",
        ],
    )
    assert create.exit_code == 0, create.output
    blocked_export = runner.invoke(
        app,
        [
            "export",
            "cli-cancel",
            "--profile",
            "generic",
            "--output",
            str(tmp_path / "forbidden-export"),
        ],
    )
    assert blocked_export.exit_code != 0
    assert not (tmp_path / "forbidden-export").exists()

    cancelled = runner.invoke(app, ["cancel", "cli-cancel"])
    assert cancelled.exit_code == 0, cancelled.output
    assert RunStore.open(runs, "cli-cancel").read_record().status == RunStatus.CANCELLED
    assert (runs / "cli-cancel/cancel.json").is_file()
