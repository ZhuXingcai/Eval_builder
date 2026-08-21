from __future__ import annotations

from pathlib import Path
from typing import Any

from env_mock_agent.graph import EnvironmentMockWorkflow
from env_mock_agent.providers import ProviderRegistry, ProviderRequest, TextProvider
from env_mock_agent.providers.helpers import artifact_result
from env_mock_agent.schemas import (
    ArtifactResult,
    DependencySpec,
    ForbiddenOutputSpec,
    RunRecord,
    RunStatus,
    TaskSpec,
)
from env_mock_agent.store import RunStore, sqlite_checkpointer


def initialize_run(tmp_path: Path, task: TaskSpec, run_id: str = "run") -> RunStore:
    store = RunStore(tmp_path / "runs", run_id)
    store.initialize(
        RunRecord(
            run_id=run_id,
            task_id=task.task_id,
            profile=task.profile,
        ),
        task,
    )
    return store


def initial_state(
    store: RunStore,
    task: TaskSpec,
    *,
    export_path: Path | None = None,
) -> dict[str, object]:
    state: dict[str, object] = {
        "run_id": store.run_id,
        "runs_root": str(store.root),
        "profile": task.profile,
        "task": task.model_dump(mode="json"),
        "overwrite": False,
    }
    if export_path is not None:
        state["export_path"] = str(export_path)
    return state


def invoke(
    workflow: EnvironmentMockWorkflow,
    store: RunStore,
    state: dict[str, object] | None,
    *,
    interrupt_after: list[str] | None = None,
) -> dict[str, Any]:
    with sqlite_checkpointer(store.root / ".checkpoints.sqlite") as checkpointer:
        graph = workflow.compile(checkpointer, interrupt_after=interrupt_after)
        return graph.invoke(  # type: ignore[attr-defined,no-any-return]
            state,
            {"configurable": {"thread_id": store.run_id}},
        )


def test_graph_runs_three_rounds_and_exports_package(tmp_path: Path) -> None:
    task = TaskSpec(
        task_id="T-1",
        task_name="input fixture",
        prompt="Use the provided input to prepare a later analysis.",
        profile="generic",
        dependencies=[
            DependencySpec(
                dependency_id="D-001",
                path="inputs/notes.txt",
                asset_type="txt",
                expected_content="Neutral source facts only.",
            )
        ],
    )
    store = initialize_run(tmp_path, task)
    output = tmp_path / "export"
    result = invoke(
        EnvironmentMockWorkflow(),
        store,
        initial_state(store, task, export_path=output),
    )
    assert result["final_status"] == RunStatus.EXPORTED.value
    assert store.read_record().status == RunStatus.EXPORTED
    assert (output / "workspace/inputs/notes.txt").is_file()
    assert len(result["reviews"]) == 3
    assert all((store.path / f"validation/round_{number}.json").is_file() for number in (1, 2, 3))


def test_graph_resumes_after_interrupted_artifact_build_without_repeating_side_effect(
    tmp_path: Path,
) -> None:
    task = TaskSpec(
        task_id="T-2",
        task_name="resumable fixture",
        prompt="Use the provided input.",
        dependencies=[
            DependencySpec(
                dependency_id="D-001",
                path="input.txt",
                asset_type="txt",
                expected_content="Stable input.",
            )
        ],
    )
    store = initialize_run(tmp_path, task)
    with sqlite_checkpointer(store.root / ".checkpoints.sqlite") as checkpointer:
        graph = EnvironmentMockWorkflow().compile(
            checkpointer,
            interrupt_after=["build_artifacts"],
        )
        graph.invoke(  # type: ignore[attr-defined]
            initial_state(store, task),
            {"configurable": {"thread_id": store.run_id}},
        )
        artifact = store.path / "staging/A-001/input.txt"
        first_mtime = artifact.stat().st_mtime_ns
        result = graph.invoke(  # type: ignore[attr-defined]
            None,
            {"configurable": {"thread_id": store.run_id}},
        )
    assert result["final_status"] == RunStatus.PACKAGED.value
    assert artifact.stat().st_mtime_ns == first_mtime


def test_p0_leakage_blocks_package_export(tmp_path: Path) -> None:
    task = TaskSpec(
        task_id="T-3",
        task_name="leak fixture",
        prompt="Use the provided input.",
        dependencies=[
            DependencySpec(
                dependency_id="D-001",
                path="input.txt",
                asset_type="txt",
                expected_content="Final recommendation: approve.",
            )
        ],
        forbidden_outputs=[
            ForbiddenOutputSpec(
                forbidden_id="F-001",
                description="final recommendation",
                semantic_patterns=["Final recommendation"],
            )
        ],
    )
    store = initialize_run(tmp_path, task)
    output = tmp_path / "export"
    result = invoke(
        EnvironmentMockWorkflow(),
        store,
        initial_state(store, task, export_path=output),
    )
    assert result["release_allowed"] is False
    assert result["final_status"] == RunStatus.FAILED.value
    assert not output.exists()


def test_unsupported_critical_format_returns_blocked_capability(tmp_path: Path) -> None:
    task = TaskSpec(
        task_id="T-4",
        task_name="unsupported fixture",
        prompt="Use the provided slide input.",
        dependencies=[
            DependencySpec(
                dependency_id="D-001",
                path="input.pptx",
                asset_type="pptx",
            )
        ],
    )
    store = initialize_run(tmp_path, task)
    result = invoke(
        EnvironmentMockWorkflow(),
        store,
        initial_state(store, task),
    )
    assert result["release_allowed"] is False
    assert result["final_status"] == RunStatus.BLOCKED_CAPABILITY.value


class RepairingTextProvider(TextProvider):
    name = "repairing_text"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ArtifactResult:
        self.calls += 1
        if self.calls == 1:
            target = request.destination()
            target.write_text("", encoding="utf-8")
            return artifact_result(request.plan.artifact_id, target, self.name)
        return super().generate(request)


def test_targeted_repair_rebuilds_only_failed_artifact(tmp_path: Path) -> None:
    provider = RepairingTextProvider()
    providers = ProviderRegistry()
    providers.register(provider)
    task = TaskSpec(
        task_id="T-5",
        task_name="repair fixture",
        prompt="Use the provided input.",
        dependencies=[
            DependencySpec(
                dependency_id="D-001",
                path="input.txt",
                asset_type="txt",
                expected_content="Repaired neutral input.",
            )
        ],
    )
    store = initialize_run(tmp_path, task)
    result = invoke(
        EnvironmentMockWorkflow(providers=providers),
        store,
        initial_state(store, task),
    )
    assert result["release_allowed"] is True
    assert provider.calls == 2
    assert result["repair_attempts"] == {"A-001": 1}
