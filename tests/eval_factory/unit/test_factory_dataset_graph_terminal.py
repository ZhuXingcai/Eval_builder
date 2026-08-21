from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from eval_factory.agent_system.dataset_graph_terminal import (
    FactoryDatasetWorkflowTerminalCommands,
)
from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.graph import (
    FactoryGraphSignalV2,
    FactoryGraphState,
)
from eval_factory.contracts.agent_system_v2 import FactoryRunStatusV2
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
    GenericAgentFactoryWorkflowError,
)
from eval_factory.team import TeamTaskStatusV1


def _ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v1",
        object_sha256="a" * 64,
    )


@dataclass(frozen=True)
class _Value:
    reference: ObjectRef

    def to_ref(self) -> ObjectRef:
        return self.reference


@dataclass(frozen=True)
class _Task:
    task_id: str
    task_kind: str
    reference: ObjectRef

    def to_ref(self) -> ObjectRef:
        return self.reference


class _Workflow:
    def __init__(
        self,
        *,
        reviews_complete: bool = True,
        delivery_count: int = 1,
    ) -> None:
        self.team = _Value(_ref("agent-team", "terminal"))
        self.blueprint = _Value(
            _ref("evaluation-blueprint", "terminal"),
        )
        self.calls: list[str] = []
        work = [
            SimpleNamespace(
                task=_Task(
                    "task-plan-review",
                    "plan-review-final-delivery",
                    _ref("team-task", "plan-review"),
                ),
                status=(TeamTaskStatusV1.COMPLETED if reviews_complete else TeamTaskStatusV1.READY),
            ),
        ]
        work.extend(
            SimpleNamespace(
                task=_Task(
                    f"task-delivery-{index}",
                    "delivery",
                    _ref("team-task", f"delivery-{index}"),
                ),
                status=TeamTaskStatusV1.PENDING,
            )
            for index in range(delivery_count)
        )
        self.store = SimpleNamespace(
            list_task_work=lambda team_id: tuple(work) if team_id == "team-terminal" else (),
        )

    def validate_team(self, team_id: str) -> Any:
        assert team_id == "team-terminal"
        return SimpleNamespace(team=self.team)

    async def dispatch_task(
        self,
        team_id: str,
        task_id: str,
        *,
        audit: object,
    ) -> object:
        del audit
        assert team_id == "team-terminal"
        self.calls.append(task_id)
        return object()


class _Store:
    def __init__(self) -> None:
        self.run = SimpleNamespace(
            status=FactoryRunStatusV2.COMPLETED,
            completion_ref=_ref("factory-run-completion", "terminal"),
            delivery_manifest_ref=_ref(
                "candidate-dataset-delivery-manifest",
                "terminal",
            ),
        )

    def get_run(self, run_id: str) -> Any:
        assert run_id == "factory-run.terminal"
        return self.run


class _Runtime:
    def __init__(self, view: object) -> None:
        self.store = _Store()
        self.view = view

    def current_view(self, **kwargs: object) -> object:
        assert kwargs["request"] is REQUEST
        return self.view


class _OutputReader:
    def __init__(self, store: _Store) -> None:
        self.store = store
        self.calls: list[str] = []

    def get(self, run_id: str) -> Any:
        self.calls.append(run_id)
        return SimpleNamespace(
            completion=_Value(self.store.run.completion_ref),
            manifest=_Value(self.store.run.delivery_manifest_ref),
        )


REQUEST = cast(
    FactoryDatasetRunRequestV2,
    SimpleNamespace(dataset_run_id="factory-run.terminal"),
)
VIEW = cast(FactoryDatasetRunViewV2, object())


def _state(workflow: _Workflow) -> FactoryGraphState:
    return {
        "factory_run_id": REQUEST.dataset_run_id,
        "expected_run_version": 1,
        "transition_count": 1,
        "signal": FactoryGraphSignalV2.CONTINUE,
        "team_ref": workflow.team.to_ref(),
        "blueprint_ref": workflow.blueprint.to_ref(),
    }


@pytest.mark.asyncio
async def test_terminal_commands_dispatch_delivery_task_and_verify_output() -> None:
    workflow = _Workflow()
    runtime = _Runtime(VIEW)
    reader = _OutputReader(runtime.store)
    commands = FactoryDatasetWorkflowTerminalCommands(
        runtime=cast(FactoryDatasetRuntime, runtime),
        workflow=cast(GenericAgentFactoryWorkflow, workflow),
        output_reader=reader,
        team_id="team-terminal",
        request=REQUEST,
        audit=cast(Any, object()),
    )
    state = _state(workflow)

    assert await commands.assemble_core_output(state) is VIEW
    assert await commands.review_final_delivery(state) is VIEW
    assert await commands.export_core_output(state) is VIEW
    assert workflow.calls == ["task-delivery-0"]
    assert reader.calls == [REQUEST.dataset_run_id]


@pytest.mark.asyncio
async def test_terminal_commands_require_completed_reviews_and_unique_delivery() -> None:
    for workflow, message, operation in (
        (
            _Workflow(reviews_complete=False),
            "completed PlanReview",
            "assemble",
        ),
        (
            _Workflow(delivery_count=2),
            "uniquely materialized",
            "delivery",
        ),
    ):
        runtime = _Runtime(VIEW)
        commands = FactoryDatasetWorkflowTerminalCommands(
            runtime=cast(FactoryDatasetRuntime, runtime),
            workflow=cast(GenericAgentFactoryWorkflow, workflow),
            output_reader=_OutputReader(runtime.store),
            team_id="team-terminal",
            request=REQUEST,
            audit=cast(Any, object()),
        )
        with pytest.raises(
            GenericAgentFactoryWorkflowError,
            match=message,
        ):
            if operation == "assemble":
                await commands.assemble_core_output(_state(workflow))
            else:
                await commands.review_final_delivery(_state(workflow))


@pytest.mark.asyncio
async def test_terminal_commands_fail_closed_on_authority_drift() -> None:
    workflow = _Workflow()
    runtime = _Runtime(VIEW)
    commands = FactoryDatasetWorkflowTerminalCommands(
        runtime=cast(FactoryDatasetRuntime, runtime),
        workflow=cast(GenericAgentFactoryWorkflow, workflow),
        output_reader=_OutputReader(runtime.store),
        team_id="team-terminal",
        request=REQUEST,
        audit=cast(Any, object()),
    )
    stale = _state(workflow)
    stale["team_ref"] = _ref("agent-team", "stale")

    with pytest.raises(ValueError, match="authority drifted"):
        await commands.assemble_core_output(stale)

    runtime.store.run = SimpleNamespace(
        status=FactoryRunStatusV2.RUNNING,
        completion_ref=_ref("factory-run-completion", "terminal"),
        delivery_manifest_ref=_ref(
            "candidate-dataset-delivery-manifest",
            "terminal",
        ),
    )
    with pytest.raises(ValueError, match="completed Factory authority"):
        await commands.export_core_output(_state(workflow))
