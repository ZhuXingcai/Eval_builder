from __future__ import annotations

from typing import Protocol

from eval_factory.agent_system.candidate_output import (
    CandidateOutputWrite,
)
from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetRuntime,
)
from eval_factory.agent_system.graph import FactoryGraphState
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
)
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
    GenericAgentFactoryWorkflowError,
)
from eval_factory.team import TeamTaskStatusV1


class FactoryDatasetOutputReader(Protocol):
    def get(self, run_id: str) -> CandidateOutputWrite: ...


class FactoryDatasetWorkflowTerminalCommands:
    """Owns control-capability dispatch and immutable delivery verification."""

    def __init__(
        self,
        *,
        runtime: FactoryDatasetRuntime,
        workflow: GenericAgentFactoryWorkflow,
        output_reader: FactoryDatasetOutputReader,
        team_id: str,
        request: FactoryDatasetRunRequestV2,
        audit: ContractAudit,
    ) -> None:
        self.runtime = runtime
        self.workflow = workflow
        self.output_reader = output_reader
        self.team_id = team_id
        self.request = request
        self.audit = audit

    async def assemble_core_output(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        incomplete_reviews = tuple(
            work.task.to_ref()
            for work in self.workflow.store.list_task_work(self.team_id)
            if work.task.task_kind.startswith("plan-review") and work.status is not TeamTaskStatusV1.COMPLETED
        )
        if incomplete_reviews:
            raise GenericAgentFactoryWorkflowError(
                "delivery closure requires completed PlanReview tasks",
            )
        return self.runtime.current_view(
            request=self.request,
            audit=self.audit,
        )

    async def review_final_delivery(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        delivery_tasks = tuple(
            work.task.task_id
            for work in self.workflow.store.list_task_work(self.team_id)
            if work.task.task_kind == "delivery"
        )
        if len(delivery_tasks) != 1:
            raise GenericAgentFactoryWorkflowError(
                "delivery task is not uniquely materialized",
            )
        await self.workflow.dispatch_task(
            self.team_id,
            delivery_tasks[0],
            audit=self.audit,
        )
        return self.runtime.current_view(
            request=self.request,
            audit=self.audit,
        )

    async def export_core_output(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        output = self.output_reader.get(
            self.request.dataset_run_id,
        )
        run = self.runtime.store.get_run(
            self.request.dataset_run_id,
        )
        if (
            run.status is not FactoryRunStatusV2.COMPLETED
            or run.completion_ref != output.completion.to_ref()
            or run.delivery_manifest_ref != output.manifest.to_ref()
        ):
            raise ValueError(
                "candidate output differs from completed Factory authority",
            )
        return self.runtime.current_view(
            request=self.request,
            audit=self.audit,
            run=run,
        )

    def _validate_state(
        self,
        state: FactoryGraphState,
    ) -> None:
        snapshot = self.workflow.validate_team(self.team_id)
        if (
            state["factory_run_id"] != self.request.dataset_run_id
            or state.get("team_ref") != snapshot.team.to_ref()
            or state.get("blueprint_ref") != self.workflow.blueprint.to_ref()
        ):
            raise ValueError(
                "dataset Graph terminal authority drifted",
            )


__all__ = [
    "FactoryDatasetOutputReader",
    "FactoryDatasetWorkflowTerminalCommands",
]
