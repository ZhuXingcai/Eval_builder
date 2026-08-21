from __future__ import annotations

from typing import Protocol

from eval_factory.agent_system.dataset_graph_handlers import (
    FactoryDatasetNodeCommand,
)
from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetRuntime,
)
from eval_factory.agent_system.graph import FactoryGraphState
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
)


class FactoryDatasetTerminalCommands(Protocol):
    async def assemble_core_output(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2: ...

    async def review_final_delivery(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2: ...

    async def export_core_output(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2: ...


class FactoryDatasetGraphNodeCommands:
    """Node-owned dataset commands used by the default Graph facade."""

    _NODES = (
        "requirement_planner",
        "review_global_plan",
        "compile_global_plan",
        "dispatch_ready_work",
        "validate_core_vertical",
        "planner_assessment",
        "assemble_core_output",
        "review_final_delivery",
        "export_core_output",
    )
    _EXECUTION_CAPABILITY_IDS = frozenset(
        {
            "capability.requirement-planning",
            "capability.trace-ingestion",
            "capability.task-authoring",
            "capability.attachment-reconstruction",
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.quality-review",
            "capability.batch-quality",
            "capability.plan-review",
        }
    )

    def __init__(
        self,
        *,
        runtime: FactoryDatasetRuntime,
        workflow: GenericAgentFactoryWorkflow,
        terminal: FactoryDatasetTerminalCommands,
        team_id: str,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> None:
        self.runtime = runtime
        self.store = runtime.store
        self.workflow = workflow
        self.terminal = terminal
        self.team_id = team_id
        self.request = request
        self.policy = policy
        self.requirement = requirement
        self.audit = audit

    def command(self, node: str) -> FactoryDatasetNodeCommand:
        commands: dict[str, FactoryDatasetNodeCommand] = {
            "requirement_planner": self.requirement_planner,
            "review_global_plan": self.review_global_plan,
            "compile_global_plan": self.compile_global_plan,
            "dispatch_ready_work": self.dispatch_ready_work,
            "validate_core_vertical": self.validate_core_vertical,
            "planner_assessment": self.planner_assessment,
            "assemble_core_output": (self.terminal.assemble_core_output),
            "review_final_delivery": (self.terminal.review_final_delivery),
            "export_core_output": (self.terminal.export_core_output),
        }
        try:
            return commands[node]
        except KeyError as exc:
            raise ValueError(
                "dataset Graph command node is unknown",
            ) from exc

    async def requirement_planner(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        return await self.runtime.plan_requirement(
            request=self.request,
            policy=self.policy,
            requirement=self.requirement,
            audit=self.audit,
        )

    async def review_global_plan(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        return self.runtime.review_global_plan(
            request=self.request,
            audit=self.audit,
        )

    async def compile_global_plan(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        self.workflow.validate_team(self.team_id)
        return self.runtime.verify_compiled_global_plan(
            request=self.request,
            audit=self.audit,
        )

    async def dispatch_ready_work(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        await self.workflow.dispatch_ready(
            self.team_id,
            audit=self.audit,
            capability_ids=self._EXECUTION_CAPABILITY_IDS,
        )
        return self.runtime.current_view(
            request=self.request,
            audit=self.audit,
        )

    async def validate_core_vertical(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        self.workflow.validate_team(self.team_id)
        return self.runtime.current_view(
            request=self.request,
            audit=self.audit,
        )

    async def planner_assessment(
        self,
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        self._validate_state(state)
        return self.runtime.current_view(
            request=self.request,
            audit=self.audit,
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
                "dataset Graph command state authority drifted",
            )


__all__ = [
    "FactoryDatasetGraphNodeCommands",
    "FactoryDatasetTerminalCommands",
]
