from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
    FactoryDatasetRuntime,
)
from eval_factory.agent_system.graph import (
    FactoryGraphNodeHandler,
    FactoryGraphNodeUpdate,
    FactoryGraphSignalV2,
    FactoryGraphState,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    PlannerAssessmentActionV2,
    PlannerAssessmentV2,
)
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
)

FactoryPlannerAssessmentFactory = Callable[
    [
        FactoryGraphState,
        FactoryDatasetRunViewV2,
        FactoryRunV2,
    ],
    PlannerAssessmentV2,
]


FactoryDatasetNodeCommand = Callable[
    [FactoryGraphState],
    Awaitable[FactoryDatasetRunViewV2],
]


class FactoryDatasetGraphCommands(Protocol):
    store: FactoryControlStore

    def command(self, node: str) -> FactoryDatasetNodeCommand: ...


class FactoryDatasetAdvanceCompatibilityCommands:
    """Explicit legacy adapter; never mount this as the product default."""

    def __init__(
        self,
        *,
        runtime: FactoryDatasetRuntime,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        core_input: FactoryDatasetCoreInput | None,
        audit: ContractAudit,
    ) -> None:
        self.runtime = runtime
        self.store = runtime.store
        self.request = request
        self.policy = policy
        self.requirement = requirement
        self.core_input = core_input
        self.audit = audit

    def command(self, node: str) -> FactoryDatasetNodeCommand:
        if node not in FactoryDatasetGraphHandlers._NODES:
            raise ValueError("dataset Graph command node is unknown")

        async def execute(
            state: FactoryGraphState,
        ) -> FactoryDatasetRunViewV2:
            del state
            return await self.runtime.advance(
                request=self.request,
                policy=self.policy,
                requirement=self.requirement,
                core_input=self.core_input,
                audit=self.audit,
            )

        return execute


class FactoryDatasetNodeCommandRegistry:
    """Requires one independently supplied command for every stable node."""

    def __init__(
        self,
        *,
        store: FactoryControlStore,
        commands: dict[str, FactoryDatasetNodeCommand],
    ) -> None:
        missing = set(FactoryDatasetGraphHandlers._NODES) - set(
            commands,
        )
        extra = set(commands) - set(
            FactoryDatasetGraphHandlers._NODES,
        )
        if missing or extra:
            raise ValueError(
                "dataset Graph command registry must cover exact nodes",
            )
        self._commands = dict(commands)
        self.store = store

    def command(self, node: str) -> FactoryDatasetNodeCommand:
        try:
            return self._commands[node]
        except KeyError as exc:
            raise ValueError(
                "dataset Graph command node is unknown",
            ) from exc


class FactoryDatasetGraphHandlers:
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

    def __init__(
        self,
        *,
        runtime: FactoryDatasetRuntime | None = None,
        commands: FactoryDatasetGraphCommands | None = None,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        core_input: FactoryDatasetCoreInput | None,
        audit: ContractAudit,
        assessment_factory: FactoryPlannerAssessmentFactory | None = None,
    ) -> None:
        if (runtime is None) == (commands is None):
            raise ValueError(
                "dataset Graph handlers require one command source",
            )
        self.request = request
        self.policy = policy
        self.requirement = requirement
        self.core_input = core_input
        self.audit = audit
        self.assessment_factory = assessment_factory
        self.runtime = runtime
        self.commands = commands or (
            FactoryDatasetAdvanceCompatibilityCommands(
                runtime=runtime,  # type: ignore[arg-type]
                request=request,
                policy=policy,
                requirement=requirement,
                core_input=core_input,
                audit=audit,
            )
        )

    def mapping(self) -> dict[str, FactoryGraphNodeHandler]:
        return {node: self._handler(node) for node in self._NODES}

    def _handler(
        self,
        node: str,
    ) -> FactoryGraphNodeHandler:
        async def handle(
            state: FactoryGraphState,
        ) -> FactoryGraphNodeUpdate:
            if state["factory_run_id"] != self.request.dataset_run_id:
                raise ValueError("graph state does not bind dataset request")
            view = await self.commands.command(node)(state)
            run = self._store().get_run(
                self.request.dataset_run_id,
            )
            if node == "planner_assessment" and self.assessment_factory is not None:
                proposed = self.assessment_factory(
                    state,
                    view,
                    run,
                )
                run = self._store().commit_planner_assessment(
                    run_id=run.run_id,
                    expected_run_version=run.run_version,
                    assessment=proposed,
                    audit=self.audit,
                    idempotency_key=(f"graph-planner-assessment-{proposed.object_sha256}"),
                )
                if run.planner_assessment_ref is None:
                    raise ValueError(
                        "committed PlannerAssessment has no run binding",
                    )
                assessment = self._store().get_planner_assessment(
                    run.planner_assessment_ref,
                )
                signal = self.signal_from_assessment(assessment)
            else:
                signal = self._signal(
                    node=node,
                    view=view,
                )
            return {
                "compiled_plan_ref": run.compiled_plan_ref,
                "completion_ref": run.completion_ref,
                "current_plan_ref": run.current_plan_ref,
                "delivery_manifest_ref": (run.delivery_manifest_ref),
                "expected_run_version": run.run_version,
                "pending_review_ref": run.pending_review_ref,
                "planner_assessment_ref": (run.planner_assessment_ref),
                "run_ref": run.to_ref(),
                "run_status": run.status,
                "signal": signal,
            }

        return handle

    def _store(self) -> FactoryControlStore:
        return self.commands.store

    @staticmethod
    def _signal(
        *,
        node: str,
        view: FactoryDatasetRunViewV2,
    ) -> FactoryGraphSignalV2:
        if view.status in {
            FactoryRunStatusV2.BLOCKED,
            FactoryRunStatusV2.FAILED,
            FactoryRunStatusV2.CANCELLED,
        }:
            return FactoryGraphSignalV2.BLOCKED
        if view.status is FactoryRunStatusV2.COMPLETED:
            return (
                FactoryGraphSignalV2.CONTINUE
                if node
                in {
                    "assemble_core_output",
                    "review_final_delivery",
                }
                else FactoryGraphSignalV2.FINISH
            )
        if view.status is FactoryRunStatusV2.WAITING_REVIEW:
            return (
                FactoryGraphSignalV2.CONTINUE if node == "requirement_planner" else FactoryGraphSignalV2.WAIT
            )
        if node in {
            "requirement_planner",
            "review_global_plan",
            "compile_global_plan",
            "dispatch_ready_work",
            "validate_core_vertical",
            "assemble_core_output",
        }:
            return FactoryGraphSignalV2.CONTINUE
        return FactoryGraphSignalV2.WAIT

    @staticmethod
    def signal_from_assessment(
        assessment: PlannerAssessmentV2,
    ) -> FactoryGraphSignalV2:
        return {
            PlannerAssessmentActionV2.CONTINUE: (FactoryGraphSignalV2.CONTINUE),
            PlannerAssessmentActionV2.RETRY: (FactoryGraphSignalV2.RETRY),
            PlannerAssessmentActionV2.REPLAN: (FactoryGraphSignalV2.REPLAN),
            PlannerAssessmentActionV2.ESCALATE: (FactoryGraphSignalV2.ESCALATE),
            PlannerAssessmentActionV2.FINISH: (FactoryGraphSignalV2.FINISH),
        }[assessment.proposed_action]


__all__ = [
    "FactoryDatasetAdvanceCompatibilityCommands",
    "FactoryDatasetGraphCommands",
    "FactoryDatasetGraphHandlers",
    "FactoryDatasetNodeCommand",
    "FactoryDatasetNodeCommandRegistry",
    "FactoryPlannerAssessmentFactory",
]
