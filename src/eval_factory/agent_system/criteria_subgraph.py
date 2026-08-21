from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from eval_factory.agent_system.criteria_agent import (
    GatewayCriteriaRubricAgent,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewError,
    PlanReviewService,
)
from eval_factory.agent_system.store import (
    FactoryControlStoreError,
)
from eval_factory.agent_system.supervisor import (
    AgentLeaseV2,
    AgentWorkerRef,
    ExecutionSupervisor,
)
from eval_factory.agent_system.workspace import AgentWorkspaceHandle
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AgentTaskV2,
    AgentWorkspaceReceiptV2,
    AgentWorkspaceStateV2,
    AttachmentQualityAssessmentV2,
    AttachmentSubgraphResultV2,
    CompiledCriteriaRubricPlanV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricPlanV2,
    CriteriaRubricResultV2,
    PlanKindV2,
    SolvabilityAssessmentV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.task_v2 import TaskDraftV2
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
)


class CriteriaRubricSubgraphError(RuntimeError):
    pass


class CriteriaRubricRunnerInjectedCrash(RuntimeError):
    pass


class CriteriaRubricRunnerFaultPoint(StrEnum):
    AFTER_RESULT_JOURNAL = "AFTER_RESULT_JOURNAL"
    AFTER_WORKSPACE_COMMIT = "AFTER_WORKSPACE_COMMIT"
    AFTER_AGENT_RESULT = "AFTER_AGENT_RESULT"
    AFTER_DOMAIN_RESULT = "AFTER_DOMAIN_RESULT"


class CriteriaRubricRunnerFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: CriteriaRubricRunnerFaultPoint,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticCriteriaRubricRunnerFaultInjector:
    crash_points: frozenset[CriteriaRubricRunnerFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: CriteriaRubricRunnerFaultPoint,
    ) -> None:
        if point in self.crash_points:
            raise CriteriaRubricRunnerInjectedCrash(f"injected criteria/rubric runner crash at {point.value}")


@dataclass(frozen=True, slots=True)
class CriteriaRubricSupervisedExecution:
    result: CriteriaRubricResultV2
    envelope: AgentResultEnvelopeV2


class SupervisedCriteriaRubricRunner:
    _RESULT_JOURNAL = "criteria-rubric-result.json"

    def __init__(
        self,
        *,
        supervisor: ExecutionSupervisor,
        agent: GatewayCriteriaRubricAgent,
        review_service: PlanReviewService | None = None,
        fault_injector: CriteriaRubricRunnerFaultInjector | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.agent = agent
        self.review_service = review_service or PlanReviewService(
            supervisor.store,
        )
        self.fault_injector = fault_injector

    async def run(
        self,
        *,
        run_id: str,
        plan: CriteriaRubricPlanV2,
        compiled_plan: CompiledCriteriaRubricPlanV2,
        task_draft: TaskDraftV2,
        attachment_quality: AttachmentQualityAssessmentV2,
        solvability: SolvabilityAssessmentV2,
        binding_definitions: tuple[EvaluatorBindingDefinition, ...],
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
        lease_duration: timedelta = timedelta(minutes=10),
    ) -> CriteriaRubricSupervisedExecution:
        try:
            self.review_service.require_resumed_plan(
                run_id=run_id,
                plan_kind=PlanKindV2.CRITERIA_RUBRIC,
                plan_ref=plan.to_ref(),
            )
        except PlanReviewError as exc:
            raise CriteriaRubricSubgraphError(
                "criteria/rubric execution requires resumed plan authority",
            ) from exc
        self._validate_preflight(
            run_id=run_id,
            plan=plan,
            compiled_plan=compiled_plan,
            task_draft=task_draft,
            attachment_quality=attachment_quality,
            solvability=solvability,
            tool_catalog=tool_catalog,
            audit=audit,
        )
        task = self.supervisor.materialize_criteria_rubric(
            run_id=run_id,
            compiled_plan=compiled_plan,
            audit=audit,
        )
        prior_envelope = self.supervisor.find_result(
            task.agent_task_id,
        )
        if prior_envelope is None:
            result, envelope = await self._execute_attempt(
                task=task,
                plan=plan,
                task_draft=task_draft,
                attachment_quality=attachment_quality,
                solvability=solvability,
                binding_definitions=binding_definitions,
                tool_catalog=tool_catalog,
                audit=audit,
                lease_duration=lease_duration,
            )
        else:
            result, envelope = self._recover_completed_attempt(
                task=task,
                plan=plan,
                prior_envelope=prior_envelope,
                audit=audit,
            )
        self.supervisor.store.commit_domain_result(
            run_id=run_id,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            result=result,
            idempotency_key=(f"commit-criteria-rubric-result-{result.object_sha256}"),
        )
        self._fault(CriteriaRubricRunnerFaultPoint.AFTER_DOMAIN_RESULT)
        return CriteriaRubricSupervisedExecution(
            result=result,
            envelope=envelope,
        )

    def _validate_preflight(
        self,
        *,
        run_id: str,
        plan: CriteriaRubricPlanV2,
        compiled_plan: CompiledCriteriaRubricPlanV2,
        task_draft: TaskDraftV2,
        attachment_quality: AttachmentQualityAssessmentV2,
        solvability: SolvabilityAssessmentV2,
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> None:
        if (
            compiled_plan.source_plan_ref != plan.to_ref()
            or compiled_plan.task_draft_ref != plan.task_draft_ref
            or compiled_plan.attachment_quality_ref != plan.attachment_quality_ref
            or compiled_plan.solvability_ref != plan.solvability_ref
            or compiled_plan.tool_catalog_ref != plan.tool_catalog_ref
            or compiled_plan.acceptance_check_refs != plan.acceptance_check_refs
        ):
            raise CriteriaRubricSubgraphError("compiled criteria/rubric plan is stale")
        try:
            attachment_material = self.supervisor.store.get_domain_result(
                run_id,
                PlanKindV2.ATTACHMENT_GENERATION,
            )
            attachment_result = AttachmentSubgraphResultV2.model_validate_json(
                attachment_material.result_record_json,
            )
        except (
            FactoryControlStoreError,
            ValidationError,
        ) as exc:
            raise CriteriaRubricSubgraphError(
                "current attachment domain result is unavailable",
            ) from exc
        if (
            attachment_material.result_ref != attachment_result.to_ref()
            or attachment_quality.attachment_subgraph_result_ref != attachment_result.to_ref()
            or solvability.attachment_subgraph_result_ref != attachment_result.to_ref()
            or solvability.quality_assessment_ref != attachment_quality.to_ref()
        ):
            raise CriteriaRubricSubgraphError(
                "criteria/rubric inputs do not bind the current attachment authority",
            )
        self.agent.validate_sources(
            plan=plan,
            task_draft=task_draft,
            attachment_quality=attachment_quality,
            solvability=solvability,
            tool_catalog=tool_catalog,
            audit=audit,
        )

    async def _execute_attempt(
        self,
        *,
        task: AgentTaskV2,
        plan: CriteriaRubricPlanV2,
        task_draft: TaskDraftV2,
        attachment_quality: AttachmentQualityAssessmentV2,
        solvability: SolvabilityAssessmentV2,
        binding_definitions: tuple[EvaluatorBindingDefinition, ...],
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
        lease_duration: timedelta,
    ) -> tuple[CriteriaRubricResultV2, AgentResultEnvelopeV2]:
        worker = AgentWorkerRef(
            worker_id="worker://criteria-rubric",
        )
        lease = self.supervisor.acquire(
            task.agent_task_id,
            worker,
            lease_duration=lease_duration,
            idempotency_key=(f"acquire-criteria-rubric-{task.compiled_plan_ref.object_sha256}"),
        )
        handle = self.supervisor.workspace_manager.recover(
            run_ref=task.run_ref,
            lease=lease,
            audit=audit,
        )
        result = self._load_result_journal(
            handle,
            plan_ref=plan.to_ref(),
        )
        if result is None:
            execution = await self.agent.author(
                task_ref=task.to_ref(),
                plan=plan,
                task_draft=task_draft,
                attachment_quality=attachment_quality,
                solvability=solvability,
                binding_definitions=binding_definitions,
                tool_catalog=tool_catalog,
                audit=audit,
            )
            result = execution.result
            self._persist_result_journal(
                handle,
                result,
            )
        self._fault(CriteriaRubricRunnerFaultPoint.AFTER_RESULT_JOURNAL)
        workspace_receipt = self.supervisor.workspace_manager.commit(
            handle,
            audit=audit,
        )
        self._fault(CriteriaRubricRunnerFaultPoint.AFTER_WORKSPACE_COMMIT)
        envelope = self._build_envelope(
            task=task,
            lease=lease,
            workspace_receipt=workspace_receipt,
            plan=plan,
            result=result,
            audit=result.audit,
        )
        committed = self.supervisor.complete(
            lease=lease,
            envelope=envelope,
        )
        self._fault(CriteriaRubricRunnerFaultPoint.AFTER_AGENT_RESULT)
        return result, committed

    def _recover_completed_attempt(
        self,
        *,
        task: AgentTaskV2,
        plan: CriteriaRubricPlanV2,
        prior_envelope: AgentResultEnvelopeV2,
        audit: ContractAudit,
    ) -> tuple[CriteriaRubricResultV2, AgentResultEnvelopeV2]:
        lease = self.supervisor.get_lease(
            prior_envelope.lease_id,
        )
        handle = self.supervisor.workspace_manager.recover(
            run_ref=task.run_ref,
            lease=lease,
            audit=audit,
        )
        result = self._load_result_journal(
            handle,
            plan_ref=plan.to_ref(),
        )
        if result is None:
            raise CriteriaRubricSubgraphError("completed criteria/rubric workspace lacks its result journal")
        expected = self._build_envelope(
            task=task,
            lease=lease,
            workspace_receipt=handle.receipt,
            plan=plan,
            result=result,
            audit=prior_envelope.audit,
        )
        if expected != prior_envelope:
            raise CriteriaRubricSubgraphError("criteria/rubric Agent result differs from workspace journal")
        return result, prior_envelope

    @staticmethod
    def _build_envelope(
        *,
        task: AgentTaskV2,
        lease: AgentLeaseV2,
        workspace_receipt: AgentWorkspaceReceiptV2,
        plan: CriteriaRubricPlanV2,
        result: CriteriaRubricResultV2,
        audit: ContractAudit,
    ) -> AgentResultEnvelopeV2:
        if result.plan_ref != plan.to_ref():
            raise CriteriaRubricSubgraphError("criteria/rubric result binds another plan")
        outcome = _agent_outcome(result.outcome)
        return AgentResultEnvelopeV2.create(
            result_id=(f"agent-result-envelope://{task.agent_task_id.rsplit('://', 1)[-1]}/{lease.attempt}"),
            task_ref=task.to_ref(),
            agent_definition_ref=lease.agent_definition_ref,
            attempt=lease.attempt,
            lease_id=lease.lease_id,
            fencing_token=lease.fencing_token,
            workspace_receipt_ref=workspace_receipt.to_ref(),
            outcome=outcome,
            output_refs=(result.to_ref(),),
            delegated_dataset_job_result_refs=(),
            gateway_receipt_refs=(result.gateway_receipt_ref,),
            validator_result_refs=_sorted_refs(plan.acceptance_check_refs),
            failure_code=(None if outcome is AgentTaskOutcomeV2.SUCCEEDED else result.reason_codes[0]),
            safe_metrics=(),
            audit=audit,
        )

    @classmethod
    def _load_result_journal(
        cls,
        handle: AgentWorkspaceHandle,
        *,
        plan_ref: ObjectRef,
    ) -> CriteriaRubricResultV2 | None:
        path = handle.path / cls._RESULT_JOURNAL
        if not path.exists():
            if handle.receipt.state is AgentWorkspaceStateV2.COMMITTED:
                raise CriteriaRubricSubgraphError(
                    "committed criteria/rubric workspace lacks its result journal"
                )
            return None
        if path.is_symlink() or not path.is_file():
            raise CriteriaRubricSubgraphError("criteria/rubric result journal is unsafe")
        payload = path.read_bytes()
        try:
            result = CriteriaRubricResultV2.model_validate_json(payload)
        except ValidationError as exc:
            raise CriteriaRubricSubgraphError("criteria/rubric result journal is invalid") from exc
        if result.canonical_json() != payload or result.plan_ref != plan_ref:
            raise CriteriaRubricSubgraphError("criteria/rubric result journal drifted")
        return result

    @classmethod
    def _persist_result_journal(
        cls,
        handle: AgentWorkspaceHandle,
        result: CriteriaRubricResultV2,
    ) -> None:
        path = handle.path / cls._RESULT_JOURNAL
        payload = result.canonical_json()
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise CriteriaRubricSubgraphError("criteria/rubric result journal conflicts")
            return
        if handle.receipt.state is not AgentWorkspaceStateV2.ACTIVE:
            raise CriteriaRubricSubgraphError("committed workspace lacks criteria/rubric result")
        _write_once(
            path,
            payload,
        )

    def _fault(
        self,
        point: CriteriaRubricRunnerFaultPoint,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point)


def _agent_outcome(
    outcome: CriteriaRubricOutcomeV2,
) -> AgentTaskOutcomeV2:
    if outcome is CriteriaRubricOutcomeV2.SUCCEEDED:
        return AgentTaskOutcomeV2.SUCCEEDED
    if outcome is CriteriaRubricOutcomeV2.ABSTAINED:
        return AgentTaskOutcomeV2.ABSTAINED
    return AgentTaskOutcomeV2.BLOCKED


def _sorted_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )


def _write_once(
    path: Path,
    payload: bytes,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise CriteriaRubricSubgraphError("criteria/rubric result journal raced") from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CriteriaRubricRunnerFaultInjector",
    "CriteriaRubricRunnerFaultPoint",
    "CriteriaRubricRunnerInjectedCrash",
    "CriteriaRubricSubgraphError",
    "CriteriaRubricSupervisedExecution",
    "StaticCriteriaRubricRunnerFaultInjector",
    "SupervisedCriteriaRubricRunner",
]
