from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from eval_factory.agent_system.grading_agent import (
    GatewayGradingDesignAgent,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewError,
    PlanReviewService,
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
    CompiledGradingDesignPlanV2,
    CriteriaRubricResultV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    GradingDesignResultV2,
    PlanKindV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    ModelRouteDecisionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.task_v2 import (
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricSetV2,
    ToolPolicyV2,
)
from eval_factory.task_authoring import (
    VerifiedModelDomainAuthorization,
)


class GradingDesignSubgraphError(RuntimeError):
    pass


class GradingDesignRunnerInjectedCrash(RuntimeError):
    pass


class GradingDesignRunnerFaultPoint(StrEnum):
    AFTER_RESULT_JOURNAL = "AFTER_RESULT_JOURNAL"
    AFTER_WORKSPACE_COMMIT = "AFTER_WORKSPACE_COMMIT"
    AFTER_AGENT_RESULT = "AFTER_AGENT_RESULT"
    AFTER_DOMAIN_RESULT = "AFTER_DOMAIN_RESULT"


class GradingDesignRunnerFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: GradingDesignRunnerFaultPoint,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticGradingDesignRunnerFaultInjector:
    crash_points: frozenset[GradingDesignRunnerFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: GradingDesignRunnerFaultPoint,
    ) -> None:
        if point in self.crash_points:
            raise GradingDesignRunnerInjectedCrash(f"injected grading design runner crash at {point.value}")


@dataclass(frozen=True, slots=True)
class GradingDesignSupervisedExecution:
    result: GradingDesignResultV2
    envelope: AgentResultEnvelopeV2


class SupervisedGradingDesignRunner:
    _RESULT_JOURNAL = "grading-design-result.json"

    def __init__(
        self,
        *,
        supervisor: ExecutionSupervisor,
        agent: GatewayGradingDesignAgent,
        review_service: PlanReviewService | None = None,
        fault_injector: (GradingDesignRunnerFaultInjector | None) = None,
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
        plan: GradingDesignPlanV2,
        compiled_plan: CompiledGradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        model_authorizations: tuple[VerifiedModelDomainAuthorization, ...],
        evaluated_at: datetime,
        audit: ContractAudit,
        lease_duration: timedelta = timedelta(minutes=10),
    ) -> GradingDesignSupervisedExecution:
        try:
            self.review_service.require_resumed_plan(
                run_id=run_id,
                plan_kind=PlanKindV2.GRADING_DESIGN,
                plan_ref=plan.to_ref(),
            )
        except PlanReviewError as exc:
            raise GradingDesignSubgraphError(
                "grading design execution requires resumed plan authority",
            ) from exc
        self._validate_preflight(
            plan=plan,
            compiled_plan=compiled_plan,
            criteria_result=criteria_result,
            criteria_route=criteria_route,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
        )
        task = self.supervisor.materialize_grading_design(
            run_id=run_id,
            compiled_plan=compiled_plan,
            audit=audit,
        )
        prior_envelope = self.supervisor.find_result(task.agent_task_id)
        if prior_envelope is None:
            result, envelope = await self._execute_attempt(
                task=task,
                plan=plan,
                criteria_result=criteria_result,
                criteria_route=criteria_route,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
                model_authorizations=model_authorizations,
                evaluated_at=evaluated_at,
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
            plan_kind=PlanKindV2.GRADING_DESIGN,
            result=result,
            idempotency_key=(f"commit-grading-design-result-{result.object_sha256}"),
        )
        self._fault(GradingDesignRunnerFaultPoint.AFTER_DOMAIN_RESULT)
        return GradingDesignSupervisedExecution(
            result=result,
            envelope=envelope,
        )

    def _validate_preflight(
        self,
        *,
        plan: GradingDesignPlanV2,
        compiled_plan: CompiledGradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
    ) -> None:
        if (
            compiled_plan.source_plan_ref != plan.to_ref()
            or compiled_plan.criteria_rubric_result_ref != plan.criteria_rubric_result_ref
            or compiled_plan.rubric_set_ref != plan.rubric_set_ref
            or compiled_plan.evaluator_spec_ref != plan.evaluator_spec_ref
            or compiled_plan.reference_policy_ref != plan.reference_policy_ref
            or compiled_plan.tool_policy_ref != plan.tool_policy_ref
            or compiled_plan.generator_model_profile_ref != plan.generator_model_profile_ref
            or compiled_plan.judge_prompt_template_ref != plan.judge_prompt_template_ref
            or compiled_plan.model_policy_ref != plan.model_policy_ref
            or compiled_plan.acceptance_check_refs != plan.acceptance_check_refs
        ):
            raise GradingDesignSubgraphError("compiled grading design plan is stale")
        self.agent.validate_sources(
            plan=plan,
            criteria_result=criteria_result,
            criteria_route=criteria_route,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
        )

    async def _execute_attempt(
        self,
        *,
        task: AgentTaskV2,
        plan: GradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        model_authorizations: tuple[VerifiedModelDomainAuthorization, ...],
        evaluated_at: datetime,
        audit: ContractAudit,
        lease_duration: timedelta,
    ) -> tuple[
        GradingDesignResultV2,
        AgentResultEnvelopeV2,
    ]:
        worker = AgentWorkerRef(worker_id="worker://grading-design")
        lease = self.supervisor.acquire(
            task.agent_task_id,
            worker,
            lease_duration=lease_duration,
            idempotency_key=(f"acquire-grading-design-{task.compiled_plan_ref.object_sha256}"),
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
                criteria_result=criteria_result,
                criteria_route=criteria_route,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
                model_authorizations=model_authorizations,
                evaluated_at=evaluated_at,
                audit=audit,
            )
            result = execution.result
            self._persist_result_journal(
                handle,
                result,
            )
        self._fault(GradingDesignRunnerFaultPoint.AFTER_RESULT_JOURNAL)
        workspace_receipt = self.supervisor.workspace_manager.commit(
            handle,
            audit=audit,
        )
        self._fault(GradingDesignRunnerFaultPoint.AFTER_WORKSPACE_COMMIT)
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
        self._fault(GradingDesignRunnerFaultPoint.AFTER_AGENT_RESULT)
        return result, committed

    def _recover_completed_attempt(
        self,
        *,
        task: AgentTaskV2,
        plan: GradingDesignPlanV2,
        prior_envelope: AgentResultEnvelopeV2,
        audit: ContractAudit,
    ) -> tuple[
        GradingDesignResultV2,
        AgentResultEnvelopeV2,
    ]:
        lease = self.supervisor.get_lease(prior_envelope.lease_id)
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
            raise GradingDesignSubgraphError("completed grading design workspace lacks its result journal")
        expected = self._build_envelope(
            task=task,
            lease=lease,
            workspace_receipt=handle.receipt,
            plan=plan,
            result=result,
            audit=prior_envelope.audit,
        )
        if expected != prior_envelope:
            raise GradingDesignSubgraphError("grading design Agent result differs from workspace journal")
        return result, prior_envelope

    @staticmethod
    def _build_envelope(
        *,
        task: AgentTaskV2,
        lease: AgentLeaseV2,
        workspace_receipt: AgentWorkspaceReceiptV2,
        plan: GradingDesignPlanV2,
        result: GradingDesignResultV2,
        audit: ContractAudit,
    ) -> AgentResultEnvelopeV2:
        if result.plan_ref != plan.to_ref():
            raise GradingDesignSubgraphError("grading design result binds another plan")
        outcome = _agent_outcome(result.outcome)
        gateway_receipt_refs = (result.gateway_receipt_ref,) if result.gateway_receipt_ref is not None else ()
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
            gateway_receipt_refs=gateway_receipt_refs,
            validator_result_refs=_sorted_refs(
                (
                    result.validation_ref,
                    *plan.acceptance_check_refs,
                )
            ),
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
    ) -> GradingDesignResultV2 | None:
        path = handle.path / cls._RESULT_JOURNAL
        if not path.exists():
            if handle.receipt.state is AgentWorkspaceStateV2.COMMITTED:
                raise GradingDesignSubgraphError(
                    "committed grading design workspace lacks its result journal"
                )
            return None
        if path.is_symlink() or not path.is_file():
            raise GradingDesignSubgraphError("grading design result journal is unsafe")
        payload = path.read_bytes()
        try:
            result = GradingDesignResultV2.model_validate_json(payload)
        except ValidationError as exc:
            raise GradingDesignSubgraphError("grading design result journal is invalid") from exc
        if result.canonical_json() != payload or result.plan_ref != plan_ref:
            raise GradingDesignSubgraphError("grading design result journal drifted")
        return result

    @classmethod
    def _persist_result_journal(
        cls,
        handle: AgentWorkspaceHandle,
        result: GradingDesignResultV2,
    ) -> None:
        path = handle.path / cls._RESULT_JOURNAL
        payload = result.canonical_json()
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise GradingDesignSubgraphError("grading design result journal conflicts")
            return
        if handle.receipt.state is not AgentWorkspaceStateV2.ACTIVE:
            raise GradingDesignSubgraphError("committed workspace lacks grading design result")
        _write_once(path, payload)

    def _fault(
        self,
        point: GradingDesignRunnerFaultPoint,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point)


def _agent_outcome(
    outcome: GradingDesignOutcomeV2,
) -> AgentTaskOutcomeV2:
    if outcome is GradingDesignOutcomeV2.SUCCEEDED:
        return AgentTaskOutcomeV2.SUCCEEDED
    if outcome is GradingDesignOutcomeV2.ABSTAINED:
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
                raise GradingDesignSubgraphError("grading design result journal raced") from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "GradingDesignRunnerFaultInjector",
    "GradingDesignRunnerFaultPoint",
    "GradingDesignRunnerInjectedCrash",
    "GradingDesignSubgraphError",
    "GradingDesignSupervisedExecution",
    "StaticGradingDesignRunnerFaultInjector",
    "SupervisedGradingDesignRunner",
]
