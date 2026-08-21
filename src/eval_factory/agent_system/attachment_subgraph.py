from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Protocol

from env_mock_agent.facade import (
    AttachmentExecutionFacade,
    FacadeObjectRef,
)
from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationAuthority,
    AttachmentR5PreparationError,
    AttachmentR5PreparationValidator,
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
from eval_factory.agent_system.workspace import (
    AgentWorkspaceHandle,
)
from eval_factory.attachment_planning import (
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
    ArtifactExecutionPolicyError,
    ArtifactGroupExecutor,
    ArtifactResultCompiler,
    ArtifactRoutingCompilationResult,
    ArtifactRoutingRequest,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AgentTaskV2,
    AgentWorkspaceStateV2,
    AttachmentGenerationPlanV2,
    AttachmentGroupResultV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
    CompiledAttachmentGenerationPlanV2,
    PlanKindV2,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactExecutionBatchV2,
    ArtifactExecutionGroupV2,
    ArtifactExecutionReceiptOutcomeV2,
    ArtifactExecutionReceiptV2,
    AttachmentReconstructionResultV2,
    artifact_execution_batch_ref,
    artifact_execution_group_ref,
    artifact_execution_receipt_ref,
    attachment_reconstruction_result_v2_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)


class AttachmentSubgraphError(RuntimeError):
    pass


class AttachmentRunnerInjectedCrash(RuntimeError):
    pass


class AttachmentRunnerFaultPoint(StrEnum):
    AFTER_BATCH_JOURNAL = "AFTER_BATCH_JOURNAL"
    AFTER_FIRST_WORKSPACE_COMMIT = "AFTER_FIRST_WORKSPACE_COMMIT"
    AFTER_FIRST_RESULT = "AFTER_FIRST_RESULT"
    BEFORE_JOIN = "BEFORE_JOIN"
    AFTER_DOMAIN_RESULT = "AFTER_DOMAIN_RESULT"


class AttachmentRunnerFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: AttachmentRunnerFaultPoint,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticAttachmentRunnerFaultInjector:
    crash_points: frozenset[AttachmentRunnerFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: AttachmentRunnerFaultPoint,
    ) -> None:
        if point in self.crash_points:
            raise AttachmentRunnerInjectedCrash(f"injected attachment runner crash at {point.value}")


@dataclass(frozen=True, slots=True)
class AttachmentR5Execution:
    execution_batch: ArtifactExecutionBatchV2
    reconstruction_result: AttachmentReconstructionResultV2


@dataclass(frozen=True, slots=True)
class AttachmentSupervisedExecution:
    r5_execution: AttachmentR5Execution
    group_results: tuple[AttachmentGroupResultV2, ...]
    envelopes: tuple[AgentResultEnvelopeV2, ...]
    subgraph_result: AttachmentSubgraphResultV2


class AttachmentR5SubgraphAdapter:
    def __init__(self) -> None:
        self._executor = ArtifactGroupExecutor()
        self._result_compiler = ArtifactResultCompiler()

    async def execute(
        self,
        *,
        plan: AttachmentGenerationPlanV2,
        routing_request: ArtifactRoutingRequest,
        routing_result: ArtifactRoutingCompilationResult,
        execution_result: ArtifactExecutionPlanningResult,
        facade: AttachmentExecutionFacade,
        audit: ContractAudit,
        prior_batch: ArtifactExecutionBatchV2 | None = None,
    ) -> AttachmentR5Execution:
        if (
            execution_result.outcome is not ArtifactExecutionPlanningOutcome.PLANNED
            or execution_result.execution_plan is None
        ):
            raise AttachmentSubgraphError("attachment subgraph requires a current R5 execution plan")
        execution_plan = execution_result.execution_plan
        self._validate_plan_mapping(plan, execution_plan.groups)
        execution_audit = _stage_audit(
            audit,
            component="artifact-execution",
            version="r5-06",
        )
        result_audit = _stage_audit(
            audit,
            component="artifact-results",
            version="r5-07",
        )
        try:
            if (
                prior_batch is not None
                and not prior_batch.retryable_artifact_ids
                and not prior_batch.dependency_blocked_artifact_ids
            ):
                batch = prior_batch
            else:
                batch = await self._executor.run(
                    execution_plan,
                    facade=facade,
                    audit=execution_audit,
                    prior_batch=prior_batch,
                )
            self._executor.validate_current(execution_plan, batch)
            reconstruction = self._result_compiler.compile(
                routing_request=routing_request,
                routing_result=routing_result,
                execution_result=execution_result,
                execution_batch=batch,
                audit=result_audit,
            )
        except ArtifactExecutionPolicyError as exc:
            raise AttachmentSubgraphError("R5 attachment execution authority is invalid") from exc
        return AttachmentR5Execution(
            execution_batch=batch,
            reconstruction_result=reconstruction,
        )

    def join(
        self,
        *,
        plan: AttachmentGenerationPlanV2,
        tasks: tuple[AgentTaskV2, ...],
        envelopes: tuple[AgentResultEnvelopeV2, ...],
        audit: ContractAudit,
    ) -> AttachmentSubgraphResultV2:
        tasks_by_ref = {task.to_ref(): task for task in tasks}
        if len(tasks_by_ref) != len(plan.works):
            raise AttachmentSubgraphError("attachment Agent tasks do not exactly cover plan work")
        if {task.plan_task_key for task in tasks} != {work.work_key for work in plan.works}:
            raise AttachmentSubgraphError("attachment Agent task keys differ from plan work")
        by_work: dict[str, AgentResultEnvelopeV2] = {}
        for envelope in envelopes:
            task = tasks_by_ref.get(envelope.task_ref)
            if task is None:
                raise AttachmentSubgraphError("attachment result belongs to another task")
            if task.plan_task_key in by_work:
                raise AttachmentSubgraphError("attachment work has duplicate results")
            by_work[task.plan_task_key] = envelope
        if set(by_work) != {work.work_key for work in plan.works}:
            raise AttachmentSubgraphError("attachment result inventory is incomplete")

        succeeded = self._refs_for_outcomes(
            by_work,
            {AgentTaskOutcomeV2.SUCCEEDED},
        )
        retryable = self._refs_for_outcomes(
            by_work,
            {AgentTaskOutcomeV2.RETRYABLE_FAILURE},
        )
        blocked = self._refs_for_outcomes(
            by_work,
            {
                AgentTaskOutcomeV2.ABSTAINED,
                AgentTaskOutcomeV2.BLOCKED,
                AgentTaskOutcomeV2.FAILED,
            },
        )
        cancelled: tuple[ObjectRef, ...] = ()
        all_refs = _sorted_refs(tuple(envelope.to_ref() for envelope in envelopes))
        if len(succeeded) == len(all_refs):
            outcome = AttachmentSubgraphOutcomeV2.SUCCEEDED
            reasons: tuple[str, ...] = ()
        elif not succeeded:
            outcome = AttachmentSubgraphOutcomeV2.BLOCKED
            reasons = ("ATTACHMENT_WORK_BLOCKED",)
        else:
            outcome = AttachmentSubgraphOutcomeV2.PARTIAL
            reasons = ("ATTACHMENT_WORK_INCOMPLETE",)
        return AttachmentSubgraphResultV2.create(
            result_id=(f"attachment-subgraph-result://{plan.object_sha256}"),
            plan_ref=plan.to_ref(),
            work_result_refs=all_refs,
            succeeded_work_result_refs=succeeded,
            retryable_work_result_refs=retryable,
            blocked_work_result_refs=blocked,
            cancelled_work_result_refs=cancelled,
            outcome=outcome,
            reason_codes=reasons,
            audit=audit,
        )

    @staticmethod
    def _validate_plan_mapping(
        plan: AttachmentGenerationPlanV2,
        groups: tuple[ArtifactExecutionGroupV2, ...],
    ) -> None:
        planned = {work.artifact_group_ref: set(work.artifact_ids) for work in plan.works}
        observed = {
            _object_ref(artifact_execution_group_ref(group)): {unit.artifact_id for unit in group.units}
            for group in groups
        }
        if planned != observed:
            raise AttachmentSubgraphError("attachment plan work differs from current R5 groups")

    @staticmethod
    def _refs_for_outcomes(
        by_work: dict[str, AgentResultEnvelopeV2],
        outcomes: set[AgentTaskOutcomeV2],
    ) -> tuple[ObjectRef, ...]:
        return _sorted_refs(
            tuple(envelope.to_ref() for envelope in by_work.values() if envelope.outcome in outcomes)
        )


class SupervisedAttachmentR5Runner:
    _BATCH_JOURNAL = "artifact-execution-batch.json"
    _GROUP_RESULT = "attachment-group-result.json"

    def __init__(
        self,
        *,
        supervisor: ExecutionSupervisor,
        adapter: AttachmentR5SubgraphAdapter | None = None,
        preparation_validator: AttachmentR5PreparationValidator | None = None,
        review_service: PlanReviewService | None = None,
        fault_injector: (AttachmentRunnerFaultInjector | None) = None,
    ) -> None:
        self.supervisor = supervisor
        self.adapter = adapter or AttachmentR5SubgraphAdapter()
        self.preparation_validator = preparation_validator or AttachmentR5PreparationValidator()
        self.review_service = review_service or PlanReviewService(
            supervisor.store,
        )
        self.fault_injector = fault_injector

    async def run(
        self,
        *,
        run_id: str,
        plan: AttachmentGenerationPlanV2,
        compiled_plan: CompiledAttachmentGenerationPlanV2,
        preparation: AttachmentR5PreparationAuthority,
        facade: AttachmentExecutionFacade,
        audit: ContractAudit,
        lease_duration: timedelta = timedelta(minutes=10),
        prior_batch: ArtifactExecutionBatchV2 | None = None,
    ) -> AttachmentSupervisedExecution:
        try:
            self.review_service.require_resumed_plan(
                run_id=run_id,
                plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
                plan_ref=plan.to_ref(),
            )
        except PlanReviewError as exc:
            raise AttachmentSubgraphError(
                "attachment execution requires resumed plan authority",
            ) from exc
        try:
            self.preparation_validator.validate_current(preparation)
        except AttachmentR5PreparationError as exc:
            raise AttachmentSubgraphError("attachment R5 preparation authority is invalid") from exc
        self._validate_supervised_plan(
            plan,
            compiled_plan,
        )
        tasks = self.supervisor.materialize_attachment(
            run_id=run_id,
            compiled_plan=compiled_plan,
            audit=audit,
        )
        leases = tuple(
            self.supervisor.acquire(
                task.agent_task_id,
                AgentWorkerRef(worker_id=(f"worker://attachment/{task.plan_task_key}")),
                lease_duration=lease_duration,
                idempotency_key=(f"acquire-attachment-{compiled_plan.object_sha256}-{task.plan_task_key}"),
            )
            for task in tasks
        )
        handles = tuple(
            self.supervisor.workspace_manager.recover(
                run_ref=task.run_ref,
                lease=lease,
                audit=audit,
            )
            for task, lease in zip(
                tasks,
                leases,
                strict=True,
            )
        )
        journal_batch = self._load_batch_journal(handles)
        if prior_batch is not None and journal_batch is not None and prior_batch != journal_batch:
            raise AttachmentSubgraphError("workspace R5 batch journal conflicts with replay input")
        r5_execution = await self.adapter.execute(
            plan=plan,
            routing_request=preparation.routing_request,
            routing_result=preparation.routing_result,
            execution_result=preparation.execution_result,
            facade=facade,
            audit=audit,
            prior_batch=journal_batch or prior_batch,
        )
        self._persist_batch_journal(
            handles,
            r5_execution.execution_batch,
        )
        self._fault(AttachmentRunnerFaultPoint.AFTER_BATCH_JOURNAL)
        group_results, envelopes = self._complete_group_results(
            plan=plan,
            tasks=tasks,
            leases=leases,
            handles=handles,
            execution=r5_execution,
            audit=audit,
        )
        self._fault(AttachmentRunnerFaultPoint.BEFORE_JOIN)
        subgraph_result = self.adapter.join(
            plan=plan,
            tasks=tasks,
            envelopes=envelopes,
            audit=audit,
        )
        self.supervisor.store.commit_domain_result(
            run_id=run_id,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            result=subgraph_result,
            idempotency_key=(f"commit-attachment-subgraph-result-{subgraph_result.object_sha256}"),
        )
        self._fault(AttachmentRunnerFaultPoint.AFTER_DOMAIN_RESULT)
        return AttachmentSupervisedExecution(
            r5_execution=r5_execution,
            group_results=group_results,
            envelopes=envelopes,
            subgraph_result=subgraph_result,
        )

    def _fault(
        self,
        point: AttachmentRunnerFaultPoint,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point)

    @classmethod
    def _load_batch_journal(
        cls,
        handles: tuple[AgentWorkspaceHandle, ...],
    ) -> ArtifactExecutionBatchV2 | None:
        paths = tuple(handle.path / cls._BATCH_JOURNAL for handle in handles)
        existing = tuple(path.is_file() for path in paths)
        if not any(existing):
            return None
        if not all(existing) or any(path.is_symlink() for path in paths):
            raise AttachmentSubgraphError("workspace R5 batch journal is incomplete or unsafe")
        values = tuple(ArtifactExecutionBatchV2.model_validate_json(path.read_bytes()) for path in paths)
        if any(
            value.canonical_json() != path.read_bytes()
            for value, path in zip(
                values,
                paths,
                strict=True,
            )
        ) or any(value != values[0] for value in values[1:]):
            raise AttachmentSubgraphError("workspace R5 batch journal drifted")
        return values[0]

    @classmethod
    def _persist_batch_journal(
        cls,
        handles: tuple[AgentWorkspaceHandle, ...],
        batch: ArtifactExecutionBatchV2,
    ) -> None:
        payload = batch.canonical_json()
        for handle in handles:
            path = handle.path / cls._BATCH_JOURNAL
            if path.exists():
                if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                    raise AttachmentSubgraphError("workspace R5 batch journal conflicts")
                continue
            if handle.receipt.state is not AgentWorkspaceStateV2.ACTIVE:
                raise AttachmentSubgraphError("committed workspace lacks R5 batch journal")
            path.write_bytes(payload)

    @staticmethod
    def _validate_supervised_plan(
        plan: AttachmentGenerationPlanV2,
        compiled_plan: CompiledAttachmentGenerationPlanV2,
    ) -> None:
        if compiled_plan.source_plan_ref != plan.to_ref():
            raise AttachmentSubgraphError("compiled attachment plan does not bind its source")
        if any(work.dependency_work_keys for work in plan.works):
            raise AttachmentSubgraphError("R5 artifact groups must remain independent Agent work")
        if any(
            work.max_model_requests or work.max_model_tokens or work.max_cost_micro_usd for work in plan.works
        ):
            raise AttachmentSubgraphError("R5 group work cannot claim direct AI Gateway budget")

    def _complete_group_results(
        self,
        *,
        plan: AttachmentGenerationPlanV2,
        tasks: tuple[AgentTaskV2, ...],
        leases: tuple[AgentLeaseV2, ...],
        handles: tuple[AgentWorkspaceHandle, ...],
        execution: AttachmentR5Execution,
        audit: ContractAudit,
    ) -> tuple[
        tuple[AttachmentGroupResultV2, ...],
        tuple[AgentResultEnvelopeV2, ...],
    ]:
        task_by_key = {task.plan_task_key: task for task in tasks}
        lease_by_task = {lease.agent_task_ref: lease for lease in leases}
        handle_by_task = {
            task.to_ref(): handle
            for task, handle in zip(
                tasks,
                handles,
                strict=True,
            )
        }
        receipts_by_group = {
            work.artifact_group_ref: tuple(
                receipt
                for receipt in execution.execution_batch.receipts
                if receipt.artifact_execution_group_ref == work.artifact_group_ref
            )
            for work in plan.works
        }
        group_results: list[AttachmentGroupResultV2] = []
        envelopes: list[AgentResultEnvelopeV2] = []
        for index, work in enumerate(plan.works):
            task = task_by_key[work.work_key]
            lease = lease_by_task[task.to_ref()]
            handle = handle_by_task[task.to_ref()]
            receipts = receipts_by_group[work.artifact_group_ref]
            if {receipt.artifact_id for receipt in receipts} != set(work.artifact_ids):
                raise AttachmentSubgraphError("R5 group receipt inventory differs from attachment work")
            outcome, reasons = _group_outcome(receipts)
            receipt_refs = tuple(artifact_execution_receipt_ref(receipt) for receipt in receipts)
            group_result = AttachmentGroupResultV2.create(
                result_id=(
                    "attachment-group-result://"
                    f"{work.work_key}/"
                    f"{execution.execution_batch.artifact_execution_batch_sha256}"
                ),
                plan_ref=plan.to_ref(),
                work_key=work.work_key,
                artifact_group_ref=work.artifact_group_ref,
                execution_batch_ref=artifact_execution_batch_ref(execution.execution_batch),
                reconstruction_result_ref=(
                    attachment_reconstruction_result_v2_ref(execution.reconstruction_result)
                ),
                execution_receipt_refs=receipt_refs,
                artifact_ids=work.artifact_ids,
                outcome=outcome,
                reason_codes=reasons,
                audit=audit,
            )
            group_result_path = handle.path / self._GROUP_RESULT
            group_payload = group_result.canonical_json()
            if group_result_path.exists():
                if (
                    group_result_path.is_symlink()
                    or not group_result_path.is_file()
                    or group_result_path.read_bytes() != group_payload
                ):
                    raise AttachmentSubgraphError("workspace attachment group result conflicts")
            elif handle.receipt.state is AgentWorkspaceStateV2.COMMITTED:
                raise AttachmentSubgraphError("committed workspace lacks attachment group result")
            else:
                group_result_path.write_bytes(group_payload)
            workspace_receipt = self.supervisor.workspace_manager.commit(
                handle,
                audit=audit,
            )
            if index == 0:
                self._fault(AttachmentRunnerFaultPoint.AFTER_FIRST_WORKSPACE_COMMIT)
            envelope = AgentResultEnvelopeV2.create(
                result_id=(
                    f"agent-result-envelope://{task.agent_task_id.rsplit('://', 1)[-1]}/{lease.attempt}"
                ),
                task_ref=task.to_ref(),
                agent_definition_ref=(lease.agent_definition_ref),
                attempt=lease.attempt,
                lease_id=lease.lease_id,
                fencing_token=lease.fencing_token,
                workspace_receipt_ref=(workspace_receipt.to_ref()),
                outcome=outcome,
                output_refs=(group_result.to_ref(),),
                delegated_dataset_job_result_refs=(),
                gateway_receipt_refs=(),
                validator_result_refs=group_result.execution_receipt_refs,
                failure_code=(None if outcome is AgentTaskOutcomeV2.SUCCEEDED else reasons[0]),
                safe_metrics=(),
                audit=audit,
            )
            envelopes.append(
                self.supervisor.complete(
                    lease=lease,
                    envelope=envelope,
                )
            )
            if index == 0:
                self._fault(AttachmentRunnerFaultPoint.AFTER_FIRST_RESULT)
            group_results.append(group_result)
        return (
            tuple(group_results),
            tuple(envelopes),
        )


def _group_outcome(
    receipts: tuple[ArtifactExecutionReceiptV2, ...],
) -> tuple[AgentTaskOutcomeV2, tuple[str, ...]]:
    outcomes = {receipt.outcome for receipt in receipts}
    if outcomes == {ArtifactExecutionReceiptOutcomeV2.SUCCEEDED}:
        return AgentTaskOutcomeV2.SUCCEEDED, ()
    if ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE in outcomes:
        return (
            AgentTaskOutcomeV2.RETRYABLE_FAILURE,
            ("ATTACHMENT_GROUP_RETRYABLE",),
        )
    return (
        AgentTaskOutcomeV2.BLOCKED,
        ("ATTACHMENT_GROUP_BLOCKED",),
    )


def _object_ref(
    value: ObjectRef | FacadeObjectRef,
) -> ObjectRef:
    return ObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _stage_audit(
    audit: ContractAudit,
    *,
    component: str,
    version: str,
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=(
            *tuple(binding for binding in audit.governing_versions if binding.component != component),
            VersionBinding(
                component=component,
                version=version,
            ),
        ),
        input_refs=audit.input_refs,
    )


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


__all__ = [
    "AttachmentR5Execution",
    "AttachmentR5SubgraphAdapter",
    "AttachmentRunnerFaultInjector",
    "AttachmentRunnerFaultPoint",
    "AttachmentRunnerInjectedCrash",
    "AttachmentSubgraphError",
    "AttachmentSupervisedExecution",
    "StaticAttachmentRunnerFaultInjector",
    "SupervisedAttachmentR5Runner",
]
