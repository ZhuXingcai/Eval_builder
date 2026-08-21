from __future__ import annotations

from env_mock_agent.facade import (
    ExecutionTelemetryV2,
    FacadeObjectRef,
    TelemetryAvailabilityV2,
    execution_telemetry_ref,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactExecutionBatchV2,
    ArtifactExecutionReceiptOutcomeV2,
    ArtifactExecutionReceiptV2,
    artifact_execution_batch_ref,
    artifact_execution_group_ref,
    artifact_execution_plan_ref,
    artifact_execution_receipt_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    FailureClass,
    FailureRecord,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.observability_v2 import (
    ArtifactMetricSliceV2,
    CostObservationV2,
    ToolMetricV2,
)
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    R6_WORK_CONTROL_POLICY_VERSION,
    ArtifactGroupFanoutV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    WorkCancellationRecordV2,
    WorkControlPolicyV2,
    WorkLeaseEventKindV2,
    WorkLeaseEventV2,
    WorkLeaseV2,
    WorkReadinessSnapshotV2,
    WorkRetryDecisionV2,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
)
from eval_factory.orchestration.job_store import (
    JobStore,
    StaleWorkLeaseError,
    WorkControlPolicyError,
)
from eval_factory.orchestration.models import WorkLeaseHeadRecord


class WorkControlService:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store

    def compile_policy(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        lease_duration_seconds: int,
        heartbeat_extension_seconds: int,
        max_attempts: int,
        retry_delay_seconds: tuple[int, ...],
        retry_lease_expiry: bool,
        audit: ContractAudit,
    ) -> WorkControlPolicyV2:
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        return WorkControlPolicyV2.create(
            resolved_job_work_graph_ref=graph_ref,
            lease_duration_seconds=lease_duration_seconds,
            heartbeat_extension_seconds=heartbeat_extension_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            retry_lease_expiry=retry_lease_expiry,
            audit=_safe_audit(audit, (graph_ref,)),
        )

    def bind_policy(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        lease_duration_seconds: int,
        heartbeat_extension_seconds: int,
        max_attempts: int,
        retry_delay_seconds: tuple[int, ...],
        retry_lease_expiry: bool,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> WorkControlPolicyV2:
        policy = self.compile_policy(
            graph=graph,
            lease_duration_seconds=lease_duration_seconds,
            heartbeat_extension_seconds=heartbeat_extension_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            retry_lease_expiry=retry_lease_expiry,
            audit=audit,
        )
        return self.job_store.bind_work_control_policy(
            policy,
            idempotency_key=idempotency_key,
        )

    def acquire(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        readiness_snapshot: WorkReadinessSnapshotV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        retry_decision: WorkRetryDecisionV2 | None,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> WorkLeaseV2:
        return self.job_store.acquire_work_lease(
            graph=graph,
            work_unit=work_unit,
            readiness_snapshot=readiness_snapshot,
            policy=policy,
            holder_ref=holder_ref,
            retry_decision=retry_decision,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def heartbeat(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> WorkLeaseEventV2:
        return self.job_store.heartbeat_work_lease(
            lease=lease,
            policy=policy,
            holder_ref=holder_ref,
            expected_lease_version=expected_lease_version,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def complete_stage(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        status: StageRunStatus,
        output_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        checkpoint_ref: ObjectRef | None,
        metrics_ref: ObjectRef | None,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> WorkLeaseEventV2:
        return self.job_store.complete_stage_work_lease(
            lease=lease,
            policy=policy,
            holder_ref=holder_ref,
            expected_lease_version=expected_lease_version,
            status=status,
            output_refs=output_refs,
            failure=failure,
            checkpoint_ref=checkpoint_ref,
            metrics_ref=metrics_ref,
            audit=audit,
            idempotency_key=idempotency_key,
            telemetry=telemetry,
        )

    def complete_artifact_group(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        fanout: ArtifactGroupFanoutV2,
        artifact_execution_planning_result: object,
        artifact_execution_batch: ArtifactExecutionBatchV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> WorkLeaseEventV2:
        if lease.stage_run_ref is not None:
            raise WorkControlPolicyError("artifact-group completion cannot bind a StageRun")
        stored_fanout = self.job_store.get_artifact_group_fanout(
            fanout.parent_attachment_work_unit_ref.object_id
        )
        if stored_fanout != fanout:
            raise WorkControlPolicyError("artifact-group fanout is not current")
        unit = next(
            (
                candidate
                for candidate in fanout.group_work_units
                if resolved_work_unit_v2_ref(candidate) == lease.work_unit_ref
            ),
            None,
        )
        if unit is None or unit.artifact_execution_group_ref is None:
            raise WorkControlPolicyError("lease does not belong to the artifact fanout")

        from eval_factory.attachment_planning import (
            ArtifactExecutionPlanCompiler,
            ArtifactExecutionPlanningOutcome,
            ArtifactExecutionPlanningResult,
            ArtifactExecutionPolicyError,
            ArtifactGroupExecutor,
        )

        try:
            planning = ArtifactExecutionPlanningResult.model_validate(artifact_execution_planning_result)
            ArtifactExecutionPlanCompiler().validate_current(planning)
            if (
                planning.outcome is not ArtifactExecutionPlanningOutcome.PLANNED
                or planning.execution_plan is None
            ):
                raise WorkControlPolicyError("artifact-group completion requires a planned R5 result")
            plan = planning.execution_plan
            if artifact_execution_plan_ref(plan) != fanout.artifact_execution_plan_ref:
                raise WorkControlPolicyError("artifact execution plan does not match the fanout")
            ArtifactGroupExecutor().validate_current(
                plan,
                artifact_execution_batch,
            )
        except (ArtifactExecutionPolicyError, ValueError) as exc:
            raise WorkControlPolicyError("artifact-group completion evidence is not current") from exc

        group = next(
            (
                candidate
                for candidate in plan.groups
                if artifact_execution_group_ref(candidate) == unit.artifact_execution_group_ref
            ),
            None,
        )
        if group is None:
            raise WorkControlPolicyError("artifact-group completion crosses the leased group")
        group_ref = artifact_execution_group_ref(group)
        receipts = tuple(
            receipt
            for receipt in artifact_execution_batch.receipts
            if receipt.artifact_execution_group_ref == group_ref
        )
        if {receipt.artifact_id for receipt in receipts} != {
            candidate.artifact_id for candidate in group.units
        }:
            raise WorkControlPolicyError("artifact-group completion does not cover the exact group")
        if not receipts or max(receipt.attempt for receipt in receipts) != lease.attempt:
            raise WorkControlPolicyError("artifact-group completion attempt does not match the lease")

        result_refs = (
            artifact_execution_batch_ref(artifact_execution_batch),
            *(artifact_execution_receipt_ref(receipt) for receipt in receipts),
        )
        outcomes = {receipt.outcome for receipt in receipts}
        if outcomes == {ArtifactExecutionReceiptOutcomeV2.SUCCEEDED}:
            event_kind = WorkLeaseEventKindV2.SUCCEEDED
            failure = None
        elif ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE in outcomes:
            event_kind = WorkLeaseEventKindV2.RETRYABLE_FAILURE
            failure = _artifact_failure(retryable=True)
        else:
            event_kind = WorkLeaseEventKindV2.TERMINAL_FAILURE
            failure = _artifact_failure(retryable=False)
        return self.job_store.complete_artifact_group_work_lease(
            lease=lease,
            policy=policy,
            holder_ref=holder_ref,
            expected_lease_version=expected_lease_version,
            event_kind=event_kind,
            result_refs=result_refs,
            failure=failure,
            audit=audit,
            idempotency_key=idempotency_key,
            telemetry=telemetry,
            artifact_slices=_artifact_metric_slices(receipts),
        )

    def expire(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> WorkLeaseEventV2:
        return self.job_store.expire_work_lease(
            lease=lease,
            policy=policy,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def cancel_job(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        policy: WorkControlPolicyV2,
        reason_code: str,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> WorkCancellationRecordV2:
        return self.job_store.cancel_job_work(
            graph=graph,
            policy=policy,
            reason_code=reason_code,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def get_retry_decision(self, work_unit_id: str) -> WorkRetryDecisionV2:
        return self.job_store.get_work_retry_decision(work_unit_id)

    def get_retry_decision_optional(
        self,
        work_unit_id: str,
    ) -> WorkRetryDecisionV2 | None:
        return self.job_store.get_work_retry_decision_optional(work_unit_id)

    def get_lease_head(self, work_lease_id: str) -> WorkLeaseHeadRecord:
        return self.job_store.get_work_lease_head(work_lease_id)


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = tuple(binding for binding in audit.governing_versions if binding.component != "work-control")
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=(
            *versions,
            VersionBinding(
                component="work-control",
                version=R6_WORK_CONTROL_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(refs, key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _artifact_failure(*, retryable: bool) -> FailureRecord:
    return FailureRecord(
        failure_class=FailureClass.INTERNAL,
        code=("artifact-group-retryable-failure" if retryable else "artifact-group-terminal-failure"),
        message="Artifact group completed with a typed non-success outcome.",
        retryable=retryable,
    )


def _artifact_metric_slices(
    receipts: tuple[ArtifactExecutionReceiptV2, ...],
) -> tuple[ArtifactMetricSliceV2, ...]:
    slices: list[ArtifactMetricSliceV2] = []
    for receipt in receipts:
        result = receipt.facade_result
        if result is None:
            slices.append(
                ArtifactMetricSliceV2(
                    artifact_id=receipt.artifact_id,
                    status=receipt.outcome.value,
                    cost=CostObservationV2.unavailable(),
                )
            )
            continue
        telemetry = result.telemetry
        telemetry_ref = _object_ref_from_facade(execution_telemetry_ref(telemetry))
        if telemetry.reported_cost.availability is TelemetryAvailabilityV2.REPORTED:
            amount = telemetry.reported_cost.amount_microusd
            if amount is None:
                raise WorkControlPolicyError("reported artifact cost is missing its amount")
            cost = CostObservationV2.reported(
                amount_microusd=amount,
                source_ref=telemetry_ref,
            )
        else:
            cost = CostObservationV2.unavailable()
        request = receipt.facade_request
        slices.append(
            ArtifactMetricSliceV2(
                artifact_id=receipt.artifact_id,
                status=receipt.outcome.value,
                telemetry_ref=telemetry_ref,
                selected_route_kind=result.selected_route_kind.value,
                selected_route_id=result.selected_route_id,
                selected_route_version=(request.selected_route_version if request is not None else None),
                worker_version=result.worker_version,
                output_ref=(
                    _object_ref_from_facade(result.output_ref) if result.output_ref is not None else None
                ),
                tool_metrics=tuple(
                    ToolMetricV2(
                        tool_family=value.tool_family.value,
                        calls=value.calls,
                    )
                    for value in telemetry.tool_family_counts
                ),
                cost=cost,
            )
        )
    return tuple(sorted(slices, key=lambda value: value.artifact_id))


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


__all__ = [
    "R6_WORK_CONTROL_POLICY_VERSION",
    "StaleWorkLeaseError",
    "WorkControlPolicyError",
    "WorkControlService",
]
