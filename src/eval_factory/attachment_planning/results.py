from __future__ import annotations

from typing import Literal

from env_mock_agent.facade import (
    FacadeObjectRef,
    attachment_execution_result_ref,
    attachment_route_decision_carried_sha256,
)
from eval_factory.attachment_planning.execution import (
    ArtifactExecutionPlanCompiler,
    ArtifactGroupExecutor,
)
from eval_factory.attachment_planning.execution_models import (
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
    ArtifactExecutionPolicyError,
)
from eval_factory.attachment_planning.routing_models import (
    ArtifactRoutingCompilationResult,
    ArtifactRoutingRequest,
    artifact_routing_compilation_result_carried_sha256,
    artifact_routing_compilation_result_ref,
    artifact_routing_request_carried_sha256,
    artifact_routing_request_ref,
)
from eval_factory.contracts.attachment import (
    ArtifactBuildResult,
    ArtifactBuildStatus,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactBuildResultOutcomeV2,
    ArtifactBuildResultV2,
    ArtifactBuildSpecV2,
    ArtifactExecutionBatchV2,
    ArtifactExecutionPlanV2,
    ArtifactExecutionReceiptV2,
    ArtifactRouteEntryOutcomeV2,
    ArtifactRouteKindV2,
    ArtifactRoutePlanEntryV2,
    AttachmentReconstructionResultV2,
    artifact_build_result_projection_gap_v2,
    artifact_build_result_v1_carried_sha256,
    artifact_build_result_v2_carried_sha256,
    artifact_build_result_v2_ref,
    artifact_build_spec_v2_carried_sha256,
    artifact_build_spec_v2_ref,
    artifact_execution_batch_ref,
    artifact_execution_failure_record_v2,
    artifact_execution_plan_ref,
    artifact_execution_receipt_ref,
    artifact_routing_plan_carried_sha256,
    artifact_routing_plan_ref,
    attachment_reconstruction_outcome_v2,
    attachment_reconstruction_result_v2_carried_sha256,
    validate_artifact_build_result_v2_identity,
    validate_attachment_reconstruction_result_v2_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
)
from eval_factory.contracts.task import AttachmentCriticality

ARTIFACT_RESULT_POLICY_VERSION: Literal["artifact-results/r5-07-v1"] = "artifact-results/r5-07-v1"


class ArtifactResultPolicyError(RuntimeError):
    pass


class ArtifactResultCompiler:
    policy_version = ARTIFACT_RESULT_POLICY_VERSION

    def compile(
        self,
        *,
        routing_request: ArtifactRoutingRequest,
        routing_result: ArtifactRoutingCompilationResult,
        execution_result: ArtifactExecutionPlanningResult,
        execution_batch: ArtifactExecutionBatchV2 | None,
        audit: ContractAudit,
    ) -> AttachmentReconstructionResultV2:
        try:
            return self._compile(
                routing_request=routing_request,
                routing_result=routing_result,
                execution_result=execution_result,
                execution_batch=execution_batch,
                audit=audit,
            )
        except ArtifactResultPolicyError:
            raise
        except (ArtifactExecutionPolicyError, ValueError) as exc:
            raise ArtifactResultPolicyError(f"artifact result compilation failed: {exc}") from exc

    def validate_current(
        self,
        result: AttachmentReconstructionResultV2,
        *,
        routing_request: ArtifactRoutingRequest,
        routing_result: ArtifactRoutingCompilationResult,
        execution_result: ArtifactExecutionPlanningResult,
        execution_batch: ArtifactExecutionBatchV2 | None,
    ) -> None:
        try:
            validated = AttachmentReconstructionResultV2.model_validate(result.model_dump(mode="python"))
            validate_attachment_reconstruction_result_v2_identity(validated)
            for artifact_result in validated.artifact_results:
                validate_artifact_build_result_v2_identity(artifact_result)
            current = self._compile(
                routing_request=routing_request,
                routing_result=routing_result,
                execution_result=execution_result,
                execution_batch=execution_batch,
                audit=result.audit,
            )
        except ArtifactResultPolicyError:
            raise
        except (ArtifactExecutionPolicyError, ValueError) as exc:
            raise ArtifactResultPolicyError(f"current artifact result validation failed: {exc}") from exc
        if (
            current.attachment_reconstruction_result_v2_sha256
            != result.attachment_reconstruction_result_v2_sha256
            or current.artifact_result_refs != result.artifact_result_refs
        ):
            raise ArtifactResultPolicyError("artifact reconstruction result is stale for current sources")

    def _compile(
        self,
        *,
        routing_request: ArtifactRoutingRequest,
        routing_result: ArtifactRoutingCompilationResult,
        execution_result: ArtifactExecutionPlanningResult,
        execution_batch: ArtifactExecutionBatchV2 | None,
        audit: ContractAudit,
    ) -> AttachmentReconstructionResultV2:
        _validate_routing_source(routing_request, routing_result)
        execution_plan = _validate_execution_source(
            routing_result=routing_result,
            execution_result=execution_result,
            execution_batch=execution_batch,
        )
        routing_plan = routing_result.routing_plan
        routing_plan_ref = artifact_routing_plan_ref(routing_plan) if routing_plan is not None else None
        execution_plan_ref = (
            artifact_execution_plan_ref(execution_plan) if execution_plan is not None else None
        )
        execution_batch_ref = (
            artifact_execution_batch_ref(execution_batch) if execution_batch is not None else None
        )
        receipts = (
            {receipt.artifact_id: receipt for receipt in execution_batch.receipts}
            if execution_batch is not None
            else {}
        )
        entries = (
            tuple(sorted(routing_plan.entries, key=lambda item: item.artifact_id))
            if routing_plan is not None
            else ()
        )
        artifact_results = tuple(
            _compile_artifact_result(
                entry=entry,
                routing_plan_ref=routing_plan_ref,
                execution_plan_ref=execution_plan_ref,
                receipt=receipts.get(entry.artifact_id),
                audit=audit,
            )
            for entry in entries
        )
        artifact_result_refs = tuple(artifact_build_result_v2_ref(item) for item in artifact_results)
        candidate_output_refs = tuple(
            sorted(
                (
                    _object_ref_from_facade(item.execution_receipt.facade_result.output_ref)
                    for item in artifact_results
                    if item.outcome is ArtifactBuildResultOutcomeV2.SUCCEEDED
                    and item.execution_receipt is not None
                    and item.execution_receipt.facade_result is not None
                    and item.execution_receipt.facade_result.output_ref is not None
                ),
                key=_ref_key,
            )
        )
        direct_failed = _artifact_ids(
            artifact_results,
            {
                ArtifactBuildResultOutcomeV2.BLOCKED_CAPABILITY,
                ArtifactBuildResultOutcomeV2.BLOCKED_POLICY,
                ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE,
                ArtifactBuildResultOutcomeV2.TERMINAL_FAILURE,
            },
        )
        resumable = _artifact_ids(
            artifact_results,
            {ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE},
        )
        dependency_blocked = _artifact_ids(
            artifact_results,
            {ArtifactBuildResultOutcomeV2.BLOCKED_DEPENDENCY},
        )
        required_incomplete = tuple(
            item.route_entry.artifact_id
            for item in artifact_results
            if item.outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED
            and item.route_entry.criticality is not AttachmentCriticality.OPTIONAL
        )
        optional_incomplete = tuple(
            item.route_entry.artifact_id
            for item in artifact_results
            if item.outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED
            and item.route_entry.criticality is AttachmentCriticality.OPTIONAL
        )
        routing_request_ref = artifact_routing_request_ref(routing_request)
        routing_compilation_ref = artifact_routing_compilation_result_ref(routing_result)
        audit_refs = [
            routing_request_ref,
            routing_compilation_ref,
            *artifact_result_refs,
        ]
        for ref in (
            routing_plan_ref,
            execution_plan_ref,
            execution_batch_ref,
        ):
            if ref is not None:
                audit_refs.append(ref)
        result = AttachmentReconstructionResultV2(
            attachment_reconstruction_result_v2_id=("attachment-reconstruction-result://pending"),
            routing_request_ref=routing_request_ref,
            routing_compilation_result_ref=routing_compilation_ref,
            attachment_planning_context_ref=(routing_request.attachment_planning_context_ref),
            producer_task_view_ref=routing_request.producer_task_view_ref,
            evidence_matrix_ref=routing_request.artifact_evidence_matrix_ref,
            artifact_routing_plan_ref=routing_plan_ref,
            artifact_execution_plan_ref=execution_plan_ref,
            artifact_execution_batch_ref=execution_batch_ref,
            artifact_results=artifact_results,
            artifact_result_refs=artifact_result_refs,
            outcome=attachment_reconstruction_outcome_v2(
                artifact_results,
                has_routing_plan=routing_plan is not None,
            ),
            candidate_output_refs=candidate_output_refs,
            accepted_artifact_refs=(),
            failed_artifact_ids=direct_failed,
            resumable_artifact_ids=resumable,
            dependency_blocked_artifact_ids=dependency_blocked,
            required_incomplete_artifact_ids=required_incomplete,
            optional_incomplete_artifact_ids=optional_incomplete,
            frozen_result=None,
            frozen_projection_gap="PRE_VALIDATION",
            environment_spec_ref=None,
            provenance_manifest_ref=None,
            quality_report_ref=None,
            package_sha256=None,
            input_state_only=None,
            policy_version=ARTIFACT_RESULT_POLICY_VERSION,
            attachment_reconstruction_result_v2_sha256="0" * 64,
            audit=_safe_audit(audit, tuple(audit_refs)),
        )
        digest = attachment_reconstruction_result_v2_carried_sha256(result)
        result = result.model_copy(
            update={
                "attachment_reconstruction_result_v2_id": (
                    f"attachment-reconstruction-result://sha256/{digest}"
                ),
                "attachment_reconstruction_result_v2_sha256": digest,
            }
        )
        validate_attachment_reconstruction_result_v2_identity(result)
        return result


def _compile_artifact_result(
    *,
    entry: ArtifactRoutePlanEntryV2,
    routing_plan_ref: ObjectRef | None,
    execution_plan_ref: ObjectRef | None,
    receipt: ArtifactExecutionReceiptV2 | None,
    audit: ContractAudit,
) -> ArtifactBuildResultV2:
    if routing_plan_ref is None:
        raise ArtifactResultPolicyError("artifact result requires routing plan")
    routed = entry.outcome in {
        ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
        ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
    }
    if routed:
        if execution_plan_ref is None or receipt is None:
            raise ArtifactResultPolicyError("routed artifact is missing execution plan or receipt")
        outcome = ArtifactBuildResultOutcomeV2(receipt.outcome.value)
    else:
        if receipt is not None:
            raise ArtifactResultPolicyError("routing-blocked artifact cannot have execution receipt")
        outcome = ArtifactBuildResultOutcomeV2(entry.outcome.value)
    projection_gap = artifact_build_result_projection_gap_v2(
        route_entry=entry,
        receipt=receipt,
    )
    frozen_result = (
        _compile_frozen_artifact_result(
            entry=entry,
            routing_plan_ref=routing_plan_ref,
            receipt=receipt,
            outcome=outcome,
            audit=audit,
        )
        if projection_gap is None
        else None
    )
    audit_refs = [routing_plan_ref]
    if execution_plan_ref is not None:
        audit_refs.append(execution_plan_ref)
    if receipt is not None:
        audit_refs.append(artifact_execution_receipt_ref(receipt))
    result = ArtifactBuildResultV2(
        artifact_build_result_v2_id="artifact-build-result://pending",
        artifact_routing_plan_ref=routing_plan_ref,
        route_entry=entry,
        artifact_execution_plan_ref=execution_plan_ref,
        execution_receipt=receipt,
        outcome=outcome,
        retryable=(outcome is ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE),
        frozen_result=frozen_result,
        frozen_projection_gap=projection_gap,
        policy_version=ARTIFACT_RESULT_POLICY_VERSION,
        artifact_build_result_v2_sha256="0" * 64,
        audit=_safe_audit(audit, tuple(audit_refs)),
    )
    digest = artifact_build_result_v2_carried_sha256(result)
    result = result.model_copy(
        update={
            "artifact_build_result_v2_id": (f"artifact-build-result://sha256/{digest}"),
            "artifact_build_result_v2_sha256": digest,
        }
    )
    validate_artifact_build_result_v2_identity(result)
    return result


def _compile_frozen_artifact_result(
    *,
    entry: ArtifactRoutePlanEntryV2,
    routing_plan_ref: ObjectRef,
    receipt: ArtifactExecutionReceiptV2 | None,
    outcome: ArtifactBuildResultOutcomeV2,
    audit: ContractAudit,
) -> ArtifactBuildResult:
    if (
        entry.build_spec is None
        or receipt is None
        or receipt.facade_result is None
        or receipt.facade_result.worker_version is None
    ):
        raise ArtifactResultPolicyError("frozen result requires observed build and execution facts")
    facade_result = receipt.facade_result
    worker_version = facade_result.worker_version
    if worker_version is None:
        raise ArtifactResultPolicyError("frozen result requires observed worker version")
    lineage = (
        routing_plan_ref,
        artifact_build_spec_v2_ref(entry.build_spec),
        receipt.artifact_execution_plan_ref,
        receipt.artifact_execution_unit_ref,
        artifact_execution_receipt_ref(receipt),
        _object_ref_from_facade(attachment_execution_result_ref(facade_result)),
    )
    failure = None
    if outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED:
        if facade_result.failure_code is None:
            raise ArtifactResultPolicyError("failed frozen result is missing execution failure code")
        failure = artifact_execution_failure_record_v2(
            outcome,
            facade_result.failure_code,
        )
    result = ArtifactBuildResult(
        artifact_build_result_id="artifact-build-result://pending",
        build_spec_ref=artifact_build_spec_v2_ref(entry.build_spec),
        build_spec_sha256=entry.build_spec.artifact_build_spec_v2_sha256,
        status=ArtifactBuildStatus(outcome.value),
        output_ref=(
            _object_ref_from_facade(facade_result.output_ref)
            if facade_result.output_ref is not None
            else None
        ),
        output_sha256=facade_result.output_sha256,
        lineage_refs=lineage,
        validation_result_refs=(),
        finding_refs=(),
        worker_version=worker_version,
        provider=(
            facade_result.selected_route_id
            if entry.build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER
            else None
        ),
        runtime=(
            facade_result.selected_route_id
            if entry.build_spec.selected_route_kind is ArtifactRouteKindV2.RUNTIME
            else None
        ),
        failure=failure,
        audit=_safe_audit(audit, lineage),
    )
    digest = artifact_build_result_v1_carried_sha256(result)
    return result.model_copy(
        update={"artifact_build_result_id": (f"artifact-build-result://sha256/{digest}")}
    )


def _validate_routing_source(
    request: ArtifactRoutingRequest,
    result: ArtifactRoutingCompilationResult,
) -> None:
    request_digest = artifact_routing_request_carried_sha256(request)
    if (
        request.request_sha256 != request_digest
        or request.request_id != f"artifact-routing-request://sha256/{request_digest}"
    ):
        raise ArtifactResultPolicyError("routing request identity is stale")
    result_digest = artifact_routing_compilation_result_carried_sha256(result)
    if (
        result.result_sha256 != result_digest
        or result.result_id != f"artifact-routing-compilation-result://sha256/{result_digest}"
    ):
        raise ArtifactResultPolicyError("routing compilation result identity is stale")
    if result.request_ref != artifact_routing_request_ref(request):
        raise ArtifactResultPolicyError("routing compilation result belongs to another request")
    plan = result.routing_plan
    if plan is None:
        if request.artifact_evidence_matrix_ref is not None:
            raise ArtifactResultPolicyError("NOT_REQUIRED routing source contains attachment work")
        return
    plan_digest = artifact_routing_plan_carried_sha256(plan)
    if (
        plan.artifact_routing_plan_sha256 != plan_digest
        or plan.artifact_routing_plan_id != f"artifact-routing-plan://sha256/{plan_digest}"
    ):
        raise ArtifactResultPolicyError("routing plan identity is stale")
    if (
        plan.attachment_planning_context_ref != request.attachment_planning_context_ref
        or plan.producer_task_view_ref != request.producer_task_view_ref
        or plan.artifact_evidence_matrix_ref != request.artifact_evidence_matrix_ref
        or plan.artifact_routing_policy_ref != request.artifact_routing_policy_ref
    ):
        raise ArtifactResultPolicyError("routing plan does not match request source refs")
    for entry in plan.entries:
        if entry.build_spec is not None:
            _validate_build_spec_identity(entry.build_spec)
        if entry.facade_route_decision is not None:
            decision_digest = attachment_route_decision_carried_sha256(entry.facade_route_decision)
            if (
                entry.facade_route_decision.route_decision_sha256 != decision_digest
                or entry.facade_route_decision.route_decision_id
                != f"attachment-route-decision://sha256/{decision_digest}"
            ):
                raise ArtifactResultPolicyError("routing plan route decision identity is stale")


def _validate_execution_source(
    *,
    routing_result: ArtifactRoutingCompilationResult,
    execution_result: ArtifactExecutionPlanningResult,
    execution_batch: ArtifactExecutionBatchV2 | None,
) -> ArtifactExecutionPlanV2 | None:
    routing_plan = routing_result.routing_plan
    if routing_plan is None:
        if (
            execution_result.outcome is not ArtifactExecutionPlanningOutcome.NOT_REQUIRED
            or execution_result.execution_plan is not None
            or execution_batch is not None
        ):
            raise ArtifactResultPolicyError("NOT_REQUIRED routing and execution sources are inconsistent")
        return None
    if not routing_plan.routed_artifact_ids:
        if (
            execution_result.outcome is not ArtifactExecutionPlanningOutcome.NO_ROUTED_ARTIFACTS
            or execution_result.execution_plan is not None
            or execution_batch is not None
        ):
            raise ArtifactResultPolicyError("no-routed-artifacts sources are inconsistent")
        return None
    plan = execution_result.execution_plan
    if (
        execution_result.outcome is not ArtifactExecutionPlanningOutcome.PLANNED
        or plan is None
        or execution_batch is None
    ):
        raise ArtifactResultPolicyError("routed artifacts require execution plan and batch")
    ArtifactExecutionPlanCompiler().validate_current(execution_result)
    ArtifactGroupExecutor().validate_current(plan, execution_batch)
    if (
        plan.artifact_routing_plan_ref != artifact_routing_plan_ref(routing_plan)
        or plan.routed_artifact_ids != routing_plan.routed_artifact_ids
    ):
        raise ArtifactResultPolicyError("execution plan does not match routing plan inventory")
    return plan


def _validate_build_spec_identity(spec: ArtifactBuildSpecV2) -> None:
    digest = artifact_build_spec_v2_carried_sha256(spec)
    if (
        spec.artifact_build_spec_v2_sha256 != digest
        or spec.artifact_build_spec_v2_id != f"artifact-build-spec://sha256/{digest}"
    ):
        raise ArtifactResultPolicyError("routing plan build spec identity is stale")


def _artifact_ids(
    results: tuple[ArtifactBuildResultV2, ...],
    outcomes: set[ArtifactBuildResultOutcomeV2],
) -> tuple[str, ...]:
    return tuple(sorted(item.route_entry.artifact_id for item in results if item.outcome in outcomes))


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    unique = {_ref_key(ref): ref for ref in refs}
    return audit.model_copy(update={"input_refs": tuple(unique[key] for key in sorted(unique))})


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
