from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from eval_factory.approval.requests import (
    ApprovalCheckpointSource,
    EnvironmentStrategyApprovalSource,
    UserApprovalPolicyError,
    UserApprovalRequestCompiler,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    EnvironmentScopeDecision,
    InvalidationScope,
    QueryPackagingChoice,
    TypedAdjustment,
    UserApprovalPolicy,
    UserApprovalRequest,
    UserDecision,
    UserDecisionRecord,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserApprovalRequestRevisionStateV2,
    UserApprovalRequestRevisionV2,
    UserDecisionAdjustmentEffectV2,
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    UserDecisionHandlingPolicyV2,
    user_approval_request_revision_v2_ref,
    user_decision_commit_result_v2_ref,
    user_decision_handling_policy_v2_ref,
    user_decision_record_carried_sha256,
    validate_user_approval_request_revision_v2_identity,
    validate_user_decision_adjustment_effect_v2_identity,
    validate_user_decision_handling_policy_v2_identity,
)
from eval_factory.contracts.approval_v2 import (
    ApprovalRequestGenerationPolicyV2,
    UserApprovalRequestCompilationOutcomeV2,
    UserApprovalRequestCompilationResultV2,
    user_approval_request_compilation_result_v2_ref,
    user_approval_request_ref,
    validate_user_approval_request_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    dataset_job_spec_v2_ref,
)
from eval_factory.contracts.task_v2 import (
    task_contract_invalidation_carried_sha256,
    task_contract_invalidation_ref,
    task_rewrite_plan_preview_carried_sha256,
    task_rewrite_plan_preview_ref,
    task_rewrite_plan_version_carried_sha256,
    task_rewrite_plan_version_ref,
    task_rewrite_preview_safety_gate_carried_sha256,
    task_rewrite_preview_safety_gate_ref,
)
from eval_factory.task_authoring.rewrite_models import (
    TaskRewriteAdjustmentResult,
    TaskRewritePlanCompileOutcome,
)


class UserDecisionPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AuthenticatedUserContext:
    authenticated_user: str
    authentication_context_ref: ObjectRef


class UserDecisionCompiler:
    def compile(
        self,
        *,
        job_spec: DatasetJobSpecV2,
        approval_policy: UserApprovalPolicy,
        generation_policy: ApprovalRequestGenerationPolicyV2,
        request_compilation: UserApprovalRequestCompilationResultV2,
        checkpoint_sources: tuple[ApprovalCheckpointSource, ...],
        request_ref: ObjectRef,
        handling_policy: UserDecisionHandlingPolicyV2,
        authentication: AuthenticatedUserContext,
        decision: UserDecision,
        adjustments: tuple[TypedAdjustment, ...],
        adjustment_effect: UserDecisionAdjustmentEffectV2 | None,
        environment_decisions: tuple[EnvironmentScopeDecision, ...],
        query_packaging: QueryPackagingChoice | None,
        reason: str,
        idempotency_key: str,
        decided_at: datetime,
        audit: ContractAudit,
    ) -> UserDecisionCommitResultV2:
        (
            job,
            policy,
            generation,
            compilation,
            handling,
        ) = _admit_common(
            job_spec=job_spec,
            approval_policy=approval_policy,
            generation_policy=generation_policy,
            request_compilation=request_compilation,
            handling_policy=handling_policy,
        )
        request = _select_request(compilation, request_ref)
        try:
            UserApprovalRequestCompiler().validate_current(
                compilation,
                job_spec=job,
                approval_policy=policy,
                generation_policy=generation,
                sources=checkpoint_sources,
                requested_by=request.requested_by,
            )
        except UserApprovalPolicyError as exc:
            raise UserDecisionPolicyError("approval request compilation is not current") from exc
        _validate_authentication(authentication, request)
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise UserDecisionPolicyError("user decision reason cannot be empty")
        if len(normalized_reason) > handling.max_reason_characters:
            raise UserDecisionPolicyError("user decision reason limit exceeded")
        if not idempotency_key.strip():
            raise UserDecisionPolicyError("user decision idempotency key cannot be empty")
        try:
            resolved_decision = decision if isinstance(decision, UserDecision) else UserDecision(decision)
        except ValueError as exc:
            raise UserDecisionPolicyError("user decision is unsupported") from exc
        if resolved_decision not in request.available_decisions:
            raise UserDecisionPolicyError("user decision is not allowed for the request")
        normalized_adjustments = _normalize_adjustments(
            adjustments,
            handling,
        )
        normalized_environment = _normalize_environment_decisions(
            environment_decisions,
            handling,
        )
        _validate_environment_choices(
            request=request,
            checkpoint_sources=checkpoint_sources,
            decision=resolved_decision,
            environment_decisions=normalized_environment,
            query_packaging=query_packaging,
        )

        effect: UserDecisionAdjustmentEffectV2 | None = None
        revision: UserApprovalRequestRevisionV2 | None = None
        resulting_object_ref: ObjectRef | None = None
        invalidation_scope = None
        if resolved_decision is UserDecision.ADJUST:
            effect = _admit_adjustment_effect(
                request=request,
                adjustments=normalized_adjustments,
                effect=adjustment_effect,
                policy=handling,
            )
            resulting_object_ref = effect.resulting_object_ref
            invalidation_scope = effect.invalidation_scope
            outcome = UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED
        elif resolved_decision is UserDecision.REQUEST_MORE_EXAMPLES:
            _require_empty_effect_inputs(
                adjustments=normalized_adjustments,
                effect=adjustment_effect,
                environment_decisions=normalized_environment,
                query_packaging=query_packaging,
                label="request-more-examples",
            )
            revision = UserApprovalRequestRevisionV2.create_requested(
                job_id=job.job_id,
                source_request_ref=user_approval_request_ref(request),
                approval_policy_ref=request.approval_policy_ref,
                checkpoint=request.checkpoint,
                requested_by=request.requested_by,
                subject_refs=request.subject_refs,
                plan_ref=request.plan_ref,
                projection_ref=request.projection_ref,
                audit=audit,
            )
            resulting_object_ref = user_approval_request_revision_v2_ref(revision)
            outcome = UserDecisionCommitOutcomeV2.MORE_EXAMPLES_REQUESTED
        else:
            if normalized_adjustments or adjustment_effect is not None:
                raise UserDecisionPolicyError("only ADJUST may carry an adjustment effect")
            if resolved_decision is UserDecision.ACCEPT:
                outcome = UserDecisionCommitOutcomeV2.ACCEPTED
            elif resolved_decision is UserDecision.REJECT:
                if normalized_environment or query_packaging is not None:
                    raise UserDecisionPolicyError("rejected decision cannot carry environment choices")
                outcome = UserDecisionCommitOutcomeV2.REJECTED
            else:
                if normalized_environment or query_packaging is not None:
                    raise UserDecisionPolicyError("deferred decision cannot carry environment choices")
                outcome = UserDecisionCommitOutcomeV2.DEFERRED

        value = UserDecisionRecord(
            decision_record_id="user-decision-record://pending",
            request_ref=user_approval_request_ref(request),
            checkpoint=request.checkpoint,
            approval_policy_ref=request.approval_policy_ref,
            subject_refs=request.subject_refs,
            plan_ref=request.plan_ref,
            projection_ref=request.projection_ref,
            authenticated_user=authentication.authenticated_user,
            decision=resolved_decision,
            adjustments=normalized_adjustments,
            environment_decisions=normalized_environment,
            query_packaging=query_packaging,
            reason=normalized_reason,
            resulting_object_ref=resulting_object_ref,
            invalidation_scope=invalidation_scope,
            hard_gate_override_requested=False,
            idempotency_key=idempotency_key,
            decided_at=decided_at,
            record_sha256="0" * 64,
        )
        digest = user_decision_record_carried_sha256(value)
        record = value.model_copy(
            update={
                "decision_record_id": (f"user-decision-record://sha256/{digest}"),
                "record_sha256": digest,
            }
        )
        return UserDecisionCommitResultV2.create(
            job_id=job.job_id,
            dataset_job_spec_ref=dataset_job_spec_v2_ref(job),
            request_compilation_result_ref=(user_approval_request_compilation_result_v2_ref(compilation)),
            request_ref=user_approval_request_ref(request),
            decision_policy_ref=user_decision_handling_policy_v2_ref(handling),
            authentication_context_ref=(authentication.authentication_context_ref),
            decision_record=record,
            adjustment_effect=effect,
            request_revision=revision,
            outcome=outcome,
            audit=audit,
        )

    def validate_current(
        self,
        result: UserDecisionCommitResultV2,
        *,
        job_spec: DatasetJobSpecV2,
        approval_policy: UserApprovalPolicy,
        generation_policy: ApprovalRequestGenerationPolicyV2,
        checkpoint_sources: tuple[ApprovalCheckpointSource, ...],
        handling_policy: UserDecisionHandlingPolicyV2,
        authentication: AuthenticatedUserContext,
    ) -> None:
        try:
            parsed = UserDecisionCommitResultV2.model_validate(result.model_dump(mode="python"))
            observed_ref = user_decision_commit_result_v2_ref(parsed)
        except (ValidationError, ValueError) as exc:
            raise UserDecisionPolicyError("user decision commit is not current") from exc
        record = parsed.decision_record
        compilation = _find_compilation_source(
            parsed,
            job_spec=job_spec,
            approval_policy=approval_policy,
            generation_policy=generation_policy,
            checkpoint_sources=checkpoint_sources,
        )
        rebuilt = self.compile(
            job_spec=job_spec,
            approval_policy=approval_policy,
            generation_policy=generation_policy,
            request_compilation=compilation,
            checkpoint_sources=checkpoint_sources,
            request_ref=parsed.request_ref,
            handling_policy=handling_policy,
            authentication=authentication,
            decision=record.decision,
            adjustments=record.adjustments,
            adjustment_effect=parsed.adjustment_effect,
            environment_decisions=record.environment_decisions,
            query_packaging=record.query_packaging,
            reason=record.reason,
            idempotency_key=record.idempotency_key,
            decided_at=record.decided_at,
            audit=parsed.audit,
        )
        if user_decision_commit_result_v2_ref(rebuilt) != observed_ref:
            raise UserDecisionPolicyError("user decision commit is not current")


class UserApprovalRequestRevisionCompiler:
    def materialize(
        self,
        *,
        pending_revision: UserApprovalRequestRevisionV2,
        successor_request: UserApprovalRequest,
        handling_policy: UserDecisionHandlingPolicyV2,
        audit: ContractAudit,
    ) -> UserApprovalRequestRevisionV2:
        try:
            pending = UserApprovalRequestRevisionV2.model_validate(pending_revision.model_dump(mode="python"))
            validate_user_approval_request_revision_v2_identity(pending)
            successor = UserApprovalRequest.model_validate(successor_request.model_dump(mode="python"))
            validate_user_approval_request_identity(successor)
            policy = UserDecisionHandlingPolicyV2.model_validate(handling_policy.model_dump(mode="python"))
            validate_user_decision_handling_policy_v2_identity(policy)
        except (ValidationError, ValueError) as exc:
            raise UserDecisionPolicyError("request revision source is stale or malformed") from exc
        if pending.state is not UserApprovalRequestRevisionStateV2.MORE_EXAMPLES_REQUESTED:
            raise UserDecisionPolicyError("only a pending request revision can be materialized")
        if pending.revision_number + 1 > policy.max_request_revision_depth:
            raise UserDecisionPolicyError("request revision depth limit exceeded")
        if (
            successor.checkpoint is not pending.checkpoint
            or successor.requested_by != pending.requested_by
            or successor.approval_policy_ref != pending.approval_policy_ref
            or successor.subject_refs != pending.subject_refs
        ):
            raise UserDecisionPolicyError("successor request authority differs from the source request")
        return UserApprovalRequestRevisionV2.create_materialized(
            pending_revision=pending,
            successor_request_ref=user_approval_request_ref(successor),
            audit=audit,
        )

    def validate_current(
        self,
        revision: UserApprovalRequestRevisionV2,
        *,
        pending_revision: UserApprovalRequestRevisionV2,
        successor_request: UserApprovalRequest,
        handling_policy: UserDecisionHandlingPolicyV2,
    ) -> None:
        try:
            parsed = UserApprovalRequestRevisionV2.model_validate(revision.model_dump(mode="python"))
            observed_ref = user_approval_request_revision_v2_ref(parsed)
        except (ValidationError, ValueError) as exc:
            raise UserDecisionPolicyError("request revision is not current") from exc
        rebuilt = self.materialize(
            pending_revision=pending_revision,
            successor_request=successor_request,
            handling_policy=handling_policy,
            audit=parsed.audit,
        )
        if user_approval_request_revision_v2_ref(rebuilt) != observed_ref:
            raise UserDecisionPolicyError("request revision is not current")


def task_rewrite_adjustment_effect_from_result(
    *,
    request: UserApprovalRequest,
    adjustments: tuple[TypedAdjustment, ...],
    result: TaskRewriteAdjustmentResult,
    audit: ContractAudit,
) -> UserDecisionAdjustmentEffectV2:
    try:
        current_request = UserApprovalRequest.model_validate(request.model_dump(mode="python"))
        validate_user_approval_request_identity(current_request)
        current_result = TaskRewriteAdjustmentResult.model_validate(result.model_dump(mode="python"))
    except (ValidationError, ValueError) as exc:
        raise UserDecisionPolicyError("task rewrite adjustment source is stale or malformed") from exc
    if current_request.checkpoint is not ApprovalCheckpoint.TASK_REWRITE_PLAN:
        raise UserDecisionPolicyError("task rewrite effect requires a TASK_REWRITE_PLAN request")
    if (
        current_result.outcome is not TaskRewritePlanCompileOutcome.PREVIEWED
        or current_result.replacement_plan_version is None
        or current_result.replacement_preview_safety_gate is None
        or current_result.replacement_preview is None
        or current_result.invalidation is None
    ):
        raise UserDecisionPolicyError("task rewrite adjustment requires complete PREVIEWED output")
    if current_result.result_id != f"task-rewrite-adjustment-result://sha256/{current_result.result_sha256}":
        raise UserDecisionPolicyError("task rewrite adjustment result identity is stale")
    source_ref = task_rewrite_plan_version_ref(current_result.source_plan_version)
    replacement = current_result.replacement_plan_version
    replacement_ref = task_rewrite_plan_version_ref(replacement)
    gate = current_result.replacement_preview_safety_gate
    gate_ref = task_rewrite_preview_safety_gate_ref(gate)
    preview = current_result.replacement_preview
    preview_ref = task_rewrite_plan_preview_ref(preview)
    invalidation = current_result.invalidation
    invalidation_ref = task_contract_invalidation_ref(invalidation)
    if (
        current_request.plan_ref != source_ref
        or task_rewrite_plan_version_carried_sha256(replacement)
        != replacement.task_rewrite_plan_version_sha256
        or replacement.task_rewrite_plan_version_id
        != (f"task-rewrite-plan-version://sha256/{replacement.task_rewrite_plan_version_sha256}")
        or task_rewrite_preview_safety_gate_carried_sha256(gate) != gate.gate_sha256
        or gate.gate_id != f"task-rewrite-preview-safety-gate://sha256/{gate.gate_sha256}"
        or task_rewrite_plan_preview_carried_sha256(preview) != preview.preview_sha256
        or preview.preview_id != f"task-rewrite-plan-preview://sha256/{preview.preview_sha256}"
        or task_contract_invalidation_carried_sha256(invalidation) != invalidation.invalidation_sha256
        or invalidation.invalidation_id
        != f"task-contract-invalidation://sha256/{invalidation.invalidation_sha256}"
        or preview.source_plan_version_ref != replacement_ref
        or preview.safety_gate_ref != gate_ref
        or invalidation.source_plan_version_ref != source_ref
        or invalidation.replacement_plan_version_ref != replacement_ref
    ):
        raise UserDecisionPolicyError("task rewrite adjustment output binding is stale")
    producer_ref = ObjectRef(
        object_type="task-rewrite-adjustment-result",
        object_id=current_result.result_id,
        object_version=current_result.policy_version,
        object_sha256=current_result.result_sha256,
    )
    invalidation_scope = InvalidationScope(
        object_refs=invalidation.invalidated_object_refs,
        stages=tuple(sorted(stage.value for stage in invalidation.required_rebuild_stages)),
    )
    return UserDecisionAdjustmentEffectV2.create(
        source_request_ref=user_approval_request_ref(current_request),
        checkpoint=current_request.checkpoint,
        source_subject_refs=current_request.subject_refs,
        source_plan_ref=current_request.plan_ref,
        adjustments=adjustments,
        producer_result_ref=producer_ref,
        resulting_object_ref=replacement_ref,
        related_result_refs=(gate_ref, preview_ref, invalidation_ref),
        invalidation_scope=invalidation_scope,
        audit=audit,
    )


def _admit_common(
    *,
    job_spec: DatasetJobSpecV2,
    approval_policy: UserApprovalPolicy,
    generation_policy: ApprovalRequestGenerationPolicyV2,
    request_compilation: UserApprovalRequestCompilationResultV2,
    handling_policy: UserDecisionHandlingPolicyV2,
) -> tuple[
    DatasetJobSpecV2,
    UserApprovalPolicy,
    ApprovalRequestGenerationPolicyV2,
    UserApprovalRequestCompilationResultV2,
    UserDecisionHandlingPolicyV2,
]:
    try:
        job = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        policy = UserApprovalPolicy.model_validate(approval_policy.model_dump(mode="python"))
        generation = ApprovalRequestGenerationPolicyV2.model_validate(
            generation_policy.model_dump(mode="python")
        )
        compilation = UserApprovalRequestCompilationResultV2.model_validate(
            request_compilation.model_dump(mode="python")
        )
        handling = UserDecisionHandlingPolicyV2.model_validate(handling_policy.model_dump(mode="python"))
        validate_user_decision_handling_policy_v2_identity(handling)
    except (ValidationError, ValueError) as exc:
        raise UserDecisionPolicyError("user decision source is stale or malformed") from exc
    if compilation.outcome is not UserApprovalRequestCompilationOutcomeV2.REQUESTED:
        raise UserDecisionPolicyError("user decision requires a REQUESTED approval compilation")
    return job, policy, generation, compilation, handling


def _select_request(
    compilation: UserApprovalRequestCompilationResultV2,
    request_ref: ObjectRef,
) -> UserApprovalRequest:
    if request_ref.object_type != "user-approval-request" or request_ref.object_version != "v2":
        raise UserDecisionPolicyError("decision request_ref must reference user-approval-request v2")
    matches = tuple(
        request for request in compilation.requests if user_approval_request_ref(request) == request_ref
    )
    if len(matches) != 1:
        raise UserDecisionPolicyError("decision request is not an exact member of the compilation")
    return matches[0]


def _validate_authentication(
    authentication: AuthenticatedUserContext,
    request: UserApprovalRequest,
) -> None:
    if not authentication.authenticated_user.strip():
        raise UserDecisionPolicyError("authenticated user cannot be empty")
    ref = authentication.authentication_context_ref
    if ref.object_type != "authenticated-user-context" or ref.object_version != "v2":
        raise UserDecisionPolicyError("authentication context must reference authenticated-user-context v2")
    if authentication.authenticated_user != request.requested_by:
        raise UserDecisionPolicyError("authenticated user does not match the requesting user")


def _normalize_adjustments(
    adjustments: tuple[TypedAdjustment, ...],
    policy: UserDecisionHandlingPolicyV2,
) -> tuple[TypedAdjustment, ...]:
    if len(adjustments) > policy.max_adjustments_per_decision:
        raise UserDecisionPolicyError("user decision adjustment limit exceeded")
    ordered = tuple(sorted(adjustments, key=lambda value: value.target_path))
    paths = tuple(value.target_path for value in ordered)
    if len(paths) != len(set(paths)):
        raise UserDecisionPolicyError("user decision adjustment paths must be unique")
    return ordered


def _normalize_environment_decisions(
    values: tuple[EnvironmentScopeDecision, ...],
    policy: UserDecisionHandlingPolicyV2,
) -> tuple[EnvironmentScopeDecision, ...]:
    if len(values) > policy.max_environment_decisions_per_decision:
        raise UserDecisionPolicyError("environment decision limit exceeded")
    ordered = tuple(sorted(values, key=lambda value: value.requirement_id))
    identities = tuple(value.requirement_id for value in ordered)
    if len(identities) != len(set(identities)):
        raise UserDecisionPolicyError("environment decision requirement IDs must be unique")
    return ordered


def _validate_environment_choices(
    *,
    request: UserApprovalRequest,
    checkpoint_sources: tuple[ApprovalCheckpointSource, ...],
    decision: UserDecision,
    environment_decisions: tuple[EnvironmentScopeDecision, ...],
    query_packaging: QueryPackagingChoice | None,
) -> None:
    is_environment = request.checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY
    requires_choices = is_environment and decision in {
        UserDecision.ACCEPT,
        UserDecision.ADJUST,
    }
    if not is_environment:
        if environment_decisions or query_packaging is not None:
            raise UserDecisionPolicyError("non-environment decision cannot carry environment choices")
        return
    if not requires_choices:
        return
    if query_packaging is None:
        raise UserDecisionPolicyError("environment decision requires query packaging")
    sources = tuple(
        source for source in checkpoint_sources if isinstance(source, EnvironmentStrategyApprovalSource)
    )
    if len(sources) != 1:
        raise UserDecisionPolicyError("environment decision requires one current strategy source")
    strategy = sources[0].strategy
    expected_ids = tuple(sorted(requirement.requirement_id for requirement in strategy.requirements))
    observed_ids = tuple(value.requirement_id for value in environment_decisions)
    if observed_ids != expected_ids:
        raise UserDecisionPolicyError("environment decision requirement inventory is not exact")


def _admit_adjustment_effect(
    *,
    request: UserApprovalRequest,
    adjustments: tuple[TypedAdjustment, ...],
    effect: UserDecisionAdjustmentEffectV2 | None,
    policy: UserDecisionHandlingPolicyV2,
) -> UserDecisionAdjustmentEffectV2:
    if not adjustments:
        raise UserDecisionPolicyError("ADJUST requires typed adjustments")
    if effect is None:
        raise UserDecisionPolicyError("ADJUST requires an exact completed adjustment effect")
    try:
        parsed = UserDecisionAdjustmentEffectV2.model_validate(effect.model_dump(mode="python"))
        validate_user_decision_adjustment_effect_v2_identity(parsed)
    except (ValidationError, ValueError) as exc:
        raise UserDecisionPolicyError("adjustment effect is stale or malformed") from exc
    if len(parsed.related_result_refs) > policy.max_effect_result_refs:
        raise UserDecisionPolicyError("adjustment effect result ref limit exceeded")
    if (
        parsed.source_request_ref != user_approval_request_ref(request)
        or parsed.checkpoint is not request.checkpoint
        or parsed.source_subject_refs != request.subject_refs
        or parsed.source_plan_ref != request.plan_ref
        or parsed.adjustments != adjustments
    ):
        raise UserDecisionPolicyError("adjustment effect does not match the approval request")
    return parsed


def _require_empty_effect_inputs(
    *,
    adjustments: tuple[TypedAdjustment, ...],
    effect: UserDecisionAdjustmentEffectV2 | None,
    environment_decisions: tuple[EnvironmentScopeDecision, ...],
    query_packaging: QueryPackagingChoice | None,
    label: str,
) -> None:
    if adjustments or effect is not None or environment_decisions or query_packaging is not None:
        raise UserDecisionPolicyError(f"{label} cannot carry adjustment or environment effects")


def _find_compilation_source(
    result: UserDecisionCommitResultV2,
    *,
    job_spec: DatasetJobSpecV2,
    approval_policy: UserApprovalPolicy,
    generation_policy: ApprovalRequestGenerationPolicyV2,
    checkpoint_sources: tuple[ApprovalCheckpointSource, ...],
) -> UserApprovalRequestCompilationResultV2:
    request = result.decision_record
    rebuilt = UserApprovalRequestCompiler().compile(
        job_spec=job_spec,
        approval_policy=approval_policy,
        generation_policy=generation_policy,
        checkpoint=request.checkpoint,
        sources=checkpoint_sources,
        requested_by=request.authenticated_user,
        audit=result.audit,
    )
    if user_approval_request_compilation_result_v2_ref(rebuilt) != result.request_compilation_result_ref:
        raise UserDecisionPolicyError("request compilation does not match the decision commit")
    return rebuilt


__all__ = [
    "AuthenticatedUserContext",
    "UserApprovalRequestRevisionCompiler",
    "UserDecisionCompiler",
    "UserDecisionPolicyError",
    "task_rewrite_adjustment_effect_from_result",
]
