from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ALL_CHECKPOINTS,
    DECISIONS_BY_CHECKPOINT,
    PLAN_CHECKPOINTS,
    ApprovalCheckpoint,
    ApprovalMode,
    EnvironmentStrategy,
    FinalReviewScope,
    LabelPlan,
    UserApprovalPolicy,
    UserApprovalRequest,
)
from eval_factory.contracts.approval_v2 import (
    USER_APPROVAL_REQUEST_POLICY_VERSION,
    ApprovalRequestGenerationPolicyV2,
    FinalDatasetReviewPreviewV2,
    UserApprovalRequestCompilationOutcomeV2,
    UserApprovalRequestCompilationResultV2,
    approval_request_generation_policy_v2_ref,
    environment_strategy_ref,
    final_dataset_review_preview_v2_ref,
    label_plan_ref,
    user_approval_policy_carried_sha256,
    user_approval_policy_ref,
    user_approval_request_carried_sha256,
    user_approval_request_compilation_result_v2_ref,
    user_approval_request_ref,
    validate_approval_request_generation_policy_v2_identity,
    validate_final_dataset_review_preview_v2_identity,
    validate_user_approval_policy_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.labeling_v2 import (
    LabelSpecV2,
    SemanticResidualSpecV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.orchestration_v2 import (
    CHECKPOINT_STAGES,
    DatasetJobSpecV2,
    dataset_job_spec_v2_ref,
)
from eval_factory.contracts.task_v2 import (
    TaskPromptSafetyGateStatusV2,
    TaskRewritePlanPreviewV2,
    TaskRewritePlanVersionV2,
    TaskRewritePreviewSafetyGateV2,
    task_rewrite_plan_carried_sha256,
    task_rewrite_plan_preview_carried_sha256,
    task_rewrite_plan_preview_ref,
    task_rewrite_plan_version_carried_sha256,
    task_rewrite_plan_version_ref,
    task_rewrite_preview_safety_gate_carried_sha256,
    task_rewrite_preview_safety_gate_ref,
)

USER_APPROVAL_PROJECTION_POLICY_VERSION = "user-approval-projection/r7-04-v1"

_AFFECTED_STAGES = {
    ApprovalCheckpoint.LABEL_PLAN: ("label",),
    ApprovalCheckpoint.TASK_REWRITE_PLAN: ("task-authoring",),
    ApprovalCheckpoint.ENVIRONMENT_STRATEGY: (
        "attachment",
        "task-authoring",
    ),
    ApprovalCheckpoint.FINAL_DATASET_REVIEW: ("release",),
}


class UserApprovalPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LabelPlanApprovalSource:
    label_spec: LabelSpecV2
    label_plan: LabelPlan


@dataclass(frozen=True, slots=True)
class TaskRewriteApprovalSource:
    plan_version: TaskRewritePlanVersionV2
    preview_safety_gate: TaskRewritePreviewSafetyGateV2
    preview: TaskRewritePlanPreviewV2


@dataclass(frozen=True, slots=True)
class EnvironmentStrategyApprovalSource:
    strategy: EnvironmentStrategy


@dataclass(frozen=True, slots=True)
class FinalDatasetReviewApprovalSource:
    preview: FinalDatasetReviewPreviewV2


ApprovalCheckpointSource = (
    LabelPlanApprovalSource
    | TaskRewriteApprovalSource
    | EnvironmentStrategyApprovalSource
    | FinalDatasetReviewApprovalSource
)


class UserApprovalPolicyCompiler:
    policy_version = USER_APPROVAL_REQUEST_POLICY_VERSION

    def compile(
        self,
        *,
        mode: ApprovalMode | None,
        custom_checkpoints: frozenset[ApprovalCheckpoint],
        final_review_scope: FinalReviewScope,
        explicit_choice: bool,
        audit: ContractAudit,
    ) -> UserApprovalPolicy:
        resolved_mode = mode or ApprovalMode.PLAN_GATES
        if resolved_mode is ApprovalMode.CUSTOM:
            if not custom_checkpoints:
                raise UserApprovalPolicyError("CUSTOM approval mode requires checkpoints")
            checkpoints = custom_checkpoints
        else:
            if custom_checkpoints:
                raise UserApprovalPolicyError("custom checkpoints require CUSTOM approval mode")
            checkpoints = {
                ApprovalMode.NONE: frozenset(),
                ApprovalMode.PLAN_GATES: PLAN_CHECKPOINTS,
                ApprovalMode.FINAL_ONLY: frozenset({ApprovalCheckpoint.FINAL_DATASET_REVIEW}),
                ApprovalMode.PLAN_AND_FINAL: ALL_CHECKPOINTS,
            }[resolved_mode]
        if resolved_mode is ApprovalMode.NONE and not explicit_choice:
            raise UserApprovalPolicyError("NONE approval mode requires an explicit choice")
        try:
            value = UserApprovalPolicy(
                policy_id="user-approval-policy://pending",
                policy_version=self.policy_version,
                mode=resolved_mode,
                enabled_checkpoints=checkpoints,
                final_review_scope=final_review_scope,
                explicit_unattended_choice=explicit_choice,
                audit=_safe_audit(audit, ()),
            )
        except ValidationError as exc:
            raise UserApprovalPolicyError("user approval policy configuration is invalid") from exc
        digest = user_approval_policy_carried_sha256(value)
        return value.model_copy(update={"policy_id": f"user-approval-policy://sha256/{digest}"})

    def validate_current(self, policy: UserApprovalPolicy) -> None:
        try:
            parsed = UserApprovalPolicy.model_validate(policy.model_dump(mode="python"))
            validate_user_approval_policy_identity(parsed)
        except (ValidationError, ValueError) as exc:
            raise UserApprovalPolicyError("user approval policy is not current") from exc
        rebuilt = self.compile(
            mode=parsed.mode,
            custom_checkpoints=(
                parsed.enabled_checkpoints if parsed.mode is ApprovalMode.CUSTOM else frozenset()
            ),
            final_review_scope=parsed.final_review_scope,
            explicit_choice=parsed.explicit_unattended_choice,
            audit=parsed.audit,
        )
        if user_approval_policy_ref(rebuilt) != user_approval_policy_ref(parsed):
            raise UserApprovalPolicyError("user approval policy is not current")


class UserApprovalRequestCompiler:
    policy_version = USER_APPROVAL_REQUEST_POLICY_VERSION

    def compile(
        self,
        *,
        job_spec: DatasetJobSpecV2,
        approval_policy: UserApprovalPolicy,
        generation_policy: ApprovalRequestGenerationPolicyV2,
        checkpoint: ApprovalCheckpoint,
        sources: tuple[ApprovalCheckpointSource, ...],
        requested_by: str,
        audit: ContractAudit,
    ) -> UserApprovalRequestCompilationResultV2:
        job, policy, generation = _admit_common(
            job_spec=job_spec,
            approval_policy=approval_policy,
            generation_policy=generation_policy,
            checkpoint=checkpoint,
            requested_by=requested_by,
        )
        policy_ref = user_approval_policy_ref(policy)
        generation_ref = approval_request_generation_policy_v2_ref(generation)
        enabled = checkpoint in policy.enabled_checkpoints
        if not enabled:
            if sources:
                raise UserApprovalPolicyError("disabled checkpoint cannot receive approval sources")
            return _empty_result(
                job=job,
                policy_ref=policy_ref,
                generation_ref=generation_ref,
                checkpoint=checkpoint,
                outcome=(UserApprovalRequestCompilationOutcomeV2.DISABLED),
                audit=audit,
            )
        if not sources:
            if checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY:
                return _empty_result(
                    job=job,
                    policy_ref=policy_ref,
                    generation_ref=generation_ref,
                    checkpoint=checkpoint,
                    outcome=(UserApprovalRequestCompilationOutcomeV2.NOT_REQUIRED),
                    audit=audit,
                )
            raise UserApprovalPolicyError("enabled checkpoint requires current approval sources")
        if len(sources) > generation.max_requests_per_checkpoint:
            raise UserApprovalPolicyError("approval request limit exceeded")
        if checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY and len(sources) != 1:
            raise UserApprovalPolicyError("environment checkpoint requires one aggregate source")
        if checkpoint is ApprovalCheckpoint.FINAL_DATASET_REVIEW and any(
            not isinstance(source, FinalDatasetReviewApprovalSource)
            or source.preview.scope is not policy.final_review_scope
            for source in sources
        ):
            raise UserApprovalPolicyError("final review preview scope does not match approval policy")

        compiled = tuple(
            _compile_source(
                checkpoint=checkpoint,
                source=_parse_source(source),
                policy_ref=policy_ref,
                generation_ref=generation_ref,
                requested_by=requested_by,
                generation_policy=generation,
                audit=audit,
            )
            for source in sources
        )
        requests = tuple(value[0] for value in compiled)
        character_counts = tuple(value[1] for value in compiled)
        request_refs = tuple(user_approval_request_ref(value) for value in requests)
        if len(request_refs) != len(set(request_refs)):
            raise UserApprovalPolicyError("duplicate approval request source identity")
        if sum(character_counts) > (generation.max_preview_characters_per_checkpoint):
            raise UserApprovalPolicyError("checkpoint preview character limit exceeded")
        return UserApprovalRequestCompilationResultV2.create(
            job_id=job.job_id,
            dataset_job_spec_ref=dataset_job_spec_v2_ref(job),
            approval_policy_ref=policy_ref,
            generation_policy_ref=generation_ref,
            checkpoint=checkpoint,
            outcome=UserApprovalRequestCompilationOutcomeV2.REQUESTED,
            requests=requests,
            request_preview_character_counts=character_counts,
            audit=audit,
        )

    def validate_current(
        self,
        result: UserApprovalRequestCompilationResultV2,
        *,
        job_spec: DatasetJobSpecV2,
        approval_policy: UserApprovalPolicy,
        generation_policy: ApprovalRequestGenerationPolicyV2,
        sources: tuple[ApprovalCheckpointSource, ...],
        requested_by: str,
    ) -> None:
        try:
            parsed = UserApprovalRequestCompilationResultV2.model_validate(result.model_dump(mode="python"))
            observed_ref = user_approval_request_compilation_result_v2_ref(parsed)
        except (ValidationError, ValueError) as exc:
            raise UserApprovalPolicyError("approval request result is not current") from exc
        rebuilt = self.compile(
            job_spec=job_spec,
            approval_policy=approval_policy,
            generation_policy=generation_policy,
            checkpoint=parsed.checkpoint,
            sources=sources,
            requested_by=requested_by,
            audit=parsed.audit,
        )
        if user_approval_request_compilation_result_v2_ref(rebuilt) != observed_ref:
            raise UserApprovalPolicyError("approval request result is not current")


def _admit_common(
    *,
    job_spec: DatasetJobSpecV2,
    approval_policy: UserApprovalPolicy,
    generation_policy: ApprovalRequestGenerationPolicyV2,
    checkpoint: ApprovalCheckpoint,
    requested_by: str,
) -> tuple[
    DatasetJobSpecV2,
    UserApprovalPolicy,
    ApprovalRequestGenerationPolicyV2,
]:
    try:
        job = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        policy = UserApprovalPolicy.model_validate(approval_policy.model_dump(mode="python"))
        generation = ApprovalRequestGenerationPolicyV2.model_validate(
            generation_policy.model_dump(mode="python")
        )
        validate_user_approval_policy_identity(policy)
        validate_approval_request_generation_policy_v2_identity(generation)
    except (ValidationError, ValueError) as exc:
        raise UserApprovalPolicyError("approval request source policy is stale or malformed") from exc
    if not requested_by.strip():
        raise UserApprovalPolicyError("approval request requires a requesting user")
    if (
        job.approval_policy_ref != user_approval_policy_ref(policy)
        or job.approval_mode is not policy.mode
        or job.enabled_checkpoints != policy.enabled_checkpoints
    ):
        raise UserApprovalPolicyError("DatasetJobSpec approval policy binding is stale")
    stage = CHECKPOINT_STAGES[checkpoint]
    if (checkpoint in policy.enabled_checkpoints) != (stage in job.requested_stages):
        raise UserApprovalPolicyError("DatasetJobSpec checkpoint stage binding is stale")
    return job, policy, generation


def _parse_source(
    source: ApprovalCheckpointSource,
) -> ApprovalCheckpointSource:
    try:
        if isinstance(source, LabelPlanApprovalSource):
            return LabelPlanApprovalSource(
                label_spec=LabelSpecV2.model_validate(source.label_spec.model_dump(mode="python")),
                label_plan=LabelPlan.model_validate(source.label_plan.model_dump(mode="python")),
            )
        if isinstance(source, TaskRewriteApprovalSource):
            return TaskRewriteApprovalSource(
                plan_version=TaskRewritePlanVersionV2.model_validate(
                    source.plan_version.model_dump(mode="python")
                ),
                preview_safety_gate=(
                    TaskRewritePreviewSafetyGateV2.model_validate(
                        source.preview_safety_gate.model_dump(mode="python")
                    )
                ),
                preview=TaskRewritePlanPreviewV2.model_validate(source.preview.model_dump(mode="python")),
            )
        if isinstance(source, EnvironmentStrategyApprovalSource):
            return EnvironmentStrategyApprovalSource(
                strategy=EnvironmentStrategy.model_validate(source.strategy.model_dump(mode="python"))
            )
        if isinstance(source, FinalDatasetReviewApprovalSource):
            return FinalDatasetReviewApprovalSource(
                preview=FinalDatasetReviewPreviewV2.model_validate(source.preview.model_dump(mode="python"))
            )
    except ValidationError as exc:
        raise UserApprovalPolicyError("approval checkpoint source is malformed") from exc
    raise UserApprovalPolicyError("approval checkpoint source type is unsupported")


def _compile_source(
    *,
    checkpoint: ApprovalCheckpoint,
    source: ApprovalCheckpointSource,
    policy_ref: ObjectRef,
    generation_ref: ObjectRef,
    requested_by: str,
    generation_policy: ApprovalRequestGenerationPolicyV2,
    audit: ContractAudit,
) -> tuple[UserApprovalRequest, int]:
    subjects: tuple[ObjectRef, ...]
    plan_ref: ObjectRef | None
    preview_refs: tuple[ObjectRef, ...]
    character_count: int
    if checkpoint is ApprovalCheckpoint.LABEL_PLAN:
        if not isinstance(source, LabelPlanApprovalSource):
            raise UserApprovalPolicyError("LABEL_PLAN source type is mismatched")
        subjects, plan_ref, preview_refs, character_count = _compile_label_source(source, generation_policy)
    elif checkpoint is ApprovalCheckpoint.TASK_REWRITE_PLAN:
        if not isinstance(source, TaskRewriteApprovalSource):
            raise UserApprovalPolicyError("TASK_REWRITE_PLAN source type is mismatched")
        subjects, plan_ref, preview_refs, character_count = _compile_rewrite_source(source, generation_policy)
    elif checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY:
        if not isinstance(source, EnvironmentStrategyApprovalSource):
            raise UserApprovalPolicyError("ENVIRONMENT_STRATEGY source type is mismatched")
        subjects, plan_ref, preview_refs, character_count = _compile_environment_source(
            source, generation_policy
        )
    else:
        if not isinstance(source, FinalDatasetReviewApprovalSource):
            raise UserApprovalPolicyError("FINAL_DATASET_REVIEW source type is mismatched")
        subjects, plan_ref, preview_refs, character_count = _compile_final_source(source, generation_policy)
    if len(subjects) > generation_policy.max_subject_refs_per_request:
        raise UserApprovalPolicyError("approval request subject ref limit exceeded")
    if len(preview_refs) > generation_policy.max_preview_refs_per_request:
        raise UserApprovalPolicyError("approval request preview ref limit exceeded")
    if character_count > (generation_policy.max_preview_characters_per_request):
        raise UserApprovalPolicyError("approval request preview character limit exceeded")
    projection_ref = _projection_ref(
        checkpoint=checkpoint,
        requested_by=requested_by,
        policy_ref=policy_ref,
        generation_ref=generation_ref,
        subject_refs=subjects,
        plan_ref=plan_ref,
        preview_refs=preview_refs,
    )
    request_seed = {
        "checkpoint": checkpoint.value,
        "requested_by": requested_by,
        "approval_policy_ref": _ref_payload(policy_ref),
        "generation_policy_ref": _ref_payload(generation_ref),
        "subject_refs": [_ref_payload(ref) for ref in subjects],
        "plan_ref": _maybe_ref_payload(plan_ref),
        "projection_ref": _ref_payload(projection_ref),
        "preview_refs": [_ref_payload(ref) for ref in preview_refs],
        "available_decisions": sorted(item.value for item in DECISIONS_BY_CHECKPOINT[checkpoint]),
        "affected_stages": list(_AFFECTED_STAGES[checkpoint]),
        "projection_policy_version": (USER_APPROVAL_PROJECTION_POLICY_VERSION),
    }
    idempotency_digest = _payload_sha256(request_seed)
    value = UserApprovalRequest(
        request_id="user-approval-request://pending",
        checkpoint=checkpoint,
        requested_by=requested_by,
        approval_policy_ref=policy_ref,
        subject_refs=subjects,
        plan_ref=plan_ref,
        projection_ref=projection_ref,
        preview_refs=preview_refs,
        available_decisions=DECISIONS_BY_CHECKPOINT[checkpoint],
        affected_stages=_AFFECTED_STAGES[checkpoint],
        idempotency_key=(f"approval-request-idempotency://sha256/{idempotency_digest}"),
        audit=_safe_audit(
            audit,
            (
                policy_ref,
                *subjects,
                *((plan_ref,) if plan_ref is not None else ()),
                projection_ref,
                *preview_refs,
            ),
        ),
    )
    request_digest = user_approval_request_carried_sha256(value)
    return (
        value.model_copy(update={"request_id": (f"user-approval-request://sha256/{request_digest}")}),
        character_count,
    )


def _compile_label_source(
    source: LabelPlanApprovalSource,
    policy: ApprovalRequestGenerationPolicyV2,
) -> tuple[
    tuple[ObjectRef, ...],
    ObjectRef,
    tuple[ObjectRef, ...],
    int,
]:
    label_ref = _label_spec_ref(source.label_spec)
    _validate_checkpoint_source_audit(
        source.label_plan.audit,
        (source.label_plan.label_spec_ref,),
        "LabelPlan",
    )
    if source.label_plan.label_spec_ref != label_ref:
        raise UserApprovalPolicyError("LabelPlan label-spec binding is stale")
    predicate_refs = _sorted_refs(
        tuple(
            _structured_predicate_ref(predicate)
            for predicate in (
                *source.label_spec.prerequisite_predicates,
                *source.label_spec.positive_predicates,
                *source.label_spec.negative_predicates,
            )
        )
    )
    semantic_refs = (
        ()
        if source.label_spec.semantic_residual is None
        else (_semantic_residual_ref(source.label_spec.semantic_residual),)
    )
    if (
        source.label_plan.deterministic_clause_refs != predicate_refs
        or source.label_plan.semantic_clause_refs != semantic_refs
    ):
        raise UserApprovalPolicyError("LabelPlan clause inventory is not exact")
    example_ids = tuple(example.example_id for example in source.label_plan.examples)
    if len(example_ids) != len(set(example_ids)):
        raise UserApprovalPolicyError("LabelPlan example identities must be unique")
    if len(source.label_plan.examples) > policy.max_examples_per_plan:
        raise UserApprovalPolicyError("LabelPlan example limit exceeded")
    plan_ref = label_plan_ref(source.label_plan)
    return (
        (label_ref,),
        plan_ref,
        (plan_ref,),
        _label_preview_characters(source.label_plan),
    )


def _compile_rewrite_source(
    source: TaskRewriteApprovalSource,
    policy: ApprovalRequestGenerationPolicyV2,
) -> tuple[
    tuple[ObjectRef, ...],
    ObjectRef,
    tuple[ObjectRef, ...],
    int,
]:
    _validate_rewrite_identity(source)
    if len(source.preview.examples) > policy.max_examples_per_plan:
        raise UserApprovalPolicyError("TaskRewritePlan example limit exceeded")
    return (
        (source.plan_version.plan.selection_context_ref,),
        task_rewrite_plan_version_ref(source.plan_version),
        (task_rewrite_plan_preview_ref(source.preview),),
        _rewrite_preview_characters(source.preview),
    )


def _compile_environment_source(
    source: EnvironmentStrategyApprovalSource,
    policy: ApprovalRequestGenerationPolicyV2,
) -> tuple[
    tuple[ObjectRef, ...],
    ObjectRef,
    tuple[ObjectRef, ...],
    int,
]:
    del policy
    requirement_ids = tuple(requirement.requirement_id for requirement in source.strategy.requirements)
    if len(requirement_ids) != len(set(requirement_ids)):
        raise UserApprovalPolicyError("EnvironmentStrategy requirement identities must be unique")
    subjects = _sorted_refs(
        tuple(
            ref for requirement in source.strategy.requirements for ref in requirement.affected_subject_refs
        )
    )
    if not subjects:
        raise UserApprovalPolicyError("EnvironmentStrategy requires affected subjects")
    _validate_checkpoint_source_audit(
        source.strategy.audit,
        subjects,
        "EnvironmentStrategy",
    )
    strategy_ref = environment_strategy_ref(source.strategy)
    return (
        subjects,
        strategy_ref,
        (strategy_ref,),
        _environment_preview_characters(source.strategy),
    )


def _compile_final_source(
    source: FinalDatasetReviewApprovalSource,
    policy: ApprovalRequestGenerationPolicyV2,
) -> tuple[
    tuple[ObjectRef, ...],
    None,
    tuple[ObjectRef, ...],
    int,
]:
    validate_final_dataset_review_preview_v2_identity(source.preview)
    if len(source.preview.sample_navigation_refs) > (policy.max_final_sample_refs):
        raise UserApprovalPolicyError("final review sample ref limit exceeded")
    return (
        source.preview.subject_refs,
        None,
        (final_dataset_review_preview_v2_ref(source.preview),),
        0,
    )


def _validate_rewrite_identity(
    source: TaskRewriteApprovalSource,
) -> None:
    embedded_plan_digest = task_rewrite_plan_carried_sha256(source.plan_version.plan)
    if source.plan_version.plan.task_rewrite_plan_id != f"task-rewrite-plan://sha256/{embedded_plan_digest}":
        raise UserApprovalPolicyError("TaskRewritePlan embedded plan identity is stale")
    plan_digest = task_rewrite_plan_version_carried_sha256(source.plan_version)
    if (
        source.plan_version.task_rewrite_plan_version_sha256 != plan_digest
        or source.plan_version.task_rewrite_plan_version_id
        != f"task-rewrite-plan-version://sha256/{plan_digest}"
    ):
        raise UserApprovalPolicyError("TaskRewritePlanVersion identity is stale")
    gate_digest = task_rewrite_preview_safety_gate_carried_sha256(source.preview_safety_gate)
    if (
        source.preview_safety_gate.gate_sha256 != gate_digest
        or source.preview_safety_gate.gate_id != f"task-rewrite-preview-safety-gate://sha256/{gate_digest}"
    ):
        raise UserApprovalPolicyError("TaskRewritePlan preview safety identity is stale")
    preview_digest = task_rewrite_plan_preview_carried_sha256(source.preview)
    if (
        source.preview.preview_sha256 != preview_digest
        or source.preview.preview_id != f"task-rewrite-plan-preview://sha256/{preview_digest}"
    ):
        raise UserApprovalPolicyError("TaskRewritePlan preview identity is stale")
    _validate_checkpoint_source_audit(
        source.plan_version.audit,
        tuple(
            ref
            for ref in (
                source.plan_version.supersedes_task_rewrite_plan_version_ref,
                source.plan_version.basis_contract_set_ref,
            )
            if ref is not None
        ),
        "TaskRewritePlanVersion",
    )
    _validate_checkpoint_source_audit(
        source.preview_safety_gate.audit,
        (
            source.preview_safety_gate.preview_candidate_ref,
            source.preview_safety_gate.leakage_reference_set_ref,
            *(
                (source.preview_safety_gate.semantic_assessment_ref,)
                if source.preview_safety_gate.semantic_assessment_ref is not None
                else ()
            ),
        ),
        "TaskRewritePlan preview safety gate",
    )
    _validate_checkpoint_source_audit(
        source.preview.audit,
        (
            source.preview.source_plan_version_ref,
            source.preview.safety_gate_ref,
        ),
        "TaskRewritePlan preview",
    )
    if (
        source.preview_safety_gate.status is not TaskPromptSafetyGateStatusV2.PASSED
        or source.preview_safety_gate.findings
    ):
        raise UserApprovalPolicyError("TaskRewritePlan preview safety must be PASSED")
    if source.preview.source_plan_version_ref != task_rewrite_plan_version_ref(
        source.plan_version
    ) or source.preview.safety_gate_ref != task_rewrite_preview_safety_gate_ref(source.preview_safety_gate):
        raise UserApprovalPolicyError("TaskRewritePlan preview binding is stale")
    plan = source.plan_version.plan
    if (
        source.preview.target_capability != plan.target_capability
        or source.preview.rewrite_style != plan.rewrite_style
        or source.preview.fidelity is not plan.fidelity
        or source.preview.operational_noise_policy != plan.operational_noise_policy
        or source.preview.forbidden_content_rules != tuple(sorted(plan.forbidden_content_rules))
        or source.preview.expected_capability_impact != plan.expected_capability_impact
        or tuple(
            (
                example.example_id,
                example.kind,
                example.input_summary,
                example.expected_treatment,
            )
            for example in source.preview.examples
        )
        != tuple(
            (
                example.example_id,
                example.kind.value,
                example.input_summary,
                example.expected_treatment,
            )
            for example in plan.examples
        )
    ):
        raise UserApprovalPolicyError("TaskRewritePlan safe preview does not match the current plan")


def _empty_result(
    *,
    job: DatasetJobSpecV2,
    policy_ref: ObjectRef,
    generation_ref: ObjectRef,
    checkpoint: ApprovalCheckpoint,
    outcome: UserApprovalRequestCompilationOutcomeV2,
    audit: ContractAudit,
) -> UserApprovalRequestCompilationResultV2:
    return UserApprovalRequestCompilationResultV2.create(
        job_id=job.job_id,
        dataset_job_spec_ref=dataset_job_spec_v2_ref(job),
        approval_policy_ref=policy_ref,
        generation_policy_ref=generation_ref,
        checkpoint=checkpoint,
        outcome=outcome,
        requests=(),
        request_preview_character_counts=(),
        audit=audit,
    )


def _projection_ref(
    *,
    checkpoint: ApprovalCheckpoint,
    requested_by: str,
    policy_ref: ObjectRef,
    generation_ref: ObjectRef,
    subject_refs: tuple[ObjectRef, ...],
    plan_ref: ObjectRef | None,
    preview_refs: tuple[ObjectRef, ...],
) -> ObjectRef:
    digest = _payload_sha256(
        {
            "checkpoint": checkpoint.value,
            "requested_by": requested_by,
            "approval_policy_ref": _ref_payload(policy_ref),
            "generation_policy_ref": _ref_payload(generation_ref),
            "subject_refs": [_ref_payload(ref) for ref in _sorted_refs(subject_refs)],
            "plan_ref": _maybe_ref_payload(plan_ref),
            "preview_refs": [_ref_payload(ref) for ref in _sorted_refs(preview_refs)],
            "projection_policy_version": (USER_APPROVAL_PROJECTION_POLICY_VERSION),
        }
    )
    return ObjectRef(
        object_type="user-approval-projection",
        object_id=f"user-approval-projection://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _label_spec_ref(value: LabelSpecV2) -> ObjectRef:
    if value.label_spec_sha256 == "0" * 64:
        raise UserApprovalPolicyError("LabelSpec identity is pending")
    return ObjectRef(
        object_type="label-spec",
        object_id=value.label_spec_id,
        object_version=value.label_version,
        object_sha256=value.label_spec_sha256,
    )


def _structured_predicate_ref(
    value: StructuredPredicateV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="structured-predicate",
        object_id=value.predicate_id,
        object_version="v2",
        object_sha256=value.canonical_sha256(),
    )


def _semantic_residual_ref(
    value: SemanticResidualSpecV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-residual-spec",
        object_id=value.residual_id,
        object_version="v2",
        object_sha256=value.canonical_sha256(),
    )


def _label_preview_characters(value: LabelPlan) -> int:
    return sum(
        len(text)
        for text in (
            value.intent,
            value.boundary,
            *value.abstain_rules,
            *value.expected_model_path,
            *value.blind_spots,
            *(
                text
                for example in value.examples
                for text in (
                    example.input_summary,
                    example.expected_treatment,
                )
            ),
        )
    )


def _rewrite_preview_characters(
    value: TaskRewritePlanPreviewV2,
) -> int:
    return sum(
        len(text)
        for text in (
            value.target_capability,
            value.rewrite_style,
            value.operational_noise_policy,
            *value.forbidden_content_rules,
            value.expected_capability_impact,
            *(
                text
                for example in value.examples
                for text in (
                    example.input_summary,
                    example.expected_treatment,
                )
            ),
        )
    )


def _environment_preview_characters(
    value: EnvironmentStrategy,
) -> int:
    return sum(
        len(text)
        for text in (
            *(
                text
                for requirement in value.requirements
                for text in (
                    requirement.description,
                    *(
                        nested
                        for alternative in requirement.alternatives
                        for nested in (
                            alternative.capability_impact,
                            alternative.blocking_reason or "",
                        )
                    ),
                )
            ),
            value.recommendation_reason or "",
        )
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = (
        *(value for value in audit.governing_versions if value.component != "user-approval-request"),
        VersionBinding(
            component="user-approval-request",
            version=USER_APPROVAL_REQUEST_POLICY_VERSION,
        ),
    )
    return audit.model_copy(
        update={
            "governing_versions": tuple(
                sorted(
                    versions,
                    key=lambda value: (
                        value.component,
                        value.version,
                        value.sha256 or "",
                    ),
                )
            ),
            "input_refs": _sorted_refs(refs),
        }
    )


def _validate_checkpoint_source_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(expected_refs):
        raise UserApprovalPolicyError(f"{label} audit input refs are not exact")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref_payload(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=False)


def _maybe_ref_payload(
    value: ObjectRef | None,
) -> dict[str, object] | None:
    return _ref_payload(value) if value is not None else None


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))
