from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    EnvironmentScopeDecision,
    EnvironmentStrategyChoice,
    ExampleKind,
    InvalidationScope,
    QueryPackagingChoice,
    TypedAdjustment,
    UserDecision,
    UserDecisionRecord,
)
from eval_factory.contracts.approval_decision_v2 import (
    USER_DECISION_POLICY_VERSION,
    UserApprovalRequestRevisionStateV2,
    UserApprovalRequestRevisionV2,
    UserDecisionAdjustmentEffectV2,
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    UserDecisionHandlingPolicyV2,
    user_approval_request_revision_v2_ref,
    user_decision_adjustment_effect_v2_ref,
    user_decision_commit_result_v2_ref,
    user_decision_handling_policy_v2_ref,
    user_decision_record_carried_sha256,
    user_decision_record_ref,
    validate_user_approval_request_revision_v2_identity,
    validate_user_decision_adjustment_effect_v2_identity,
    validate_user_decision_commit_result_v2_identity,
    validate_user_decision_handling_policy_v2_identity,
    validate_user_decision_record_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-05/{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(
    *refs: ObjectRef,
    created_at: datetime = NOW,
    created_by: str = "r7-05-contract-test",
) -> ContractAudit:
    by_key = {
        (
            ref.object_type,
            ref.object_id,
            ref.object_version,
            ref.object_sha256,
        ): ref
        for ref in refs
    }
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(
            VersionBinding(
                component="user-decision",
                version=USER_DECISION_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(by_key[key] for key in sorted(by_key)),
    )


def _policy() -> UserDecisionHandlingPolicyV2:
    return UserDecisionHandlingPolicyV2.create(
        max_adjustments_per_decision=10,
        max_environment_decisions_per_decision=20,
        max_effect_result_refs=20,
        max_request_revision_depth=5,
        max_reason_characters=1000,
        audit=_audit(),
    )


def _adjustment() -> TypedAdjustment:
    return TypedAdjustment(
        target_path="rewrite_style",
        operation="SET",
        value="Concise approved style.",
        reason="Use the approved style.",
    )


def _effect(
    *,
    subject_ref: ObjectRef | None = None,
    plan_ref: ObjectRef | None = None,
    invalidation_scope: InvalidationScope | None = None,
) -> UserDecisionAdjustmentEffectV2:
    request_ref = _ref("user-approval-request", "rewrite")
    resolved_subject_ref = subject_ref or _ref("selection-context", "rewrite")
    resolved_plan_ref = plan_ref or _ref("task-rewrite-plan-version", "source")
    producer_ref = _ref(
        "task-rewrite-adjustment-result",
        "result",
        version="task-rewrite/r4-09-v1",
    )
    result_ref = _ref("task-rewrite-plan-version", "replacement")
    related = (
        _ref("task-rewrite-plan-preview", "replacement"),
        _ref("task-rewrite-preview-safety-gate", "replacement"),
    )
    resolved_invalidation_scope = invalidation_scope or InvalidationScope(
        object_refs=(_ref("r4-task-contract-set", "source"),),
        stages=("task-authoring", "attachment", "item-quality", "release"),
    )
    return UserDecisionAdjustmentEffectV2.create(
        source_request_ref=request_ref,
        checkpoint=ApprovalCheckpoint.TASK_REWRITE_PLAN,
        source_subject_refs=(resolved_subject_ref,),
        source_plan_ref=resolved_plan_ref,
        adjustments=(_adjustment(),),
        producer_result_ref=producer_ref,
        resulting_object_ref=result_ref,
        related_result_refs=related,
        invalidation_scope=resolved_invalidation_scope,
        audit=_audit(),
    )


def _decision_record(
    *,
    decision: UserDecision = UserDecision.ACCEPT,
    checkpoint: ApprovalCheckpoint = ApprovalCheckpoint.TASK_REWRITE_PLAN,
    request_ref: ObjectRef | None = None,
    subject_refs: tuple[ObjectRef, ...] | None = None,
    plan_ref: ObjectRef | None = None,
    projection_ref: ObjectRef | None = None,
    resulting_object_ref: ObjectRef | None = None,
    invalidation_scope: InvalidationScope | None = None,
    adjustments: tuple[TypedAdjustment, ...] = (),
    environment_decisions: tuple[EnvironmentScopeDecision, ...] = (),
    query_packaging: QueryPackagingChoice | None = None,
    idempotency_key: str = "decision-r7-05",
) -> UserDecisionRecord:
    resolved_plan_ref = plan_ref if plan_ref is not None else _ref("task-rewrite-plan-version", "source")
    value = UserDecisionRecord(
        decision_record_id="user-decision-record://pending",
        request_ref=request_ref or _ref("user-approval-request", "rewrite"),
        checkpoint=checkpoint,
        approval_policy_ref=_ref("user-approval-policy", "policy"),
        subject_refs=subject_refs or (_ref("selection-context", "rewrite"),),
        plan_ref=resolved_plan_ref,
        projection_ref=projection_ref or _ref("user-approval-projection", "rewrite"),
        authenticated_user="requesting-user",
        decision=decision,
        adjustments=adjustments,
        environment_decisions=environment_decisions,
        query_packaging=query_packaging,
        reason="USER_ACCEPTED" if decision is UserDecision.ACCEPT else "USER_ADJUSTED",
        resulting_object_ref=resulting_object_ref,
        invalidation_scope=invalidation_scope,
        hard_gate_override_requested=False,
        idempotency_key=idempotency_key,
        decided_at=NOW,
        record_sha256="0" * 64,
    )
    digest = user_decision_record_carried_sha256(value)
    return value.model_copy(
        update={
            "decision_record_id": f"user-decision-record://sha256/{digest}",
            "record_sha256": digest,
        }
    )


def _pending_revision(
    *,
    job_id: str = "job://r7-05",
    approval_policy_ref: ObjectRef | None = None,
) -> UserApprovalRequestRevisionV2:
    return UserApprovalRequestRevisionV2.create_requested(
        job_id=job_id,
        source_request_ref=_ref("user-approval-request", "label"),
        approval_policy_ref=approval_policy_ref or _ref("user-approval-policy", "policy"),
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        requested_by="requesting-user",
        subject_refs=(_ref("label-spec", "label"),),
        plan_ref=_ref("label-plan", "label"),
        projection_ref=_ref("user-approval-projection", "label"),
        audit=_audit(),
    )


def _commit(
    *,
    record: UserDecisionRecord | None = None,
    effect: UserDecisionAdjustmentEffectV2 | None = None,
    revision: UserApprovalRequestRevisionV2 | None = None,
    outcome: UserDecisionCommitOutcomeV2 = UserDecisionCommitOutcomeV2.ACCEPTED,
) -> UserDecisionCommitResultV2:
    decision = record or _decision_record()
    return UserDecisionCommitResultV2.create(
        job_id="job://r7-05",
        dataset_job_spec_ref=_ref("dataset-job-spec", "job"),
        request_compilation_result_ref=_ref(
            "user-approval-request-compilation-result",
            "result",
        ),
        request_ref=decision.request_ref,
        decision_policy_ref=user_decision_handling_policy_v2_ref(_policy()),
        authentication_context_ref=_ref("authenticated-user-context", "user"),
        decision_record=decision,
        adjustment_effect=effect,
        request_revision=revision,
        outcome=outcome,
        audit=_audit(),
    )


def test_decision_policy_is_strict_current_and_audit_independent() -> None:
    first = _policy()
    second = UserDecisionHandlingPolicyV2.create(
        max_adjustments_per_decision=first.max_adjustments_per_decision,
        max_environment_decisions_per_decision=(first.max_environment_decisions_per_decision),
        max_effect_result_refs=first.max_effect_result_refs,
        max_request_revision_depth=first.max_request_revision_depth,
        max_reason_characters=first.max_reason_characters,
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
        ),
    )

    assert user_decision_handling_policy_v2_ref(first) == (user_decision_handling_policy_v2_ref(second))
    validate_user_decision_handling_policy_v2_identity(first)
    with pytest.raises(ValidationError):
        UserDecisionHandlingPolicyV2.model_validate({**first.model_dump(mode="python"), "unknown": True})


def test_adjustment_effect_is_effect_first_and_checkpoint_typed() -> None:
    effect = _effect()

    validate_user_decision_adjustment_effect_v2_identity(effect)
    assert user_decision_adjustment_effect_v2_ref(effect).object_sha256 == (effect.effect_sha256)
    assert effect.hard_gate_revalidation_required is True

    with pytest.raises(ValidationError, match="producer"):
        UserDecisionAdjustmentEffectV2.model_validate(
            {
                **effect.model_dump(mode="python"),
                "producer_result_ref": _ref("label-plan-adjustment-result", "wrong"),
            }
        )
    with pytest.raises(ValidationError):
        UserDecisionAdjustmentEffectV2.model_validate(
            {
                **effect.model_dump(mode="python"),
                "hard_gate_revalidation_required": False,
            }
        )
    with pytest.raises(ValidationError, match="restricted"):
        _effect(
            invalidation_scope=InvalidationScope(
                object_refs=(_ref("private-reference", "forbidden"),),
                stages=("release",),
            )
        )


def test_request_revision_is_immutable_and_materialized_by_successor() -> None:
    pending = _pending_revision()
    validate_user_approval_request_revision_v2_identity(pending)
    assert pending.state is (UserApprovalRequestRevisionStateV2.MORE_EXAMPLES_REQUESTED)
    assert pending.required_example_kinds == (
        ExampleKind.POSITIVE,
        ExampleKind.NEGATIVE,
        ExampleKind.AMBIGUOUS,
        ExampleKind.ABSTAIN,
    )

    materialized = UserApprovalRequestRevisionV2.create_materialized(
        pending_revision=pending,
        successor_request_ref=_ref("user-approval-request", "label-successor"),
        audit=_audit(),
    )
    assert materialized.state is UserApprovalRequestRevisionStateV2.MATERIALIZED
    assert materialized.revision_number == 2
    assert materialized.predecessor_revision_ref == (user_approval_request_revision_v2_ref(pending))
    validate_user_approval_request_revision_v2_identity(materialized)

    with pytest.raises(ValidationError, match="successor"):
        UserApprovalRequestRevisionV2.model_validate(
            {
                **pending.model_dump(mode="python"),
                "state": UserApprovalRequestRevisionStateV2.MATERIALIZED,
            }
        )


def test_frozen_decision_record_has_current_carried_identity() -> None:
    record = _decision_record()
    validate_user_decision_record_identity(record)
    assert user_decision_record_ref(record).object_sha256 == record.record_sha256

    stale = record.model_copy(update={"record_sha256": "f" * 64})
    with pytest.raises(ValueError, match="stale"):
        validate_user_decision_record_identity(stale)


def test_commit_result_binds_decision_effect_or_revision_exactly() -> None:
    accepted = _commit()
    validate_user_decision_commit_result_v2_identity(accepted)
    assert user_decision_commit_result_v2_ref(accepted).object_sha256 == (accepted.commit_sha256)

    effect = _effect()
    adjusted_record = _decision_record(
        decision=UserDecision.ADJUST,
        resulting_object_ref=effect.resulting_object_ref,
        invalidation_scope=effect.invalidation_scope,
        adjustments=effect.adjustments,
    )
    adjusted = _commit(
        record=adjusted_record,
        effect=effect,
        outcome=UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED,
    )
    validate_user_decision_commit_result_v2_identity(adjusted)

    pending = _pending_revision()
    more_record = _decision_record(
        decision=UserDecision.REQUEST_MORE_EXAMPLES,
        checkpoint=pending.checkpoint,
        request_ref=pending.source_request_ref,
        subject_refs=pending.subject_refs,
        plan_ref=pending.plan_ref,
        projection_ref=pending.projection_ref,
        resulting_object_ref=user_approval_request_revision_v2_ref(pending),
    )
    more = _commit(
        record=more_record,
        revision=pending,
        outcome=UserDecisionCommitOutcomeV2.MORE_EXAMPLES_REQUESTED,
    )
    validate_user_decision_commit_result_v2_identity(more)

    with pytest.raises(ValidationError, match="effect"):
        UserDecisionCommitResultV2.model_validate(
            {
                **adjusted.model_dump(mode="python"),
                "adjustment_effect": None,
            }
        )
    with pytest.raises(ValidationError, match="effect"):
        _commit(
            record=adjusted_record,
            effect=_effect(subject_ref=_ref("selection-context", "other")),
            outcome=UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED,
        )
    with pytest.raises(ValidationError, match="revision"):
        _commit(
            record=more_record,
            revision=_pending_revision(
                approval_policy_ref=_ref("user-approval-policy", "other"),
            ),
            outcome=UserDecisionCommitOutcomeV2.MORE_EXAMPLES_REQUESTED,
        )


def test_non_effect_outcomes_cannot_fabricate_resulting_objects() -> None:
    with pytest.raises(ValidationError, match="resulting object"):
        _commit(
            record=_decision_record(
                resulting_object_ref=_ref("task-rewrite-plan-version", "fabricated"),
            )
        )


def test_environment_decisions_are_canonical_and_acceptance_only() -> None:
    first = EnvironmentScopeDecision(
        requirement_id="environment-requirement://b",
        strategy=EnvironmentStrategyChoice.EXCLUDE_TASK,
    )
    second = EnvironmentScopeDecision(
        requirement_id="environment-requirement://a",
        strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
    )
    common = {
        "checkpoint": ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        "subject_refs": (_ref("task-draft", "environment"),),
        "plan_ref": _ref("environment-strategy", "environment"),
        "environment_decisions": (first, second),
        "query_packaging": QueryPackagingChoice.OMIT_QUERY_YAML,
    }
    with pytest.raises(ValidationError, match="sorted and unique"):
        _commit(record=_decision_record(**common))
    with pytest.raises(ValidationError, match="environment choices"):
        _commit(
            record=_decision_record(
                **{**common, "environment_decisions": (second, first)},
                decision=UserDecision.REJECT,
            ),
            outcome=UserDecisionCommitOutcomeV2.REJECTED,
        )


def test_new_decision_schemas_are_closed_and_content_free() -> None:
    def property_names(value: object) -> set[str]:
        if isinstance(value, dict):
            names = set(value.get("properties", {}))
            for child in value.values():
                names.update(property_names(child))
            return names
        if isinstance(value, list):
            names: set[str] = set()
            for child in value:
                names.update(property_names(child))
            return names
        return set()

    forbidden = {
        "raw_trace",
        "trace_text",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "evaluator_material",
        "attachment_bytes",
        "physical_path",
        "credential",
        "provider_payload",
        "reviewer_id",
        "quorum",
        "claim_lease",
        "appeal",
    }
    for model in (
        UserDecisionHandlingPolicyV2,
        UserDecisionAdjustmentEffectV2,
        UserApprovalRequestRevisionV2,
        UserDecisionCommitResultV2,
    ):
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert forbidden.isdisjoint(property_names(schema))
