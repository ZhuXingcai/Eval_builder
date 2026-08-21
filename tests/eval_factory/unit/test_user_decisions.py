from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_task_rewrite import (
    _audit as _r4_audit,
)
from test_task_rewrite import (
    _clear_safety_proposal,
    _compile_plan,
    _source_chain,
)
from test_user_approval_requests import (
    _compile as _compile_request,
)
from test_user_approval_requests import (
    _environment_source,
    _final_source,
    _generation_policy,
    _job_spec,
    _label_source,
    _policy,
)

from eval_factory.approval.decisions import (
    AuthenticatedUserContext,
    UserApprovalRequestRevisionCompiler,
    UserDecisionCompiler,
    UserDecisionPolicyError,
    task_rewrite_adjustment_effect_from_result,
)
from eval_factory.approval.requests import (
    TaskRewriteApprovalSource,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    EnvironmentScopeDecision,
    EnvironmentStrategyChoice,
    FinalReviewScope,
    InvalidationScope,
    QueryPackagingChoice,
    TypedAdjustment,
    UserDecision,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserApprovalRequestRevisionStateV2,
    UserDecisionAdjustmentEffectV2,
    UserDecisionCommitOutcomeV2,
    UserDecisionHandlingPolicyV2,
    user_approval_request_revision_v2_ref,
)
from eval_factory.contracts.approval_v2 import (
    user_approval_request_carried_sha256,
    user_approval_request_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.task_v2 import task_rewrite_plan_version_ref
from eval_factory.task_authoring import TaskRewriteAdjustmentCompiler

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/approval" / "r7-05-user-decisions-v1.json"


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


def _audit(*refs: ObjectRef) -> ContractAudit:
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
        created_at=NOW,
        created_by="r7-05-decision-test",
        governing_versions=(
            VersionBinding(
                component="user-decision",
                version="user-decision/r7-05-v1",
            ),
        ),
        input_refs=tuple(by_key[key] for key in sorted(by_key)),
    )


def _handling_policy(
    *,
    max_adjustments: int = 10,
    max_environment_decisions: int = 20,
    max_effect_refs: int = 20,
    max_revision_depth: int = 5,
    max_reason_characters: int = 1000,
) -> UserDecisionHandlingPolicyV2:
    return UserDecisionHandlingPolicyV2.create(
        max_adjustments_per_decision=max_adjustments,
        max_environment_decisions_per_decision=max_environment_decisions,
        max_effect_result_refs=max_effect_refs,
        max_request_revision_depth=max_revision_depth,
        max_reason_characters=max_reason_characters,
        audit=_audit(),
    )


def _authentication(
    user: str = "requesting-user",
) -> AuthenticatedUserContext:
    return AuthenticatedUserContext(
        authenticated_user=user,
        authentication_context_ref=_ref(
            "authenticated-user-context",
            user,
        ),
    )


def _label_request():
    policy = _policy(ApprovalMode.PLAN_GATES)
    source = _label_source()
    result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
    )
    return policy, source, result, result.requests[0]


def _label_effect(
    request,
    adjustments: tuple[TypedAdjustment, ...],
    *,
    related_result_refs: tuple[ObjectRef, ...] | None = None,
) -> UserDecisionAdjustmentEffectV2:
    return UserDecisionAdjustmentEffectV2.create(
        source_request_ref=user_approval_request_ref(request),
        checkpoint=request.checkpoint,
        source_subject_refs=request.subject_refs,
        source_plan_ref=request.plan_ref,
        adjustments=adjustments,
        producer_result_ref=_ref("label-plan-adjustment-result", "result"),
        resulting_object_ref=_ref("label-plan", "replacement"),
        related_result_refs=related_result_refs
        if related_result_refs is not None
        else (_ref("label-spec", "replacement"),),
        invalidation_scope=InvalidationScope(
            object_refs=(_ref("label-plan", "source"),),
            stages=("label", "release"),
        ),
        audit=_audit(),
    )


def _successor_request(
    request,
    *,
    requested_by: str | None = None,
    approval_policy_ref: ObjectRef | None = None,
) -> object:
    replacement_plan = _ref("label-plan", "replacement")
    replacement_projection = _ref("user-approval-projection", "replacement")
    successor_policy_ref = approval_policy_ref or request.approval_policy_ref
    successor_audit = _audit(
        successor_policy_ref,
        *request.subject_refs,
        replacement_plan,
        replacement_projection,
        replacement_plan,
    ).model_copy(update={"governing_versions": request.audit.governing_versions})
    successor = request.model_copy(
        update={
            "request_id": "user-approval-request://pending",
            "requested_by": requested_by or request.requested_by,
            "approval_policy_ref": successor_policy_ref,
            "plan_ref": replacement_plan,
            "projection_ref": replacement_projection,
            "preview_refs": (replacement_plan,),
            "idempotency_key": "successor-request",
            "audit": successor_audit,
        }
    )
    successor_digest = user_approval_request_carried_sha256(successor)
    return successor.model_copy(update={"request_id": f"user-approval-request://sha256/{successor_digest}"})


def _compile_decision(
    *,
    policy,
    sources,
    request_result,
    request,
    decision: UserDecision,
    selected_request_ref: ObjectRef | None = None,
    handling_policy: UserDecisionHandlingPolicyV2 | None = None,
    authentication: AuthenticatedUserContext | None = None,
    adjustments: tuple[TypedAdjustment, ...] = (),
    adjustment_effect: UserDecisionAdjustmentEffectV2 | None = None,
    environment_decisions: tuple[EnvironmentScopeDecision, ...] = (),
    query_packaging: QueryPackagingChoice | None = None,
    reason: str = "Approved decision.",
    idempotency_key: str | None = None,
):
    return UserDecisionCompiler().compile(
        job_spec=_job_spec(policy),
        approval_policy=policy,
        generation_policy=_generation_policy(),
        request_compilation=request_result,
        checkpoint_sources=sources,
        request_ref=selected_request_ref or user_approval_request_ref(request),
        handling_policy=handling_policy or _handling_policy(),
        authentication=authentication or _authentication(),
        decision=decision,
        adjustments=adjustments,
        adjustment_effect=adjustment_effect,
        environment_decisions=environment_decisions,
        query_packaging=query_packaging,
        reason=reason,
        idempotency_key=idempotency_key
        if idempotency_key is not None
        else f"decision-{decision.value.casefold()}",
        decided_at=NOW,
        audit=_audit(),
    )


def test_accept_binds_exact_current_request_and_authenticated_user() -> None:
    policy, source, request_result, request = _label_request()

    result = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ACCEPT,
    )

    record = result.decision_record
    assert result.outcome is UserDecisionCommitOutcomeV2.ACCEPTED
    assert record.request_ref == user_approval_request_ref(request)
    assert record.approval_policy_ref == request.approval_policy_ref
    assert record.subject_refs == request.subject_refs
    assert record.plan_ref == request.plan_ref
    assert record.projection_ref == request.projection_ref
    assert record.authenticated_user == request.requested_by
    assert record.resulting_object_ref is None
    assert record.invalidation_scope is None

    UserDecisionCompiler().validate_current(
        result,
        job_spec=_job_spec(policy),
        approval_policy=policy,
        generation_policy=_generation_policy(),
        checkpoint_sources=(source,),
        handling_policy=_handling_policy(),
        authentication=_authentication(),
    )


def test_decision_rejects_auth_mismatch_and_non_requested_result() -> None:
    policy, source, request_result, request = _label_request()

    with pytest.raises(UserDecisionPolicyError, match="authenticated"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            authentication=_authentication("other-user"),
        )

    none = _policy(ApprovalMode.NONE)
    disabled = _compile_request(
        policy=none,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(),
    )
    with pytest.raises(UserDecisionPolicyError, match="REQUESTED"):
        UserDecisionCompiler().compile(
            job_spec=_job_spec(none),
            approval_policy=none,
            generation_policy=_generation_policy(),
            request_compilation=disabled,
            checkpoint_sources=(),
            request_ref=_ref("user-approval-request", "missing"),
            handling_policy=_handling_policy(),
            authentication=_authentication(),
            decision=UserDecision.ACCEPT,
            adjustments=(),
            adjustment_effect=None,
            environment_decisions=(),
            query_packaging=None,
            reason="Should fail.",
            idempotency_key="disabled-decision",
            decided_at=NOW,
            audit=_audit(),
        )


def test_adjust_requires_exact_effect_and_derives_result_and_invalidation() -> None:
    policy, source, request_result, request = _label_request()
    adjustment = TypedAdjustment(
        target_path="boundary",
        operation="SET",
        value="Use current executed facts only.",
        reason="Tighten the boundary.",
    )
    effect = _label_effect(request, (adjustment,))

    result = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=effect,
    )

    assert result.outcome is UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED
    assert result.decision_record.resulting_object_ref == (effect.resulting_object_ref)
    assert result.decision_record.invalidation_scope == effect.invalidation_scope
    assert result.adjustment_effect == effect

    with pytest.raises(UserDecisionPolicyError, match="effect"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ADJUST,
            adjustments=(adjustment,),
            adjustment_effect=None,
        )


def test_task_rewrite_adjust_binds_real_r4_replacement_and_invalidation() -> None:
    chain = _source_chain()
    contract_set, compiled = _compile_plan(chain)
    assert compiled.plan_version is not None
    assert compiled.preview is not None
    assert compiled.preview_safety_gate is not None
    adjustment = TypedAdjustment(
        target_path="rewrite_style",
        operation="SET",
        value="Direct and concise.",
        reason="Use the configured task style.",
    )
    r4_compiler = TaskRewriteAdjustmentCompiler()
    safety_request = r4_compiler.build_safety_request(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=(adjustment,),
        task_draft=chain["task_draft"],
        leakage_reference_set=chain["reference_set"],
        model_profile="internal-task-rewrite-safety-v1",
        prompt_version="task-rewrite-safety-prompt/v1",
        audit=_r4_audit(),
    )
    r4_result = r4_compiler.apply(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=(adjustment,),
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        safety_request=safety_request,
        safety_proposal=_clear_safety_proposal(safety_request),
        preserved_object_refs=(
            chain["task_draft"].selection_context_ref,
            *chain["task_draft"].task_episode_refs,
            ObjectRef(
                object_type="tool-capability-catalog",
                object_id=chain["catalog"].catalog_id,
                object_version=chain["catalog"].catalog_version,
                object_sha256=chain["catalog"].catalog_sha256,
            ),
        ),
        audit=_r4_audit(),
    )
    assert r4_result.replacement_plan_version is not None
    assert r4_result.invalidation is not None

    policy = _policy(ApprovalMode.PLAN_GATES)
    source = TaskRewriteApprovalSource(
        plan_version=compiled.plan_version,
        preview_safety_gate=compiled.preview_safety_gate,
        preview=compiled.preview,
    )
    request_result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.TASK_REWRITE_PLAN,
        sources=(source,),
    )
    request = request_result.requests[0]
    effect = task_rewrite_adjustment_effect_from_result(
        request=request,
        adjustments=(adjustment,),
        result=r4_result,
        audit=_audit(),
    )
    result = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=effect,
    )

    assert result.decision_record.resulting_object_ref == (
        task_rewrite_plan_version_ref(r4_result.replacement_plan_version)
    )
    assert result.decision_record.invalidation_scope is not None
    assert set(result.decision_record.invalidation_scope.object_refs) == set(
        r4_result.invalidation.invalidated_object_refs
    )
    assert "TASK_DRAFT" in result.decision_record.invalidation_scope.stages


def test_environment_accept_requires_exact_requirement_choices_and_query() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    source = _environment_source()
    request_result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(source,),
    )
    request = request_result.requests[0]
    choices = tuple(
        EnvironmentScopeDecision(
            requirement_id=requirement.requirement_id,
            strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
        )
        for requirement in source.strategy.requirements
    )

    result = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ACCEPT,
        environment_decisions=choices,
        query_packaging=QueryPackagingChoice.INCLUDE_QUERY_YAML,
    )

    assert result.decision_record.environment_decisions == choices
    assert result.decision_record.query_packaging is QueryPackagingChoice.INCLUDE_QUERY_YAML

    with pytest.raises(UserDecisionPolicyError, match="environment"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            environment_decisions=(),
            query_packaging=QueryPackagingChoice.INCLUDE_QUERY_YAML,
        )


def test_request_more_creates_revision_and_materializes_new_request() -> None:
    policy, source, request_result, request = _label_request()

    result = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.REQUEST_MORE_EXAMPLES,
    )

    revision = result.request_revision
    assert revision is not None
    assert result.outcome is (UserDecisionCommitOutcomeV2.MORE_EXAMPLES_REQUESTED)
    assert result.decision_record.resulting_object_ref == (user_approval_request_revision_v2_ref(revision))

    successor = _successor_request(request)
    materialized = UserApprovalRequestRevisionCompiler().materialize(
        pending_revision=revision,
        successor_request=successor,
        handling_policy=_handling_policy(),
        audit=_audit(),
    )
    assert materialized.state is UserApprovalRequestRevisionStateV2.MATERIALIZED
    assert materialized.successor_request_ref == user_approval_request_ref(successor)


def test_reject_and_final_defer_have_no_effect_or_satisfaction_claim() -> None:
    policy, source, request_result, request = _label_request()
    rejected = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.REJECT,
    )
    assert rejected.outcome is UserDecisionCommitOutcomeV2.REJECTED
    assert rejected.adjustment_effect is None
    assert rejected.request_revision is None
    assert rejected.decision_record.resulting_object_ref is None

    final_policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.PROMPTS,
    )
    final_source = _final_source(FinalReviewScope.PROMPTS)
    final_result = _compile_request(
        policy=final_policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(final_source,),
    )
    deferred = _compile_decision(
        policy=final_policy,
        sources=(final_source,),
        request_result=final_result,
        request=final_result.requests[0],
        decision=UserDecision.DEFER,
    )
    assert deferred.outcome is UserDecisionCommitOutcomeV2.DEFERRED
    assert deferred.decision_record.hard_gate_override_requested is False
    assert deferred.decision_record.resulting_object_ref is None


def test_policy_budgets_reject_before_record_construction() -> None:
    policy, source, request_result, request = _label_request()
    with pytest.raises(UserDecisionPolicyError, match="reason"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            handling_policy=_handling_policy(max_reason_characters=5),
            reason="Too long.",
        )


def test_common_admission_rejects_stale_authority_and_invalid_input() -> None:
    policy, source, request_result, request = _label_request()

    with pytest.raises(UserDecisionPolicyError, match="current"):
        _compile_decision(
            policy=policy,
            sources=(_label_source("different"),),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
        )
    with pytest.raises(UserDecisionPolicyError, match="request_ref"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            selected_request_ref=_ref("label-plan", "wrong-type"),
            decision=UserDecision.ACCEPT,
        )
    with pytest.raises(UserDecisionPolicyError, match="exact member"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            selected_request_ref=_ref("user-approval-request", "unknown"),
            decision=UserDecision.ACCEPT,
        )
    with pytest.raises(UserDecisionPolicyError, match="reason"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            reason=" ",
        )
    with pytest.raises(UserDecisionPolicyError, match="idempotency"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            idempotency_key=" ",
        )
    with pytest.raises(UserDecisionPolicyError, match="unsupported"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision="UNKNOWN",  # type: ignore[arg-type]
            idempotency_key="unsupported-decision",
        )
    with pytest.raises(UserDecisionPolicyError, match="not allowed"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.DEFER,
        )


def test_auth_context_and_non_environment_effects_fail_closed() -> None:
    policy, source, request_result, request = _label_request()
    with pytest.raises(UserDecisionPolicyError, match="cannot be empty"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            authentication=_authentication(""),
        )
    with pytest.raises(UserDecisionPolicyError, match="authentication context"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            authentication=AuthenticatedUserContext(
                authenticated_user="requesting-user",
                authentication_context_ref=_ref("user-session", "wrong"),
            ),
        )
    with pytest.raises(UserDecisionPolicyError, match="non-environment"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            environment_decisions=(
                EnvironmentScopeDecision(
                    requirement_id="environment-requirement://unexpected",
                    strategy=EnvironmentStrategyChoice.EXCLUDE_TASK,
                ),
            ),
        )


def test_adjustment_limits_duplicates_effect_drift_and_mismatch() -> None:
    policy, source, request_result, request = _label_request()
    first = TypedAdjustment(
        target_path="boundary",
        operation="SET",
        value="New boundary.",
        reason="Tighten boundary.",
    )
    second = TypedAdjustment(
        target_path="intent",
        operation="SET",
        value="New intent.",
        reason="Clarify intent.",
    )
    effect = _label_effect(request, (first, second))

    with pytest.raises(UserDecisionPolicyError, match="limit"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ADJUST,
            handling_policy=_handling_policy(max_adjustments=1),
            adjustments=(first, second),
            adjustment_effect=effect,
        )
    with pytest.raises(UserDecisionPolicyError, match="unique"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ADJUST,
            adjustments=(first, first),
            adjustment_effect=effect,
        )
    oversized_effect = _label_effect(
        request,
        (first,),
        related_result_refs=(
            _ref("label-spec", "one"),
            _ref("label-spec", "two"),
        ),
    )
    with pytest.raises(UserDecisionPolicyError, match="result ref limit"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ADJUST,
            handling_policy=_handling_policy(max_effect_refs=1),
            adjustments=(first,),
            adjustment_effect=oversized_effect,
        )
    stale_effect = _label_effect(request, (first,)).model_copy(update={"effect_sha256": "f" * 64})
    with pytest.raises(UserDecisionPolicyError, match="stale"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ADJUST,
            adjustments=(first,),
            adjustment_effect=stale_effect,
        )
    wrong_effect = UserDecisionAdjustmentEffectV2.create(
        source_request_ref=_ref("user-approval-request", "other"),
        checkpoint=request.checkpoint,
        source_subject_refs=request.subject_refs,
        source_plan_ref=request.plan_ref,
        adjustments=(first,),
        producer_result_ref=_ref("label-plan-adjustment-result", "result"),
        resulting_object_ref=_ref("label-plan", "replacement"),
        related_result_refs=(),
        invalidation_scope=InvalidationScope(stages=("label",)),
        audit=_audit(),
    )
    with pytest.raises(UserDecisionPolicyError, match="does not match"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ADJUST,
            adjustments=(first,),
            adjustment_effect=wrong_effect,
        )
    with pytest.raises(UserDecisionPolicyError, match="only ADJUST"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            adjustment_effect=_label_effect(request, (first,)),
        )
    with pytest.raises(UserDecisionPolicyError, match="cannot carry"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.REQUEST_MORE_EXAMPLES,
            adjustments=(first,),
        )


def test_environment_choice_budgets_duplicates_and_reject_shape() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    source = _environment_source()
    request_result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(source,),
    )
    request = request_result.requests[0]
    requirement_id = source.strategy.requirements[0].requirement_id
    choice = EnvironmentScopeDecision(
        requirement_id=requirement_id,
        strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
    )
    with pytest.raises(UserDecisionPolicyError, match="query packaging"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            environment_decisions=(choice,),
        )
    with pytest.raises(UserDecisionPolicyError, match="unique"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            environment_decisions=(choice, choice),
            query_packaging=QueryPackagingChoice.ASK_PER_TASK,
        )
    with pytest.raises(UserDecisionPolicyError, match="limit"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            handling_policy=_handling_policy(max_environment_decisions=1),
            environment_decisions=(
                choice,
                EnvironmentScopeDecision(
                    requirement_id="environment-requirement://extra",
                    strategy=EnvironmentStrategyChoice.EXCLUDE_TASK,
                ),
            ),
            query_packaging=QueryPackagingChoice.ASK_PER_TASK,
        )
    with pytest.raises(UserDecisionPolicyError, match="inventory"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.ACCEPT,
            environment_decisions=(
                EnvironmentScopeDecision(
                    requirement_id="environment-requirement://unknown",
                    strategy=EnvironmentStrategyChoice.EXCLUDE_TASK,
                ),
            ),
            query_packaging=QueryPackagingChoice.OMIT_QUERY_YAML,
        )
    with pytest.raises(UserDecisionPolicyError, match="rejected"):
        _compile_decision(
            policy=policy,
            sources=(source,),
            request_result=request_result,
            request=request,
            decision=UserDecision.REJECT,
            environment_decisions=(choice,),
            query_packaging=QueryPackagingChoice.OMIT_QUERY_YAML,
        )


def test_revision_depth_authority_and_currentness_fail_closed() -> None:
    policy, source, request_result, request = _label_request()
    committed = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.REQUEST_MORE_EXAMPLES,
    )
    pending = committed.request_revision
    assert pending is not None
    successor = _successor_request(request)
    compiler = UserApprovalRequestRevisionCompiler()

    with pytest.raises(UserDecisionPolicyError, match="depth"):
        compiler.materialize(
            pending_revision=pending,
            successor_request=successor,
            handling_policy=_handling_policy(max_revision_depth=1),
            audit=_audit(),
        )
    wrong_user = _successor_request(request, requested_by="other-user")
    with pytest.raises(UserDecisionPolicyError, match="authority"):
        compiler.materialize(
            pending_revision=pending,
            successor_request=wrong_user,
            handling_policy=_handling_policy(),
            audit=_audit(),
        )
    wrong_policy = _successor_request(
        request,
        approval_policy_ref=_ref("user-approval-policy", "other"),
    )
    with pytest.raises(UserDecisionPolicyError, match="authority"):
        compiler.materialize(
            pending_revision=pending,
            successor_request=wrong_policy,
            handling_policy=_handling_policy(),
            audit=_audit(),
        )
    materialized = compiler.materialize(
        pending_revision=pending,
        successor_request=successor,
        handling_policy=_handling_policy(),
        audit=_audit(),
    )
    compiler.validate_current(
        materialized,
        pending_revision=pending,
        successor_request=successor,
        handling_policy=_handling_policy(),
    )
    with pytest.raises(UserDecisionPolicyError, match="pending"):
        compiler.materialize(
            pending_revision=materialized,
            successor_request=successor,
            handling_policy=_handling_policy(),
            audit=_audit(),
        )
    stale = materialized.model_copy(update={"revision_sha256": "f" * 64})
    with pytest.raises(UserDecisionPolicyError, match="current"):
        compiler.validate_current(
            stale,
            pending_revision=pending,
            successor_request=successor,
            handling_policy=_handling_policy(),
        )


def test_user_decision_gold_is_complete_and_content_free() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == ("eval-factory/user-decision-gold/r7-05-v1")
    assert payload["decision_policy_version"] == "user-decision/r7-05-v1"
    assert payload["effect_first_adjustment"] is True
    assert payload["pending_adjustment_intent_count"] == 0
    assert payload["synthetic_decision_count"] == 0
    assert payload["stable_hash_seeds"] == [1, 321]
    scenarios = {value["scenario_id"]: value for value in payload["scenarios"]}
    assert set(scenarios) == {
        "label-accept",
        "task-rewrite-adjust",
        "environment-accept",
        "label-more-examples",
        "task-rewrite-more-examples",
        "reject",
        "final-defer",
        "idempotent-replay",
        "changed-same-key",
        "second-key-same-request",
        "authenticated-user-mismatch",
        "stale-request",
        "hard-gate-override",
    }
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "raw_trace",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "attachment_bytes",
        "physical_path",
        "credential",
        "provider_payload",
        "reviewer_id",
        "quorum",
        "claim_lease",
        "appeal",
    ):
        assert forbidden not in serialized
