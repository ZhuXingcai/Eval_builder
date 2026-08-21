from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    InvalidationScope,
    TypedAdjustment,
)
from eval_factory.contracts.approval_application_v2 import (
    USER_PLAN_APPLICATION_POLICY_VERSION,
    DirectedRevalidationPlanV2,
    DirectedRevalidationReportOutcomeV2,
    DirectedRevalidationReportV2,
    EnvironmentStrategyAdjustmentResultV2,
    FinalDatasetAdjustmentResultV2,
    LabelPlanAdjustmentResultV2,
    RevalidationWorkItemV2,
    RevalidationWorkOutcomeV2,
    RevalidationWorkResultV2,
    RevalidationWorkScopeV2,
    UserPlanApplicationPolicyV2,
    UserPlanApplicationV2,
    directed_revalidation_plan_v2_ref,
    directed_revalidation_report_v2_ref,
    environment_strategy_adjustment_result_v2_ref,
    final_dataset_adjustment_result_v2_ref,
    label_plan_adjustment_result_v2_ref,
    revalidation_work_item_v2_ref,
    revalidation_work_result_v2_ref,
    user_plan_application_policy_v2_ref,
    user_plan_application_v2_ref,
    validate_directed_revalidation_plan_v2_identity,
    validate_directed_revalidation_report_v2_identity,
    validate_environment_strategy_adjustment_result_v2_identity,
    validate_final_dataset_adjustment_result_v2_identity,
    validate_label_plan_adjustment_result_v2_identity,
    validate_revalidation_work_item_v2_identity,
    validate_revalidation_work_result_v2_identity,
    validate_user_plan_application_policy_v2_identity,
    validate_user_plan_application_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration_v2 import StageNameV2

NOW = datetime(2026, 8, 2, tzinfo=UTC)
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
        object_id=f"{object_type}://r7-06/{suffix}",
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
        created_by="r7-06-contract-test",
        governing_versions=(
            VersionBinding(
                component="user-plan-application",
                version=USER_PLAN_APPLICATION_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(by_key[key] for key in sorted(by_key)),
    )


def _policy() -> UserPlanApplicationPolicyV2:
    return UserPlanApplicationPolicyV2.create(
        max_adjustments=10,
        max_invalidated_refs=100,
        max_preserved_refs=100,
        max_affected_items=50,
        max_work_items=200,
        max_outputs_per_work=20,
        max_revalidation_attempts=3,
        max_failure_code_characters=128,
        audit=_audit(),
    )


def _adjustment(path: str = "label_plan.boundary") -> TypedAdjustment:
    return TypedAdjustment(
        target_path=path,
        operation="SET",
        value="Use the adjusted boundary.",
        reason="Apply the requesting-user decision.",
    )


def _label_result() -> LabelPlanAdjustmentResultV2:
    source_request = _ref("user-approval-request", "label")
    source_spec = _ref("label-spec", "source")
    source_plan = _ref("label-plan", "source")
    replacement_spec = _ref("label-spec", "replacement")
    replacement_plan = _ref("label-plan", "replacement")
    invalidation = InvalidationScope(
        object_refs=(_ref("label-decision", "source"),),
        stages=(StageNameV2.LABEL.value, StageNameV2.RELEASE.value),
    )
    return LabelPlanAdjustmentResultV2.create(
        source_request_ref=source_request,
        source_label_spec_ref=source_spec,
        source_label_plan_ref=source_plan,
        adjustments=(_adjustment(),),
        replacement_label_spec_ref=replacement_spec,
        replacement_label_plan_ref=replacement_plan,
        invalidation_scope=invalidation,
        audit=_audit(),
    )


def _environment_result() -> EnvironmentStrategyAdjustmentResultV2:
    source_request = _ref("user-approval-request", "environment")
    source_strategy = _ref("environment-strategy", "source")
    replacement_strategy = _ref("environment-strategy", "replacement")
    invalidation = InvalidationScope(
        object_refs=(_ref("environment-spec", "source"),),
        stages=(StageNameV2.ATTACHMENT.value, StageNameV2.RELEASE.value),
    )
    return EnvironmentStrategyAdjustmentResultV2.create(
        source_request_ref=source_request,
        source_strategy_ref=source_strategy,
        adjustments=(
            _adjustment(
                "requirements[aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa].description"
            ),
        ),
        replacement_strategy_ref=replacement_strategy,
        invalidation_scope=invalidation,
        audit=_audit(),
    )


def _final_result() -> FinalDatasetAdjustmentResultV2:
    source_request = _ref("user-approval-request", "final")
    source_preview = _ref("final-dataset-review-preview", "source")
    replacement_preview = _ref("final-dataset-review-preview", "replacement")
    invalidation = InvalidationScope(
        object_refs=(_ref("evaluation-item", "item-a"),),
        stages=(StageNameV2.ITEM_QUALITY.value, StageNameV2.RELEASE.value),
    )
    return FinalDatasetAdjustmentResultV2.create(
        source_request_ref=source_request,
        source_preview_ref=source_preview,
        adjustments=(
            _adjustment(
                "items[aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa].restart_stage"
            ),
        ),
        replacement_preview_ref=replacement_preview,
        target_item_ids=("item://a",),
        restart_stages=(StageNameV2.ITEM_QUALITY,),
        invalidation_scope=invalidation,
        audit=_audit(),
    )


def _application() -> UserPlanApplicationV2:
    policy_ref = user_plan_application_policy_v2_ref(_policy())
    return UserPlanApplicationV2.create(
        job_id="job://r7-06",
        decision_commit_ref=_ref("user-decision-commit", "commit"),
        decision_record_ref=_ref("user-decision-record", "record"),
        adjustment_effect_ref=_ref("user-decision-adjustment-effect", "effect"),
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        source_root_refs=(
            _ref("label-plan", "source"),
            _ref("label-spec", "source"),
        ),
        replacement_root_refs=(
            _ref("label-plan", "replacement"),
            _ref("label-spec", "replacement"),
        ),
        invalidated_object_refs=(_ref("label-decision", "source"),),
        preserved_object_refs=(_ref("trace-envelope", "source"),),
        affected_item_ids=("item://a",),
        excluded_item_ids=(),
        restart_stages=(
            StageNameV2.LABEL,
            StageNameV2.ITEM_QUALITY,
            StageNameV2.BATCH_QUALITY,
            StageNameV2.RELEASE,
        ),
        required_checkpoint_reapprovals=(
            ApprovalCheckpoint.TASK_REWRITE_PLAN,
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
            ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        ),
        policy_ref=policy_ref,
        audit=_audit(),
    )


def _work_item(
    *,
    application_ref: ObjectRef,
    stage: StageNameV2 = StageNameV2.ITEM_QUALITY,
    depends_on: tuple[ObjectRef, ...] = (),
) -> RevalidationWorkItemV2:
    return RevalidationWorkItemV2.create(
        application_ref=application_ref,
        scope=RevalidationWorkScopeV2.ITEM,
        item_id="item://a",
        stage=stage,
        invalidated_input_refs=(_ref("quality-report", "source"),),
        replacement_input_refs=(_ref("r4-task-contract-set", "replacement"),),
        depends_on_work_item_refs=depends_on,
        required_output_types=("item-quality-compilation-result",),
        hard_gate_required=True,
    )


def test_policy_is_strict_current_and_audit_independent() -> None:
    first = _policy()
    second = UserPlanApplicationPolicyV2.create(
        max_adjustments=first.max_adjustments,
        max_invalidated_refs=first.max_invalidated_refs,
        max_preserved_refs=first.max_preserved_refs,
        max_affected_items=first.max_affected_items,
        max_work_items=first.max_work_items,
        max_outputs_per_work=first.max_outputs_per_work,
        max_revalidation_attempts=first.max_revalidation_attempts,
        max_failure_code_characters=first.max_failure_code_characters,
        audit=ContractAudit(
            created_at=datetime(2026, 8, 3, tzinfo=UTC),
            created_by="other-actor",
            governing_versions=first.audit.governing_versions,
            input_refs=(),
        ),
    )

    assert user_plan_application_policy_v2_ref(first) == (user_plan_application_policy_v2_ref(second))
    validate_user_plan_application_policy_v2_identity(first)
    with pytest.raises(ValidationError):
        UserPlanApplicationPolicyV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "unknown": True,
            }
        )
    with pytest.raises(ValueError, match="stale"):
        validate_user_plan_application_policy_v2_identity(
            first.model_copy(update={"policy_sha256": "f" * 64})
        )


def test_checkpoint_producer_results_are_effect_first_and_safe() -> None:
    label = _label_result()
    environment = _environment_result()
    final = _final_result()

    validate_label_plan_adjustment_result_v2_identity(label)
    validate_environment_strategy_adjustment_result_v2_identity(environment)
    validate_final_dataset_adjustment_result_v2_identity(final)
    assert label_plan_adjustment_result_v2_ref(label).object_type == ("label-plan-adjustment-result")
    assert environment_strategy_adjustment_result_v2_ref(environment).object_type == (
        "environment-strategy-adjustment-result"
    )
    assert final_dataset_adjustment_result_v2_ref(final).object_type == ("final-dataset-adjustment-result")
    assert label.hard_gate_revalidation_required is True
    assert final.target_item_ids == ("item://a",)

    with pytest.raises(ValidationError, match="restricted"):
        LabelPlanAdjustmentResultV2.create(
            source_request_ref=label.source_request_ref,
            source_label_spec_ref=label.source_label_spec_ref,
            source_label_plan_ref=label.source_label_plan_ref,
            adjustments=label.adjustments,
            replacement_label_spec_ref=label.replacement_label_spec_ref,
            replacement_label_plan_ref=label.replacement_label_plan_ref,
            invalidation_scope=InvalidationScope(
                object_refs=(_ref("private-reference", "forbidden"),),
                stages=(StageNameV2.RELEASE.value,),
            ),
            audit=_audit(),
        )


def test_application_and_revalidation_plan_are_exact_dags() -> None:
    application = _application()
    application_ref = user_plan_application_v2_ref(application)
    validate_user_plan_application_v2_identity(application)

    item_quality = _work_item(application_ref=application_ref)
    batch_quality = RevalidationWorkItemV2.create(
        application_ref=application_ref,
        scope=RevalidationWorkScopeV2.JOB,
        item_id=None,
        stage=StageNameV2.BATCH_QUALITY,
        invalidated_input_refs=(_ref("batch-quality-report", "source"),),
        replacement_input_refs=(),
        depends_on_work_item_refs=(revalidation_work_item_v2_ref(item_quality),),
        required_output_types=("batch-quality-report",),
        hard_gate_required=True,
    )
    plan = DirectedRevalidationPlanV2.create(
        application_ref=application_ref,
        resolved_job_work_graph_ref=_ref("resolved-job-work-graph", "graph"),
        work_items=(item_quality, batch_quality),
        preserved_item_ids=("item://b",),
        excluded_item_ids=(),
        audit=_audit(),
    )

    validate_revalidation_work_item_v2_identity(item_quality)
    validate_directed_revalidation_plan_v2_identity(plan)
    assert plan.work_item_refs == (
        revalidation_work_item_v2_ref(item_quality),
        revalidation_work_item_v2_ref(batch_quality),
    )
    with pytest.raises(ValidationError, match="preceding"):
        DirectedRevalidationPlanV2.create(
            application_ref=application_ref,
            resolved_job_work_graph_ref=plan.resolved_job_work_graph_ref,
            work_items=(batch_quality, item_quality),
            preserved_item_ids=plan.preserved_item_ids,
            excluded_item_ids=(),
            audit=_audit(),
        )


def test_work_results_and_report_publish_heads_only_when_complete() -> None:
    application = _application()
    application_ref = user_plan_application_v2_ref(application)
    work = _work_item(application_ref=application_ref)
    plan = DirectedRevalidationPlanV2.create(
        application_ref=application_ref,
        resolved_job_work_graph_ref=_ref("resolved-job-work-graph", "graph"),
        work_items=(work,),
        preserved_item_ids=(),
        excluded_item_ids=(),
        audit=_audit(),
    )
    work_result = RevalidationWorkResultV2.create(
        work_item_ref=revalidation_work_item_v2_ref(work),
        stage_result_ref=_ref("stage-result", "result", version="record/v1"),
        outcome=RevalidationWorkOutcomeV2.SUCCEEDED,
        output_refs=(_ref("item-quality-compilation-result", "replacement"),),
        failure_code=None,
        audit=_audit(),
    )
    report = DirectedRevalidationReportV2.create(
        application_ref=application_ref,
        plan_ref=directed_revalidation_plan_v2_ref(plan),
        work_results=(work_result,),
        replacement_current_refs=work_result.output_refs,
        invalidated_prior_refs=application.invalidated_object_refs,
        excluded_item_ids=(),
        required_checkpoint_reapprovals=(ApprovalCheckpoint.FINAL_DATASET_REVIEW,),
        outcome=DirectedRevalidationReportOutcomeV2.COMPLETE,
        audit=_audit(),
    )

    validate_revalidation_work_result_v2_identity(work_result)
    validate_directed_revalidation_report_v2_identity(report)
    assert revalidation_work_result_v2_ref(work_result) in report.work_result_refs
    assert directed_revalidation_report_v2_ref(report).object_sha256 == (report.report_sha256)

    with pytest.raises(ValidationError, match="replacement current"):
        DirectedRevalidationReportV2.create(
            application_ref=application_ref,
            plan_ref=directed_revalidation_plan_v2_ref(plan),
            work_results=(
                RevalidationWorkResultV2.create(
                    work_item_ref=revalidation_work_item_v2_ref(work),
                    stage_result_ref=_ref(
                        "stage-result",
                        "blocked",
                        version="record/v1",
                    ),
                    outcome=RevalidationWorkOutcomeV2.BLOCKED_POLICY,
                    output_refs=(),
                    failure_code="HARD_GATE_BLOCKED",
                    audit=_audit(),
                ),
            ),
            replacement_current_refs=(_ref("quality-report", "unsafe"),),
            invalidated_prior_refs=application.invalidated_object_refs,
            excluded_item_ids=(),
            required_checkpoint_reapprovals=(ApprovalCheckpoint.FINAL_DATASET_REVIEW,),
            outcome=DirectedRevalidationReportOutcomeV2.BLOCKED,
            audit=_audit(),
        )


def test_new_schemas_are_closed_and_content_free() -> None:
    models = (
        UserPlanApplicationPolicyV2,
        LabelPlanAdjustmentResultV2,
        EnvironmentStrategyAdjustmentResultV2,
        FinalDatasetAdjustmentResultV2,
        UserPlanApplicationV2,
        RevalidationWorkItemV2,
        DirectedRevalidationPlanV2,
        RevalidationWorkResultV2,
        DirectedRevalidationReportV2,
    )
    forbidden = {
        "raw_trace",
        "trace_text",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "attachment_bytes",
        "physical_path",
        "credential",
        "provider_payload",
        "exception_text",
        "release_decision",
        "checkpoint_satisfied",
    }

    for model in models:
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        serialized = str(schema).lower()
        for field in forbidden:
            assert field not in serialized
