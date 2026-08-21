from __future__ import annotations

import hashlib

import pytest
from test_user_approval_requests import (
    _compile as _compile_request,
)
from test_user_approval_requests import (
    _environment_source,
    _final_source,
    _label_source,
    _policy,
)
from test_user_decisions import _audit

from eval_factory.approval.adjustments import (
    EnvironmentStrategyAdjustmentCompiler,
    FinalDatasetAdjustmentCompiler,
    LabelPlanAdjustmentCompiler,
    UserPlanAdjustmentPolicyError,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    FinalReviewScope,
    TypedAdjustment,
)
from eval_factory.contracts.approval_application_v2 import (
    environment_strategy_adjustment_result_v2_ref,
    final_dataset_adjustment_result_v2_ref,
    label_plan_adjustment_result_v2_ref,
)
from eval_factory.contracts.approval_decision_v2 import (
    user_decision_adjustment_effect_v2_ref,
)
from eval_factory.contracts.approval_v2 import (
    environment_strategy_ref,
    final_dataset_review_preview_v2_ref,
    label_plan_ref,
)
from eval_factory.contracts.orchestration_v2 import StageNameV2


def _selector(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_label_adjustment_creates_new_spec_plan_and_exact_effect() -> None:
    source = _label_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request_result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
    )
    request = request_result.requests[0]
    source_spec_json = source.label_spec.model_dump_json()
    source_plan_json = source.label_plan.model_dump_json()
    example = source.label_plan.examples[0]
    adjustments = (
        TypedAdjustment(
            target_path="label_plan.boundary",
            operation="SET",
            value="Use only current executed tool facts.",
            reason="Tighten the label boundary.",
        ),
        TypedAdjustment(
            target_path=(f"label_plan.examples[{_selector(example.example_id)}].expected_treatment"),
            operation="REPLACE",
            value="Require an executed and completed search call.",
            reason="Clarify the positive boundary.",
        ),
    )

    compiled = LabelPlanAdjustmentCompiler().compile(
        request=request,
        source_label_spec=source.label_spec,
        source_label_plan=source.label_plan,
        adjustments=adjustments,
        audit=_audit(),
    )

    assert source.label_spec.model_dump_json() == source_spec_json
    assert source.label_plan.model_dump_json() == source_plan_json
    assert compiled.replacement_label_spec.label_version == "v2"
    assert compiled.replacement_label_spec.label_spec_sha256 != (source.label_spec.label_spec_sha256)
    assert label_plan_ref(compiled.replacement_label_plan) == (compiled.result.replacement_label_plan_ref)
    assert compiled.effect.producer_result_ref == (label_plan_adjustment_result_v2_ref(compiled.result))
    assert compiled.effect.resulting_object_ref == (label_plan_ref(compiled.replacement_label_plan))
    assert StageNameV2.LABEL.value in compiled.effect.invalidation_scope.stages
    user_decision_adjustment_effect_v2_ref(compiled.effect)


def test_label_adjustment_supports_complete_closed_path_surface() -> None:
    source = _label_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
    ).requests[0]
    predicate = source.label_spec.positive_predicates[0]
    example = source.label_plan.examples[1]
    adjustments = (
        TypedAdjustment(
            target_path="label_spec.requirement",
            operation="SET",
            value="Detect a verified search-tool execution.",
            reason="Clarify the requirement.",
        ),
        TypedAdjustment(
            target_path="label_spec.decision_threshold",
            operation="SET",
            value=0.9,
            reason="Set the deterministic threshold.",
        ),
        TypedAdjustment(
            target_path="label_spec.review_threshold",
            operation="REPLACE",
            value=0.8,
            reason="Set the review threshold.",
        ),
        TypedAdjustment(
            target_path=(f"label_spec.predicates[{_selector(predicate.predicate_id)}].expected_value"),
            operation="SET",
            value="fetch",
            reason="Change the expected structured value.",
        ),
        TypedAdjustment(
            target_path="label_plan.intent",
            operation="SET",
            value="Detect verified search execution.",
            reason="Clarify the plan intent.",
        ),
        TypedAdjustment(
            target_path="label_plan.abstain_rules",
            operation="ADD",
            value="Abstain when execution completion is unavailable.",
            reason="Add an abstain boundary.",
        ),
        TypedAdjustment(
            target_path="label_plan.blind_spots",
            operation="ADD",
            value="Provider truncation may hide completion.",
            reason="Record a blind spot.",
        ),
        TypedAdjustment(
            target_path=(f"label_plan.examples[{_selector(example.example_id)}].input_summary"),
            operation="REPLACE",
            value="A bounded ambiguous search example.",
            reason="Clarify the example input.",
        ),
    )

    compiled = LabelPlanAdjustmentCompiler().compile(
        request=request,
        source_label_spec=source.label_spec,
        source_label_plan=source.label_plan,
        adjustments=adjustments,
        audit=_audit(),
    )

    assert compiled.replacement_label_spec.requirement == ("Detect a verified search-tool execution.")
    assert compiled.replacement_label_spec.decision_threshold == 0.9
    assert compiled.replacement_label_spec.review_threshold == 0.8
    assert compiled.replacement_label_spec.positive_predicates[0].expected_value == "fetch"
    assert "Provider truncation may hide completion." in (compiled.replacement_label_plan.blind_spots)


def test_label_adjustment_rejects_invalid_replacement_contract() -> None:
    source = _label_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
    ).requests[0]

    with pytest.raises(UserPlanAdjustmentPolicyError, match="replacement label specification"):
        LabelPlanAdjustmentCompiler().compile(
            request=request,
            source_label_spec=source.label_spec,
            source_label_plan=source.label_plan,
            adjustments=(
                TypedAdjustment(
                    target_path="label_spec.decision_threshold",
                    operation="SET",
                    value=0.5,
                    reason="This would put decision below review.",
                ),
            ),
            audit=_audit(),
        )


@pytest.mark.parametrize(
    "target_path",
    (
        "label_spec.label_spec_id",
        "label_spec.audit",
        "label_plan.examples[0].expected_treatment",
        "label_plan.expected_model_path",
    ),
)
def test_label_adjustment_rejects_authority_and_numeric_paths(
    target_path: str,
) -> None:
    source = _label_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
    ).requests[0]

    with pytest.raises(UserPlanAdjustmentPolicyError, match="not allowed"):
        LabelPlanAdjustmentCompiler().compile(
            request=request,
            source_label_spec=source.label_spec,
            source_label_plan=source.label_plan,
            adjustments=(
                TypedAdjustment(
                    target_path=target_path,
                    operation="SET",
                    value="caller-controlled",
                    reason="Must fail closed.",
                ),
            ),
            audit=_audit(),
        )


def test_environment_adjustment_preserves_authority_and_creates_effect() -> None:
    source = _environment_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(source,),
    ).requests[0]
    requirement = source.strategy.requirements[0]
    adjustments = (
        TypedAdjustment(
            target_path=(f"requirements[{_selector(requirement.requirement_id)}].description"),
            operation="SET",
            value="Use a reproducible local workspace.",
            reason="Clarify the environment requirement.",
        ),
    )

    compiled = EnvironmentStrategyAdjustmentCompiler().compile(
        request=request,
        source_strategy=source.strategy,
        adjustments=adjustments,
        audit=_audit(),
    )

    assert compiled.replacement_strategy.requirements[0].evidence_refs == (requirement.evidence_refs)
    assert compiled.replacement_strategy.query_packaging_options == (source.strategy.query_packaging_options)
    assert compiled.replacement_strategy.credential_fabrication_forbidden is True
    assert environment_strategy_ref(compiled.replacement_strategy) == (
        compiled.result.replacement_strategy_ref
    )
    assert compiled.effect.producer_result_ref == (
        environment_strategy_adjustment_result_v2_ref(compiled.result)
    )

    with pytest.raises(UserPlanAdjustmentPolicyError, match="not allowed"):
        EnvironmentStrategyAdjustmentCompiler().compile(
            request=request,
            source_strategy=source.strategy,
            adjustments=(
                TypedAdjustment(
                    target_path="credential_fabrication_forbidden",
                    operation="SET",
                    value=False,
                    reason="Unsafe caller request.",
                ),
            ),
            audit=_audit(),
        )


def test_environment_adjustment_supports_complete_closed_path_surface() -> None:
    source = _environment_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(source,),
    ).requests[0]
    requirement = source.strategy.requirements[0]
    choice = requirement.alternatives[0].strategy
    adjustments = (
        TypedAdjustment(
            target_path="recommended_strategy",
            operation="SET",
            value="TRACE_FAITHFUL_MOCK",
            reason="Prefer the trace-faithful mock.",
        ),
        TypedAdjustment(
            target_path="recommendation_reason",
            operation="REPLACE",
            value="Preserve the evaluated capability without credentials.",
            reason="Clarify the recommendation.",
        ),
        TypedAdjustment(
            target_path=(f"requirements[{_selector(requirement.requirement_id)}].description"),
            operation="SET",
            value="Use a reproducible isolated workspace.",
            reason="Clarify the requirement.",
        ),
        TypedAdjustment(
            target_path=(
                f"requirements[{_selector(requirement.requirement_id)}]"
                f".alternatives.{choice.value}.capability_impact"
            ),
            operation="SET",
            value="Preserves the original environment capability.",
            reason="Clarify the capability impact.",
        ),
    )

    compiled = EnvironmentStrategyAdjustmentCompiler().compile(
        request=request,
        source_strategy=source.strategy,
        adjustments=adjustments,
        audit=_audit(),
    )

    assert compiled.replacement_strategy.recommended_strategy.value == ("TRACE_FAITHFUL_MOCK")
    assert compiled.replacement_strategy.recommendation_reason == (
        "Preserve the evaluated capability without credentials."
    )
    assert compiled.replacement_strategy.requirements[0].alternatives[0].capability_impact == (
        "Preserves the original environment capability."
    )


def test_environment_adjustment_rejects_invalid_replacement_contract() -> None:
    source = _environment_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(source,),
    ).requests[0]
    requirement = source.strategy.requirements[0]

    with pytest.raises(UserPlanAdjustmentPolicyError, match="replacement environment strategy"):
        EnvironmentStrategyAdjustmentCompiler().compile(
            request=request,
            source_strategy=source.strategy,
            adjustments=(
                TypedAdjustment(
                    target_path=(f"requirements[{_selector(requirement.requirement_id)}].description"),
                    operation="SET",
                    value="x" * 4001,
                    reason="This exceeds the contract boundary.",
                ),
            ),
            audit=_audit(),
        )


@pytest.mark.parametrize(
    ("directive", "expected_stage"),
    (
        ("REBUILD_FROM_TASK_AUTHORING", StageNameV2.TASK_AUTHORING),
        ("REBUILD_FROM_ATTACHMENT", StageNameV2.ATTACHMENT),
        ("REBUILD_FROM_ITEM_QUALITY", StageNameV2.ITEM_QUALITY),
        ("EXCLUDE_ITEM", StageNameV2.ITEM_QUALITY),
    ),
)
def test_final_adjustment_supports_each_closed_restart_directive(
    directive: str,
    expected_stage: StageNameV2,
) -> None:
    source = _final_source(FinalReviewScope.SELECTED_ITEMS)
    policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.SELECTED_ITEMS,
    )
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
    ).requests[0]
    item_ref = source.preview.subject_refs[0]

    compiled = FinalDatasetAdjustmentCompiler().compile(
        request=request,
        source_preview=source.preview,
        adjustments=(
            TypedAdjustment(
                target_path=(f"items[{_selector(item_ref.object_id)}].restart_stage"),
                operation="REPLACE",
                value=directive,
                reason="Apply a closed restart directive.",
            ),
        ),
        audit=_audit(),
    )

    assert compiled.result.restart_stages == (expected_stage,)


@pytest.mark.parametrize(
    ("operation", "value", "expected"),
    (
        ("ADD", "REBUILD_FROM_ATTACHMENT", "SET/REPLACE"),
        ("SET", 1, "closed string"),
        ("SET", "PATCH_CONTENT", "unsupported"),
    ),
)
def test_final_adjustment_rejects_invalid_restart_directive(
    operation: str,
    value: str | int,
    expected: str,
) -> None:
    source = _final_source(FinalReviewScope.SELECTED_ITEMS)
    policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.SELECTED_ITEMS,
    )
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
    ).requests[0]
    item_ref = source.preview.subject_refs[0]

    with pytest.raises(UserPlanAdjustmentPolicyError, match=expected):
        FinalDatasetAdjustmentCompiler().compile(
            request=request,
            source_preview=source.preview,
            adjustments=(
                TypedAdjustment(
                    target_path=(f"items[{_selector(item_ref.object_id)}].restart_stage"),
                    operation=operation,
                    value=value,
                    reason="Reject an invalid restart directive.",
                ),
            ),
            audit=_audit(),
        )


def test_final_adjustment_is_directed_invalidation_not_content_patch() -> None:
    source = _final_source(FinalReviewScope.SELECTED_ITEMS)
    policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.SELECTED_ITEMS,
    )
    request = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
    ).requests[0]
    item_ref = source.preview.subject_refs[0]
    adjustment = TypedAdjustment(
        target_path=f"items[{_selector(item_ref.object_id)}].restart_stage",
        operation="SET",
        value="REBUILD_FROM_ITEM_QUALITY",
        reason="Re-run quality for the selected Item.",
    )

    compiled = FinalDatasetAdjustmentCompiler().compile(
        request=request,
        source_preview=source.preview,
        adjustments=(adjustment,),
        audit=_audit(),
    )

    assert compiled.result.target_item_ids == (item_ref.object_id,)
    assert compiled.result.restart_stages == (StageNameV2.ITEM_QUALITY,)
    assert final_dataset_review_preview_v2_ref(compiled.replacement_preview) == (
        compiled.result.replacement_preview_ref
    )
    assert compiled.effect.producer_result_ref == (final_dataset_adjustment_result_v2_ref(compiled.result))
    assert compiled.replacement_preview.prompt_projection_refs == (source.preview.prompt_projection_refs)
    assert compiled.replacement_preview.selected_item_projection_refs == (
        source.preview.selected_item_projection_refs
    )

    with pytest.raises(UserPlanAdjustmentPolicyError, match="not allowed"):
        FinalDatasetAdjustmentCompiler().compile(
            request=request,
            source_preview=source.preview,
            adjustments=(
                TypedAdjustment(
                    target_path="selected_item_projection_refs",
                    operation="REPLACE",
                    value="patched-content",
                    reason="Direct content patch must fail.",
                ),
            ),
            audit=_audit(),
        )
