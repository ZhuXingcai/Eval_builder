from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.agent_system_v2 import (
    CompiledGradingDesignPlanV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    GradingDesignResultV2,
    JudgeAggregationModeV2,
    JudgeDesignSpecV2,
    JudgeDesignValidationOutcomeV2,
    JudgeDesignValidationV2,
    JudgeTaskMappingV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "grading",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(
    at: datetime = datetime(2026, 8, 6, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=at,
        created_by="grading-contract-test",
        governing_versions=(
            VersionBinding(
                component="grading-design",
                version="v1",
            ),
        ),
    )


def _tasks() -> tuple[JudgeTaskMappingV2, ...]:
    return (
        JudgeTaskMappingV2(
            task_key="judge-main",
            criterion_ids=(
                "rubric-criterion://one",
                "rubric-criterion://two",
            ),
            evaluator_binding_id="evaluator-binding://main",
            score_weight_basis_points=10_000,
        ),
    )


def _plan(
    *,
    tasks: tuple[JudgeTaskMappingV2, ...] | None = None,
    version: int = 1,
    predecessor: ObjectRef | None = None,
    required_output_fields: tuple[str, ...] = (
        "ABSTAIN",
        "EVIDENCE_IDS",
        "FAILURE_CLASS",
        "SCORE_BASIS_POINTS",
    ),
    prompt_template_ref: ObjectRef | None = None,
    audit: ContractAudit | None = None,
) -> GradingDesignPlanV2:
    return GradingDesignPlanV2.create(
        plan_id="grading-design-plan://example",
        run_ref=_ref("factory-run"),
        plan_version=version,
        predecessor_plan_ref=predecessor,
        criteria_rubric_result_ref=_ref("criteria-rubric-result"),
        rubric_set_ref=_ref("rubric-set"),
        evaluator_spec_ref=_ref("evaluator-spec"),
        reference_policy_ref=_ref("reference-policy"),
        tool_policy_ref=_ref("tool-policy"),
        generator_model_profile_ref=_ref(
            "model-capability-profile",
            "generator",
        ),
        judge_tasks=tasks or _tasks(),
        aggregation_mode=JudgeAggregationModeV2.WEIGHTED_SUM,
        passing_score_basis_points=7000,
        minimum_confidence_basis_points=8000,
        escalate_on_reference_unavailable=True,
        judge_input_schema_ref=_ref(
            "json-schema",
            "judge-input",
        ),
        judge_output_schema_ref=_ref(
            "json-schema",
            "judge-output",
        ),
        required_output_fields=required_output_fields,
        allowed_judge_model_profile_refs=(
            _ref("model-capability-profile", "generator"),
            _ref("model-capability-profile", "judge"),
        ),
        judge_prompt_template_ref=(prompt_template_ref or _ref("prompt-template")),
        agent_role="grading-design-agent",
        required_capability_ids=("agent-capability://grading-design",),
        specialist_tool_ids=("reference-grant-read",),
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
        model_policy_ref=_ref("model-routing-policy"),
        acceptance_check_refs=(_ref("validator", "grading-design"),),
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=16_000,
        max_cost_micro_usd=500_000,
        audit=audit or _audit(),
    )


def _compiled(
    plan: GradingDesignPlanV2,
) -> CompiledGradingDesignPlanV2:
    return CompiledGradingDesignPlanV2.create(
        compiled_plan_id="compiled-grading-design-plan://example",
        source_plan_ref=plan.to_ref(),
        policy_ref=_ref("factory-run-policy"),
        agent_definition_ref=_ref("agent-definition"),
        capability_refs=(_ref("agent-capability"),),
        criteria_rubric_result_ref=(plan.criteria_rubric_result_ref),
        rubric_set_ref=plan.rubric_set_ref,
        evaluator_spec_ref=plan.evaluator_spec_ref,
        reference_policy_ref=plan.reference_policy_ref,
        tool_policy_ref=plan.tool_policy_ref,
        generator_model_profile_ref=(plan.generator_model_profile_ref),
        judge_prompt_template_ref=(plan.judge_prompt_template_ref),
        model_policy_ref=plan.model_policy_ref,
        acceptance_check_refs=plan.acceptance_check_refs,
        audit=_audit(),
    )


def _spec(
    plan: GradingDesignPlanV2,
) -> JudgeDesignSpecV2:
    return JudgeDesignSpecV2.create(
        design_spec_id="judge-design-spec://example",
        plan_ref=plan.to_ref(),
        criteria_rubric_result_ref=(plan.criteria_rubric_result_ref),
        rubric_set_ref=plan.rubric_set_ref,
        evaluator_spec_ref=plan.evaluator_spec_ref,
        reference_policy_ref=plan.reference_policy_ref,
        tool_policy_ref=plan.tool_policy_ref,
        prompt_template_ref=plan.judge_prompt_template_ref,
        prompt_rendering_ref=_ref("prompt-rendering"),
        route_decision_ref=_ref("model-route-decision"),
        gateway_receipt_ref=_ref("gateway-receipt"),
        gateway_invocation_result_ref=_ref("gateway-invocation-result"),
        proposal_ref=_ref("judge-design-proposal"),
        selected_judge_model_profile_ref=_ref(
            "model-capability-profile",
            "judge",
        ),
        generator_model_profile_ref=(plan.generator_model_profile_ref),
        judge_tasks=plan.judge_tasks,
        aggregation_mode=plan.aggregation_mode,
        passing_score_basis_points=(plan.passing_score_basis_points),
        minimum_confidence_basis_points=(plan.minimum_confidence_basis_points),
        escalate_on_reference_unavailable=(plan.escalate_on_reference_unavailable),
        judge_input_schema_ref=plan.judge_input_schema_ref,
        judge_output_schema_ref=plan.judge_output_schema_ref,
        required_output_fields=plan.required_output_fields,
        reference_grant_refs=(),
        private_material_inventory_sha256=HASH,
        audit=_audit(),
    )


def _validation(
    plan: GradingDesignPlanV2,
    spec: JudgeDesignSpecV2,
) -> JudgeDesignValidationV2:
    return JudgeDesignValidationV2.create(
        validation_id="judge-design-validation://example",
        plan_ref=plan.to_ref(),
        design_spec_ref=spec.to_ref(),
        criterion_ids=(
            "rubric-criterion://one",
            "rubric-criterion://two",
        ),
        validated_output_fields=plan.required_output_fields,
        reference_access_result_refs=(),
        outcome=JudgeDesignValidationOutcomeV2.VALID,
        reason_codes=(),
        audit=_audit(),
    )


def test_grading_plan_is_strict_frozen_and_audit_independent() -> None:
    first = _plan()
    second = _plan(audit=_audit(datetime(2026, 8, 7, tzinfo=UTC)))

    assert first.to_ref() == second.to_ref()
    assert _compiled(first).source_plan_ref == first.to_ref()
    with pytest.raises(ValidationError):
        GradingDesignPlanV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        first.plan_version = 2  # type: ignore[misc]


def test_plan_rejects_partition_schema_weight_and_sensitive_refs() -> None:
    duplicate = (
        JudgeTaskMappingV2(
            task_key="judge-a",
            criterion_ids=("rubric-criterion://one",),
            evaluator_binding_id="evaluator-binding://main",
            score_weight_basis_points=5000,
        ),
        JudgeTaskMappingV2(
            task_key="judge-b",
            criterion_ids=("rubric-criterion://one",),
            evaluator_binding_id="evaluator-binding://main",
            score_weight_basis_points=5000,
        ),
    )
    with pytest.raises(ValidationError, match="criterion IDs"):
        _plan(tasks=duplicate)

    with pytest.raises(ValidationError, match="output fields"):
        _plan(
            required_output_fields=(
                "ABSTAIN",
                "EVIDENCE_IDS",
                "EXTRA_FIELD",
                "SCORE_BASIS_POINTS",
            )
        )

    with pytest.raises(ValidationError, match="sensitive"):
        _plan(
            prompt_template_ref=_ref(
                "prompt-template",
                "grader-rule",
            )
        )


def test_plan_requires_contiguous_predecessor_shape() -> None:
    first = _plan()
    with pytest.raises(ValidationError, match="predecessor"):
        _plan(version=2)

    successor = _plan(
        version=2,
        predecessor=first.to_ref(),
    )
    assert successor.predecessor_plan_ref == first.to_ref()


def test_spec_requires_generator_judge_separation() -> None:
    plan = _plan()
    with pytest.raises(ValidationError, match="differ"):
        JudgeDesignSpecV2.create(
            design_spec_id="judge-design-spec://collision",
            plan_ref=plan.to_ref(),
            criteria_rubric_result_ref=(plan.criteria_rubric_result_ref),
            rubric_set_ref=plan.rubric_set_ref,
            evaluator_spec_ref=plan.evaluator_spec_ref,
            reference_policy_ref=plan.reference_policy_ref,
            tool_policy_ref=plan.tool_policy_ref,
            prompt_template_ref=plan.judge_prompt_template_ref,
            prompt_rendering_ref=_ref("prompt-rendering"),
            route_decision_ref=_ref("model-route-decision"),
            gateway_receipt_ref=_ref("gateway-receipt"),
            gateway_invocation_result_ref=_ref("gateway-invocation-result"),
            proposal_ref=_ref("judge-design-proposal"),
            selected_judge_model_profile_ref=(plan.generator_model_profile_ref),
            generator_model_profile_ref=(plan.generator_model_profile_ref),
            judge_tasks=plan.judge_tasks,
            aggregation_mode=plan.aggregation_mode,
            passing_score_basis_points=(plan.passing_score_basis_points),
            minimum_confidence_basis_points=(plan.minimum_confidence_basis_points),
            escalate_on_reference_unavailable=(plan.escalate_on_reference_unavailable),
            judge_input_schema_ref=plan.judge_input_schema_ref,
            judge_output_schema_ref=plan.judge_output_schema_ref,
            required_output_fields=plan.required_output_fields,
            reference_grant_refs=(),
            private_material_inventory_sha256=HASH,
            audit=_audit(),
        )


def test_result_requires_complete_success_or_no_current_spec() -> None:
    plan = _plan()
    spec = _spec(plan)
    validation = _validation(plan, spec)
    result = GradingDesignResultV2.create(
        result_id="grading-design-result://example",
        plan_ref=plan.to_ref(),
        criteria_rubric_result_ref=(plan.criteria_rubric_result_ref),
        route_decision_ref=spec.route_decision_ref,
        gateway_receipt_ref=spec.gateway_receipt_ref,
        gateway_invocation_result_ref=(spec.gateway_invocation_result_ref),
        proposal_ref=spec.proposal_ref,
        judge_design_spec_ref=spec.to_ref(),
        validation_ref=validation.to_ref(),
        outcome=GradingDesignOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=_audit(),
    )
    assert result.judge_design_spec_ref == spec.to_ref()

    with pytest.raises(ValidationError, match="cannot publish"):
        GradingDesignResultV2.create(
            result_id="grading-design-result://partial",
            plan_ref=plan.to_ref(),
            criteria_rubric_result_ref=(plan.criteria_rubric_result_ref),
            route_decision_ref=spec.route_decision_ref,
            gateway_receipt_ref=spec.gateway_receipt_ref,
            gateway_invocation_result_ref=(spec.gateway_invocation_result_ref),
            proposal_ref=spec.proposal_ref,
            judge_design_spec_ref=spec.to_ref(),
            validation_ref=validation.to_ref(),
            outcome=GradingDesignOutcomeV2.BLOCKED_POLICY,
            reason_codes=("REFERENCE_ACCESS_DENIED",),
            audit=_audit(),
        )

    blocked = GradingDesignResultV2.create(
        result_id="grading-design-result://blocked",
        plan_ref=plan.to_ref(),
        criteria_rubric_result_ref=(plan.criteria_rubric_result_ref),
        route_decision_ref=None,
        gateway_receipt_ref=None,
        gateway_invocation_result_ref=None,
        proposal_ref=None,
        judge_design_spec_ref=None,
        validation_ref=JudgeDesignValidationV2.create(
            validation_id="judge-design-validation://blocked",
            plan_ref=plan.to_ref(),
            design_spec_ref=None,
            criterion_ids=(),
            validated_output_fields=(),
            reference_access_result_refs=(),
            outcome=(JudgeDesignValidationOutcomeV2.BLOCKED_POLICY),
            reason_codes=("GENERATOR_JUDGE_COLLISION",),
            audit=_audit(),
        ).to_ref(),
        outcome=GradingDesignOutcomeV2.BLOCKED_POLICY,
        reason_codes=("GENERATOR_JUDGE_COLLISION",),
        audit=_audit(),
    )
    assert blocked.route_decision_ref is None
    assert blocked.judge_design_spec_ref is None
