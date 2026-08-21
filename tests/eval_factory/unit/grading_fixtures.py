from __future__ import annotations

from datetime import UTC, datetime

from eval_factory.agent_system.grading_registry import (
    GradingDesignAgentRegistryConfig,
    build_grading_design_agent_registry,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunPolicyV2,
    GradingDesignPlanV2,
    JudgeAggregationModeV2,
    JudgeTaskMappingV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)

HASH = "a" * 64


def ref(
    object_type: str,
    suffix: str = "grading",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=HASH,
    )


def audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by="grading-fixture",
        governing_versions=(
            VersionBinding(
                component="grading-design",
                version="v1",
            ),
        ),
    )


def registry():
    return build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=ref("prompt-template"),
            model_policy_ref=ref("model-routing-policy"),
        ),
        audit=audit(),
    )


def policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://grading",
        allowed_task_kinds=("grading-design",),
        max_transitions=100,
        max_plan_revisions=3,
        max_agent_attempts=3,
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=5_000_000,
        audit=audit(),
    )


def plan(
    *,
    run_ref: ObjectRef | None = None,
    criteria_rubric_result_ref: ObjectRef | None = None,
    rubric_set_ref: ObjectRef | None = None,
    evaluator_spec_ref: ObjectRef | None = None,
    reference_policy_ref: ObjectRef | None = None,
    tool_policy_ref: ObjectRef | None = None,
    generator_model_profile_ref: ObjectRef | None = None,
    judge_tasks: tuple[JudgeTaskMappingV2, ...] | None = None,
    allowed_judge_model_profile_refs: tuple[ObjectRef, ...] | None = None,
    minimum_confidence_basis_points: int = 8000,
    escalate_on_reference_unavailable: bool = True,
    required_capability_ids: tuple[str, ...] = ("agent-capability://grading-design",),
    specialist_tool_ids: tuple[str, ...] = ("reference-grant-read",),
    data_classifications: tuple[str, ...] = ("RESTRICTED_EVALUATOR_CONTROL",),
    prompt_template_ref: ObjectRef | None = None,
    model_policy_ref: ObjectRef | None = None,
    acceptance_check_refs: tuple[ObjectRef, ...] | None = None,
    max_model_tokens: int = 16_000,
) -> GradingDesignPlanV2:
    definition = registry().definitions[0]
    return GradingDesignPlanV2.create(
        plan_id="grading-design-plan://fixture",
        run_ref=run_ref or ref("factory-run"),
        plan_version=1,
        predecessor_plan_ref=None,
        criteria_rubric_result_ref=(criteria_rubric_result_ref or ref("criteria-rubric-result")),
        rubric_set_ref=rubric_set_ref or ref("rubric-set"),
        evaluator_spec_ref=(evaluator_spec_ref or ref("evaluator-spec")),
        reference_policy_ref=(reference_policy_ref or ref("reference-policy")),
        tool_policy_ref=tool_policy_ref or ref("tool-policy"),
        generator_model_profile_ref=(
            generator_model_profile_ref
            or ref(
                "model-capability-profile",
                "generator",
            )
        ),
        judge_tasks=judge_tasks
        or (
            JudgeTaskMappingV2(
                task_key="judge-main",
                criterion_ids=("rubric-criterion://main",),
                evaluator_binding_id="evaluator-binding://main",
                score_weight_basis_points=10_000,
            ),
        ),
        aggregation_mode=JudgeAggregationModeV2.WEIGHTED_SUM,
        passing_score_basis_points=7000,
        minimum_confidence_basis_points=(minimum_confidence_basis_points),
        escalate_on_reference_unavailable=(escalate_on_reference_unavailable),
        judge_input_schema_ref=ref(
            "json-schema",
            "judge-input",
        ),
        judge_output_schema_ref=ref(
            "json-schema",
            "judge-output",
        ),
        required_output_fields=(
            "ABSTAIN",
            "EVIDENCE_IDS",
            "FAILURE_CLASS",
            "SCORE_BASIS_POINTS",
        ),
        allowed_judge_model_profile_refs=(
            allowed_judge_model_profile_refs
            or (
                ref("model-capability-profile", "generator"),
                ref("model-capability-profile", "judge"),
            )
        ),
        judge_prompt_template_ref=(prompt_template_ref or definition.prompt_template_ref),
        agent_role=definition.agent_role,
        required_capability_ids=required_capability_ids,
        specialist_tool_ids=specialist_tool_ids,
        data_classifications=data_classifications,
        model_policy_ref=(model_policy_ref or definition.model_policy_ref),
        acceptance_check_refs=(
            acceptance_check_refs if acceptance_check_refs is not None else definition.validator_refs
        ),
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=max_model_tokens,
        max_cost_micro_usd=500_000,
        audit=audit(),
    )
