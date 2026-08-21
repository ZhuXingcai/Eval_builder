from __future__ import annotations

from pathlib import Path

import pytest
from grading_fixtures import (
    audit,
)
from grading_fixtures import (
    plan as grading_plan,
)
from grading_fixtures import (
    registry as grading_registry,
)
from test_domain_plan_review import (
    RUN_ID,
    USER,
)
from test_domain_plan_review import (
    _registry as domain_registry,
)
from test_domain_plan_review import (
    _setup as domain_setup,
)

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
)
from eval_factory.agent_system.plan_adapters import (
    AttachmentGenerationPlanAdapter,
    GlobalBuildPlanAdapter,
    GradingDesignPlanAdapter,
    ReviewablePlanAdapterError,
    ReviewablePlanAdapterRegistry,
)
from eval_factory.agent_system.plan_review import (
    GradingDesignPlanReviewEditSubmissionV1,
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.planner import (
    DatasetBuildPlanCompiler,
)
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.contracts.agent_system_v2 import (
    PlanDecisionKindV2,
    PlanKindV2,
    PlanReviewStateV2,
)


def _service(
    tmp_path: Path,
) -> tuple[PlanReviewService, GradingDesignPlanCompiler]:
    _, store, _ = domain_setup(tmp_path)
    domain = domain_registry()
    grading = grading_registry()
    registry = AgentRegistry(
        capabilities=(
            *domain.capabilities,
            *grading.capabilities,
        ),
        definitions=(
            *domain.definitions,
            *grading.definitions,
        ),
    )
    global_compiler = DatasetBuildPlanCompiler(registry)
    attachment_compiler = AttachmentGenerationPlanCompiler(registry)
    grading_compiler = GradingDesignPlanCompiler(registry)
    service = PlanReviewService(
        store,
        compiler=global_compiler,
        adapter_registry=ReviewablePlanAdapterRegistry(
            (
                GlobalBuildPlanAdapter(global_compiler),
                AttachmentGenerationPlanAdapter(attachment_compiler),
                GradingDesignPlanAdapter(grading_compiler),
            )
        ),
    )
    return service, grading_compiler


def _commit(
    service: PlanReviewService,
    compiler: GradingDesignPlanCompiler,
):
    run = service.store.get_run(RUN_ID)
    source = grading_plan(run_ref=run.to_ref())
    compiled = compiler.compile(
        plan=source,
        policy=service.store.get_policy(run.policy_ref.object_id),
        audit=audit(),
    )
    service.store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.GRADING_DESIGN,
        plan=source,
        compiled_plan=compiled,
        audit=audit(),
        idempotency_key="commit-grading-plan",
    )
    return source


def _open(
    service: PlanReviewService,
):
    return service.open_plan(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.GRADING_DESIGN,
        requested_by=USER,
        idempotency_key="open-grading-review",
        audit=audit(),
    )


def test_grading_plan_approve_and_resume_uses_shared_state_machine(
    tmp_path: Path,
) -> None:
    service, compiler = _service(tmp_path)
    source = _commit(service, compiler)
    opened = _open(service)

    approved = service.decide(
        opened.request.review_request_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=source.plan_version,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="GRADING_PLAN_APPROVED",
            idempotency_key="approve-grading-plan",
        ),
        audit=audit(),
    )
    resumed = service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=source.plan_version,
            resumed_by=USER,
            idempotency_key="resume-grading-plan",
        ),
        audit=audit(),
    )

    assert approved.result.state is PlanReviewStateV2.APPROVED
    assert resumed.result.state is PlanReviewStateV2.RESUMED
    assert "judge_tasks" in opened.presentation.editable_paths


def test_grading_plan_edit_compiles_narrow_successor(
    tmp_path: Path,
) -> None:
    service, compiler = _service(tmp_path)
    source = _commit(service, compiler)
    opened = _open(service)
    run = service.store.get_run(RUN_ID)
    successor = type(source).create(
        plan_id=source.plan_id,
        run_ref=run.to_ref(),
        plan_version=2,
        predecessor_plan_ref=source.to_ref(),
        criteria_rubric_result_ref=(source.criteria_rubric_result_ref),
        rubric_set_ref=source.rubric_set_ref,
        evaluator_spec_ref=source.evaluator_spec_ref,
        reference_policy_ref=source.reference_policy_ref,
        tool_policy_ref=source.tool_policy_ref,
        generator_model_profile_ref=(source.generator_model_profile_ref),
        judge_tasks=source.judge_tasks,
        aggregation_mode=source.aggregation_mode,
        passing_score_basis_points=7500,
        minimum_confidence_basis_points=8500,
        escalate_on_reference_unavailable=True,
        judge_input_schema_ref=source.judge_input_schema_ref,
        judge_output_schema_ref=source.judge_output_schema_ref,
        required_output_fields=source.required_output_fields,
        allowed_judge_model_profile_refs=(source.allowed_judge_model_profile_refs[1:]),
        judge_prompt_template_ref=(source.judge_prompt_template_ref),
        agent_role=source.agent_role,
        required_capability_ids=(source.required_capability_ids),
        specialist_tool_ids=source.specialist_tool_ids,
        data_classifications=source.data_classifications,
        model_policy_ref=source.model_policy_ref,
        acceptance_check_refs=source.acceptance_check_refs,
        max_attempts=source.max_attempts,
        max_model_requests=source.max_model_requests,
        max_model_tokens=8000,
        max_cost_micro_usd=source.max_cost_micro_usd,
        audit=audit(),
    )

    edited = service.edit_plan(
        opened.request.review_request_id,
        GradingDesignPlanReviewEditSubmissionV1(
            expected_plan_version=1,
            edited_plan=successor,
            changed_paths=(
                "allowed_judge_model_profile_refs",
                "max_model_tokens",
                "minimum_confidence_basis_points",
                "passing_score_basis_points",
            ),
            decided_by=USER,
            reason_code="REFINE_GRADING_PLAN",
            idempotency_key="edit-grading-plan",
        ),
        audit=audit(),
    )
    resumed = service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-edited-grading-plan",
        ),
        audit=audit(),
    )

    assert edited.result.state is (PlanReviewStateV2.REVISION_REQUESTED)
    assert resumed.plan == successor
    assert resumed.result.state is PlanReviewStateV2.RESUMED


def test_grading_plan_edit_rejects_model_scope_widening(
    tmp_path: Path,
) -> None:
    service, compiler = _service(tmp_path)
    source = _commit(service, compiler)
    widened = source.model_copy(
        update={
            "plan_version": 2,
            "predecessor_plan_ref": source.to_ref(),
            "allowed_judge_model_profile_refs": (
                *source.allowed_judge_model_profile_refs,
                source.generator_model_profile_ref.model_copy(
                    update={"object_id": ("model-capability-profile://extra")}
                ),
            ),
        }
    )

    with pytest.raises(
        ReviewablePlanAdapterError,
        match="widens judge model",
    ):
        GradingDesignPlanAdapter(compiler).validate_successor(
            source,
            widened,
        )
