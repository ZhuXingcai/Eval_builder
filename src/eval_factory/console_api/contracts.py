from __future__ import annotations

from typing import Literal

from pydantic import Field

from eval_factory.agent_system.plan_review import (
    AttachmentPlanReviewEditSubmissionV1,
    CriteriaRubricPlanReviewEditSubmissionV1,
    DatasetDeliveryPlanReviewEditSubmissionV1,
    GradingDesignPlanReviewEditSubmissionV1,
    PlanReviewDecisionSubmissionV1,
    PlanReviewEditSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewViewV1,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    CriteriaRubricGoalV2,
    CriteriaRubricPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    GradingDesignPlanV2,
    JudgeAggregationModeV2,
    JudgeTaskMappingV2,
    PlanDecisionKindV2,
    PlanKindV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.dataset_runtime_v2 import (
    DatasetDeliveryPlanV2,
)
from eval_factory.contracts.task import ReferenceMode


class PlanReviewDecisionCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-decision-command/v1"] = (
        "eval-factory/plan-review-decision-command/v1"
    )
    review_id: str
    submission: PlanReviewDecisionSubmissionV1


class GlobalPlanEditableFieldsV1(ContractModelV2):
    schema_version: Literal["eval-factory/global-plan-editable-fields/v1"] = (
        "eval-factory/global-plan-editable-fields/v1"
    )
    goals: tuple[str, ...] = Field(min_length=1, max_length=128)
    user_constraints: tuple[str, ...] = Field(default=(), max_length=256)
    assumptions: tuple[str, ...] = Field(default=(), max_length=256)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=256)
    stage_order: tuple[str, ...] = Field(min_length=1, max_length=64)
    tasks: tuple[DatasetBuildPlanTaskV2, ...] = Field(min_length=1, max_length=100_000)
    required_review_kinds: tuple[PlanKindV2, ...] = Field(default=(), max_length=7)
    total_model_requests: int = Field(ge=0, le=10_000_000)
    total_model_tokens: int = Field(ge=0, le=10_000_000_000)
    total_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)


class AttachmentPlanEditableFieldsV1(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-plan-editable-fields/v1"] = (
        "eval-factory/attachment-plan-editable-fields/v1"
    )
    works: tuple[AttachmentMockWorkV2, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    max_parallel_groups: int = Field(ge=1, le=10_000)


class CriteriaRubricPlanEditableFieldsV1(ContractModelV2):
    schema_version: Literal["eval-factory/criteria-rubric-plan-editable-fields/v1"] = (
        "eval-factory/criteria-rubric-plan-editable-fields/v1"
    )
    criterion_goals: tuple[CriteriaRubricGoalV2, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    allowed_evaluator_binding_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    allowed_reference_modes: tuple[ReferenceMode, ...] = Field(
        min_length=1,
        max_length=5,
    )
    selected_reference_mode: ReferenceMode
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=1, le=1_000_000)
    max_model_tokens: int = Field(ge=1, le=1_000_000_000)
    max_cost_micro_usd: int = Field(ge=1, le=1_000_000_000_000)


class GradingDesignPlanEditableFieldsV1(ContractModelV2):
    schema_version: Literal["eval-factory/grading-design-plan-editable-fields/v1"] = (
        "eval-factory/grading-design-plan-editable-fields/v1"
    )
    judge_tasks: tuple[JudgeTaskMappingV2, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    aggregation_mode: JudgeAggregationModeV2
    passing_score_basis_points: int = Field(ge=1, le=10_000)
    minimum_confidence_basis_points: int = Field(
        ge=0,
        le=10_000,
    )
    escalate_on_reference_unavailable: bool
    allowed_judge_model_profile_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=1, le=1_000_000)
    max_model_tokens: int = Field(ge=1, le=1_000_000_000)
    max_cost_micro_usd: int = Field(
        ge=1,
        le=1_000_000_000_000,
    )


class DatasetDeliveryPlanEditableFieldsV1(ContractModelV2):
    schema_version: Literal["eval-factory/dataset-delivery-plan-editable-fields/v1"] = (
        "eval-factory/dataset-delivery-plan-editable-fields/v1"
    )
    max_files: int = Field(ge=1, le=10_000_000)
    max_total_bytes: int = Field(
        ge=1,
        le=10_000_000_000_000,
    )


class PlanReviewEditPayloadV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-edit-payload/v1"] = (
        "eval-factory/plan-review-edit-payload/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    edited_plan: (
        GlobalPlanEditableFieldsV1
        | AttachmentPlanEditableFieldsV1
        | CriteriaRubricPlanEditableFieldsV1
        | DatasetDeliveryPlanEditableFieldsV1
        | GradingDesignPlanEditableFieldsV1
    )
    changed_paths: tuple[str, ...] = Field(min_length=1, max_length=256)
    invalidated_object_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    decided_by: str = Field(min_length=3, max_length=256)
    reason_code: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=3, max_length=256)


class PlanReviewEditCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-edit-command/v1"] = (
        "eval-factory/plan-review-edit-command/v1"
    )
    review_id: str
    submission: PlanReviewEditPayloadV1


class PlanReviewResumeCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-resume-command/v1"] = (
        "eval-factory/plan-review-resume-command/v1"
    )
    review_id: str
    submission: PlanReviewResumeSubmissionV1


class PlanReviewApiContractV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-api-contract/v1"] = (
        "eval-factory/plan-review-api-contract/v1"
    )
    decision_actions: tuple[PlanDecisionKindV2, ...]
    review_states: tuple[PlanReviewStateV2, ...]
    principal_header: Literal["X-Eval-Factory-Principal"] = "X-Eval-Factory-Principal"


def plan_review_api_contract() -> PlanReviewApiContractV1:
    return PlanReviewApiContractV1(
        decision_actions=tuple(PlanDecisionKindV2),
        review_states=tuple(PlanReviewStateV2),
    )


def compile_plan_review_edit(
    view: PlanReviewViewV1,
    payload: PlanReviewEditPayloadV1,
    *,
    audit: ContractAudit,
) -> (
    PlanReviewEditSubmissionV1
    | AttachmentPlanReviewEditSubmissionV1
    | CriteriaRubricPlanReviewEditSubmissionV1
    | DatasetDeliveryPlanReviewEditSubmissionV1
    | GradingDesignPlanReviewEditSubmissionV1
):
    if payload.expected_plan_version != view.request.plan_version:
        raise ValueError("plan edit payload uses a stale plan version")
    editable = payload.edited_plan
    if view.request.plan_kind is PlanKindV2.ATTACHMENT_GENERATION:
        if not isinstance(
            view.plan,
            AttachmentGenerationPlanV2,
        ) or not isinstance(
            editable,
            AttachmentPlanEditableFieldsV1,
        ):
            raise ValueError("attachment plan edit payload has the wrong projection")
        attachment_successor = AttachmentGenerationPlanV2.create(
            plan_id=view.plan.plan_id,
            run_ref=view.current_run_ref,
            plan_version=view.plan.plan_version + 1,
            predecessor_plan_ref=view.plan.to_ref(),
            producer_task_view_ref=(view.plan.producer_task_view_ref),
            evidence_bundle_ref=view.plan.evidence_bundle_ref,
            attachment_planning_context_ref=(view.plan.attachment_planning_context_ref),
            works=editable.works,
            max_parallel_groups=editable.max_parallel_groups,
            quality_policy_ref=view.plan.quality_policy_ref,
            solvability_policy_ref=(view.plan.solvability_policy_ref),
            total_model_requests=sum(work.max_model_requests for work in editable.works),
            total_model_tokens=sum(work.max_model_tokens for work in editable.works),
            total_cost_micro_usd=sum(work.max_cost_micro_usd for work in editable.works),
            audit=audit,
        )
        return AttachmentPlanReviewEditSubmissionV1(
            expected_plan_version=(payload.expected_plan_version),
            edited_plan=attachment_successor,
            changed_paths=payload.changed_paths,
            decided_by=payload.decided_by,
            reason_code=payload.reason_code,
            idempotency_key=payload.idempotency_key,
        )
    if view.request.plan_kind is PlanKindV2.CRITERIA_RUBRIC:
        if not isinstance(
            view.plan,
            CriteriaRubricPlanV2,
        ) or not isinstance(
            editable,
            CriteriaRubricPlanEditableFieldsV1,
        ):
            raise ValueError("criteria/rubric plan edit payload has the wrong projection")
        criteria_successor = CriteriaRubricPlanV2.create(
            plan_id=view.plan.plan_id,
            run_ref=view.current_run_ref,
            plan_version=view.plan.plan_version + 1,
            predecessor_plan_ref=view.plan.to_ref(),
            task_draft_ref=view.plan.task_draft_ref,
            attachment_quality_ref=view.plan.attachment_quality_ref,
            solvability_ref=view.plan.solvability_ref,
            allowed_prompt_requirement_ids=(view.plan.allowed_prompt_requirement_ids),
            required_prompt_requirement_ids=(view.plan.required_prompt_requirement_ids),
            allowed_attachment_dependency_ids=(view.plan.allowed_attachment_dependency_ids),
            required_attachment_dependency_ids=(view.plan.required_attachment_dependency_ids),
            allowed_task_tool_ids=view.plan.allowed_task_tool_ids,
            required_task_tool_ids=view.plan.required_task_tool_ids,
            criterion_goals=editable.criterion_goals,
            allowed_evaluator_binding_ids=(editable.allowed_evaluator_binding_ids),
            allowed_reference_modes=editable.allowed_reference_modes,
            selected_reference_mode=editable.selected_reference_mode,
            tool_catalog_ref=view.plan.tool_catalog_ref,
            agent_role=view.plan.agent_role,
            required_capability_ids=view.plan.required_capability_ids,
            specialist_tool_ids=view.plan.specialist_tool_ids,
            data_classifications=view.plan.data_classifications,
            prompt_template_ref=view.plan.prompt_template_ref,
            model_policy_ref=view.plan.model_policy_ref,
            acceptance_check_refs=editable.acceptance_check_refs,
            max_attempts=editable.max_attempts,
            max_model_requests=editable.max_model_requests,
            max_model_tokens=editable.max_model_tokens,
            max_cost_micro_usd=editable.max_cost_micro_usd,
            audit=audit,
        )
        return CriteriaRubricPlanReviewEditSubmissionV1(
            expected_plan_version=payload.expected_plan_version,
            edited_plan=criteria_successor,
            changed_paths=payload.changed_paths,
            decided_by=payload.decided_by,
            reason_code=payload.reason_code,
            idempotency_key=payload.idempotency_key,
        )
    if view.request.plan_kind is PlanKindV2.GRADING_DESIGN:
        if not isinstance(
            view.plan,
            GradingDesignPlanV2,
        ) or not isinstance(
            editable,
            GradingDesignPlanEditableFieldsV1,
        ):
            raise ValueError("grading design edit payload has the wrong projection")
        grading_successor = GradingDesignPlanV2.create(
            plan_id=view.plan.plan_id,
            run_ref=view.current_run_ref,
            plan_version=view.plan.plan_version + 1,
            predecessor_plan_ref=view.plan.to_ref(),
            criteria_rubric_result_ref=(view.plan.criteria_rubric_result_ref),
            rubric_set_ref=view.plan.rubric_set_ref,
            evaluator_spec_ref=view.plan.evaluator_spec_ref,
            reference_policy_ref=(view.plan.reference_policy_ref),
            tool_policy_ref=view.plan.tool_policy_ref,
            generator_model_profile_ref=(view.plan.generator_model_profile_ref),
            judge_tasks=editable.judge_tasks,
            aggregation_mode=editable.aggregation_mode,
            passing_score_basis_points=(editable.passing_score_basis_points),
            minimum_confidence_basis_points=(editable.minimum_confidence_basis_points),
            escalate_on_reference_unavailable=(editable.escalate_on_reference_unavailable),
            judge_input_schema_ref=(view.plan.judge_input_schema_ref),
            judge_output_schema_ref=(view.plan.judge_output_schema_ref),
            required_output_fields=(view.plan.required_output_fields),
            allowed_judge_model_profile_refs=(editable.allowed_judge_model_profile_refs),
            judge_prompt_template_ref=(view.plan.judge_prompt_template_ref),
            agent_role=view.plan.agent_role,
            required_capability_ids=(view.plan.required_capability_ids),
            specialist_tool_ids=view.plan.specialist_tool_ids,
            data_classifications=(view.plan.data_classifications),
            model_policy_ref=view.plan.model_policy_ref,
            acceptance_check_refs=(editable.acceptance_check_refs),
            max_attempts=editable.max_attempts,
            max_model_requests=editable.max_model_requests,
            max_model_tokens=editable.max_model_tokens,
            max_cost_micro_usd=(editable.max_cost_micro_usd),
            audit=audit,
        )
        return GradingDesignPlanReviewEditSubmissionV1(
            expected_plan_version=payload.expected_plan_version,
            edited_plan=grading_successor,
            changed_paths=payload.changed_paths,
            decided_by=payload.decided_by,
            reason_code=payload.reason_code,
            idempotency_key=payload.idempotency_key,
        )
    if view.request.plan_kind is PlanKindV2.FINAL_DELIVERY:
        if not isinstance(
            view.plan,
            DatasetDeliveryPlanV2,
        ) or not isinstance(
            editable,
            DatasetDeliveryPlanEditableFieldsV1,
        ):
            raise ValueError("dataset delivery edit payload has the wrong projection")
        delivery_successor = DatasetDeliveryPlanV2.create(
            plan_id=view.plan.plan_id,
            run_ref=view.current_run_ref,
            plan_version=view.plan.plan_version + 1,
            predecessor_plan_ref=view.plan.to_ref(),
            aggregate_result_ref=(view.plan.aggregate_result_ref),
            candidate_item_refs=(view.plan.candidate_item_refs),
            candidate_projection_refs=(view.plan.candidate_projection_refs),
            candidate_stage_head_refs=(view.plan.candidate_stage_head_refs),
            rejected_binding_refs=(view.plan.rejected_binding_refs),
            blocked_binding_refs=(view.plan.blocked_binding_refs),
            output_target_ref=view.plan.output_target_ref,
            max_files=editable.max_files,
            max_total_bytes=editable.max_total_bytes,
            audit=audit,
        )
        return DatasetDeliveryPlanReviewEditSubmissionV1(
            expected_plan_version=(payload.expected_plan_version),
            edited_plan=delivery_successor,
            changed_paths=payload.changed_paths,
            decided_by=payload.decided_by,
            reason_code=payload.reason_code,
            idempotency_key=payload.idempotency_key,
        )
    if (
        view.request.plan_kind is not PlanKindV2.GLOBAL_BUILD
        or not isinstance(view.plan, DatasetBuildPlanV2)
        or not isinstance(editable, GlobalPlanEditableFieldsV1)
    ):
        raise ValueError("global plan edit payload has the wrong projection")
    global_successor = DatasetBuildPlanV2.create(
        plan_id=view.plan.plan_id,
        run_ref=view.current_run_ref,
        plan_version=view.plan.plan_version + 1,
        predecessor_plan_ref=view.plan.to_ref(),
        goals=editable.goals,
        user_constraints=editable.user_constraints,
        assumptions=editable.assumptions,
        unresolved_questions=editable.unresolved_questions,
        stage_order=editable.stage_order,
        tasks=editable.tasks,
        required_review_kinds=editable.required_review_kinds,
        total_model_requests=editable.total_model_requests,
        total_model_tokens=editable.total_model_tokens,
        total_cost_micro_usd=editable.total_cost_micro_usd,
        audit=audit,
    )
    return PlanReviewEditSubmissionV1(
        expected_plan_version=payload.expected_plan_version,
        edited_plan=global_successor,
        changed_paths=payload.changed_paths,
        invalidated_object_refs=payload.invalidated_object_refs,
        decided_by=payload.decided_by,
        reason_code=payload.reason_code,
        idempotency_key=payload.idempotency_key,
    )


__all__ = [
    "AttachmentPlanEditableFieldsV1",
    "CriteriaRubricPlanEditableFieldsV1",
    "DatasetDeliveryPlanEditableFieldsV1",
    "GlobalPlanEditableFieldsV1",
    "GradingDesignPlanEditableFieldsV1",
    "PlanReviewApiContractV1",
    "PlanReviewDecisionCommandV1",
    "PlanReviewEditCommandV1",
    "PlanReviewEditPayloadV1",
    "PlanReviewResumeCommandV1",
    "compile_plan_review_edit",
    "plan_review_api_contract",
]
