export type PlanDecision =
  | "APPROVE"
  | "EDIT"
  | "REJECT"
  | "DEFER"
  | "REQUEST_MORE";

export type PlanReviewState =
  | "PENDING_REVIEW"
  | "APPROVED"
  | "REJECTED"
  | "DEFERRED"
  | "REVISION_REQUESTED"
  | "SUPERSEDED"
  | "RESUMED";

export interface ObjectRef {
  object_type: string;
  object_id: string;
  object_version: string;
  object_sha256: string;
}

export interface ContractAudit {
  schema_version: string;
  created_at: string;
  created_by: string;
  governing_versions: Array<{
    component: string;
    version: string;
    sha256: string;
  }>;
  input_refs: ObjectRef[];
}

export interface DatasetBuildPlanTask {
  schema_version: string;
  task_key: string;
  stage: string;
  task_kind: string;
  agent_role: string;
  dependency_task_keys: string[];
  input_object_types: string[];
  output_object_types: string[];
  required_capability_ids: string[];
  acceptance_check_refs: ObjectRef[];
  plan_review_kind: string | null;
  max_attempts: number;
  max_model_requests: number;
  max_model_tokens: number;
  max_cost_micro_usd: number;
}

export interface DatasetBuildPlan {
  schema_version: string;
  object_id: string;
  object_sha256: string;
  audit: ContractAudit;
  plan_id: string;
  run_ref: ObjectRef;
  plan_version: number;
  predecessor_plan_ref: ObjectRef | null;
  goals: string[];
  user_constraints: string[];
  assumptions: string[];
  unresolved_questions: string[];
  stage_order: string[];
  tasks: DatasetBuildPlanTask[];
  required_review_kinds: string[];
  total_model_requests: number;
  total_model_tokens: number;
  total_cost_micro_usd: number;
}

export interface AttachmentMockWork {
  schema_version: string;
  work_key: string;
  artifact_group_ref: ObjectRef;
  artifact_ids: string[];
  agent_role: string;
  dependency_work_keys: string[];
  input_object_types: string[];
  output_object_types: string[];
  required_capability_ids: string[];
  allowed_tool_ids: string[];
  data_purposes: string[];
  data_classifications: string[];
  workspace_policy_ref: ObjectRef;
  acceptance_check_refs: ObjectRef[];
  max_attempts: number;
  max_model_requests: number;
  max_model_tokens: number;
  max_cost_micro_usd: number;
}

export interface AttachmentGenerationPlan {
  schema_version: "eval-factory/attachment-generation-plan/v2";
  object_id: string;
  object_sha256: string;
  audit: ContractAudit;
  plan_id: string;
  run_ref: ObjectRef;
  plan_version: number;
  predecessor_plan_ref: ObjectRef | null;
  producer_task_view_ref: ObjectRef;
  evidence_bundle_ref: ObjectRef;
  attachment_planning_context_ref: ObjectRef;
  works: AttachmentMockWork[];
  max_parallel_groups: number;
  quality_policy_ref: ObjectRef;
  solvability_policy_ref: ObjectRef;
  total_model_requests: number;
  total_model_tokens: number;
  total_cost_micro_usd: number;
}

export type RubricJudgedObjectKind =
  | "CONTESTANT_RESPONSE"
  | "OUTPUT_ARTIFACT"
  | "WORKSPACE_STATE"
  | "TOOL_BEHAVIOR"
  | "STRUCTURED_VALUE";

export type ReferenceMode =
  | "NONE"
  | "STRUCTURED_EXPECTATIONS"
  | "PRIVATE_ANSWER"
  | "TRACE_BEHAVIOR"
  | "HUMAN_ONLY";

export interface CriteriaRubricGoal {
  schema_version: "eval-factory/criteria-rubric-goal/v2";
  goal_id: string;
  goal_summary: string;
  judged_object_kind: RubricJudgedObjectKind;
  prompt_requirement_ids: string[];
  attachment_dependency_ids: string[];
  allowed_tool_ids: string[];
  evaluator_binding_id: string;
  weight_basis_points: number;
  visibility: "EVALUATOR_ONLY";
}

export interface CriteriaRubricPlan {
  schema_version: "eval-factory/criteria-rubric-plan/v2";
  object_id: string;
  object_sha256: string;
  audit: ContractAudit;
  plan_id: string;
  run_ref: ObjectRef;
  plan_version: number;
  predecessor_plan_ref: ObjectRef | null;
  task_draft_ref: ObjectRef;
  attachment_quality_ref: ObjectRef;
  solvability_ref: ObjectRef;
  allowed_prompt_requirement_ids: string[];
  required_prompt_requirement_ids: string[];
  allowed_attachment_dependency_ids: string[];
  required_attachment_dependency_ids: string[];
  allowed_task_tool_ids: string[];
  required_task_tool_ids: string[];
  criterion_goals: CriteriaRubricGoal[];
  allowed_evaluator_binding_ids: string[];
  allowed_reference_modes: ReferenceMode[];
  selected_reference_mode: ReferenceMode;
  tool_catalog_ref: ObjectRef;
  agent_role: string;
  required_capability_ids: string[];
  specialist_tool_ids: string[];
  data_purpose: "criteria-rubric-authoring";
  data_classifications: string[];
  prompt_template_ref: ObjectRef;
  model_policy_ref: ObjectRef;
  acceptance_check_refs: ObjectRef[];
  max_attempts: number;
  max_model_requests: number;
  max_model_tokens: number;
  max_cost_micro_usd: number;
}

export type JudgeAggregationMode =
  | "WEIGHTED_SUM"
  | "ALL_CRITICAL";

export interface JudgeTaskMapping {
  schema_version: "eval-factory/judge-task-mapping/v2";
  task_key: string;
  criterion_ids: string[];
  evaluator_binding_id: string;
  score_weight_basis_points: number;
}

export interface GradingDesignPlan {
  schema_version: "eval-factory/grading-design-plan/v2";
  object_id: string;
  object_sha256: string;
  audit: ContractAudit;
  plan_id: string;
  run_ref: ObjectRef;
  plan_version: number;
  predecessor_plan_ref: ObjectRef | null;
  criteria_rubric_result_ref: ObjectRef;
  rubric_set_ref: ObjectRef;
  evaluator_spec_ref: ObjectRef;
  reference_policy_ref: ObjectRef;
  tool_policy_ref: ObjectRef;
  generator_model_profile_ref: ObjectRef;
  judge_tasks: JudgeTaskMapping[];
  aggregation_mode: JudgeAggregationMode;
  passing_score_basis_points: number;
  minimum_confidence_basis_points: number;
  escalate_on_reference_unavailable: boolean;
  judge_input_schema_ref: ObjectRef;
  judge_output_schema_ref: ObjectRef;
  required_output_fields: string[];
  allowed_judge_model_profile_refs: ObjectRef[];
  judge_prompt_template_ref: ObjectRef;
  agent_role: string;
  required_capability_ids: string[];
  specialist_tool_ids: string[];
  data_purpose: "grading-design-authoring";
  data_classifications: string[];
  model_policy_ref: ObjectRef;
  acceptance_check_refs: ObjectRef[];
  max_attempts: number;
  max_model_requests: number;
  max_model_tokens: number;
  max_cost_micro_usd: number;
}

export interface DatasetDeliveryPlan {
  schema_version: "eval-factory/dataset-delivery-plan/v2";
  object_id: string;
  object_sha256: string;
  audit: ContractAudit;
  plan_id: string;
  run_ref: ObjectRef;
  plan_version: number;
  predecessor_plan_ref: ObjectRef | null;
  aggregate_result_ref: ObjectRef;
  candidate_item_refs: ObjectRef[];
  candidate_projection_refs: ObjectRef[];
  rejected_binding_refs: ObjectRef[];
  blocked_binding_refs: ObjectRef[];
  output_target_ref: ObjectRef;
  max_files: number;
  max_total_bytes: number;
  production_release_allowed: false;
}

export type ReviewablePlan =
  | DatasetBuildPlan
  | AttachmentGenerationPlan
  | CriteriaRubricPlan
  | GradingDesignPlan
  | DatasetDeliveryPlan;

export interface PlanReviewRequest {
  review_request_id: string;
  plan_kind: string;
  plan_version: number;
  plan_ref: ObjectRef;
  requested_by: string;
}

export interface PlanReviewPresentation {
  title: string;
  summary_lines: string[];
  editable_paths: string[];
  warning_codes: string[];
}

export interface PlanReviewResult {
  state: PlanReviewState;
  plan_version: number;
  successor_plan_ref: ObjectRef | null;
  resume_token_ref: ObjectRef | null;
}

export interface PlanReviewView {
  schema_version: "eval-factory/plan-review-view/v1";
  run_id: string;
  current_run_ref: ObjectRef;
  request: PlanReviewRequest;
  presentation: PlanReviewPresentation;
  result: PlanReviewResult;
  plan: ReviewablePlan;
  decision: {
    decision: PlanDecision;
    reason_code: string;
    decided_by: string;
  } | null;
  revision: {
    changed_paths: string[];
    successor_plan_ref: ObjectRef;
  } | null;
}

export interface PlanReviewPage {
  schema_version: "eval-factory/plan-review-page/v1";
  items: PlanReviewView[];
  offset: number;
  limit: number;
  total: number;
}

export interface PlanReviewApiContract {
  schema_version: "eval-factory/plan-review-api-contract/v1";
  decision_actions: PlanDecision[];
  review_states: PlanReviewState[];
  principal_header: "X-Eval-Factory-Principal";
}

export interface GlobalPlanEditableFields {
  goals: string[];
  user_constraints: string[];
  assumptions: string[];
  unresolved_questions: string[];
  stage_order: string[];
  tasks: DatasetBuildPlanTask[];
  required_review_kinds: string[];
  total_model_requests: number;
  total_model_tokens: number;
  total_cost_micro_usd: number;
}

export interface AttachmentPlanEditableFields {
  works: AttachmentMockWork[];
  max_parallel_groups: number;
}

export interface CriteriaRubricPlanEditableFields {
  criterion_goals: CriteriaRubricGoal[];
  allowed_evaluator_binding_ids: string[];
  allowed_reference_modes: ReferenceMode[];
  selected_reference_mode: ReferenceMode;
  acceptance_check_refs: ObjectRef[];
  max_attempts: number;
  max_model_requests: number;
  max_model_tokens: number;
  max_cost_micro_usd: number;
}

export interface GradingDesignPlanEditableFields {
  judge_tasks: JudgeTaskMapping[];
  aggregation_mode: JudgeAggregationMode;
  passing_score_basis_points: number;
  minimum_confidence_basis_points: number;
  escalate_on_reference_unavailable: boolean;
  allowed_judge_model_profile_refs: ObjectRef[];
  acceptance_check_refs: ObjectRef[];
  max_attempts: number;
  max_model_requests: number;
  max_model_tokens: number;
  max_cost_micro_usd: number;
}

export interface DatasetDeliveryPlanEditableFields {
  max_files: number;
  max_total_bytes: number;
}

export type PlanEditableFields =
  | GlobalPlanEditableFields
  | AttachmentPlanEditableFields
  | CriteriaRubricPlanEditableFields
  | GradingDesignPlanEditableFields
  | DatasetDeliveryPlanEditableFields;

export function editableFields(plan: ReviewablePlan): PlanEditableFields {
  if (isAttachmentGenerationPlan(plan)) {
    return {
      works: plan.works,
      max_parallel_groups: plan.max_parallel_groups,
    };
  }
  if (isCriteriaRubricPlan(plan)) {
    return {
      criterion_goals: plan.criterion_goals,
      allowed_evaluator_binding_ids:
        plan.allowed_evaluator_binding_ids,
      allowed_reference_modes: plan.allowed_reference_modes,
      selected_reference_mode: plan.selected_reference_mode,
      acceptance_check_refs: plan.acceptance_check_refs,
      max_attempts: plan.max_attempts,
      max_model_requests: plan.max_model_requests,
      max_model_tokens: plan.max_model_tokens,
      max_cost_micro_usd: plan.max_cost_micro_usd,
    };
  }
  if (isGradingDesignPlan(plan)) {
    return {
      judge_tasks: plan.judge_tasks,
      aggregation_mode: plan.aggregation_mode,
      passing_score_basis_points:
        plan.passing_score_basis_points,
      minimum_confidence_basis_points:
        plan.minimum_confidence_basis_points,
      escalate_on_reference_unavailable:
        plan.escalate_on_reference_unavailable,
      allowed_judge_model_profile_refs:
        plan.allowed_judge_model_profile_refs,
      acceptance_check_refs: plan.acceptance_check_refs,
      max_attempts: plan.max_attempts,
      max_model_requests: plan.max_model_requests,
      max_model_tokens: plan.max_model_tokens,
      max_cost_micro_usd: plan.max_cost_micro_usd,
    };
  }
  if (isDatasetDeliveryPlan(plan)) {
    return {
      max_files: plan.max_files,
      max_total_bytes: plan.max_total_bytes,
    };
  }
  return {
    goals: plan.goals,
    user_constraints: plan.user_constraints,
    assumptions: plan.assumptions,
    unresolved_questions: plan.unresolved_questions,
    stage_order: plan.stage_order,
    tasks: plan.tasks,
    required_review_kinds: plan.required_review_kinds,
    total_model_requests: plan.total_model_requests,
    total_model_tokens: plan.total_model_tokens,
    total_cost_micro_usd: plan.total_cost_micro_usd,
  };
}

export function isAttachmentGenerationPlan(
  plan: ReviewablePlan,
): plan is AttachmentGenerationPlan {
  return (
    plan.schema_version ===
    "eval-factory/attachment-generation-plan/v2"
  );
}

export function isDatasetBuildPlan(
  plan: ReviewablePlan,
): plan is DatasetBuildPlan {
  return plan.schema_version === "eval-factory/dataset-build-plan/v2";
}

export function isCriteriaRubricPlan(
  plan: ReviewablePlan,
): plan is CriteriaRubricPlan {
  return (
    plan.schema_version ===
    "eval-factory/criteria-rubric-plan/v2"
  );
}

export function isGradingDesignPlan(
  plan: ReviewablePlan,
): plan is GradingDesignPlan {
  return (
    plan.schema_version ===
    "eval-factory/grading-design-plan/v2"
  );
}

export function isDatasetDeliveryPlan(
  plan: ReviewablePlan,
): plan is DatasetDeliveryPlan {
  return (
    plan.schema_version ===
    "eval-factory/dataset-delivery-plan/v2"
  );
}

export function isPlanReviewView(value: unknown): value is PlanReviewView {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  const result = record.result;
  const request = record.request;
  return (
    record.schema_version === "eval-factory/plan-review-view/v1" &&
    typeof record.run_id === "string" &&
    !!result &&
    typeof result === "object" &&
    typeof (result as Record<string, unknown>).state === "string" &&
    !!request &&
    typeof request === "object" &&
    typeof (request as Record<string, unknown>).review_request_id === "string"
  );
}

export function isPlanReviewPage(value: unknown): value is PlanReviewPage {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return (
    record.schema_version === "eval-factory/plan-review-page/v1" &&
    Array.isArray(record.items) &&
    record.items.every(isPlanReviewView) &&
    typeof record.total === "number"
  );
}
