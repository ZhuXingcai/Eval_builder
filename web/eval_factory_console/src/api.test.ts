import { afterEach, describe, expect, it, vi } from "vitest";

import openapi from "./generated/openapi.json";
import {
  decidePlanReview,
  editPlanReview,
  listPlanReviews,
} from "./api";
import {
  editableFields,
  type AttachmentGenerationPlan,
  type CriteriaRubricPlan,
  type DatasetDeliveryPlan,
  type GradingDesignPlan,
  type PlanReviewView,
} from "./types";

const ref = {
  object_type: "dataset-build-plan",
  object_id: "dataset-build-plan://sha256/example",
  object_version: "v2",
  object_sha256: "a".repeat(64),
};

const audit = {
  schema_version: "eval-factory/contract-audit/v1",
  created_at: "2026-08-06T00:00:00Z",
  created_by: "test",
  governing_versions: [],
  input_refs: [],
};

const view: PlanReviewView = {
  schema_version: "eval-factory/plan-review-view/v1",
  run_id: "factory-run://example",
  current_run_ref: { ...ref, object_type: "factory-run" },
  request: {
    review_request_id: "plan-review-request://example/1",
    plan_kind: "GLOBAL_BUILD",
    plan_version: 1,
    plan_ref: ref,
    requested_by: "user://plan-owner",
  },
  presentation: {
    title: "Review plan",
    summary_lines: ["One task"],
    editable_paths: ["goals"],
    warning_codes: [],
  },
  result: {
    state: "PENDING_REVIEW",
    plan_version: 1,
    successor_plan_ref: null,
    resume_token_ref: null,
  },
  plan: {
    schema_version: "eval-factory/dataset-build-plan/v2",
    object_id: ref.object_id,
    object_sha256: ref.object_sha256,
    audit,
    plan_id: "dataset-build-plan://example",
    run_ref: { ...ref, object_type: "factory-run" },
    plan_version: 1,
    predecessor_plan_ref: null,
    goals: ["Build source-grounded tasks."],
    user_constraints: [],
    assumptions: [],
    unresolved_questions: [],
    stage_order: ["core"],
    tasks: [],
    required_review_kinds: ["GLOBAL_BUILD"],
    total_model_requests: 0,
    total_model_tokens: 0,
    total_cost_micro_usd: 0,
  },
  decision: null,
  revision: null,
};

const attachmentPlan: AttachmentGenerationPlan = {
  schema_version: "eval-factory/attachment-generation-plan/v2",
  object_id: "attachment-generation-plan://sha256/example",
  object_sha256: ref.object_sha256,
  audit,
  plan_id: "attachment-generation-plan://example",
  run_ref: { ...ref, object_type: "factory-run" },
  plan_version: 1,
  predecessor_plan_ref: null,
  producer_task_view_ref: { ...ref, object_type: "producer-task-view" },
  evidence_bundle_ref: {
    ...ref,
    object_type: "evidence-bundle",
    object_version: "v1",
  },
  attachment_planning_context_ref: {
    ...ref,
    object_type: "attachment-planning-context",
  },
  works: [
    {
      schema_version: "eval-factory/attachment-mock-work/v2",
      work_key: "work-a",
      artifact_group_ref: {
        ...ref,
        object_type: "artifact-execution-group",
      },
      artifact_ids: ["artifact-a"],
      agent_role: "attachment-mock-agent",
      dependency_work_keys: [],
      input_object_types: ["attachment-planning-context"],
      output_object_types: ["attachment-group-result"],
      required_capability_ids: ["agent-capability://attachment-mock"],
      allowed_tool_ids: ["attachment-execution"],
      data_purposes: ["attachment-production"],
      data_classifications: ["RESTRICTED_TRACE_DERIVED"],
      workspace_policy_ref: {
        ...ref,
        object_type: "agent-workspace-policy",
      },
      acceptance_check_refs: [
        { ...ref, object_type: "acceptance-check" },
      ],
      max_attempts: 2,
      max_model_requests: 2,
      max_model_tokens: 4096,
      max_cost_micro_usd: 100_000,
    },
  ],
  max_parallel_groups: 1,
  quality_policy_ref: {
    ...ref,
    object_type: "attachment-quality-policy",
  },
  solvability_policy_ref: {
    ...ref,
    object_type: "solvability-policy",
  },
  total_model_requests: 2,
  total_model_tokens: 4096,
  total_cost_micro_usd: 100_000,
};

const criteriaPlan: CriteriaRubricPlan = {
  schema_version: "eval-factory/criteria-rubric-plan/v2",
  object_id: "criteria-rubric-plan://sha256/example",
  object_sha256: ref.object_sha256,
  audit,
  plan_id: "criteria-rubric-plan://example",
  run_ref: { ...ref, object_type: "factory-run" },
  plan_version: 1,
  predecessor_plan_ref: null,
  task_draft_ref: { ...ref, object_type: "task-draft" },
  attachment_quality_ref: {
    ...ref,
    object_type: "attachment-quality-assessment",
  },
  solvability_ref: {
    ...ref,
    object_type: "solvability-assessment",
  },
  allowed_prompt_requirement_ids: ["requirement://visible"],
  required_prompt_requirement_ids: ["requirement://visible"],
  allowed_attachment_dependency_ids: ["dependency://input"],
  required_attachment_dependency_ids: ["dependency://input"],
  allowed_task_tool_ids: ["file-read"],
  required_task_tool_ids: [],
  criterion_goals: [
    {
      schema_version: "eval-factory/criteria-rubric-goal/v2",
      goal_id: "criteria-goal://workspace",
      goal_summary: "Judge the requested workspace state.",
      judged_object_kind: "WORKSPACE_STATE",
      prompt_requirement_ids: ["requirement://visible"],
      attachment_dependency_ids: ["dependency://input"],
      allowed_tool_ids: [],
      evaluator_binding_id: "evaluator-binding://workspace",
      weight_basis_points: 10_000,
      visibility: "EVALUATOR_ONLY",
    },
  ],
  allowed_evaluator_binding_ids: [
    "evaluator-binding://workspace",
  ],
  allowed_reference_modes: ["NONE"],
  selected_reference_mode: "NONE",
  tool_catalog_ref: {
    ...ref,
    object_type: "tool-capability-catalog",
    object_version: "tool-capability-catalog/r4-07-v1",
  },
  agent_role: "criteria-rubric-agent",
  required_capability_ids: [
    "agent-capability://criteria-rubric",
  ],
  specialist_tool_ids: ["rubric-candidate-read"],
  data_purpose: "criteria-rubric-authoring",
  data_classifications: [
    "RESTRICTED_EVALUATOR_CONTROL",
  ],
  prompt_template_ref: {
    ...ref,
    object_type: "prompt-template",
  },
  model_policy_ref: {
    ...ref,
    object_type: "model-routing-policy",
  },
  acceptance_check_refs: [
    { ...ref, object_type: "validator" },
  ],
  max_attempts: 2,
  max_model_requests: 1,
  max_model_tokens: 16_000,
  max_cost_micro_usd: 500_000,
};

const gradingPlan: GradingDesignPlan = {
  schema_version: "eval-factory/grading-design-plan/v2",
  object_id: "grading-design-plan://sha256/example",
  object_sha256: ref.object_sha256,
  audit,
  plan_id: "grading-design-plan://example",
  run_ref: { ...ref, object_type: "factory-run" },
  plan_version: 1,
  predecessor_plan_ref: null,
  criteria_rubric_result_ref: {
    ...ref,
    object_type: "criteria-rubric-result",
  },
  rubric_set_ref: { ...ref, object_type: "rubric-set" },
  evaluator_spec_ref: {
    ...ref,
    object_type: "evaluator-spec",
  },
  reference_policy_ref: {
    ...ref,
    object_type: "reference-policy",
  },
  tool_policy_ref: { ...ref, object_type: "tool-policy" },
  generator_model_profile_ref: {
    ...ref,
    object_type: "model-capability-profile",
  },
  judge_tasks: [
    {
      schema_version: "eval-factory/judge-task-mapping/v2",
      task_key: "judge-main",
      criterion_ids: ["rubric-criterion://main"],
      evaluator_binding_id: "evaluator-binding://main",
      score_weight_basis_points: 10_000,
    },
  ],
  aggregation_mode: "WEIGHTED_SUM",
  passing_score_basis_points: 7000,
  minimum_confidence_basis_points: 8000,
  escalate_on_reference_unavailable: true,
  judge_input_schema_ref: {
    ...ref,
    object_type: "json-schema",
    object_id: "json-schema://judge-input",
  },
  judge_output_schema_ref: {
    ...ref,
    object_type: "json-schema",
    object_id: "json-schema://judge-output",
  },
  required_output_fields: [
    "ABSTAIN",
    "EVIDENCE_IDS",
    "FAILURE_CLASS",
    "SCORE_BASIS_POINTS",
  ],
  allowed_judge_model_profile_refs: [
    {
      ...ref,
      object_type: "model-capability-profile",
      object_id: "model-capability-profile://judge",
    },
  ],
  judge_prompt_template_ref: {
    ...ref,
    object_type: "prompt-template",
  },
  agent_role: "grading-design-agent",
  required_capability_ids: [
    "agent-capability://grading-design",
  ],
  specialist_tool_ids: ["reference-grant-read"],
  data_purpose: "grading-design-authoring",
  data_classifications: [
    "RESTRICTED_EVALUATOR_CONTROL",
  ],
  model_policy_ref: {
    ...ref,
    object_type: "model-routing-policy",
  },
  acceptance_check_refs: [
    { ...ref, object_type: "validator" },
  ],
  max_attempts: 2,
  max_model_requests: 1,
  max_model_tokens: 16_000,
  max_cost_micro_usd: 500_000,
};

const deliveryPlan: DatasetDeliveryPlan = {
  schema_version: "eval-factory/dataset-delivery-plan/v2",
  object_id: "dataset-delivery-plan://sha256/example",
  object_sha256: ref.object_sha256,
  audit,
  plan_id: "dataset-delivery-plan://example",
  run_ref: { ...ref, object_type: "factory-run" },
  plan_version: 1,
  predecessor_plan_ref: null,
  aggregate_result_ref: {
    ...ref,
    object_type: "factory-dataset-aggregate-result",
  },
  candidate_item_refs: [
    { ...ref, object_type: "evaluation-item" },
  ],
  candidate_projection_refs: [
    { ...ref, object_type: "release-projection-result" },
  ],
  rejected_binding_refs: [],
  blocked_binding_refs: [],
  output_target_ref: {
    ...ref,
    object_type: "candidate-output-target",
  },
  max_files: 100,
  max_total_bytes: 1_000_000,
  production_release_allowed: false,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("shared API contract", () => {
  it("contains every frontend plan-review operation", () => {
    expect(openapi.paths["/api/plan-reviews"]).toHaveProperty("get");
    expect(openapi.paths["/api/plan-reviews/show"]).toHaveProperty("get");
    expect(openapi.paths["/api/plan-reviews/export"]).toHaveProperty("get");
    expect(openapi.paths["/api/plan-reviews/decision"]).toHaveProperty("post");
    expect(openapi.paths["/api/plan-reviews/edit"]).toHaveProperty("post");
    expect(openapi.paths["/api/plan-reviews/resume"]).toHaveProperty("post");
  });

  it("decodes list responses through the shared view guard", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            schema_version: "eval-factory/plan-review-page/v1",
            items: [view],
            offset: 0,
            limit: 100,
            total: 1,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    await expect(listPlanReviews()).resolves.toMatchObject({ total: 1 });
  });

  it("sends direct decisions with principal and expected version", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
      return new Response(
        JSON.stringify({
          ...view,
          result: { ...view.result, state: "APPROVED" },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await decidePlanReview(
      view,
      "APPROVE",
      "user://plan-owner",
      "APPROVE_BY_TEST",
    );
    expect(result.result.state).toBe("APPROVED");
    const init = fetchMock.mock.calls[0]?.[1];
    expect(new Headers(init?.headers).get("X-Eval-Factory-Principal")).toBe(
      "user://plan-owner",
    );
    expect(String(init?.body)).toContain('"expected_plan_version":1');
  });

  it("projects and sends only attachment-owned editable fields", async () => {
    const attachmentView: PlanReviewView = {
      ...view,
      request: {
        ...view.request,
        plan_kind: "ATTACHMENT_GENERATION",
        plan_ref: {
          ...ref,
          object_type: "attachment-generation-plan",
        },
      },
      plan: attachmentPlan,
    };
    const projection = editableFields(attachmentPlan);
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify(attachmentView), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await editPlanReview(
      attachmentView,
      projection,
      ["max_parallel_groups"],
      "user://plan-owner",
    );

    expect(projection).toEqual({
      works: attachmentPlan.works,
      max_parallel_groups: 1,
    });
    const body = String(fetchMock.mock.calls[0]?.[1]?.body);
    expect(body).toContain('"max_parallel_groups":1');
    expect(body).not.toContain('"producer_task_view_ref"');
  });

  it("projects and sends only criteria-owned editable fields", async () => {
    const criteriaView: PlanReviewView = {
      ...view,
      request: {
        ...view.request,
        plan_kind: "CRITERIA_RUBRIC",
        plan_ref: {
          ...ref,
          object_type: "criteria-rubric-plan",
        },
      },
      plan: criteriaPlan,
    };
    const projection = editableFields(criteriaPlan);
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify(criteriaView), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await editPlanReview(
      criteriaView,
      projection,
      ["criterion_goals", "max_model_tokens"],
      "user://plan-owner",
    );

    expect(projection).toEqual({
      criterion_goals: criteriaPlan.criterion_goals,
      allowed_evaluator_binding_ids:
        criteriaPlan.allowed_evaluator_binding_ids,
      allowed_reference_modes:
        criteriaPlan.allowed_reference_modes,
      selected_reference_mode:
        criteriaPlan.selected_reference_mode,
      acceptance_check_refs:
        criteriaPlan.acceptance_check_refs,
      max_attempts: 2,
      max_model_requests: 1,
      max_model_tokens: 16_000,
      max_cost_micro_usd: 500_000,
    });
    const body = String(fetchMock.mock.calls[0]?.[1]?.body);
    expect(body).toContain('"criterion_goals"');
    expect(body).not.toContain('"task_draft_ref"');
    expect(body).not.toContain('"attachment_quality_ref"');
    expect(body).not.toContain('"tool_catalog_ref"');
  });

  it("projects and sends only grading-owned editable fields", async () => {
    const gradingView: PlanReviewView = {
      ...view,
      request: {
        ...view.request,
        plan_kind: "GRADING_DESIGN",
        plan_ref: {
          ...ref,
          object_type: "grading-design-plan",
        },
      },
      plan: gradingPlan,
    };
    const projection = editableFields(gradingPlan);
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify(gradingView), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await editPlanReview(
      gradingView,
      projection,
      ["minimum_confidence_basis_points"],
      "user://plan-owner",
    );

    expect(projection).toEqual({
      judge_tasks: gradingPlan.judge_tasks,
      aggregation_mode: "WEIGHTED_SUM",
      passing_score_basis_points: 7000,
      minimum_confidence_basis_points: 8000,
      escalate_on_reference_unavailable: true,
      allowed_judge_model_profile_refs:
        gradingPlan.allowed_judge_model_profile_refs,
      acceptance_check_refs:
        gradingPlan.acceptance_check_refs,
      max_attempts: 2,
      max_model_requests: 1,
      max_model_tokens: 16_000,
      max_cost_micro_usd: 500_000,
    });
    const body = String(fetchMock.mock.calls[0]?.[1]?.body);
    expect(body).toContain('"judge_tasks"');
    expect(body).not.toContain('"criteria_rubric_result_ref"');
    expect(body).not.toContain('"rubric_set_ref"');
    expect(body).not.toContain('"evaluator_spec_ref"');
    expect(body).not.toContain('"reference_policy_ref"');
    expect(body).not.toContain('"tool_policy_ref"');
    expect(body).not.toContain('"generator_model_profile_ref"');
    expect(body).not.toContain('"judge_input_schema_ref"');
    expect(body).not.toContain('"judge_output_schema_ref"');
    expect(body).not.toContain('"judge_prompt_template_ref"');
    expect(body).not.toContain('"model_policy_ref"');
  });

  it("projects and sends only delivery-owned editable fields", async () => {
    const deliveryView: PlanReviewView = {
      ...view,
      request: {
        ...view.request,
        plan_kind: "FINAL_DELIVERY",
        plan_ref: {
          ...ref,
          object_type: "dataset-delivery-plan",
        },
      },
      plan: deliveryPlan,
    };
    const projection = editableFields(deliveryPlan);
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify(deliveryView), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await editPlanReview(
      deliveryView,
      projection,
      ["max_files", "max_total_bytes"],
      "user://plan-owner",
    );

    expect(projection).toEqual({
      max_files: 100,
      max_total_bytes: 1_000_000,
    });
    const body = String(fetchMock.mock.calls[0]?.[1]?.body);
    expect(body).toContain('"max_files":100');
    expect(body).toContain('"max_total_bytes":1000000');
    expect(body).not.toContain('"aggregate_result_ref"');
    expect(body).not.toContain('"candidate_item_refs"');
    expect(body).not.toContain('"candidate_projection_refs"');
    expect(body).not.toContain('"output_target_ref"');
    expect(body).not.toContain('"production_release_allowed"');
  });
});
