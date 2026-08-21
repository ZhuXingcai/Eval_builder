import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type {
  ContractAudit,
  GradingDesignPlan,
  ObjectRef,
  PlanReviewView,
} from "./types";

const objectRef = (
  objectType: string,
  suffix: string,
): ObjectRef => ({
  object_type: objectType,
  object_id: `${objectType}://${suffix}`,
  object_version: "v2",
  object_sha256: "a".repeat(64),
});

const audit: ContractAudit = {
  schema_version: "eval-factory/contract-audit/v1",
  created_at: "2026-08-06T00:00:00Z",
  created_by: "grading-web-test",
  governing_versions: [],
  input_refs: [],
};

const gradingPlan: GradingDesignPlan = {
  schema_version: "eval-factory/grading-design-plan/v2",
  object_id: "grading-design-plan://sha256/example",
  object_sha256: "a".repeat(64),
  audit,
  plan_id: "grading-design-plan://example",
  run_ref: objectRef("factory-run", "example"),
  plan_version: 1,
  predecessor_plan_ref: null,
  criteria_rubric_result_ref: objectRef(
    "criteria-rubric-result",
    "example",
  ),
  rubric_set_ref: objectRef("rubric-set", "example"),
  evaluator_spec_ref: objectRef("evaluator-spec", "example"),
  reference_policy_ref: objectRef(
    "reference-policy",
    "example",
  ),
  tool_policy_ref: objectRef("tool-policy", "example"),
  generator_model_profile_ref: objectRef(
    "model-capability-profile",
    "generator",
  ),
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
  judge_input_schema_ref: objectRef(
    "json-schema",
    "judge-input",
  ),
  judge_output_schema_ref: objectRef(
    "json-schema",
    "judge-output",
  ),
  required_output_fields: [
    "ABSTAIN",
    "EVIDENCE_IDS",
    "FAILURE_CLASS",
    "SCORE_BASIS_POINTS",
  ],
  allowed_judge_model_profile_refs: [
    objectRef("model-capability-profile", "judge"),
  ],
  judge_prompt_template_ref: objectRef(
    "prompt-template",
    "grading-design",
  ),
  agent_role: "grading-design-agent",
  required_capability_ids: [
    "agent-capability://grading-design",
  ],
  specialist_tool_ids: ["reference-grant-read"],
  data_purpose: "grading-design-authoring",
  data_classifications: [
    "RESTRICTED_EVALUATOR_CONTROL",
  ],
  model_policy_ref: objectRef(
    "model-routing-policy",
    "grading-design",
  ),
  acceptance_check_refs: [
    objectRef("validator", "grading-design"),
  ],
  max_attempts: 2,
  max_model_requests: 1,
  max_model_tokens: 16_000,
  max_cost_micro_usd: 500_000,
};

const review: PlanReviewView = {
  schema_version: "eval-factory/plan-review-view/v1",
  run_id: "factory-run://example",
  current_run_ref: gradingPlan.run_ref,
  request: {
    review_request_id: "plan-review-request://grading/1",
    plan_kind: "GRADING_DESIGN",
    plan_version: 1,
    plan_ref: objectRef("grading-design-plan", "example"),
    requested_by: "user://plan-owner",
  },
  presentation: {
    title: "Review grading design",
    summary_lines: ["One evaluator-only judge task"],
    editable_paths: ["judge_tasks"],
    warning_codes: [],
  },
  result: {
    state: "PENDING_REVIEW",
    plan_version: 1,
    successor_plan_ref: null,
    resume_token_ref: null,
  },
  plan: gradingPlan,
  decision: null,
  revision: null,
};

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.replaceChildren();
});

describe("grading plan rendering", () => {
  it("renders a judge ledger and a safe editable projection", async () => {
    Object.assign(globalThis, {
      IS_REACT_ACT_ENVIRONMENT: true,
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        const payload = url.endsWith(
          "/api/plan-reviews/contract",
        )
          ? {
              schema_version:
                "eval-factory/plan-review-api-contract/v1",
              decision_actions: [
                "APPROVE",
                "EDIT",
                "REJECT",
                "DEFER",
                "REQUEST_MORE",
              ],
              review_states: ["PENDING_REVIEW"],
              principal_header:
                "X-Eval-Factory-Principal",
            }
          : {
              schema_version:
                "eval-factory/plan-review-page/v1",
              items: [review],
              offset: 0,
              limit: 100,
              total: 1,
            };
        return new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<App />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(container.textContent).toContain("1 个评分任务");
    expect(container.textContent).toContain("judge-main");
    expect(container.textContent).toContain("绑定 1 条准则");
    expect(container.textContent).toContain("WEIGHTED_SUM");

    expect(container.querySelector(".workbench-shell")).not.toBeNull();
    expect(container.textContent).toContain("控制平面在线");

    const search = container.querySelector<HTMLInputElement>(
      'input[type="search"]',
    );
    expect(search).not.toBeNull();
    await act(async () => {
      if (!search) return;
      setInputValue(search, "missing review");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(container.textContent).toContain("没有匹配的审核项");
    await act(async () => {
      if (!search) return;
      setInputValue(search, "");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const activityButton = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((button) => button.getAttribute("aria-label") === "审核动态");
    expect(activityButton).not.toBeUndefined();
    await act(async () => activityButton?.click());
    const activity = container.querySelector<HTMLElement>(
      '[role="dialog"][aria-label="审核动态"]',
    );
    expect(activity).not.toBeNull();
    expect(activity?.textContent).toContain("Pending Review");
    expect(activity?.textContent).toContain("暂无警告");
    expect(activity?.textContent).not.toContain("Token 用量");
    expect(activity?.textContent).not.toContain("运行进度");

    const jsonTab = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        '[role="tab"]',
      ),
    ).find((button) => button.textContent?.includes("计划 JSON"));
    expect(jsonTab).not.toBeUndefined();
    await act(async () => jsonTab?.click());
    expect(jsonTab?.getAttribute("aria-selected")).toBe("true");

    const editor = container.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="可编辑计划 JSON"]',
    );
    expect(editor).not.toBeNull();
    expect(editor?.dataset.editorReviewId).toBe(
      review.request.review_request_id,
    );
    expect(editor?.dataset.editorPlanVersion).toBe(
      String(review.request.plan_version),
    );
    const projection = JSON.parse(editor?.value ?? "{}") as Record<
      string,
      unknown
    >;
    expect(projection).toHaveProperty("judge_tasks");
    expect(projection).not.toHaveProperty(
      "criteria_rubric_result_ref",
    );
    expect(projection).not.toHaveProperty("rubric_set_ref");
    expect(projection).not.toHaveProperty(
      "judge_prompt_template_ref",
    );

    await act(async () => root.unmount());
  });
});

function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;
  setter?.call(input, value);
}
