import {
  type PlanEditableFields,
  type PlanDecision,
  type PlanReviewApiContract,
  type PlanReviewPage,
  type PlanReviewView,
  isPlanReviewPage,
  isPlanReviewView,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const PRINCIPAL_HEADER = "X-Eval-Factory-Principal";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export async function fetchApiContract(): Promise<PlanReviewApiContract> {
  const value = await requestJson("/api/plan-reviews/contract");
  if (!value || typeof value !== "object") throw invalidResponse();
  const record = value as Record<string, unknown>;
  if (
    record.schema_version !== "eval-factory/plan-review-api-contract/v1" ||
    !Array.isArray(record.decision_actions) ||
    !Array.isArray(record.review_states) ||
    record.principal_header !== PRINCIPAL_HEADER
  ) {
    throw invalidResponse();
  }
  return value as PlanReviewApiContract;
}

export async function listPlanReviews(
  state = "PENDING_REVIEW",
): Promise<PlanReviewPage> {
  const parameters = new URLSearchParams({ state });
  const value = await requestJson(`/api/plan-reviews?${parameters.toString()}`);
  if (!isPlanReviewPage(value)) throw invalidResponse();
  return value;
}

export async function showPlanReview(reviewId: string): Promise<PlanReviewView> {
  const parameters = new URLSearchParams({ review_id: reviewId });
  const value = await requestJson(`/api/plan-reviews/show?${parameters.toString()}`);
  if (!isPlanReviewView(value)) throw invalidResponse();
  return value;
}

export async function decidePlanReview(
  view: PlanReviewView,
  decision: Exclude<PlanDecision, "EDIT">,
  principal: string,
  reasonCode: string,
): Promise<PlanReviewView> {
  return command("/api/plan-reviews/decision", principal, {
    review_id: view.request.review_request_id,
    submission: {
      expected_plan_version: view.request.plan_version,
      decision,
      decided_by: principal,
      reason_code: reasonCode,
      idempotency_key: createIdempotencyKey(
        `${decision}-${view.request.review_request_id}`,
      ),
    },
  });
}

export async function editPlanReview(
  view: PlanReviewView,
  editedPlan: PlanEditableFields,
  changedPaths: string[],
  principal: string,
): Promise<PlanReviewView> {
  return command("/api/plan-reviews/edit", principal, {
    review_id: view.request.review_request_id,
    submission: {
      expected_plan_version: view.request.plan_version,
      edited_plan: editedPlan,
      changed_paths: [...new Set(changedPaths)].sort(),
      invalidated_object_refs: [],
      decided_by: principal,
      reason_code: "USER_EDITED_PLAN",
      idempotency_key: createIdempotencyKey(
        `EDIT-${view.request.review_request_id}`,
      ),
    },
  });
}

export async function resumePlanReview(
  view: PlanReviewView,
  principal: string,
): Promise<PlanReviewView> {
  return command("/api/plan-reviews/resume", principal, {
    review_id: view.request.review_request_id,
    submission: {
      expected_plan_version: view.request.plan_version,
      resumed_by: principal,
      idempotency_key: createIdempotencyKey(
        `RESUME-${view.request.review_request_id}`,
      ),
    },
  });
}

async function command(
  path: string,
  principal: string,
  body: object,
): Promise<PlanReviewView> {
  const value = await requestJson(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      [PRINCIPAL_HEADER]: principal,
    },
    body: JSON.stringify(body),
  });
  if (!isPlanReviewView(value)) throw invalidResponse();
  return value;
}

export async function requestJson(
  path: string,
  init?: RequestInit,
): Promise<unknown> {
  const response = await fetch(apiUrl(path), init);
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    if (response.ok) throw invalidResponse();
    throw new ApiError(
      response.status,
      "HTTP_ERROR",
      "Agent Host 请求失败。",
    );
  }
  if (!response.ok) {
    const record =
      body && typeof body === "object" ? (body as Record<string, unknown>) : {};
    throw new ApiError(
      response.status,
      typeof record.error_code === "string" ? record.error_code : "HTTP_ERROR",
      typeof record.message === "string" ? record.message : "Agent Host 请求失败。",
    );
  }
  return body;
}

function invalidResponse(): ApiError {
  return new ApiError(
    500,
    "INVALID_RESPONSE",
    "Agent Host 返回了无效响应。",
  );
}

export function createIdempotencyKey(seed: string): string {
  const suffix =
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return `${seed}-${suffix}`;
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
