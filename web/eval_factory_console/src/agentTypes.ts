import type { ObjectRef } from "./types";

export type AgentShellExecutionPhase =
  | "NOT_CONFIGURED"
  | "WAITING_REQUIREMENT"
  | "WAITING_SOURCE"
  | "READY"
  | "RUNNING"
  | "WAITING_REVIEW"
  | "VERIFICATION_REQUIRED"
  | "BLOCKED"
  | "COMPLETED";

export type AgentShellWorkspaceKind =
  | "CONVERSATION"
  | "PLAN_REVIEW"
  | "TRACE"
  | "TASK"
  | "ATTACHMENT"
  | "RUBRIC"
  | "GRADING"
  | "QUALITY"
  | "TEAM"
  | "ACTIVITY"
  | "DELIVERY";

export type AgentShellWorkspaceStatus =
  | "AVAILABLE"
  | "EMPTY"
  | "BLOCKED"
  | "COMPLETE";

export type AgentShellConnectionState =
  | "connecting"
  | "online"
  | "reconnecting"
  | "offline";

export interface AgentShellSessionSummary {
  schema_version: "eval-factory/agent-shell-session-summary/v1";
  session_id: string;
  title: string | null;
  status: "ACTIVE" | "BLOCKED" | "CLOSED";
  session_version: number;
  last_event_sequence: number;
  created_at: string;
  updated_at: string;
}

export interface AgentShellSessionPage {
  schema_version: "eval-factory/agent-shell-session-page/v1";
  offset: number;
  limit: number;
  total: number;
  sessions: AgentShellSessionSummary[];
}

export interface AgentShellTranscriptEntry {
  schema_version: "eval-factory/agent-shell-transcript-entry/v1";
  message_id: string;
  role: "USER" | "ASSISTANT";
  content: string;
  artifact_envelope_refs: ObjectRef[];
  created_at: string;
}

export interface RuntimeUsage {
  schema_version: "env-mock-agent/unified-runtime-usage/v2";
  model_requests: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
  tool_calls: number;
  turns: number;
  duration_ms: number | null;
  reported_cost: {
    availability: string;
    amount_micro_usd: number | null;
    currency: string | null;
  };
}

export interface AgentShellEvent {
  schema_version: "eval-factory/agent-shell-event/v1";
  event_id: string;
  sequence: number;
  authority_version: number;
  family: string;
  event_kind: string;
  occurred_at: string;
  turn_id: string | null;
  step_id: string | null;
  runtime_id: string | null;
  tool_family: string | null;
  usage: RuntimeUsage | null;
  failure_code: string | null;
}

export interface AgentShellFactorySummary {
  schema_version: "eval-factory/agent-shell-factory-summary/v1";
  run_ref: ObjectRef;
  run_version: number;
  status: string;
  next_action: string;
  pending_review_refs: ObjectRef[];
  candidate_count: number;
  rejected_count: number;
  blocked_count: number;
  incomplete_count: number;
  aggregate_result_ref: ObjectRef | null;
  delivery_manifest_ref: ObjectRef | null;
}

export interface AgentShellGraphSummary {
  schema_version: "eval-factory/agent-shell-graph-summary/v1";
  binding_ref: ObjectRef;
  checkpoint_ref: ObjectRef | null;
  checkpoint_phase: string | null;
  checkpoint_outcome: string | null;
  node: string | null;
  transition_number: number;
  reason_codes: string[];
}

export interface AgentShellTeamMemberSummary {
  schema_version: "eval-factory/agent-shell-team-member-summary/v1";
  member_ref: ObjectRef;
  member_id: string;
  role: string;
  status: string;
  independent_session_ref: ObjectRef;
  capability_definition_refs: ObjectRef[];
  is_coordinator: boolean;
}

export interface AgentShellTeamTaskSummary {
  schema_version: "eval-factory/agent-shell-team-task-summary/v1";
  task_ref: ObjectRef;
  task_id: string;
  task_kind: string;
  status: string;
  assigned_member_id: string | null;
  dependency_task_ids: string[];
  capability_definition_ref: ObjectRef;
  attempt: number;
  event_version: number;
  claim_ref: ObjectRef | null;
  lease_ref: ObjectRef | null;
  latest_event_ref: ObjectRef | null;
}

export interface AgentShellTeamSummary {
  schema_version: "eval-factory/agent-shell-team-summary/v1";
  team_ref: ObjectRef;
  team_version: number;
  lifecycle: string;
  roster_ref: ObjectRef;
  task_graph_ref: ObjectRef;
  authority_ref: ObjectRef;
  checkpoint_ref: ObjectRef | null;
  members: AgentShellTeamMemberSummary[];
  tasks: AgentShellTeamTaskSummary[];
  artifact_head_refs: ObjectRef[];
  open_conflict_refs: ObjectRef[];
}

export interface AgentShellPlanReviewSummary {
  schema_version: "eval-factory/agent-shell-plan-review-summary/v1";
  run_id: string;
  request_ref: ObjectRef;
  result_ref: ObjectRef;
  plan_ref: ObjectRef;
  plan_kind: string;
  plan_version: number;
  state: string;
  decision_ref: ObjectRef | null;
  revision_ref: ObjectRef | null;
}

export interface AgentShellInteractionCard {
  schema_version: "eval-factory/agent-shell-interaction-card/v1";
  interaction_id: string;
  kind: "CLARIFICATION" | "PERMISSION" | "PLAN_REVIEW" | "VERIFICATION";
  state: "PENDING" | "BLOCKED";
  subject_ref: ObjectRef;
  questions: string[];
  reason_codes: string[];
}

export interface AgentShellWorkspaceDescriptor {
  schema_version: "eval-factory/agent-shell-workspace/v1";
  kind: AgentShellWorkspaceKind;
  status: AgentShellWorkspaceStatus;
  owner_refs: ObjectRef[];
  item_count: number;
  reason_codes: string[];
}

export interface AgentShellDeliverySummary {
  schema_version: "eval-factory/agent-shell-delivery-summary/v1";
  completion_ref: ObjectRef;
  manifest_ref: ObjectRef;
  inventory_ref: ObjectRef;
  item_count: number;
  file_count: number;
  total_bytes: number;
  bundle_sha256: string;
}

export interface AgentShellProjection {
  schema_version: "eval-factory/agent-shell-projection/v1";
  session: AgentShellSessionSummary;
  transcript: AgentShellTranscriptEntry[];
  activity: AgentShellEvent[];
  requirement_outcome: string | null;
  evidence_class: string | null;
  source_admission_ref: ObjectRef | null;
  execution_phase: AgentShellExecutionPhase;
  factory: AgentShellFactorySummary | null;
  graph: AgentShellGraphSummary | null;
  team: AgentShellTeamSummary | null;
  plan_reviews: AgentShellPlanReviewSummary[];
  pending_interactions: AgentShellInteractionCard[];
  workspaces: AgentShellWorkspaceDescriptor[];
  delivery: AgentShellDeliverySummary | null;
  reconnect_cursor: number;
  source_fingerprint: string;
}

export interface AgentShellTeamMessageSummary {
  schema_version: "eval-factory/agent-shell-team-message-summary/v1";
  message_ref: ObjectRef;
  sender_member_ref: ObjectRef;
  audience: string;
  message_kind: string;
  recipient_member_refs: ObjectRef[];
  task_ref: ObjectRef | null;
  artifact_envelope_refs: ObjectRef[];
  reply_to_message_ref: ObjectRef | null;
}

export interface AgentShellMemberProjection {
  schema_version: "eval-factory/agent-shell-member-projection/v1";
  team_ref: ObjectRef;
  member: AgentShellTeamMemberSummary;
  task_refs: ObjectRef[];
  artifact_envelope_refs: ObjectRef[];
  messages: AgentShellTeamMessageSummary[];
  subscription_refs: ObjectRef[];
  acceptance_check_refs: ObjectRef[];
  source_fingerprint: string;
}

export interface AgentShellSourceFile {
  schema_version: "eval-factory/agent-shell-source-file/v1";
  kind: "MANIFEST" | "TRACE";
  relative_name: string;
  media_type: "text/csv" | "application/x-ndjson";
  size_bytes: number;
  sha256: string;
  source_ref: ObjectRef | null;
  artifact_envelope_ref: ObjectRef | null;
}

export interface AgentShellSourceAdmission {
  schema_version: "eval-factory/agent-shell-source-admission/v1";
  status: "ADMITTED";
  admission_ref: ObjectRef;
  session_id: string;
  session_version: number;
  manifest_sha256: string;
  source_count: number;
  total_source_bytes: number;
  files: AgentShellSourceFile[];
  artifact_envelope_refs: ObjectRef[];
  created_at: string;
}

export interface AgentShellSourceContract {
  schema_version: "eval-factory/agent-shell-source-admission-contract/v1";
  manifest_name: "manifest.csv";
  manifest_columns: string[];
  max_source_files: number;
  max_manifest_bytes: number;
  max_source_bytes: number;
  max_total_source_bytes: number;
  max_request_bytes: number;
  error_codes: string[];
}

export interface AgentShellApiContract {
  schema_version: "eval-factory/agent-shell-api-contract/v1";
  principal_header: "X-Eval-Factory-Principal";
  event_stream_media_type: "text/event-stream";
  event_families: string[];
  max_event_page_size: 500;
}

export function isAgentShellProjection(
  value: unknown,
): value is AgentShellProjection {
  if (!isRecord(value)) return false;
  return (
    value.schema_version ===
      "eval-factory/agent-shell-projection/v1" &&
    isSessionSummary(value.session) &&
    Array.isArray(value.transcript) &&
    value.transcript.every(isTranscriptEntry) &&
    Array.isArray(value.activity) &&
    value.activity.every(isAgentShellEvent) &&
    Array.isArray(value.pending_interactions) &&
    Array.isArray(value.workspaces) &&
    typeof value.execution_phase === "string" &&
    typeof value.reconnect_cursor === "number" &&
    typeof value.source_fingerprint === "string"
  );
}

export function isAgentShellSessionPage(
  value: unknown,
): value is AgentShellSessionPage {
  return (
    isRecord(value) &&
    value.schema_version ===
      "eval-factory/agent-shell-session-page/v1" &&
    typeof value.total === "number" &&
    Array.isArray(value.sessions) &&
    value.sessions.every(isSessionSummary)
  );
}

export function isAgentShellSourceAdmission(
  value: unknown,
): value is AgentShellSourceAdmission {
  return (
    isRecord(value) &&
    value.schema_version ===
      "eval-factory/agent-shell-source-admission/v1" &&
    value.status === "ADMITTED" &&
    typeof value.session_id === "string" &&
    Array.isArray(value.files) &&
    Array.isArray(value.artifact_envelope_refs)
  );
}

export function isAgentShellSourceContract(
  value: unknown,
): value is AgentShellSourceContract {
  return (
    isRecord(value) &&
    value.schema_version ===
      "eval-factory/agent-shell-source-admission-contract/v1" &&
    value.manifest_name === "manifest.csv" &&
    typeof value.max_source_files === "number"
  );
}

export function isAgentShellApiContract(
  value: unknown,
): value is AgentShellApiContract {
  return (
    isRecord(value) &&
    value.schema_version ===
      "eval-factory/agent-shell-api-contract/v1" &&
    value.principal_header === "X-Eval-Factory-Principal" &&
    value.event_stream_media_type === "text/event-stream"
  );
}

export function isAgentShellMemberProjection(
  value: unknown,
): value is AgentShellMemberProjection {
  return (
    isRecord(value) &&
    value.schema_version ===
      "eval-factory/agent-shell-member-projection/v1" &&
    isRecord(value.member) &&
    typeof value.member.member_id === "string" &&
    Array.isArray(value.messages) &&
    typeof value.source_fingerprint === "string"
  );
}

export function isAgentShellEvent(
  value: unknown,
): value is AgentShellEvent {
  return (
    isRecord(value) &&
    value.schema_version === "eval-factory/agent-shell-event/v1" &&
    typeof value.event_id === "string" &&
    typeof value.sequence === "number" &&
    typeof value.family === "string" &&
    typeof value.event_kind === "string"
  );
}

function isSessionSummary(
  value: unknown,
): value is AgentShellSessionSummary {
  return (
    isRecord(value) &&
    value.schema_version ===
      "eval-factory/agent-shell-session-summary/v1" &&
    typeof value.session_id === "string" &&
    typeof value.session_version === "number" &&
    typeof value.last_event_sequence === "number"
  );
}

function isTranscriptEntry(
  value: unknown,
): value is AgentShellTranscriptEntry {
  return (
    isRecord(value) &&
    value.schema_version ===
      "eval-factory/agent-shell-transcript-entry/v1" &&
    typeof value.message_id === "string" &&
    typeof value.content === "string" &&
    (value.role === "USER" || value.role === "ASSISTANT")
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
