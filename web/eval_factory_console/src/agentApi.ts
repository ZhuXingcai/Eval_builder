import {
  ApiError,
  apiUrl,
  createIdempotencyKey,
  requestJson,
} from "./api";
import {
  isAgentShellApiContract,
  isAgentShellEvent,
  isAgentShellMemberProjection,
  isAgentShellProjection,
  isAgentShellSessionPage,
  isAgentShellSourceAdmission,
  isAgentShellSourceContract,
  type AgentShellApiContract,
  type AgentShellConnectionState,
  type AgentShellEvent,
  type AgentShellMemberProjection,
  type AgentShellProjection,
  type AgentShellSessionPage,
  type AgentShellSessionSummary,
  type AgentShellSourceAdmission,
  type AgentShellSourceContract,
} from "./agentTypes";

const PRINCIPAL_HEADER = "X-Eval-Factory-Principal";
const SSE_RETRY_DELAY_MS = 1200;
const SSE_IDLE_TIMEOUT_MS = 20_000;

export async function fetchAgentShellContract(): Promise<AgentShellApiContract> {
  const value = await requestJson("/api/harness/contract");
  if (!isAgentShellApiContract(value)) throw invalidResponse();
  return value;
}

export async function fetchSourceContract(): Promise<AgentShellSourceContract> {
  const value = await requestJson(
    "/api/harness/source-admission/contract",
  );
  if (!isAgentShellSourceContract(value)) throw invalidResponse();
  return value;
}

export async function listAgentSessions(): Promise<AgentShellSessionPage> {
  const value = await requestJson("/api/harness/sessions?limit=100");
  if (!isAgentShellSessionPage(value)) throw invalidResponse();
  return value;
}

export async function createAgentSession(
  principal: string,
): Promise<AgentShellProjection> {
  const suffix = identitySuffix();
  const value = await requestJson("/api/harness/sessions", {
    method: "POST",
    headers: jsonHeaders(principal),
    body: JSON.stringify({
      session_id: `session-ui-${suffix}`,
      incarnation_id: `session-ui-${suffix}-incarnation`,
      idempotency_key: createIdempotencyKey(
        `create-session-ui-${suffix}`,
      ),
    }),
  });
  return projection(value);
}

export async function closeAgentSession(
  session: AgentShellSessionSummary,
  principal: string,
): Promise<AgentShellProjection> {
  const value = await requestJson(
    `/api/harness/sessions/${encodeURIComponent(
      session.session_id,
    )}`,
    {
      method: "DELETE",
      headers: jsonHeaders(principal),
      body: JSON.stringify({
        expected_session_version: session.session_version,
        idempotency_key: createIdempotencyKey(
          `close-session-${session.session_id}`,
        ),
      }),
    },
  );
  return projection(value);
}

export async function getAgentSession(
  sessionId: string,
): Promise<AgentShellProjection> {
  return projection(
    await requestJson(
      `/api/harness/sessions/${encodeURIComponent(sessionId)}`,
    ),
  );
}

export async function postAgentMessage(
  session: AgentShellProjection,
  content: string,
  principal: string,
  source: AgentShellSourceAdmission | null,
): Promise<AgentShellProjection> {
  const value = await requestJson(
    `/api/harness/sessions/${encodeURIComponent(
      session.session.session_id,
    )}/messages`,
    {
      method: "POST",
      headers: jsonHeaders(principal),
      body: JSON.stringify({
        expected_session_version:
          session.session.session_version,
        content,
        artifact_envelope_refs:
          source?.artifact_envelope_refs ?? [],
        idempotency_key: createIdempotencyKey(
          `message-${session.session.session_id}`,
        ),
      }),
    },
  );
  return projection(value);
}

export async function reconcileAgentSession(
  session: AgentShellProjection,
  principal: string,
): Promise<AgentShellProjection> {
  const value = await requestJson(
    `/api/harness/sessions/${encodeURIComponent(
      session.session.session_id,
    )}/reconcile`,
    {
      method: "POST",
      headers: jsonHeaders(principal),
      body: JSON.stringify({
        expected_session_version:
          session.session.session_version,
        idempotency_key: createIdempotencyKey(
          `reconcile-${session.session.session_id}`,
        ),
      }),
    },
  );
  return projection(value);
}

export async function getAgentSource(
  sessionId: string,
): Promise<AgentShellSourceAdmission | null> {
  try {
    const value = await requestJson(
      `/api/harness/sessions/${encodeURIComponent(sessionId)}/sources`,
    );
    if (!isAgentShellSourceAdmission(value)) throw invalidResponse();
    return value;
  } catch (reason) {
    if (reason instanceof ApiError && reason.status === 404) return null;
    throw reason;
  }
}

export async function getAgentMember(
  sessionId: string,
  memberId: string,
): Promise<AgentShellMemberProjection> {
  const value = await requestJson(
    `/api/harness/sessions/${encodeURIComponent(
      sessionId,
    )}/members/${encodeURIComponent(memberId)}`,
  );
  if (!isAgentShellMemberProjection(value)) throw invalidResponse();
  return value;
}

export async function admitAgentSource(
  session: AgentShellProjection,
  files: File[],
  principal: string,
): Promise<AgentShellSourceAdmission> {
  const manifest = files.filter(
    (file) => file.name === "manifest.csv",
  );
  const traces = files
    .filter((file) => file.name.endsWith(".jsonl"))
    .sort((left, right) => left.name.localeCompare(right.name));
  if (
    manifest.length !== 1 ||
    traces.length < 1 ||
    manifest.length + traces.length !== files.length
  ) {
    throw new ApiError(
      422,
      "SOURCE_SELECTION_INVALID",
      "请选择一个 manifest.csv 和至少一个 JSONL 文件。",
    );
  }
  const [manifestSha256, ...traceHashes] = await Promise.all([
    sha256(manifest[0]),
    ...traces.map(sha256),
  ]);
  const command = {
    expected_session_version: session.session.session_version,
    manifest_sha256: manifestSha256,
    manifest_size_bytes: manifest[0].size,
    trace_files: traces.map((file, index) => ({
      relative_name: file.name,
      expected_sha256: traceHashes[index],
      expected_size_bytes: file.size,
    })),
    idempotency_key: createIdempotencyKey(
      `admit-source-${session.session.session_id}`,
    ),
  };
  const body = new FormData();
  body.append("command", JSON.stringify(command));
  body.append("manifest", manifest[0], manifest[0].name);
  for (const trace of traces) {
    body.append("traces", trace, trace.name);
  }
  const value = await requestJson(
    `/api/harness/sessions/${encodeURIComponent(
      session.session.session_id,
    )}/sources`,
    {
      method: "POST",
      headers: {
        [PRINCIPAL_HEADER]: principal,
      },
      body,
    },
  );
  if (!isAgentShellSourceAdmission(value)) throw invalidResponse();
  return value;
}

export function subscribeAgentSession(
  sessionId: string,
  afterSequence: number,
  onEvent: (event: AgentShellEvent) => void,
  onState: (state: AgentShellConnectionState) => void,
): () => void {
  let cancelled = false;
  let cursor = afterSequence;
  let connectionAttempt = 0;
  let controller: AbortController | null = null;
  let retryTimer: number | null = null;

  const connect = async () => {
    if (cancelled) return;
    const requestController = new AbortController();
    controller = requestController;
    let idleTimer: number | null = null;
    let idleExpired = false;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    const armIdleTimer = () => {
      if (idleTimer !== null) window.clearTimeout(idleTimer);
      idleTimer = window.setTimeout(() => {
        idleExpired = true;
        void reader?.cancel();
        requestController.abort();
      }, SSE_IDLE_TIMEOUT_MS);
    };
    onState(
      connectionAttempt === 0 ? "connecting" : "reconnecting",
    );
    connectionAttempt += 1;
    armIdleTimer();
    try {
      const parameters = new URLSearchParams({
        after_sequence: String(cursor),
        follow: "true",
      });
      const response = await fetch(
        apiUrl(
          `/api/harness/sessions/${encodeURIComponent(
            sessionId,
          )}/stream?${parameters.toString()}`,
        ),
        {
          headers: { Accept: "text/event-stream" },
          signal: requestController.signal,
        },
      );
      if (!response.ok || !response.body) {
        throw new ApiError(
          response.status,
          "SSE_CONNECTION_FAILED",
          "事件流连接失败。",
        );
      }
      onState("online");
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (!cancelled) {
        const chunk = await reader.read();
        if (chunk.done) break;
        armIdleTimer();
        buffer += decoder.decode(chunk.value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const event = parseEventFrame(frame);
          if (!event) continue;
          cursor = Math.max(cursor, event.sequence);
          onEvent(event);
        }
      }
    } catch (reason) {
      if (cancelled || (isAbortError(reason) && !idleExpired)) return;
      onState("offline");
    } finally {
      if (idleTimer !== null) window.clearTimeout(idleTimer);
    }
    if (!cancelled) {
      onState("reconnecting");
      retryTimer = window.setTimeout(() => {
        void connect();
      }, SSE_RETRY_DELAY_MS);
    }
  };

  void connect();
  return () => {
    cancelled = true;
    controller?.abort();
    if (retryTimer !== null) window.clearTimeout(retryTimer);
  };
}

export function parseEventFrame(
  frame: string,
): AgentShellEvent | null {
  const eventName = frame
    .split(/\r?\n/)
    .find((line) => line.startsWith("event:"))
    ?.slice("event:".length)
    .trim();
  if (eventName !== "session-event") return null;
  const data = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n");
  if (!data) return null;
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    return null;
  }
  return isAgentShellEvent(value) ? value : null;
}

function jsonHeaders(principal: string): Record<string, string> {
  return {
    "Content-Type": "application/json",
    [PRINCIPAL_HEADER]: principal,
  };
}

function projection(value: unknown): AgentShellProjection {
  if (!isAgentShellProjection(value)) throw invalidResponse();
  return value;
}

async function sha256(file: File): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    await file.arrayBuffer(),
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function identitySuffix(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  ).replaceAll("-", "");
}

function invalidResponse(): ApiError {
  return new ApiError(
    500,
    "INVALID_RESPONSE",
    "Agent Shell 响应未通过契约校验。",
  );
}

function isAbortError(reason: unknown): boolean {
  return (
    reason instanceof DOMException &&
    reason.name === "AbortError"
  );
}
