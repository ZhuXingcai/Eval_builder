import { afterEach, describe, expect, it, vi } from "vitest";

import {
  admitAgentSource,
  closeAgentSession,
  parseEventFrame,
  subscribeAgentSession,
} from "./agentApi";
import type { AgentShellProjection } from "./agentTypes";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Agent Shell API", () => {
  it("parses only typed session-event frames", () => {
    const event = {
      schema_version: "eval-factory/agent-shell-event/v1",
      event_id: "session-event://example/2",
      sequence: 2,
      authority_version: 2,
      family: "MESSAGE",
      event_kind: "USER_MESSAGE_RECORDED",
      occurred_at: "2026-08-23T00:00:00Z",
      turn_id: null,
      step_id: null,
      runtime_id: null,
      tool_family: null,
      usage: null,
      failure_code: null,
    };

    expect(
      parseEventFrame(
        `id: 2\nevent: session-event\ndata: ${JSON.stringify(event)}`,
      ),
    ).toEqual(event);
    expect(
      parseEventFrame(
        'event: heartbeat\ndata: {"connection":"alive"}',
      ),
    ).toBeNull();
    expect(
      parseEventFrame("event: session-event\ndata: not-json"),
    ).toBeNull();
  });

  it("builds a sorted, hash-bound multipart source command", async () => {
    const subtle = {
      digest: vi.fn(
        async (_algorithm: string, value: ArrayBuffer) =>
          new Uint8Array(32).fill(value.byteLength).buffer,
      ),
    };
    vi.stubGlobal("crypto", {
      randomUUID: () => "00000000-0000-4000-8000-000000000001",
      subtle,
    });
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, init?: RequestInit) => {
        const body = init?.body;
        expect(body).toBeInstanceOf(FormData);
        const form = body as FormData;
        const command = JSON.parse(
          String(form.get("command")),
        ) as {
          trace_files: Array<{
            relative_name: string;
            expected_sha256: string;
          }>;
        };
        expect(
          command.trace_files.map((file) => file.relative_name),
        ).toEqual(["A_1.jsonl", "B_1.jsonl"]);
        expect(command.trace_files[0].expected_sha256).toHaveLength(
          64,
        );
        return json({
          schema_version:
            "eval-factory/agent-shell-source-admission/v1",
          status: "ADMITTED",
          admission_ref: objectRef("harness-source-admission"),
          session_id: "session-ui-test",
          session_version: 1,
          manifest_sha256: "a".repeat(64),
          source_count: 2,
          total_source_bytes: 4,
          files: [],
          artifact_envelope_refs: [
            objectRef("artifact-envelope"),
          ],
          created_at: "2026-08-23T00:00:00Z",
        });
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await admitAgentSource(
      projection(),
      [
        file("bb", "B_1.jsonl"),
        file("manifest", "manifest.csv"),
        file("aa", "A_1.jsonl"),
      ],
      "user://agent-shell",
    );

    expect(result.status).toBe("ADMITTED");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("closes a session with version and principal authority", async () => {
    vi.stubGlobal("crypto", {
      randomUUID: () => "00000000-0000-4000-8000-000000000002",
    });
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, init?: RequestInit) => {
        expect(init?.method).toBe("DELETE");
        expect(init?.headers).toMatchObject({
          "Content-Type": "application/json",
          "X-Eval-Factory-Principal": "user://agent-shell",
        });
        expect(JSON.parse(String(init?.body))).toMatchObject({
          expected_session_version: 1,
          idempotency_key:
            "close-session-session-ui-test-00000000-0000-4000-8000-000000000002",
        });
        const closed = projection();
        closed.session.status = "CLOSED";
        closed.session.session_version = 2;
        closed.session.last_event_sequence = 2;
        return json(closed);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const closed = await closeAgentSession(
      projection().session,
      "user://agent-shell",
    );

    expect(closed.session.status).toBe("CLOSED");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("labels every connection after the first as reconnecting", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("connection unavailable");
      }),
    );
    const states: string[] = [];
    const stop = subscribeAgentSession(
      "session-ui-test",
      1,
      () => undefined,
      (state) => states.push(state),
    );

    await Promise.resolve();
    await Promise.resolve();
    expect(states).toEqual([
      "connecting",
      "offline",
      "reconnecting",
    ]);

    await vi.advanceTimersByTimeAsync(1200);
    expect(states.filter((state) => state === "connecting")).toHaveLength(
      1,
    );
    expect(states.at(-1)).toBe("reconnecting");
    stop();
  });
});

function projection(): AgentShellProjection {
  return {
    schema_version: "eval-factory/agent-shell-projection/v1",
    session: {
      schema_version:
        "eval-factory/agent-shell-session-summary/v1",
      session_id: "session-ui-test",
      title: null,
      status: "ACTIVE",
      session_version: 1,
      last_event_sequence: 1,
      created_at: "2026-08-23T00:00:00Z",
      updated_at: "2026-08-23T00:00:00Z",
    },
    transcript: [],
    activity: [],
    requirement_outcome: null,
    evidence_class: null,
    source_admission_ref: null,
    execution_phase: "WAITING_REQUIREMENT",
    factory: null,
    graph: null,
    team: null,
    plan_reviews: [],
    pending_interactions: [],
    workspaces: [],
    delivery: null,
    reconnect_cursor: 1,
    source_fingerprint: "a".repeat(64),
  };
}

function objectRef(objectType: string) {
  return {
    object_type: objectType,
    object_id: `${objectType}://example`,
    object_version: "v1",
    object_sha256: "a".repeat(64),
  };
}

function file(content: string, name: string): File {
  const value = new File([content], name);
  Object.defineProperty(value, "arrayBuffer", {
    value: async () => new TextEncoder().encode(content).buffer,
  });
  return value;
}

function json(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
