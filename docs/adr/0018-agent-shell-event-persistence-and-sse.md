# ADR 0018: Project Runtime Events Through Harness Session Authority

- Status: Accepted
- Date: 2026-08-23
- Spec: `specs/003-evaluation-agent-harness/spec.md`
- Trellis: `.trellis/tasks/08-17-conversation-first-agent-shell-v1/`
- Beads: `env_mock_agent-ujc.18`

## Context

Stage 5.0 created a provider-neutral runtime event protocol, but those events
were not durable Harness observations and the product host had no resumable
event stream. Streaming Provider-native events directly would expose private
source references and make a connection-local counter compete with Harness
session authority.

## Decision

Persist each normalized runtime event and its `SessionEventV1` projection in
one `HarnessSessionStore` transaction. The transaction also advances the
session state/version/sequence, rebuilds the session projection, and records
idempotency.

Expose only `AgentShellEventV1` over HTTP and SSE. Runtime event source refs,
raw payloads, paths, tool arguments/results, and runtime handles remain below
the public boundary. Public runtime activity carries only normalized runtime
identity, event kind, tool family, usage, and closed failure code.

SSE uses the committed Harness sequence:

```text
id: <session sequence>
event: session-event
data: <AgentShellEventV1>
```

`Last-Event-ID` and `after_sequence` must agree. Replay starts after that
sequence. Heartbeats have no event ID and are not persisted. Disconnecting a
client stops only its observer.

## Integrity Rules

- Runtime source sequence is contiguous per session/runtime/run/work stream.
- The first event is `RUN_STARTED`; a stream cannot restart.
- `RUN_COMPLETED` and `RUN_FAILED` require prior `USAGE_REPORTED`.
- No event follows a terminal event.
- One runtime event identity belongs to one Harness session.
- Runtime JSON, hashes, materialized columns, and SessionEvent linkage are
  validated on replay/read.
- Public event pages use one captured watermark so concurrent commits appear
  on the next page.

## Compatibility

- Existing PlanReview routes and `create_app(PlanReviewService)` remain
  unchanged.
- `create_agent_app` adds `/api/harness/**` to the same FastAPI application.
- The checked-in OpenAPI document is generated from the combined app.
- No Factory, Team, Graph, JobStore, artifact-head, approval, or release
  authority moves into the Agent Shell.

## Follow-Up

Stage 5.2 adds bounded source admission. Stage 5.3 composes Team, Factory,
Graph, PlanReview, workspace, and delivery projections into the shell host.
Those stages must continue to use committed Harness sequences for observation
and typed HTTP commands for mutation.
