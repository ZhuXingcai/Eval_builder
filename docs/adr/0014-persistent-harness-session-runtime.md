# ADR 0014: Add A Persistent Harness Session Runtime

- Status: Accepted
- Date: 2026-08-17
- Spec: `specs/003-evaluation-agent-harness/spec.md`
- Trellis: `.trellis/tasks/08-17-persistent-session-agent-loop-v1/`
- Depends on: ADR 0012, ADR 0013

## Context

ADR 0013 froze provider-neutral Harness contracts but intentionally added no
runtime. The current product still enters through assembled Factory request
files and fixture-oriented commands. A conversation UI cannot be authoritative
unless messages, model-visible context, Gateway use, clarification, results,
and recovery are durable before the frontend exists.

Using `FactoryControlStore` as a chat database would mix conversation and
Factory run lifecycles. Using Gateway transcripts as the session log would
make product recovery vendor-dependent. Treating WebSocket/SSE frames as
authority would lose state on disconnect.

## Decision

Add an independent `HarnessSessionStore` and `HarnessSessionService`.

`HarnessSessionStore` owns:

- stable session/incarnation identity and pinned composition;
- immutable commands, user/assistant messages, session events, and turns;
- typed requirement interpretation and requirement-policy authority;
- Gateway invocation preparation/commit/unknown-outcome journal;
- command idempotency;
- rebuildable current session projection.

The store uses SQLite WAL, one connection per operation, foreign keys, busy
timeout, and `BEGIN IMMEDIATE` writes. One command transaction commits all
session-side facts and its idempotency response together.

## Gateway Boundary

Semantic interpretation uses the existing transport-neutral `AIGateway`.

The order is:

```text
private prompt rendering
-> persisted route decision
-> PREPARED session invocation journal
-> Gateway invocation
-> persisted Gateway receipt/result
-> strict typed output from private store
-> deterministic requirement/policy compilation
-> committed session turn/events/projection/idempotency
```

Raw provider response bodies remain in the private store. Session state keeps
only typed assistant text, structured interpretation/policy, safe refs,
normalized usage, provider-reported cost, closed failure codes, and evidence
class.

The model cannot choose object IDs, authority refs, permissions, policy
defaults, or session versions.

## Recovery

The session runtime does not assume a distributed transaction across
SessionStore, GatewayRecordStore, and private CAS.

- A committed session turn replays without a Gateway call.
- A PREPARED journal with exact Gateway result may reconcile into the same
  turn.
- A PREPARED journal without provable Gateway authority becomes
  `UNKNOWN_OUTCOME`; automatic retry is forbidden.
- Private CAS bytes without Session/Gateway authority are unreachable orphans.
- Mutable projection drift is rejected on read and repaired only by explicit
  rebuild.

This prefers a visible blocked state over a duplicate non-idempotent model
call.

## Streaming

Stage 1 exposes an async stream of committed `SessionEventV1` values. It yields
the user event immediately, then bounded assistant chunks and final result
events after governed Gateway output has been validated. Stage 1 does not
change the Gateway provider protocol to claim token-level provider streaming.
HTTP/SSE transport belongs to Stage 5.

## Evidence Classes

```text
REAL_SEMANTIC
MECHANISM_FIXTURE
```

Only a configured approved provider and separately authorized live execution
may produce `REAL_SEMANTIC`. Offline provider-boundary fixtures must remain
`MECHANISM_FIXTURE` and cannot close the semantic acceptance gate.

## Compatibility

- Spec 002 domain services and stores remain unchanged.
- FactoryControlStore, JobStore, GatewayRecordStore, and AgentMemoryStore
  retain separate ownership.
- Stage 1 does not execute Team work, Factory graphs, capabilities, delivery,
  release, attestation, or production publication.
- Existing PlanReview CLI/API/Web behavior remains unchanged.

## Consequences

Positive:

- conversation survives client and process restart;
- model use is auditable without exposing provider bodies;
- clarification and requirement authority become first-class;
- retries can distinguish replay, reconciliation, and unknown outcome;
- Stage 5 can render server truth rather than inventing client state.

Negative:

- another SQLite authority and reconciliation path are introduced;
- Gateway/provider token streaming remains deferred;
- a crash before Gateway authority is provable may require user/operator
  verification instead of automatic continuation;
- real semantic acceptance requires a separately approved live call.

## Validation

- strict/frozen runtime contracts and denied-field scans;
- same/different-key replay and concurrent command races;
- event monotonicity and bounded pagination;
- projection rebuild equality and corruption rejection;
- faults before/after route, PREPARED journal, Gateway commit, and session
  commit;
- exact provider-call counts across restart;
- private response non-disclosure;
- fixture-versus-real evidence-class enforcement;
- Stage 0, AI Gateway, Graph, PlanReview, generated, full-product, Node,
  privacy, production-blocked, and continuity regressions.
