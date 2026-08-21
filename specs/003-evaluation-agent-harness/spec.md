# Feature Specification: Evaluation Dataset Agent Harness

**Feature Branch**: `main`

**Created**: 2026-08-17

**Status**: Approved

**Approved**: 2026-08-17

**Revision**: v1 - Vertical Edition Harness runtime

**Predecessor**: `specs/002-eval-dataset-factory/spec.md`

**Input**: Turn the existing Trace-driven Eval Dataset Factory into a
conversation-first Agent Harness whose domain capabilities remain independently
invocable and whose first product profile produces generic Agent evaluation
data.

## Supersession Boundary

This specification does not rewrite or invalidate Spec 002. Spec 002 remains
the authoritative domain contract for Trace ingestion, provenance, labeling,
task/attachment/evaluation construction, quality, and release.

This specification supersedes only Spec 002's product-shell and orchestration
scope:

- the product entry is now a persistent Agent session rather than an internal
  DatasetJob request;
- the existing Trace vertical is mounted as the static
  `generic-agent-trace-pack`;
- Harness session, capability, artifact, permission, projection, Team, and
  Blueprint authorities are first-class;
- future dynamic Pack lifecycle remains a V2 Platform Edition concern.

## Product Goal

The V1 product converts a natural-language evaluation-data goal plus admitted
sources into an auditable candidate dataset through one persistent interaction
loop:

```text
user message
  -> durable Harness session event
  -> governed Gateway requirement interpretation
  -> clarification or typed requirement authority
  -> fixed EvaluationBlueprintV1
  -> approved collaborative Team and Factory graph
  -> typed capability/artifact execution
  -> PlanReview and quality rework
  -> candidate dataset delivery
```

Conversation coordinates the work but is never business authority. Canonical
requirements, policies, plans, artifacts, assessments, decisions, and delivery
remain strict versioned objects.

## User Stories

### User Story 1 - Start And Continue A Persistent Agent Session (P1)

An evaluation engineer creates a session, describes a dataset goal, closes the
client, and later reopens the same conversation without losing messages,
Gateway decisions, usage, or current interpretation.

**Independent Test**: Create a session, submit a message, restart the service,
reload transcript/projection/events, and continue with the next command.

**Acceptance Scenarios**:

1. A session pins one exact Harness composition and never resolves a newer
   Pack implicitly.
2. User and assistant messages are immutable, content-hashed, and referenced
   by append-only session events.
3. Event sequence and authority versions are monotonic and reconstruct the
   same server projection after restart.
4. A repeated exact command returns the original response; changed reuse of an
   idempotency key conflicts.

### User Story 2 - Clarify Before Compiling Requirements (P1)

The Agent interprets the user's goal through the governed AI Gateway. Missing
goal, source, target capability, quality, budget, or delivery information
produces explicit clarification rather than fabricated assumptions.

**Independent Test**: Submit one incomplete and one complete requirement. The
first produces a durable clarification request; the second produces typed
requirement and policy authorities.

**Acceptance Scenarios**:

1. Model output must validate against one closed requirement-interpretation
   schema loaded from private material.
2. Clarification carries bounded questions and missing-field codes.
3. A ready interpretation carries explicit goals, constraints, assumptions,
   source expectations, target capability, quality intent, budget, and
   delivery intent.
4. Deterministic compilation owns IDs, refs, policy defaults, hashes, and
   currentness; model output cannot grant permission.

### User Story 3 - Observe Governed Model Use (P1)

The user can inspect which route, model profile, prompt template, receipt,
usage, provider-reported cost, result, and evidence class produced an
interpretation without seeing provider-private response bodies.

**Independent Test**: Execute one successful and one failed Gateway turn and
inspect the session projection after restart.

**Acceptance Scenarios**:

1. The route is persisted before invocation.
2. Success binds exact invocation, receipt, result, model profile, prompt, and
   output authority.
3. Failure stores a closed failure code and zero/private refs according to
   Gateway contracts; no raw exception crosses the boundary.
4. Fixture-backed results are labeled `MECHANISM_FIXTURE` and cannot satisfy
   `REAL_SEMANTIC` acceptance.

### User Story 4 - Recover Without Duplicate External Calls (P1)

The runtime can restart at every command/Gateway/session commit boundary
without silently repeating a possibly non-idempotent model call.

**Independent Test**: Inject faults before Gateway invocation, after Gateway
commit, and before session commit; reopen stores and reconcile.

**Acceptance Scenarios**:

1. The session store commits a PREPARED invocation journal before calling the
   Gateway.
2. If Gateway authority exists after restart, reconciliation completes the
   same turn without another provider call.
3. If a PREPARED invocation has no provable Gateway result, it becomes
   `UNKNOWN_OUTCOME` and blocks automatic retry.
4. Reads never auto-repair immutable drift; explicit projection rebuild is the
   only mutable projection repair.

## Functional Requirements

### Harness Session Runtime

- **HR-001**: Session identity MUST include a stable session ID, incarnation
  ID, composition ref, revision, and closed status.
- **HR-002**: The session store MUST use SQLite WAL, foreign keys, busy timeout,
  one connection per operation, and `BEGIN IMMEDIATE` for writes.
- **HR-003**: Commands, messages, events, turns, interpretations, policies,
  Gateway journals, idempotency rows, and projection heads MUST have explicit
  ownership and versioning.
- **HR-004**: Every model-visible value MUST be reconstructable from a durable
  message/event or immutable artifact ref.
- **HR-005**: Event pages MUST be bounded by offset/sequence and limit.
- **HR-006**: Session projection MUST be server-owned and rebuildable from
  immutable session facts.

### Gateway Agent Loop

- **HR-007**: All semantic interpretation MUST use the transport-neutral
  `AIGateway`; Harness code MUST NOT import vendor SDKs.
- **HR-008**: Prompt rendering and provider output bodies MUST remain in the
  private object store.
- **HR-009**: The model may return only the typed semantic proposal. The
  deterministic compiler owns requirement/policy authority.
- **HR-010**: A route and PREPARED invocation journal MUST commit before the
  provider call.
- **HR-011**: A successful turn MUST retain route, invocation, receipt, result,
  model profile, prompt, usage, reported-cost, output, and evidence-class refs.
- **HR-012**: Silent provider fallback is forbidden. A successor route requires
  the existing Gateway failure-receipt protocol.

### Clarification And Requirement Authority

- **HR-013**: Missing required fields MUST produce
  `CLARIFICATION_REQUIRED`, not inferred values.
- **HR-014**: Clarification questions and assistant text MUST be bounded,
  content-safe, and stored as message authority.
- **HR-015**: `READY` output MUST compile immutable Harness requirement and
  requirement-policy objects.
- **HR-016**: Stage 2 alone adapts Harness requirement authority into existing
  domain capability requests; Stage 1 MUST NOT run the Factory pipeline.

### Typed Service API

- **HR-017**: One service API MUST support create, get, list, post-message,
  event-tail, projection, reconcile, and explicit projection rebuild.
- **HR-018**: `stream_turn` MUST yield only committed events, including the
  user message, bounded assistant chunks, final assistant message, Gateway
  result, and requirement/clarification result.
- **HR-019**: A client disconnect MUST NOT cancel or roll back committed
  authority.
- **HR-020**: API/service errors MUST use closed codes without SQL, path,
  provider payload, credentials, or raw exception text.

### Collaborative Team Runtime

- **HR-021**: Stage 3 MUST use a dedicated SQLite `TeamStore` that is
  physically distinct from Harness sessions, Factory control, JobStore,
  Gateway, and later graph-checkpoint databases.
- **HR-022**: Team, roster, graph, execution-authority, task event, message,
  Capability result, Artifact Head, subscription, conflict, projection,
  checkpoint, outbox, and idempotency authority MUST be immutable and
  content-addressed; mutable current heads MUST rebuild from history.
- **HR-023**: Claims, attempts, fencing tokens, event versions, Team message
  sequence, and Blackboard sequence MUST allocate under `BEGIN IMMEDIATE`.
  Provider, model, filesystem, or other owner work MUST NOT run inside a Team
  transaction.
- **HR-024**: Direct peer messages MUST bypass Coordinator relay and may carry
  only bounded coordination-body refs plus visible Artifact Envelope refs.
  Messages cannot mutate approvals, grants, budgets, graph authority, or
  canonical artifact bodies.
- **HR-025**: Successful Capability execution MUST commit the exact call,
  result, task completion, successor Envelope/Head, authority data scope,
  outbox, and idempotency response atomically. Non-success MUST publish no
  canonical Artifact Head.
- **HR-026**: Every member MUST retain a distinct Harness session and receive
  only assigned tasks, immediate dependency/dependent neighbors, authorized
  Artifact Envelopes, relevant direct/broadcast messages, subscriptions, and
  acceptance checks.
- **HR-027**: Team outbox reconciliation MUST be TeamStore-first and
  session-second. Exact reconciliation replay MUST not duplicate either the
  Team source record or member session event.
- **HR-028**: Coordinator may dispatch, checkpoint, converge, and create an
  in-scope graph/authority successor, but MUST NOT relay ordinary peer
  messages or publish Requirement, Trace, Task, or Quality Artifact Heads.
- **HR-029**: The Stage 3 mechanism fixture MUST execute typed
  Requirement-to-Trace-to-Task-to-Quality handoff, direct Quality challenge,
  bounded Task/Quality successor revisions, final convergence, checkpoint,
  member-session reconciliation, and restart recovery.
- **HR-030**: Initial session-owned input authority required by a Capability
  role, including `evaluation-requirement`, MUST enter Blackboard through an
  idempotent unowned-input seed operation. It cannot impersonate task output
  ownership or bypass exact Stage 2 input-role validation.

## Authority And Storage Boundaries

```text
HarnessSessionStore
  conversation commands/messages/events/turns
  interpretation and policy refs
  Gateway invocation journal
  rebuildable session projection

GatewayRecordStore
  route/invocation/receipt/result authority

FactoryPrivateObjectStore
  prompt rendering, provider response, typed semantic output bytes

FactoryControlStore / JobStore
  unchanged; not called by Stage 1

AgentMemoryStore
  unchanged; Stage 1 does not write long-term memory

TeamStore
  Team composition, task events, P2P mailbox, Blackboard, convergence
  rebuildable Team heads, checkpoint, outbox, Team idempotency

HarnessCapabilityRuntime
  exact AGENT_TOOL binding, Balanced Autonomy, provider-neutral invocation
```

Cross-store composition uses immutable refs. No distributed transaction is
assumed. Recovery reconciles source authority in order and fails closed when
proof is missing.

## Non-Functional Requirements

- **HNFR-001**: Python 3.12, Pydantic v2, SQLite, and existing Gateway contracts
  remain the baseline.
- **HNFR-002**: Identical semantic inputs, composition, prompt, route facts,
  and deterministic compiler version MUST produce identical behavior refs.
- **HNFR-003**: Every new deep service module requires at least 80% branch
  coverage.
- **HNFR-004**: Real semantic acceptance requires an explicitly configured,
  approved provider and a separately authorized live call.
- **HNFR-005**: Fixture evidence is mechanism evidence only.
- **HNFR-006**: Stage 1 grants no Team, Factory execution, delivery, release,
  attestation, or production authority.
- **HNFR-007**: Existing Spec 002 safety, privacy, provenance, and
  `PRODUCTION_RELEASE_BLOCKED` invariants remain unchanged.

## Stage 1 Out Of Scope

- Collaborative Agent Team runtime and P2P mailbox execution.
- Capability provider adapters over all EDF domains.
- FactoryControlGraph execution.
- React conversation shell, SSE/HTTP transport, and desktop packaging.
- Dynamic Pack lifecycle, schema registry, Marketplace, or second Pack.
- Live semantic call without explicit user authorization.

## Success Criteria

- **HSC-001**: Session create/reopen/continue passes across process restart.
- **HSC-002**: Exact command replay performs zero duplicate Gateway calls.
- **HSC-003**: Clarification and READY interpretations are durable and typed.
- **HSC-004**: Projection rebuild is byte-equivalent to the current projection.
- **HSC-005**: Fault injection proves reconcile-or-block behavior at every
  Gateway boundary.
- **HSC-006**: Gateway usage/provider/model/prompt/evidence class is inspectable
  without private response disclosure.
- **HSC-007**: A separately authorized real-provider run satisfies
  `REAL_SEMANTIC`; until then Stage 1 remains semantically pending.
- **HSC-008**: Existing Stage 0, AI Gateway, Graph, PlanReview, Spec 002 safety,
  generated, static, and full-product regressions pass.
- **HSC-009**: Requirement and Trace Artifact Heads unblock Task under one
  current Team authority; Task then unblocks Quality.
- **HSC-010**: Quality sends Task a direct challenge, Task and Quality publish
  exact revision-2 successors, and Coordinator reaches COMPLETE without
  specialist authorship or message relay.
- **HSC-011**: Claim/lease/heartbeat/expiry/retry/cancellation/stale-fence,
  mailbox, subscription, conflict, projection, checkpoint, rebuild, and exact
  replay tests pass under both configured hash seeds.
- **HSC-012**: The Stage 3 fixture remains `MECHANISM_FIXTURE`; real Team
  semantic evidence remains `REAL_SEMANTIC_PENDING` until a separately
  authorized Provider execution is available.

## Stage 4 Default Factory Graph

- **HS4-001**: `evalfactory agent run` MUST resolve exact Pack `1.2.0`,
  READY Harness sessions, current Team/Blueprint authority, Factory request,
  Graph journal, and async checkpoint stores before execution.
- **HS4-002**: The default command MUST use `FactoryDatasetGraphRuntime`.
  `FactoryDatasetRuntime.advance()` is available only with
  `--direct-runtime-compatibility`; missing Graph authority MUST fail closed.
- **HS4-003**: All ten first-party capabilities MUST execute through
  `TeamCapabilityRunner` and `HarnessCapabilityRuntime` while canonical owner
  stores retain business authority.
- **HS4-004**: Trace dispositions MAY only narrow the concrete Team graph.
  The successor MUST preserve Team incarnation, composition, roster, grants,
  budgets, usage monotonicity, and static Blueprint authority.
- **HS4-005**: Capability requests after narrowing MUST bind the current
  successor `TeamTaskV1` ref and MUST reject any changed capability definition.
- **HS4-006**: `CoreVerticalResultV2` MUST retain the current governed global
  planning route plus two semantic routes per candidate. A deterministic
  non-candidate frontier therefore has one route; no route may be fabricated.
- **HS4-007**: Exhausted Team failure MUST commit
  `PlannerAssessmentV2(ESCALATE / TERMINAL_WORK_FAILED)` instead of looping as
  missing work. A committed non-success Batch aggregate projects a stable
  `BLOCKED / NONE` product view and exact replay performs no Graph transition.
- **HS4-008**: Zero candidates MUST commit `NO_ELIGIBLE_ITEMS`; one candidate
  MUST commit `INSUFFICIENT_BATCH_CANDIDATES` without cross-item provider work.
  Neither path may enter Final Delivery.
- **HS4-009**: Two eligible fixture candidates MUST traverse global, domain,
  and final PlanReview on one thread, then produce current Batch Quality,
  candidate JSON/JSONL, immutable delivery manifest, and `COMPLETED`.
- **HS4-010**: Complete zero-, one-, and two-candidate CLI paths MUST pass
  under `PYTHONHASHSEED=1` and `321`. Evidence remains `MECHANISM_FIXTURE`;
  live Provider execution requires separate authorization.
