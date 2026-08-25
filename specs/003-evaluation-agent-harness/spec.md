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

### Unified Runtime Protocol

- **HR-016A**: Gateway and external Agent runtime observations MUST normalize
  into one strict, versioned lifecycle/tool/usage/failure event vocabulary.
- **HR-016B**: Normalized runtime events MUST contain only closed metadata,
  normalized tool families, bounded usage, and safe/private evidence refs.
  Raw Provider payloads, tool arguments/results, physical paths, credentials,
  and runtime session handles MUST NOT cross this boundary.
- **HR-016C**: Runtime readiness MUST distinguish protocol availability from
  credential and sandbox readiness. Unknown credentials or absent sandbox
  enforcement MUST NOT be reported as fully `READY`.
- **HR-016D**: The unified protocol is an observation boundary only. Existing
  Gateway receipts, runtime transcripts, Harness sessions, Factory/Team
  stores, and Graph journals retain their current authority.
- **HR-016E**: Persisting a normalized runtime event MUST atomically write its
  immutable runtime record, `SessionEventV1`, session version/sequence,
  rebuildable projection, and idempotency result.
- **HR-016F**: Runtime event persistence MUST enforce one start, contiguous
  source sequence, prior usage before terminal, no post-terminal event, and
  one-session ownership of each runtime event identity.

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
- **HR-020A**: Agent Shell SSE IDs MUST be committed Harness session
  sequences. `Last-Event-ID` and `after_sequence` MUST agree, replay MUST
  begin after the acknowledged sequence, and heartbeats MUST carry no event ID
  or persistent authority.
- **HR-020B**: Public runtime activity MUST omit private runtime source refs
  and expose only closed event kind, runtime ID, normalized tool family,
  bounded usage, and closed failure code.
- **HR-020C**: V1 source admission MUST accept exactly one `manifest.csv` plus
  the exact referenced lowercase `.jsonl` members through a bounded multipart
  command. Browser paths and additional fields MUST be rejected.
- **HR-020D**: The server MUST stream uploads into private staging, verify
  declared and observed byte counts and SHA-256 values, validate the exact
  manifest inventory and `raw_traj_v1` metadata, and remove staging on every
  terminal path.
- **HR-020E**: Validated bytes MUST publish to private content-addressed
  storage before one immutable SQLite admission transaction. One session has
  one admission; identical content MAY be reused by different sessions
  without sharing session admission authority.
- **HR-020F**: Public source results and OpenAPI MUST expose only safe relative
  names, media types, hashes, sizes, counts, Trace source refs, and Artifact
  Envelope refs. Physical paths, private content refs, full envelopes, and
  manifest/Trace bodies MUST remain private.
- **HR-020G**: Source admission MUST NOT start Graph execution or grant Team,
  Factory, artifact-head, approval, delivery, or release authority.

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

### Agent Shell Service Composition

- **HR-031**: The Agent Shell MUST rebuild Factory, Graph, Team, PlanReview,
  workspace, interaction, and delivery summaries through their owner APIs. It
  MUST NOT persist a duplicate aggregate business authority.
- **HR-032**: A READY turn MAY start the fixed Pack `1.2.0` Graph only when
  one immutable source admission exists and the READY user message explicitly
  carries every admitted source Artifact Envelope ref.
- **HR-033**: Source execution MUST use a private, rebuildable, hash-verified
  execution view. Stable Harness `trace-source/v2` refs MUST survive Pack
  registration without exposing the physical execution path. The private
  runner URI MAY be the deterministic execution-view `file://` URI, but it
  MUST NOT become public source identity or API output.
- **HR-034**: Every projected Graph binding MUST equal the current Harness
  requirement policy, Factory run/version, Team/roster/task graph/execution
  authority/checkpoint, and Graph checkpoint authority. Drift MUST fail
  closed rather than merge snapshots.
- **HR-035**: Team and Graph outbox reconciliation MUST remain owner-first and
  idempotent. The public reconcile command MUST bind session version,
  principal, idempotency key, and the pre-effect Graph binding ref before
  effects. Exact replay MUST retry an unadvanced binding and MUST NOT advance
  a committed successor binding. Rebuilding existing Pack authority MUST use
  the committed binding audit rather than a new reconcile audit.
- **HR-036**: A Graph-bound V1 session MUST reject a new requirement command
  before Harness mutation. Exact replay of the original command remains
  legal; requirement replacement and Graph supersession are later-version
  behavior.
- **HR-037**: `evalfactory agent serve` MUST require either complete explicit
  Agent Shell authority/config paths or explicit `--plan-review-only`
  compatibility mode. Missing configuration MUST fail before server startup.
- **HR-038**: The built-in Shell fixture Host MUST remain
  `DEVELOPMENT_FIXTURE_ONLY`, use the real Router/Gateway/private
  store/Harness loop, and replace only the Provider boundary. It MUST report
  `MECHANISM_FIXTURE` and perform no live Provider request.

### Conversation-First UI

- **HR-039**: The default React entry MUST be the Agent Shell. A Host that
  explicitly lacks `/api/harness/contract` MAY fall back to the compatible
  PlanReview-only workbench; other bootstrap failures MUST remain visible.
- **HR-040**: Browser code MUST validate unknown API/SSE payloads at one typed
  boundary and render only server-owned projections. It MUST NOT infer
  Harness, Factory, Graph, Team, review, or delivery authority.
- **HR-041**: Session, workspace, and Team-member selection MUST be
  URL-addressable and reloadable. A delayed projection refresh MUST resolve
  the current URL when it commits and MUST NOT restore a stale workspace.
- **HR-042**: SSE observation MUST reconnect from the latest committed
  sequence after disconnect or heartbeat-aware idle timeout. Initial and
  reconnecting states MUST remain distinct, and HTTP projection refresh is
  the recovery source of truth.
- **HR-043**: Browser source selection MUST require explicit confirmation of
  exactly one `manifest.csv` plus one or more JSONL files before upload.
- **HR-044**: Conversation, inline interaction cards, PlanReview, Team,
  Activity, Trace, task-domain summaries, and Delivery empty/result states
  MUST remain in one Shell with Chinese operational labels.
- **HR-045**: The Shell MUST provide stable named layout regions, visible
  focus, keyboard composer behavior, modal drawer focus containment/return,
  Escape close, reduced motion, and no horizontal page overflow at
  1440/1024/768/375 widths.
- **HR-046**: The Shell MUST use a light desktop-Agent hierarchy derived from
  the MIT-licensed CC-Haha interaction model without copying its brand assets:
  warm-neutral session navigation, restrained top modes, a readable document
  column, hugged right-aligned user messages, borderless full-column Agent
  prose, an inline Team status strip, and a floating composer. This visual
  composition MUST NOT change server-owned authority or browser permissions.
- **HR-047**: Primary navigation MUST expose user tasks rather than backend
  projection types: Conversation, Evaluation Workbench, and Run History.
  PlanReview, Trace, Task, Attachment, Rubric, Grading, Quality, Team,
  Activity, and Delivery remain URL-addressable contextual subviews.
- **HR-048**: An empty session MUST center the context composer; an active
  session MUST dock it below the transcript. The composer MAY provide local
  goal templates, Agent/Plan presentation modes, slash commands, bounded
  source-directory selection, and quotation of already-public Agent output.
  It MUST NOT claim project-directory, Skill, or plugin authority until the
  Host publishes typed permission and registry contracts.
- **HR-049**: Session removal MUST be an idempotent, version-bound
  `SESSION_CLOSED` transition. The default session list MUST hide closed
  sessions while direct lookup, events, and audit evidence remain available.
- **HR-050**: The Composer upper surface MUST contain only the text input.
  Its lower toolbar MUST place add and one mode selector on the left, and
  one model-and-reasoning trigger, voice input, and send controls on the
  right. The model trigger MUST combine model selection and a discrete effort
  slider constrained to the selected model's declared capability levels;
  voice input MUST use an available browser speech API or fail visibly.
- **HR-051**: Browser-local custom models MAY store only display name,
  provider, API protocol, model ID, Base URL, credential environment-variable
  name, supported effort levels, default effort, and source. Deserialization
  MUST reject unknown fields. Configuration MUST reject plaintext
  credentials, URL credentials/query/fragment, remote non-HTTPS endpoints,
  and remote endpoints without a credential reference; loopback HTTP MAY be
  used for local development.
- **HR-052**: Model/effort selection MUST be persisted per session, restored
  after reload, and clamped to the selected model's supported levels. Until
  the Host publishes typed model-registry, secret, routing-policy,
  Provider-client, and message-authority contracts, these values remain local
  preferences and MUST NOT claim or imply an executed Provider route.

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

HarnessSourceAdmissionStore
  private upload staging and manifest/trace CAS
  immutable per-session admission/member/envelope linkage and idempotency
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
- **HSC-013**: A valid `manifest.csv` and exact JSONL inventory produce one
  immutable per-session admission and safe Artifact Envelope refs; exact
  replay and cross-session content reuse create no duplicate physical bytes.
- **HSC-014**: Unsafe names, aliases, malformed CSV/JSONL, inventory or
  metadata mismatch, size/hash drift, budget excess, changed replay, injected
  faults, and corrupted SQLite/CAS authority fail closed.
- **HSC-015**: API, OpenAPI, SQLite metadata, logs, and diagnostics expose no
  physical path or raw manifest/Trace content; Graph execution remains
  untouched by admission.
- **HSC-016**: READY plus admitted and explicitly confirmed source authority
  starts Pack `1.2.0`, creates one Graph binding, and projects the current
  Factory/Graph/Team/PlanReview state.
- **HSC-017**: Restart and exact reconcile replay preserve the same aggregate
  fingerprint and create no duplicate session, Team, Graph, or Provider
  effect.
- **HSC-018**: Team, member, workspace, delivery, and reconcile HTTP routes
  return closed errors and expose no private body, content ref, physical path,
  or unrestricted payload.
- **HSC-019**: The React application opens to a persistent session rail,
  conversation transcript, and fixed composer, with PlanReview mounted as a
  contextual workspace.
- **HSC-020**: Deep-link reload preserves session/workspace identity and API
  restart returns SSE to online from the committed cursor without duplicate
  commands.
- **HSC-021**: Mobile drawers trap focus, close with Escape, restore their
  trigger, and honor reduced-motion preferences.
- **HSC-022**: Full fixture Browser E2E passes at 1440, 1024, 768, and 375
  pixels with no horizontal overflow or denied private marker in DOM,
  diagnostics, console metadata, or network paths.
- **HSC-023**: The PlanReview-only Host retains the existing browser behavior
  and all compatibility scenarios.
- **HSC-024**: Stage 5.4 evidence remains `MECHANISM_FIXTURE`; Delivery
  displays an explicit empty state until Stage 6 creates a package.
- **HSC-025**: Stage 5.4.1 visual evidence passes Web check/test/build, the
  seven-scenario Full Agent Shell Browser suite, and all 22 PlanReview
  compatibility scenarios. Screenshots at 1440/1024/768/375 show no
  horizontal overflow and retain the same private-data boundary.
- **HSC-026**: Stage 5.4.2 preserves all workspace deep links and source,
  review, Team, reconnect, responsive, and privacy scenarios after primary
  navigation consolidation and context-composer enhancement.
- **HSC-027**: Stage 5.4.3 preserves immutable session history through
  soft-close, verifies delete confirmation focus/Escape/return, and renders
  the revised Composer without input focus borders, control overlap, or
  horizontal overflow at 1440/1024/768/375 widths.

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
