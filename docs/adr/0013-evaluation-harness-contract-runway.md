# ADR 0013: Add An Evaluation Harness Contract Runway

- Status: Accepted
- Date: 2026-08-17
- Beads: `env_mock_agent-ujc.12`
- Trellis: `.trellis/tasks/08-17-harness-contract-runway-v1/`
- Depends on: ADR 0008, ADR 0011, ADR 0012

## Context

The existing Eval Dataset Factory has durable domain services, a Graph
Engineering control plane, specialist Agent definitions, immutable
`ObjectRef` authority, PlanReview, and production-neutral candidate delivery.
Its product and contracts remain specialized around a Trace-driven evaluation
vertical.

The approved product direction is a conversation-first Evaluation Dataset
Agent Harness. The first edition keeps the existing Trace vertical, while a
later platform edition may install professional domain Packs. The expensive
boundaries to retrofit are session events, capability invocation, artifact
exchange, collaborative Team authority, permission decisions, Blueprint
connectivity, projections, and composition identity.

Implementing runtime behavior before those contracts exist would hard-code
Trace assumptions into the Harness. Building a dynamic plugin platform before
one vertical is proven would instead overgeneralize unknown domains.

## Decision

Add a provider-neutral, additive Stage 0 contract runway under
`eval_factory`:

```text
eval_factory.harness
  session event and command contracts
  capability definition/provider/consumer contracts
  universal Artifact Envelope
  Balanced Autonomy permission contracts
  server-owned projection definitions
  static Pack registration and pinned composition

eval_factory.team
  stable Team/member identity
  revisioned task DAG
  claims and leases
  P2P coordination messages
  artifact subscriptions, conflicts, context projections, and checkpoints

eval_factory.blueprints
  fixed EvaluationBlueprintV1
  schema and capability data-flow validation

eval_factory.packs.generic_agent_trace
  one statically registered built-in Pack
```

All new authorities are strict, frozen Pydantic contracts and reuse existing
`ObjectRef`, `ContractAudit`, and deterministic canonical hashing. The static
Pack maps existing EDF service boundaries without moving business logic.

The Harness Kernel contains no Trace-specific field. Trace assumptions live
only in the built-in `generic-agent-trace-pack`.

## Authority Boundaries

- Existing `FactoryControlStore`, `JobStore`, LangGraph checkpoints, private
  CAS, output CAS, Gateway records, PlanReview, and memory stores remain
  unchanged.
- Stage 0 creates no persistence, provider call, Team scheduler, graph
  transition, API, CLI command, UI behavior, release decision, or production
  authority.
- Collaboration messages carry bounded coordination-body refs and Artifact
  Envelope refs. They cannot carry canonical artifact bodies, user approvals,
  or grant mutations.
- Balanced Autonomy allows only exact work inside a pinned Team graph, grants,
  providers, data scopes, and budgets. Source admission, budget widening,
  external execution, authority widening, destructive action, and release
  require explicit successor approval.
- React and other clients will consume server-owned projections; Stage 0 does
  not implement those reducers or transports.

## Static First, Dynamic Later

V1 uses one in-process `StaticPackRegistry` with exact manifest and composition
refs. It does not implement plugin discovery, installation, upgrade, removal,
dependency solving, marketplace behavior, arbitrary schema authoring, or a
generic graph editor.

V2 may replace static registration with lifecycle-managed registration only
after V1 freezes semantic, recovery, and delivery evidence. Exact V1
composition must remain replayable.

## Spec Compatibility Gate

The active `002-eval-dataset-factory` specification lists free-form Team
runtime behavior outside its first-release scope. This ADR authorizes the
additive Stage 0 contract spike only; it does not silently change that runtime
scope.

Before Stage 1 or any later task changes default product behavior, the active
Spec Kit requirement set must be revised or superseded to cover the approved
Harness session, Team, capability, permission, and version strategy.

## Consequences

Positive:

- future session, Team, and Pack work depends on stable typed seams;
- the current Trace vertical becomes a Pack instead of the Kernel definition;
- domain services remain independently invocable and authoritative;
- permission and replay identity are established before side effects exist;
- V1 can remain a focused product after V2 adds dynamic composition.

Negative:

- the contract surface grows before user-visible behavior exists;
- Stage 1 must preserve exact composition and authority refs;
- static Pack declarations require compatibility maintenance as existing
  domain contracts evolve;
- later runtime work requires a formal Spec Kit scope revision.

## Validation

- strict/frozen/unknown-field/version tests;
- audit-independent deterministic identity tests;
- contiguous session event and exact command idempotency tests;
- Artifact Envelope and no-inline-payload tests;
- Team DAG, write ownership, message, grant, conflict, and checkpoint tests;
- Balanced Autonomy allow/approval/deny matrix;
- static Pack exact-resolution and drift rejection;
- Blueprint schema connectivity and acyclic data-flow tests;
- no Trace-specific Harness Kernel field;
- existing ObjectRef/CAS, Graph Engineering, PlanReview, import, privacy, and
  production-blocked regressions.
