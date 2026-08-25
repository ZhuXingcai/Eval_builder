# ADR 0020: Compose Agent Shell Reads Around Existing Authorities

- Status: Accepted
- Date: 2026-08-23
- Spec: `specs/003-evaluation-agent-harness/spec.md`
- Trellis: `.trellis/tasks/08-17-conversation-first-agent-shell-v1/`
- Beads: `env_mock_agent-ujc.20`
- Depends on: ADR 0016, ADR 0018, ADR 0019

## Context

Harness sessions, bounded source admission, Team, Factory, Graph journal,
PlanReview, and candidate output already own separate durable state. The Agent
Shell needs one coherent product projection and one message-to-Graph start
path without copying those owners into another database or introducing a
distributed transaction.

## Decision

Add `AgentShellCompositionService` as an owner reader and coordinator. It:

- rebuilds the public session view from current owner APIs;
- validates that Graph bindings still match current Harness, Factory, Team,
  checkpoint, and Pack authority;
- projects bounded Factory, Graph, Team, PlanReview, workspace, interaction,
  and delivery summaries;
- reconciles Team and Graph outboxes into Harness events in owner-first order;
- starts the fixed `generic-agent-trace` Pack `1.2.0` Graph only after a READY
  turn and successful source preflight.

The composition service stores no copied business state. Its
`source_fingerprint` is a deterministic digest of the owner-backed public
projection.

## Source Confirmation And Execution

A READY interpretation is insufficient to start execution. The user message
that produced the current READY authority must explicitly carry every
Artifact Envelope ref from the session's immutable source admission:

```text
admitted source Envelope refs
  subset of READY user-message Envelope refs
```

The source store materializes a rebuildable private execution directory from
CAS using hard links, verifies its exact inventory and hashes, and passes
stable Harness `trace-source/v2` refs into the Pack source registry. Physical
execution paths never cross the public API.

Graph-bound V1 sessions accept only exact replay of the original message
command. A new requirement message is rejected before Harness mutation because
V1 owns one immutable Graph binding per session and does not yet define
requirement replacement or graph supersession.

## Recovery And Idempotency

Cross-store writes remain ordered and idempotent:

```text
Team owner commit
  -> TeamSessionReconciler
  -> Harness member-session event

Graph checkpoint commit
  -> FactoryGraphSessionReconciler
  -> Harness main-session event
```

The public reconcile command first records a content-addressed intent in the
Harness idempotency authority. Exact replay may rerun both reconcilers because
their owner outboxes and Harness event keys are independently idempotent. A
changed request or principal under the same key conflicts before reconciliation.

The intent also records the Graph binding ref observed before effects. A retry
advances the Graph only while that binding remains current. If a successor
binding already exists, the retry drains owner outboxes without starting the
next transition. This closes both crash windows: a failure after intent commit
can retry the original binding, while a failure after Graph commit cannot
double-advance it.

Rebuilding an existing Pack product uses the committed binding audit rather
than the new reconcile-command audit. Team bootstrap refs and idempotency
hashes therefore remain stable. Harness source identity remains the admitted
`trace-source/v2` ref, while the private `TraceSourceRef.source_uri` remains
the deterministic `file://` execution-view path required by the R1 runner.
That URI is not part of the public Shell projection.

## Product Host

`evalfactory agent serve` now has two explicit modes:

- the default unified Agent Shell requires every Store/workspace path plus one
  strict `AgentShellFixtureHostConfigV1`;
- legacy PlanReview-only hosting requires `--plan-review-only`.

The full V1 host is deliberately
`DEVELOPMENT_FIXTURE_ONLY`. Requirement interpretation uses the real Router,
Gateway record store, private object store, and Harness loop with only the
Provider boundary replaced by a deterministic fixture. This evidence remains
`MECHANISM_FIXTURE` and makes no real semantic claim.

## Public Boundary

The aggregate API may expose only closed enums, bounded counts, safe names and
hashes, immutable refs, Team task/message metadata, checkpoint metadata,
PlanReview state, and candidate delivery inventory summaries. It excludes raw
traces, message-body refs, provider bodies, prompts, grader rules, private
content refs, credentials, physical paths, and bundle paths.

## Consequences

- Restart reconstructs the same aggregate projection from owner stores.
- READY plus admitted and explicitly confirmed sources enters the existing
  Pack `1.2.0` Graph and pauses at real PlanReview authority.
- Missing, stale, duplicate, or drifted owners fail closed through typed HTTP
  errors.
- Stage 6 spreadsheet/ZIP delivery, dynamic Packs, real Provider execution,
  and production release remain out of scope.
