# ADR 0012: Add A Graph-Engineered Eval Factory Control Plane

- Status: Accepted
- Date: 2026-08-06
- Task: `.trellis/tasks/08-06-graph-engineered-multi-agent-eval-factory/`
- Supersedes: none

## Context

The Eval Dataset Factory already has durable R1-R8 services, but they are
coordinated as fixed pipelines. The product now needs a user-specific plan,
specialist agents, editable plan gates, model governance, and crash-safe
orchestration without weakening the existing release and privacy boundaries.

Extending the existing `JobStore` into a global agent database would give two
different lifecycles one mutable authority. Treating a LangGraph checkpoint
as business truth would also make replay and audit depend on an execution
cursor rather than immutable domain records.

## Decision

Add an opt-in parent control plane under `eval_factory.agent_system`.

`FactoryControlStore` is the only authority for Factory runs, requirement
specifications, plan proposals and compiled plans, agent definitions and
tasks, leases and results, plan review, model route and receipt refs,
completion assessments, and delivery manifests.

The existing `JobStore` remains the only authority for delegated
`DatasetJob`, Item, StageRun, StageResult, R6-R8, and release state. The
stores exchange immutable `ObjectRef` values and never share mutable foreign
keys or duplicate current-head fields.

The parent LangGraph uses a separate SQLite checkpointer. Its state contains
only run identity, expected authority versions, bounded counters, and safe
object refs. Every node reloads `FactoryControlStore` and rejects drift.
Committed domain results are reused; checkpoint replay does not repeat an
external side effect.

Planning is split into a semantic proposal and a deterministic compiler.
Only the compiled plan is executable. The Execution Supervisor is
deterministic, uses fenced leases, and accepts only typed
`AgentResultEnvelopeV2` values.

All model work uses the transport-neutral AI Gateway. Routing order is:

```text
governance -> capability -> quality baseline -> health/latency/cost
```

A route is immutable before invocation. Provider failure requires a failure
receipt and successor route; silent fallback is forbidden.

CLI and Web call one `PlanReviewService`. An edit creates a successor plan,
recompiles it, and invalidates only affected downstream work. Neither adapter
reads SQLite or private object storage directly.

## Security And Data Boundaries

- Raw traces, prompt renderings, RAG text, model bodies, attachments,
  credentials, evaluator-only material, and unrestricted paths stay in
  private stores.
- Global records contain refs, hashes, closed states, counts, safe summaries,
  and bounded audit metadata only.
- Original user prompts remain source-span-bound immutable objects.
  `InferredUserIntentV2` is evidence-bound derivative data and cannot replace
  or overwrite them.
- New code may import attachment behavior only through
  `env_mock_agent.facade`.
- This control plane grants no production authority and does not change
  `PRODUCTION_RELEASE_BLOCKED`.

## Consequences

Positive:

- control, recovery, and audit have one explicit authority per aggregate;
- user-specific DAGs do not require dynamic LangGraph recompilation;
- model, prompt, RAG, budget, and permission policy are shared across agents;
- CLI/Web decisions have identical concurrency and idempotency behavior.

Negative:

- a new SQLite authority and explicit cross-store refs increase operational
  surface;
- every graph node must perform authority currentness checks;
- plan edits require immutable successor records and directed invalidation;
- provider failures are visible and may require user action instead of an
  automatic fallback.

## Validation

- strict/frozen/hash-stable public contracts and sensitive-field audits;
- transaction faults, same/different-key races, projection drift, and rebuild;
- separate database/checkpoint paths and no JobStore migration;
- graph crash/resume with committed-result reuse and drift rejection;
- deterministic route and successor-route tests;
- stale plan edit and concurrent decision tests;
- full R0-R8, frozen v1, Node, and production-blocked regressions.
