# ADR 0015: Add A Recoverable Collaborative Team Runtime

- Status: Accepted
- Date: 2026-08-18
- Spec: `specs/003-evaluation-agent-harness/spec.md`
- Trellis: `.trellis/tasks/08-17-collaborative-team-runtime-v1/`
- Depends on: ADR 0013, ADR 0014

## Context

Stage 0 froze Team, task DAG, mailbox, Artifact, permission, context, and
checkpoint contracts. Stage 2 made ten existing EDF capabilities independently
invocable through one provider-neutral Runtime. Durable peer collaboration
still required one business authority that could survive restart without
merging five independent Agent contexts or moving owner persistence into the
Harness.

Using `FactoryControlStore`, `JobStore`, or Harness session tables for Team
work would mix distinct lifecycles. Letting the Coordinator relay peer traffic
would make it a hidden shared context. Running an owner while SQLite is open
would make external recovery depend on an impossible distributed transaction.

## Decision

Add a dedicated SQLite `TeamStore` with one public facade and internal modules
owned by invariant:

```text
_store_db.py             SQLite schema, connections, canonical records,
                         transaction/idempotency/outbox primitives
_store_authority.py      Team composition and fenced task lifecycle
_store_collaboration.py  P2P mailbox, Blackboard, subscriptions, conflicts
_store_projection.py     member context, checkpoint, rework, rebuild/recovery
store.py                 stable public TeamStore facade
```

The split is internal. Callers use `TeamStore`; no service issues SQL.

## Transaction And Recovery Order

Every Team mutation uses `BEGIN IMMEDIATE` and commits its immutable records,
current cache, Team outbox, and operation idempotency response together.
Claims allocate attempts; leases allocate monotonic fences; messages and
Blackboard events allocate contiguous Team-local sequences.

Capability execution order is:

```text
read current Team authority
-> claim and acquire fenced lease
-> close Team transaction
-> invoke HarnessCapabilityRuntime through AGENT_TOOL
-> reopen Team transaction and revalidate Team/graph/authority/member/fence
-> commit result, task event, successful Envelope/Head, outbox, idempotency
```

Owner work is never performed inside SQLite. A crash after an idempotent owner
effect reuses the exact Team incarnation/graph/task/attempt/fence operation
key. An unprovable non-idempotent outcome remains blocked.

## Collaboration Boundary

Direct peer messages are committed for sender and recipient sessions without
Coordinator mediation. They carry only a `team-message-body` ref and visible
Artifact Envelope refs. Approval, grant, budget, graph, and canonical body
authority cannot enter through mailbox content.

Artifact Head write ownership comes only from the current task graph. A
successful successor binds both predecessor Envelope and predecessor Head.
Initial session-owned inputs required by exact Stage 2 roles enter through an
idempotent unowned-input seed operation; the seed cannot target any
task-owned output head.

Each member context is rebuilt from current immutable Team authority and
contains only its assigned task neighborhood, authorized Envelope refs,
relevant messages, subscriptions, and acceptance checks.

## Coordinator Boundary

Coordinator owns convergence and bounded in-scope rework only. A direct
Quality challenge resets the approved Task and Quality work under successor
graph, authority, and Team revisions without widening grants or budgets.
Coordinator cannot publish specialist Artifact Heads or relay the challenge.

## Cross-Store Reconciliation

TeamStore commits first. `TeamSessionReconciler` appends typed Team/checkpoint
events to each existing Harness session idempotently, then marks that
outbox/session delivery. No distributed transaction is assumed.

## Compatibility

- Existing Stage 0 contracts are additive and remain accepted.
- Stage 1 session and Gateway ownership are unchanged.
- Stage 2 Pack `1.0.0` and `1.1.0` exact resolution is unchanged.
- FactoryControlStore, JobStore, Agent Memory, private/output CAS, and future
  LangGraph checkpoints remain separate authorities.
- Stage 3 adds no UI, transport, release, attestation, or production authority.

## Evidence

The offline fixture proves:

```text
evaluation-requirement seed -> Requirement -> Trace -> Task -> Quality
Quality --direct CHALLENGE--> Task
Task revision 2 -> Quality revision 2 -> Coordinator COMPLETE
Team outbox -> five independent Harness sessions
restart -> current heads and checkpoint rebuild
```

This is `MECHANISM_FIXTURE`, not semantic evidence. No live Provider call is
authorized by this ADR; `REAL_SEMANTIC_PENDING` remains explicit.
