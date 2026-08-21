# ADR 0016: Make FactoryControlGraph The Recoverable Product Control Plane

- Status: Accepted
- Date: 2026-08-19
- Spec: `specs/003-evaluation-agent-harness/spec.md`
- Trellis: `.trellis/tasks/08-17-factory-graph-closure-v1/`
- Depends on: ADR 0013, ADR 0014, ADR 0015

## Context

The V1 Harness already had durable conversation authority, ten independently
callable Pack capabilities, and a recoverable collaborative Team. Dataset
execution still used `FactoryDatasetRuntime.advance()` as the product path,
while each LangGraph node called that same monolith. The graph therefore did
not own the next legal transition and could not prove which Factory, Team,
Blueprint, Blackboard, or review authority it had observed.

LangGraph checkpoints cannot become business truth: Factory, Team, Harness,
Job, and domain stores own their records independently and no distributed
transaction exists across them.

## Decision

Make `FactoryControlGraph` the default product control plane. Bind one graph
thread to exact Harness, requirement, Pack composition, Blueprint, Factory,
Team, checkpoint, and Blackboard refs through
`HarnessGraphExecutionBindingV1`.

Every consequential node uses this protocol:

```text
resolve current owner authority
-> commit PRE_TRANSITION to FactoryGraphJournalStore
-> execute one node-owned idempotent command outside SQLite transactions
-> resolve owner authority again
-> commit successor binding and POST_TRANSITION
-> project the checkpoint to HarnessSessionStore through an outbox
```

`FactoryGraphJournalStore` is a dedicated WAL SQLite authority. LangGraph
uses a physically separate `AsyncSqliteSaver` only for continuation cursor
state. Neither source may repair Factory, Team, Harness, Job, or domain truth.

## Authority And Team Rules

`EvaluationBlueprintV1` pins the baseline Team composition, roster, task
graph, grants, budgets, and data flow. Current Team authority may be a
validated successor only when identity and composition remain fixed, grants
and budgets do not widen, usage is monotonic, and data scope grows solely from
authorized Artifact Envelopes.

The Coordinator remains taskless and convergence-only. A non-coordinator
`control` member owns PlanReview and delivery Artifact Heads. Empty task
grants are valid, but any task-bound action still requires an assigned task.

## PlannerAssessment And Delivery

`PlannerAssessmentV2` is compiled from committed Factory and Team facts and
persisted atomically with the Factory run successor. Graph routing maps only
the committed assessment action.

`FINISH` means the planning and specialist execution frontier is complete
and the graph may enter delivery closure. It does not mean the Factory run is
already complete. The final delivery review, immutable JSON/JSONL bundle,
`CandidateDatasetDeliveryManifestV2`, and
`FactoryRunCompletionV2.COMPLETE` remain mandatory before the run reaches
`COMPLETED`. Requiring the manifest before `FINISH` would create a control
cycle because the manifest is produced only after the assessment route enters
delivery.

## Recovery

- PRE with unchanged owners: reuse the PRE and rerun the exact idempotent
  command.
- PRE with a provable owner successor: reconcile and write POST.
- PRE with an unprovable outcome: record `VERIFICATION_REQUIRED` and fail
  closed.
- POST with a missing Harness event: replay the journal outbox.
- PlanReview resume: accept only the exact current resumed result and preserve
  the same session, Factory run, Team incarnation, composition, and thread.

## Compatibility

The direct dataset runtime remains available only behind the explicit
`--direct-runtime-compatibility` flag. Missing Graph binding, journal,
checkpointer, Team authority, Blueprint, or exact Pack registration blocks;
the product path never silently falls back.

Stage 4 adds no Stage 5 UI, Stage 6 spreadsheet/ZIP packaging, attestation, or
production-release authority. No live Provider call is authorized. Mechanism
fixtures remain `MECHANISM_FIXTURE`, and `REAL_SEMANTIC_PENDING` remains
explicit.

## Implementation Closure: Default Pack 1.2.0 CLI

As of 2026-08-20, `evalfactory agent run` mounts the complete first-party Pack
`1.2.0` composition by default:

```text
READY Harness sessions
-> manifest source admission
-> ten first-party Providers and owner material
-> deterministic Team task graph and non-widening refinement
-> Blueprint and current Team checkpoint
-> Graph binding, journal, and AsyncSqliteSaver
-> PlanReview continuation and terminal candidate delivery
```

The command requires explicit Graph runtime, Harness, Team, capability
request, journal, and checkpoint paths. Missing authority fails closed before
execution. `FactoryDatasetRuntime.advance()` remains reachable only through
`--direct-runtime-compatibility`.

Narrowing may change a task's content-addressed ref when dependencies are
removed. Capability request preparation therefore resolves the current Team
task by stable task ID and verifies that its capability definition is
unchanged before committing the request binding.

The Core fan-in always includes the governed Factory planning route and adds
two semantic routes per candidate. This permits a deterministic
non-candidate frontier without fabricating semantic routes.

## Terminal Candidate Cardinality

Batch Quality remains the owner of candidate cardinality outcomes:

```text
zero candidates -> NO_ELIGIBLE_ITEMS
one candidate   -> BLOCKED / INSUFFICIENT_BATCH_CANDIDATES
two or more     -> ordinary Batch Quality and Delivery gates
```

After the allowed retry budget is exhausted, PlannerAssessment commits
`ESCALATE / TERMINAL_WORK_FAILED` rather than treating the failed task as
ordinary missing output. The product projects the committed non-success
aggregate as `BLOCKED / NONE`; exact replay does not run another Graph
transition and cannot enter Final Delivery.

Delivery candidate and export materials are sorted by their final
`ObjectRef`, making request identity independent of Python hash seed.
Zero-, one-, and two-candidate CLI paths pass under `PYTHONHASHSEED=1` and
`321`. This is mechanism evidence only; no live Provider was called.

## Recovery Fault Matrix Closure

Stage 4 recovery is verified across these process boundaries:

```text
PRE committed, owner not started
owner committed, POST absent
POST committed, session event absent
session event committed, delivery acknowledgement absent
PlanReview successor committed, reconciliation incomplete
PlannerAssessment transaction interrupted
candidate bundle renamed, Factory authority absent
```

An unchanged owner reuses the unfinished PRE. An owner successor that cannot
be proven command-by-command becomes
`RECONCILED / VERIFICATION_REQUIRED`; replay of the same recovery command
returns that exact committed checkpoint. Session projection relies on the
Harness event idempotency key before acknowledging the graph-journal outbox.
Candidate output verifies and reuses an exact orphan bundle.

The repository may have multiple active Beads tasks. Their handoffs are
task-scoped records, so multiple `in_progress` handoffs are valid when each
matches its Beads authority. `PROJECT_STATE.md` remains the aggregate
cross-task continuity view.
