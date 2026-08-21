# Implementation Plan: Trace-Driven Eval Dataset Factory

**Branch**: `main` | **Date**: 2026-07-20

**Spec**: `specs/002-eval-dataset-factory/spec.md`

**Epic Map**: `specs/002-eval-dataset-factory/tasks.md`

**Root Beads Epic**: `env_mock_agent-ujc`

**Status**: Active execution baseline

## Summary

Build the Eval Dataset Factory as a set of bounded contexts around a deterministic Trace Intelligence
spine. Reuse the existing attachment generator as the attachment reconstruction subsystem. Keep facts,
policies, state transitions, and release gates deterministic; use specialized agents only for semantic
labeling, task authorship, open-ended artifact work, and independent quality review.

Delivery is split into R0-R8:

- R0 freezes governance, safety boundaries, data, and contracts.
- R1-R2 build the deterministic trace and safety foundation.
- R3-R5 add labeling, task reconstruction, attachments, and item QA.
- R6 adds resource-aware batch orchestration.
- R7 completes batch review, configured user checkpoints, and non-production release.
- R8 establishes statistical and operational production readiness.

`E2E_CANARY_COMPLETE` is achieved at R7. `PRODUCTION_READY` requires R8 and a valid
ProductionReadinessAttestation.

## Technical Context

**Language/Version**: Python 3.12; Node.js 22 remains limited to the existing Pi bridge.

**Primary Dependencies**: Pydantic v2, LangGraph, Typer, SQLite, JSONL, SQLite FTS5, existing Provider,
Validator, Runtime, and Profile abstractions.

**Storage**:

- Raw trace: immutable filesystem references or content-addressed objects.
- Canonical trace facts: JSONL.
- Metadata, job state, leases, and indexes: SQLite.
- Content payloads: content-addressed storage.
- Analysis export: Parquet where required by labeling and benchmark work.
- Existing attachment runs: referenced from the new Job Store, not copied into it.

**Testing**: pytest unit, contract, integration, property, golden, fault-injection, and agent-evaluation
suites; existing Node checks for Pi; offline-by-default with live credentials isolated by markers.

**Target Platform**: Trusted and monitored local CLI/batch execution for the first release.

**Project Type**: Python CLI and workflow library with bounded contexts in the existing repository,
pending the accepted Repo Boundary ADR.

**Performance Goals**:

- Parse each raw trace once per adapter/policy version.
- Avoid full-trace model disclosure.
- Resume at trace, stage, item, and artifact granularity.
- Determine default concurrency through R8 staircase tests, not a hard-coded guess.

**Constraints**:

- No answer, requested final deliverable, grader rule, hidden condition, or final-output-derived content
  can be released or human-overridden.
- Missing critical capabilities block.
- Raw traces and quarantine content never enter exported packages.
- Production release requires R8 attestation.

**Scale/Scope**:

- R0: 20-30 development canaries and at least five complete review-ready EvaluationItem reference
  fixtures with user-checkpoint previews.
- R8 label quality: at least 50 positive and 50 negative traces per structured label; at least 100
  frozen independent traces per semantic label.
- R8 stability: at least 100 independent eligible real traces.
- R8 scheduler load: 1000 synthetic/replay jobs, reported as load-only evidence.

## Constitution Check

| Principle | Plan compliance | Gate |
|---|---|---|
| Input-State Integrity | Safety/Taint ADR, ProducerTaskView, leakage validator, immutable release subject | PASS |
| Evidence Before Synthesis | EvidenceBundle, evidence priority, external source lineage | PASS |
| Deterministic Capabilities First | TraceIR, policy services, providers, validators, state machines | PASS |
| Structural And Semantic Gates | Format validators plus three isolated sequential reviewers | PASS |
| Replaceable Runtimes | Existing shared runtime contract retained | PASS |
| Durable Project Truth | Spec/ADR/contracts/Beads/project-state truth domains retained | PASS |
| Test And Evaluation Driven Development | Each Beads issue has acceptance evidence and test ownership | PASS |
| Least Privilege And Explicit Degradation | Identity-scoped views, blocked capability, no silent fallback | PASS |

The check MUST be repeated after R0 contract freeze and whenever an ADR changes a safety, runtime, or
repository boundary.

## Architecture

```text
DatasetJobOrchestrator
  |
  +-- Trace Intelligence
  |     raw_traj_v1 -> TraceIR -> SourceSpan/RepairMap -> query/index
  |
  +-- Safety and Evidence
  |     provenance -> file timeline -> taint -> safe views -> EvidenceBundle
  |
  +-- Candidate Mining
  |     LabelSpec -> deterministic query -> semantic residual -> LabelDecision
  |
  +-- Task Authoring
  |     SelectionContext -> TaskEpisode -> TaskDraft -> evaluation contracts
  |
  +-- Attachment Reconstruction
  |     ProducerTaskView -> ArtifactEvidenceMatrix -> existing attachment subgraph
  |
  +-- Quality and Release
        item validators -> 3 semantic rounds -> batch QA -> configured user checkpoints -> release
```

### Control Boundaries

1. TraceAdapter converts source format to TraceIR and never calls an LLM.
2. TraceQueryService is the only downstream trace-evidence access path.
3. Stage agents exchange typed artifacts, not shared chat history.
4. Attachment workers receive ProducerTaskView, not full TaskDraft or evaluation contracts.
5. ReleaseDecision is the only approval and release authority.
6. Beads owns task state; runtime transcripts and Markdown do not.
7. UserApprovalPolicy controls optional HITL; no separate human-review subsystem is active.
8. Data-classification v2 replaces v1 human-role projections with bounded requesting-user checkpoint
   projections while retaining all v1 privacy and model-disclosure controls.

## Project Structure

R0-02 owns the final repository-boundary decision. The implementation baseline assumes the following
bounded contexts in the current repository and prohibits a split before that ADR is accepted:

```text
src/
  env_mock_agent/                 # existing attachment subsystem and compatibility CLI
    facade/                       # provider-owned attachment request/result boundary
  eval_factory/
    contracts/                    # factory-internal and dataset-facing Pydantic schemas
    orchestration/                # Job/Item/StageRun state and scheduler
    trace/
      adapters/
      parsing/
      indexing/
      query/
    safety/
      provenance/
      taint/
      redaction/
      views/
    labeling/
    task_authoring/
    review/
    dataset/

tests/
  eval_factory/
    contract/
    unit/
    integration/
    property/
    golden/
    fault_injection/

evals/
  golden/
    eval_factory/
      parser/
      safety/
      labeling/
      task_authoring/
      attachments/
      release/
  benchmarks/
  results/

specs/002-eval-dataset-factory/
  spec.md
  plan.md
  tasks.md
  research.md
  data-model.md
  quickstart.md
  contracts/
```

**Structure Decision**: Keep `env_mock_agent` behavior stable behind compatibility contracts. New
factory concerns live under `eval_factory`. Cross-boundary imports flow through versioned contracts;
Trace parsing and dataset release logic do not enter Provider or Runtime modules.

The attachment boundary is provider-owned: `env_mock_agent.facade` defines
`AttachmentReconstructionRequest` and `AttachmentReconstructionResult`, and `eval_factory` imports that
facade. `env_mock_agent` never imports `eval_factory`. The first release registers an independent
`evalfactory = eval_factory.cli:app` entry point while preserving `envmock = env_mock_agent.cli:app`.

## Delivery Sequence

| Release | Beads Epic | Exit artifact | Exit gate |
|---|---|---|---|
| R0 | `env_mock_agent-ujc.1` | ADRs, canary manifest, annotation policy, frozen contracts | No unapproved governing boundary |
| R1 | `env_mock_agent-ujc.2` | Recoverable TraceIR/index/query vertical slice | Parser and state-machine gold pass |
| R2 | `env_mock_agent-ujc.3` | Safe EvidenceBundle and ArtifactEvidenceMatrix | Known output contamination recall 100% |
| R3 | `env_mock_agent-ujc.4` | Evidence-bound LabelDecisions | Functional canary gate, no fake statistical claim |
| R4 | `env_mock_agent-ujc.5` | Candidate task and evaluation contracts | Prompt/reference/selection firewall pass |
| R5 | `env_mock_agent-ujc.6` | Safe attachment package and item QualityReport | Three modes succeed; P0/P1 closed |
| R6 | `env_mock_agent-ujc.7` | Resource-aware batch pipeline | Resume, backpressure, budget, isolation pass |
| R7 | `env_mock_agent-ujc.8` | BatchQualityReport and canary/internal release | `E2E_CANARY_COMPLETE` |
| R8 | `env_mock_agent-ujc.9` | ProductionReadinessAttestation and production release | `PRODUCTION_READY` |

## Critical Path

```text
R0-01
  -> R0 governance and contract closure
  -> R1 parser/state/query
  -> R2 safety/evidence
  -> R3 labeling
  -> R4 task/evaluation contracts
  -> R5 attachments/item QA
  -> R6 batch orchestration
  -> R7 batch QA/release
  -> R8 production gates
```

R6 intentionally depends on real R1, R3, R4, and R5 outputs. It must not be built as an empty
orchestrator before stage contracts exist.

## R0 Immediate Execution Plan

R0 is governance and evidence preparation, not placeholder work.

### Wave R0-A - Approval and Materialization

| Work | Beads ID | Result |
|---|---|---|
| Approve Spec and materialize graph | `env_mock_agent-ujc.1.1` | Approved Spec and original 88-node acyclic graph |

### Wave R0-B - Parallel Governing Decisions

After R0-01 closes, these can proceed independently:

| Work | Beads ID | Durable output |
|---|---|---|
| Repository boundary | `env_mock_agent-ujc.1.2` | ADR 0008 |
| TraceIR and recovery principles | `env_mock_agent-ujc.1.3` | ADR 0009 |
| Provenance, taint, and data views | `env_mock_agent-ujc.1.4` | ADR 0010 |
| Privacy and model disclosure | `env_mock_agent-ujc.1.5` | Approved data-classification record |

### Wave R0-C - Dependent Policy and Data Work

| Work | Beads ID | Dependencies |
|---|---|---|
| Historical human review policy v1 | `env_mock_agent-ujc.1.6` | R0-01, R0-04 |
| Canary selection | `env_mock_agent-ujc.1.7` | R0-03, R0-04, R0-05 |
| Annotation guides | `env_mock_agent-ujc.1.8` | R0-03, R0-04, R0-06, R0-07 |
| Review-ready EvaluationItem fixtures and checkpoint previews | `env_mock_agent-ujc.1.9` | R0-07, R0-08, R0-10, R0-13 |
| Cross-stage contract freeze | `env_mock_agent-ujc.1.10` | R0-02 through R0-06 |
| Labels and release profiles | `env_mock_agent-ujc.1.11` | R0-01, R0-07 |
| User-directed approval and data-projection architecture | `env_mock_agent-ujc.1.13` | R0-01, R0-04, R0-06, R0-08, R0-10 |
| Fresh-session continuity | `env_mock_agent-ujc.1.12` | R0-01 through R0-11, R0-13 |

### R0 Exit

Close `env_mock_agent-ujc.1` only when every child acceptance criterion has evidence and a fresh
bootstrap identifies the new Spec, Epic, active work, blockers, and exact next command.

## Parallel Execution Policy

Parallelism is allowed only when all conditions hold:

1. Beads dependencies are closed.
2. Tasks do not write the same contract or state owner.
3. Each task has an independent acceptance command.
4. Shared safety policy versions are fixed for the run.
5. Merge order cannot alter deterministic output hashes.

Approved parallel lanes:

- R0: repository, TraceIR, safety/taint, and privacy decisions after R0-01.
- R1: state contracts and source adapter can begin separately; parser stages remain ordered.
- R2: provenance, file timeline, and redaction can begin separately.
- R3: deterministic label compiler and semantic residual contract after LabelSpec.
- R4: TaskEpisode and SelectionContext; rubric and evaluator work after TaskDraft.
- R5: independent artifact groups after WorldLedger and build specs.
- R8: frozen quality evaluation, canary regression, real-trace stability, and load tests.

Disallowed parallelism:

- Semantic review rounds across the same item.
- Conflicting WorldLedger or file-version writes.
- Release publication and object mutation.
- Two workers owning the same path or user checkpoint request.

## Testing Strategy

### Contract First

Every cross-stage object gets:

- serialization round-trip;
- schema-version rejection and migration tests;
- content-hash stability;
- forbidden-field projection tests;
- stale-reference behavior.

### Trace and Safety

- Strict, repaired, and streaming parser gold.
- Raw byte and decoded-character SourceSpan checks.
- Tool pair, orphan, duplicate-ID, and truncation cases.
- File read/edit/write timelines.
- Derived taint across copy, summary, translation, and embedding.
- Secret, PII, prompt injection, final output, grader, and private-reference gates.

### Workflow and Fault Injection

- Legal and illegal state transitions.
- Transaction rollback and outbox recovery.
- Duplicate idempotency key.
- Lease expiry and stale worker result.
- Model timeout, 429, process death, renderer OOM, and partial network download.
- User-checkpoint interruption and export crash.

### Agent Evaluation

- Structured facts are never delegated to an LLM.
- Semantic labeler supports abstain.
- Task author preserves intent without trajectory leakage.
- Artifact writers obey ProducerTaskView.
- Reviewers use isolated roles and clean contexts.
- Enabled user checkpoints bind exact plan/subject hashes; disabled checkpoints emit no synthetic
  decision.

## Quality Gates

### Per Work Package

- Claimed Beads issue.
- Governing Spec and ADRs read.
- Tests added or explicit documentation validation defined.
- Ruff/format/mypy/pytest scopes selected by blast radius.
- Acceptance evidence recorded in Beads.
- Context checkpoint created before handoff.

### Per Epic

- All child issues closed with evidence.
- `bd dep cycles`, `bd orphans`, and `bd lint` clean.
- Epic-specific gold suite passes.
- No open P0/P1 or unresolved security exception.
- Documentation and version migration updated.

### Production

- R7 canary/internal release complete.
- R8 frozen statistical and real-trace stability gates pass.
- Safety, privacy, permission, and operational attestations exist.
- Every user checkpoint required by the selected UserApprovalPolicy is current.
- Active attestation covers current system/schema/policy versions.
- Release subject and package hashes match.

## Compatibility and Migration

- Existing `envmock generate/resume/export` behavior remains available.
- Existing Run Store is read-only compatible; new Job Store references attachment runs.
- Existing Generic, CC, and LH adapters are not rewritten in R0.
- Existing LH packages and `cc-mock-env` remain read-only benchmark sources.
- Schema changes require explicit migration and version negotiation.
- Runtime session resume remains distinct from workflow checkpoint resume.

## Observability and Cost

Record per job, item, stage, model, tool, and artifact:

- status and reason code;
- latency and queue time;
- token and request usage;
- cache hit;
- retry and resume count;
- provider/runtime version;
- evidence and output hashes;
- cost where available.

Do not rank runtimes or models when all trials are blocked before execution.

## Risk Controls

| Risk | Control |
|---|---|
| Invalid nested JSON loses traces | Three-level parser, RepairMap, partial capability matrix |
| Original answer reaches attachments | Provenance timeline, taint graph, ProducerTaskView, non-waivable gate |
| Long traces exhaust model context | Deterministic index and bounded TraceQueryService |
| Semantic labels become expensive scripts | Structured-first compiler and residual model calls |
| Multi-agent work drifts | Central orchestrator, typed handoffs, immutable versions |
| Parallel artifacts contradict | Artifact groups, WorldLedger, ownership locks |
| Throughput hides poor quality | Separate canary, statistical, and load claims |
| User plan decision becomes stale | Request/plan/subject hash, immutable decision, invalidation graph |
| Old attachment behavior regresses | Compatibility boundary and existing regression suite |
| Context loss stalls development | Spec, ADR, Beads, generated state, typed handoff |

## Governance

- Constitution changes require amendment.
- Product behavior changes require Spec revision.
- Architecture boundary changes require an append-only ADR.
- Cross-stage schema changes require migration.
- Task status and blockers change only in Beads.
- Generated project state is a recovery view and must validate against its sources.

## Definition of Done

The factory is complete only when:

1. R0-R7 achieve `E2E_CANARY_COMPLETE`.
2. R8 achieves `PRODUCTION_READY`.
3. All 89 materialized nodes are closed or explicitly superseded with traceable replacements.
4. Production release is reproducible from immutable inputs, policies, models, configured user
   decisions, quality results, and hashes.
5. A fresh session can recover the active state without chat history.
