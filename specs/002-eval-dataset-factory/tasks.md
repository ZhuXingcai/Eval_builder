# Epic Map: Trace-Driven Eval Dataset Factory

**Spec**: `specs/002-eval-dataset-factory/spec.md`

**Status**: Approved v2 - original graph materialized 2026-07-19; R0-13 change decision added
2026-07-20

**Root Epic**: `env_mock_agent-ujc`

**Approval/materialization issue**: `env_mock_agent-ujc.1.1`

**Predecessor drafting issue**: `env_mock_agent-njh` (closed)

Truth is separated by domain: Constitution governs non-negotiable principles; Spec and ADRs govern
approved requirements and design; versioned contracts govern executable interfaces; Beads alone governs
task status, dependencies, blockers, and completion evidence. This document records the approved issue
graph and acceptance mapping; it MUST NOT be maintained as an independent status checklist.
The graph below has been materialized. Use `bd show <id>` for current status and evidence; this file
records stable intent and ID mapping only.

## Root Epic

| Key | Beads ID | Title | Type | Priority | Completion definition |
|---|---|---|---|---|---|
| `EDF` | `env_mock_agent-ujc` | Trace-Driven Eval Dataset Factory R0-R8 | Epic | P0 | R0-R8 gates complete; E2E canary and production-readiness claims remain distinct; production release requires a valid attestation |

The root Epic supersedes no historical issue. The closed `env_mock_agent-3nj` Epic remains the
authoritative record for the attachment-agent M0-M5 foundation.

## Authoritative Beads ID Mapping

| Keys | Beads IDs | Count |
|---|---|---:|
| `EDF` | `env_mock_agent-ujc` | 1 |
| `EDF-R0`, `R0-01` through `R0-13` | `env_mock_agent-ujc.1`, `env_mock_agent-ujc.1.1` through `.1.13` | 14 |
| `EDF-R1`, `R1-01` through `R1-09` | `env_mock_agent-ujc.2`, `env_mock_agent-ujc.2.1` through `.2.9` | 10 |
| `EDF-R2`, `R2-01` through `R2-07` | `env_mock_agent-ujc.3`, `env_mock_agent-ujc.3.1` through `.3.7` | 8 |
| `EDF-R3`, `R3-01` through `R3-06` | `env_mock_agent-ujc.4`, `env_mock_agent-ujc.4.1` through `.4.6` | 7 |
| `EDF-R4`, `R4-01` through `R4-09` | `env_mock_agent-ujc.5`, `env_mock_agent-ujc.5.1` through `.5.9` | 10 |
| `EDF-R5`, `R5-01` through `R5-10` | `env_mock_agent-ujc.6`, `env_mock_agent-ujc.6.1` through `.6.10` | 11 |
| `EDF-R6`, `R6-01` through `R6-07` | `env_mock_agent-ujc.7`, `env_mock_agent-ujc.7.1` through `.7.7` | 8 |
| `EDF-R7`, `R7-01` through `R7-09` | `env_mock_agent-ujc.8`, `env_mock_agent-ujc.8.1` through `.8.9` | 10 |
| `EDF-R8`, `R8-01` through `R8-09` | `env_mock_agent-ujc.9`, `env_mock_agent-ujc.9.1` through `.9.9` | 10 |
| **Total** | Root + 9 child Epics + 79 work packages | **89** |

## Epic Overview

| Key | Beads ID | Epic | Primary outcome | Depends on | Parallel boundary |
|---|---|---|---|---|---|
| `EDF-R0` | `env_mock_agent-ujc.1` | Governance, contracts, and reference foundation | Approved Spec v2, ADRs, canary manifest, historical v1 baseline, active v2 overlays, and review-ready reference fixtures | None | ADRs and data preparation may proceed in parallel after Spec review |
| `EDF-R1` | `env_mock_agent-ujc.2` | Durable control skeleton and TraceIR | Recoverable parser/index slice with deterministic base segments and query service | `EDF-R0` | Parser, storage, query, and state tests may split after schemas freeze |
| `EDF-R2` | `env_mock_agent-ujc.3` | Provenance, taint, and EvidenceBundle | Default-safe evidence views with input/output isolation | `EDF-R1` | Provenance rules, file timeline, redaction, and policy tests may parallelize |
| `EDF-R3` | `env_mock_agent-ujc.4` | LabelSpec and candidate mining | Structured-first and semantic-residual batch labeling | `EDF-R2` | Label compiler and semantic residual may parallelize against stable query contracts |
| `EDF-R4` | `env_mock_agent-ujc.5` | Task and evaluation-contract reconstruction | Candidate TaskDraft, rubrics, evaluator and reference policies | `EDF-R3` | TaskEpisode, task authoring, rubric, and evaluator work split after SelectionContext |
| `EDF-R5` | `env_mock_agent-ujc.6` | Three-mode attachment reconstruction and item QA | Safe input package, immutable build results, three-round item QualityReport | `EDF-R4` | Independent artifact groups and deterministic validators may parallelize |
| `EDF-R6` | `env_mock_agent-ujc.7` | Resource-aware DatasetJob orchestration | Batch fan-out, limits, backpressure, checkpoint, cancellation, CLI | `EDF-R1`, `EDF-R3`, `EDF-R4`, `EDF-R5` | Scheduler and observability may parallelize after state contracts |
| `EDF-R7` | `env_mock_agent-ujc.8` | Batch QA, user checkpoints, and canary release | BatchQualityReport, optional version-bound user decisions, canary/internal release | `EDF-R5`, `EDF-R6` | Batch validators and checkpoint interfaces may parallelize |
| `EDF-R8` | `env_mock_agent-ujc.9` | Statistical, load, safety, and production gates | ProductionReadinessAttestation and guarded production release | `EDF-R7` | Quality, load, cost, safety, privacy, and operations evaluations may run in parallel on frozen versions |

## Canonical Dependency Graph

```text
EDF-R0
  -> EDF-R1
      -> EDF-R2
          -> EDF-R3
              -> EDF-R4
                  -> EDF-R5

EDF-R1 + EDF-R3 + EDF-R4 + EDF-R5
  -> EDF-R6

EDF-R5 + EDF-R6
  -> EDF-R7
      -> EDF-R8
```

An Epic may prepare non-behavioral design work before all dependencies close, but it cannot claim its
acceptance gate until all listed dependencies are closed with evidence.

## EDF-R0 - Governance, Contracts, and Reference Foundation

**Objective**: Convert the approved architecture proposal into durable project truth before behavioral
implementation.

**Proposed work packages**:

| Key | Work package | Type | Depends on | Acceptance evidence |
|---|---|---|---|---|
| `R0-01` | Approve `002-eval-dataset-factory` Spec and materialize the Beads graph | decision | `env_mock_agent-njh` | Approved spec status; real Beads IDs written into this map |
| `R0-02` | Decide repository boundary | decision | `R0-01` | Accepted Repo Boundary ADR |
| `R0-03` | Freeze TraceIR and source-recovery principles | decision | `R0-01` | Accepted TraceIR ADR with partial-recovery semantics |
| `R0-04` | Freeze provenance, taint, and producer/evaluator/contestant views | decision | `R0-01` | Accepted Safety/Taint ADR |
| `R0-05` | Approve privacy classification and internal-model disclosure policy | decision | `R0-01` | Data-classification record and policy owner |
| `R0-06` | Approve immutable human editing, invalidation, and review policy v1 | decision | `R0-01`, `R0-04` | Historical v1 policy retained; active review subsystem superseded by R0-13 |
| `R0-07` | Select 20-30 stratified development canaries | task | `R0-03`, `R0-04`, `R0-05` | Immutable manifest, source hashes, category coverage |
| `R0-08` | Define parser, safety, label, task, attachment, and release annotation guides | task | `R0-03`, `R0-04`, `R0-06`, `R0-07` | Versioned annotation schemas; v1 independent-adjudication workflow retained as superseded history |
| `R0-09` | Prepare five review-ready EvaluationItem reference fixtures and user checkpoint previews | task | `R0-07`, `R0-08`, `R0-10`, `R0-13` | Source-bound complete candidate scaffolds and label/rewrite/environment/final previews; no mandatory user acceptance |
| `R0-10` | Freeze v1 cross-stage contracts | task | `R0-02`, `R0-03`, `R0-04`, `R0-05`, `R0-06` | Contract files and schema compatibility policy |
| `R0-11` | Approve first representative labels and release profiles | decision | `R0-01`, `R0-07` | Approved label definitions; LH requirement; Generic decision |
| `R0-12` | Validate fresh-session continuity for the new Spec and Epic | task | `R0-01` through `R0-11`, `R0-13` | Context bootstrap identifies Spec v2, active decisions, Beads graph, blockers, and next command |
| `R0-13` | Replace mandatory independent review with user-directed approval checkpoints | decision | `R0-01`, `R0-04`, `R0-06`, `R0-08`, `R0-10` | ADR 0011, User Approval Policy v2, active contract v2, graph and fixture migration |

**Epic completion gate**:

- The Spec has no unresolved product-boundary placeholder.
- Repo Boundary, TraceIR, and Safety/Taint ADRs are accepted.
- The privacy owner, user approval policy, and first release profiles are explicit.
- The canary manifest covers strict parse, repair, continuation, pre/post mutation, truncation, missing
  reads, search, shell errors, and visible final outputs.
- At least five canaries have complete review-ready EvaluationItem reference fixtures and user
  checkpoint previews.
- Cross-stage contracts and their versioning rules are frozen.
- No R1 behavioral implementation starts against an unapproved governing contract.

## EDF-R1 - Durable Control Skeleton and TraceIR

**Objective**: Build the deterministic evidence spine and minimum recoverable control plane.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R1-01` | Implement Job, Item, StageRun, immutable StageResult, legal transitions, idempotency, and outbox transactions | None | No | State contract and illegal-transition tests |
| `R1-02` | Implement `raw_traj_v1` probe and immutable source registration | None | Yes | Adapter contract and source-hash tests |
| `R1-03` | Implement strict parse and audited local repair with RepairMap | `R1-02` | Yes | Golden parser tests |
| `R1-04` | Implement streaming recovery and capability-level parse quality | `R1-03` | No | Partial and blocked admission tests |
| `R1-05` | Normalize events, tool families, call/result pairs, errors, and truncation | `R1-03`, `R1-04` | No | TraceIR contract tests |
| `R1-06` | Build deterministic InteractionSegments and file observations | `R1-05` | No | Stable-ID and repeatability tests |
| `R1-07` | Store JSONL canonical facts, CAS content, SQLite metadata, and FTS indexes | `R1-01`, `R1-05` | No | Reload and query-equivalence tests |
| `R1-08` | Implement bounded TraceQueryService API and CLI | `R1-07` | No | Consumer, purpose, size-limit, and source-span tests |
| `R1-09` | Implement single-process serial runner, checkpoint, resume, and cancel | `R1-01`, `R1-07` | No | Fault-injected recovery integration tests |

**Epic completion gate**:

- Parser event recovery on approved gold is at least 99%.
- Explicit-ID tool pairs are 100% correct.
- Every repair and derived event is source-addressable.
- Base segments and deterministic IDs are repeatable.
- Queries do not require loading a complete trace into a downstream agent.
- Parse/index interruption resumes without duplicate side effects.

## EDF-R2 - Provenance, Taint, and EvidenceBundle

**Objective**: Make input-state evidence safe by construction and inaccessible by default when unsafe.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R2-01` | Implement provenance classes and deterministic decision table | None | Yes | Safety gold decisions |
| `R2-02` | Reconstruct pre-write and post-write file-version timelines | None | Yes | In-place mutation tests |
| `R2-03` | Implement derived-content edges and taint propagation | `R2-01`, `R2-02` | No | Copy, summary, translation, and embedding tests |
| `R2-04` | Implement secret and configured-PII redaction | None | Yes | Redaction gold tests |
| `R2-05` | Implement privileged audit, default-safe, producer, evaluator, and contestant views | `R2-01`, `R2-03`, `R2-04` | No | Authorization contract tests |
| `R2-06` | Compile EvidenceBundle and ArtifactEvidenceMatrix | `R2-02`, `R2-03`, `R2-05` | No | Hash, lineage, gap, and quarantine tests |
| `R2-07` | Add prompt-injection-as-data enforcement | `R2-05` | No | Injection execution count remains zero |

**Epic completion gate**:

- Every known output-side gold object is quarantined.
- Content first observed after mutation never enters input evidence.
- Taint survives all tested derivations.
- Default downstream identities cannot read raw or quarantined content.
- No secret or configured PII appears in safe evidence views.

## EDF-R3 - LabelSpec and Candidate Mining

**Objective**: Replace ad hoc keyword scripts with evidence-bound, structured-first labeling.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R3-01` | Define LabelSpec and LabelDecision contracts | None | No | Schema and compatibility tests |
| `R3-02` | Compile structured predicates, negatives, sequences, windows, and error signatures | `R3-01` | Yes | Deterministic label gold tests |
| `R3-03` | Implement semantic residual request and abstain contract | `R3-01` | Yes | Model-fake contract tests |
| `R3-04` | Implement confidence merge, abstention, and typed unresolved routing | `R3-02`, `R3-03` | No | Disagreement, threshold, abstention, and configured user-inspection tests |
| `R3-05` | Implement batch progress, checkpoint, JSONL, and Parquet export | `R3-01` | Yes | Resume and export tests |
| `R3-06` | Calibrate first structured and semantic labels on development canaries | `R3-04`, `R3-05` | No | Evidence-level human comparison |

**Epic completion gate**:

- Every canary decision and evidence span matches gold or yields an approved ambiguity abstention.
- No unresolved functional blocking finding remains.
- Structured rejection avoids unnecessary semantic model calls.
- Interrupted labeling resumes only incomplete trace-label pairs.
- Canary scores are reported as development evidence, not production statistics.

## EDF-R4 - Task and Evaluation-Contract Reconstruction

**Objective**: Turn selected evidence into a self-contained candidate task and isolated evaluation
contract.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R4-01` | Implement TaskEpisode semantic grouping over immutable base segments | None | Yes | Episode lineage tests |
| `R4-02` | Implement SelectionContext firewall | None | Yes | Hidden-signal non-disclosure tests |
| `R4-03` | Implement TaskDraft authoring and requirement lineage | `R4-01`, `R4-02` | No | Gold task-intent review |
| `R4-04` | Implement final-answer, completed-deliverable, and accidental-path prompt gates | `R4-03` | No | Prompt leakage gold tests |
| `R4-05` | Implement RubricSet import/candidate generation and reachability | `R4-03` | Yes | Judged-object and reachability tests |
| `R4-06` | Implement EvaluatorSpec and ReferencePolicy | `R4-03`, `R4-05` | No | Isolation and failure-classification tests |
| `R4-07` | Implement ToolPolicy generation, contestant projection, and version/hash invalidation | `R4-03`, `R4-06` | No | Allow/deny, projection, and stale-policy tests |
| `R4-08` | Implement deterministic ProducerTaskView projection | `R4-03`, `R4-06`, `R4-07` | No | Schema-diff and storage-authorization tests |
| `R4-09` | Add task rewrite previews and version-bound user adjustment handling | `R4-04`, `R4-05`, `R4-06`, `R4-07`, `R4-08` | No | Preview projection, immutable version, and directed invalidation tests |

**Epic completion gate**:

- Candidate prompts preserve approved task intent and are understandable without raw trace access.
- Every critical prompt requirement has evidence lineage.
- Prompt and producer view contain no private reference, grader rule, hidden selection signal, final answer,
  or completed output.
- Every criterion has a judged object, reachable inputs, evaluator binding, and visibility.
- ToolPolicy has explicit allow/deny semantics, contestant projection, versioning, and hash invalidation.
- R4 produces `CANDIDATE_TASK`, not a released EvaluationItem.

## EDF-R5 - Three-Mode Attachment Reconstruction and Item QA

**Objective**: Reuse and extend the attachment subsystem behind least-privilege typed contracts.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R5-01` | Bridge ProducerTaskView and EvidenceBundle into attachment planning | None | No | Boundary contract tests |
| `R5-02` | Compute artifact-level evidence modes and task-level `MIXED` | `R5-01` | Yes | Mode matrix gold tests |
| `R5-03` | Implement prompt-only dependency discovery | `R5-01` | Yes | Dependency coverage, evidence-priority, and forbidden-output tests |
| `R5-04` | Connect approved Search/Fetch evidence and SourceEvidence lineage | `R5-01`, `R5-03` | No | Retrieval provenance tests |
| `R5-05` | Connect RuntimeRouter and deterministic-provider-first policy | `R5-01` | Yes | Routing and blocked-capability tests |
| `R5-06` | Fan out independent artifact groups with WorldLedger consistency | `R5-02`, `R5-04`, `R5-05` | No | Concurrency and consistency tests |
| `R5-07` | Emit ArtifactBuildResult and AttachmentReconstructionResult | `R5-06` | No | Partial failure and minimal retry tests |
| `R5-08` | Run deterministic item validators | `R5-07` | No | Format, secret, package, and non-overridable leakage tests |
| `R5-09` | Run three isolated sequential semantic review and targeted-repair rounds | `R5-08` | No | Separate role, StageRun, clean-context, round-dependency, and stale-finding tests |
| `R5-10` | Produce version-bound item QualityReport | `R5-09` | No | Open-finding and hash-invalidation tests |

**Epic completion gate**:

- `TRACE_RICH`, `SKELETON_GUIDED`, and `PROMPT_ONLY` each have at least one safe executable success;
  separate canaries verify blocked outcomes.
- Mode changes strategy inside one workflow rather than selecting duplicate agents.
- Critical missing capabilities block.
- A failed artifact retries independently.
- Shared-fact groups remain consistent.
- Output packages contain input-state material only.
- No item with open P0/P1 has an approvable QualityReport.
- Answer leakage, grader disclosure, and original-final-output derivation cannot be human-overridden.

## EDF-R6 - Resource-Aware DatasetJob Orchestration

**Objective**: Scale the serial R1 runner into a bounded, observable, resumable batch control plane.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R6-01` | Implement canonical DAG validation and resolved-plan persistence | None | No | Missing and reverse dependency tests |
| `R6-02` | Implement item and artifact fan-out with dependency joins | `R6-01` | No | Mixed success and isolation tests |
| `R6-03` | Implement model RPM/TPM buckets, concurrency pools, and backpressure | `R6-01` | Yes | Deterministic scheduler tests |
| `R6-04` | Implement process, renderer, network, storage, and budget pools | `R6-01` | Yes | Resource exhaustion tests |
| `R6-05` | Implement lease, heartbeat, cancellation, stale-result rejection, and retry policy | `R6-01`, `R6-02` | No | Fault-injection tests |
| `R6-06` | Implement batch audit events, metrics, and cost accounting | `R6-02`, `R6-03`, `R6-04`, `R6-05` | No | Event completeness tests |
| `R6-07` | Implement required CLI lifecycle commands | `R6-02`, `R6-05`, `R6-06` | No | CLI integration tests |

**Epic completion gate**:

- One command can produce a candidate dataset from a raw-trace manifest.
- Any stage resumes from the smallest incomplete unit.
- One item failure does not block unrelated items.
- Rate limits and budgets produce explicit state, not silent quality degradation.
- Stale workers and duplicate idempotency keys cannot duplicate side effects.

## EDF-R7 - Batch QA, User Checkpoints, and Canary Release

**Objective**: Close cross-item quality risks, apply configured user checkpoints without a separate
human-review subsystem, and provide a non-production release workflow.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R7-01` | Implement duplicate task and near-duplicate attachment detection | None | Yes | Cluster gold tests |
| `R7-02` | Implement cross-item leakage, answer reuse, and contamination checks | None | Yes | Batch safety gold tests |
| `R7-03` | Implement lineage audit and BatchQualityReport | `R7-01`, `R7-02` | No | Item/batch finding ownership tests |
| `R7-04` | Implement UserApprovalPolicy and checkpoint request generation | None | Yes | Mode/checkpoint, bounded preview, and no-synthetic-decision tests |
| `R7-05` | Implement immutable UserDecisionRecord and plan decision handling | `R7-04` | No | Hash, stale decision, idempotency, adjustment, and hard-gate tests |
| `R7-06` | Implement user-plan invalidation and directed R3/R4/R5 revalidation | `R7-03`, `R7-05` | No | Plan and object invalidation graph tests |
| `R7-07` | Implement CLI/JSON user checkpoint interaction and resume | `R7-04`, `R7-05` | Yes | Checkpoint interaction, resume, disabled-mode, and state mapping tests |
| `R7-08` | Implement ReleaseDecision v2, complete EvaluationItem including ToolPolicy, and Item projection transaction | `R7-03`, `R7-05`, `R7-06` | No | Projection rebuild, required-child-contract, required-checkpoint, and integrity tests |
| `R7-09` | Implement LH canary/internal release manifest and registry isolation | `R7-08` | No | Package-hash and channel tests |

**Epic completion gate**:

- Every approved item has current item and batch quality reports and complete lineage.
- Item-scoped and batch-scoped findings have one explicit owner and closure path.
- Required user decisions bind exact request, plan, subject, and policy hashes; disabled checkpoints
  create no synthetic decision.
- User plan adjustments invalidate and re-run only affected checks.
- R7 cannot write to the production registry.

## EDF-R8 - Statistical, Load, Safety, and Production Gates

**Objective**: Establish evidence for production readiness without substituting throughput or replay
volume for quality.

**Proposed work packages**:

| Key | Work package | Depends on | Parallel | Acceptance evidence |
|---|---|---|---|---|
| `R8-01` | Freeze independent label test sets at required sample sizes | None | No | Dataset hashes, access controls, freeze record |
| `R8-02` | Run structured and semantic label quality evaluation | `R8-01` | No | Point estimates, 95% intervals, sample counts |
| `R8-03` | Run 20-30 canary end-to-end regression | None | Yes | Typed success/failure report |
| `R8-04` | Run 100-independent-real-trace stability regression | None | Yes | Unique-trace coverage and stability-threshold report |
| `R8-05` | Run synthetic/replay 1000-job scheduler load tests | None | Yes | Throughput and recovery report labeled load-only |
| `R8-06` | Run concurrency staircase, cost, cache, and latency analysis | `R8-03`, `R8-04`, `R8-05` | No | Default resource-policy recommendation |
| `R8-07` | Complete safety, privacy, release-permission, and operations-SLO reviews | `R8-02`, `R8-03`, `R8-04`, `R8-06` | No | Approval records |
| `R8-08` | Issue version-scoped ProductionReadinessAttestation | `R8-02`, `R8-04`, `R8-06`, `R8-07` | No | Active attestation with invalidation policy |
| `R8-09` | Verify guarded production ReleaseDecision | `R8-08` | No | Positive and negative production-publish tests |

**Epic completion gate**:

- Frozen independent label sets meet approved statistical thresholds.
- Stability regression covers at least 100 independent eligible real traces and meets the approved
  stability threshold; otherwise the gate remains `STATISTICAL_GATE_PENDING`.
- Real, synthetic, and replay evidence are reported separately.
- Default concurrency and budgets are measured rather than guessed.
- Safety, privacy, permissions, and operations approvals are complete.
- No statistical gate, P0/P1, or high-risk review remains open.
- Production publication fails without a current attestation and succeeds only for an unchanged approved
  release subject and package.

## Beads Materialization Record

Materialization completed with these controls:

1. One root Epic and nine child Epics were created.
2. All original 78 work-package issues were created under their owning Epic; R0-13 was added as the
   approved requirement-change decision, bringing the active total to 79.
3. Approved dependencies were added with `bd dep add`.
4. `bd dep cycles` reports no cycle and `bd orphans` reports no orphan.
5. Every node carries `specs/002-eval-dataset-factory/spec.md` as `spec_id`.
6. Current status, assignee, blockers, and completion evidence come only from Beads.
7. Closed `env_mock_agent-3nj.*` issues remain unchanged.

## Initial Materialization Order

```text
1. EDF root Epic
2. EDF-R0 child Epic
3. R0-01 Spec approval decision
4. R0-02 through R0-06 governance decisions
5. R0-07 through R0-11 data and contract work
6. R0-13 user-approval architecture change
7. R0-12 continuity validation
8. EDF-R1 through EDF-R8 child Epics with dependency edges
```

The predecessor issue `env_mock_agent-njh` records draft validation. User approval authorized
materialization through `env_mock_agent-ujc.1.1`; behavioral work proceeds one claimed Beads issue at a
time.
