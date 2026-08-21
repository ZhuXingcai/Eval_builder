# Feature Specification: Trace-Driven Eval Dataset Factory

**Feature Branch**: `main`

**Created**: 2026-07-19

**Status**: Approved

**Approved**: 2026-07-19

**Revision**: v2 - user-directed approval checkpoints

**Revision Approved**: 2026-07-20

**Root Beads Epic**: `env_mock_agent-ujc`

**Input**: Upgrade the existing environment attachment generator into a trace-driven, auditable,
multi-stage evaluation dataset production system.

**Source Design**:
`../../../.trae/documents/评测集生产多Agent系统_Trace驱动总体开发方案.md`

**Supersession**: This specification does not rewrite or invalidate
`specs/001-agent-foundation`. The existing attachment generator remains a supported subsystem and
becomes the attachment reconstruction bounded context.

## Product Goal

The system turns a batch of real execution traces into reviewed evaluation items through one durable
pipeline:

```text
immutable raw trace
  -> deterministic TraceIR and evidence index
  -> provenance and taint classification
  -> structured and semantic labeling
  -> task and evaluation-contract reconstruction
  -> input-state attachment reconstruction
  -> item and batch quality review
  -> optional user-directed approval checkpoints
  -> canary or production release
```

The shared product primitive is trace evidence, not an autonomous agent. Deterministic services own
fact recovery, indexing, lineage, policy enforcement, state transitions, and release gates. Specialized
agents only handle semantic judgments that cannot be expressed reliably as deterministic rules.

## User Scenarios & Testing

### User Story 1 - Build a Safe Trace Evidence Index (Priority: P1)

An evaluation engineer submits a manifest of `raw_traj` files. The system parses each trace once,
recovers as much valid structure as possible, records every repair, reconstructs tool calls and file
observations, classifies input-side and output-side evidence, and exposes bounded query APIs without
giving downstream agents unrestricted access to the raw trace.

**Why this priority**: Every later stage depends on an accurate and safe evidence plane. A wrong input
and output boundary can mass-produce leaked evaluation items.

**Independent Test**: Ingest the R0 parser and safety canary set and query messages, tool calls, errors,
file versions, truncation markers, provenance decisions, and source spans without loading complete
trace text into a downstream agent.

**Acceptance Scenarios**:

1. **Given** a strictly valid `raw_traj`, **When** it is ingested, **Then** the system emits stable
   TraceIR events, tool-call pairs, file observations, deterministic interaction segments, and source
   spans.
2. **Given** an inner JSON string with a repairable invalid escape, **When** it is ingested, **Then**
   the system performs audited local repair and preserves a byte-level RepairMap.
3. **Given** a partially recoverable trace, **When** parsing completes, **Then** each capability is
   marked complete, partial, or unavailable instead of the trace being presented as fully recovered.
4. **Given** a file created or modified by the original agent, **When** evidence is compiled, **Then**
   generated content and all derived content remain tainted or quarantined by default.
5. **Given** a downstream stage identity, **When** it requests evidence, **Then** it receives only
   authorized spans within an explicit character budget and cannot read raw or quarantined content.

---

### User Story 2 - Label and Select Candidate Traces (Priority: P2)

An evaluation engineer describes a desired trace property as a LabelSpec. The system compiles
structured conditions into deterministic queries and invokes a semantic labeling agent only for the
remaining contextual judgment. Every decision includes evidence, confidence, rule and model versions,
and an abstain path. When enabled by UserApprovalPolicy, the system first presents the detailed label
plan, examples, deterministic/semantic split, and blind spots for user acceptance or adjustment.

**Why this priority**: Candidate mining must be more accurate and auditable than keyword scripts while
remaining inexpensive enough for batches of thousands of traces.

**Independent Test**: Apply one structured label and one semantic label to the R0 canary set, compare
decisions and evidence spans with frozen reference fixtures, exercise user accept/adjust on the label
plan, and resume after an injected interruption without reprocessing completed traces.

**Acceptance Scenarios**:

1. **Given** a label for an executed PowerShell error signature, **When** traces are scanned, **Then**
   only actual matching tool executions are accepted and user-quoted examples are excluded.
2. **Given** a label with deterministic and semantic clauses, **When** a trace fails a deterministic
   prerequisite, **Then** no semantic model call is made for that trace.
3. **Given** ambiguous semantic evidence, **When** the model cannot decide reliably, **Then** the
   result is abstain or needs-review rather than an unsupported positive label.
4. **Given** an interrupted batch, **When** the labeling job resumes, **Then** only incomplete
   trace-label pairs are scheduled.
5. **Given** `LABEL_PLAN` is enabled, **When** the user adjusts a rule or example boundary, **Then** a
   new LabelSpec version is created and labeling waits for the current plan decision.

---

### User Story 3 - Reconstruct a Candidate Evaluation Task (Priority: P3)

An evaluation author selects a labeled trace. The system groups deterministic interaction segments into
semantic task episodes, reconstructs a self-contained one-turn prompt, derives candidate rubrics and
evaluator configuration, and preserves requirement lineage without exposing the original agent's final
answer or accidental implementation path. When configured, the user accepts or adjusts rewrite style,
fidelity, and examples before batch authoring.

**Why this priority**: A trace is not itself an evaluation question. The task must preserve the intended
capability while remaining understandable, solvable, and free of selection leakage.

**Independent Test**: Reconstruct a task from an approved canary trace and verify that a reader who
cannot access the trace can understand the task, while private rubrics, selection signals, and reference
material remain outside the contestant and attachment-producer views.

**Acceptance Scenarios**:

1. **Given** a trace containing retries and continuation turns, **When** the task is reconstructed,
   **Then** the prompt describes the stable user intent without replaying operational noise.
2. **Given** an original agent final answer, **When** prompt and rubric candidates are produced,
   **Then** final-answer content and accidental solution steps do not enter the visible prompt.
3. **Given** a rubric criterion, **When** reachability is checked, **Then** the criterion identifies
   its judged object and can be satisfied from the prompt, allowed tools, and reconstructable inputs.
4. **Given** hidden selection signals, **When** the task-authoring agent runs, **Then** it receives
   only the approved SelectionContext projection.
5. **Given** `TASK_REWRITE_PLAN` is enabled, **When** the user adjusts rewrite style or fidelity,
   **Then** a new versioned plan is applied before task generation and prior examples remain immutable.

---

### User Story 4 - Reconstruct Input-State Attachments (Priority: P4)

An operator requests attachments for a candidate task. The system computes evidence sufficiency per
artifact, selects `TRACE_RICH`, `SKELETON_GUIDED`, `PROMPT_ONLY`, or `BLOCKED`, retrieves approved
public evidence where required, and invokes deterministic providers or specialized workers to build a
safe input-state package.

**Why this priority**: Attachments make the task executable, but they are also the highest-risk path for
copying the original agent's answer into a test.

**Independent Test**: Build representative artifacts in all three reconstruction modes, inject one
artifact failure, and verify that only the failed artifact is retried while successful artifacts and
their lineage remain unchanged.

**Acceptance Scenarios**:

1. **Given** complete untainted pre-mutation evidence, **When** an artifact is planned, **Then** it may
   use `TRACE_RICH` restoration with source spans and content hashes.
2. **Given** structure but incomplete safe content, **When** an artifact is planned, **Then** it uses
   `SKELETON_GUIDED` reconstruction and records all external grounding.
3. **Given** only the task prompt, **When** dependencies are discovered, **Then** the system uses
   `PROMPT_ONLY` synthesis and does not pretend to have recovered an original file.
4. **Given** a critical missing capability, **When** preflight runs, **Then** the artifact is
   `BLOCKED_CAPABILITY` and is not silently replaced with a low-fidelity file.
5. **Given** an attachment worker identity, **When** it requests task context, **Then** it can read
   ProducerTaskView and approved evidence only, not TaskDraft, rubrics, evaluator rules, hidden labels,
   or private references.
6. **Given** a task needs a special account or non-standard environment, **When**
   `ENVIRONMENT_STRATEGY` is enabled, **Then** the user chooses trace-faithful mock, standard-environment
   rewrite, or task exclusion, and separately chooses whether `query.yaml` is included.

---

### User Story 5 - Validate, Optionally Inspect, and Release Evaluation Items (Priority: P5)

Deterministic validators and three sequential semantic reviewer agents assess coverage and solvability,
realism and consistency, and leakage and executability. Batch checks detect cross-item duplication and
contamination. The requesting user may optionally inspect rewritten prompts, selected items, or the
complete dataset and accept, adjust, reject, or defer against immutable object versions.

**Why this priority**: Prompt and attachment quality alone do not create a valid evaluation item. The
published object must include scoring contracts, provenance, quality evidence, and a controlled release
decision.

**Independent Test**: Assemble a reference EvaluationItem, inject an answer leak and a stale user
decision, and verify that both block release when the checkpoint is required. Then release an unchanged
item to the canary channel with all configured checkpoints satisfied and confirm that production still
fails without a valid ProductionReadinessAttestation.

**Acceptance Scenarios**:

1. **Given** an item with an open P0 or P1 finding, **When** approval is requested, **Then** no
   approvable QualityReport or ReleaseDecision is produced.
2. **Given** a modified plan, prompt, artifact, rubric, policy, or quality report, **When** an older
   user decision is evaluated, **Then** its subject hash mismatch invalidates the decision.
3. **Given** a canary-approved item, **When** it is exported, **Then** it enters an isolated canary or
   internal-review registry and cannot appear in production.
4. **Given** an active ProductionReadinessAttestation covering current versions, **When** a new
   production decision validates the same release subject and package hash, **Then** the item may enter
   the production registry.
5. **Given** a batch-scoped leakage finding, **When** release is evaluated, **Then** every referenced
   item is blocked until the BatchQualityReport finding is closed or superseded.
6. **Given** answer-bearing content derived from the original agent's final output, **When** the user
   attempts to accept the item, **Then** the release gate still rejects it; the content must be
   removed, reconstructed, or the item rejected.
7. **Given** final dataset review is disabled, **When** automated quality and release gates pass,
   **Then** the system does not fabricate a user approval record.

---

### User Story 6 - Run and Resume the End-to-End Factory (Priority: P6)

An operator submits a DatasetJobSpec for one or many traces. A deterministic orchestrator resolves the
stage DAG, fans out independent work within resource and model limits, records immutable stage results,
and resumes safely after process, model, network, or user-checkpoint interruption.

**Why this priority**: The system is valuable at production scale only when batch execution is bounded,
recoverable, and observable.

**Independent Test**: Run a multi-trace canary job, inject failures at parser, model, artifact, user
checkpoint, and export boundaries, then verify that each item resumes from its smallest incomplete unit
without duplicating successful external side effects.

**Acceptance Scenarios**:

1. **Given** a stage list with `label` before `trace_index`, **When** the job is validated, **Then** it
   is rejected or deterministically resolved and the resolved plan is persisted.
2. **Given** a retryable stage failure, **When** retry occurs, **Then** a new StageRun references the
   failed run and the old StageResult remains immutable.
3. **Given** model rate limits, **When** pressure increases, **Then** token and request buckets apply
   backpressure without silently switching to a lower-quality path.
4. **Given** a cancelled job, **When** cancellation completes, **Then** checkpoints and successful
   artifacts remain auditable and no new work is dispatched.

### Edge Cases

- The outer trace record parses but the nested request or response is malformed.
- A trace is truncated after a tool call but before its result.
- Duplicate or missing tool-call IDs prevent deterministic pairing.
- A file is read, edited in place, then read again under the same path.
- The original agent writes a file before any input-side observation exists.
- A generated file is copied, summarized, translated, or embedded in another file.
- Public search evidence conflicts with trace evidence or has no stable retrieval source.
- Prompt-only dependency discovery proposes the requested final deliverable as an input.
- Two artifact workers attempt to write the same path.
- A semantic review finishes after an upstream artifact hash has changed.
- A user adjusts a label or rewrite plan after downstream work exists.
- A required user checkpoint has a stale plan or subject hash.
- A task requires credentials or a special service that cannot be safely mocked.
- `query.yaml` is requested for some profiles and omitted for others.
- A canary ReleaseDecision is presented to the production registry.
- An attestation expires or a safety policy changes after production approval.
- A job contains no selected items, some successful items, or only final failures.

## Requirements

### Functional Requirements

#### Trace Intelligence

- **FR-001**: The system MUST accept a DatasetJobSpec and an immutable manifest of trace references.
- **FR-002**: The first adapter MUST support the current `raw_traj` format without modifying source
  files.
- **FR-003**: Parsing MUST implement strict parse, audited local repair, and streaming recovery as
  separate outcomes.
- **FR-004**: Every repair MUST record the source byte range, decoded character range, rule, transform,
  exactness, and raw content hash.
- **FR-005**: TraceIR MUST represent user, assistant, system, tool-call, tool-result, runtime-error,
  continuation, and attachment-reference events.
- **FR-006**: The system MUST normalize tool families, pair calls and results where evidence permits,
  detect orphan records, and retain original source spans.
- **FR-007**: The system MUST construct an immutable file-version timeline with operation, observed
  range, completeness, truncation, content hash, and source events.
- **FR-008**: R1 MUST create stable deterministic InteractionSegments. Semantic TaskEpisodes MAY group
  or annotate those segments but MUST NOT rewrite TraceIR membership or source facts.
- **FR-009**: TraceQueryService MUST enforce consumer identity, purpose, privacy profile, taint policy,
  and response-size limits for every query.
- **FR-010**: Downstream agents MUST NOT receive unrestricted raw trace text.

#### Provenance, Privacy, and Safety

- **FR-011**: Every evidence object MUST be classified as user input, pre-existing workspace input,
  harness context, agent-retrieved external content, agent-generated intermediate, agent-generated
  final, or unknown.
- **FR-012**: The system MUST propagate taint through writes, edits, copies, summaries, translations,
  embeddings, and other derived-content edges.
- **FR-013**: Agent-generated and unknown content MUST default to quarantine or review; an agent MUST
  NOT clear taint by itself.
- **FR-014**: Evidence views MUST separate privileged audit data, default-safe data, attachment-producer
  data, evaluator data, and contestant-visible data.
- **FR-015**: Secrets and configured personal data MUST be redacted before model disclosure and MUST
  NOT enter an exported package.
- **FR-016**: First-release model disclosure MUST default to approved internal model domains only.
- **FR-017**: Trace content that contains instructions MUST be treated as untrusted data, not executable
  system instructions.

#### Labeling and Selection

- **FR-018**: Natural-language labeling requirements MUST compile into a versioned LabelSpec before
  batch execution.
- **FR-019**: Structured predicates, negative conditions, sequences, windows, and error signatures MUST
  execute deterministically.
- **FR-020**: Semantic model calls MUST be limited to residual conditions that cannot be decided from
  structured evidence.
- **FR-021**: Every LabelDecision MUST include positive, negative, and semantic evidence as applicable,
  confidence, rule version, model profile, prompt version, and decision status.
- **FR-022**: The labeling path MUST support abstain and MUST route low-confidence or conflicting
  decisions to typed unresolved or user-inspection queues according to UserApprovalPolicy.
- **FR-023**: Task reconstruction MUST consume an approved SelectionContext that excludes hidden label
  signals and error signatures not intended for the task author.

#### Task and Evaluation Contract Reconstruction

- **FR-024**: The system MUST produce a versioned TaskDraft with prompt, task intent, evaluation claim,
  capabilities, allowed tools, forbidden outputs, attachment dependencies, uncertainties, and
  requirement lineage.
- **FR-025**: The visible prompt MUST exclude final answers, already completed deliverables, private
  references, grader rules, and accidental trajectory-specific solution steps.
- **FR-026**: The system MUST support imported rubrics and generated rubric candidates with explicit
  judged objects, reachability evidence, visibility, evaluator binding, and approval status.
- **FR-027**: EvaluatorSpec MUST distinguish contestant failure, evaluator failure, environment failure,
  and indeterminate results.
- **FR-028**: ReferencePolicy MUST support no reference, structured expectations, private answer,
  trace-behavior reference, and human-only modes.
- **FR-029**: Private references MUST be inaccessible to contestant and attachment-producer identities.
- **FR-030**: ProducerTaskView MUST be generated by a deterministic projection and MUST exclude
  evaluation claims, rubrics, selection signals, lineage, evaluator configuration, and private
  references.

#### Attachment Reconstruction

- **FR-031**: Reconstruction mode MUST be computed per artifact, not once per task.
- **FR-032**: Supported modes MUST include `TRACE_RICH`, `SKELETON_GUIDED`, `PROMPT_ONLY`, and
  `BLOCKED`; a task MAY aggregate to `MIXED`.
- **FR-033**: Mode selection MUST use an ArtifactEvidenceMatrix containing path, type, structure,
  untainted coverage, pre-mutation evidence, provenance confidence, truncation, criticality, and
  blocking uncertainty.
- **FR-034**: Artifact workers MUST receive only ProducerTaskView, ArtifactBuildSpec, and authorized
  evidence spans.
- **FR-035**: Deterministic providers MUST be selected before open-ended runtimes when they satisfy the
  artifact contract.
- **FR-036**: External research MUST record source, retrieval time, usage basis, content hash, and the
  artifact claims it grounds.
- **FR-037**: Critical capability gaps MUST return an explicit blocked status and MUST NOT silently
  produce a lower-fidelity substitute.
- **FR-038**: ArtifactBuildResult MUST bind output content, lineage, validation, findings, worker
  version, and retry scope to an immutable build-spec hash.
- **FR-039**: AttachmentReconstructionResult MUST identify the smallest failed and resumable artifact
  sets and MUST map accepted artifacts to EnvironmentSpec, ProvenanceManifest, and QualityReport.
- **FR-040**: Exported workspace packages MUST contain input-state files only.

#### Quality, User Approval, and Release

- **FR-041**: Deterministic validators MUST run before semantic review and their P0 or P1 findings MUST
  NOT be downgraded by a reviewer agent.
- **FR-042**: Standard item review MUST execute three semantic rounds in order: coverage and
  solvability; realism and consistency; leakage and executability.
- **FR-043**: A later semantic round MUST consume the repaired and revalidated output of prior rounds.
- **FR-044**: Every finding and result MUST bind to the hashes of the objects it evaluated and become
  stale when those hashes change.
- **FR-045**: Batch review MUST distinguish item-scoped findings from findings that exist only across a
  set of items.
- **FR-046**: DatasetJobSpec MUST bind a versioned UserApprovalPolicy that enables no checkpoints,
  plan checkpoints, final review, both, or an explicit custom subset.
- **FR-047**: Enabled checkpoints MUST use immutable UserApprovalRequests and UserDecisionRecords bound
  to exact plan, subject, policy, and projection hashes; disabled checkpoints MUST NOT create synthetic
  approval records.
- **FR-047A**: Requesting-user checkpoint projections MUST follow Data Classification Policy v2 and
  MUST NOT disclose raw traces, private evaluator material, credentials, or content outside the
  checkpoint purpose.
- **FR-048**: A complete EvaluationItem MUST reference QuerySpec, EnvironmentSpec, RubricSet,
  EvaluatorSpec, ReferencePolicy, ToolPolicy, ProvenanceManifest, QualityReport, and ReleaseDecision.
- **FR-049**: ReleaseDecision MUST be the sole authority for approval and release state; item state MUST
  be a transactionally consistent projection.
- **FR-050**: Release approval and publication MUST revalidate release-subject and package hashes.
- **FR-051**: Canary and internal-review packages MUST remain isolated from the production registry.
- **FR-052**: Production publication MUST require a current ProductionReadinessAttestation covering
  the active system, schema, and policy versions.

#### Orchestration and Audit

- **FR-053**: The control plane MUST validate a canonical stage DAG beginning with `trace_index` before
  `label`.
- **FR-054**: Job, Item, StageRun, user checkpoint, and release transitions MUST reject unspecified
  state changes.
- **FR-055**: Stage results and release decisions MUST be immutable; retries and later decisions MUST
  create linked records.
- **FR-056**: State transition, output references, checkpoint, and outbox event MUST commit atomically.
- **FR-057**: Repeated idempotency keys MUST return committed results without repeating external side
  effects.
- **FR-058**: The scheduler MUST enforce independent model request, token, process, renderer, network,
  and storage resource limits.
- **FR-059**: The system MUST checkpoint at job, item, stage, and artifact granularity and resume only
  incomplete work.
- **FR-060**: The first release MUST provide CLI operations for trace ingestion, inspection, labeling,
  task reconstruction, attachment construction, pipeline run and resume, user checkpoint decisions,
  and export.
- **FR-061**: Every released item MUST be traceable to source hashes, parser and policy versions, label
  decisions, prompts, models, tools, configured user decisions, quality reports, and package hashes.
- **FR-062**: LH export MUST remain supported. Additional Generic export MUST be enabled only by an
  approved release-profile decision.

#### Cross-Cutting Hard Gates

- **FR-063**: Every required attachment dependency MUST bind to prompt, trace, rubric, existing-input,
  or approved-public-source evidence. Evidence priority MUST be direct observable facts, explicit task
  requirements, then inferred dependencies; lower-priority inference MUST NOT override contradictory
  higher-priority evidence.
- **FR-064**: Content that exposes the requested final answer or deliverable, private reference,
  grader rule, hidden pass condition, or original-agent final-output derivation MUST be a
  release-blocking finding that no agent or user decision can waive. Resolution is limited to removal,
  safe reconstruction, or item rejection. An unresolved finding MUST block every release channel.
- **FR-065**: Each semantic quality round MUST use its own reviewer role, StageRun, and clean runtime
  context, isolated from artifact-construction transcripts and other reviewers' hidden reasoning. A
  later round MAY consume version-bound findings, repairs, and validated outputs from earlier rounds.
- **FR-066**: An enabled `LABEL_PLAN` checkpoint MUST present label boundaries, deterministic and
  semantic clauses, positive/negative/ambiguous examples, abstain rules, expected model path, and blind
  spots before batch labeling.
- **FR-067**: An enabled `TASK_REWRITE_PLAN` checkpoint MUST present capability intent, rewrite style,
  fidelity, examples, operational-noise handling, and forbidden-content rules before batch task
  authoring.
- **FR-068**: For tasks needing special accounts or environments, an enabled
  `ENVIRONMENT_STRATEGY` checkpoint MUST support `TRACE_FAITHFUL_MOCK`, `REWRITE_STANDARD_ENV`, and
  `EXCLUDE_TASK`, with capability impact and invalidation recorded.
- **FR-069**: Environment strategy MUST independently support `INCLUDE_QUERY_YAML`,
  `OMIT_QUERY_YAML`, and `ASK_PER_TASK`.
- **FR-070**: The active workflow MUST NOT require independent reviewer identities, review quorums,
  claim leases, appeals, or fabricated HumanReviewRecords.
- **FR-071**: Final prompt, item, or dataset inspection MUST be optional and user-configured; skipping
  it MUST NOT disable deterministic, semantic, privacy, provenance, or release gates.
- **FR-072**: One idempotent Graph Engineering lifecycle command MUST accept a versioned evaluation
  requirement and admitted `raw_traj_v1` manifest, advance until the next mandatory review or typed
  terminal state, and resume the same dataset run without repeating completed side effects.
- **FR-073**: One dataset parent run MUST own global plan, Core partition, BatchQuality, Final
  Delivery, and output authority. Every candidate MUST use one deterministic child run and exact
  `FactoryItemRunBindingV2`; non-candidates and Core-blocked traces MUST NOT receive children.
- **FR-074**: `FactoryControlStore`, `JobStore`, LangGraph checkpoint storage, private CAS, and output
  CAS MUST retain distinct ownership. Cross-store composition MUST use immutable refs; checkpoint
  state MUST contain only cursor, closed state, bounded counts, and safe refs.
- **FR-075**: Candidate, non-candidate, rejected, and blocked source partitions MUST be exact and
  disjoint. One typed candidate failure MUST NOT prevent unrelated child runs from reaching candidate
  delivery.
- **FR-076**: Factory `ITEM_QUALITY` authority MUST preserve Attachment-stage base quality. Criteria
  private authority and BatchQuality MUST bind the Criteria-reviewed quality for the complete R4
  contract; JobStore ItemQuality witnesses and R7 candidate dependencies MUST use that reviewed
  quality.
- **FR-077**: R7 candidate projection MUST require current Task Authoring, Attachment base
  ItemQuality, Criteria material/result, Grading result, reviewed ItemQuality, BatchQuality, JobStore
  work, package, provenance, and configured user-decision authority.
- **FR-078**: A reviewed `DatasetDeliveryPlanV2` MUST freeze equal candidate Item, R7 projection, and
  current successful `RELEASE_CANDIDATE` stage-head inventories. A Final Delivery edit MAY narrow
  output limits but MUST NOT replace fixed candidate authority.
- **FR-079**: Candidate output MUST use staging, complete hash/inventory verification, fsync, atomic
  rename, orphan verification/reuse, and one authority transaction. A physical bundle without current
  Store authority MUST remain unreachable.
- **FR-080**: Candidate completion MUST remain separate from canary/internal publication and
  `PRODUCTION_READY`. This lifecycle MUST NOT issue SC-010 through SC-015 evidence or change
  `PRODUCTION_RELEASE_BLOCKED / REPOSITORY_PENDING_ONLY`.

### Key Entities

- **DatasetJobSpec**: User-requested pipeline, inputs, privacy profile, model profiles, budgets,
  concurrency, selection rules, and export target.
- **TraceEnvelope / TraceEvent**: Immutable normalized trace identity, parse quality, capabilities, and
  ordered event facts.
- **SourceSpan / RepairMap**: Auditable mapping from derived facts to raw bytes and any recovery
  transforms.
- **ToolCallRecord / FileObservation / InteractionSegment**: Deterministic indexes for tool execution,
  file versions, and conversation boundaries.
- **ProvenanceDecision / TaintEdge**: Classification and derived-content propagation records.
- **EvidenceBundle**: Safe, purpose-bounded references to approved evidence without copying the complete
  trace.
- **LabelSpec / LabelDecision**: Versioned labeling intent, execution plan, result, confidence, and
  evidence.
- **SelectionContext**: Minimal task-authoring context that removes hidden selection signals.
- **TaskEpisode / TaskDraft**: Semantic task view and versioned reconstructed candidate task.
- **ProducerTaskView**: Deterministic least-privilege projection for attachment workers.
- **ArtifactEvidenceMatrix / ArtifactBuildSpec**: Per-artifact information sufficiency and construction
  contract.
- **ArtifactBuildResult / AttachmentReconstructionResult**: Immutable artifact and package outcomes,
  lineage, validation, and retry scopes.
- **QuerySpec / EnvironmentSpec / ToolPolicy**: Contestant-visible task, input workspace, and allowed
  capabilities.
- **RubricSet / EvaluatorSpec / ReferencePolicy**: Isolated evaluation contract and private-reference
  controls.
- **QualityReport / BatchQualityReport / ReviewerFinding**: Item and cross-item quality evidence.
- **FactoryDatasetRunRequest / FactoryItemRunBinding / FactoryItemStageHead**: Versioned parent intake,
  deterministic parent-child binding, and immutable-history/rebuildable-current stage authority for
  the production-neutral multi-Agent lifecycle.
- **DatasetDeliveryPlan / CandidateDatasetDeliveryManifest**: Reviewed candidate/R7 stage-head
  inventory and the physical candidate bundle's exact Item, projection, quality, provenance, and
  inventory authority.
- **UserApprovalPolicy / UserApprovalRequest / UserDecisionRecord**: User-configured stage checkpoints,
  exact version-bound requests, and immutable accept/adjust/reject/defer decisions.
- **EvaluationItem / ReleaseDecision**: Complete versioned evaluation object and authoritative
  canary/production decision chain.
- **ProductionReadinessAttestation**: System-level proof that statistical, safety, privacy, stability,
  and operational gates passed for a version set.

## Non-Functional Requirements

- **NFR-001**: Python 3.12, Pydantic v2, `uv`, and the existing LangGraph control plane remain the
  baseline unless superseded by an approved ADR.
- **NFR-002**: Provider-neutral modules MUST NOT import vendor SDKs.
- **NFR-003**: Raw trace parsing, indexing, provenance propagation, state transitions, and release gates
  MUST be deterministic and testable without live model credentials.
- **NFR-004**: Identical raw hashes, adapter versions, repair policies, and index policies MUST produce
  identical deterministic TraceIR and indexes.
- **NFR-005**: The parser canary event-recovery target is at least 99%; call/result pairs with valid
  explicit IDs must be 100% accurate.
- **NFR-006**: Known output-side contamination recall and known answer-leakage recall on the approved
  safety gold set MUST be 100%.
- **NFR-007**: Quarantined-content access and trace prompt-injection execution counts MUST be zero.
- **NFR-008**: Structured-label precision and recall targets are at least 0.99 on a frozen independent
  set containing at least 50 positive and 50 negative traces per structured label. Semantic-label
  macro-F1 target is at least 0.85 on at least 100 frozen independent traces per semantic label, with
  class balance used where the source population permits. Train, development, and test partitions MUST
  be isolated by trace ID.
- **NFR-009**: Canary metrics MUST NOT be reported as production statistical evidence.
- **NFR-010**: No critical attachment may be silently degraded.
- **NFR-011**: A single artifact failure MUST NOT require rebuilding successful unrelated artifacts.
- **NFR-012**: Job cancellation and retry MUST leave durable checkpoints and no untracked external
  side effects.
- **NFR-013**: Batch scheduling MUST apply backpressure before provider rate limits are exhausted.
- **NFR-014**: User-visible packages MUST exclude raw traces, quarantine content, process transcripts,
  internal paths, secrets, grader prompts, and privileged provenance.
- **NFR-015**: Local process execution remains trusted and monitored only; untrusted unattended
  execution is outside the first release.
- **NFR-016**: All schemas, policies, prompts, adapters, providers, validators, evaluators, and export
  profiles MUST be versioned.
- **NFR-017**: The system MUST remain recoverable in a fresh development session through Spec Kit,
  Beads, ADRs, generated project state, and typed handoffs.
- **NFR-018**: A disabled user checkpoint MUST add no queue, reviewer identity, quorum, lease, or
  approval latency, and MUST emit no synthetic decision.

## Success Criteria

### R0-R7: `E2E_CANARY_COMPLETE`

- **SC-001**: At least 20 development canaries traverse the requested pipeline, with every failure
  assigned a typed cause rather than hidden or retried indefinitely.
- **SC-002**: At least five canaries have complete review-ready EvaluationItem reference fixtures and
  user-checkpoint previews; user acceptance is recorded only when enabled.
- **SC-003**: The parser meets the approved R1 event and pairing targets on parser gold.
- **SC-004**: All known output-side content in the R0 safety gold set is quarantined, and no
  post-mutation-only content enters input evidence.
- **SC-005**: At least one structured and one semantic label produce evidence-bound decisions, including
  abstain and user-plan adjustment paths.
- **SC-006**: `TRACE_RICH`, `SKELETON_GUIDED`, and `PROMPT_ONLY` each produce at least one safe,
  executable successful package. Separate canaries MUST verify `BLOCKED` and `BLOCKED_CAPABILITY`;
  blocked outcomes do not count as successful mode coverage.
- **SC-007**: Every canary item completes deterministic validation and three ordered semantic review
  rounds, or is explicitly rejected.
- **SC-008**: Approved items can be exported to an isolated canary or internal-review registry and
  reconstructed from provenance and package hashes.
- **SC-009**: Injected failures at parser, labeling, artifact, user-checkpoint, and export boundaries
  resume from the smallest incomplete unit.
- **SC-009A**: One production-neutral lifecycle run produces at least two current
  `EvaluationItemV2` candidates while isolating at least one typed blocked source, then rebuilds
  byte-identical parent/item/output authority in fresh processes under `PYTHONHASHSEED=1` and `321`.

### R8: `PRODUCTION_READY`

- **SC-010**: Frozen independent label tests meet NFR-008 with point estimates, 95% confidence intervals,
  sample counts, dataset version, and freeze evidence.
- **SC-011**: Stability regression uses at least 100 independent real traces. If 100 eligible traces
  are unavailable, the result remains `STATISTICAL_GATE_PENDING` and no ProductionReadinessAttestation
  may be issued. Synthetic and replay loads are reported separately and never counted as quality
  coverage.
- **SC-012**: Concurrency experiments establish default resource limits with measured throughput,
  p50/p95 latency, model usage, cost, queue wait, error rate, and resume success.
- **SC-013**: Security, privacy, release-permission, and operations-SLO approvals are complete.
- **SC-014**: No statistical gate, P0/P1 finding, required user checkpoint, or release-integrity check
  remains open.
- **SC-015**: A valid ProductionReadinessAttestation is issued and a new production ReleaseDecision
  publishes an unchanged approved subject to the production registry.

## Scope Boundaries

### In Scope for the First Release

- The current single `raw_traj` source format through a versioned adapter.
- Approved internal model domains.
- Structured and semantic labeling.
- One-turn task reconstruction and evaluation-contract authoring.
- Trace-rich, skeleton-guided, and prompt-only attachment reconstruction.
- Existing deterministic providers, validators, runtimes, and LH profile integration.
- CLI/JSON user approval checkpoints with versioned plan previews and decisions.
- Local batch CLI and durable resource-aware orchestration.
- Canary/internal and production release separation.

### Out of Scope for the First Release

- Free-form autonomous agent-to-agent handoff or swarm planning.
- A separate agent implementation for each information mode.
- Direct downstream-model ingestion of full raw traces.
- Additional trace-source adapters beyond `raw_traj_v1`.
- An untrusted multi-tenant execution service.
- Fully automatic approval of high-risk provenance or private-reference changes.
- Mandatory independent-human review, reviewer-role queues, dual-review quorums, leases, appeals, and
  Argilla-owned workflow state.
- Declaring production quality from development canaries, synthetic traces, or repeated copies.
- Replacing the existing attachment providers and runtimes without migration evidence.

## Assumptions

- The first implementation remains in the current repository as bounded contexts until the Repo
  Boundary ADR is approved.
- Trace data may be processed only in approved internal model domains under the selected privacy
  profile, and model calls may receive only authorized, redacted, purpose-bound spans through
  TraceQueryService. No model identity receives unrestricted raw trace text.
- The current attachment generator is reused behind typed request/result contracts rather than
  rewritten in R0.
- R0 uses 20-30 development canaries selected from the current `raw_traj` corpus; at least five receive
  complete review-ready EvaluationItem reference scaffolds.
- LH is the mandatory first release profile. Generic release remains an approval-gated extension.
- User plan adjustments are versioned and auditable; they never overwrite source TraceIR, prior plans,
  task versions, artifacts, rubrics, quality results, or decisions.

## R0 Approval Gates

| Decision | Draft default | Required durable output |
|---|---|---|
| Repository boundary | Current monorepo with bounded contexts | Repo Boundary ADR |
| Canary composition | 20-30 stratified traces; at least five complete review-ready fixtures | Canary manifest and annotation guide |
| Model data classification | Internal-only; least necessary disclosure | Privacy/data classification record |
| User approval model | Optional stage checkpoints; no independent-review subsystem | ADR 0011 and User Approval Policy v2 |
| Representative first labels | Search-tool usage, PowerShell error signature, one contextual semantic behavior | Approved LabelSpecs and gold annotations |
| Release profiles | LH required; Generic deferred until explicitly approved | Profile ADR or release decision |

No implementation beyond R0 contract spikes may start while its governing decision remains unapproved.
