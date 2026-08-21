# ADR 0010: Enforce Input-State Provenance, Taint, And Least-Privilege Evidence Views

- Status: Accepted
- Date: 2026-07-19
- Beads: `env_mock_agent-ujc.1.4`
- Spec: `specs/002-eval-dataset-factory/spec.md`
- Depends on: ADR 0009

## Context

The source trace records both the environment that existed before the original agent acted and the
agent's later reasoning, writes, conclusions, and final deliverables. Recovering attachment content
without a temporal and derivation boundary can copy the answer into the evaluation input.

The existing attachment subsystem blocks obvious final deliverables, secrets, and answer-bearing text,
but the factory must also handle file mutation, copied or summarized output, private references,
retrieved public material, unknown origin, and different visibility needs across labeling, task
authoring, attachment production, evaluation, audit, and contestant execution.

Safety cannot be represented by one `trusted` flag or one provenance enum. Origin, derivation, content
risk, visibility, and release disposition are related but distinct facts.

## Decision

Every content-bearing subject receives a versioned ProvenanceDecision. Subjects include:

- TraceEvent content blocks;
- tool-call arguments and tool results;
- file observations and immutable file versions;
- interaction and task annotations;
- external-source snapshots;
- generated prompts and attachments;
- reference and evaluation objects.

The decision separates:

```text
origin_class
taint_labels[]
content_risk_labels[]
visibility
disposition
derived_from[]
rule_ids[]
source_event_ids[]
confidence
review_required
policy_version
subject_hash
```

`confidence` describes classification certainty. It never grants access or release by itself.

## Origin Classes

Initial origin classes are:

```text
USER_SUPPLIED_INPUT
PREEXISTING_WORKSPACE_INPUT
SYSTEM_OR_HARNESS_CONTEXT
AGENT_RETRIEVED_EXTERNAL
AGENT_GENERATED_INTERMEDIATE
AGENT_GENERATED_FINAL
UNKNOWN
```

Origin answers who introduced the subject and when. It does not answer whether content is safe.

Examples:

- User-provided material can still be `ANSWER_BEARING`.
- Agent-retrieved public material can still reveal the solution.
- Agent-generated intermediate content remains quarantined even if it is not final.
- Unknown origin is never promoted to input evidence by confidence alone.

## File-Version Boundary

Each normalized logical path has an immutable version timeline:

```text
V0 unknown or pre-existing
  -- observed read ranges --> V0 evidence
  -- write/edit by agent --> V1 agent-generated
  -- later read --> V1 observation, still tainted
  -- later write/edit --> V2 agent-generated
```

Rules:

1. A version existing before the target task segment and observed before its first write/edit may be
   `PREEXISTING_WORKSPACE_INPUT` only when file-mutation capability is complete from the segment boundary
   through the observation.
2. Only observed V0 ranges are usable unless an approved input workspace provides the complete immutable
   V0 and matching hash.
3. If the first observed operation is write/edit, no V0 content is recoverable. Path, media type, and
   non-answer-bearing schema may remain structure leads.
4. Every post-write version is agent-generated regardless of later read operations.
5. Copy, rename, archive, export, and re-read do not reset origin or taint.
6. Conflicting path normalizations or ambiguous write order force `UNKNOWN` and quarantine.
7. Shell, code execution, subagent, task-management, unknown tools, and any tool with unavailable
   arguments are treated as possible mutators unless a versioned tool policy proves otherwise.
8. A possible unmodeled mutation produces an unknown-version boundary for every path in its possible
   write scope. If scope cannot be bounded, all later workspace observations are `UNKNOWN` until an
   independently hashed approved workspace snapshot establishes a new baseline.

## Taint And Content-Risk Namespaces

Taint records derivation. Initial `TaintLabel` values are:

```text
FINAL_OUTPUT_DERIVED
PRIVATE_REFERENCE_DERIVED
GRADER_RULE_DERIVED
UNKNOWN_DERIVATION
SENSITIVE_SOURCE_DERIVED
UNTRUSTED_INSTRUCTION_DERIVED
```

Content risk records what the target bytes or structure expose. Initial `ContentRiskLabel` values are:

```text
ANSWER_BEARING
HIDDEN_PASS_CONDITION
SECRET
RESTRICTED_PII
PROMPT_INJECTION
UNSCANNABLE_CONTENT
```

`AGENT_GENERATED_FINAL` is assigned when any deterministic condition holds:

- path or media role matches the requested deliverable;
- the assistant declares completion or submission;
- the object is exported, packaged, submitted, or cited as the final result;
- content matches a private reference, grader target, or rubric conclusion;
- the object is used as a result for acceptance rather than an input for continued work.

Other generated content begins as `AGENT_GENERATED_INTERMEDIATE` plus quarantine. Intermediate is not a
safe-input designation.

## Derivation Graph

The system records directed, versioned edges:

```text
source subject --operation/rule--> derived subject
```

Deterministic edges include:

- file version to read result;
- tool arguments to written or edited file version;
- exact or near-exact content transfer;
- copy, move, archive, and extraction operations;
- tool result to assistant block through call identity;
- assistant block to final response through source order;
- object to normalized or redacted projection.

Semantic candidates may identify summary, translation, paraphrase, or structural transformation. A
semantic candidate can add taint; failure to detect a semantic edge can never remove taint.

## Propagation

Propagation is a total, deterministic function over parents, the derivation edge, the target actor, and
the target subject:

1. `origin_class` is assigned from the actor and temporal event that created the target; origin is not
   inherited from a parent.
2. `taint_labels` is the union of all parent taint labels plus labels added by the edge rule.
3. `content_risk_labels` is evaluated against the exact target bytes and structure. A parent risk is
   conservatively retained unless a versioned deterministic transform and matching validator prove that
   specific risk absent from the new projection.
4. `disposition` is the most restrictive of every parent disposition, the edge-rule minimum, and the
   target-rule result.
5. Missing parents, unknown transforms, incomplete content, validator failure, or ambiguous rules add
   `UNKNOWN_DERIVATION` or `UNSCANNABLE_CONTENT` and fail closed to quarantine or reject.

Safety restrictiveness is:

```text
REJECT
  > QUARANTINE
  > NEEDS_REVIEW
  > ALLOW_EXTERNAL_LEAD_ONLY
  > ALLOW_STRUCTURE_ONLY
  > ALLOW_INPUT_EVIDENCE
```

The same function applies to zero, one, or many parents; there is no single-parent bypass. No operation
mutates a tainted subject into a clean subject. If independently verified clean evidence
supports equivalent content, the system creates a new subject with its own lineage and hash; the old
subject remains tainted.

Redaction removes disclosed sensitive spans from a new projection and may clear `SECRET` or
`RESTRICTED_PII` content risk only when the redaction validator proves the target projection contains no
such spans. It retains `SENSITIVE_SOURCE_DERIVED`. Redaction never clears answer, final-output, grader,
private-reference, or hidden-pass-condition restrictions.

## Content-Risk Classification

Answer leakage is evaluated relative to:

```text
Evaluation Claim
  + visible Prompt
  + RubricSet
  + ReferencePolicy
  + ToolPolicy
  + trace and artifact lineage
```

A subject is release-blocking when it:

1. contains a result the contestant must derive, calculate, choose, or create;
2. contains all or a material part of a private reference answer;
3. derives from the original agent's final output;
4. exposes an accidental trajectory as the required solution path;
5. materially lowers the capability measured by the evaluation claim;
6. reveals grader rules, thresholds, hidden assertions, or pass conditions;
7. lets the task be completed by copying, renaming, or superficial rewriting.

Public facts, user-provided conclusions, examples, schemas, and interfaces are not automatically
leakage. They still require evaluation-claim-aware review. A user-provided answer-like document may be
valid when the task explicitly asks the contestant to analyze that document.

## Non-Waivable Release Gate

The following predicate cannot be waived by an agent, one reviewer, two reviewers, or an administrator:

```text
taint_labels intersects {
  FINAL_OUTPUT_DERIVED,
  PRIVATE_REFERENCE_DERIVED,
  GRADER_RULE_DERIVED
}
OR
content_risk_labels intersects {
  ANSWER_BEARING,
  HIDDEN_PASS_CONDITION
}
```

Resolution is limited to:

- remove the subject from every visible package;
- create a new independently grounded safe reconstruction;
- change the evaluation claim through a new approved task version and re-run all dependent gates;
- reject the item.

An unresolved non-waivable finding blocks canary, internal-review, and production release.

## Dispositions

Every decision uses exactly one disposition:

```text
ALLOW_INPUT_EVIDENCE
ALLOW_STRUCTURE_ONLY
ALLOW_EXTERNAL_LEAD_ONLY
NEEDS_REVIEW
QUARANTINE
REJECT
```

Only `ALLOW_INPUT_EVIDENCE` can provide content to attachment reconstruction.

`ALLOW_STRUCTURE_ONLY` permits only fields named in a versioned recursive structure-projection schema.
The initial schema can include normalized relative path, approved media type, dimensions, sheet/table
counts, field names, and type signatures when each field passes its risk rule. It cannot expose
answer-bearing file names, values, comments, formulas, sample answers, hidden worksheets, reference
identifiers, arbitrary metadata, or unknown extension fields.

`ALLOW_EXTERNAL_LEAD_ONLY` permits source locator and retrieval intent. The content must be fetched
again, verified, and recorded as a new SourceEvidence object before use.

`NEEDS_REVIEW` cannot enter a contestant or producer view while unresolved.

## Least-Privilege Views

Views are deterministic projections bound to authenticated principals and policy versions. Each
principal-purpose pair has a versioned `ProjectionPolicy`:

```text
principal_type
purpose
source_schema_versions[]
recursive_allowed_field_paths[]
recursive_denied_field_paths[]
maximum_items
maximum_characters
unknown_field_policy=REJECT
projection_policy_version
```

Projection is deny by default at every nesting level. Unknown fields, polymorphic variants, and schema
extensions are rejected until the policy is versioned. Each projection creates a new subject with
source hashes, projection hash, ProvenanceDecision, and audit event.

| Principal | Allowed view | Explicitly denied |
|---|---|---|
| ingestion/index service | raw source and canonical facts required for parsing | release authority |
| privileged auditor | approved raw and quarantine inspection with reason and audit event | content mutation, release |
| structured labeler | indexes, normalized facts, authorized spans | raw store, private reference, generated final content |
| semantic labeler | bounded safe spans for declared residual question | full trace, hidden task authoring data |
| task author | SelectionContext, safe interaction evidence, approved capabilities | hidden selection signals, final output, private reference |
| attachment producer | ProducerTaskView, build specs, safe EvidenceBundle spans | full TaskDraft, rubric, evaluator, private reference, quarantine |
| coverage reviewer | QuerySpec public view, environment manifest, public rubric requirements, bounded safe EvidenceBundle, typed prior validation | artifact build transcript, private reference, other reviewers' hidden reasoning |
| realism reviewer | repaired artifact content, WorldLedger, approved SourceEvidence, typed Round 1 findings and resolutions | build transcript, Round 1 hidden reasoning, private grader rules |
| leakage reviewer | repaired item, taint/lineage projection, evaluator contract, reference fingerprints, executability results, typed prior findings | build transcript, other reviewers' hidden reasoning, raw private reference unless separately authorized |
| evaluator | contestant outputs and evaluator-authorized reference objects | producer context, raw trace |
| contestant | QuerySpec, EnvironmentSpec, contestant ToolPolicy | provenance, rubrics unless public, grader, references, trace |
| release authority | hashes, quality reports, review records, policy attestations | direct content editing |

Request fields can only narrow principal grants. Stage runtimes have no direct credentials for raw,
canonical, quarantine, or private-reference stores.

## Privileged Audit And Review Exceptions

Privileged audit is break-glass access:

- separate role and credential;
- purpose, case, subject, returned spans, and duration logged;
- no export or copy into stage workspaces;
- access expires;
- audit cannot mutate provenance facts.

Two distinct reviewer identities are required for non-answer-leakage exceptions such as an approved
provenance ambiguity resolution or a Private Reference Policy change. Both records bind the exact
subject hash and policy version.

No review exception can override the non-waivable predicate above.

## External Material

Original-agent search and fetch results are `AGENT_RETRIEVED_EXTERNAL` and default to
`ALLOW_EXTERNAL_LEAD_ONLY`.

Before use, the factory must:

1. fetch from an approved source under current policy;
2. record source URI, retrieval time, usage basis, content hash, and supported claims;
3. scan secrets, PII, injection, licensing policy, and answer leakage;
4. create a new SourceEvidence object;
5. preserve the original trace result as restricted lineage, not as source truth.

## Secrets, PII, And Prompt Injection

Secret and privacy policy details are owned by R0-05. This ADR fixes the safety behavior:

- scans run before model disclosure and package export;
- raw and redacted projections have separate hashes and derivation edges;
- secrets are `REJECT` for package content;
- restricted PII is denied unless the approved privacy policy and purpose permit a redacted projection;
- trace instructions, tool results, and file text remain untrusted data and are never loaded as system,
  developer, tool, MCP, or plugin configuration.

## Invalidation

Changing any of these values invalidates dependent decisions and quality results:

- subject content hash or version;
- Evaluation Claim, Prompt, RubricSet, EvaluatorSpec, ReferencePolicy, or ToolPolicy;
- provenance, taint, redaction, or visibility policy version;
- lineage edges;
- external SourceEvidence content or retrieval policy.

Invalidation creates new decisions. Prior decisions and review records remain immutable.

## Release Enforcement

The release subject includes hashes for:

```text
QuerySpec
EnvironmentSpec
RubricSet
EvaluatorSpec
ReferencePolicy
ToolPolicy
ProvenanceManifest
QualityReport
BatchQualityReport
export profile and profile version
release channel and target registry
ProductionReadinessAttestation for production, or explicit none for non-production
```

The package inventory is canonical and exhaustive. It lists every directory, regular file, and nested
container member with normalized path, media type, size, and content hash. Symlinks are rejected in the
first release. ZIP-based documents, archives, workbooks, comments, hidden sheets, formulas, embedded
objects, macros, metadata, and other profile-relevant nested content must be enumerated by a
format-specific validator.

The inventory and ProvenanceManifest must form an exact set equality for all package content. Every
inventory member has one current ProvenanceDecision and complete derivation closure. Extra manifest
entries, unlisted package members, stale decisions, truncated extraction, unsupported encryption, and
unscannable content reject release.

Release requires:

- no unresolved non-waivable or P0/P1 finding;
- no unresolved `UNKNOWN` that can affect evaluation difficulty;
- current item and batch quality reports;
- human records matching the exact subject and policy versions;
- contestant package containing only approved contestant projections;
- package manifest and release subject hashes matching the ReleaseDecision.

The existing attachment `LeakageValidator` remains defense in depth for path and extractable-text
patterns. A clean result from that validator never grants visibility, clears taint, proves derivation
closure, or authorizes release.

## Alternatives Considered

### Strip Files Named Like Final Deliverables

Rejected because answers can be copied, renamed, summarized, or embedded elsewhere.

### Treat All Agent-Generated Content As Final

Rejected as an origin fact because intermediate and final have different audit meaning. Both remain
quarantined by default, while final-derived taint is non-waivable.

### Let A Reviewer Clear Taint In Place

Rejected because it destroys lineage and makes prior release evidence ambiguous.

### Detect Leakage Only During Final Review

Rejected because unsafe content would already have reached task authors and attachment producers.

### Give Every Stage A Redacted Full Trace

Rejected because redaction does not remove final-answer, grader, or selection leakage and violates least
necessary disclosure.

## Consequences

Positive:

- input/output separation follows versions and derivation rather than names;
- answer leakage remains blocked across all release channels;
- every stage receives the minimum view needed for its role;
- human exceptions are immutable and cannot silently clear prior taint;
- external material is re-grounded rather than trusted from the original trace.

Negative:

- conservative quarantine increases review and reconstruction work;
- derivation graphs and version timelines add storage and test complexity;
- semantic lineage detection can add taint but cannot safely prove absence;
- policy changes invalidate substantial downstream work by design.

## Validation

R2, R4, R5, and R7 must provide:

- pre-read/write/edit timeline gold tests;
- copy, rename, archive, summary, translation, and paraphrase propagation tests;
- multi-parent most-restrictive disposition tests;
- zero- and single-parent propagation totality tests;
- possible mutator and incomplete-interval V0 rejection tests;
- non-waivable leakage tests for every release channel and reviewer role;
- recursive projection allowlist, unknown-field rejection, and per-review-round view tests;
- external lead re-fetch and lineage tests;
- authenticated view and direct-store bypass tests;
- private-reference access tests;
- subject-hash and policy-version invalidation tests;
- batch cross-item leakage and contamination tests;
- package inventory/ProvenanceManifest exact-set and nested-container enumeration tests;
- truncated, encrypted, unsupported, and unscannable member rejection tests;
- proof that a clean legacy LeakageValidator result does not authorize release;
- contestant package boundary tests.
