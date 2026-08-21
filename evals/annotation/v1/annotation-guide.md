# Eval Dataset Factory Annotation Guide

```text
Guide version: eval-factory-annotation-guide/v1
Schema version: eval-factory-annotation/v1
Status: Approved
Frozen canary manifest:
  evals/golden/eval_factory/canary_manifest.v4.json
Frozen manifest SHA-256:
  1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6
```

## 1. Scope

This guide defines development-gold annotation for:

1. parser and recovery truth;
2. provenance, taint, and leakage truth;
3. structured and semantic label truth;
4. reconstructed task truth;
5. per-artifact reconstruction truth;
6. item release truth.

The records support R0-R7 canary development. They are not production statistical evidence and cannot
replace R8 independent frozen sets.

The authoritative contracts are the seven JSON Schemas in `schemas/` and the cross-record validator:

`contract-manifest.json` records the exact guide, schema, and validator SHA-256 values. Every
annotation record binds that manifest's SHA-256 through `contract_manifest_sha256`.

```bash
uv run python scripts/validate_eval_factory_annotations.py \
  path/to/annotations.jsonl \
  --require-adjudicated
```

## 2. Non-Negotiable Rules

1. **Bind exact versions.** Every judgment binds the source, subject, governing-policy, guide, and
   manifest hashes it evaluated.
2. **Evidence before narrative.** A rationale explains a decision; it never substitutes for evidence
   references.
3. **Missing is not negative.** Absence can support a negative decision only when the relevant parser
   capability is complete for the full decision window.
4. **Candidate is not truth.** Values under `annotation_candidates` in the canary manifest are routing
   hints only. They cannot become labels, safety findings, or task facts without annotation.
5. **Partial recovery is capability-scoped.** A trace may support positive facts from recovered spans
   while remaining unusable for negative predicates over missing ranges.
6. **Generated output is not input.** Original-agent final output, answer-bearing content, private
   reference derivation, grader rules, and hidden pass conditions are non-waivable release blockers.
7. **Unknown fails closed.** Unknown derivation, unscannable content, ambiguous mutation order, or
   unavailable critical evidence yields `INDETERMINATE`, `BLOCKED`, `QUARANTINE`, or `REJECT`.
8. **Do not copy raw content.** Annotation JSONL contains source coordinates, hashes, typed facts, and
   bounded rationale. It must not contain raw trace text, secrets, private references, or final output.
9. **No in-place correction.** A correction creates a new annotation ID and lists prior IDs in
   `supersedes`. Prior records remain unchanged.
10. **Human approval cannot waive deterministic hard gates.**

## 3. Evidence Contract

Each evidence reference identifies:

- immutable source reference and SHA-256;
- typed locator and locator reference;
- whether it supports a positive, negative, uncertainty, or contradiction judgment;
- the capability under which the evidence was observed;
- whether that capability is complete;
- whether the source coordinate is approximate.

Evidence priority is:

```text
DIRECT OBSERVABLE FACT
  > EXPLICIT TASK REQUIREMENT
  > INFERRED DEPENDENCY
```

Lower-priority evidence cannot override contradictory higher-priority evidence. Contradictions remain
explicit until adjudication.

Approximate spans may support routing and low-risk positive facts. They cannot alone support a
high-risk safety, provenance exception, or release decision.

### Negative Evidence

A negative decision requires all of:

1. an explicit `NEGATIVE` evidence reference;
2. `capability_complete=true` on that evidence;
3. the corresponding parser capability marked `COMPLETE`;
4. no unresolved contradictory higher-priority evidence.

Otherwise use `ABSTAIN`, `INDETERMINATE`, or a positive fact limited to the observed range.

## 4. Record Lifecycle

Annotation status has the following meaning:

| Status | Meaning |
|---|---|
| `DRAFT` | Work in progress; not consumable as gold |
| `SUBMITTED` | Immutable independent annotation awaiting adjudication |
| `ADJUDICATED` | Final gold candidate bound to review records and input annotations |
| `ABSTAINED` | Evidence or reviewer competence was insufficient |
| `SUPERSEDED` | Reserved for a materialized validity projection; the source log is not mutated |

Normal flow:

```text
independent SUBMITTED record(s)
  -> schema and policy validation
  -> independent adjudication
  -> ADJUDICATED record
  -> downstream gold use
```

An adjudicated record is not a mutation of its input records. It is a new record with:

- exact input annotation IDs;
- adjudicator identity and role;
- immutable HumanReviewRecord references;
- disagreement codes;
- decision and reason.

## 5. Adjudication

### Identity Rules

- The adjudicator differs from every input annotator.
- Input annotations come from distinct human identities.
- The subject author cannot provide the only approval.
- Safety and release gold require two independent submitted annotations plus an independent
  adjudicator.
- Parser, label, task, and attachment gold require at least one submitted annotation plus an
  independent adjudicator. A second annotation is required when the first annotator or adjudicator
  records material uncertainty.

### Decisions

| Decision | Use |
|---|---|
| `ACCEPT` | Input truth is supported without material change |
| `REVISE` | Adjudicator emits a corrected payload with explicit disagreement codes |
| `ABSTAIN` | Available evidence cannot resolve the question |
| `REJECT` | Inputs are invalid, policy-incompatible, or address the wrong subject/version |

Do not resolve disagreement by majority vote alone. The adjudicator must resolve the evidence and
policy interpretation, or abstain.

Recommended disagreement codes:

```text
PARSE_BOUNDARY
REPAIR_SEMANTICS
CAPABILITY_COMPLETENESS
CALL_RESULT_PAIRING
FILE_VERSION_ORIGIN
DERIVATION_CLOSURE
ANSWER_BEARING
REQUIREMENT_LINEAGE
SELECTION_FIREWALL
ATTACHMENT_SET
RECONSTRUCTION_MODE
CRITICALITY
RELEASE_GATE
EVIDENCE_CONTRADICTION
POLICY_INTERPRETATION
INSUFFICIENT_EVIDENCE
```

## 6. Parser Annotation

### Truth Units

- overall parse outcome: `STRICT`, `REPAIRED`, `PARTIAL`, or `UNPARSEABLE`;
- capability completeness by conversation, tool event, pairing, file timeline, final response, and
  attachment observation;
- every expected repair and whether it is semantically critical;
- event counts by normalized event type;
- explicit call/result relationships, including orphan, ambiguity, and duplicate IDs;
- source-span accuracy;
- capabilities eligible for negative predicates.

### Rules

1. `STRICT` means standards-compliant nested JSON and schema validation, not permissive decoder
   acceptance.
2. `REPAIRED` requires an allowlisted deterministic rule and RepairMap.
3. `PARTIAL` identifies recovered complete units and missing ranges separately.
4. A repair that touches role, tool name, call ID, path, or content semantics is not automatically
   accepted and requires review.
5. A valid explicit call/result ID must be paired exactly. Temporal proximity is not enough.
6. A capability listed under `negative_predicate_capabilities` must be `COMPLETE`.
7. Heuristic tool signals from malformed request text are annotation leads, not parser truth.

## 7. Safety Annotation

### Truth Units

Each subject receives:

- file-version stage;
- origin class;
- derivation taint labels;
- content-risk labels;
- disposition;
- non-waivable status;
- evidence references.

The record also states known output contamination, answer leakage, package/provenance equality, package
disposition, and blockers.

### File Timeline

```text
observed pre-mutation V0 -> potentially usable observed input ranges
write/edit -> V1 generated content
later read -> V1 observation, still generated and tainted
unknown possible mutation -> unknown-version boundary, fail closed
```

Copy, rename, archive, extraction, translation, summary, or re-read never resets origin or taint.

### Hard Rules

- `FINAL_OUTPUT_DERIVED`, `PRIVATE_REFERENCE_DERIVED`, and `GRADER_RULE_DERIVED` cannot receive an allow
  disposition.
- `ANSWER_BEARING`, `HIDDEN_PASS_CONDITION`, and package `SECRET` content are non-waivable.
- `UNKNOWN_DERIVATION` and `UNSCANNABLE_CONTENT` fail closed.
- A clean keyword leakage scan does not clear taint or authorize release.
- Package inventory and ProvenanceManifest must be exact sets before release.
- High-risk conclusions cannot rely only on approximate spans.

Answer leakage is evaluated relative to the visible prompt, evaluation claim, rubric, evaluator,
reference policy, tool policy, lineage, and measured capability. It is not a context-free keyword
classification.

## 8. Label Annotation

### Truth Units

- exact LabelSpec and version;
- expected `MATCH`, `NO_MATCH`, or `ABSTAIN`;
- positive, negative, and semantic evidence sets;
- structured capability completeness;
- semantic residual status;
- confidence, rule version, model profile, prompt version, and review status.

### Rules

1. Evaluate deterministic structured predicates first.
2. Call semantic judgment only for a declared residual condition.
3. `MATCH` requires positive evidence.
4. `NO_MATCH` requires complete structured capability and explicit complete negative evidence.
5. Conflicting structured and semantic evidence routes to review; confidence cannot hide conflict.
6. Model uncertainty, missing spans, or an underspecified LabelSpec produces `ABSTAIN`.
7. Manifest `verified_traits` may locate evidence but do not replace an adjudicated LabelDecision.

## 9. Task Annotation

### Truth Units

- TaskEpisode references;
- visible prompt;
- task intent and evaluation claim;
- required capabilities and allowed tools;
- forbidden outputs;
- attachment dependencies;
- requirement lineage and conflict status;
- Selection Firewall checks;
- task usability.

### Rules

1. Every requirement and attachment dependency binds evidence.
2. Direct facts outrank explicit requirements, which outrank inferred dependencies.
3. An inference cannot override a contradictory direct observation.
4. A `USABLE` task has no unresolved requirement conflict and passes every Selection Firewall check.
5. The visible prompt excludes final answers, completed deliverables, private references, grader rules,
   hidden selection signals, and accidental trajectory-specific solution steps.
6. Hidden error signatures used to select a trace are not exposed unless independently required by the
   visible task.
7. Material ambiguity about the evaluated capability yields `NEEDS_REVIEW` or `BLOCKED`.

## 10. Attachment Annotation

Attachment mode is decided per artifact:

| Mode | Minimum basis |
|---|---|
| `TRACE_RICH` | Direct safe pre-mutation content or high-coverage structure and content evidence |
| `SKELETON_GUIDED` | Approved path/type/structure skeleton with incomplete safe content |
| `PROMPT_ONLY` | Requirement evidence but no usable trace or skeleton content |
| `BLOCKED` | Critical uncertainty or capability gap prevents a valid input artifact |

A task aggregates to `MIXED` when its artifacts use multiple non-blocked modes.

For every artifact annotate:

- logical path and media type;
- criticality;
- path/type/structure evidence quality;
- untainted content coverage and pre-mutation evidence;
- provenance confidence and truncation;
- blocking uncertainties;
- expected non-answer-bearing structure;
- expected outcome and evidence.

Hard rules:

- critical artifacts cannot be omitted as optional;
- critical blocking uncertainty cannot silently produce `BUILD`;
- a buildable package has a complete required artifact set and contains input-state files only;
- `silent_degradation_allowed` is always false;
- generated final content cannot be reused as an input artifact;
- external trace results are leads until independently fetched, scanned, and recorded as SourceEvidence.

## 11. Release Annotation

Release truth binds:

- QuerySpec;
- EnvironmentSpec;
- RubricSet;
- EvaluatorSpec;
- ReferencePolicy;
- ToolPolicy;
- ProvenanceManifest;
- QualityReport;
- package manifest;
- review roles and records;
- channel, registry, profile, and production attestation.

`release_allowed=true` requires:

1. exact package/provenance set equality;
2. no open P0/P1 or non-waivable finding;
3. current subject, package, policy, quality, and review hashes;
4. required role and separation-of-duty satisfaction;
5. channel-appropriate registry isolation;
6. a current ProductionReadinessAttestation for `PRODUCTION`.

`CANARY` and `INTERNAL_REVIEW` records never authorize production publication. ReleaseDecision remains
the sole release-state authority.

## 12. R0 Gold Procedure

The first five complete EvaluationItem gold records should include:

- one strict, large, pre/post-mutation trace;
- one malformed request or response requiring partial/repaired adjudication;
- one search/fetch trace;
- one PowerShell error trace with call/result identity evidence;
- one no-read or truncation candidate requiring explicit abstain or resolution.

Recommended initial IDs from the frozen manifest are `LH_005`, `LH_011`, `LH_015`, `LH_058`, and
`LH_067`. R0-09 may substitute an ID only when the replacement preserves or improves this coverage and
records the reason.

For each complete gold item:

1. verify source SHA-256 against the frozen manifest;
2. create parser truth before any negative label;
3. create safety truth and file-version decisions before task or attachment truth;
4. annotate representative LabelDecisions;
5. annotate task and attachment truth from safe projections;
6. assemble Query, Environment, Rubric, Evaluator, Reference, Tool, Provenance, and Quality contracts;
7. annotate release truth;
8. adjudicate and validate the complete JSONL bundle;
9. record immutable review references and hashes.

## 13. Versioning

After R0-08 closes, `v1` schemas and this guide are frozen. A semantic change creates `v2`.

Corrections to annotation truth:

1. create a new annotation record and subject version;
2. reference prior annotation IDs in `supersedes`;
3. attach new review records;
4. invalidate dependent gold and derived objects;
5. retain all prior records.

Changes to the Spec, ADRs, policy, schema, guide, source hash, subject hash, or evidence projection
require an explicit `NO_EFFECT`, `REVALIDATE`, or `INVALIDATE` compatibility decision. Missing or
ambiguous compatibility defaults to `INVALIDATE`.
