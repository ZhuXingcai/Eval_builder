# Eval Dataset Factory Human Review And Immutable Editing Policy

- Status: Approved
- Version: 1.0.0
- Date: 2026-07-19
- Beads: `env_mock_agent-ujc.1.6`
- Spec: `specs/002-eval-dataset-factory/spec.md`
- Depends on: ADR 0010

## Purpose

Human review resolves semantic uncertainty and approves versioned objects. It does not rewrite trace
facts, erase findings, clear non-waivable leakage, or substitute an opinion for deterministic hard
gates.

This policy defines review assignment, claims, immutable conclusions, editing, invalidation,
separation of duties, dual review, reopening, and release binding.

## Reviewable Subjects

The initial subject types are:

```text
LABEL_DECISION
TASK_DRAFT
PROVENANCE_DECISION
ARTIFACT
RUBRIC_SET
EVALUATOR_SPEC
REFERENCE_POLICY
TOOL_POLICY
QUALITY_REPORT
BATCH_QUALITY_REPORT
EVALUATION_ITEM
RELEASE_DECISION
POLICY_EXCEPTION
```

Raw traces, TraceIR events, SourceSpans, RepairMaps, immutable StageResults, and prior review records are
not editable review subjects. Corrections create new parser, policy, object, or decision versions.

## Roles

Roles are assigned by policy and subject:

```text
LABEL_REVIEWER
TASK_REVIEWER
PROVENANCE_REVIEWER
ARTIFACT_REVIEWER
RUBRIC_REVIEWER
EVALUATOR_REVIEWER
PRIVACY_REVIEWER
SAFETY_EXCEPTION_REVIEWER
ITEM_REVIEWER
RELEASE_AUTHORITY
AUDITOR
SUBJECT_EDITOR
REVIEW_ADMIN
APPEAL_TRIAGER
```

A role grant records principal, scope, policy version, issuer, issue time, expiry, and conflict set.
Expired or scope-mismatched roles cannot claim or decide a case.

## Separation Of Duties

1. A subject author cannot provide its only required approval.
2. An attachment worker cannot approve its own artifact or QualityReport.
3. A semantic reviewer cannot approve the finding it authored as resolved without a matching
   deterministic recheck or another assigned reviewer.
4. A content reviewer cannot publish the item.
5. ReleaseAuthority cannot edit the release subject.
6. Two-person review requires two distinct human identities; aliases, service accounts, delegated
   sessions, and the same supervising identity do not satisfy separation.
7. The same identity cannot occupy two roles in the same required quorum.
8. Auditor access does not grant edit, approval, exception, or release authority.
9. ReviewAdmin may force-release a lease but cannot decide that case unless independently assigned a
   non-conflicting reviewer role.
10. AppealTriager for an accepted appeal must differ from the subject author, original deciding
    reviewer, and ReleaseAuthority for the affected decision.

Conflict-of-interest checks run at claim time and decision time.

## Mutable Case And Immutable Record

HumanReviewCase is mutable work coordination:

```text
case_id
subject_type
subject_ref
subject_version
subject_hash
required_role
review_policy_id
review_policy_version
projection_ref
projection_hash
required_check_refs[]
quorum_group_id
quorum_size
conflict_set[]
idempotency_key
status
assignee
claim_token_hash
lock_version
lease_expires_at
created_at
updated_at
```

HumanReviewRecord is an immutable conclusion:

```text
review_record_id
case_id
subject_type
subject_ref
subject_version
subject_hash
reviewer_id
reviewer_role
review_policy_id
review_policy_version
conclusion
reason
finding_refs[]
diff_ref
diff_hash
supersedes[]
decided_at
record_sha256
```

Records are append-only. Corrections or appeals create a new case and record linked through
`supersedes`; they never mutate the prior record.

Later invalidation is represented by a separate immutable ReviewRecordInvalidation:

```text
invalidation_id
review_record_id
review_record_hash
reason_code
source_ref
source_hash
policy_version
invalidated_at
invalidation_sha256
```

A rebuildable validity projection joins records with invalidation events. It never writes a reverse
reference into the original record.

## Case States

```text
PENDING
CLAIMED
APPROVED
REJECTED
CHANGES_REQUESTED
EDITED
ABSTAINED
SUPERSEDED
```

Legal transitions:

| From | Action | To | Required guard |
|---|---|---|---|
| `PENDING` | claim | `CLAIMED` | CAS status + lock version; valid role, no conflict, active policy |
| `CLAIMED` | heartbeat | `CLAIMED` | CAS status + lock version + token + unexpired lease |
| `CLAIMED` | release | `PENDING` | CAS status + lock version + token + unexpired lease |
| `CLAIMED` | admin force release | `PENDING` | ReviewAdmin, CAS status + lock version, reason and audit event |
| `CLAIMED` | expire | `PENDING` | reaper CAS proves token hash, lease expiry, and lock version |
| `CLAIMED` | approve | `APPROVED` | CAS token/lease/lock; current subject/projection hashes, checks, no blocker |
| `CLAIMED` | reject | `REJECTED` | CAS token/lease/lock; current subject hash and reason/finding |
| `CLAIMED` | request changes | `CHANGES_REQUESTED` | CAS token/lease/lock; target, reason, findings, revalidation scope |
| `CLAIMED` | abstain | `ABSTAINED` | CAS token/lease/lock; reason; routing creates a new case if required |
| `CHANGES_REQUESTED` | submit edit | `EDITED` | atomic new subject version and immutable diff |
| `EDITED` | queue revalidation | `SUPERSEDED` | new subject gets a new `PENDING` case |
| `APPROVED` | invalidate | `SUPERSEDED` | subject, dependency, or policy hash changed |
| `REJECTED` | reopen | `SUPERSEDED` | authorized appeal reason; new case created |
| non-terminal | supersede | `SUPERSEDED` | newer subject/case or administrative cancellation |

All unspecified transitions fail.

`APPROVED`, `REJECTED`, `ABSTAINED`, and `SUPERSEDED` are terminal for that case. Reopen never returns
the same case to pending.

## Claim And Lease

- Claim uses compare-and-swap over status and lock version.
- At most one active claim exists per subject version, required role, and review policy.
- Claim token is random, stored only as a hash, scoped to the case, and expires with the lease.
- Heartbeat extends an unexpired lease up to policy maximum.
- Reaper atomically releases expired claims.
- Late, duplicate, or mismatched-token decisions are rejected.
- Administrative force release requires a separate role, reason, and audit event.
- Claim, record creation, case transition, invalidation event, and outbox message use database
  transactions appropriate to their operation.

## Review Inputs

Every role receives the recursive deny-by-default projection defined by ADR 0010. A case freezes:

- subject hash and version;
- projection policy and hash;
- relevant finding and evidence references;
- governing Spec, ADR, and policy versions;
- required checks and quorum.

Reviewers cannot browse raw stores or expand their own projection. Requests for more evidence create an
audited case action and a new approved projection; silent data expansion is forbidden.

Semantic review rounds remain separate:

1. coverage and solvability;
2. realism and consistency;
3. leakage and executability.

Each uses a separate role, StageRun, clean context, and typed findings. Later rounds see prior typed
findings, resolutions, and repaired outputs, not hidden reasoning or build transcripts.

## Conclusions

Allowed conclusions are policy-specific subsets of:

```text
APPROVE
REJECT
REQUEST_CHANGES
ABSTAIN
CONFIRM_EXCEPTION
DENY_EXCEPTION
```

`ABSTAIN` never counts toward approval quorum. It returns the case to routing with a reason and may
require a different qualified reviewer.

An approval is evidence that an assigned human accepted the exact subject under the exact policy. It is
not a waiver for deterministic P0/P1, non-waivable leakage, secret, integrity, or stale-hash failures.

## Editing

Reviewers do not edit source objects directly.

Editability is closed by subject type:

| Subject | Editable | Required editor scope | Allowed edit | Required follow-up |
|---|---:|---|---|---|
| LabelDecision | yes | `SUBJECT_EDITOR:LABEL_DECISION` | adjudicated replacement | selection and all descendants |
| TaskDraft | yes | `SUBJECT_EDITOR:TASK_DRAFT` | typed prompt/requirement patch | prompt gates and all descendants |
| ProvenanceDecision | no direct edit | none | new classifier/policy decision or exception case | provenance and all descendants |
| Artifact | yes | `SUBJECT_EDITOR:ARTIFACT` | full replacement or typed format edit | format/safety QA and descendants |
| RubricSet | yes | `SUBJECT_EDITOR:RUBRIC_SET` | typed criterion patch | reachability/evaluator/QA/release |
| EvaluatorSpec | yes | `SUBJECT_EDITOR:EVALUATOR_SPEC` | typed evaluator patch | evaluator/reference/QA/release |
| ReferencePolicy | yes | `SUBJECT_EDITOR:REFERENCE_POLICY` | typed policy patch | reference/leakage/QA/release |
| ToolPolicy | yes | `SUBJECT_EDITOR:TOOL_POLICY` | typed allow/deny patch | reachability/executability/QA/release |
| QualityReport | no | none | recompute from immutable inputs | new report version |
| BatchQualityReport | no | none | recompute from immutable inputs | new report version |
| EvaluationItem | no | none | reassemble from approved child versions | new item version |
| ReleaseDecision | no | none | append a new decision-chain record | transactional Item projection |
| PolicyException | no after submission | none | supersede with a new proposal | full exception quorum |

An edit command contains:

```text
base_subject_ref
base_subject_hash
edit_type
typed_patch_or_replacement
reason
finding_refs[]
editor_identity
editor_role
idempotency_key
```

The edit transaction:

1. verifies base hash and editor authority;
2. validates the typed patch against subject schema and edit policy;
3. creates a new immutable subject version and content hash;
4. writes an immutable semantic and structural diff;
5. marks the old case `EDITED` then `SUPERSEDED`;
6. emits invalidation events for all dependents;
7. queues deterministic validation and required semantic review for the new version;
8. creates new review cases bound to the new hash.

If any required write fails, no new version or partial state is committed.

Raw trace, TraceIR, SourceSpan, RepairMap, prior subject versions, prior diffs, findings, review records,
and ReleaseDecisions are never overwritten.

## Invalidation Matrix

| Changed object | Invalidated dependents |
|---|---|
| LabelDecision or SelectionContext | selection, TaskDraft and all downstream objects |
| TaskDraft or QuerySpec | attachment plan/results, rubric reachability, ToolPolicy projection, all QA and release |
| Artifact or EnvironmentSpec | affected validators, current and later semantic rounds, QualityReport and release |
| RubricSet | reachability, prompt/attachment leakage checks, EvaluatorSpec binding, QualityReport and release |
| EvaluatorSpec | evaluator validation, ReferencePolicy binding, QualityReport and release |
| ReferencePolicy or private reference | evaluator checks, leakage fingerprints, QualityReport and release |
| ToolPolicy | reachability, executability, contestant projection, QualityReport and release |
| provenance/taint/view policy | affected evidence projections and every descendant object |
| lineage edge or derivation closure | affected provenance, views, artifacts, QA and release |
| SourceEvidence content, retrieval, usage, or approval | every claim, artifact and QA result using it |
| redaction policy or redacted projection | every model/view consumer and descendant object |
| projection policy or projection hash | every stage result, review and descendant using that view |
| ReviewRecordInvalidation | every ReviewQuorum using that record; dependent approval, Item and release |
| ReviewQuorum or PolicyException aggregate | every dependent exception, approval, Item and release |
| review policy version | pending/claimed cases, quorum aggregates and release decisions in policy scope |
| governing Spec or ADR | objects named by its mandatory compatibility-impact declaration |
| QualityReport | Item approval and release |
| BatchQualityReport | release of every referenced item |
| release profile/channel/attestation | package and ReleaseDecision |

Invalidation is transitive, hash-addressed, and append-only. It does not mark old results fixed or
delete them.

Every Spec, ADR, or review-policy change includes a machine-readable compatibility impact for each
governed object type and prior policy version:

```text
NO_EFFECT
REVALIDATE
INVALIDATE
```

`NO_EFFECT` requires evidence and an approved compatibility record. `REVALIDATE` keeps immutable
objects but invalidates current approvals until checks rerun. `INVALIDATE` supersedes pending cases and
invalidates records, quorums, Items, and releases in scope. A missing, ambiguous, or unapproved impact
declaration defaults to `INVALIDATE` for the entire governed scope.

## Exceptions And Dual Review

Two distinct qualified reviewers are required for:

- a non-answer-bearing provenance ambiguity exception;
- a Private Reference Policy change;
- a privacy exception allowed by the organization policy;
- release-policy deviation explicitly marked dual-reviewable.

Each reviewer sees the same subject hash, policy, evidence projection, and exception proposal. The
exception is effective only when both immutable records approve and the aggregate exception hash
matches.

A ReviewQuorum is an immutable-input aggregate with a mutable coordination head:

```text
quorum_group_id
subject_ref
subject_hash
policy_version
required_roles[]
required_distinct_identities
case_ids[]
record_hashes[]
status=PENDING | SATISFIED | DENIED | STALE
lock_version
aggregate_hash
```

Each required role receives an independent case. The aggregator uses CAS over group status/lock version,
revalidates every record, identity, role, projection, subject and policy hash, and writes the aggregate
record plus outbox event atomically. A required rejection denies the quorum unless the governing policy
explicitly routes a new proposal; changed inputs mark it stale.

The following are not reviewable exceptions:

- requested answer or final-deliverable leakage;
- original-agent final-output derivation;
- private-reference answer leakage;
- grader rule or hidden pass condition disclosure;
- secret in package or model disclosure;
- package/provenance inventory mismatch;
- stale or mismatched subject hash.

Those conditions require removal, safe reconstruction, policy-compliant redaction where applicable, or
rejection.

## Release Binding

ReleaseDecision references:

- exact release-subject hash and item version;
- required HumanReviewRecord hashes and reviewer roles;
- review policy versions;
- item and batch QualityReport hashes;
- package inventory and hash;
- channel, registry, export profile, and production attestation where required.

Approval and publication both re-evaluate record validity, separation of duties, subject hashes,
invalidations, and hard gates in one transaction with the Item status projection and outbox event.

Content review and release authorization are separate actions by separate roles.

ReleaseDecision is an immutable chain and the sole release-state authority:

```text
CANDIDATE
NEEDS_REVIEW
REJECTED
APPROVED
RELEASED
REVOKED
```

| Current chain head | Action | New decision | Guard |
|---|---|---|---|
| `CANDIDATE` | submit review | `NEEDS_REVIEW` | complete child contracts and prechecks |
| `NEEDS_REVIEW` | approve | `APPROVED` | current subject/package hashes, quorum, all hard gates |
| `NEEDS_REVIEW` | reject | `REJECTED` | current subject hash and blocking reason |
| `APPROVED` | publish canary/internal | `RELEASED` | non-production channel, registry isolation, package hash |
| `APPROVED` | publish production | `RELEASED` | production channel plus current attestation |
| `APPROVED` or `RELEASED` | invalidate/revoke | `REVOKED` | dependency, policy, quality, package or attestation invalidation |

Every action appends a record with chain ID, previous decision hash, subject/package/channel hashes,
idempotency key, actor and timestamp. It uses CAS against the current chain head. Unlisted actions fail.
ReleaseDecision append, Item status projection, integrity hashes and outbox event commit in one
transaction. Item status is rebuilt from the current valid chain head and cannot be written directly by
another service.

## Appeals And Reopening

AppealRecord is immutable and AppealCase has:

```text
appeal_case_id
target_record_hash
current_subject_ref
current_subject_hash
current_policy_version
appellant_identity
reason
new_evidence_refs[]
status=PENDING | CLAIMED | DENIED | ACCEPTED | SUPERSEDED
assignee
claim_token_hash
lock_version
lease_expires_at
idempotency_key
```

Appeal claim and decision use the same token/lease/lock CAS rules. `APPEAL_TRIAGER` may:

- deny the appeal with an immutable record;
- supersede the old case and open a new case;
- request a policy change through its own approval path.

Legal AppealCase transitions are:

| From | Action | To | Guard |
|---|---|---|---|
| `PENDING` | claim | `CLAIMED` | triager role/conflict check; status + lock CAS |
| `CLAIMED` | heartbeat | `CLAIMED` | token + unexpired lease + lock CAS |
| `CLAIMED` | release | `PENDING` | token + unexpired lease + lock CAS |
| `CLAIMED` | expire/admin force release | `PENDING` | expiry or ReviewAdmin reason; lock CAS |
| `CLAIMED` | deny | `DENIED` | current target, token/lease/lock CAS, immutable reason |
| `CLAIMED` | accept | `ACCEPTED` | current target, no conflict, token/lease/lock CAS |
| `PENDING` or `CLAIMED` | supersede | `SUPERSEDED` | target/policy no longer current or duplicate appeal |

`DENIED`, `ACCEPTED`, and `SUPERSEDED` are terminal. Unlisted transitions fail.

An accepted appeal is forbidden when the triager conflicts with the subject author, original deciding
reviewer, or affected ReleaseAuthority. The appeal remains `ACCEPTED` and atomically creates a new
ReviewCase bound to the current subject hash, current policy version and newly approved evidence
projection. If the target subject is no longer current before decision, the appeal becomes
`SUPERSEDED` or a new appeal must target the current version.

Appeal does not pause safety invalidation or restore release. A revoked item remains revoked until a new
subject and ReleaseDecision pass every current gate. Duplicate idempotency keys return the committed
appeal result.

## Review Adapters

JSONL and CLI are the first authoritative review interfaces. An optional Argilla adapter may mirror
assignments and conclusions, but:

- Job Store remains authoritative;
- adapter states map to policy states and cannot add transitions;
- adapter edits use the same typed edit command;
- adapter identity maps to an authenticated principal;
- sync conflicts fail closed and create an audit event.

## Audit And Metrics

Audit records include case, subject/version/hash, action, actor, role, lease, policy, evidence projection,
reason, findings, diff, invalidations, quorum, and timestamp. They exclude secret, raw trace, and private
reference text.

Operational metrics include queue age, claim duration, expiry rate, abstain rate, disagreement,
request-changes rate, invalidation rate, edit rate, override rate, and post-review defect rate. Metrics do
not expose subject content.

## Alternatives Considered

### In-Place Editing

Rejected because prior approvals and package hashes would become ambiguous.

### Unversioned Comments As Approval

Rejected because comments do not bind subject hash, role, policy, or conclusion.

### Automatic Approval After Human Edit

Rejected because edits invalidate prior validation and semantic review.

### One Reviewer For Content And Release

Rejected because it collapses authoring, exception, and publication authority.

### UI State As The Review Source Of Truth

Rejected because adapters can be unavailable, stale, or inconsistent; the Job Store owns review state.

## Validation

R4 and R7 must include:

- every legal and illegal case transition;
- concurrent claim, heartbeat, expiry, force-release, and late-token tests;
- role expiry, scope, conflict, author-self-review, and duplicate-identity tests;
- immutable record and chain reconstruction tests;
- typed edit success and transaction rollback tests;
- full invalidation-matrix tests;
- governance compatibility NO_EFFECT/REVALIDATE/INVALIDATE and missing-declaration tests;
- recompute-only report/item and append-only ReleaseDecision edit-boundary tests;
- ReviewRecordInvalidation append/rebuild tests;
- quorum aggregate CAS, mixed conclusion, stale input, and identity tests;
- new-version revalidation and no-auto-approval tests;
- dual-review quorum and non-waivable exception rejection tests;
- item/ReleaseDecision atomic projection tests;
- appeal/reopen and revoked-item tests;
- appeal role conflict, current-subject rebinding, lease, and idempotency tests;
- every legal/illegal AppealCase transition and accepted-appeal ReviewCase creation test;
- immutable ReleaseDecision chain, illegal transition, CAS, and Item projection rebuild tests;
- JSONL/CLI/Argilla state mapping and sync-conflict tests;
- content-free audit and metric tests.
