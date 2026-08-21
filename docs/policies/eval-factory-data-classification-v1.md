# Eval Dataset Factory Data Classification And Model Disclosure Policy

- Status: Approved
- Version: 1.0.0
- Date: 2026-07-19
- Beads: `env_mock_agent-ujc.1.5`
- Spec: `specs/002-eval-dataset-factory/spec.md`
- Policy owner: Eval Dataset Factory Data Owner
- Initial accountable project owner: `project-owner`

## Scope

This project-local policy governs data handled by the first trusted local CLI/batch release. It does not
replace company-wide security, privacy, legal, retention, or model-approval policy. Where an
organization policy is stricter, the stricter policy wins and the affected job is blocked until the
conflict is resolved.

The policy covers raw traces, TraceIR, evidence spans, labels, prompts, attachments, evaluation
contracts, private references, logs, model calls, human review, and exported packages.

## Local Trust Admission

The first release accepts only `TRUSTED_MONITORED_LOCAL` jobs. Every real-data job requires a signed or
otherwise independently verifiable LocalTrustAdmission:

```text
admission_id
manifest_hash
source_roots[]
operator_principal
approved_processing_classes[]
monitoring_mode
isolated_home
isolated_workdir
tool_allowlist_version
path_guard_version
timeout_policy_version
data_lifecycle_profile_ref
incident_route_ref
approval_authority
expires_at
```

Admission enforces a read-only approved manifest, isolated home/temp/work directories, path guards,
explicit tool allowlists, timeouts, process-group termination, and monitored execution. Permission
bypass modes are forbidden. Missing, expired, or mismatched admission returns `BLOCKED_POLICY`.

R0 manifest-only analysis may use a versioned no-model profile that reads allowlisted sources without
copying raw content and emits content-free hashes and classifications to a restricted output directory.
Any persistent TraceIR, raw-derived content, semantic model call, or unattended execution requires the
full lifecycle and incident configuration below.

Untrusted or unattended inputs require a later container workspace and are blocked in the first release.

## Default

Data is denied to models and stage workspaces unless an authenticated principal, declared purpose,
approved policy version, and explicit projection allow it.

The first release permits model processing only in approved internal model domains. There is no external
model fallback. Public-source retrieval over approved network tools is not permission to send content to
an external model.

No model receives a complete raw trace.

## Project Processing Classes

These classes are implementation controls, not official corporate labels.

### `PUBLIC_SOURCE`

Publicly retrievable material with recorded source, retrieval time, usage basis, and content hash.

Examples:

- public standards;
- public product documentation;
- public research and datasets allowed by source policy.

### `INTERNAL_PROJECT`

Factory metadata that does not contain raw trace content, secrets, personal data, private references, or
hidden evaluation controls.

Examples:

- schema names and versions;
- non-sensitive metrics;
- Beads IDs;
- package hashes;
- public release-profile names.

### `RESTRICTED_TRACE_RAW`

Complete or substantially complete raw trace data, outer records, nested request/response/extra strings,
canonical TraceIR content stores, and raw workspace observations.

### `RESTRICTED_TRACE_SPAN`

Minimal, purpose-bound trace or file spans selected through TraceQueryService after policy filtering and
redaction.

### `RESTRICTED_EVAL_CONTROL`

Hidden selection signals, private rubric criteria, grader prompts, thresholds, hidden assertions,
private evaluator configuration, and non-public release controls.

### `PRIVATE_REFERENCE`

Private reference answers, expected outputs, reference files, or trace-behavior references available
only to an explicitly authorized evaluator or human-review role.

### `SENSITIVE_SECRET`

Credentials and authentication material, including API keys, access tokens, cookies, passwords,
private keys, session secrets, and equivalent values.

### `SENSITIVE_PII`

Personal or uniquely identifying data governed by the selected privacy profile. This includes direct
identifiers and configured indirect identifiers. The scanner and policy versions define the exact set.

## Classification And Visibility Composition

Processing classes are additive. An object retains every applicable class; assigning `PUBLIC_SOURCE` or
`INTERNAL_PROJECT` cannot remove a restricted or sensitive class. Access resolution evaluates all
classes and uses the most restrictive applicable result. Deny overrides allow.

Visibility is a separate, versioned decision:

```text
PRIVATE_STORE
PRIVILEGED_AUDIT
STAGE_PROJECTION
EVALUATOR_PROJECTION
CONTESTANT_VISIBLE
PUBLIC_RELEASE
```

`CONTESTANT_VISIBLE` applies only to a new, independently hashed projection that passed all content,
provenance, policy, and release gates. It is never a processing class and cannot downgrade source data.

## Disclosure Matrix

| Data class | Deterministic services | Internal semantic stage model | Internal evaluator model | Human reviewer | Contestant/export |
|---|---:|---:|---:|---:|---:|
| `PUBLIC_SOURCE` | allow | allow through recorded SourceEvidence | allow if evaluator needs it | allow | only if item policy permits |
| `INTERNAL_PROJECT` | allow | allow by purpose | allow by purpose | allow | deny unless projected |
| `RESTRICTED_TRACE_RAW` | ingestion/index only | deny | deny | privileged audit only | deny |
| `RESTRICTED_TRACE_SPAN` | allow | allow by principal, purpose, budget, and redaction | allow by evaluator purpose | allow by assigned case | deny unless separately released |
| `RESTRICTED_EVAL_CONTROL` | control/evaluation services | deny direct access; allow only independent approved SelectionContext projection | allow assigned subset | assigned role only | deny |
| `PRIVATE_REFERENCE` | reference store/evaluator only | deny | allow only to named evaluator principal | assigned reference reviewer only | deny |
| `SENSITIVE_SECRET` | scanner and isolated credential store only | deny | deny | masked evidence only | deny |
| `SENSITIVE_PII` | scanner/redaction service | deny unless an approved redacted projection removes governed identifiers | deny unless evaluator policy explicitly allows a redacted projection | assigned privacy reviewer | deny unless approved redacted projection |

An allow entry never overrides provenance, taint, answer-leakage, source-license, or release policy.

## Approved Model Domains

Model profiles are resolved to a versioned domain record:

```text
model_profile
provider
endpoint
domain=INTERNAL_APPROVED | INTERNAL_EVALUATOR_APPROVED
data_classes[]
purposes[]
retention_mode
training_use
region
approval_ref
expires_at
```

`approval_ref` resolves to an independently verifiable ModelDomainApproval:

```text
approval_id
approval_authority
organization_policy_id
organization_policy_version
provider
endpoint
region
data_classes[]
purposes[]
training_use
retention_mode
evidence_hashes[]
issued_at
expires_at
signature_or_registry_proof
```

Requirements:

1. `INTERNAL_APPROVED` may receive `PUBLIC_SOURCE`, `INTERNAL_PROJECT`, and authorized
   `RESTRICTED_TRACE_SPAN`.
2. `INTERNAL_EVALUATOR_APPROVED` may additionally receive the exact `PRIVATE_REFERENCE` objects named
   in its EvaluatorSpec.
3. An absent, expired, or mismatched approval blocks the model call.
4. Model provider or endpoint changes create a new profile version and invalidate affected evidence.
5. Training use must be disabled or explicitly approved under organization policy.
6. Retention mode must satisfy the organization policy selected by the job.
7. External endpoints are denied in the first release even for public-only prompts.
8. The approval authority is independent of the project runtime and project data owner.
9. The project owner cannot self-sign an organization model approval.
10. Missing authority configuration, policy reference, scope, proof, or successful verification returns
    `BLOCKED_POLICY`.

Runtime availability or authentication does not grant data approval.

## TraceQueryService Disclosure

Stage models receive trace data only through TraceQueryService using the trusted-principal rules in ADR
0009.

Before disclosure, the service verifies:

- StageRun principal and model profile;
- declared stage and purpose;
- trace and EvidenceBundle scope;
- allowed data and provenance classes;
- taint and quarantine exclusion;
- redaction policy and result;
- character, event, and call budgets;
- destination model-domain approval and expiry.

The request can only reduce scope. It cannot self-assert a higher privacy profile or request raw-store
credentials.

Every response records:

```text
stage_run_id
principal_id
model_profile_version
purpose
source_span_ids[]
redacted_projection_hash
data_classes[]
characters
events
policy_versions[]
timestamp
```

Logs contain IDs, counts, decisions, and hashes, not returned span text.

## Logs, Telemetry, And Diagnostics

All sinks use a versioned structured-field allowlist. This includes application logs, audit logs,
runtime stdout/stderr, tool errors, HTTP diagnostics, SDK debug output, exception stacks, telemetry,
profiles, and crash handling.

Allowed fields are limited to identifiers, hashes, enum reason codes, counts, sizes, durations, policy
versions, and approved non-content metadata. The following are forbidden from normal sinks:

- prompt or response text;
- returned spans;
- tool arguments and results;
- file content or extracted text;
- raw request/response bodies;
- secrets, PII, private references, hidden controls, and redaction maps.

SDK/HTTP body logging and crash dumps are disabled for restricted processes. Raw stdout/stderr,
exceptions, or tool failures that may contain content first enter a restricted quarantine sink; a
deterministic sanitizer emits the normal diagnostic record. Sanitization failure emits only a generic
reason code and restricted event reference.

Third-party telemetry must satisfy the same destination approval and field allowlist. Unknown fields
are rejected. Tests cover every configured sink, not only TraceQuery and model-call logs.

## Redaction

Redaction is deterministic and versioned.

- The raw subject and redacted projection have separate content hashes.
- Redaction creates a derivation edge and does not modify the raw subject.
- Replacement markers contain no original value.
- A redaction map is stored only in the restricted audit store.
- Redaction validators check that governed identifiers do not remain in visible bytes, metadata, file
  names, comments, formulas, hidden content, or nested containers.
- Failed, partial, unsupported, or unscannable redaction blocks disclosure.
- Redaction does not clear answer, final-output, private-reference, or grader-derived restrictions.

## Secrets And Credentials

- Credentials come from environment variables or an isolated credential store.
- Secrets are never written to project files, Prompt artifacts, TraceIR projections, model prompts,
  review queues, or exported packages.
- Secret scanner findings record type, subject hash, and redacted location, not the secret.
- A model or tool error containing a credential is quarantined and redacted before diagnostics leave the
  restricted store.
- Credential-bearing jobs cannot be approved by a content reviewer; they require removal and rerun.

## Personal Data

The first release is data minimization by default:

- direct identifiers are removed or consistently pseudonymized before semantic processing;
- account, user, device, contact, and internal identity fields are denied unless the purpose explicitly
  requires an approved redacted projection;
- free text is scanned as well as structured metadata;
- human review queues expose only the minimum subject and purpose;
- exported packages must pass PII scanning for files, names, metadata, and nested content.

If a task genuinely requires personal data to measure its evaluation claim, the job requires a separate
privacy approval and profile version. Until that exists, it is `BLOCKED_POLICY`.

## Private Evaluation Material

Private references, hidden rubrics, grader prompts, and selection signals are stored separately from
producer and contestant data.

- Attachment producers cannot resolve their references.
- Task authors receive only the SelectionContext projection.
- Reviewers receive only the view assigned to their review round.
- Evaluator principals receive only objects bound in EvaluatorSpec.
- Release manifests store hashes and private references, not private content in the contestant package.
- Model-call logs never contain private reference text.

## Public Retrieval

Approved Search/Fetch tools may retrieve public material under the network and source policy.

Retrieved material:

- is treated as untrusted data;
- receives source and content hashes;
- is scanned before model or artifact use;
- cannot install tools, load MCP/plugin configuration, or change system instructions;
- cannot be treated as the original input solely because it appeared in the original trace.

## Storage And Access

- Raw, quarantine, private-reference, and redaction-map stores use restrictive local permissions and are
  not mounted into stage workspaces.
- Stage services receive opaque references and TraceQueryService capability tokens.
- Contestant workspaces contain release-approved projections only.
- CAS deduplication never broadens authorization; authorization is checked on each reference resolution.
- Temporary files inherit the source data class and are deleted according to the mandatory
  DataLifecycleProfile.
- Backup, export, and diagnostic paths obey the same class and view rules.

## Audit

Audit events are immutable and include:

- access decision and reason;
- actor/principal and role;
- purpose and case/job/stage;
- source and projection hashes;
- model or tool destination;
- policy/profile versions;
- volume and timestamp;
- override or break-glass reference.

Audit events exclude raw content, secrets, and private reference text.

Break-glass raw access requires a privileged auditor, case reason, expiration, and separate event. It
cannot grant release or write access.

## Lifecycle, Retention, And Deletion

Organization-approved retention values must be configured before production. This policy does not invent
corporate retention durations.

Every real-data job requires a versioned DataLifecycleProfile:

```text
profile_id
profile_version
applicable_processing_classes[]
restricted_storage_roots[]
temporary_root
directory_permissions
file_permissions
cleanup_on_success
cleanup_on_failure
cleanup_on_cancel
startup_scavenger
retention_approval_refs[]
backup_policy
deletion_authority
incident_route_ref
expires_at
```

The profile requires restrictive roots and permissions, cleanup for every terminal outcome, and a
startup scavenger for abandoned workspaces. Cleanup and scavenging emit content-free audit events.

The system must also support:

- retention class and expiry on every stored object;
- legal or investigation hold reference where applicable;
- deletion of derived CAS objects when no authorized reference remains;
- immutable deletion audit events without retaining deleted content;
- re-indexing or invalidation when source data is removed;
- independent retention for raw, redacted, audit, review, and released objects.

Missing lifecycle or retention approval blocks all real-trace persistence and model jobs. Only
synthetic data, or an approved R0 manifest-only profile that persists no raw-derived content, may run
without organization retention values. Missing configuration always blocks production attestation.

## Incident Routing

Every real-data admission and lifecycle profile references a verified IncidentRouteRef:

```text
incident_route_id
owner_role
primary_contact_ref
escalation_contact_ref
organization_channel_ref
runbook_ref
issued_at
expires_at
verification_proof
```

The project does not assume a particular organization channel. Missing, expired, or unverifiable
incident routing returns `BLOCKED_POLICY`.

## Incident And Revocation

On suspected unauthorized disclosure:

1. stop affected jobs and model calls;
2. revoke capability tokens and model profile;
3. quarantine affected projections and packages;
4. revoke associated ReleaseDecisions;
5. preserve content-free audit evidence;
6. notify the policy owner and verified IncidentRouteRef contacts;
7. identify all descendants through lineage and invalidate them;
8. resume only under a new approved policy/profile version.

## Policy Ownership

The `Eval Dataset Factory Data Owner` role owns:

- processing-class definitions;
- project use of the independently approved model-domain allowlist and expiry;
- privacy/redaction profiles;
- retention-class requirements;
- disclosure exceptions that organization policy permits;
- incident and revocation coordination.

The initial project owner recorded in Beads is accountable for obtaining organization approvals; it
cannot sign those approvals as the independent authority and does not gain permission to waive answer
leakage, secret handling, or organization policy.

## Validation

R1-R8 must include:

- model-domain allow/deny and expiry tests;
- raw-trace and direct-store denial tests for all stage identities;
- trace-query scope, budget, and audit tests;
- structured and free-text secret/PII fixtures;
- nested document metadata and hidden-content redaction tests;
- private-reference producer/contestant denial tests;
- content-free log tests;
- stdout/stderr, exception, HTTP, SDK, telemetry, and crash-sink allowlist tests;
- CAS authorization and temporary-file inheritance tests;
- LocalTrustAdmission isolation, timeout, path, tool, and bypass-denial tests;
- DataLifecycleProfile success/failure/cancel cleanup and startup-scavenger tests;
- ModelDomainApproval authority, signature/registry, scope, and self-sign rejection tests;
- IncidentRouteRef missing, expiry, and verification blocker tests;
- break-glass expiry and no-release-authority tests;
- retention-missing production-attestation blocker;
- incident lineage invalidation and release revocation tests.
