# Eval Dataset Factory Contract Compatibility Policy v1

```text
Policy ID: eval-factory-contract-compatibility/v1
Status: Approved
Owner: Eval Dataset Factory contract owner
Beads: env_mock_agent-ujc.1.10
```

## Scope

This policy governs:

- every Pydantic contract under `eval_factory.contracts`;
- `AttachmentReconstructionRequest` and `AttachmentReconstructionResult` under
  `env_mock_agent.facade`;
- generated JSON Schemas and their manifest;
- persisted objects, StageResults, HumanReviewRecords, EvaluationItems, and ReleaseDecisions that bind
  those contracts.

It does not permit R1-R8 behavioral implementation to change a frozen contract implicitly.

## Version Identity

Every top-level cross-stage object has a literal schema version:

```text
<owner>/<contract-name>/v<major>
```

Examples:

```text
eval-factory/trace-envelope/v1
eval-factory/evaluation-item/v1
env-mock-agent/attachment-reconstruction-request/v1
```

The version identifies validation, field meaning, enums, invariants, canonical serialization, and
visibility semantics. A consumer rejects unknown versions before processing the object.

## Frozen v1 Rule

After the v1 contract manifest is approved, the following are immutable:

- required and optional fields;
- field types, constraints, defaults, and aliases;
- enum members and their meaning;
- cross-field validators;
- canonical serialization and hash behavior;
- security, projection, taint, review, and release semantics.

Even an optional field addition changes the recursive closed schema and requires a new contract
version. Documentation-only clarification may remain in v1 only when it does not change any reasonable
producer or consumer behavior and has an approved `NO_EFFECT` declaration.

## Compatibility Declaration

Every change creates a `CompatibilityDeclaration` with:

- changed contract;
- prior and new versions;
- every affected object type;
- exactly one impact;
- migration or invalidation procedure where required;
- compatibility evidence;
- immutable approving HumanReviewRecord references.

Impacts:

| Impact | Meaning | Required action |
|---|---|---|
| `NO_EFFECT` | No producer, consumer, projection, hash, validation, review, or release meaning changes | Evidence plus approved compatibility record |
| `REVALIDATE` | Existing bytes remain parseable, but a policy, projection, or decision must rerun | New validation/review results; stale approvals invalidated |
| `INVALIDATE` | Existing object meaning or structure is incompatible | New object version and transitive invalidation/migration |

A missing, ambiguous, or unapproved declaration defaults to `INVALIDATE` for the complete governed
scope.

## Change Classification

The following always require a new major contract version and default to `INVALIDATE`:

- remove, rename, or change a field;
- add a required or optional field;
- add, remove, or reinterpret an enum member;
- relax a fail-closed safety, privacy, review, or release gate;
- change hash canonicalization;
- change object identity or reference semantics;
- permit a previously forbidden principal view;
- make an immutable object mutable;
- change `ReleaseDecision` authority or EvaluationItem composition.

The following require a new version and at least `REVALIDATE`:

- change a default, regex, numeric range, or cross-field validator;
- change a projection policy or evidence completeness rule;
- change attachment mode selection;
- change quality thresholds or review quorum.

## Migration

Migration is explicit and directional:

```text
source bytes + source schema version
  -> named migration implementation and version
  -> target bytes + target schema version
  -> target validation
  -> immutable migration record
```

Requirements:

1. Never overwrite source bytes or the prior object.
2. Record source and target hashes.
3. Reject migration when the source version is unknown.
4. Run safety, projection, quality, review, and release invalidation named by the declaration.
5. Preserve prior references for audit.
6. Do not auto-approve a migrated object.
7. Release remains revoked or stale until a new ReleaseDecision passes current gates.

No generic “best effort” migration or permissive extra-field parsing is allowed.

## Serialization And Hashing

Factory contracts use:

- UTF-8 JSON;
- Pydantic JSON-mode values;
- lexicographically sorted object keys;
- compact separators;
- explicit `null` for present optional fields;
- no non-finite numbers.

`canonical_sha256()` hashes those exact bytes. ObjectRef hashes bind the referenced serialized object,
not a database row or local path.

Generated JSON Schemas are projections of the Pydantic source contracts. The frozen manifest binds:

- every contract source file;
- every generated schema file;
- this policy;
- schema and generator versions.

Source, schema, and manifest drift fails contract validation.

## Producer And Consumer Rules

- Producers emit one declared version only.
- Consumers validate the version and full closed schema before reading fields.
- Unknown fields fail; they are not ignored.
- Cross-stage state uses `ObjectRef` and content hashes, not shared mutable objects.
- StageResult, HumanReviewRecord, QualityReport, EvaluationItem, and ReleaseDecision are append-only.
- A stale reference or hash mismatch fails closed.
- `env_mock_agent` never imports `eval_factory`.
- `eval_factory` may import only `env_mock_agent.facade` across the bounded-context boundary.

## Release Rules

A release action revalidates:

- every component schema version and hash;
- contract manifest version;
- compatibility declarations and migrations;
- current quality and review records;
- package and ProvenanceManifest exact-set equality;
- channel/profile and production attestation.

An active production attestation covers an exact contract manifest. A new contract version invalidates
the attestation unless an approved declaration explicitly proves otherwise.

## Required Tests

Every frozen contract set must pass:

- serialization round-trip;
- deterministic canonical hash;
- extra-field rejection;
- unknown-version rejection;
- generated-schema drift check;
- representative cross-field gate tests;
- stale reference and hash mismatch tests;
- import-boundary tests;
- built-wheel import check for both packages;
- explicit migration tests before any v2 consumer is enabled.
