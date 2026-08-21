# Eval Factory Contract Compatibility Policy v2

## Status

- Policy: `eval-factory-contract-compatibility/v2`
- Status: Frozen
- Governing decision: ADR 0011
- Beads: `env_mock_agent-ujc.1.13`

## Contract Set

The active v2 contract set is an overlay on the immutable v1 base. The v1 manifest and its 73 schemas
remain byte-for-byte historical evidence. The v2 manifest binds the exact v1 manifest hash and adds
only the contracts changed by the user-directed approval architecture.

The overlay supersedes these contracts for v2 jobs:

- `DatasetJobSpec` v1 with `DatasetJobSpecV2`;
- `ReleaseDecision` v1 with `ReleaseDecisionV2`;
- `EvaluationItem` v1 with `EvaluationItemV2`.

It adds `UserApprovalPolicy`, plan and environment previews, `UserApprovalRequest`, and
`UserDecisionRecord`. All other v1 contracts remain the active base definitions unless a later overlay
explicitly supersedes them.

## Selection Rule

A job declares exactly one root job schema version. A v2 job MUST use the v2 overlay manifest and MUST
NOT schedule v1 human-review records, quorums, adjudication queues, claim leases, or appeals.

A v1 persisted object remains readable against the frozen v1 manifest. It is not silently interpreted
as a v2 object. Moving a v1 candidate into a v2 run requires an explicit migration that:

1. preserves every v1 object and hash;
2. creates a v2 UserApprovalPolicy;
3. creates a new DatasetJobSpecV2 or v2 release chain;
4. drops no deterministic safety or provenance evidence;
5. creates no synthetic UserDecisionRecord;
6. revalidates every affected object and release subject.

## Compatibility Classes

- **Base-compatible**: a v1 contract remains active without a v2 replacement.
- **Overlay-required**: a v2 replacement is mandatory for v2 jobs.
- **Historical-only**: a v1 object remains auditable but cannot be scheduled by v2.
- **Invalidating**: a change requires new dependent object versions and directed revalidation.

`HumanReviewRecord`, `ReviewRecordInvalidation`, and `ReviewQuorum` are historical-only. They are not
approval evidence for a v2 release.

## Hash And Version Rules

- Every schema is closed and uses a literal `schema_version`.
- v2 canonicalization recursively sorts unordered containers before hashing, so hashes remain stable
  across process hash seeds.
- Every cross-stage reference binds object type, ID, version, and canonical SHA-256.
- A stale plan, subject, policy, projection, package, or quality hash fails closed.
- An adjustment creates a new object version and an explicit invalidation scope.
- The overlay manifest records source-file, schema, policy, and v1-base-manifest hashes.
- Generated schemas and manifests are checked for drift in CI.

## User Decision Semantics

Disabled checkpoints emit no request and no decision. Enabled checkpoints require an exact
`UserApprovalRequest` and, before affected work proceeds, a matching `UserDecisionRecord`.

User decisions may select or adjust product choices. They cannot waive deterministic leakage, secret,
privacy, provenance, P0/P1, package-integrity, stale-hash, or production-attestation gates.

## Release Rules

`ReleaseDecisionV2` is the sole release-state authority for v2 items. Approval or publication requires:

- passing automated quality gates;
- zero open P0, P1, or non-waivable blockers;
- every checkpoint required by the bound UserApprovalPolicy satisfied by a current decision;
- matching release-subject and package hashes;
- a current ProductionReadinessAttestation for production.

Canary and internal-review decisions do not imply production compatibility or readiness.
