# Eval Dataset Factory Annotation Guide v2

```text
Guide version: eval-factory-annotation-guide/v2
Schema version: eval-factory-annotation/v2
Status: Approved
Governing decision: ADR 0011
Beads: env_mock_agent-ujc.1.13
```

## Scope

This guide defines evidence-bound development reference annotations for parser, safety, label, task,
attachment, and release truth. It supersedes the v1 independent-submission and adjudication workflow
for active execution. The frozen v1 guide and schemas remain historical evidence.

Reference annotations support development canaries. They are not production statistical evidence,
automatic user approval, or a substitute for R8 frozen independent test sets.

## Lifecycle

Allowed record states are:

```text
DRAFT
REFERENCE_READY
ABSTAINED
SUPERSEDED
```

`REFERENCE_READY` means the record passed schema and deterministic cross-record validation and may be
used as a development reference. It does not mean a user accepted a label plan, rewrite plan,
environment strategy, item, or dataset.

The record identifies an `author_kind`:

```text
DETERMINISTIC_SERVICE
SEMANTIC_AGENT
USER
```

No independent reviewer identity, submission slot, adjudicator, quorum, claim, lease, or appeal is
part of the active annotation contract. Corrections create a new annotation ID and list prior IDs in
`supersedes`.

## Evidence Rules

1. Every annotation binds exact source, subject, guide, contract-manifest, policy, and content hashes.
2. Evidence references, not rationale text, support conclusions.
3. Missing ranges cannot prove negative facts. Negative conclusions require complete capability and
   explicit complete negative evidence.
4. Canary `annotation_candidates` are routing hints, not truth.
5. Approximate evidence cannot be the sole basis for safety or release conclusions.
6. Raw trace text, final output, secrets, private references, and grader content do not enter
   annotation JSONL.
7. Original-agent final-output derivation, answers, private references, grader rules, hidden pass
   conditions, secrets, unknown derivation, and unscannable critical content fail closed.
8. Package inventory and ProvenanceManifest must be exact sets before release.

The v1 domain payload definitions remain the base for parser, safety, label, task, and attachment
annotations. The v2 envelope and release payload are authoritative where semantics changed.

## User Checkpoints

User checkpoints steer product choices and remain separate from annotation truth:

```text
LABEL_PLAN
TASK_REWRITE_PLAN
ENVIRONMENT_STRATEGY
FINAL_DATASET_REVIEW
```

An annotation may reference bounded checkpoint previews. It never fabricates a UserDecisionRecord.
Disabled checkpoints have no request or decision. Enabled checkpoints bind exact request, plan,
subject, policy, and projection hashes.

User decisions cannot waive deterministic safety, privacy, provenance, P0/P1, integrity, stale-hash,
or production-attestation gates.

## Release Annotation

A v2 release annotation binds:

- all required EvaluationItem component hashes;
- package and provenance exact-set status;
- automated quality status and hard-gate findings;
- UserApprovalPolicy;
- required checkpoint set;
- typed checkpoint/request/UserDecisionRecord bindings for satisfied checkpoints;
- release channel, registry isolation, action, state, and production attestation.

`release_allowed=true` requires automated quality to pass, no open P0/P1 or non-waivable finding,
required and satisfied checkpoint sets to match, no synthetic decision references, package/provenance
equality, registry isolation, a valid action/state pair, and a production attestation for production.

## Validation

```bash
uv run python scripts/validate_eval_factory_annotations_v2.py \
  path/to/annotations.jsonl \
  --require-reference-ready
```

The validator checks schema closure, frozen canary and manifest hashes, evidence closure, capability
completeness, safety and leakage hard gates, task firewall, critical attachment handling, and v2
release semantics.
