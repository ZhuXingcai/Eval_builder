# ADR 0011: Replace Mandatory Independent Human Review With User-Directed Approval Checkpoints

- Status: Accepted
- Date: 2026-07-20
- Beads: `env_mock_agent-ujc.1.13`
- Spec: `specs/002-eval-dataset-factory/spec.md`
- Supersedes in active architecture:
  `docs/policies/eval-factory-human-review-v1.md`

## Context

The original R0 design introduced an internal human-review subsystem with reviewer roles, independent
identities, claim leases, quorums, adjudication, appeals, and release-bound HumanReviewRecords. That
design treated human review as a mandatory production stage.

The product owner clarified that this is not the intended interaction model. Human judgment belongs to
the user-facing workflow and is requested only where the user wants to steer or inspect the dataset:

- accept or adjust the detailed labeling scheme and examples;
- accept or adjust task-rewrite style and examples;
- decide whether generated task prompts are included as `query.yaml`;
- decide how tasks with special accounts or environments are handled;
- optionally inspect rewritten prompts, individual items, or the completed dataset.

The system must not invent independent reviewer identities or turn optional user inspection into a
mandatory multi-person operational subsystem.

## Decision

Remove mandatory independent human review from the active factory architecture.

Introduce a versioned `UserApprovalPolicy` with user-directed checkpoints:

```text
LABEL_PLAN
TASK_REWRITE_PLAN
ENVIRONMENT_STRATEGY
FINAL_DATASET_REVIEW
```

The policy supports:

```text
NONE
PLAN_GATES
FINAL_ONLY
PLAN_AND_FINAL
CUSTOM
```

Interactive first use defaults to `PLAN_GATES`. `NONE` requires an explicit user choice and does not
disable deterministic or semantic quality gates.

The active privacy projection model uses
`docs/policies/eval-factory-data-classification-v2.md`; data-classification v1 remains the immutable
base for non-review controls.

### Label Plan

Before batch labeling when enabled, present:

- label intent and boundary;
- structured predicates and semantic residual;
- positive, negative, ambiguous, and abstain examples;
- expected cost and model-use path;
- known blind spots.

The user may accept, adjust, reject, or request more examples. Accepted changes create a new LabelSpec
version; they do not mutate the prior version.

### Task Rewrite Plan

Before batch task reconstruction when enabled, present:

- intended evaluation capability;
- rewrite style and fidelity;
- handling of operational trace noise;
- prompt examples;
- forbidden answer, grader, private-reference, and trajectory content.

The user may accept or adjust the plan and examples. A decision binds the exact plan hash.

### Environment Strategy

When selected tasks contain attachments, special accounts, proprietary services, or non-standard
environment requirements, present the affected task classes and require a strategy when the checkpoint
is enabled:

```text
TRACE_FAITHFUL_MOCK
REWRITE_STANDARD_ENV
EXCLUDE_TASK
```

Independently decide query packaging:

```text
INCLUDE_QUERY_YAML
OMIT_QUERY_YAML
ASK_PER_TASK
```

`TRACE_FAITHFUL_MOCK` reconstructs safe environment dependencies from approved evidence while
preserving the original capability. `REWRITE_STANDARD_ENV` changes the task and evaluation contracts
to remove special-account or special-environment dependence. `EXCLUDE_TASK` removes the item from the
candidate set. No strategy may fabricate credentials or bypass access controls.

### Final Dataset Review

Final prompt/item/dataset inspection is optional and user-configured. It can produce accepted,
adjusted, rejected, or deferred decisions for exact object hashes. It is not required for every run.

## Automated Quality Review

The three ordered semantic quality rounds remain:

1. coverage and solvability;
2. realism and consistency;
3. leakage and executability.

These are isolated reviewer agents and deterministic validators, not human reviewers. Their findings
are version-bound and become stale when inputs change.

## Safety And Release

User decisions cannot waive:

- answer or requested-deliverable leakage;
- original-agent final-output derivation;
- private-reference or grader leakage;
- hidden pass conditions;
- secrets in model disclosure or package content;
- package/ProvenanceManifest mismatch;
- open deterministic P0/P1 findings;
- stale hashes or missing production attestation.

ReleaseDecision remains the sole release-state authority. It binds deterministic quality evidence and
any user checkpoint decisions required by the selected UserApprovalPolicy. It does not require
independent reviewer identities, role quorums, claim leases, or appeals.

## State And Audit

`UserApprovalRequest` and `UserDecisionRecord` are immutable, hash-bound records used for pause/resume
and audit. They record one authenticated requesting user; they do not model reviewer independence.

The orchestrator may pause only at checkpoints enabled by policy. When a checkpoint is disabled, it
continues without manufacturing an approval record.

Plan adjustment creates new plan and downstream object versions and invalidates affected work. It
never overwrites raw trace, TraceIR, prior plans, prior user decisions, artifacts, or release decisions.

## Deprecated Active Components

The following v1 concepts remain historical artifacts but are not active v2 workflow components:

- HumanReviewCase;
- HumanReviewRecord as mandatory content approval;
- ReviewQuorum and dual-human approval;
- reviewer claim, lease, heartbeat, and appeal state machines;
- Argilla as an authoritative or required adapter.

The v1 policy and schemas remain immutable for audit. Active contracts use v2.

## Consequences

Positive:

- HITL matches actual user workflow and remains optional;
- the user controls labeling, rewrite, and environment trade-offs at useful decision points;
- no synthetic reviewer identity or fake quorum is needed;
- fewer operational states and less blocking infrastructure;
- deterministic safety remains non-waivable.

Negative:

- unattended jobs must explicitly choose an approval policy;
- user-plan changes can invalidate substantial downstream work;
- final content quality may vary when the user disables optional inspection;
- UI/CLI must summarize plans and examples clearly enough for informed decisions.

## Validation

The active design must test:

- each approval mode and checkpoint;
- no pause for disabled checkpoints;
- exact plan/subject hash binding;
- accept, adjust, reject, and defer behavior;
- invalidation after plan adjustment;
- query.yaml packaging choices;
- all three special-environment strategies;
- no credential fabrication or policy bypass;
- deterministic hard gates rejecting user override attempts;
- release with required checkpoints satisfied and failure when a required decision is stale.
