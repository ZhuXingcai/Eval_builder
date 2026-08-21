# Eval Factory User Approval Policy v2

```text
Policy ID: eval-factory-user-approval/v2
Status: Approved
Owner: Eval Dataset Factory product owner
Governing ADR: ADR 0011
Supersedes for active execution: eval-factory-human-review/v1
```

## Purpose

User approval steers dataset-production choices. It is not an independent-review system, a safety
override, or a replacement for deterministic and semantic quality validation.

## Policy Modes

| Mode | Enabled checkpoints |
|---|---|
| `NONE` | None; must be selected explicitly |
| `PLAN_GATES` | Label plan, task rewrite plan, relevant environment strategy |
| `FINAL_ONLY` | Optional final dataset review only |
| `PLAN_AND_FINAL` | All plan gates plus final dataset review |
| `CUSTOM` | Explicit configured subset |

Interactive first use defaults to `PLAN_GATES`. Batch automation may use another mode only when the
DatasetJobSpec records the explicit choice.

## Checkpoints

### `LABEL_PLAN`

Request contains the LabelSpec candidate, deterministic/semantic split, examples, abstain rules, cost
path, and blind spots.

Allowed decisions:

```text
ACCEPT
ADJUST
REJECT
REQUEST_MORE_EXAMPLES
```

### `TASK_REWRITE_PLAN`

Request contains target capability, rewrite style, fidelity, examples, noise-removal policy, and
forbidden-content rules.

Allowed decisions:

```text
ACCEPT
ADJUST
REJECT
REQUEST_MORE_EXAMPLES
```

### `ENVIRONMENT_STRATEGY`

Request contains affected task classes, evidence sufficiency, special account/environment
requirements, executable alternatives, and expected capability impact.

Each affected class or task receives one strategy:

```text
TRACE_FAITHFUL_MOCK
REWRITE_STANDARD_ENV
EXCLUDE_TASK
```

Query packaging receives one strategy:

```text
INCLUDE_QUERY_YAML
OMIT_QUERY_YAML
ASK_PER_TASK
```

### `FINAL_DATASET_REVIEW`

Request contains exported prompt/item/dataset references and hashes, quality summary, unresolved P2/P3
findings, and sample navigation. It does not include private references or raw traces.

Allowed decisions:

```text
ACCEPT
ADJUST
REJECT
DEFER
```

## Request And Decision Binding

Every UserApprovalRequest records:

- checkpoint;
- requesting user;
- exact subject and plan references/hashes;
- bounded preview references;
- available choices;
- affected downstream stages;
- policy version and idempotency key.

Every UserDecisionRecord records:

- request reference and hash;
- authenticated user;
- decision;
- selected options and typed adjustments;
- reason;
- resulting plan/object reference;
- invalidation scope;
- timestamp and record hash.

No claim, lease, quorum, reviewer role, independent identity, appeal, or dual approval is required.

## Execution Rules

1. The orchestrator pauses only for enabled checkpoints.
2. Disabled checkpoints create no synthetic approval record.
3. An enabled checkpoint requires a current matching UserDecisionRecord before affected work starts.
4. `REQUEST_MORE_EXAMPLES` creates a new request version, not an in-place mutation.
5. `ADJUST` creates a new plan/object version and invalidates named dependents.
6. `REJECT` stops or excludes the affected scope with a typed reason.
7. `DEFER` never counts as release acceptance.
8. Duplicate idempotency keys return the committed decision.
9. A stale request or subject hash is rejected.

## Environment Rules

### Trace-Faithful Mock

- use only approved, safe evidence;
- preserve measured capability and environment shape;
- mock inaccessible services without real credentials;
- block when critical semantics cannot be reproduced safely;
- never copy original final output into inputs.

### Standard-Environment Rewrite

- create a new TaskDraft, QuerySpec, EnvironmentSpec, RubricSet, EvaluatorSpec, ReferencePolicy, and
  ToolPolicy as affected;
- state the capability trade-off;
- remove special-account and special-environment assumptions;
- rerun prompt, reachability, attachment, quality, and release gates.

### Exclude Task

- preserve source and selection history;
- record exclusion reason;
- create no attachment package or released EvaluationItem.

## Non-Waivable Gates

User decisions cannot approve or downgrade:

- answer-bearing or final-output-derived package content;
- private-reference, grader, or hidden-condition leakage;
- secrets or disallowed PII;
- unknown/unscannable critical content;
- open P0/P1 findings;
- stale object or package hashes;
- package/provenance set mismatch;
- production release without a current attestation.

## Optional Final Review

The user may inspect:

- rewritten prompt samples before batch execution;
- all rewritten prompts after task authoring;
- selected EvaluationItems;
- the complete candidate dataset before release;
- no content at all.

The chosen scope is part of UserApprovalPolicy. Skipping optional inspection does not bypass automated
quality gates.

## Compatibility

This policy requires active v2 approval and release contracts. Legacy v1 HumanReview schemas and
queues remain historical and must not be scheduled by a v2 DatasetJobSpec.
