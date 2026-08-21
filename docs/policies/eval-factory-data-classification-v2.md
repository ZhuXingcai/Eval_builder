# Eval Dataset Factory Data Classification Overlay v2

- Status: Approved
- Version: 2.0.0
- Date: 2026-07-20
- Beads: `env_mock_agent-ujc.1.13`
- Governing ADR: ADR 0011
- Base policy: `eval-factory-data-classification-v1`

## Purpose

This overlay preserves the data classes, local trust admission, lifecycle, model-domain approval,
redaction, retention, and incident controls from v1. It replaces only v1 references to mandatory human
review roles and queues.

For active v2 execution:

- automated semantic reviewers are stage models with purpose-bound projections;
- the requesting user receives a bounded checkpoint projection only when UserApprovalPolicy enables
  that checkpoint;
- no independent human-review principal, queue, case, quorum, claim, lease, or appeal is created;
- disabled checkpoints disclose no additional data and create no request or decision;
- UserDecisionRecord is an audit record, not permission to widen data visibility.

## Disclosure Matrix

| Data class | Deterministic services | Internal semantic stage model | Internal evaluator model | Requesting user checkpoint | Contestant/export |
|---|---:|---:|---:|---:|---:|
| `PUBLIC_SOURCE` | allow | recorded SourceEvidence only | allow if required | bounded preview if enabled | only if item policy permits |
| `INTERNAL_PROJECT` | allow | allow by purpose | allow by purpose | bounded preview if enabled | deny unless projected |
| `RESTRICTED_TRACE_RAW` | ingestion/index only | deny | deny | deny | deny |
| `RESTRICTED_TRACE_SPAN` | allow | purpose, budget, and redaction | evaluator purpose only | deny; preview contains derived summaries and refs only | deny unless separately released |
| `RESTRICTED_EVAL_CONTROL` | control/evaluation only | exact stage projection | assigned evaluator subset | deny except explicit non-secret plan choices | deny |
| `PRIVATE_REFERENCE` | reference/evaluator only | deny | named evaluator only | deny | deny |
| `SENSITIVE_SECRET` | scanner/credential store only | deny | deny | deny | deny |
| `SENSITIVE_PII` | scanner/redaction only | approved redacted projection only | approved redacted projection only | approved redacted preview only | approved redacted projection only |

An allow entry never overrides provenance, taint, answer leakage, source license, or release policy.

## Checkpoint Projections

`LABEL_PLAN` exposes label intent, deterministic/semantic split, examples, abstain rules, cost path,
and blind spots. It does not expose complete raw traces.

`TASK_REWRITE_PLAN` exposes capability intent, rewrite style, fidelity, examples, noise handling, and
forbidden-content rules. It does not expose private references, grader rules, hidden labels, or the
original final answer.

`ENVIRONMENT_STRATEGY` exposes affected task classes, safe evidence sufficiency summaries, executable
alternatives, capability impact, and query packaging choices. It does not expose credentials.

`FINAL_DATASET_REVIEW` exposes configured prompt, item, or dataset projections plus quality summaries.
It does not expose raw traces or private evaluator material.

## Personal Data

The v1 minimization and privacy-approval requirements remain unchanged. Where v1 says a human review
queue or assigned human reviewer, v2 means only an enabled, authenticated requesting-user checkpoint
with an approved redacted projection. No user checkpoint is created solely to inspect personal data.

## Compatibility

V1 records remain historical. A v2 DatasetJobSpec binds this overlay through its active policy set.
Changing checkpoint scope, projection fields, or user identity creates a new policy or projection
version and invalidates affected requests and decisions.
