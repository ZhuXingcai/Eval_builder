# Unfinished Work

This document distinguishes implemented mechanism from unverified or planned
product behavior. Do not infer completion from the presence of contracts,
fixtures, mock providers, or UI placeholders.

## Blocking Gate

### Real Semantic Provider

Status:

```text
REAL_SEMANTIC_PENDING
PROVIDER_AUTHENTICATION_FAILED
```

The configured Provider exposes model metadata but rejects Messages and key
status requests with HTTP 401. The latest authorized gate used only synthetic
requirements and produced zero model tokens and no model output.

Required before a V1 Stable claim:

1. configure a valid Messages API credential locally;
2. rerun the two-case requirement interpretation gate under a fresh run root;
3. obtain one `CLARIFICATION_REQUIRED` and one `READY` result;
4. preserve exact provider/model/prompt/schema/usage authority;
5. execute the frozen task/rubric/grading/dataset usefulness benchmark;
6. bind passing semantic evidence into the V1 evidence freeze.

Never commit credentials or copy them into issues, logs, fixtures, or
documentation.

## Stage 5: Conversation-First Agent Shell

Status: planned, not implemented.

Remaining work:

- unified Agent Host over Harness, Team, Factory Graph, PlanReview, and
  delivery authorities;
- persistent session navigation and rename;
- conversation transcript and fixed composer;
- bounded `manifest.csv` plus JSONL source upload;
- typed command API and sequence-based SSE reconnect;
- inline clarification, permission, review, failure, and delivery cards;
- Activity and Team workbenches;
- contextual Trace, Task, Attachment, Rubric, Grading, Quality, PlanReview,
  and Delivery workspaces;
- desktop/tablet/mobile accessibility and Browser E2E.

The current React application is a PlanReview/Graph workbench, not the final
Agent Shell.

## Stage 6: Delivery And Evidence

Status: planned, not implemented.

Remaining work:

- deterministic `dataset.csv`;
- deterministic `dataset.xlsx`;
- bounded, replay-stable `attachments.zip`;
- delivery summary and immutable delivery-package manifest;
- manifest-bound download API and Delivery workspace;
- frozen V1 support matrix and known-limits document;
- semantic, recovery, fault, concurrency, API, frontend, Browser, privacy, and
  production-blocked evidence aggregation.

The current Stage 4 output contains authoritative candidate JSON/JSONL and an
immutable candidate manifest. It is not yet the final user-facing package.

## V2 Platform Edition

Status: design only.

Blocked until V1 Stable:

- dynamic Pack installation, upgrade, rollback, and removal;
- registry-driven Blueprint composition;
- cross-Pack schema and dependency resolution;
- Pack certification SDK and Marketplace;
- a materially different second first-party Pack;
- cross-domain delivery and workspace contributions.

## Dependency Audit Follow-Up

Status: remediation pending.

The current lockfiles reproduce the tested Node/Web builds, but `npm audit`
reports unresolved advisories:

- `web/eval_factory_console`: 1 high severity advisory;
- `node/pi_bridge`: 2 moderate and 2 high severity advisories.

These counts do not by themselves establish exploitability in this product,
but each advisory must be triaged and resolved or explicitly accepted before
a stable production release.

## Production

Production release remains blocked. Existing release, attestation, and
publication fixtures are mechanism evidence only and do not authorize a
production registry write.

## Current Verified Baseline

Fully provisioned local workspace:

```text
Python product suite: 2787 passed
Stage 0-3 focused suite: 133 passed
Node/Web/Browser gates: passed
Ruff/format/mypy: passed
Generated contracts and wheel imports: passed
Continuity validation: passed
```

Sanitized public snapshot:

```text
Python product suite: 2750 passed, 39 skipped
```

Public-snapshot skips identify resources that are intentionally not published:
the private 91-trace corpus, private CC/LH adapter fixtures, and local
Beads/continuity state.

These results prove deterministic implementation and recovery behavior, not
real semantic quality or production readiness.
