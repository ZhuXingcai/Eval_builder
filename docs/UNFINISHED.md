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

Status: Stage 5.0-5.4 mechanism complete.

Completed:

- provider-neutral runtime readiness/event/usage/failure protocol;
- atomic runtime-event to Harness session projection;
- strict session/message/event HTTP contracts;
- committed-sequence SSE replay, reconnect, heartbeat, and disconnect
  isolation;
- bounded `manifest.csv` plus exact `raw_traj_v1` JSONL admission;
- private server-owned staging/CAS, immutable SQLite admission authority,
  exact replay/recovery, and safe Artifact Envelope refs;
- unified owner-backed Harness/Source/Team/Factory/Graph/PlanReview/workspace/
  delivery projection;
- READY plus explicitly confirmed admitted source to fixed Pack `1.2.0`
  Graph start;
- replay-safe Team/Graph outbox reconciliation, Graph-bound new-turn
  rejection, and restart-equivalent aggregate projection;
- full fixture Agent Host with explicit authority/config paths and explicit
  `--plan-review-only` compatibility mode;
- combined Agent Shell/PlanReview OpenAPI;
- conversation-first session rail, transcript, fixed composer, and explicit
  source selection/confirmation;
- inline clarification, permission, PlanReview, verification, and delivery
  states;
- Activity and Team inspectors plus contextual Trace, Task, Attachment,
  Rubric, Grading, Quality, PlanReview, and Delivery workspaces;
- URL-addressable session/workspace/member state;
- heartbeat-aware SSE reconnect with HTTP projection refresh;
- 1440/1024/768/375 responsive behavior, modal drawer focus/Escape handling,
  reduced motion, and Full Host Browser E2E.

Remaining work:

- session rename and explicit cancel/interrupt controls;
- separately authorized real-provider semantic acceptance;
- Stage 6 delivery package generation and download actions.

The React application now opens as the Agent Shell. PlanReview remains
available as a contextual workspace and as the explicit compatibility Host.

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

## Production

Production release remains blocked. Existing release, attestation, and
publication fixtures are mechanism evidence only and do not authorize a
production registry write.

## Current Verified Baseline

```text
Full-suite run: 2868 passed plus one corrected Stage 5.4 Browser assertion
Final Full Agent Shell Browser E2E: 7 passed
PlanReview compatibility Browser E2E: 22 passed
Focused Shell/Source/SSE/Graph/Host backend: 31 passed
Web Vitest: 14 passed
Node Pi: 4 passed
Ruff lint and mypy: passed
Full format: one pre-existing tests/integration/test_context.py drift
Generated contracts: 73/415/7/5
Wheel build, isolated install, cold imports, and CLI help: passed
Continuity validation: passed
```

These results prove deterministic implementation and recovery behavior, not
real semantic quality or production readiness.
