# Task Map: Evaluation Dataset Agent Harness

**Spec**: `specs/003-evaluation-agent-harness/spec.md`

**Status**: Active

## V1 Vertical Edition

| Stage | Trellis task | Outcome | Dependency |
|---|---|---|---|
| 0 | `08-17-harness-contract-runway-v1` | Frozen Harness/Team/Blueprint/static Pack contracts | none |
| 1 | `08-17-persistent-session-agent-loop-v1` | Persistent session, clarification, Gateway loop, recovery | Stage 0 |
| 2 | `08-17-capability-generic-pack-adapters-v1` | Ten EDF capability adapters, requirement bridge, and fixed Blueprint | Stage 1 mechanism gate |
| 3 | `08-17-collaborative-team-runtime-v1` | P2P Team Runtime and Graph Blackboard | Stage 2 |
| 4 | `08-17-factory-graph-closure-v1` | Default Factory graph and full vertical closure | Stage 3 |
| 5 | `08-17-conversation-first-agent-shell-v1` | HTTP/SSE and conversation-first Agent UI | Stage 4 |
| 6 | `08-17-delivery-evidence-v1` | Spreadsheet/ZIP delivery and evidence freeze | Stage 5 |

## Stage 1 Work Packages

Stage 1 is one bounded implementation task with four internal gates:

1. **Runtime contracts**
   - session state and message authority;
   - requirement interpretation/policy;
   - Gateway invocation journal and turn result;
   - event page and session projection.
2. **Session store**
   - additive SQLite schema;
   - atomic commands/messages/events/turns/idempotency;
   - bounded reads, reconcile, and explicit projection rebuild.
3. **Gateway Agent loop**
   - private intake/rendering/output;
   - route-before-invoke and PREPARED journal;
   - clarification/READY deterministic compilation;
   - committed-event streaming and closed failure handling.
4. **Verification**
   - fault/restart/replay/concurrency/property tests;
   - mechanism fixture labeling;
   - separately authorized live semantic gate.

Stage 1 mechanism completion permits Stage 2 planning and implementation.
`REAL_SEMANTIC` may remain pending when no usable provider API is available,
but must pass before V1 Stable evidence is frozen.

## Stage 2 Work Packages

1. **Capability runtime**
   - exact static composition/provider/consumer resolution;
   - Balanced Autonomy authorization;
   - canonical calls, results, and Artifact Envelopes.
2. **Generic Pack adapters**
   - explicit Pack-owned request bundles;
   - ten providers including Batch Quality;
   - direct-owner/runtime parity.
3. **Requirement bridge**
   - READY interpretation to existing Factory requirement/policy/run request;
   - no inferred source, budget, registry, output, or release authority.
4. **Fixed Blueprint**
   - exact ten-capability provider registry;
   - complete acyclic schema/data flow and pinned replay.

## Stage 4 Work Packages

1. **Graph authority and journal**
   - strict cross-store execution binding;
   - dedicated pre/post transition journal and Harness-session outbox;
   - async LangGraph SQLite cursor storage kept separate from business truth.
2. **Committed routing and continuation**
   - deterministic persisted `PlannerAssessmentV2`;
   - exact PlanReview successor reconciliation on one thread;
   - fail-closed PRE recovery for unknown owner outcomes.
3. **Team and Pack execution**
   - fixed ten-capability DAG through Team/Harness Runtime;
   - Blueprint baseline with non-widening Team successors;
   - taskless Coordinator and specialist `control` owner.
4. **Delivery and default product path**
   - execution-frontier `FINISH` routes into final review and immutable
     candidate delivery;
   - CLI/facade Graph bootstrap is required by default;
   - direct runtime remains explicit compatibility only.

## Stage 5 Work Packages

1. **Unified runtime protocol**
   - provider-neutral readiness, event, usage, tool, and failure contracts;
   - legacy runtime and Gateway adapters with private source evidence.
2. **Agent Shell event transport**
   - strict public session/message/event projections;
   - atomic runtime-event to SessionEvent persistence;
   - committed-sequence HTTP pagination and SSE reconnect.
3. **Source admission**
   - bounded `manifest.csv` plus JSONL admission;
   - private staging/CAS, immutable per-session authority, exact replay, and
     safe Artifact Envelope refs.
4. **Service composition**
   - owner-backed Team/Factory/Graph/PlanReview/workspace/delivery
     projections;
   - READY plus explicit admitted-source confirmation to Pack `1.2.0` Graph
     start;
   - replay-safe Team/Graph outbox reconciliation and restart recovery;
   - explicit full-Shell versus PlanReview-only `agent serve` composition.
5. **Conversation-first UI and browser evidence**
   - React session shell, composer, Activity/Team/workspaces;
   - CC-Haha-inspired warm-light workbench hierarchy, asymmetric
     user/Agent messages, inline Team status, and floating composer;
   - task-level primary navigation plus a centered/docked context composer
     with source directories, goal templates, Plan mode, slash commands,
     explicit extension status, and safe Agent-output quotation;
   - auditable soft-close session removal with confirmation, plus a
     text-only Composer surface and responsive lower toolbar for add/mode, one
     model-and-reasoning trigger, browser voice input, and send;
   - searchable built-in/custom model catalog, per-model discrete effort
     slider, strict non-secret custom model metadata, and per-session
     browser-local preferences that do not claim Host routing authority;
   - responsive, accessibility, reconnect, and privacy E2E.

Stage 5.4 mechanism evidence covers the full fixture conversation-to-review
flow, PlanReview-only compatibility, API restart/SSE reconnect, deep-link
reload, Delivery empty state, drawer/model/delete-dialog focus/Escape, reduced
motion, custom model reload/delete, and 1440/1024/768/375 no-overflow
rendering. This remains
`MECHANISM_FIXTURE`; real Provider semantics and Stage 6 packages are separate
gates.

## Authority

Beads owns active/closed task state and evidence. Trellis owns implementation
planning/context. This file records stable work-package intent only and is not
a parallel status tracker.

## V2 Platform Edition Gate

Dynamic Pack lifecycle, schema registry, registry-driven Blueprint compiler,
second Pack, certification SDK, and platform UI remain blocked until V1 Stage
6 freezes the static composition and semantic/recovery evidence.
