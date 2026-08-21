# Implementation Plan: Evaluation Dataset Agent Harness

**Branch**: `main`

**Date**: 2026-08-17

**Spec**: `specs/003-evaluation-agent-harness/spec.md`

**Status**: Active V1 execution baseline

## Architecture

```text
Agent Shell / future transports
  -> HarnessSessionService
      -> HarnessSessionStore
      -> GatewayRequirementAgentLoop
          -> AIGateway
          -> GatewayRecordStore
          -> FactoryPrivateObjectStore
      -> server-owned session projection

later:
  -> fixed EvaluationBlueprintV1
  -> Team Runtime
  -> FactoryControlGraph
  -> generic-agent-trace Pack capabilities
```

## Store Ownership

| Store | Owns | Does not own |
|---|---|---|
| HarnessSessionStore | session commands, messages, events, turns, interpretation/policy refs, invocation journal, projection | provider bodies, Factory runs, Team tasks |
| GatewayRecordStore | route, invocation, receipt, result | conversation transcript |
| FactoryPrivateObjectStore | prompt/rendering/provider output bytes | public/session state |
| FactoryControlStore | existing Factory run/plan/Agent authority | conversation event log |
| JobStore | existing DatasetJob/Item/Stage authority | Harness session |

## Stage Sequence

### Stage 0 - Contract Runway

Completed under ADR 0013:

- session/capability/artifact/permission/projection/composition contracts;
- Team and Blueprint contracts;
- static `generic-agent-trace-pack`.

### Stage 1 - Persistent Session And Agent Loop

- add runtime contracts for session state, messages, interpretation, policy,
  Gateway journal, turn result, projection, and bounded event page;
- implement SQLite session authority and projection rebuild;
- implement governed Gateway interpretation and clarification loop;
- expose typed Python service and committed-event async stream;
- prove restart, replay, reconciliation, unknown-outcome blocking, private
  boundaries, and evidence-class separation.

### Stage 2 - Capability And Pack Adapters

- adapt existing EDF services behind one provider-neutral capability runtime
  and ten static Pack providers, including Batch Quality;
- bridge READY Harness requirement authority into existing Factory
  requirement, policy, and run-request contracts;
- compile fixed EvaluationBlueprintV1;
- prove standalone/Agent parity.

### Stage 3 - Collaborative Team Runtime

- persist roster, task DAG, mailbox, blackboard, subscriptions, conflicts,
  checkpoints, and context projections;
- run Requirement/Trace/Task/Quality peer-rework slice.

### Stage 4 - Factory Graph Closure

- make FactoryControlGraph the default product execution path;
- bind Harness, Factory, Team, Blueprint, Blackboard, and one recoverable
  graph thread through a dedicated pre/post transition journal;
- route only from committed PlannerAssessment authority;
- resume PlanReview on the same run, Team incarnation, composition, and
  thread;
- execute the fixed ten-capability Pack workflow and close immutable candidate
  delivery after the execution-frontier `FINISH` signal;
- retain direct `FactoryDatasetRuntime.advance()` only behind explicit
  compatibility mode.

### Stage 5 - Agent Shell

- add HTTP/SSE and conversation-first React UI;
- retain PlanReview as a workspace.

### Stage 6 - Delivery And Evidence

- add spreadsheet/ZIP delivery and complete semantic/recovery evidence freeze.

## Stage 1 Data Flow

```text
PostMessageCommand
  -> commit command + user message + user event
  -> build private RequirementIntake
  -> persist Gateway route
  -> commit PREPARED invocation journal
  -> invoke Gateway
  -> read strict semantic output from private store
  -> deterministic interpretation/policy compilation
  -> commit Gateway closure + assistant chunks/message + result events
  -> rebuild/advance session projection + idempotency
```

On restart:

```text
COMMITTED journal -> replay committed turn
PREPARED + Gateway result -> reconcile, no provider call
PREPARED + no Gateway result -> UNKNOWN_OUTCOME, block retry
projection drift -> explicit rebuild only
```

## Testing Strategy

- contract: strict/frozen/versions/ref shapes and denied fields;
- unit: store transitions, exact replay, projection rebuild, pagination;
- property: event order/idempotency/rebuild determinism;
- fault injection: before/after route, journal, Gateway result, session commit;
- integration: real Router/PromptRegistry/GatewayRecordStore/session loop with
  only the provider boundary faked and labeled `MECHANISM_FIXTURE`;
- semantic: separately authorized real provider only;
- regression: Stage 0, AI Gateway, Graph, PlanReview, generated, static, full
  Python, Node, privacy, production block, continuity.

## Compatibility And Rollback

- Stage 1 is additive and does not mount a default CLI/Web route.
- Existing `evalfactory agent run`, PlanReview API, and web workbench remain
  unchanged.
- Rollback removes the unmounted session service/store modules and tables.
- No migration touches FactoryControlStore, JobStore, or GatewayRecordStore.
- Stage 2 may start after Stage 1 mechanism gates pass. Semantic completion
  remains separately gated and may be deferred when no usable provider API is
  available; it must pass before V1 Stable evidence, not before adapter
  development.
