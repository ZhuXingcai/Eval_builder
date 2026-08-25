# ADR 0017: Add A Unified Runtime Observation Protocol

- Status: Accepted
- Date: 2026-08-23
- Spec: `specs/003-evaluation-agent-harness/spec.md`
- Trellis: `.trellis/tasks/08-17-conversation-first-agent-shell-v1/`
- Beads: `env_mock_agent-ujc.17`

## Context

The repository has two valid but separate execution boundaries:

- `AIGateway` governs structured model calls and persists final receipts and
  results;
- `AgentRuntime` streams Claude/Pi lifecycle, tool, usage, and terminal events
  for open-ended attachment work.

The first boundary is final-result oriented. The second carries provider-native
payloads, paths, tool names, and external session handles that cannot be sent
directly to the Stage 5 API or browser. Runtime probes also established
installation or protocol availability while their messages implied credential
readiness.

## Decision

Add a provider-owned `runtime-protocol/stage5-0-v1` facade with:

- explicit readiness, credential, feature, sandbox, event, usage, and failure
  taxonomies;
- deterministic content-addressed normalized runtime events;
- private source-event references instead of raw runtime payloads;
- normalized tool families and hashed tool-call correlation;
- exact adapters for legacy `AgentRuntime` streams and Gateway
  receipt/result closures.

`available` in the legacy runtime contract continues to mean that the runtime
protocol can start. The unified readiness projection reports
`READY`, `DEGRADED`, or `BLOCKED`; unknown credentials and missing sandbox
enforcement are visible degradation reasons.

The protocol does not move persistence or business authority. Raw legacy
runtime events remain private provider evidence. Gateway records, Harness
sessions, Factory/Team stores, Graph journals, and domain artifacts retain
their existing owners.

## Compatibility

- Existing `AgentRuntime.run/resume/cancel` signatures remain unchanged.
- Existing R5 routing may continue to select a protocol-available runtime.
- `RuntimeCapabilities` gains additive readiness metadata with conservative
  defaults.
- The Pi bridge keeps its legacy `available` field and adds explicit protocol,
  credential, stream, progress, and sandbox facts.
- No Provider call, database migration, Graph transition, or frontend behavior
  is introduced.

## Follow-Up

Stage 5.1 will persist normalized runtime observations into the Harness event
stream and expose sequence-based SSE. Later slices add context compaction,
tool execution policy, and enforced sandbox providers. They must consume this
protocol rather than provider-native payloads.
