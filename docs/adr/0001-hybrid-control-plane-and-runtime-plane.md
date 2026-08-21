# ADR 0001: Separate Control And Runtime Planes

- Status: Accepted
- Date: 2026-07-18

## Context

Attachment generation needs durable workflow state and also needs mature coding-agent execution. Treating one
agent transcript as both concerns makes recovery, comparison, and validation unreliable.

## Decision

LangGraph owns the control plane. Claude Agent SDK, Claude Code CLI, Pi, and deterministic providers are
runtime-plane implementations behind typed contracts.

## Consequences

- Runtime sessions can be replaced without migrating business state.
- Every tool and model event must be normalized.
- The control plane adds integration work but prevents vendor lock-in.
