# ADR 0003: Route Provider, Claude, Then Pi

- Status: Accepted
- Date: 2026-07-18

## Context

No single model or harness is best at deterministic Office rendering, natural text, code construction, web
research, and format validation.

## Decision

Default routing is deterministic provider, Claude Agent SDK, Claude Code CLI, Pi, then
`BLOCKED_CAPABILITY`. Runtime choice and fallback reasons are recorded in the Run Store.

## Consequences

- Claude Code's mature behavior is preserved.
- Pi supplies provider-neutral and highly customizable execution.
- The same-model benchmark separates harness effects from model effects.
