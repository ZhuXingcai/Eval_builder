# ADR 0002: Use LangGraph For The Control Plane

- Status: Accepted
- Date: 2026-07-18

## Context

The workflow includes long-running retrieval, artifact-level fan-out, retries, human inspection, three review
rounds, targeted repair, and resumability after process failure.

## Decision

Use LangGraph with a SQLite checkpointer for the local release. Put network calls, file writes, downloads, and
runtime executions in explicit side-effect nodes/tasks with idempotency keys.

## Consequences

- Runs resume by stable thread/run ID.
- Node boundaries become part of the compatibility contract.
- PostgreSQL or Temporal is deferred until multi-machine operation is required.
