# ADR 0005: Use Spec Kit And Beads For Project Continuity

- Status: Accepted
- Date: 2026-07-18

## Context

Long development cannot depend on a model context window, transcript format, or compaction summary.

## Decision

Spec Kit stores intent, technical plans, data contracts, and quality gates. Beads stores dependency-aware
active work and persistent insights. ADRs store decisions. `PROJECT_STATE.md` and typed handoffs are generated
from Git and Beads state.

## Consequences

- Fresh sessions can recover without replaying chat.
- Specifications and tasks have separate ownership but explicit references.
- Markdown task checklists are non-authoritative.
