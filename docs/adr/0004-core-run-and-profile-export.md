# ADR 0004: Use A Core Run Store And Profile Exports

- Status: Accepted
- Date: 2026-07-18

## Context

Generic tasks, cc-mock-env, and LH packages have different visible directory contracts. Process evidence must
not leak into evaluation packages.

## Decision

Store normalized tasks, evidence, staging files, validation, reviews, transcripts, and checkpoints in a common
Run Store. Export user-visible files through Generic, CC, or LH profiles.

## Consequences

- Internal state is stable across business profiles.
- LH exports can enforce the top-level three-item boundary.
- Profile compatibility is tested independently from generation.
