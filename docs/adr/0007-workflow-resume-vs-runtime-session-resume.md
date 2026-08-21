# ADR 0007: Separate Workflow Resume From Runtime Session Resume

- Status: Accepted
- Date: 2026-07-18

## Context

The product must recover a run after the control process exits. Individual agent harnesses may also
support continuing their own conversation sessions. These are different guarantees:

- Workflow resume restores durable business state and skips completed side effects.
- Runtime session resume continues a vendor conversation from a persisted session identifier or file.

Treating a warning or a fresh retry as runtime resume makes capability reporting and benchmark results
misleading.

## Decision

LangGraph checkpoints and idempotent side-effect records provide mandatory workflow resume by run ID.

Each runtime reports `supports_resume=true` only when it can persist a `RuntimeSessionRef` and continue
that session from a new process through a `RuntimeResumeRequest`. Runtimes without that behavior report
the capability as unavailable; the control plane may still retry their node idempotently.

Runtime transcripts and session files remain execution evidence. They do not own task, evidence,
validation, review, or package state.

## Consequences

- Interrupted runs can recover even if a selected runtime cannot continue its conversation.
- Runtime contract tests must exercise resume across a fresh runtime instance.
- Session identifiers and files become explicit Run Store references.
- A fresh retry and a true conversation resume are reported separately in evaluation metrics.
