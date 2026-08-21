# Environment Mock Agent Constitution

## Core Principles

### I. Input-State Integrity

The product creates the environment visible before a tested agent starts. It MUST NOT pre-place the
deliverable requested by the task, model answers, scoring hints, derived conclusions, or fragments that
materially reveal the final solution. Explicit and implicit answer leakage are release blockers.

### II. Evidence Before Synthesis

Every dependency MUST have evidence from prompt, trace, rubrics, existing attachments, or an approved
public source. Source priority is direct observable evidence, explicit task requirements, then inferred
dependencies. Public evidence MUST record provenance, retrieval time, usage basis, and content hash.

### III. Deterministic Capabilities First

If a typed provider can generate or transform an artifact reliably, the workflow MUST use it before an
open-ended agent. Agent runtimes are reserved for ambiguous planning, research, natural-language
authorship, code construction, and recovery that cannot be expressed deterministically.

### IV. Structural And Semantic Gates

File existence is not quality. Every artifact MUST pass its format validator and the package MUST pass
three independent semantic review rounds: coverage/solvability, realism/consistency, and
leakage/executability. Script-green without semantic acceptance is not deliverable.

### V. Replaceable Runtimes

Claude Agent SDK, Claude Code CLI, Pi, and future runtimes MUST implement one typed contract. Business
state, source evidence, package rules, and quality decisions MUST NOT live inside a vendor transcript.
The control plane records every routing decision and fallback.

### VI. Durable Project Truth

Specifications, ADRs, Beads tasks, validation evidence, project state, and handoffs in this repository
are authoritative. Chat history, compaction summaries, and runtime sessions are disposable evidence.
Every active task MUST be recoverable in a fresh session through `envmock context bootstrap`.

### VII. Test And Evaluation Driven Development

No task is closed without its acceptance evidence and validation command. Deterministic unit, contract,
integration, property, and agent evaluation suites MUST scale with risk. Changes to models, prompts,
tools, runtimes, providers, validators, or profiles MUST run the relevant golden regression suite.

### VIII. Least Privilege And Explicit Degradation

Local process mode is not a sandbox. It is limited to trusted, monitored inputs, isolated work
directories, explicit tool allowlists, timeouts, and path guards. A missing critical capability MUST
produce `BLOCKED_CAPABILITY`; silent fallback to a low-fidelity artifact is prohibited.

## Technical Constraints

- Python 3.12 with `uv` and a committed lock file is the control-plane baseline.
- LangGraph owns workflow state and checkpointing.
- Pydantic v2 schemas define all external and cross-runtime contracts.
- Node.js 22 hosts the Pi bridge.
- The official Anthropic SDK is required in Anthropic-specific modules.
- Provider-neutral modules MUST NOT import a vendor SDK.
- Runtime outputs live under `runs/` and are not committed.
- Secrets are provided through environment variables or isolated credential stores, never project files.
- The first release supports local CLI and batch execution; untrusted unattended execution requires a
  later container workspace.

## Development Workflow

1. Run `bd prime` and `envmock context bootstrap`.
2. Claim one ready Beads issue.
3. Read the governing spec and ADRs before editing.
4. Add or update tests before closing behavioral work.
5. Keep file edits scoped to the claimed task.
6. Run the issue's acceptance commands.
7. Update the Beads issue with evidence.
8. Run `envmock context checkpoint` before handoff.

ADRs are append-only; supersede an old decision rather than rewriting history. Markdown TODO lists are
not a task system. Any discovered work that cannot be completed in scope becomes a Beads issue.

## Governance

This constitution supersedes ad hoc prompts and runtime behavior. Amendments require:

1. A reviewed specification change.
2. An ADR explaining the reason and compatibility impact.
3. Updated tests and migration steps.
4. A version increment below.

**Version**: 1.0.0 | **Ratified**: 2026-07-18 | **Last Amended**: 2026-07-18
