# Feature Specification: Agent Foundation

**Feature Branch**: `main`  
**Created**: 2026-07-18  
**Status**: Approved  
**Input**: Build a durable, evidence-grounded environment attachment generation agent.

## User Scenarios

### Scenario 1: Generate an input-state package

Given a task prompt, dependency metadata, optional trace, rubrics, and existing files, the operator runs
one CLI command. The system produces a task-supporting input package without any requested final
deliverable or answer leakage.

### Scenario 2: Resume an interrupted run

Given a run that stopped during retrieval, generation, validation, or review, the operator resumes by
run ID. Completed side effects are not repeated and the workflow continues from persisted state.

### Scenario 3: Compare agent harnesses

Given the same task and model, the operator runs Claude Code CLI, Claude Agent SDK, and Pi. The system
records normalized events, outputs, quality, usage, cost, and duration so harness behavior can be
compared independently of the model.

### Scenario 4: Continue development in a fresh session

Given no prior chat transcript, a new coding agent reads `AGENTS.md` and runs `envmock context
bootstrap`. It can identify the current goal, architecture, active task, decisions, blockers, tests, and
next action without asking for project history.

## Functional Requirements

- **FR-001**: The system MUST normalize Generic YAML/JSON, cc-mock-env CSV, and LH package inputs.
- **FR-002**: It MUST classify dependencies separately from final outputs.
- **FR-003**: It MUST store source evidence for retrieved public material.
- **FR-004**: It MUST route deterministic providers before open-ended runtimes.
- **FR-005**: It MUST support Claude Code CLI, Claude Agent SDK, Pi, and a fake test runtime.
- **FR-006**: It MUST persist run state and normalized runtime events.
- **FR-007**: It MUST validate artifacts with format-specific validators.
- **FR-008**: Standard quality mode MUST perform three distinct review rounds.
- **FR-009**: P0 findings MUST block package export.
- **FR-010**: It MUST export Generic, CC-compatible, and LH-compatible package profiles.
- **FR-011**: Critical capability gaps MUST return `BLOCKED_CAPABILITY`, never silent degradation.
- **FR-012**: Development state MUST be recoverable from repository artifacts and Beads.

## Non-Functional Requirements

- **NFR-001**: Python control-plane contracts are Pydantic v2 models.
- **NFR-002**: Runtime implementations are replaceable behind one protocol.
- **NFR-003**: Local process mode is limited to trusted, monitored inputs.
- **NFR-004**: Tests do not require live credentials by default.
- **NFR-005**: Every run is auditable through JSON/JSONL files.
- **NFR-006**: User-visible package outputs exclude process evidence.

## Success Criteria

- All critical dependencies are represented and validated.
- No corrupt attachment, real secret, or final-answer leak is exported.
- A deliberately interrupted test run resumes successfully.
- The three runtimes pass the same contract suite.
- A fresh-session continuity test recovers project state without chat history.
- LH_023 and LH_067 produce complete benchmark evidence for all planned trials or explicit failures.
