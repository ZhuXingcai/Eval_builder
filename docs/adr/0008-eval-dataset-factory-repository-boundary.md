# ADR 0008: Keep Eval Dataset Factory In The Current Repository As Bounded Contexts

- Status: Accepted
- Date: 2026-07-19
- Beads: `env_mock_agent-ujc.1.2`
- Spec: `specs/002-eval-dataset-factory/spec.md`

## Context

The repository already contains the attachment generation control plane, runtime adapters, deterministic
providers, validators, profile exporters, continuity tooling, tests, and packaging configuration. The
Eval Dataset Factory adds trace parsing, safety policy, labeling, task authoring, batch orchestration,
review, and dataset release around that existing subsystem.

Creating a separate repository before the cross-stage contracts are implemented would force versioned
remote boundaries across components that are still changing together. Keeping every new concern inside
`env_mock_agent` would create the opposite problem: trace intelligence and dataset release would become
coupled to attachment-specific schemas and workflow nodes.

## Decision

The first release remains in this repository and uses explicit Python bounded contexts:

```text
src/
  env_mock_agent/     # existing attachment reconstruction subsystem
  eval_factory/       # new factory control plane and stage services
```

The `eval_factory` package owns:

- factory-internal and dataset-facing cross-stage contracts;
- Job, Item, StageRun, and release orchestration;
- trace adapters, parsing, indexing, and bounded query;
- provenance, taint, redaction, and evidence views;
- labeling and candidate selection;
- task and evaluation-contract authoring;
- batch review, human review, and dataset release.

The `env_mock_agent` package continues to own:

- existing attachment providers and validators;
- Claude CLI, Claude SDK, Pi, and fake runtimes;
- attachment-specific LangGraph nodes and Run Store;
- Generic, CC, and LH compatibility adapters and profiles.

Integration occurs through versioned Pydantic contracts, initially
`AttachmentReconstructionRequest` and `AttachmentReconstructionResult`.

The physical attachment facade and these two provider-owned contracts live under
`env_mock_agent.facade`. They use stable primitive values and attachment-owned schemas only; they MUST
NOT import any `eval_factory` type. `eval_factory` imports and calls this facade as its consumer. All
other factory contracts live under `eval_factory.contracts`.

The attachment subsystem MUST NOT import trace parsing, labeling, task-authoring, review, release, or
factory-internal contract implementations.

## Dependency Rules

Allowed:

```text
eval_factory -> env_mock_agent.facade
env_mock_agent.facade -> env_mock_agent attachment implementation
env_mock_agent -> shared provider/runtime abstractions already owned by env_mock_agent
both packages -> provider-neutral third-party libraries
```

Forbidden:

```text
env_mock_agent -> eval_factory.trace
env_mock_agent -> eval_factory.labeling
env_mock_agent -> eval_factory.task_authoring
env_mock_agent -> eval_factory.review
env_mock_agent -> eval_factory.dataset
env_mock_agent -> eval_factory.contracts
provider-neutral modules -> vendor SDKs
```

Cross-context state is passed by immutable references and typed values, not by importing another
context's storage internals or sharing runtime transcripts.

## Packaging And CLI

The wheel build currently includes only `src/env_mock_agent`. Before the first `eval_factory` module is
released, packaging MUST include both packages and contract tests MUST import them from a built wheel.

The existing `envmock = env_mock_agent.cli:app` entry point remains backward compatible.

The first factory release uses a separate entry point:

```toml
[project.scripts]
envmock = "env_mock_agent.cli:app"
evalfactory = "eval_factory.cli:app"
```

`evalfactory` owns trace ingestion, inspection, labeling, task reconstruction, attachment orchestration,
pipeline run/resume, review, and export commands. It calls attachment behavior through
`env_mock_agent.facade`; it does not duplicate provider or runtime implementations. A future CLI
consolidation requires a separate compatibility decision.

## Storage Boundary

- Existing attachment Run Store remains readable by `env_mock_agent`.
- The new factory Job Store owns factory state and references attachment run IDs.
- Factory state MUST NOT copy attachment process files or vendor transcripts.
- Raw traces remain immutable references or content-addressed objects outside exported packages.

## Migration And Split Criteria

No repository split is planned for the first release. A later split requires a new ADR and evidence
that at least one condition is true:

1. independent deployment or release cadence is required;
2. organizational ownership requires separate access control;
3. data classification requires a repository or service boundary;
4. measured build or test coupling makes the monorepo materially harmful;
5. the versioned attachment facade has remained stable across at least one production release.

The split plan must define contract versioning, package publication, data migration, CI ownership, and
rollback. Directory movement alone is not an acceptable split plan.

## Alternatives Considered

### New Top-Level Repository Now

Rejected for the first release. It would introduce package publication, remote contract negotiation,
cross-repository CI, and coordinated migrations before TraceIR and release contracts are stable.

### Put All New Code Under `env_mock_agent`

Rejected. It would preserve one import root at the cost of mixing trace intelligence, labeling, release,
and attachment generation ownership.

### Nested Repository Or Git Submodule

Rejected. It complicates local development and continuity while providing neither a stable service
boundary nor independent deployment.

### Immediate Service Split

Rejected. The first release is a trusted local CLI/batch system, and no measured operational need
justifies a network boundary.

## Consequences

Positive:

- existing providers, validators, runtimes, and profiles are reused without copying;
- one Spec/Beads/ADR continuity system covers the whole factory;
- atomic local changes can evolve contracts and their consumers together;
- bounded imports preserve a future split path.

Negative:

- the repository and test matrix become larger;
- packaging must be updated for a second Python package;
- architectural rules require import-boundary tests and review discipline;
- a later split remains real migration work rather than a directory rename.

## Validation

R0-10 and R1 work MUST add:

- wheel-import tests for both Python packages;
- import-boundary tests for forbidden dependency directions;
- attachment facade contract tests;
- migration tests for any shared schema version change;
- existing `envmock` CLI regression tests.
