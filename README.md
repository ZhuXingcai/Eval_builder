# Eval Builder

Eval Builder is an AI-native harness for producing auditable evaluation
datasets from natural-language requirements, admitted agent traces, and
versioned evaluation policies.

The project is evolving from an attachment reconstruction tool into an
**Evaluation Dataset Agent Harness**. Its control plane resembles a coding
agent rather than a chat application with miscellaneous tools:

- a persistent Coordinator session owns user interaction;
- specialist capabilities remain independently invocable;
- a collaborative Agent Team shares a task DAG, mailbox, and Artifact
  Blackboard;
- LangGraph controls transitions, interruption, recovery, and convergence;
- domain stores retain canonical business authority;
- every output is linked to typed evidence, provenance, and immutable refs.

> **Current status:** V1 Stages 0-4 are mechanism-complete and fully
> regression-tested. Stage 5 Agent Shell and Stage 6 user delivery are
> specified but not implemented. Real semantic acceptance is blocked by
> Provider authentication. See [Unfinished Work](docs/UNFINISHED.md).

## Product Model

```text
Agent Shell
  conversation | composer | activity | team | workspaces | approvals
                              |
Harness Kernel                |
  sessions | capability runtime | permissions | artifact envelopes
                              |
Graph + Team Runtime
  FactoryControlGraph | PlannerAssessment | checkpoints | DAG | mailbox
                              |
generic-agent-trace Pack 1.2.0
  requirement | trace | task | attachment | quality
  rubric | grading | batch quality | plan review | delivery
                              |
Canonical Stores And CAS
  Factory | Job | Team | Gateway | Graph journal | domain artifacts
```

The Coordinator distributes work and resolves convergence. It does not
overwrite specialist artifacts, relay every peer message, or grant authority.

## Implemented Capabilities

### Harness And Collaboration

- Strict, frozen session, capability, artifact, permission, Team, Blueprint,
  Graph binding, and checkpoint contracts.
- Append-only SQLite Harness sessions with deterministic projection rebuild.
- Capability runtime with exact provider/composition/request validation.
- Team task DAG, leases and fencing, P2P mailbox, subscriptions, conflicts,
  Blackboard heads, member context, and checkpoints.

### Graph Engineering

- Pack `1.2.0` first-party Graph composition mounted as the default CLI path.
- Dedicated graph journal with PRE, POST, and reconciliation checkpoints.
- Committed `PlannerAssessment` routing for continue, retry, replan, escalate,
  and finish.
- Exact PlanReview continuation across Factory, Team, Graph, and session
  authority.
- Trace-disposition-driven, non-widening Team graph narrowing.
- Explicit `--direct-runtime-compatibility`; no implicit fallback.

### Evaluation Dataset Pipeline

- Requirement planning and trace admission.
- Raw trajectory, runtime snapshot, and curated trajectory adapters.
- Candidate selection, task authoring, attachment reconstruction, criteria and
  rubric design, grading design, item quality, batch quality, safety,
  duplicate/leakage checks, and lineage validation.
- Canonical candidate JSON/JSONL output and immutable delivery manifest.
- Zero-, one-, and multi-candidate terminal behavior with exact replay.

### Existing Web Surface

- React 19/Vite PlanReview workbench.
- FastAPI typed PlanReview API.
- Responsive desktop/tablet/mobile behavior.
- Vitest and Playwright coverage.

The conversation-first Agent Shell described in Stage 5 is not yet available.

## Version Lines

| Line | Purpose | Status |
|---|---|---|
| EDF Foundation | Trace-to-evaluation dataset pipeline and release safety | Implemented |
| V1 Stages 0-4 | Static `generic-agent-trace` Harness, Team, and Graph runtime | Mechanism complete |
| V1 Stage 5 | Conversation-first Agent Shell and HTTP/SSE product host | Planned |
| V1 Stage 6 | CSV/XLSX/ZIP delivery and V1 evidence freeze | Planned |
| V2 Platform | Installable cross-domain Eval Packs and dynamic Blueprint composition | Blocked until V1 Stable |

See [Versioning](docs/VERSIONING.md) for branch/tag meaning and
[Unfinished Work](docs/UNFINISHED.md) for exact remaining gates.

## Repository Layout

```text
src/env_mock_agent/             attachment reconstruction subsystem
src/eval_factory/harness/       session, capability, permission, projection
src/eval_factory/team/          collaborative Team runtime
src/eval_factory/agent_system/  Factory, Graph, review, candidate delivery
src/eval_factory/packs/         static evaluation Packs
src/eval_factory/trace/         parsing, recovery, normalization, storage
src/eval_factory/console_api/   typed FastAPI surface
web/eval_factory_console/       React/Vite workbench
tests/                          contract, unit, property, integration, E2E
specs/                          product and generated contract specifications
docs/adr/                       architecture decisions
evals/                          safe checked-in fixtures and manifests
```

## Requirements

- Python 3.12
- `uv`
- Node.js 22
- Chromium for Browser E2E
- Provider credentials only for separately authorized live semantic tests

## Setup

```bash
git clone https://github.com/ZhuXingcai/Eval_builder.git
cd Eval_builder
uv sync --all-extras
npm install --prefix node/pi_bridge
npm install --prefix web/eval_factory_console
```

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q

npm run check --prefix node/pi_bridge
npm run build --prefix node/pi_bridge
npm test --prefix node/pi_bridge

npm run contract:check --prefix web/eval_factory_console
npm run check --prefix web/eval_factory_console
npm test --prefix web/eval_factory_console
npm run build --prefix web/eval_factory_console
npm run test:e2e --prefix web/eval_factory_console
```

Latest verified Python baseline:

```text
2787 passed
```

The public clone does not contain the private 91-trace corpus or local
Beads/continuity state. Tests that require those resources skip explicitly
when the resources are absent; all assertions still run in an authorized local
workspace where they are installed.

## Current CLI

```text
envmock doctor
envmock ingest
envmock plan
envmock generate
envmock resume
envmock inspect
envmock export

evalfactory agent run
evalfactory agent serve
evalfactory pipeline ...
```

`evalfactory agent run` uses the Factory Graph by default. The legacy direct
runtime is available only through `--direct-runtime-compatibility`.

## Web Console

The current UI is a PlanReview/Graph workbench:

```bash
uv run evalfactory agent serve \
  --factory-store PATH \
  --registry PATH

npm run dev --prefix web/eval_factory_console
```

Stage 5 will make persistent conversation the default surface and add session,
Team, Activity, source admission, SSE, and delivery workspaces.

## Evidence Classes

The repository never treats deterministic fixtures as proof of model quality:

- `MECHANISM_FIXTURE`: deterministic implementation/recovery evidence.
- `REAL_SEMANTIC`: separately authorized real-provider evidence.
- `SEMANTIC_PENDING`: mechanism passed, but real semantic evidence is
  unavailable or blocked.

The current semantic gate is `SEMANTIC_PENDING` because the configured
Provider credential is rejected. No production-readiness claim is made.

## Security And Privacy

- Raw traces and runtime outputs are not committed.
- `.env*`, credentials, tokens, private keys, local Agent state, Beads,
  handoffs, and generated run directories are ignored.
- Browser/API projections exclude raw traces, provider response bodies,
  private references, grader rules, hidden conditions, and original final
  answers.
- The local-process execution profile is for trusted, monitored inputs and is
  not a sandbox.
- Production publication remains explicitly blocked.

## Documentation

- [Project specification](specs/003-evaluation-agent-harness/spec.md)
- [Implementation plan](specs/003-evaluation-agent-harness/plan.md)
- [Versioning](docs/VERSIONING.md)
- [Unfinished work](docs/UNFINISHED.md)
- [Architecture decisions](docs/adr/)
- [Data classification](docs/policies/eval-factory-data-classification-v2.md)

## License

No open-source license has been granted yet. All rights are reserved until a
license is added explicitly.
