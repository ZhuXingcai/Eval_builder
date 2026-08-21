# Tasks: Agent Foundation

Beads is the authoritative task tracker. This file maps the approved specification to durable Beads
issues and MUST NOT be used as a separate mutable checklist.

| Milestone | Beads issue | Verification |
|---|---|---|
| Project continuity foundation | `env_mock_agent-3nj.1` | Fresh-session bootstrap test |
| Schema, CLI, run store, profiles | `env_mock_agent-3nj.2` | Unit, contract, adapter and export tests |
| Claude CLI/SDK and Pi runtimes | `env_mock_agent-3nj.3` | Shared runtime contract suite |
| Evidence, providers, validators | `env_mock_agent-3nj.4` | Provider/validator round trips |
| LangGraph review/repair workflow | `env_mock_agent-3nj.5` | Interrupted run and P0 export gate |
| LH benchmark experiments | `env_mock_agent-3nj.6` | Benchmark result manifest and report |

Run `bd show <id>` for current state, acceptance criteria, dependencies, and evidence.
