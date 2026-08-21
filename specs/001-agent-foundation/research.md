# Research: Agent Foundation

## Decisions

- LangGraph is the control plane because the workload needs durable state, conditional routing, and
  idempotent side-effect boundaries.
- Pi remains a runtime. Its stable SDK/RPC and extension system are useful, while the official
  orchestrator package is experimental and does not implement the domain workflow.
- Claude Code CLI is preserved as a behavioral baseline.
- Claude Agent SDK is added for typed hooks and production integration while retaining Claude Code's
  tool loop.
- Spec Kit stores intent and architecture; Beads stores active dependency-aware work.
- Runtime sessions and compaction summaries are evidence, not project memory.

## Primary Sources

- `<workspace>/private-research/environment-attachment-agent-architecture.md`
- `https://pi.dev/docs/latest`
- `https://docs.langchain.com/oss/python/langgraph/durable-execution`
- `https://platform.claude.com/docs/en/agent-sdk/overview`
- `https://github.com/github/spec-kit`
- `https://github.com/gastownhall/beads`
