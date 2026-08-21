# Runtime Selection After M5

- Status: Provisional
- Date: 2026-07-19
- Benchmark suite: `evals/benchmarks/lh_vertical_slice.yaml`
- Canonical result: `evals/results/20260719T080250Z-lh_023_lh_067_runtime_comparison`

## Evidence

The suite planned 18 same-model main trials:

- `LH_023` and `LH_067`
- Claude Code CLI, Claude Agent SDK, and Pi
- `claude-sonnet-4-6`
- Three repetitions per task and runtime

It also planned one Pi + Ark/GLM subset trial per task. All 20 trial records were emitted as
`blocked_preflight`:

| Runtime | Planned records | Preflight result |
|---|---:|---|
| Claude Code CLI | 6 | Live smoke returned `auth_unavailable` |
| Claude Agent SDK | 6 | `ANTHROPIC_API_KEY` missing |
| Pi + Claude | 6 | `ANTHROPIC_API_KEY` missing |
| Pi + Ark/GLM | 2 | `ARK_API_KEY` missing |

The canonical `manifest.json`, `metrics.jsonl`, and `report.md` contain the per-trial evidence. The
earlier `20260719T075316Z` result is superseded because its Claude CLI probe did not terminate a
variadic tool option before the prompt; it measured a CLI invocation error instead of credential
readiness.

## Decision

Do not rank Runtime quality, stability, cost, or latency from M5. No compared Runtime completed a
generation trial, so a harness-quality conclusion would be unsupported.

Keep ADR 0003's default routing:

1. Deterministic Provider for supported formats.
2. Claude Agent SDK.
3. Claude Code CLI.
4. Pi.
5. `BLOCKED_CAPABILITY`.

This is an implementation policy, not a benchmark winner. Pi remains a provider-neutral runtime and
does not replace LangGraph.

## Inspect AI Status

Inspect AI task/scorer integration is deferred. Every M5 trial stopped before package generation, so
there is no artifact set for an external scorer to evaluate. The independent JSONL benchmark evidence
is the M5 evaluation output; Inspect scoring becomes actionable after the first credential-ready suite
produces packages.

## Reopen Criteria

Rerun the same suite only after:

- Claude CLI live smoke reaches `runtime_finished`.
- `ANTHROPIC_API_KEY` enables both SDK and Pi with `claude-sonnet-4-6`.
- All three main runtimes use the same model and limits.
- `ARK_API_KEY` enables the declared Pi + Ark/GLM subset.

Then compare completion rate, release-gate pass rate, P0/P1 counts, duration, cost, tool calls, and
three-run variance. Until then, M5 satisfies only the explicit-failure-evidence branch of its
acceptance criteria.
