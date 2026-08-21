# Implementation Plan: Agent Foundation

## Technical Context

- Python 3.12, `uv`, Pydantic v2, Typer, LangGraph, SQLite checkpointing.
- Node.js 22 bridge for Pi 0.80.10.
- Claude Code CLI and official Python Claude Agent SDK.
- Local process execution for trusted inputs only.
- JSON/JSONL run store; package output controlled by profiles.

## Architecture

1. Adapters normalize external inputs to `TaskSpec`.
2. LangGraph nodes plan dependencies, capabilities, evidence, artifacts, validation, reviews, repair,
   assembly, and export.
3. Providers create deterministic artifacts.
4. Runtimes handle open-ended work using one event contract.
5. Validators emit typed findings.
6. The problem ledger drives targeted repairs.
7. Profiles export only user-visible package data.

## Delivery Sequence

1. Project continuity foundation.
2. Schemas, adapters, CLI, run store, and profile export.
3. Runtime implementations and router.
4. Evidence, providers, and validators.
5. LangGraph review/repair workflow.
6. LH benchmark trials and decision report.

## Quality Gates

- Ruff, mypy, pytest, Node type checking.
- Runtime and provider contract suites.
- Interrupted-run recovery integration test.
- Fresh-session continuity test.
- LH package boundary test.
- Live tests isolated by pytest markers.

## Compatibility

- Existing cc-mock-env and LH packages remain read-only.
- CC CSV values are preserved; rubric scores are not rewritten.
- LH export retains `query.yaml`, `.eval/rubrics.json`, and `workspace/`.
