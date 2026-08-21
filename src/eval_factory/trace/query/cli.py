from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2
from eval_factory.orchestration import JobStore, SerialTraceRunner, TraceRunnerError
from eval_factory.trace.query import (
    QueryKind,
    TraceQueryError,
    TraceQueryRequest,
    TraceQueryService,
    TrustedTracePrincipal,
)
from eval_factory.trace.source_registry import TraceSourceRegistry
from eval_factory.trace.storage import TraceIndexStore

app = typer.Typer(help="Eval Factory trace evidence tools.")
trace_app = typer.Typer(help="Bounded TraceQueryService commands.")
run_app = typer.Typer(help="Local serial Eval Factory runner commands.")
app.add_typer(trace_app, name="trace")
app.add_typer(run_app, name="run")


@trace_app.command("query")
def trace_query(
    store: Annotated[Path, typer.Option("--store", exists=True, file_okay=False, readable=True)],
    principal: Annotated[Path, typer.Option("--principal", exists=True, readable=True)],
    trace: Annotated[str, typer.Option("--trace")],
    kind: Annotated[QueryKind, typer.Option("--kind")],
    purpose: Annotated[str, typer.Option("--purpose")],
    event_type: Annotated[str | None, typer.Option("--event-type")] = None,
    event_id: Annotated[str | None, typer.Option("--event-id")] = None,
    span_id: Annotated[str | None, typer.Option("--span-id")] = None,
    tool_family: Annotated[str | None, typer.Option("--tool-family")] = None,
    logical_path: Annotated[str | None, typer.Option("--logical-path")] = None,
    boundary_method: Annotated[str | None, typer.Option("--boundary-method")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
    max_characters: Annotated[int, typer.Option("--max-characters")] = 2000,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    filters = _filters(
        event_type=event_type,
        event_id=event_id,
        span_id=span_id,
        tool_family=tool_family,
        logical_path=logical_path,
        boundary_method=boundary_method,
    )
    _run_query(
        store=store,
        principal_path=principal,
        trace=trace,
        purpose=purpose,
        kind=kind,
        filters=filters,
        limit=limit,
        max_characters=max_characters,
    )


@trace_app.command("search")
def trace_search(
    store: Annotated[Path, typer.Option("--store", exists=True, file_okay=False, readable=True)],
    principal: Annotated[Path, typer.Option("--principal", exists=True, readable=True)],
    trace: Annotated[str, typer.Option("--trace")],
    purpose: Annotated[str, typer.Option("--purpose")],
    text: Annotated[str, typer.Option("--text")],
    limit: Annotated[int, typer.Option("--limit")] = 20,
    max_characters: Annotated[int, typer.Option("--max-characters")] = 2000,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _run_query(
        store=store,
        principal_path=principal,
        trace=trace,
        purpose=purpose,
        kind=QueryKind.TEXT_SEARCH,
        filters={"text": text},
        limit=limit,
        max_characters=max_characters,
    )


@run_app.command("trace-index")
def run_trace_index(
    job_store: Annotated[Path, typer.Option("--job-store")],
    source_registry: Annotated[Path, typer.Option("--source-registry")],
    trace_store: Annotated[Path, typer.Option("--trace-store")],
    spec: Annotated[Path, typer.Option("--spec", exists=True, readable=True)],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    runner = _runner(job_store=job_store, source_registry=source_registry, trace_store=trace_store)
    job_spec = DatasetJobSpecV2.model_validate_json(spec.read_text(encoding="utf-8"))
    try:
        summary = runner.run(job_spec)
    except TraceRunnerError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(summary.model_dump_json(indent=2))


@run_app.command("resume")
def run_resume(
    job_store: Annotated[Path, typer.Option("--job-store", exists=True, dir_okay=False, readable=True)],
    source_registry: Annotated[Path, typer.Option("--source-registry")],
    trace_store: Annotated[Path, typer.Option("--trace-store", file_okay=False)],
    job: Annotated[str, typer.Option("--job")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    runner = _runner(job_store=job_store, source_registry=source_registry, trace_store=trace_store)
    try:
        summary = runner.resume(job)
    except TraceRunnerError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(summary.model_dump_json(indent=2))


@run_app.command("cancel")
def run_cancel(
    job_store: Annotated[Path, typer.Option("--job-store", exists=True, dir_okay=False, readable=True)],
    job: Annotated[str, typer.Option("--job")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    runner = SerialTraceRunner(
        job_store=JobStore(job_store),
        source_registry=TraceSourceRegistry(job_store.parent / "unused-source-registry.sqlite3"),
        trace_store=TraceIndexStore(job_store.parent / "unused-trace-store"),
    )
    try:
        summary = runner.cancel(job)
    except TraceRunnerError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(summary.model_dump_json(indent=2))


def _run_query(
    *,
    store: Path,
    principal_path: Path,
    trace: str,
    purpose: str,
    kind: QueryKind,
    filters: dict[str, str | int | bool],
    limit: int,
    max_characters: int,
) -> None:
    principal = TrustedTracePrincipal.model_validate_json(principal_path.read_text(encoding="utf-8"))
    service = TraceQueryService(TraceIndexStore(store))
    request = TraceQueryRequest(
        principal=principal,
        trace_ir_version_id=trace,
        purpose=purpose,
        query_kind=kind,
        filters=filters,
        limit=limit,
        max_characters=max_characters,
    )
    try:
        response = service.query(request)
    except TraceQueryError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(response.model_dump_json(indent=2))


def _runner(
    *,
    job_store: Path,
    source_registry: Path,
    trace_store: Path,
) -> SerialTraceRunner:
    return SerialTraceRunner(
        job_store=JobStore(job_store),
        source_registry=TraceSourceRegistry(source_registry),
        trace_store=TraceIndexStore(trace_store),
    )


def _filters(
    *,
    event_type: str | None,
    event_id: str | None,
    span_id: str | None,
    tool_family: str | None,
    logical_path: str | None,
    boundary_method: str | None,
) -> dict[str, str | int | bool]:
    values = {
        "event_type": event_type,
        "event_id": event_id,
        "span_id": span_id,
        "tool_family": tool_family,
        "logical_path": logical_path,
        "boundary_method": boundary_method,
    }
    return {key: value for key, value in values.items() if value is not None}
