from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from env_mock_agent import __version__
from env_mock_agent.adapters import load_tasks
from env_mock_agent.config import find_project_root, get_settings
from env_mock_agent.context.bootstrap import collect_bootstrap_state, render_bootstrap
from env_mock_agent.context.checkpoint import validate_context, write_checkpoint
from env_mock_agent.evals import BenchmarkRunner, BenchmarkSuite
from env_mock_agent.graph import EnvironmentMockWorkflow
from env_mock_agent.profiles import get_profile
from env_mock_agent.providers import ProviderRegistry
from env_mock_agent.schemas import (
    ArtifactPlan,
    InputAdapterType,
    RunRecord,
    RunStatus,
    TaskSpec,
)
from env_mock_agent.store import RunStore, sqlite_checkpointer

app = typer.Typer(help="Generate and validate evidence-grounded environment attachment packages.")
context_app = typer.Typer(help="Recover and persist development context.")
app.add_typer(context_app, name="context")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    del version


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Audit the development and attachment toolchain without changing it."""
    tools = [
        ("python", [sys.executable, "--version"], True),
        ("git", ["git", "--version"], True),
        ("bd", ["bd", "version"], True),
        ("node", ["node", "--version"], True),
        ("npm", ["npm", "--version"], True),
        ("claude", ["claude", "--version"], False),
        ("pi", ["pi", "--version"], False),
        ("libreoffice", ["libreoffice", "--version"], False),
        ("ffmpeg", ["ffmpeg", "-version"], False),
        ("pandoc", ["pandoc", "--version"], False),
        ("pdftotext", ["pdftotext", "-v"], False),
        ("tesseract", ["tesseract", "--version"], False),
        ("qpdf", ["qpdf", "--version"], False),
        ("gs", ["gs", "--version"], False),
        ("magick", ["magick", "--version"], False),
    ]
    results: list[dict[str, object]] = []
    for name, command, required in tools:
        executable = shutil.which(command[0])
        status = "missing"
        version = ""
        if executable:
            process = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            output = (process.stdout or process.stderr).strip().splitlines()
            status = "ok" if process.returncode == 0 else "error"
            version = output[0] if output else ""
        results.append(
            {
                "name": name,
                "required": required,
                "status": status,
                "path": executable,
                "version": version,
            }
        )

    payload = {
        "project_root": str(find_project_root()),
        "healthy": all(item["status"] == "ok" for item in results if item["required"]),
        "tools": results,
        "providers": [
            capability.model_dump(mode="json") for capability in ProviderRegistry.default().capabilities()
        ],
        "security": {
            "local_process_mode": True,
            "sandboxed": False,
            "trusted_monitored_inputs_only": True,
        },
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    table = Table(title="Environment Mock Agent Toolchain")
    table.add_column("Tool")
    table.add_column("Level")
    table.add_column("Status")
    table.add_column("Version / Path")
    for item in results:
        detail = str(item["version"] or item["path"] or "")
        table.add_row(
            str(item["name"]),
            "required" if item["required"] else "optional",
            str(item["status"]),
            detail,
        )
    console.print(table)
    if not payload["healthy"]:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    adapter: Annotated[InputAdapterType, typer.Option("--adapter", help="Input adapter type.")],
    input_path: Annotated[Path, typer.Option("--input", exists=True, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", help="Output JSON file or directory.")] = None,
    task_id: Annotated[str | None, typer.Option("--task-id", help="Select one task.")] = None,
) -> None:
    """Normalize Generic, CC CSV, or LH inputs into TaskSpec JSON."""
    tasks = load_tasks(adapter, input_path, task_id)
    if not tasks:
        raise typer.BadParameter("no matching tasks found")
    if output is None and len(tasks) == 1:
        typer.echo(tasks[0].model_dump_json(indent=2))
        return
    destination = (output or Path("normalized_tasks")).expanduser().resolve()
    if len(tasks) == 1 and destination.suffix.lower() == ".json":
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(tasks[0].model_dump_json(indent=2) + "\n", encoding="utf-8")
        typer.echo(destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        target = destination / f"{task.task_id}.json"
        target.write_text(task.model_dump_json(indent=2) + "\n", encoding="utf-8")
        typer.echo(target)


@app.command("plan")
def plan_task(
    task_path: Annotated[Path, typer.Option("--task", exists=True, readable=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Create deterministic artifact plans from a normalized task."""
    task = TaskSpec.model_validate_json(task_path.read_text(encoding="utf-8"))
    plans = [
        ArtifactPlan(
            artifact_id=f"A-{index:03d}",
            dependency_id=dependency.dependency_id,
            relative_path=dependency.path,
            asset_type=dependency.asset_type,
            validators=[dependency.asset_type, "common", "leakage"],
            source_evidence_ids=[],
            seed=index,
        )
        for index, dependency in enumerate(task.dependencies, start=1)
    ]
    payload = {"task_id": task.task_id, "artifacts": [item.model_dump(mode="json") for item in plans]}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output:
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        typer.echo(destination)
    else:
        typer.echo(text, nl=False)


def _new_run_id(task_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_task = "".join(character if character.isalnum() else "-" for character in task_id).strip("-")
    return f"{timestamp}-{safe_task}-{uuid.uuid4().hex[:8]}"


@app.command()
def create_run(
    task_path: Annotated[Path, typer.Option("--task", exists=True, readable=True)],
    profile: Annotated[str, typer.Option("--profile")] = "generic",
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
) -> None:
    """Create a durable Run Store without executing generation."""
    task = TaskSpec.model_validate_json(task_path.read_text(encoding="utf-8"))
    identifier = run_id or _new_run_id(task.task_id)
    store = RunStore(get_settings().runs_root, identifier)
    record = RunRecord(
        run_id=identifier,
        task_id=task.task_id,
        profile=profile,
        task_path=str(task_path.expanduser().resolve()),
    )
    store.initialize(record, task)
    typer.echo(identifier)


@app.command()
def generate(
    task_path: Annotated[Path, typer.Option("--task", exists=True, readable=True)],
    profile: Annotated[str, typer.Option("--profile")] = "generic",
    quality: Annotated[str, typer.Option("--quality")] = "standard",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    runtime_name: Annotated[str | None, typer.Option("--runtime")] = None,
    model_provider: Annotated[str | None, typer.Option("--model-provider")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Run the durable generation, validation, review, repair, and export workflow."""
    if quality != "standard":
        raise typer.BadParameter("only the standard three-round quality profile is implemented")
    task = TaskSpec.model_validate_json(task_path.read_text(encoding="utf-8"))
    task.profile = profile
    if runtime_name or model_provider or model:
        if not all((runtime_name, model_provider, model)):
            raise typer.BadParameter("--runtime, --model-provider, and --model must be provided together")
        task.metadata["runtime_name"] = runtime_name
        task.metadata["model_profile"] = {
            "provider": model_provider,
            "model": model,
        }
    settings = get_settings()
    identifier = run_id or _new_run_id(task.task_id)
    store = RunStore(settings.runs_root, identifier)
    store.initialize(
        RunRecord(
            run_id=identifier,
            task_id=task.task_id,
            profile=profile,
            task_path=str(task_path.expanduser().resolve()),
        ),
        task,
    )
    state: dict[str, object] = {
        "run_id": identifier,
        "runs_root": str(settings.runs_root),
        "profile": profile,
        "task": task.model_dump(mode="json"),
        "overwrite": overwrite,
    }
    if output is not None:
        state["export_path"] = str(output.expanduser().resolve())
    with sqlite_checkpointer(settings.checkpoint_path) as checkpointer:
        graph = cast(Any, EnvironmentMockWorkflow().compile(checkpointer))
        result = graph.invoke(
            state,
            {"configurable": {"thread_id": identifier}},
        )
    typer.echo(
        json.dumps(
            {
                "run_id": identifier,
                "status": result.get("final_status", store.read_record().status.value),
                "package_path": result.get("package_path"),
                "export_path": result.get("export_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def resume(run_id: Annotated[str, typer.Argument()]) -> None:
    """Resume a workflow from its durable LangGraph checkpoint."""
    settings = get_settings()
    store = RunStore.open(settings.runs_root, run_id)
    if store.read_record().status == RunStatus.CANCELLED:
        raise typer.BadParameter(f"run is cancelled and cannot be resumed: {run_id}")
    with sqlite_checkpointer(settings.checkpoint_path) as checkpointer:
        graph = cast(Any, EnvironmentMockWorkflow().compile(checkpointer))
        result = graph.invoke(
            None,
            {"configurable": {"thread_id": run_id}},
        )
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "status": result.get("final_status", store.read_record().status.value),
                "package_path": result.get("package_path"),
                "export_path": result.get("export_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def cancel(run_id: Annotated[str, typer.Argument()]) -> None:
    """Persist a cancellation request for an active local workflow."""
    store = RunStore.open(get_settings().runs_root, run_id)
    record = store.read_record()
    if record.status in {RunStatus.PACKAGED, RunStatus.EXPORTED}:
        raise typer.BadParameter(f"completed run cannot be cancelled: {run_id}")
    store.write_json(
        "cancel.json",
        {
            "run_id": run_id,
            "requested_at": datetime.now(UTC).isoformat(),
        },
    )
    store.update_status(
        RunStatus.CANCELLED,
        current_node=record.current_node,
        error="cancelled by operator",
    )
    typer.echo(run_id)


@app.command()
def benchmark(
    suite_path: Annotated[Path, typer.Option("--suite", exists=True, readable=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Run a benchmark suite or emit explicit preflight failure evidence."""
    suite = BenchmarkSuite.load(suite_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = (
        output.expanduser().resolve()
        if output is not None
        else find_project_root() / "evals/results" / f"{timestamp}-{suite.suite_id}"
    )
    result_path = BenchmarkRunner(suite, destination).run()
    typer.echo(result_path)


@app.command("inspect")
def inspect_run(
    target: Annotated[str, typer.Argument(help="Run ID or package path.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Inspect a persisted run or package boundary."""
    path = Path(target).expanduser()
    payload: dict[str, object]
    if path.exists():
        resolved = path.resolve()
        payload = {
            "type": "package",
            "path": str(resolved),
            "top_level": sorted(item.name for item in resolved.iterdir()) if resolved.is_dir() else [],
        }
    else:
        store = RunStore.open(get_settings().runs_root, target)
        payload = {
            "type": "run",
            "record": store.read_record().model_dump(mode="json"),
            "events": list(store.events.read()),
        }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        console.print_json(data=payload)


@app.command("export")
def export_run(
    run_id: Annotated[str, typer.Argument()],
    profile: Annotated[str, typer.Option("--profile")],
    output: Annotated[Path, typer.Option("--output")],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Export a Core Run through a package profile."""
    store = RunStore.open(get_settings().runs_root, run_id)
    task = store.read_model("task/normalized_task.json", TaskSpec)
    destination = get_profile(profile).export(store, task, output, overwrite=overwrite)
    typer.echo(destination)


@context_app.command("bootstrap")
def context_bootstrap(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Print the canonical context required by a fresh development session."""
    state = collect_bootstrap_state()
    typer.echo(json.dumps(state, ensure_ascii=False, indent=2) if json_output else render_bootstrap(state))


@context_app.command("checkpoint")
def context_checkpoint(
    task_id: Annotated[str | None, typer.Option("--task-id", help="Active Beads task ID.")] = None,
    validation: Annotated[
        list[str] | None,
        typer.Option("--validation", help="Validation result to persist; repeatable."),
    ] = None,
    failure: Annotated[
        list[str] | None,
        typer.Option("--failure", help="Current failure or blocker to persist; repeatable."),
    ] = None,
    next_command: Annotated[
        str | None,
        typer.Option("--next-command", help="Exact command the next session should run."),
    ] = None,
) -> None:
    """Persist current project and optional task handoff state."""
    state_path, handoff_path = write_checkpoint(
        task_id=task_id,
        validation_results=validation,
        current_failures=failure,
        next_command=next_command,
    )
    errors = validate_context()
    typer.echo(f"project_state: {state_path}")
    if handoff_path:
        typer.echo(f"handoff: {handoff_path}")
    if errors:
        for error in errors:
            typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1)


@context_app.command("validate")
def context_validate() -> None:
    """Validate PROJECT_STATE freshness and durable handoff records."""
    errors = validate_context()
    if errors:
        for error in errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo("context records valid")


if __name__ == "__main__":
    app()
