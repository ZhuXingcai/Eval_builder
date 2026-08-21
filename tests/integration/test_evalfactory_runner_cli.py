from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    JobStatus,
    ResourceBudget,
    StageRunStatus,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2, StageNameV2
from eval_factory.orchestration import JobStore
from eval_factory.trace.adapters import RawTrajV1Adapter
from eval_factory.trace.query.cli import app

NOW = datetime(2026, 7, 22, tzinfo=UTC)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="evalfactory-runner-cli-test",
        governing_versions=(VersionBinding(component="serial-trace-runner", version="v1"),),
    )


def _ref(object_type: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://runner-cli",
        object_version="v1",
        object_sha256="a" * 64,
    )


def _trace(path: Path, needle: str = "NeedleRunnerCli") -> TraceSourceRef:
    request = {
        "messages": [
            {"role": "user", "content": needle},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "cli.txt"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": needle}],
            },
        ]
    }
    outer = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "RunnerCli",
        "endpoint": "offline",
        "event_time": "2026-07-22T00:00:00Z",
        "extra": '{"req_lost_number":0,"resp_lost_number":0}',
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-07-22",
        "request": json.dumps(request, sort_keys=True, separators=(",", ":")),
        "response": '{"content":[],"role":"assistant"}',
        "sid": "synthetic-session",
        "source": None,
    }
    path.write_text(json.dumps(outer, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    probe = RawTrajV1Adapter().probe(path)
    assert probe.raw_sha256 is not None
    return TraceSourceRef(
        source_trace_id="source-trace://runner-cli",
        source_uri=path.resolve().as_uri(),
        raw_sha256=probe.raw_sha256,
        adapter_name="raw_traj_v1",
        adapter_version="1.0.0",
        processing_class="RESTRICTED_TRACE_RAW",
    )


def _spec(path: Path, trace: TraceSourceRef) -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id="job://runner-cli",
        traces=(trace,),
        requested_stages=(StageNameV2.TRACE_INDEX,),
        privacy_profile="trusted-monitored-local",
        budget=ResourceBudget(
            max_model_requests=0,
            max_model_tokens=0,
            max_processes=1,
            max_renderers=0,
            max_network_requests=0,
            max_storage_bytes=1024 * 1024,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        approval_policy_ref=_ref("user-approval-policy"),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key="create-job-runner-cli",
        audit=_audit(),
    )


def _write_spec(tmp_path: Path) -> Path:
    spec = _spec(tmp_path / "spec.json", _trace(tmp_path / "runner-cli.jsonl"))
    path = tmp_path / "job-spec.json"
    path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_evalfactory_run_trace_index_and_resume_output_bounded_json(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    job_store = tmp_path / "job.sqlite3"
    source_registry = tmp_path / "source-registry.sqlite3"
    trace_store = tmp_path / "trace-store"

    run = CliRunner().invoke(
        app,
        [
            "run",
            "trace-index",
            "--job-store",
            str(job_store),
            "--source-registry",
            str(source_registry),
            "--trace-store",
            str(trace_store),
            "--spec",
            str(spec_path),
            "--json",
        ],
    )
    assert run.exit_code == 0, run.output
    payload = json.loads(run.output)
    assert payload["job_status"] == "SUCCEEDED"
    assert payload["item_summaries"][0]["output_refs"][0]["object_type"] == "trace-envelope"
    assert "NeedleRunnerCli" not in run.output

    resume = CliRunner().invoke(
        app,
        [
            "run",
            "resume",
            "--job-store",
            str(job_store),
            "--source-registry",
            str(source_registry),
            "--trace-store",
            str(trace_store),
            "--job",
            "job://runner-cli",
            "--json",
        ],
    )
    assert resume.exit_code == 0, resume.output
    replay = json.loads(resume.output)
    assert replay["executed_count"] == 0
    assert replay["skipped_completed_count"] == 1


def test_evalfactory_run_cancel_marks_active_stage_and_blocks_resume(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "spec.json", _trace(tmp_path / "runner-cli.jsonl"))
    job_store_path = tmp_path / "job.sqlite3"
    store = JobStore(job_store_path, clock=lambda: NOW)
    job = store.create_job(spec)
    running_job = store.transition_job(
        job.job_id,
        JobStatus.RUNNING,
        expected_version=job.row_version,
        idempotency_key="runner-cli-job-running",
    )
    item = store.create_item(
        running_job.job_id,
        "item://runner-cli",
        idempotency_key="runner-cli-create-item",
    )
    stage = store.create_stage_run(
        stage_run_id="stage-run://runner-cli",
        job_id=running_job.job_id,
        item_id=item.item_id,
        stage=StageNameV2.TRACE_INDEX,
        attempt=1,
        principal_ref=_ref("stage-principal"),
        input_refs=(_ref("trace-source"),),
        idempotency_key="runner-cli-create-stage",
    )
    store.transition_stage_run(
        stage.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=stage.row_version,
        idempotency_key="runner-cli-start-stage",
    )

    cancel = CliRunner().invoke(
        app,
        [
            "run",
            "cancel",
            "--job-store",
            str(job_store_path),
            "--job",
            "job://runner-cli",
            "--json",
        ],
    )

    assert cancel.exit_code == 0, cancel.output
    payload = json.loads(cancel.output)
    assert payload["job_status"] == "CANCELLED"
    assert store.get_stage_result_for_run(stage.stage_run_id).status is StageRunStatus.CANCELLED
