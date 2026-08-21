from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    JobStatus,
    ResourceBudget,
    StageRunStatus,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2, StageNameV2
from eval_factory.orchestration import (
    JobStore,
    SerialTraceRunner,
    StaticTraceRunnerFaultInjector,
    TraceRunnerCancelledError,
    TraceRunnerFaultPoint,
    TraceRunnerInjectedCrash,
    TraceRunnerStageError,
)
from eval_factory.trace import QueryKind, TraceQueryRequest, TraceQueryService, TrustedTracePrincipal
from eval_factory.trace.adapters import RawTrajV1Adapter
from eval_factory.trace.source_registry import TraceSourceRegistry
from eval_factory.trace.storage import TraceIndexStore

NOW = datetime(2026, 7, 22, tzinfo=UTC)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="serial-trace-runner-test",
        governing_versions=(VersionBinding(component="serial-trace-runner", version="v1"),),
    )


def _outer(request: dict[str, object]) -> dict[str, object]:
    return {
        "account": "synthetic",
        "api_type": "chat",
        "business": "RunnerGold",
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


def _write_trace(path: Path, needle: str = "NeedleRunner") -> TraceSourceRef:
    request = {
        "messages": [
            {"role": "user", "content": f"{needle} inspect runner.txt"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "runner.txt"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": needle}],
            },
        ]
    }
    path.write_text(
        json.dumps(_outer(request), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    probe = RawTrajV1Adapter().probe(path)
    assert probe.raw_sha256 is not None
    return TraceSourceRef(
        source_trace_id=f"source-trace://{path.stem}",
        source_uri=path.resolve().as_uri(),
        raw_sha256=probe.raw_sha256,
        adapter_name="raw_traj_v1",
        adapter_version="1.0.0",
        processing_class="RESTRICTED_TRACE_RAW",
    )


def _spec(tmp_path: Path, *traces: TraceSourceRef) -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id="job://r1-09/test",
        traces=traces or (_write_trace(tmp_path / "runner.jsonl"),),
        requested_stages=(StageNameV2.TRACE_INDEX,),
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
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
        idempotency_key="create-job-r1-09-test",
        audit=_audit(),
    )


def _ref(object_type: str) -> object:
    from eval_factory.contracts.core import ObjectRef

    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r1-09-test",
        object_version="v1",
        object_sha256="a" * 64,
    )


def _runner(
    tmp_path: Path,
    *,
    fault_injector: StaticTraceRunnerFaultInjector | None = None,
) -> tuple[SerialTraceRunner, JobStore, TraceIndexStore]:
    job_store = JobStore(tmp_path / "job.sqlite3", clock=lambda: NOW)
    trace_store = TraceIndexStore(tmp_path / "trace-store")
    runner = SerialTraceRunner(
        job_store=job_store,
        source_registry=TraceSourceRegistry(tmp_path / "source-registry.sqlite3"),
        trace_store=trace_store,
        clock=lambda: NOW,
        fault_injector=fault_injector,
    )
    return runner, job_store, trace_store


def _principal(trace_id: str) -> TrustedTracePrincipal:
    return TrustedTracePrincipal(
        principal_id="trace-principal://runner-test",
        consumer_stage="label",
        consumer_agent="runner-verifier",
        allowed_purposes=frozenset({"labeling"}),
        allowed_trace_ir_version_ids=frozenset({trace_id}),
        max_events=20,
        max_characters=200,
        audit=False,
    )


def test_serial_runner_completes_trace_index_and_replay_skips_completed_work(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    runner, job_store, trace_store = _runner(tmp_path)

    first = runner.run(spec)
    replay = runner.run(spec)

    assert first.job_status is JobStatus.SUCCEEDED
    assert first.executed_count == 1
    assert replay.job_status is JobStatus.SUCCEEDED
    assert replay.executed_count == 0
    assert replay.skipped_completed_count == 1
    item = first.item_summaries[0]
    assert item.item_status is ItemStatus.APPROVED
    assert item.stage_status is StageRunStatus.SUCCEEDED
    assert item.output_refs and item.checkpoint_ref == item.output_refs[0]
    assert [event.event_type for event in job_store.list_outbox()].count("stage-run-completed") == 1

    response = TraceQueryService(trace_store).query(
        TraceQueryRequest(
            principal=_principal(item.output_refs[0].object_id),
            trace_ir_version_id=item.output_refs[0].object_id,
            purpose="labeling",
            query_kind=QueryKind.TEXT_SEARCH,
            filters={"text": "NeedleRunner"},
            limit=10,
            max_characters=200,
        )
    )
    assert response.result_refs


@pytest.mark.parametrize(
    "point",
    [
        TraceRunnerFaultPoint.BEFORE_SOURCE_REGISTRATION,
        TraceRunnerFaultPoint.AFTER_SOURCE_REGISTRATION,
        TraceRunnerFaultPoint.AFTER_TRACE_STORE_PERSIST,
        TraceRunnerFaultPoint.BEFORE_STAGE_COMPLETION,
    ],
)
def test_runner_retryable_faults_resume_with_linked_stage_run(
    tmp_path: Path,
    point: TraceRunnerFaultPoint,
) -> None:
    spec = _spec(tmp_path)
    runner, job_store, _trace_store = _runner(
        tmp_path,
        fault_injector=StaticTraceRunnerFaultInjector(failure_points=frozenset({point})),
    )

    with pytest.raises(TraceRunnerStageError, match="Injected runner failure"):
        runner.run(spec)
    failed = job_store.list_stage_runs(job_id=spec.job_id, stage=StageNameV2.TRACE_INDEX)[0]
    failed_result = job_store.get_stage_result_for_run(failed.stage_run_id)
    assert failed_result is not None
    assert failed_result.status is StageRunStatus.RETRYABLE_FAILURE

    resumed_runner, _job_store, _trace_store = _runner(tmp_path)
    summary = resumed_runner.resume(spec.job_id)

    runs = job_store.list_stage_runs(job_id=spec.job_id, stage=StageNameV2.TRACE_INDEX)
    assert summary.job_status is JobStatus.SUCCEEDED
    assert summary.retry_count == 1
    assert len(runs) == 2
    assert runs[1].retry_of_stage_run_id == failed.stage_run_id
    assert job_store.get_stage_result_for_run(runs[1].stage_run_id).status is StageRunStatus.SUCCEEDED


def test_runner_crash_after_persist_replays_storage_and_completes_missing_stage_result(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    runner, job_store, trace_store = _runner(
        tmp_path,
        fault_injector=StaticTraceRunnerFaultInjector(
            crash_points=frozenset({TraceRunnerFaultPoint.AFTER_TRACE_STORE_PERSIST})
        ),
    )

    with pytest.raises(TraceRunnerInjectedCrash):
        runner.run(spec)
    stage = job_store.list_stage_runs(job_id=spec.job_id, stage=StageNameV2.TRACE_INDEX)[0]
    assert stage.status is StageRunStatus.RUNNING
    assert job_store.get_stage_result_for_run(stage.stage_run_id) is None

    resumed_runner, _job_store, _trace_store = _runner(tmp_path)
    summary = resumed_runner.resume(spec.job_id)

    assert summary.job_status is JobStatus.SUCCEEDED
    assert summary.executed_count == 1
    assert trace_store.load(summary.item_summaries[0].output_refs[0].object_id)


def test_runner_cancel_preserves_completed_output_and_blocks_resume(tmp_path: Path) -> None:
    first_trace = _write_trace(tmp_path / "first.jsonl", "NeedleFirst")
    second_trace = _write_trace(tmp_path / "second.jsonl", "NeedleSecond")
    spec = _spec(tmp_path, first_trace, second_trace)
    runner, job_store, _trace_store = _runner(tmp_path)
    job_store.create_job(spec)
    job_store.transition_job(
        spec.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="manual-job-running",
    )
    item = job_store.create_item(
        spec.job_id,
        "item://manual/cancel",
        idempotency_key="manual-create-item-cancel",
    )
    stage = job_store.create_stage_run(
        stage_run_id="stage-run://manual/cancel",
        job_id=spec.job_id,
        item_id=item.item_id,
        stage=StageNameV2.TRACE_INDEX,
        attempt=1,
        principal_ref=_ref("stage-principal"),
        input_refs=(_ref("trace-source"),),
        idempotency_key="manual-create-stage-cancel",
    )
    job_store.transition_stage_run(
        stage.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=stage.row_version,
        idempotency_key="manual-start-stage-cancel",
    )

    summary = runner.cancel(spec.job_id)

    assert summary.job_status is JobStatus.CANCELLED
    assert summary.cancelled is True
    assert job_store.get_stage_result_for_run(stage.stage_run_id).status is StageRunStatus.CANCELLED
    with pytest.raises(TraceRunnerCancelledError):
        runner.resume(spec.job_id)
