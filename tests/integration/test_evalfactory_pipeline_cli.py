from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from eval_factory.cli import app
from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.cli_v2 import PipelineControlConfigV2
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    StageRunStatus,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
    WorkReadinessV2,
)
from eval_factory.orchestration import (
    JobStore,
    WorkControlService,
    WorkReadinessEvaluator,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64
FORBIDDEN_OUTPUT_KEYS = (
    "raw_trace",
    "prompt",
    "output_bytes",
    "tool_payload",
    "provider_payload",
    "credential",
    "private_reference",
    "grader_rule",
    "hidden_condition",
    "final_answer",
    "queue_wait_samples_us",
    "execution_samples_us",
    "total_samples_us",
)
GOLD_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals/golden/eval_factory/orchestration"
    / "r6-07-cli-lifecycle-v1.json"
)


def _ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-07/{suffix}",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r6-07-pipeline-cli-test",
        governing_versions=(
            VersionBinding(
                component="eval-factory-spec",
                version="approved-v2",
            ),
        ),
    )


def _spec(
    *,
    job_id: str,
    idempotency_key: str,
) -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id=job_id,
        traces=(
            TraceSourceRef(
                source_trace_id=f"source-trace://{job_id}",
                source_uri=f"raw-traj://{job_id}",
                raw_sha256=HASH,
                adapter_name="raw-traj-v1",
                adapter_version="v1",
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        ),
        requested_stages=(StageNameV2.TRACE_INDEX,),
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=0,
            max_model_tokens=0,
            max_processes=1,
            max_renderers=1,
            max_network_requests=1,
            max_storage_bytes=1024,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        approval_policy_ref=_ref(
            "user-approval-policy",
            job_id,
        ),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key=idempotency_key,
        audit=_audit(),
    )


def _write_inputs(
    tmp_path: Path,
    *,
    job_id: str = "job://r6-07/pipeline-cli",
    idempotency_key: str = "create-r6-07-pipeline-cli",
) -> tuple[Path, Path, DatasetJobSpecV2]:
    spec = _spec(
        job_id=job_id,
        idempotency_key=idempotency_key,
    )
    control = PipelineControlConfigV2(
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=2,
        retry_delay_seconds=(5,),
        retry_lease_expiry=True,
    )
    spec_path = tmp_path / "job-spec.json"
    control_path = tmp_path / "control.json"
    spec_path.write_text(
        spec.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    control_path.write_text(
        control.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return spec_path, control_path, spec


def _invoke(arguments: list[str]):
    result = CliRunner().invoke(app, arguments)
    payload = json.loads(result.output)
    _assert_content_free(result.output)
    return result, payload


def _assert_content_free(output: str) -> None:
    lowered = output.casefold()
    for forbidden in FORBIDDEN_OUTPUT_KEYS:
        assert forbidden not in lowered


def test_pipeline_cli_runs_all_lifecycle_commands(
    tmp_path: Path,
) -> None:
    spec_path, control_path, spec = _write_inputs(tmp_path)
    store_path = tmp_path / "pipeline.sqlite3"

    planned, plan_payload = _invoke(
        [
            "pipeline",
            "plan",
            "--spec",
            str(spec_path),
            "--json",
        ]
    )
    assert planned.exit_code == 0, planned.output
    assert plan_payload["job_id"] == spec.job_id
    assert plan_payload["work_unit_count"] == 1

    create_args = [
        "pipeline",
        "create",
        "--job-store",
        str(store_path),
        "--spec",
        str(spec_path),
        "--control",
        str(control_path),
        "--idempotency-key",
        spec.idempotency_key,
        "--json",
    ]
    created, create_payload = _invoke(create_args)
    replayed, replay_payload = _invoke(create_args)
    assert created.exit_code == 0, created.output
    assert replayed.exit_code == 0, replayed.output
    assert create_payload == replay_payload
    assert create_payload["job_status"] == "CREATED"

    status, status_payload = _invoke(
        [
            "pipeline",
            "status",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--offset",
            "0",
            "--limit",
            "1",
            "--json",
        ]
    )
    assert status.exit_code == 0, status.output
    assert status_payload["observability_availability"] == "REFRESH_REQUIRED"
    assert len(status_payload["work_units"]) == 1

    resumed, resume_payload = _invoke(
        [
            "pipeline",
            "resume",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--expected-job-version",
            "0",
            "--idempotency-key",
            "resume-r6-07-pipeline-cli",
            "--json",
        ]
    )
    assert resumed.exit_code == 0, resumed.output
    assert resume_payload["action"] == "STARTED"
    assert resume_payload["status"]["job_status"] == "RUNNING"
    assert JobStore(store_path).list_stage_runs(job_id=spec.job_id) == ()
    progressed_replay, progressed_payload = _invoke(create_args)
    assert progressed_replay.exit_code == 0
    assert progressed_payload == create_payload

    unavailable, unavailable_payload = _invoke(
        [
            "pipeline",
            "events",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--json",
        ]
    )
    assert unavailable.exit_code == 2
    assert unavailable_payload["error_code"] == "REFRESH_REQUIRED"

    events, events_payload = _invoke(
        [
            "pipeline",
            "events",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--refresh",
            "--idempotency-key",
            "refresh-r6-07-pipeline-cli",
            "--json",
        ]
    )
    assert events.exit_code == 0, events.output
    assert events_payload["audit_outcome"] == "COMPLETE"
    assert events_payload["events"]

    metrics, metrics_payload = _invoke(
        [
            "pipeline",
            "metrics",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--scope",
            "JOB",
            "--json",
        ]
    )
    assert metrics.exit_code == 0, metrics.output
    assert metrics_payload["audit_outcome"] == "COMPLETE"
    assert metrics_payload["attempt_metrics_refs"] == []
    assert all(value["scope"] == "JOB" for value in metrics_payload["metric_slices"])

    cancelled, cancel_payload = _invoke(
        [
            "pipeline",
            "cancel",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--reason-code",
            "operator-requested",
            "--idempotency-key",
            "cancel-r6-07-pipeline-cli",
            "--json",
        ]
    )
    assert cancelled.exit_code == 0, cancelled.output
    assert cancel_payload["job_id"] == spec.job_id
    assert cancel_payload["mode"] == "WORK"
    assert cancel_payload["pending_physical_acknowledgements"] == 0

    stale_events, stale_payload = _invoke(
        [
            "pipeline",
            "events",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--json",
        ]
    )
    assert stale_events.exit_code == 2
    assert stale_payload["error_code"] == "REFRESH_REQUIRED"

    terminal_resume, terminal_payload = _invoke(
        [
            "pipeline",
            "resume",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--expected-job-version",
            "2",
            "--idempotency-key",
            "resume-cancelled-r6-07-pipeline-cli",
            "--json",
        ]
    )
    assert terminal_resume.exit_code == 2
    assert terminal_payload["error_code"] == "POLICY_ERROR"


def test_pipeline_cli_preserves_incomplete_audit_truth(
    tmp_path: Path,
) -> None:
    spec_path, control_path, spec = _write_inputs(
        tmp_path,
        job_id="job://r6-07/incomplete",
        idempotency_key="create-r6-07-incomplete",
    )
    store_path = tmp_path / "incomplete.sqlite3"
    created, _payload = _invoke(
        [
            "pipeline",
            "create",
            "--job-store",
            str(store_path),
            "--spec",
            str(spec_path),
            "--control",
            str(control_path),
            "--idempotency-key",
            spec.idempotency_key,
        ]
    )
    assert created.exit_code == 0
    resumed, _payload = _invoke(
        [
            "pipeline",
            "resume",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--expected-job-version",
            "0",
            "--idempotency-key",
            "resume-r6-07-incomplete",
        ]
    )
    assert resumed.exit_code == 0

    store = JobStore(store_path, clock=lambda: NOW)
    graph = store.get_job_work_graph(spec.job_id)
    unit = graph.work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=unit,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )
    assert readiness.readiness is WorkReadinessV2.READY
    store.record_work_readiness(
        readiness,
        idempotency_key="ready-r6-07-incomplete",
    )
    control = WorkControlService(store)
    policy = store.get_work_control_policy(spec.job_id)
    holder = ObjectRef(
        object_type="worker-principal",
        object_id="worker-principal://r6-07/incomplete",
        object_version="v1",
        object_sha256=HASH,
    )
    lease = control.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-07-incomplete",
    )
    control.complete_stage(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(_ref("trace-envelope", "incomplete"),),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-r6-07-incomplete",
    )
    with sqlite3.connect(store_path) as connection:
        connection.execute(
            "DELETE FROM work_attempt_metrics WHERE job_id = ?",
            (spec.job_id,),
        )

    events, payload = _invoke(
        [
            "pipeline",
            "events",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--refresh",
            "--idempotency-key",
            "refresh-r6-07-incomplete",
        ]
    )

    assert events.exit_code == 0, events.output
    assert payload["audit_outcome"] == "INCOMPLETE"
    assert "MISSING_TERMINAL_METRICS" in payload["finding_codes"]


def test_pipeline_cli_returns_bounded_errors_for_missing_and_corrupt_state(
    tmp_path: Path,
) -> None:
    missing, missing_payload = _invoke(
        [
            "pipeline",
            "status",
            "--job-store",
            str(tmp_path / "missing.sqlite3"),
            "--job",
            "job://r6-07/missing",
        ]
    )
    assert missing.exit_code == 2
    assert missing_payload["error_code"] == "NOT_FOUND"

    spec_path, control_path, spec = _write_inputs(
        tmp_path,
        job_id="job://r6-07/corrupt",
        idempotency_key="create-r6-07-corrupt",
    )
    store_path = tmp_path / "corrupt.sqlite3"
    created, _payload = _invoke(
        [
            "pipeline",
            "create",
            "--job-store",
            str(store_path),
            "--spec",
            str(spec_path),
            "--control",
            str(control_path),
            "--idempotency-key",
            spec.idempotency_key,
        ]
    )
    assert created.exit_code == 0
    with sqlite3.connect(store_path) as connection:
        connection.execute("UPDATE work_control_policies SET record_json = '{}'")

    corrupt, corrupt_payload = _invoke(
        [
            "pipeline",
            "status",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
        ]
    )

    assert corrupt.exit_code == 2
    assert corrupt_payload["error_code"] == "INTEGRITY_ERROR"
    assert "sqlite" not in corrupt.output.casefold()
    assert str(store_path) not in corrupt.output


def test_root_cli_preserves_legacy_groups() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "pipeline" in result.output
    assert "trace" in result.output
    assert "run" in result.output


def test_pipeline_cli_gold_is_current(tmp_path: Path) -> None:
    expected = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert _gold_payload(tmp_path) == expected


def _gold_payload(tmp_path: Path) -> dict[str, object]:
    spec_path, control_path, spec = _write_inputs(
        tmp_path,
        job_id="job://r6-07/gold",
        idempotency_key="create-r6-07-gold",
    )
    store_path = tmp_path / "gold.sqlite3"
    _planned, plan = _invoke(
        [
            "pipeline",
            "plan",
            "--spec",
            str(spec_path),
        ]
    )
    _created, created = _invoke(
        [
            "pipeline",
            "create",
            "--job-store",
            str(store_path),
            "--spec",
            str(spec_path),
            "--control",
            str(control_path),
            "--idempotency-key",
            spec.idempotency_key,
        ]
    )
    _status_result, status = _invoke(
        [
            "pipeline",
            "status",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
        ]
    )
    _resumed, resumed = _invoke(
        [
            "pipeline",
            "resume",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--expected-job-version",
            "0",
            "--idempotency-key",
            "resume-r6-07-gold",
        ]
    )
    _events_result, events = _invoke(
        [
            "pipeline",
            "events",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--refresh",
            "--idempotency-key",
            "refresh-r6-07-gold",
        ]
    )
    _metrics_result, metrics = _invoke(
        [
            "pipeline",
            "metrics",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
        ]
    )
    _cancelled, cancelled = _invoke(
        [
            "pipeline",
            "cancel",
            "--job-store",
            str(store_path),
            "--job",
            spec.job_id,
            "--reason-code",
            "gold-cancelled",
            "--idempotency-key",
            "cancel-r6-07-gold",
        ]
    )
    payload: dict[str, object] = {
        "schema_version": "eval-factory/r6-07-cli-gold/v1",
        "plan": {
            key: plan[key]
            for key in (
                "job_id",
                "job_spec_ref",
                "resolved_plan_ref",
                "work_graph_ref",
                "resolved_stages",
                "work_unit_count",
                "work_units",
            )
        },
        "create": created,
        "status": {
            key: status[key]
            for key in (
                "job_id",
                "job_status",
                "job_version",
                "item_count",
                "work_unit_count",
                "terminal_attempt_count",
                "work_units",
                "observability_availability",
                "audit_outcome",
            )
        },
        "resume": {
            "action": resumed["action"],
            "job_status": resumed["status"]["job_status"],
            "job_version": resumed["status"]["job_version"],
            "work_unit_count": (resumed["status"]["work_unit_count"]),
            "work_units": resumed["status"]["work_units"],
        },
        "events": {
            "audit_outcome": events["audit_outcome"],
            "finding_codes": events["finding_codes"],
            "total_events": events["total_events"],
            "event_kinds": [value["event_kind"] for value in events["events"]],
        },
        "metrics": {
            "audit_outcome": metrics["audit_outcome"],
            "finding_codes": metrics["finding_codes"],
            "total_attempts": metrics["total_attempts"],
            "attempt_metrics_refs": metrics["attempt_metrics_refs"],
            "metric_slices": metrics["metric_slices"],
        },
        "cancel": cancelled,
    }
    _assert_content_free(json.dumps(payload, sort_keys=True))
    return payload
