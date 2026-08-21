from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.model_control_v2 import ModelUsageSourceV2, ModelUsageV2
from eval_factory.contracts.observability_v2 import (
    BatchAuditEventKindV2,
    BatchAuditEventV2,
    BatchAuditFindingCodeV2,
    BatchAuditFindingV2,
    BatchAuditOutcomeV2,
    BatchAuditReportV2,
    BatchMetricsSnapshotV2,
    CacheUseV2,
    CostObservationV2,
    ItemMetricsSnapshotV2,
    MetricAvailabilityV2,
    MetricScopeV2,
    MetricSliceV2,
    ToolMetricV2,
    WorkAttemptMetricsV2,
    batch_audit_event_v2_ref,
    batch_metrics_snapshot_v2_ref,
    item_metrics_snapshot_v2_ref,
    work_attempt_metrics_v2_ref,
)
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkLeaseEventKindV2,
    WorkUnitScopeV2,
)
from eval_factory.contracts.resource_v2 import NonModelResourceVectorV2, ResourceUsageV2

HASH = "a" * 64
NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _ref(object_type: str, suffix: str, *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-06/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _audit(*refs: ObjectRef, created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r6-06-contract-test",
        governing_versions=(
            VersionBinding(component="batch-observability", version="batch-observability/r6-06-v1"),
        ),
        input_refs=tuple(sorted(refs, key=_key)),
    )


def _model_usage(*, source: ModelUsageSourceV2 = ModelUsageSourceV2.REPORTED) -> ModelUsageV2:
    return ModelUsageV2(
        requests=1,
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=20,
        charged_tokens=180,
        source=source,
    )


def _resource_usage() -> ResourceUsageV2:
    return ResourceUsageV2(
        observed=NonModelResourceVectorV2(
            processes=1,
            renderers=0,
            network_requests=2,
            storage_bytes=512,
        )
    )


def _attempt(*, created_at: datetime = NOW) -> WorkAttemptMetricsV2:
    graph_ref = _ref("resolved-job-work-graph", "graph")
    unit_ref = _ref("resolved-work-unit", "unit")
    lease_ref = _ref("work-lease", "lease")
    stage_run_ref = _ref("stage-run", "run", version="identity/v1")
    profile_ref = _ref("model-profile", "model", version="v1")
    telemetry_ref = _ref("execution-telemetry", "telemetry")
    result_ref = _ref("model-control-receipt", "receipt")
    output_ref = _ref("semantic-residual-result", "output")
    return WorkAttemptMetricsV2.create(
        resolved_job_work_graph_ref=graph_ref,
        work_unit_ref=unit_ref,
        work_lease_ref=lease_ref,
        stage_run_ref=stage_run_ref,
        job_id="job://r6-06/metrics",
        item_id="item://r6-06/metrics",
        scope=WorkUnitScopeV2.ITEM,
        stage=StageNameV2.LABEL,
        artifact_execution_group_ref=None,
        attempt=1,
        fencing_token=3,
        terminal_lease_version=1,
        terminal_event_kind=WorkLeaseEventKindV2.SUCCEEDED,
        terminal_status=StageRunStatus.SUCCEEDED,
        reason_code=None,
        ready_at=NOW,
        eligible_at=NOW + timedelta(seconds=1),
        acquired_at=NOW + timedelta(seconds=3),
        completed_at=NOW + timedelta(seconds=8),
        model_profile_ref=profile_ref,
        model_usage=_model_usage(),
        resource_usage=_resource_usage(),
        telemetry_ref=telemetry_ref,
        tool_metrics=(ToolMetricV2(tool_family="SEARCH", calls=2),),
        cost=CostObservationV2.reported(amount_microusd=1_250, source_ref=result_ref),
        selected_route_kind="RUNTIME",
        selected_route_id="claude-agent-sdk",
        selected_route_version="1.2.3",
        worker_version="worker-v1",
        retry_of_attempt_metrics_ref=None,
        result_refs=(result_ref,),
        output_refs=(output_ref,),
        artifact_slices=(),
        audit=_audit(
            graph_ref,
            unit_ref,
            lease_ref,
            stage_run_ref,
            profile_ref,
            telemetry_ref,
            result_ref,
            output_ref,
            created_at=created_at,
        ),
    )


def _recreate_attempt(
    value: WorkAttemptMetricsV2,
    **updates: object,
) -> WorkAttemptMetricsV2:
    inputs: dict[str, object] = {
        "resolved_job_work_graph_ref": value.resolved_job_work_graph_ref,
        "work_unit_ref": value.work_unit_ref,
        "work_lease_ref": value.work_lease_ref,
        "stage_run_ref": value.stage_run_ref,
        "job_id": value.job_id,
        "item_id": value.item_id,
        "scope": value.scope,
        "stage": value.stage,
        "artifact_execution_group_ref": value.artifact_execution_group_ref,
        "attempt": value.attempt,
        "fencing_token": value.fencing_token,
        "terminal_lease_version": value.terminal_lease_version,
        "terminal_event_kind": value.terminal_event_kind,
        "terminal_status": value.terminal_status,
        "reason_code": value.reason_code,
        "ready_at": value.ready_at,
        "eligible_at": value.eligible_at,
        "acquired_at": value.acquired_at,
        "completed_at": value.completed_at,
        "model_profile_ref": value.model_profile_ref,
        "model_usage": value.model_usage,
        "resource_usage": value.resource_usage,
        "telemetry_ref": value.telemetry_ref,
        "tool_metrics": value.tool_metrics,
        "cost": value.cost,
        "selected_route_kind": value.selected_route_kind,
        "selected_route_id": value.selected_route_id,
        "selected_route_version": value.selected_route_version,
        "worker_version": value.worker_version,
        "retry_of_attempt_metrics_ref": value.retry_of_attempt_metrics_ref,
        "result_refs": value.result_refs,
        "output_refs": value.output_refs,
        "artifact_slices": value.artifact_slices,
        "audit": value.audit,
    }
    inputs.update(updates)
    return WorkAttemptMetricsV2.create(**inputs)  # type: ignore[arg-type]


def test_attempt_metrics_derive_exact_timing_cache_and_identity() -> None:
    first = _attempt()
    second = _attempt(created_at=NOW - timedelta(days=1))

    assert first.queue_wait_us == 2_000_000
    assert first.execution_duration_us == 5_000_000
    assert first.total_duration_us == 8_000_000
    assert first.cache_use is CacheUseV2.HIT
    assert first.work_attempt_metrics_id == second.work_attempt_metrics_id
    assert first.work_attempt_metrics_sha256 == second.work_attempt_metrics_sha256
    assert work_attempt_metrics_v2_ref(first).object_sha256 == first.work_attempt_metrics_sha256
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WorkAttemptMetricsV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "prompt": "do not persist",
            }
        )


def test_attempt_metrics_reject_clock_regression_and_false_cost() -> None:
    value = _attempt()
    with pytest.raises(ValueError, match="timestamp"):
        _recreate_attempt(
            value,
            completed_at=value.acquired_at - timedelta(microseconds=1),
        )
    with pytest.raises(ValidationError, match="availability"):
        CostObservationV2.model_validate(
            {
                "availability": MetricAvailabilityV2.UNAVAILABLE,
                "currency": "USD",
                "amount_microusd": 0,
                "source_ref": None,
                "rounding": None,
            }
        )


def test_conservative_usage_makes_cache_unavailable() -> None:
    value = _attempt()
    conservative = ModelUsageV2.conservative(requests=1, tokens=180)
    rebuilt = _recreate_attempt(value, model_usage=conservative)
    assert rebuilt.cache_use is CacheUseV2.UNAVAILABLE


def test_audit_event_is_content_free_and_has_stable_source_identity() -> None:
    source_ref = _ref("work-lease-event", "terminal")
    related_ref = work_attempt_metrics_v2_ref(_attempt())
    event = BatchAuditEventV2.create(
        job_id="job://r6-06/metrics",
        item_id="item://r6-06/metrics",
        work_unit_ref=_ref("resolved-work-unit", "unit"),
        stage=StageNameV2.LABEL,
        artifact_id=None,
        attempt=1,
        event_kind=BatchAuditEventKindV2.WORK_LEASE_TERMINAL,
        status=StageRunStatus.SUCCEEDED.value,
        reason_code=None,
        occurred_at=NOW,
        source_family="WORK_LEASE_EVENT",
        source_version=1,
        source_ref=source_ref,
        outbox_event_id="outbox-event://r6-06/terminal",
        related_refs=(related_ref,),
        result_refs=(related_ref,),
        audit=_audit(source_ref, related_ref),
    )

    assert batch_audit_event_v2_ref(event).object_sha256 == event.batch_audit_event_sha256
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BatchAuditEventV2.model_validate(
            {
                **event.model_dump(mode="python"),
                "raw_exception": "secret",
            }
        )


def test_item_batch_snapshots_and_complete_report_bind_exact_sets() -> None:
    attempt = _attempt()
    attempt_ref = work_attempt_metrics_v2_ref(attempt)
    source_ref = _ref("work-lease-event", "terminal")
    event = BatchAuditEventV2.create(
        job_id=attempt.job_id,
        item_id=attempt.item_id,
        work_unit_ref=attempt.work_unit_ref,
        stage=attempt.stage,
        artifact_id=None,
        attempt=attempt.attempt,
        event_kind=BatchAuditEventKindV2.WORK_LEASE_TERMINAL,
        status=attempt.terminal_status.value,
        reason_code=None,
        occurred_at=attempt.completed_at,
        source_family="WORK_LEASE_EVENT",
        source_version=1,
        source_ref=source_ref,
        outbox_event_id="outbox-event://r6-06/terminal",
        related_refs=(attempt_ref,),
        result_refs=(attempt_ref,),
        audit=_audit(source_ref, attempt_ref),
    )
    event_ref = batch_audit_event_v2_ref(event)
    metric_slice = MetricSliceV2(
        scope=MetricScopeV2.ITEM,
        scope_id=attempt.item_id or "item://r6-06/missing",
        work_units=1,
        terminal_attempts=1,
        succeeded_attempts=1,
        retryable_attempts=0,
        terminal_non_success_attempts=0,
        cancelled_attempts=0,
        retry_attempts=0,
        resume_attempts=0,
        model_requests=1,
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=10,
        charged_tokens=180,
        reported_usage_attempts=1,
        conservative_usage_attempts=0,
        cache_hit_attempts=1,
        cache_miss_attempts=0,
        cache_unavailable_attempts=0,
        process_starts=0,
        renderer_operations=0,
        network_requests=0,
        retained_storage_bytes=0,
        tool_calls=2,
        reported_cost_microusd=1_250,
        unavailable_cost_attempts=0,
        backpressure_events=0,
        error_events=0,
        queue_wait_samples_us=(2_000_000,),
        execution_samples_us=(5_000_000,),
        total_samples_us=(8_000_000,),
    )
    item_ref = _ref("item-record", "metrics", version="record/v1")
    item = ItemMetricsSnapshotV2.create(
        subject_ref=item_ref,
        predecessor_ref=None,
        source_fingerprint="b" * 64,
        attempt_metrics_refs=(attempt_ref,),
        audit_event_refs=(event_ref,),
        metric_slices=(metric_slice,),
        audit=_audit(item_ref, attempt_ref, event_ref),
    )
    item_metrics_ref = item_metrics_snapshot_v2_ref(item)
    graph_ref = attempt.resolved_job_work_graph_ref
    batch = BatchMetricsSnapshotV2.create(
        subject_ref=graph_ref,
        predecessor_ref=None,
        source_fingerprint="b" * 64,
        attempt_metrics_refs=(attempt_ref,),
        audit_event_refs=(event_ref,),
        item_metrics_refs=(item_metrics_ref,),
        metric_slices=(
            metric_slice.model_copy(update={"scope": MetricScopeV2.JOB, "scope_id": attempt.job_id}),
        ),
        audit=_audit(graph_ref, attempt_ref, event_ref, item_metrics_ref),
    )
    batch_ref = batch_metrics_snapshot_v2_ref(batch)
    report = BatchAuditReportV2.create(
        resolved_job_work_graph_ref=graph_ref,
        source_fingerprint="b" * 64,
        outcome=BatchAuditOutcomeV2.COMPLETE,
        expected_work_unit_refs=(attempt.work_unit_ref,),
        audit_event_refs=(event_ref,),
        attempt_metrics_refs=(attempt_ref,),
        item_metrics_refs=(item_metrics_ref,),
        batch_metrics_ref=batch_ref,
        findings=(),
        audit=_audit(
            graph_ref,
            attempt.work_unit_ref,
            event_ref,
            attempt_ref,
            item_metrics_ref,
            batch_ref,
        ),
    )

    assert report.outcome is BatchAuditOutcomeV2.COMPLETE
    assert report.findings == ()
    with pytest.raises(ValidationError, match="COMPLETE"):
        BatchAuditReportV2.create(
            resolved_job_work_graph_ref=graph_ref,
            source_fingerprint="b" * 64,
            outcome=BatchAuditOutcomeV2.COMPLETE,
            expected_work_unit_refs=(attempt.work_unit_ref,),
            audit_event_refs=(event_ref,),
            attempt_metrics_refs=(attempt_ref,),
            item_metrics_refs=(item_metrics_ref,),
            batch_metrics_ref=batch_ref,
            findings=(
                BatchAuditFindingV2(
                    code=BatchAuditFindingCodeV2.MISSING_OUTBOX_WITNESS,
                    subject_refs=(source_ref,),
                ),
            ),
            audit=_audit(
                graph_ref,
                attempt.work_unit_ref,
                event_ref,
                attempt_ref,
                item_metrics_ref,
                batch_ref,
                source_ref,
            ),
        )
