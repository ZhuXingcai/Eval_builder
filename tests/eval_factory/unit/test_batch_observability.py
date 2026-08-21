from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from test_artifact_execution import (
    _audit as _artifact_audit,
)
from test_artifact_execution import (
    _ControlledExecutionFacade,
    _definition,
    _execution_plan,
)
from test_job_store import _audit
from test_model_control import (
    _Clock as _ModelClock,
)
from test_model_control import (
    _demand as _model_demand,
)
from test_model_control import (
    _prepared as _model_prepared,
)
from test_model_control import (
    _ref as _model_ref,
)
from test_resource_control import (
    _Clock as _ResourceClock,
)
from test_resource_control import (
    _demand as _resource_demand,
)
from test_resource_control import (
    _prepared as _resource_prepared,
)
from test_work_control import HASH, _holder, _prepared
from test_work_fanout import _planning_result

from env_mock_agent.facade import (
    ExecutionTelemetryV2,
    ReportedCostV2,
    TelemetryAvailabilityV2,
    ToolFamilyCountV2,
    ToolFamilyV2,
    attachment_execution_result_carried_sha256,
)
from eval_factory.attachment_planning import ArtifactGroupExecutor
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.model_control_v2 import (
    ModelUsageSourceV2,
    ModelUsageV2,
    ProviderBackpressureKindV2,
)
from eval_factory.contracts.observability_v2 import (
    BatchAuditEventKindV2,
    BatchAuditFindingCodeV2,
    BatchAuditOutcomeV2,
    MetricScopeV2,
    batch_audit_report_v2_ref,
    batch_metrics_snapshot_v2_ref,
    work_attempt_metrics_v2_ref,
)
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import WorkLeaseStateV2
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
    ResourceUsageV2,
)
from eval_factory.orchestration import (
    ArtifactGroupFanoutCompiler,
    BatchObservabilityIntegrityError,
    BatchObservabilityService,
    ImmutableResultError,
    JobStore,
    WorkLeaseHeadRecord,
    WorkReadinessEvaluator,
)

GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/orchestration"
    / "r6-06-batch-observability-v1.json"
)


def _complete_one_attempt(tmp_path: Path):
    _clock, store, control, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()
    lease = control.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-06-attempt",
    )
    control.complete_stage(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(
            ObjectRef(
                object_type="trace-envelope",
                object_id="trace-envelope://r6-06/success",
                object_version="v2",
                object_sha256=HASH,
            ),
        ),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-r6-06-attempt",
    )
    return store, graph, lease


class _ReportedTelemetryExecutionFacade(_ControlledExecutionFacade):
    @staticmethod
    def _result(request, *, status):
        base = _ControlledExecutionFacade._result(
            request,
            status=status,
        )
        telemetry = ExecutionTelemetryV2.create(
            duration_us=8_000,
            duration_availability=TelemetryAvailabilityV2.REPORTED,
            tool_family_counts=(
                ToolFamilyCountV2(
                    tool_family=ToolFamilyV2.FILE,
                    calls=1,
                ),
                ToolFamilyCountV2(
                    tool_family=ToolFamilyV2.SEARCH,
                    calls=2,
                ),
            ),
            tool_availability=TelemetryAvailabilityV2.DERIVED,
            reported_cost=ReportedCostV2.reported(
                amount_microusd=750,
            ),
            observed_at=base.telemetry.observed_at,
        )
        pending = base.model_copy(
            update={
                "execution_result_id": ("attachment-execution-result://pending"),
                "telemetry": telemetry,
                "execution_result_sha256": "0" * 64,
            }
        )
        digest = attachment_execution_result_carried_sha256(pending)
        return pending.model_copy(
            update={
                "execution_result_id": (f"attachment-execution-result://sha256/{digest}"),
                "execution_result_sha256": digest,
            }
        )


async def _complete_artifact_attempt(tmp_path: Path):
    _clock, store, control, graph, _unit, _readiness, policy = _prepared(tmp_path)
    parent = next(unit for unit in graph.work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact-observability",
        (("artifact://reported", "inputs/reported.txt"),),
        (_definition("artifact://reported"),),
    )
    planning = _planning_result(execution_plan)
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=planning,
        audit=_audit(),
    )
    store.create_artifact_group_fanout(
        fanout,
        idempotency_key="create-r6-06-artifact-fanout",
    )
    group = fanout.group_work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=group,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        audit=_audit(),
    )
    store.record_work_readiness(
        readiness,
        idempotency_key="ready-r6-06-artifact",
    )
    holder = _holder("r6-06-artifact")
    lease = control.acquire(
        graph=graph,
        work_unit=group,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-06-artifact",
    )
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=_ReportedTelemetryExecutionFacade(
            expected_parallelism=1,
        ),
        audit=_artifact_audit(),
    )
    control.complete_artifact_group(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        fanout=fanout,
        artifact_execution_planning_result=planning,
        artifact_execution_batch=batch,
        audit=_audit(),
        idempotency_key="complete-r6-06-artifact",
    )
    return store, graph, lease


def test_refresh_is_idempotent_and_rebuilds_complete_observability(
    tmp_path: Path,
) -> None:
    store, graph, _lease = _complete_one_attempt(tmp_path)
    service = BatchObservabilityService(store)

    first = service.refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06",
    )
    replay = service.refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06",
    )
    rebuilt = service.rebuild_job(graph.job_id)

    assert replay == first
    assert rebuilt == first
    assert service.list_audit_events(graph.job_id) == first.audit_events
    assert service.list_attempt_metrics(graph.job_id) == (first.attempt_metrics)
    assert service.get_item_metrics(graph.item_ids[0]) == (first.item_metrics[0])
    assert service.get_batch_metrics(graph.job_id) == (first.batch_metrics)
    assert service.get_audit_report(graph.job_id) == first.report
    assert first.report.outcome is BatchAuditOutcomeV2.COMPLETE
    assert first.report.findings == ()
    assert len(first.attempt_metrics) == 1
    assert {event.event_kind for event in first.audit_events} >= {
        BatchAuditEventKindV2.WORK_GRAPH_CREATED,
        BatchAuditEventKindV2.WORK_READINESS_RECORDED,
        BatchAuditEventKindV2.WORK_DISPATCHED,
        BatchAuditEventKindV2.WORK_LEASE_ACQUIRED,
        BatchAuditEventKindV2.WORK_LEASE_TERMINAL,
        BatchAuditEventKindV2.STAGE_RUN_CREATED,
        BatchAuditEventKindV2.STAGE_RUN_COMPLETED,
        BatchAuditEventKindV2.ATTEMPT_METRICS_RECORDED,
    }
    job_slice = next(value for value in first.batch_metrics.metric_slices if value.scope is MetricScopeV2.JOB)
    assert job_slice.work_units == len(graph.work_units)
    assert job_slice.terminal_attempts == 1
    assert job_slice.succeeded_attempts == 1
    assert job_slice.unavailable_cost_attempts == 1
    assert {
        value.scope_id for value in first.batch_metrics.metric_slices if value.scope is MetricScopeV2.STAGE
    } == {unit.stage.value for unit in graph.work_units}


def test_missing_terminal_metrics_yields_typed_incomplete_report(
    tmp_path: Path,
) -> None:
    store, graph, lease = _complete_one_attempt(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM work_attempt_metrics WHERE work_lease_id = ?",
            (lease.work_lease_id,),
        )

    result = BatchObservabilityService(store).refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-missing-metrics",
    )

    assert result.report.outcome is BatchAuditOutcomeV2.INCOMPLETE
    assert {finding.code for finding in result.report.findings} == {
        BatchAuditFindingCodeV2.MISSING_TERMINAL_METRICS
    }


def test_refresh_rejects_lease_head_drift_from_deleted_source_event(
    tmp_path: Path,
) -> None:
    store, graph, lease = _complete_one_attempt(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM work_lease_events WHERE work_lease_id = ?",
            (lease.work_lease_id,),
        )

    with pytest.raises(ImmutableResultError, match="lease head"):
        BatchObservabilityService(store).refresh_job(
            job_id=graph.job_id,
            audit=_audit(),
            idempotency_key="refresh-r6-06-missing-event",
        )


def test_orphan_metrics_without_terminal_event_are_incomplete(
    tmp_path: Path,
) -> None:
    store, graph, lease = _complete_one_attempt(tmp_path)
    active_head = WorkLeaseHeadRecord(
        work_lease_id=lease.work_lease_id,
        work_unit_id=lease.work_unit_ref.object_id,
        state=WorkLeaseStateV2.ACTIVE,
        lease_version=0,
        fencing_token=lease.fencing_token,
        expires_at=lease.expires_at,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM work_lease_events WHERE work_lease_id = ?",
            (lease.work_lease_id,),
        )
        connection.execute(
            """
            UPDATE work_lease_heads
            SET state = ?, lease_version = ?, expires_at = ?,
                record_json = ?
            WHERE work_lease_id = ?
            """,
            (
                active_head.state.value,
                active_head.lease_version,
                active_head.expires_at.isoformat(),
                store._record_json(active_head),
                lease.work_lease_id,
            ),
        )

    result = BatchObservabilityService(store).refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-orphan-metrics",
    )

    assert result.report.outcome is BatchAuditOutcomeV2.INCOMPLETE
    assert BatchAuditFindingCodeV2.UNEXPECTED_SOURCE_RECORD in {
        finding.code for finding in result.report.findings
    }


@pytest.mark.asyncio
async def test_artifact_metrics_roll_up_exact_output_cost_and_tool_families(
    tmp_path: Path,
) -> None:
    store, graph, lease = await _complete_artifact_attempt(
        tmp_path,
    )
    result = BatchObservabilityService(store).refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-artifact",
    )

    metrics = store.get_work_attempt_metrics_for_lease(
        lease.work_lease_id,
    )
    artifact = metrics.artifact_slices[0]
    assert artifact.output_ref is not None
    assert metrics.output_refs == (artifact.output_ref,)
    job_slice = next(
        value for value in result.batch_metrics.metric_slices if value.scope is MetricScopeV2.JOB
    )
    assert job_slice.reported_cost_microusd == 750
    assert job_slice.unavailable_cost_attempts == 0
    assert job_slice.tool_calls == 3
    file_slice = next(
        value
        for value in result.batch_metrics.metric_slices
        if value.scope is MetricScopeV2.TOOL_FAMILY and value.scope_id == ToolFamilyV2.FILE.value
    )
    search_slice = next(
        value
        for value in result.batch_metrics.metric_slices
        if value.scope is MetricScopeV2.TOOL_FAMILY and value.scope_id == ToolFamilyV2.SEARCH.value
    )
    assert file_slice.tool_calls == 1
    assert search_slice.tool_calls == 2
    artifact_slice = next(
        value for value in result.batch_metrics.metric_slices if value.scope is MetricScopeV2.ARTIFACT
    )
    assert artifact_slice.reported_cost_microusd == 750
    assert artifact_slice.model_requests == 0
    assert artifact_slice.process_starts == 0


def test_rebuild_rejects_missing_projection_row(tmp_path: Path) -> None:
    store, graph, _lease = _complete_one_attempt(tmp_path)
    service = BatchObservabilityService(store)
    result = service.refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-corruption",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM batch_audit_events WHERE batch_audit_event_id = ?",
            (result.audit_events[0].batch_audit_event_id,),
        )

    with pytest.raises(BatchObservabilityIntegrityError, match="projection"):
        service.rebuild_job(graph.job_id)


def test_refresh_fault_rolls_back_all_projection_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, graph, _lease = _complete_one_attempt(tmp_path)
    service = BatchObservabilityService(store)

    def fail_batch_snapshot(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected batch projection failure")

    monkeypatch.setattr(
        service,
        "_persist_batch_snapshot",
        fail_batch_snapshot,
    )
    with pytest.raises(RuntimeError, match="projection failure"):
        service.refresh_job(
            job_id=graph.job_id,
            audit=_audit(),
            idempotency_key="refresh-r6-06-fault",
        )

    with sqlite3.connect(store.path) as connection:
        for table in (
            "batch_audit_events",
            "item_metrics_snapshots",
            "batch_metrics_snapshots",
            "batch_audit_reports",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM idempotency_records
            WHERE scope = ?
            """,
            ("refresh-batch-observability:job://r6-05/control",),
        ).fetchone() == (0,)


def test_historical_store_reopen_adds_projection_tables(
    tmp_path: Path,
) -> None:
    store, _graph, _lease = _complete_one_attempt(tmp_path)
    JobStoreType = type(store)
    JobStoreType(store.path)

    with sqlite3.connect(store.path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
    assert {
        "batch_audit_events",
        "item_metrics_snapshots",
        "batch_metrics_snapshots",
        "batch_audit_reports",
    } <= tables


def test_model_usage_and_cache_are_aggregated_from_control_facts(
    tmp_path: Path,
) -> None:
    store = JobStore(
        tmp_path / "model.sqlite3",
        clock=_ModelClock(),
    )
    prepared = _model_prepared(
        store,
        job_id="job://r6-06/model-observability",
    )
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _model_demand(
        models,
        graph,
        unit,
        job_policy,
        pool,
        profile,
    )
    holder = _holder("r6-06-model")
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-06-model",
    )
    assert admitted.lease is not None
    assert admitted.reservation is not None
    usage = ModelUsageV2(
        requests=1,
        input_tokens=50,
        output_tokens=30,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=5,
        charged_tokens=95,
        source=ModelUsageSourceV2.REPORTED,
    )
    telemetry = ExecutionTelemetryV2.create(
        duration_us=12_000,
        duration_availability=TelemetryAvailabilityV2.REPORTED,
        tool_family_counts=(
            ToolFamilyCountV2(
                tool_family=ToolFamilyV2.SEARCH,
                calls=2,
            ),
        ),
        tool_availability=TelemetryAvailabilityV2.DERIVED,
        reported_cost=ReportedCostV2.reported(amount_microusd=1_250),
        observed_at=_ModelClock().value,
    )
    models.complete_stage(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(
            _model_ref(
                "model-control-receipt",
                "r6-06",
                version="v2",
            ),
        ),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        usage=usage,
        audit=_audit(),
        idempotency_key="complete-r6-06-model",
        telemetry=telemetry,
    )

    result = BatchObservabilityService(store).refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-model",
    )

    assert result.report.outcome is BatchAuditOutcomeV2.COMPLETE
    attempt = result.attempt_metrics[0]
    assert attempt.cost.amount_microusd == 1_250
    assert attempt.tool_metrics[0].tool_family == "SEARCH"
    assert attempt.tool_metrics[0].calls == 2
    kinds = {event.event_kind for event in result.audit_events}
    assert {
        BatchAuditEventKindV2.MODEL_POLICY_BOUND,
        BatchAuditEventKindV2.MODEL_ADMISSION_DECIDED,
        BatchAuditEventKindV2.MODEL_USAGE_RECORDED,
    } <= kinds
    job_slice = next(
        value for value in result.batch_metrics.metric_slices if value.scope is MetricScopeV2.JOB
    )
    assert job_slice.model_requests == 1
    assert job_slice.charged_tokens == 95
    assert job_slice.cache_hit_attempts == 1
    assert job_slice.reported_usage_attempts == 1
    assert job_slice.tool_calls == 2
    assert job_slice.reported_cost_microusd == 1_250
    actual_gold = {
        "schema_version": ("eval-factory/r6-06-batch-observability-gold/v1"),
        "claim_scope": ("PRIVATE_CONTENT_FREE_OPERATIONAL_OBSERVABILITY_ONLY"),
        "source_fingerprint": result.report.source_fingerprint,
        "report_ref": batch_audit_report_v2_ref(result.report).model_dump(mode="json"),
        "batch_metrics_ref": batch_metrics_snapshot_v2_ref(result.batch_metrics).model_dump(mode="json"),
        "attempt_metrics_ref": work_attempt_metrics_v2_ref(attempt).model_dump(mode="json"),
        "event_kinds": sorted(event.event_kind.value for event in result.audit_events),
        "job_metrics": {
            "terminal_attempts": job_slice.terminal_attempts,
            "model_requests": job_slice.model_requests,
            "charged_tokens": job_slice.charged_tokens,
            "cache_hit_attempts": job_slice.cache_hit_attempts,
            "tool_calls": job_slice.tool_calls,
            "reported_cost_microusd": (job_slice.reported_cost_microusd),
            "unavailable_cost_attempts": (job_slice.unavailable_cost_attempts),
        },
    }
    assert actual_gold == json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(actual_gold, sort_keys=True).lower()
    for forbidden in (
        "prompt",
        "model_output",
        "trace_text",
        "tool_arguments",
        "private_reference",
        "grader",
        "credential",
        "/private/",
    ):
        assert forbidden not in serialized


def test_resource_usage_is_aggregated_from_control_facts(
    tmp_path: Path,
) -> None:
    store = JobStore(
        tmp_path / "resource.sqlite3",
        clock=_ResourceClock(),
    )
    prepared = _resource_prepared(
        store,
        job_id="job://r6-06/resource-observability",
    )
    resources, graph, unit, readiness, control, job_policy, _pool = prepared
    demand = _resource_demand(resources, graph, unit, job_policy)
    holder = _holder("r6-06-resource")
    admitted = resources.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-06-resource",
    )
    assert admitted.lease is not None
    assert admitted.reservation is not None
    usage = ResourceUsageV2(
        observed=NonModelResourceVectorV2(
            processes=1,
            renderers=0,
            network_requests=0,
            storage_bytes=128,
        )
    )
    resources.complete_stage(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        usage=usage,
        audit=_audit(),
        idempotency_key="complete-r6-06-resource",
    )

    result = BatchObservabilityService(store).refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-resource",
    )

    assert result.report.outcome is BatchAuditOutcomeV2.COMPLETE
    kinds = {event.event_kind for event in result.audit_events}
    assert {
        BatchAuditEventKindV2.RESOURCE_POLICY_BOUND,
        BatchAuditEventKindV2.RESOURCE_ADMISSION_DECIDED,
        BatchAuditEventKindV2.RESOURCE_USAGE_RECORDED,
    } <= kinds
    job_slice = next(
        value for value in result.batch_metrics.metric_slices if value.scope is MetricScopeV2.JOB
    )
    assert job_slice.process_starts == 1
    assert job_slice.network_requests == 0
    assert job_slice.retained_storage_bytes == 128


def test_provider_backpressure_is_audited_with_retry_metrics(
    tmp_path: Path,
) -> None:
    clock = _ModelClock()
    store = JobStore(
        tmp_path / "backpressure.sqlite3",
        clock=clock,
    )
    prepared = _model_prepared(
        store,
        job_id="job://r6-06/backpressure-observability",
    )
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _model_demand(
        models,
        graph,
        unit,
        job_policy,
        pool,
        profile,
    )
    holder = _holder("r6-06-backpressure")
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-06-backpressure",
    )
    assert admitted.lease is not None
    assert admitted.reservation is not None
    usage = ModelUsageV2(
        requests=1,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=0,
        source=ModelUsageSourceV2.REPORTED,
    )
    models.complete_stage_backpressured(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        facade_receipt_ref=_model_ref(
            "model-control-receipt",
            "r6-06-429",
            version="v2",
        ),
        kind=ProviderBackpressureKindV2.RATE_LIMITED,
        eligible_at=clock.value + timedelta(seconds=1),
        usage=usage,
        audit=_audit(),
        idempotency_key="complete-r6-06-backpressure",
    )

    result = BatchObservabilityService(store).refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-backpressure",
    )

    assert result.report.outcome is BatchAuditOutcomeV2.COMPLETE
    assert any(
        event.event_kind is BatchAuditEventKindV2.MODEL_BACKPRESSURE_RECORDED for event in result.audit_events
    )
    job_slice = next(
        value for value in result.batch_metrics.metric_slices if value.scope is MetricScopeV2.JOB
    )
    assert job_slice.retryable_attempts == 1
    assert job_slice.backpressure_events == 1


@pytest.mark.asyncio
async def test_artifact_group_emits_artifact_event_and_scope_slice(
    tmp_path: Path,
) -> None:
    _clock, store, control, graph, _unit, _readiness, policy = _prepared(tmp_path / "artifact")
    parent = next(unit for unit in graph.work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact-plan",
        (("artifact://r6-06/a", "inputs/a.txt"),),
        (_definition("artifact://r6-06/a"),),
    )
    planning = _planning_result(execution_plan)
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=planning,
        audit=_audit(),
    )
    store.create_artifact_group_fanout(
        fanout,
        idempotency_key="create-r6-06-artifact-fanout",
    )
    group = fanout.group_work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=group,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        audit=_audit(),
    )
    store.record_work_readiness(
        readiness,
        idempotency_key="ready-r6-06-artifact-group",
    )
    holder = _holder("r6-06-artifact")
    lease = control.acquire(
        graph=graph,
        work_unit=group,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-06-artifact-group",
    )
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=_ControlledExecutionFacade(expected_parallelism=1),
        audit=_artifact_audit(),
    )
    control.complete_artifact_group(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        fanout=fanout,
        artifact_execution_planning_result=planning,
        artifact_execution_batch=batch,
        audit=_audit(),
        idempotency_key="complete-r6-06-artifact-group",
    )

    result = BatchObservabilityService(store).refresh_job(
        job_id=graph.job_id,
        audit=_audit(),
        idempotency_key="refresh-r6-06-artifact-group",
    )

    assert lease.stage_run_ref is None
    artifact_event = next(
        event
        for event in result.audit_events
        if event.event_kind is BatchAuditEventKindV2.ARTIFACT_RESULT_RECORDED
    )
    assert artifact_event.artifact_id == "artifact://r6-06/a"
    artifact_slice = next(
        value for value in result.batch_metrics.metric_slices if value.scope is MetricScopeV2.ARTIFACT
    )
    assert artifact_slice.scope_id == "artifact://r6-06/a"
    assert artifact_slice.succeeded_attempts == 1
    assert artifact_slice.unavailable_cost_attempts == 1
