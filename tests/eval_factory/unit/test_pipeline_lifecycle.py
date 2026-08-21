from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
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
    _resource_demand as _combined_resource_demand,
)
from test_resource_control import (
    _demand as _resource_demand,
)
from test_resource_control import (
    _prepared as _resource_prepared,
)
from test_work_control import _holder

from env_mock_agent.facade.model_control_v2 import (
    ModelInvocationCancellationOutcomeV2,
    ModelInvocationCancellationResultV2,
    model_invocation_cancellation_request_ref,
)
from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.cli_v2 import (
    PipelineCancellationModeV2,
    PipelineControlConfigV2,
    PipelineObservabilityAvailabilityV2,
    PipelineResumeActionV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.observability_v2 import (
    BatchAuditOutcomeV2,
    MetricScopeV2,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    JobStatus,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
)
from eval_factory.orchestration import (
    DatasetJobPlanCompiler,
    DatasetJobWorkGraphCompiler,
    IdempotencyConflictError,
    JobStore,
    WorkControlService,
)
from eval_factory.orchestration.lifecycle import (
    PipelineLifecyclePolicyError,
    PipelineLifecycleService,
    PipelineObservabilityRefreshRequiredError,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64


def _ref(object_type: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-07/lifecycle",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r6-07-lifecycle-test",
        governing_versions=(
            VersionBinding(
                component="eval-factory-spec",
                version="approved-v2",
            ),
        ),
    )


def _spec(
    *,
    job_id: str = "job://r6-07/lifecycle",
    idempotency_key: str = "create-r6-07-lifecycle",
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
        approval_policy_ref=_ref("user-approval-policy"),
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


def _control(
    *,
    max_attempts: int = 2,
    retry_delay_seconds: tuple[int, ...] = (5,),
) -> PipelineControlConfigV2:
    return PipelineControlConfigV2(
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        retry_lease_expiry=True,
    )


def _service(tmp_path: Path) -> PipelineLifecycleService:
    return PipelineLifecycleService(JobStore(tmp_path / "pipeline.sqlite3", clock=lambda: NOW))


def test_plan_and_create_are_deterministic_and_replay_safe(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    spec = _spec()

    planned = service.plan(job_spec=spec, audit=_audit())
    created = service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )
    replay = service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )

    assert planned.job_id == spec.job_id
    assert planned.resolved_stages == (StageNameV2.TRACE_INDEX,)
    assert planned.work_unit_count == 1
    assert created == replay
    assert created.job_status is JobStatus.CREATED
    assert created.item_count == 1
    assert created.work_unit_count == 1

    with pytest.raises(
        IdempotencyConflictError,
        match="idempotency",
    ):
        service.create(
            job_spec=spec,
            control=_control(
                max_attempts=3,
                retry_delay_seconds=(5, 10),
            ),
            audit=_audit(),
            idempotency_key=spec.idempotency_key,
        )
    with pytest.raises(
        PipelineLifecyclePolicyError,
        match="must match",
    ):
        service.create(
            job_spec=spec,
            control=_control(),
            audit=_audit(),
            idempotency_key="different-command-key",
        )


def test_create_recovers_a_committed_job_prefix(tmp_path: Path) -> None:
    service = _service(tmp_path)
    spec = _spec()
    plan = DatasetJobPlanCompiler().compile(
        job_spec=spec,
        audit=_audit(),
    )
    graph = DatasetJobWorkGraphCompiler().compile(
        job_spec=spec,
        resolved_plan=plan,
        audit=_audit(),
    )
    control_policy = WorkControlService(service.job_store).compile_policy(
        graph=graph,
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=2,
        retry_delay_seconds=(5,),
        retry_lease_expiry=True,
        audit=_audit(),
    )
    service.job_store.create_planned_job(spec, plan)

    created = service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )

    assert service.job_store.get_job_work_graph(spec.job_id) == graph
    assert service.job_store.get_work_control_policy(spec.job_id) == control_policy
    assert created.job_status is JobStatus.CREATED


def test_status_and_resume_do_not_execute_or_refresh_work(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    spec = _spec()
    created = service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )

    status = service.status(job_id=spec.job_id, offset=0, limit=100)
    resumed = service.resume(
        job_id=spec.job_id,
        expected_job_version=0,
        idempotency_key="resume-r6-07-created",
        offset=0,
        limit=100,
    )
    replay = service.resume(
        job_id=spec.job_id,
        expected_job_version=0,
        idempotency_key="resume-r6-07-created",
        offset=0,
        limit=100,
    )
    create_replay = service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )

    assert status.observability_availability is PipelineObservabilityAvailabilityV2.REFRESH_REQUIRED
    assert status.audit_report_ref is None
    assert status.terminal_attempt_count == 0
    assert resumed.action is PipelineResumeActionV2.STARTED
    assert resumed.status.job_status is JobStatus.RUNNING
    assert replay.action is PipelineResumeActionV2.ALREADY_RUNNING
    assert create_replay == created
    assert service.job_store.list_stage_runs(job_id=spec.job_id) == ()
    assert service.job_store.list_work_leases(job_id=spec.job_id) == ()


def test_blocked_resume_reopens_and_terminal_resume_is_rejected(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    spec = _spec()
    service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )
    running = service.job_store.transition_job(
        spec.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r6-07-blocked",
    )
    blocked = service.job_store.transition_job(
        spec.job_id,
        JobStatus.BLOCKED,
        expected_version=running.row_version,
        idempotency_key="block-r6-07-job",
    )

    reopened = service.resume(
        job_id=spec.job_id,
        expected_job_version=blocked.row_version,
        idempotency_key="resume-r6-07-blocked",
        offset=0,
        limit=100,
    )
    failed = service.job_store.transition_job(
        spec.job_id,
        JobStatus.FAILED,
        expected_version=reopened.status.job_version,
        idempotency_key="fail-r6-07-job",
    )

    assert reopened.action is PipelineResumeActionV2.REOPENED
    with pytest.raises(
        PipelineLifecyclePolicyError,
        match="terminal",
    ):
        service.resume(
            job_id=spec.job_id,
            expected_job_version=failed.row_version,
            idempotency_key="resume-r6-07-failed",
            offset=0,
            limit=100,
        )


def test_work_only_cancel_is_durable_and_replay_safe(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    spec = _spec()
    service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )

    cancelled = service.cancel(
        job_id=spec.job_id,
        reason_code="operator-requested",
        audit=_audit(),
        idempotency_key="cancel-r6-07-job",
    )
    replay = service.cancel(
        job_id=spec.job_id,
        reason_code="operator-requested",
        audit=_audit(),
        idempotency_key="cancel-r6-07-job",
    )

    assert cancelled == replay
    assert cancelled.mode is PipelineCancellationModeV2.WORK
    assert cancelled.pending_physical_acknowledgements == 0
    assert service.job_store.get_job(spec.job_id).status is JobStatus.CANCELLED


def test_model_only_cancel_returns_pending_acknowledgement(
    tmp_path: Path,
) -> None:
    job_id = "job://r6-07/model-cancel"
    store = JobStore(
        tmp_path / "model.sqlite3",
        clock=_ModelClock(),
    )
    prepared = _model_prepared(store, job_id=job_id)
    (
        models,
        graph,
        unit,
        readiness,
        control,
        job_policy,
        pool,
        profile,
    ) = prepared
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=_model_demand(
            models,
            graph,
            unit,
            job_policy,
            pool,
            profile,
        ),
        holder_ref=_holder("r6-07-model"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-07-model",
    )
    assert admitted.reservation is not None
    service = PipelineLifecycleService(store)

    cancelled = service.cancel(
        job_id=job_id,
        reason_code="operator-requested",
        audit=_audit(),
        idempotency_key="cancel-r6-07-model",
    )

    assert service.model_control.get_job_policy(job_id) == job_policy
    assert cancelled.job_id == job_id
    assert cancelled.mode is PipelineCancellationModeV2.MODEL
    assert len(cancelled.model_cancellation_request_refs) == 1
    assert cancelled.pending_physical_acknowledgements == 1

    domain = models.cancel_job(
        graph=graph,
        control_policy=control,
        job_policy=job_policy,
        reason_code="operator-requested",
        audit=_audit(),
        idempotency_key="cancel-r6-07-model",
    )
    request = domain.cancellation_requests[0]
    facade_request = models.to_facade_cancellation_request(request)
    models.record_cancellation(
        request=request,
        facade_result=ModelInvocationCancellationResultV2.create(
            cancellation_request_ref=(model_invocation_cancellation_request_ref(facade_request)),
            outcome=(ModelInvocationCancellationOutcomeV2.CANCELLED),
            completed_at=NOW,
        ),
        audit=_audit(),
        idempotency_key="ack-r6-07-model",
    )
    acknowledged = service.cancel(
        job_id=job_id,
        reason_code="operator-requested",
        audit=_audit(),
        idempotency_key="cancel-r6-07-model",
    )
    assert acknowledged.model_cancellation_request_refs == ()
    assert acknowledged.pending_physical_acknowledgements == 0


def test_resource_only_cancel_returns_pending_termination(
    tmp_path: Path,
) -> None:
    job_id = "job://r6-07/resource-cancel"
    store = JobStore(tmp_path / "resource.sqlite3")
    prepared = _resource_prepared(store, job_id=job_id)
    resources, graph, unit, readiness, control, job_policy, _pool = prepared
    admitted = resources.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=_resource_demand(
            resources,
            graph,
            unit,
            job_policy,
        ),
        holder_ref=_holder("r6-07-resource"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-07-resource",
    )
    assert admitted.reservation is not None
    service = PipelineLifecycleService(store)

    cancelled = service.cancel(
        job_id=job_id,
        reason_code="operator-requested",
        audit=_audit(),
        idempotency_key="cancel-r6-07-resource",
    )

    assert service.resource_control.get_job_policy(job_id) == job_policy
    assert cancelled.job_id == job_id
    assert cancelled.mode is PipelineCancellationModeV2.RESOURCE
    assert len(cancelled.resource_termination_request_refs) == 1
    assert cancelled.pending_physical_acknowledgements == 1


def test_combined_cancel_returns_both_pending_requests(
    tmp_path: Path,
) -> None:
    job_id = "job://r6-07/combined-cancel"
    store = JobStore(
        tmp_path / "combined.sqlite3",
        clock=_ModelClock(),
    )
    prepared = _model_prepared(store, job_id=job_id)
    (
        models,
        graph,
        unit,
        readiness,
        control,
        model_policy,
        pool,
        profile,
    ) = prepared
    _resources, resource_policy, resource_demand = _combined_resource_demand(
        store,
        graph,
        unit,
        process_capacity=1,
    )
    admitted = models.acquire_combined(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=model_policy,
        demand=_model_demand(
            models,
            graph,
            unit,
            model_policy,
            pool,
            profile,
        ),
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("r6-07-combined"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-07-combined",
    )
    assert admitted.model_reservation is not None
    assert admitted.resource_reservation is not None
    service = PipelineLifecycleService(store)

    cancelled = service.cancel(
        job_id=job_id,
        reason_code="operator-requested",
        audit=_audit(),
        idempotency_key="cancel-r6-07-combined",
    )

    assert service.model_control.get_job_policy(job_id) == model_policy
    assert service.resource_control.get_job_policy(job_id) == resource_policy
    assert cancelled.job_id == job_id
    assert cancelled.mode is PipelineCancellationModeV2.COMBINED
    assert len(cancelled.model_cancellation_request_refs) == 1
    assert len(cancelled.resource_termination_request_refs) == 1
    assert cancelled.pending_physical_acknowledgements == 2


def test_events_and_metrics_require_explicit_refresh(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    spec = _spec()
    service.create(
        job_spec=spec,
        control=_control(),
        audit=_audit(),
        idempotency_key=spec.idempotency_key,
    )

    with pytest.raises(PipelineObservabilityRefreshRequiredError):
        service.events(
            job_id=spec.job_id,
            offset=0,
            limit=100,
            refresh=False,
            audit=None,
            idempotency_key=None,
        )

    events = service.events(
        job_id=spec.job_id,
        offset=0,
        limit=100,
        refresh=True,
        audit=_audit(),
        idempotency_key="refresh-r6-07-events",
    )
    metrics = service.metrics(
        job_id=spec.job_id,
        item_id=None,
        scope=MetricScopeV2.JOB,
        offset=0,
        limit=100,
        refresh=False,
        audit=None,
        idempotency_key=None,
    )

    assert events.audit_outcome is BatchAuditOutcomeV2.COMPLETE
    assert events.finding_codes == ()
    assert metrics.audit_outcome is BatchAuditOutcomeV2.COMPLETE
    assert all(metric.scope is MetricScopeV2.JOB for metric in metrics.metric_slices)
    assert metrics.attempt_metrics_refs == ()
