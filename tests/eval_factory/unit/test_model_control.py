from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from test_artifact_execution import (
    _audit as _artifact_audit,
)
from test_artifact_execution import (
    _ControlledExecutionFacade,
    _definition,
    _execution_plan,
)
from test_job_store import _audit, _job_spec, _plan, _work_graph
from test_resource_control import _facade_termination_result
from test_work_control import _holder
from test_work_fanout import _planning_result

from env_mock_agent.facade.model_control_v2 import (
    ModelInvocationCancellationOutcomeV2,
    ModelInvocationCancellationResultV2,
    model_invocation_cancellation_request_ref,
)
from eval_factory.attachment_planning import ArtifactGroupExecutor
from eval_factory.contracts.attachment_v2 import (
    artifact_execution_batch_ref,
    artifact_execution_receipt_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.model_control_v2 import (
    ModelAdmissionOutcomeV2,
    ModelDemandModeV2,
    ModelRateDimensionV2,
    ModelUsageSourceV2,
    ModelUsageV2,
    ProviderBackpressureKindV2,
    WorkModelEventKindV2,
    WorkModelReservationStateV2,
)
from eval_factory.contracts.orchestration import JobStatus, ResourceBudget, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    WorkLeaseEventKindV2,
    WorkReadinessV2,
    resolved_work_unit_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
    ResourceAdmissionOutcomeV2,
    ResourceUsageV2,
    WorkResourceTerminationOutcomeV2,
)
from eval_factory.orchestration import (
    ArtifactGroupFanoutCompiler,
    ImmutableResultError,
    JobStore,
    WorkControlService,
    WorkReadinessEvaluator,
)
from eval_factory.orchestration.model_control import (
    RATE_CREDITS_PER_UNIT,
    ModelControlPolicyError,
    ModelControlService,
    _classify,
    _refill_pool_head,
)
from eval_factory.orchestration.models import (
    JobModelHeadRecord,
    ModelRatePoolHeadRecord,
)
from eval_factory.orchestration.resources import ResourceControlService

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)


@dataclass
class _Clock:
    value: datetime = NOW

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _ref(object_type: str, suffix: str, *, version: str = "v1") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-03/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _model_spec(job_id: str) -> DatasetJobSpecV2:
    source = _job_spec(job_id=job_id, idempotency_key=f"create-{job_id}")
    payload = source.model_dump(mode="python")
    payload["model_profiles"] = ("model-profile://r6-03/main",)
    payload["budget"] = ResourceBudget(
        max_model_requests=10,
        max_model_tokens=10_000,
        max_processes=source.budget.max_processes,
        max_renderers=source.budget.max_renderers,
        max_network_requests=source.budget.max_network_requests,
        max_storage_bytes=source.budget.max_storage_bytes,
    )
    return DatasetJobSpecV2.model_validate(payload)


def _prepared(
    store: JobStore,
    *,
    job_id: str,
    pool_policy=None,
    requests_per_minute: int = 60,
    request_burst: int = 2,
    max_concurrent_requests: int = 2,
):
    spec = _model_spec(job_id)
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(graph, idempotency_key=f"graph-{job_id}")
    created = store.get_job(job_id)
    store.transition_job(
        job_id,
        JobStatus.RUNNING,
        expected_version=created.row_version,
        idempotency_key=f"start-{job_id}",
    )
    unit = graph.work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=unit,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )
    assert readiness.readiness is WorkReadinessV2.READY
    store.record_work_readiness(readiness, idempotency_key=f"ready-{job_id}")
    control = WorkControlService(store)
    control_policy = control.bind_policy(
        graph=graph,
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=2,
        retry_delay_seconds=(0,),
        retry_lease_expiry=True,
        audit=_audit(),
        idempotency_key=f"control-{job_id}",
    )
    models = ModelControlService(store)
    profile_ref = _ref("model-profile", "main")
    if pool_policy is None:
        pool_policy = models.bind_pool_policy(
            provider_bucket_ref=_ref("provider-rate-bucket", "shared"),
            allowed_model_profile_refs=(profile_ref,),
            requests_per_minute=requests_per_minute,
            tokens_per_minute=60_000,
            request_burst=request_burst,
            token_burst=1_000,
            max_concurrent_requests=max_concurrent_requests,
            minimum_backpressure_seconds=5,
            audit=_audit(),
            idempotency_key="model-pool-shared",
        )
    job_policy = models.bind_job_policy(
        graph=graph,
        pool_policies=(pool_policy,),
        model_profile_refs=(profile_ref,),
        audit=_audit(),
        idempotency_key=f"job-model-policy-{job_id}",
    )
    return (
        models,
        graph,
        unit,
        readiness,
        control_policy,
        job_policy,
        pool_policy,
        profile_ref,
    )


def _demand(models, graph, unit, job_policy, pool_policy, profile_ref):
    return models.create_demand(
        graph=graph,
        work_unit=unit,
        job_policy=job_policy,
        pool_policy=pool_policy,
        model_profile_ref=profile_ref,
        operation_ref=_ref("semantic-residual-request", unit.resolved_work_unit_id, version="v2"),
        model_execution_profile_ref=_ref("model-execution-profile", "direct"),
        attempt=1,
        mode=ModelDemandModeV2.DIRECT_REQUEST,
        request_allowance=1,
        input_token_allowance=200,
        output_token_allowance=250,
        cache_token_allowance=50,
        concurrency_units=1,
        audit=_audit(),
    )


def _resource_demand(
    store: JobStore,
    graph,
    unit,
    *,
    process_capacity: int,
):
    resources = ResourceControlService(store)
    pool = resources.bind_pool_policy(
        resource_domain_ref=_ref(
            "resource-domain",
            f"combined-{process_capacity}",
        ),
        capacity=NonModelResourceVectorV2(
            processes=process_capacity,
            renderers=1,
            network_requests=1,
            storage_bytes=4096,
        ),
        audit=_audit(),
        idempotency_key=f"combined-resource-pool-{process_capacity}",
    )
    policy = resources.bind_job_policy(
        graph=graph,
        pool_policy=pool,
        audit=_audit(),
        idempotency_key=f"combined-resource-policy-{graph.job_id}",
    )
    demand = resources.create_demand(
        graph=graph,
        work_unit=unit,
        job_policy=policy,
        execution_profile_ref=_ref("resource-execution-profile", "combined"),
        attempt=1,
        capacity_units=NonModelResourceVectorV2(
            processes=1,
            renderers=0,
            network_requests=0,
            storage_bytes=1024,
        ),
        budget_allowance=NonModelResourceVectorV2(
            processes=1,
            renderers=0,
            network_requests=0,
            storage_bytes=1024,
        ),
        termination_required=True,
        audit=_audit(),
    )
    return resources, policy, demand


def test_model_admission_atomically_creates_lease_and_reservation(tmp_path: Path) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-03/admitted")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)

    result = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("model"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-model-r6-03",
    )
    replay = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("model"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-model-r6-03",
    )

    assert replay == result
    assert result.decision.outcome is ModelAdmissionOutcomeV2.ADMITTED
    assert result.lease is not None and result.reservation is not None
    assert result.lease.stage_run_ref is not None
    assert store.get_stage_run(result.lease.stage_run_ref.object_id).status is StageRunStatus.RUNNING
    pool_head = models.get_pool_head(pool.model_rate_pool_policy_id)
    assert pool_head.active_concurrency == 1
    assert pool_head.request_credits == 1 * 60_000_000
    assert pool_head.token_credits == 500 * 60_000_000
    assert (
        models.get_reservation_head(result.reservation.work_model_reservation_id).state
        is WorkModelReservationStateV2.ACTIVE
    )
    descriptor, grant = models.to_facade_controls(
        demand=demand,
        reservation=result.reservation,
        idempotency_key="invoke-model-r6-03",
    )
    assert descriptor.operation_ref.object_id == demand.operation_ref.object_id
    assert descriptor.model_profile_ref.object_id == demand.model_profile_ref.object_id
    assert descriptor.request_allowance == demand.request_allowance
    assert descriptor.token_allowance == demand.token_allowance
    assert grant.reservation_ref.object_id == result.reservation.work_model_reservation_id
    assert grant.fencing_token == result.reservation.fencing_token
    assert grant.output_token_limit == demand.output_token_allowance
    grant.validate_descriptor(descriptor)


def test_model_policy_binding_replays_and_rejects_conflicting_pool(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/policy-replay")
    models, graph, _unit, _readiness, _control, job_policy, pool, profile = prepared

    pool_replay = models.bind_pool_policy(
        provider_bucket_ref=_ref("provider-rate-bucket", "shared"),
        allowed_model_profile_refs=(profile,),
        requests_per_minute=60,
        tokens_per_minute=60_000,
        request_burst=2,
        token_burst=1_000,
        max_concurrent_requests=2,
        minimum_backpressure_seconds=5,
        audit=_audit(),
        idempotency_key="model-pool-shared",
    )
    job_replay = models.bind_job_policy(
        graph=graph,
        pool_policies=(pool,),
        model_profile_refs=(profile,),
        audit=_audit(),
        idempotency_key=f"job-model-policy-{graph.job_id}",
    )

    assert pool_replay == pool
    assert job_replay == job_policy
    with pytest.raises(ImmutableResultError, match="already has"):
        models.bind_pool_policy(
            provider_bucket_ref=_ref("provider-rate-bucket", "shared"),
            allowed_model_profile_refs=(profile,),
            requests_per_minute=61,
            tokens_per_minute=60_000,
            request_burst=2,
            token_burst=1_000,
            max_concurrent_requests=2,
            minimum_backpressure_seconds=5,
            audit=_audit(),
            idempotency_key="model-pool-conflict",
        )


def test_model_classification_covers_all_dimensions_and_clock_regression(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/classification")
    models, graph, unit, _readiness, _control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)
    pool_head = models.get_pool_head(pool.model_rate_pool_policy_id)
    job_head = models.get_job_head(job_policy.job_model_policy_id)

    assert _classify(
        pool_policy=pool,
        job_policy=job_policy,
        pool_head=pool_head,
        job_head=job_head,
        demand=demand,
        now=NOW,
    ) == (ModelAdmissionOutcomeV2.ADMITTED, (), None)
    unsatisfiable_tokens = _classify(
        pool_policy=pool,
        job_policy=job_policy,
        pool_head=pool_head,
        job_head=job_head,
        demand=demand.model_copy(update={"token_allowance": pool.token_burst + 1}),
        now=NOW,
    )
    assert unsatisfiable_tokens[0] is ModelAdmissionOutcomeV2.UNSATISFIABLE_DEMAND
    assert unsatisfiable_tokens[1] == (ModelRateDimensionV2.TOKENS,)
    unsatisfiable_concurrency = _classify(
        pool_policy=pool,
        job_policy=job_policy,
        pool_head=pool_head,
        job_head=job_head,
        demand=demand.model_copy(update={"concurrency_units": pool.max_concurrent_requests + 1}),
        now=NOW,
    )
    assert unsatisfiable_concurrency[1] == (ModelRateDimensionV2.CONCURRENCY,)
    exhausted = _classify(
        pool_policy=pool,
        job_policy=job_policy,
        pool_head=pool_head,
        job_head=JobModelHeadRecord(
            job_model_policy_id=job_policy.job_model_policy_id,
            active_concurrency=0,
            reserved_requests=0,
            reserved_tokens=0,
            consumed_requests=job_policy.max_model_requests,
            consumed_tokens=job_policy.max_model_tokens,
            row_version=0,
        ),
        demand=demand,
        now=NOW,
    )
    assert exhausted[0] is ModelAdmissionOutcomeV2.BUDGET_EXHAUSTED
    assert exhausted[1] == (
        ModelRateDimensionV2.REQUESTS,
        ModelRateDimensionV2.TOKENS,
    )
    blocked_until = NOW + timedelta(seconds=10)
    provider_wait = _classify(
        pool_policy=pool,
        job_policy=job_policy,
        pool_head=pool_head.model_copy(update={"blocked_until": blocked_until}),
        job_head=job_head,
        demand=demand,
        now=NOW,
    )
    assert provider_wait == (
        ModelAdmissionOutcomeV2.WAITING_PROVIDER,
        (ModelRateDimensionV2.PROVIDER_BACKPRESSURE,),
        blocked_until,
    )
    token_wait = _classify(
        pool_policy=pool,
        job_policy=job_policy,
        pool_head=pool_head.model_copy(update={"token_credits": 0}),
        job_head=job_head,
        demand=demand,
        now=NOW,
    )
    assert token_wait[0] is ModelAdmissionOutcomeV2.WAITING_RATE
    assert token_wait[1] == (ModelRateDimensionV2.TOKENS,)
    concurrency_wait = _classify(
        pool_policy=pool,
        job_policy=job_policy,
        pool_head=pool_head.model_copy(update={"active_concurrency": pool.max_concurrent_requests}),
        job_head=job_head,
        demand=demand,
        now=NOW,
    )
    assert concurrency_wait == (
        ModelAdmissionOutcomeV2.WAITING_CONCURRENCY,
        (ModelRateDimensionV2.CONCURRENCY,),
        None,
    )
    stale_head = ModelRatePoolHeadRecord(
        model_rate_pool_policy_id=pool.model_rate_pool_policy_id,
        request_credits=RATE_CREDITS_PER_UNIT,
        token_credits=RATE_CREDITS_PER_UNIT,
        active_concurrency=0,
        blocked_until=None,
        effective_at=NOW + timedelta(microseconds=1),
        bucket_version=0,
        row_version=0,
    )
    with pytest.raises(ImmutableResultError, match="backwards"):
        _refill_pool_head(stale_head, pool, NOW)


def test_rate_pressure_waits_without_lease_and_admits_at_exact_boundary(tmp_path: Path) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    first = _prepared(
        store,
        job_id="job://r6-03/rate-a",
        requests_per_minute=1,
        request_burst=1,
        max_concurrent_requests=2,
    )
    second = _prepared(
        store,
        job_id="job://r6-03/rate-b",
        pool_policy=first[6],
    )
    first_demand = _demand(first[0], first[1], first[2], first[5], first[6], first[7])
    second_demand = _demand(second[0], second[1], second[2], second[5], second[6], second[7])
    first[0].acquire(
        graph=first[1],
        work_unit=first[2],
        readiness_snapshot=first[3],
        control_policy=first[4],
        job_policy=first[5],
        demand=first_demand,
        holder_ref=_holder("rate-a"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-rate-a",
    )

    waiting = second[0].acquire(
        graph=second[1],
        work_unit=second[2],
        readiness_snapshot=second[3],
        control_policy=second[4],
        job_policy=second[5],
        demand=second_demand,
        holder_ref=_holder("rate-b"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="wait-rate-b",
    )
    assert waiting.decision.outcome is ModelAdmissionOutcomeV2.WAITING_RATE
    assert waiting.decision.eligible_at == NOW + timedelta(seconds=60)
    assert waiting.lease is None and waiting.reservation is None

    clock.advance(60)
    admitted = second[0].acquire(
        graph=second[1],
        work_unit=second[2],
        readiness_snapshot=second[3],
        control_policy=second[4],
        job_policy=second[5],
        demand=second_demand,
        holder_ref=_holder("rate-b"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-rate-b",
    )
    assert admitted.decision.outcome is ModelAdmissionOutcomeV2.ADMITTED


def test_unsatisfiable_demand_is_terminal_without_lease(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/unsatisfiable", request_burst=1)
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = models.create_demand(
        graph=graph,
        work_unit=unit,
        job_policy=job_policy,
        pool_policy=pool,
        model_profile_ref=profile,
        operation_ref=_ref("opaque-runtime-request", "too-large", version="v2"),
        model_execution_profile_ref=_ref("model-execution-profile", "runtime"),
        attempt=1,
        mode=ModelDemandModeV2.OPAQUE_RUNTIME_ENVELOPE,
        request_allowance=2,
        input_token_allowance=200,
        output_token_allowance=250,
        cache_token_allowance=50,
        concurrency_units=1,
        audit=_audit(),
    )

    result = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("unsatisfiable"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="unsatisfiable-model",
    )

    assert result.decision.outcome is ModelAdmissionOutcomeV2.UNSATISFIABLE_DEMAND
    assert result.lease is None and result.reservation is None
    unit_ref = resolved_work_unit_v2_ref(unit)
    downstream = next(
        candidate for candidate in graph.work_units if unit_ref in candidate.depends_on_work_unit_refs
    )
    terminal = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=downstream,
        stage_completions=(),
        item_records=(),
        model_admission_decisions=(result.decision,),
        work_model_demands=(demand,),
        audit=_audit(),
    )
    assert terminal.readiness is WorkReadinessV2.BLOCKED_DEPENDENCY
    assert (
        store.record_work_readiness(
            terminal,
            idempotency_key="terminal-model-readiness",
        )
        == terminal
    )


def test_completion_refunds_unused_rate_and_charges_actual_job_budget(tmp_path: Path) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-03/completed")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("completed"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-completed",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    usage = ModelUsageV2(
        requests=1,
        input_tokens=50,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=100,
        source=ModelUsageSourceV2.REPORTED,
    )
    receipt_ref = _ref("model-control-receipt", "completed", version="v2")
    with pytest.raises(ModelControlPolicyError, match="receipt"):
        models.complete_stage(
            lease=admitted.lease,
            control_policy=control,
            reservation=admitted.reservation,
            holder_ref=_holder("completed"),
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(_ref("semantic-residual-result", "completed", version="v2"),),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            usage=usage,
            audit=_audit(),
            idempotency_key="complete-model-missing-receipt",
        )
    completed = models.complete_stage(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=_holder("completed"),
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(
            _ref("semantic-residual-result", "completed", version="v2"),
            receipt_ref,
        ),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        usage=usage,
        audit=_audit(),
        idempotency_key="complete-model",
    )

    assert completed.model_event.usage == usage
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 0
    job_head = models.get_job_head(job_policy.job_model_policy_id)
    assert job_head.reserved_tokens == 0
    assert job_head.consumed_tokens == 100


def test_provider_backpressure_sets_bucket_window_and_retry_decision(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-03/backpressure")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("backpressure"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-backpressure",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    usage = ModelUsageV2(
        requests=1,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=0,
        source=ModelUsageSourceV2.REPORTED,
    )
    provider_eligible_at = NOW + timedelta(seconds=1)
    expected_eligible_at = NOW + timedelta(seconds=5)

    completed = models.complete_stage_backpressured(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=_holder("backpressure"),
        expected_lease_version=0,
        facade_receipt_ref=_ref("model-control-receipt", "429", version="v2"),
        kind=ProviderBackpressureKindV2.RATE_LIMITED,
        eligible_at=provider_eligible_at,
        usage=usage,
        audit=_audit(),
        idempotency_key="complete-backpressure",
    )

    assert completed.model_event.event_kind is WorkModelEventKindV2.BACKPRESSURED_RELEASED
    assert models.get_pool_head(pool.model_rate_pool_policy_id).blocked_until == (expected_eligible_at)
    retry = WorkControlService(store).get_retry_decision(unit.resolved_work_unit_id)
    assert retry.eligible_at == NOW


def test_pool_head_corruption_is_detected_from_immutable_events(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/corrupt")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)
    models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("corrupt"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-corrupt",
    )
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT record_json FROM model_rate_pool_heads WHERE model_rate_pool_policy_id = ?",
            (pool.model_rate_pool_policy_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(str(row[0]))
        payload["active_concurrency"] = 0
        connection.execute(
            "UPDATE model_rate_pool_heads SET record_json = ? WHERE model_rate_pool_policy_id = ?",
            (json.dumps(payload, sort_keys=True), pool.model_rate_pool_policy_id),
        )

    with pytest.raises(ImmutableResultError, match="model rate pool head"):
        models.get_pool_head(pool.model_rate_pool_policy_id)


@pytest.mark.parametrize(
    ("outcome", "expected_state", "expected_active"),
    [
        (
            ModelInvocationCancellationOutcomeV2.CANCELLED,
            WorkModelReservationStateV2.RELEASED,
            0,
        ),
        (
            ModelInvocationCancellationOutcomeV2.FAILED,
            WorkModelReservationStateV2.QUARANTINED,
            1,
        ),
    ],
)
def test_expiry_conservatively_charges_then_releases_or_quarantines_on_ack(
    tmp_path: Path,
    outcome: ModelInvocationCancellationOutcomeV2,
    expected_state: WorkModelReservationStateV2,
    expected_active: int,
) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / f"{outcome.value}.sqlite3", clock=clock)
    prepared = _prepared(store, job_id=f"job://r6-03/expiry-{outcome.value}")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder(f"expiry-{outcome.value}"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key=f"admit-expiry-{outcome.value}",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    clock.advance(30)

    expired = models.expire(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        audit=_audit(),
        idempotency_key=f"expire-{outcome.value}",
    )
    expired_replay = models.expire(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        audit=_audit(),
        idempotency_key=f"expire-{outcome.value}",
    )

    assert expired_replay == expired
    assert (
        models.get_reservation_head(admitted.reservation.work_model_reservation_id).state
        is WorkModelReservationStateV2.RELEASE_PENDING_ACK
    )
    pending_job = models.get_job_head(job_policy.job_model_policy_id)
    assert pending_job.reserved_tokens == 0
    assert pending_job.consumed_tokens == demand.token_allowance
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 1
    facade_request = models.to_facade_cancellation_request(expired.cancellation_request)
    facade_result = ModelInvocationCancellationResultV2.create(
        cancellation_request_ref=model_invocation_cancellation_request_ref(facade_request),
        outcome=outcome,
        completed_at=clock.value,
    )
    recorded = models.record_cancellation(
        request=expired.cancellation_request,
        facade_result=facade_result,
        audit=_audit(),
        idempotency_key=f"record-{outcome.value}",
    )
    recorded_replay = models.record_cancellation(
        request=expired.cancellation_request,
        facade_result=facade_result,
        audit=_audit(),
        idempotency_key=f"record-{outcome.value}",
    )

    assert recorded_replay == recorded
    assert models.get_reservation_head(admitted.reservation.work_model_reservation_id).state is expected_state
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == (expected_active)
    assert recorded.cancellation_result.outcome.value == outcome.value


def test_combined_admission_creates_one_lease_and_both_reservations(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/combined")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    model_demand = _demand(models, graph, unit, job_policy, pool, profile)
    resources, resource_policy, resource_demand = _resource_demand(
        store,
        graph,
        unit,
        process_capacity=1,
    )

    result = models.acquire_combined(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=model_demand,
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-combined",
    )
    replay = models.acquire_combined(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=model_demand,
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-combined",
    )

    assert replay == result
    assert result.lease is not None
    assert result.model_reservation is not None
    assert result.resource_reservation is not None
    assert result.model_reservation.work_lease_ref == result.resource_reservation.work_lease_ref
    assert len(store.list_stage_runs(job_id=graph.job_id)) == 1
    assert (
        resources.get_pool_head(resource_policy.resource_pool_policy_ref.object_id).active_capacity.processes
        == 1
    )


def test_combined_resource_denial_does_not_debit_model_or_create_lease(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/combined-denied")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    model_demand = _demand(models, graph, unit, job_policy, pool, profile)
    _resources, resource_policy, resource_demand = _resource_demand(
        store,
        graph,
        unit,
        process_capacity=0,
    )
    before = models.get_pool_head(pool.model_rate_pool_policy_id)

    result = models.acquire_combined(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=model_demand,
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined-denied"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="deny-combined",
    )

    assert result.model_decision is None
    assert result.resource_decision is not None
    assert result.resource_decision.outcome is ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND
    assert result.lease is None
    assert models.get_pool_head(pool.model_rate_pool_policy_id) == before
    assert store.list_stage_runs(job_id=graph.job_id) == ()


def test_combined_completion_releases_both_control_families_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/combined-complete")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    model_demand = _demand(models, graph, unit, job_policy, pool, profile)
    resources, resource_policy, resource_demand = _resource_demand(
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
        job_policy=job_policy,
        demand=model_demand,
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined-complete"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-combined-complete",
    )
    assert (
        admitted.lease is not None
        and admitted.model_reservation is not None
        and admitted.resource_reservation is not None
    )
    model_usage = ModelUsageV2(
        requests=1,
        input_tokens=50,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=100,
        source=ModelUsageSourceV2.REPORTED,
    )
    resource_usage = ResourceUsageV2(
        observed=NonModelResourceVectorV2(
            processes=1,
            renderers=0,
            network_requests=0,
            storage_bytes=0,
        )
    )
    original_release = ModelControlService._release_completed

    def fail_model_release(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("injected combined model release failure")

    monkeypatch.setattr(
        ModelControlService,
        "_release_completed",
        fail_model_release,
    )
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        models.complete_stage_combined(
            lease=admitted.lease,
            control_policy=control,
            model_reservation=admitted.model_reservation,
            resource_reservation=admitted.resource_reservation,
            holder_ref=_holder("combined-complete"),
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(
                _ref(
                    "semantic-residual-result",
                    "combined",
                    version="v2",
                ),
                _ref(
                    "model-control-receipt",
                    "combined",
                    version="v2",
                ),
            ),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            model_usage=model_usage,
            resource_usage=resource_usage,
            audit=_audit(),
            idempotency_key="complete-combined",
        )
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 1
    assert (
        resources.get_pool_head(resource_policy.resource_pool_policy_ref.object_id).active_capacity.processes
        == 1
    )
    assert store.get_stage_run(admitted.lease.stage_run_ref.object_id).status is (StageRunStatus.RUNNING)
    monkeypatch.setattr(
        ModelControlService,
        "_release_completed",
        original_release,
    )

    completed = models.complete_stage_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        holder_ref=_holder("combined-complete"),
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(
            _ref("semantic-residual-result", "combined", version="v2"),
            _ref("model-control-receipt", "combined", version="v2"),
        ),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        model_usage=model_usage,
        resource_usage=resource_usage,
        audit=_audit(),
        idempotency_key="complete-combined",
    )
    replay = models.complete_stage_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        holder_ref=_holder("combined-complete"),
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(
            _ref("semantic-residual-result", "combined", version="v2"),
            _ref("model-control-receipt", "combined", version="v2"),
        ),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        model_usage=model_usage,
        resource_usage=resource_usage,
        audit=_audit(),
        idempotency_key="complete-combined",
    )

    assert replay == completed
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 0
    assert (
        resources.get_pool_head(resource_policy.resource_pool_policy_ref.object_id).active_capacity.processes
        == 0
    )


@pytest.mark.asyncio
async def test_stage_run_free_artifact_group_releases_model_reservation(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/artifact")
    models, graph, _unit, _readiness, control, job_policy, pool, profile = prepared
    parent = next(unit for unit in graph.work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
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
        idempotency_key="model-artifact-fanout",
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
        idempotency_key="model-artifact-ready",
    )
    demand = models.create_demand(
        graph=graph,
        work_unit=group,
        job_policy=job_policy,
        pool_policy=pool,
        model_profile_ref=profile,
        operation_ref=_ref("attachment-execution-request", "artifact", version="v2"),
        model_execution_profile_ref=_ref("model-execution-profile", "runtime"),
        attempt=1,
        mode=ModelDemandModeV2.OPAQUE_RUNTIME_ENVELOPE,
        request_allowance=2,
        input_token_allowance=100,
        output_token_allowance=300,
        cache_token_allowance=100,
        concurrency_units=1,
        audit=_audit(),
    )
    holder = _holder("model-artifact")
    admitted = models.acquire(
        graph=graph,
        work_unit=group,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="model-artifact-admit",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    assert admitted.lease.stage_run_ref is None
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=_ControlledExecutionFacade(expected_parallelism=1),
        audit=_artifact_audit(),
    )
    result_refs = (
        artifact_execution_batch_ref(batch),
        *(artifact_execution_receipt_ref(receipt) for receipt in batch.receipts),
        _ref("model-control-receipt", "artifact", version="v2"),
    )

    completed = models.complete_artifact_group(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        event_kind=WorkLeaseEventKindV2.SUCCEEDED,
        result_refs=result_refs,
        failure=None,
        usage=ModelUsageV2(
            requests=1,
            input_tokens=50,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=100,
            source=ModelUsageSourceV2.REPORTED,
        ),
        audit=_audit(),
        idempotency_key="model-artifact-complete",
    )
    replay = models.complete_artifact_group(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        event_kind=WorkLeaseEventKindV2.SUCCEEDED,
        result_refs=result_refs,
        failure=None,
        usage=ModelUsageV2(
            requests=1,
            input_tokens=50,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=100,
            source=ModelUsageSourceV2.REPORTED,
        ),
        audit=_audit(),
        idempotency_key="model-artifact-complete",
    )

    assert replay == completed
    assert completed.lease_event.event_kind is WorkLeaseEventKindV2.SUCCEEDED
    assert completed.model_event.event_kind is WorkModelEventKindV2.COMPLETED_RELEASED
    assert store.list_stage_runs(job_id=graph.job_id) == ()
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 0


@pytest.mark.asyncio
async def test_stage_run_free_artifact_group_releases_combined_reservations(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/artifact-combined")
    models, graph, _unit, _readiness, control, job_policy, pool, profile = prepared
    parent = next(unit for unit in graph.work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact-combined",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
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
        idempotency_key="combined-artifact-fanout",
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
        idempotency_key="combined-artifact-ready",
    )
    resources, resource_policy, resource_demand = _resource_demand(
        store,
        graph,
        group,
        process_capacity=1,
    )
    model_demand = models.create_demand(
        graph=graph,
        work_unit=group,
        job_policy=job_policy,
        pool_policy=pool,
        model_profile_ref=profile,
        operation_ref=_ref(
            "attachment-execution-request",
            "artifact-combined",
            version="v2",
        ),
        model_execution_profile_ref=_ref("model-execution-profile", "runtime"),
        attempt=1,
        mode=ModelDemandModeV2.OPAQUE_RUNTIME_ENVELOPE,
        request_allowance=2,
        input_token_allowance=100,
        output_token_allowance=300,
        cache_token_allowance=100,
        concurrency_units=1,
        audit=_audit(),
    )
    holder = _holder("combined-artifact")
    admitted = models.acquire_combined(
        graph=graph,
        work_unit=group,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=model_demand,
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="combined-artifact-admit",
    )
    assert (
        admitted.lease is not None
        and admitted.model_reservation is not None
        and admitted.resource_reservation is not None
    )
    assert admitted.lease.stage_run_ref is None
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=_ControlledExecutionFacade(expected_parallelism=1),
        audit=_artifact_audit(),
    )
    result_refs = (
        artifact_execution_batch_ref(batch),
        *(artifact_execution_receipt_ref(receipt) for receipt in batch.receipts),
        _ref("model-control-receipt", "artifact-combined", version="v2"),
    )

    completed = models.complete_artifact_group_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        holder_ref=holder,
        expected_lease_version=0,
        event_kind=WorkLeaseEventKindV2.SUCCEEDED,
        result_refs=result_refs,
        failure=None,
        model_usage=ModelUsageV2(
            requests=1,
            input_tokens=50,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=100,
            source=ModelUsageSourceV2.REPORTED,
        ),
        resource_usage=ResourceUsageV2(
            observed=NonModelResourceVectorV2(
                processes=1,
                renderers=0,
                network_requests=0,
                storage_bytes=0,
            )
        ),
        audit=_audit(),
        idempotency_key="combined-artifact-complete",
    )
    replay = models.complete_artifact_group_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        holder_ref=holder,
        expected_lease_version=0,
        event_kind=WorkLeaseEventKindV2.SUCCEEDED,
        result_refs=result_refs,
        failure=None,
        model_usage=ModelUsageV2(
            requests=1,
            input_tokens=50,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=100,
            source=ModelUsageSourceV2.REPORTED,
        ),
        resource_usage=ResourceUsageV2(
            observed=NonModelResourceVectorV2(
                processes=1,
                renderers=0,
                network_requests=0,
                storage_bytes=0,
            )
        ),
        audit=_audit(),
        idempotency_key="combined-artifact-complete",
    )

    assert replay == completed
    assert completed.lease_event.event_kind is WorkLeaseEventKindV2.SUCCEEDED
    assert completed.model_event.work_lease_event_ref == (completed.resource_event.work_lease_event_ref)
    assert store.list_stage_runs(job_id=graph.job_id) == ()
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 0
    assert (
        resources.get_pool_head(resource_policy.resource_pool_policy_ref.object_id).active_capacity.processes
        == 0
    )


def test_combined_backpressure_releases_both_control_families_atomically(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-03/combined-backpressure")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    model_demand = _demand(models, graph, unit, job_policy, pool, profile)
    resources, resource_policy, resource_demand = _resource_demand(
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
        job_policy=job_policy,
        demand=model_demand,
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined-backpressure"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-combined-backpressure",
    )
    assert (
        admitted.lease is not None
        and admitted.model_reservation is not None
        and admitted.resource_reservation is not None
    )

    completed = models.complete_stage_combined_backpressured(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        holder_ref=_holder("combined-backpressure"),
        expected_lease_version=0,
        facade_receipt_ref=_ref("model-control-receipt", "combined-529", version="v2"),
        kind=ProviderBackpressureKindV2.OVERLOADED,
        eligible_at=NOW + timedelta(seconds=45),
        model_usage=ModelUsageV2(
            requests=1,
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=0,
            source=ModelUsageSourceV2.REPORTED,
        ),
        resource_usage=ResourceUsageV2(
            observed=NonModelResourceVectorV2(
                processes=1,
                renderers=0,
                network_requests=0,
                storage_bytes=0,
            )
        ),
        audit=_audit(),
        idempotency_key="complete-combined-backpressure",
    )
    replay = models.complete_stage_combined_backpressured(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        holder_ref=_holder("combined-backpressure"),
        expected_lease_version=0,
        facade_receipt_ref=_ref("model-control-receipt", "combined-529", version="v2"),
        kind=ProviderBackpressureKindV2.OVERLOADED,
        eligible_at=NOW + timedelta(seconds=45),
        model_usage=ModelUsageV2(
            requests=1,
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=0,
            source=ModelUsageSourceV2.REPORTED,
        ),
        resource_usage=ResourceUsageV2(
            observed=NonModelResourceVectorV2(
                processes=1,
                renderers=0,
                network_requests=0,
                storage_bytes=0,
            )
        ),
        audit=_audit(),
        idempotency_key="complete-combined-backpressure",
    )

    assert replay == completed
    assert completed.model_event.event_kind is WorkModelEventKindV2.BACKPRESSURED_RELEASED
    assert completed.lease_event.event_kind is WorkLeaseEventKindV2.RETRYABLE_FAILURE
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 0
    assert (
        resources.get_pool_head(resource_policy.resource_pool_policy_ref.object_id).active_capacity.processes
        == 0
    )


def test_usage_overrun_closes_attempt_and_charges_actual_usage(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/usage-overrun")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("usage-overrun"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-usage-overrun",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    usage = ModelUsageV2(
        requests=1,
        input_tokens=550,
        output_tokens=550,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=1100,
        source=ModelUsageSourceV2.REPORTED,
    )

    completed = models.complete_usage_overrun(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=_holder("usage-overrun"),
        expected_lease_version=0,
        facade_receipt_ref=_ref(
            "model-control-receipt",
            "usage-overrun",
            version="v2",
        ),
        usage=usage,
        audit=_audit(),
        idempotency_key="complete-usage-overrun",
    )
    replay = models.complete_usage_overrun(
        lease=admitted.lease,
        control_policy=control,
        reservation=admitted.reservation,
        holder_ref=_holder("usage-overrun"),
        expected_lease_version=0,
        facade_receipt_ref=_ref(
            "model-control-receipt",
            "usage-overrun",
            version="v2",
        ),
        usage=usage,
        audit=_audit(),
        idempotency_key="complete-usage-overrun",
    )

    assert replay == completed
    assert completed.lease_event.event_kind is WorkLeaseEventKindV2.TERMINAL_FAILURE
    assert completed.model_event.event_kind is WorkModelEventKindV2.USAGE_OVERRUN_RELEASED
    assert completed.lease_event.result_refs
    assert models.get_job_head(job_policy.job_model_policy_id).consumed_tokens == 1100
    pool_head = models.get_pool_head(pool.model_rate_pool_policy_id)
    assert pool_head.active_concurrency == 0
    assert pool_head.token_credits == -100 * RATE_CREDITS_PER_UNIT


def test_combined_usage_overrun_closes_and_charges_both_control_families(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/combined-overrun")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    resources, resource_policy, resource_demand = _resource_demand(
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
        job_policy=job_policy,
        demand=_demand(models, graph, unit, job_policy, pool, profile),
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined-overrun"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-combined-overrun",
    )
    assert (
        admitted.lease is not None
        and admitted.model_reservation is not None
        and admitted.resource_reservation is not None
    )
    model_usage = ModelUsageV2(
        requests=1,
        input_tokens=550,
        output_tokens=550,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=1100,
        source=ModelUsageSourceV2.REPORTED,
    )
    resource_usage = ResourceUsageV2(
        observed=NonModelResourceVectorV2(
            processes=1,
            renderers=0,
            network_requests=0,
            storage_bytes=0,
        )
    )

    completed = models.complete_usage_overrun_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        holder_ref=_holder("combined-overrun"),
        expected_lease_version=0,
        facade_receipt_ref=_ref(
            "model-control-receipt",
            "combined-overrun",
            version="v2",
        ),
        model_usage=model_usage,
        resource_usage=resource_usage,
        audit=_audit(),
        idempotency_key="complete-combined-overrun",
    )

    assert completed.lease_event.event_kind is WorkLeaseEventKindV2.TERMINAL_FAILURE
    assert completed.model_event.event_kind is WorkModelEventKindV2.USAGE_OVERRUN_RELEASED
    assert completed.model_event.work_lease_event_ref == (completed.resource_event.work_lease_event_ref)
    assert models.get_pool_head(pool.model_rate_pool_policy_id).token_credits == (
        -100 * RATE_CREDITS_PER_UNIT
    )
    assert (
        resources.get_pool_head(resource_policy.resource_pool_policy_ref.object_id).active_capacity.processes
        == 0
    )


def test_model_job_cancellation_is_atomic_and_replays(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/cancel")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    demand = _demand(models, graph, unit, job_policy, pool, profile)
    admitted = models.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control,
        job_policy=job_policy,
        demand=demand,
        holder_ref=_holder("cancel"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-model-cancel",
    )
    assert admitted.reservation is not None

    cancelled = models.cancel_job(
        graph=graph,
        control_policy=control,
        job_policy=job_policy,
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-job-model",
    )
    replay = models.cancel_job(
        graph=graph,
        control_policy=control,
        job_policy=job_policy,
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-job-model",
    )

    assert replay == cancelled
    assert len(cancelled.model_events) == 1
    assert len(cancelled.cancellation_requests) == 1
    assert cancelled.model_events[0].event_kind is WorkModelEventKindV2.RELEASE_PENDING_ACK
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 1
    assert store.get_job(graph.job_id).status is JobStatus.CANCELLED
    cancellation_request = cancelled.cancellation_requests[0]
    facade_request = models.to_facade_cancellation_request(cancellation_request)
    models.record_cancellation(
        request=cancellation_request,
        facade_result=ModelInvocationCancellationResultV2.create(
            cancellation_request_ref=(model_invocation_cancellation_request_ref(facade_request)),
            outcome=ModelInvocationCancellationOutcomeV2.CANCELLED,
            completed_at=NOW,
        ),
        audit=_audit(),
        idempotency_key="ack-job-model-cancel",
    )
    post_ack_replay = models.cancel_job(
        graph=graph,
        control_policy=control,
        job_policy=job_policy,
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-job-model",
    )
    assert post_ack_replay == cancelled


def test_combined_job_cancellation_closes_both_reservations_atomically(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/combined-cancel")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    resources, resource_policy, resource_demand = _resource_demand(
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
        job_policy=job_policy,
        demand=_demand(models, graph, unit, job_policy, pool, profile),
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined-cancel"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-combined-cancel",
    )
    assert admitted.model_reservation is not None and admitted.resource_reservation is not None

    cancelled = models.cancel_job_combined(
        graph=graph,
        control_policy=control,
        model_job_policy=job_policy,
        resource_job_policy=resource_policy,
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-job-combined",
    )
    replay = models.cancel_job_combined(
        graph=graph,
        control_policy=control,
        model_job_policy=job_policy,
        resource_job_policy=resource_policy,
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-job-combined",
    )

    assert replay == cancelled
    assert len(cancelled.model_events) == 1
    assert len(cancelled.resource_events) == 1
    assert len(cancelled.model_cancellation_requests) == 1
    assert len(cancelled.resource_termination_requests) == 1
    assert cancelled.model_events[0].work_lease_event_ref == (
        cancelled.resource_events[0].work_lease_event_ref
    )
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 1
    assert (
        resources.get_pool_head(resource_policy.resource_pool_policy_ref.object_id).active_capacity.processes
        == 1
    )


def test_combined_expiry_closes_one_lease_and_both_reservations(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-03/combined-expiry")
    models, graph, unit, readiness, control, job_policy, pool, profile = prepared
    resource_service, resource_policy, resource_demand = _resource_demand(
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
        job_policy=job_policy,
        demand=_demand(models, graph, unit, job_policy, pool, profile),
        resource_job_policy=resource_policy,
        resource_demand=resource_demand,
        holder_ref=_holder("combined-expiry"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-combined-expiry",
    )
    assert (
        admitted.lease is not None
        and admitted.model_reservation is not None
        and admitted.resource_reservation is not None
    )
    clock.advance(30)

    expired = models.expire_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        audit=_audit(),
        idempotency_key="expire-combined",
    )
    replay = models.expire_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        audit=_audit(),
        idempotency_key="expire-combined",
    )

    assert replay == expired
    assert expired.model_event.work_lease_event_ref == expired.resource_event.work_lease_event_ref
    assert expired.model_event.event_kind is WorkModelEventKindV2.RELEASE_PENDING_ACK
    assert expired.resource_event.event_kind.value == "RELEASE_PENDING_TERMINATION"
    assert expired.model_cancellation_request is not None
    assert expired.resource_termination_request is not None
    assert models.get_pool_head(pool.model_rate_pool_policy_id).active_concurrency == 1
    assert (
        resource_service.get_pool_head(
            resource_policy.resource_pool_policy_ref.object_id
        ).active_capacity.processes
        == 1
    )
    facade_request = models.to_facade_cancellation_request(expired.model_cancellation_request)
    models.record_cancellation(
        request=expired.model_cancellation_request,
        facade_result=ModelInvocationCancellationResultV2.create(
            cancellation_request_ref=(model_invocation_cancellation_request_ref(facade_request)),
            outcome=ModelInvocationCancellationOutcomeV2.CANCELLED,
            completed_at=clock.value,
        ),
        audit=_audit(),
        idempotency_key="ack-combined-expiry-model",
    )
    resource_service.record_termination(
        request=expired.resource_termination_request,
        facade_result=_facade_termination_result(
            expired.resource_termination_request,
            WorkResourceTerminationOutcomeV2.TERMINATED,
        ),
        audit=_audit(),
        idempotency_key="ack-combined-expiry-resource",
    )
    post_ack_replay = models.expire_combined(
        lease=admitted.lease,
        control_policy=control,
        model_reservation=admitted.model_reservation,
        resource_reservation=admitted.resource_reservation,
        audit=_audit(),
        idempotency_key="expire-combined",
    )
    assert post_ack_replay == expired


def test_shared_pool_concurrency_race_admits_exactly_one_job(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    first = _prepared(
        store,
        job_id="job://r6-03/race-a",
        request_burst=2,
        max_concurrent_requests=1,
    )
    second = _prepared(
        store,
        job_id="job://r6-03/race-b",
        pool_policy=first[6],
    )
    barrier = Barrier(2)

    def acquire(prepared, suffix: str):
        barrier.wait()
        return prepared[0].acquire(
            graph=prepared[1],
            work_unit=prepared[2],
            readiness_snapshot=prepared[3],
            control_policy=prepared[4],
            job_policy=prepared[5],
            demand=_demand(
                prepared[0],
                prepared[1],
                prepared[2],
                prepared[5],
                prepared[6],
                prepared[7],
            ),
            holder_ref=_holder(suffix),
            retry_decision=None,
            audit=_audit(),
            idempotency_key=f"race-{suffix}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            future.result()
            for future in (
                executor.submit(acquire, first, "a"),
                executor.submit(acquire, second, "b"),
            )
        )

    assert sorted(result.decision.outcome.value for result in results) == [
        "ADMITTED",
        "WAITING_CONCURRENCY",
    ]
    assert first[0].get_pool_head(first[6].model_rate_pool_policy_id).active_concurrency == 1


def test_model_admission_outbox_failure_rolls_back_every_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/fault")
    demand = _demand(
        prepared[0],
        prepared[1],
        prepared[2],
        prepared[5],
        prepared[6],
        prepared[7],
    )
    before = prepared[0].get_pool_head(prepared[6].model_rate_pool_policy_id)

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("injected model outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        prepared[0].acquire(
            graph=prepared[1],
            work_unit=prepared[2],
            readiness_snapshot=prepared[3],
            control_policy=prepared[4],
            job_policy=prepared[5],
            demand=demand,
            holder_ref=_holder("fault"),
            retry_decision=None,
            audit=_audit(),
            idempotency_key="model-fault",
        )

    assert prepared[0].get_pool_head(prepared[6].model_rate_pool_policy_id) == before
    assert store.list_stage_runs(job_id=prepared[1].job_id) == ()
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM work_model_reservations").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM model_admission_decisions").fetchone() == (0,)


def test_reopened_store_rebuilds_model_heads_from_immutable_events(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3", clock=_Clock())
    prepared = _prepared(store, job_id="job://r6-03/reopen")
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=_demand(
            prepared[0],
            prepared[1],
            prepared[2],
            prepared[5],
            prepared[6],
            prepared[7],
        ),
        holder_ref=_holder("reopen"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="model-reopen",
    )
    assert admitted.reservation is not None
    expected_pool = prepared[0].get_pool_head(prepared[6].model_rate_pool_policy_id)
    expected_job = prepared[0].get_job_head(prepared[5].job_model_policy_id)
    expected_reservation = prepared[0].get_reservation_head(admitted.reservation.work_model_reservation_id)

    reopened = ModelControlService(JobStore(store.path, clock=_Clock()))

    assert reopened.get_pool_head(prepared[6].model_rate_pool_policy_id) == expected_pool
    assert reopened.get_job_head(prepared[5].job_model_policy_id) == expected_job
    assert (
        reopened.get_reservation_head(admitted.reservation.work_model_reservation_id) == expected_reservation
    )
