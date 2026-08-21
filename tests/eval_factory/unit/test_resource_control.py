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
from test_work_control import _holder
from test_work_fanout import _planning_result

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.resource_v2 import (
    AttachmentResourceTerminationOutcomeV2,
    AttachmentResourceTerminationRequestV2,
    AttachmentResourceTerminationResultV2,
    attachment_resource_termination_request_ref,
)
from eval_factory.attachment_planning import ArtifactGroupExecutor
from eval_factory.contracts.attachment_v2 import (
    artifact_execution_batch_ref,
    artifact_execution_receipt_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import JobStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    WorkLeaseEventKindV2,
    WorkReadinessV2,
)
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
    ResourceAdmissionOutcomeV2,
    ResourceUsageV2,
    WorkResourceEventKindV2,
    WorkResourceReservationStateV2,
    WorkResourceTerminationOutcomeV2,
    WorkResourceTerminationRequestV2,
    work_resource_termination_result_v2_ref,
)
from eval_factory.orchestration import (
    ArtifactGroupFanoutCompiler,
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    WorkControlPolicyError,
    WorkControlService,
    WorkReadinessEvaluator,
)
from eval_factory.orchestration.resources import (
    ResourceControlPolicyError,
    ResourceControlService,
)

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/orchestration/r6-04-resource-pools-v1.json"
)


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
        object_id=f"{object_type}://r6-04/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _facade_termination_result(
    request: WorkResourceTerminationRequestV2,
    outcome: WorkResourceTerminationOutcomeV2,
) -> AttachmentResourceTerminationResultV2:
    facade_request = AttachmentResourceTerminationRequestV2.create(
        reservation_ref=_facade_ref(request.work_resource_reservation_ref),
        execution_handle_ref=_facade_ref(request.execution_handle_ref),
        fencing_token=request.fencing_token,
        reason=request.reason.value,
        requested_at=request.requested_at,
    )
    return AttachmentResourceTerminationResultV2.create(
        termination_request_ref=attachment_resource_termination_request_ref(facade_request),
        outcome=AttachmentResourceTerminationOutcomeV2(outcome.value),
        completed_at=NOW,
    )


class _TerminationFacade:
    def __init__(
        self,
        outcome: AttachmentResourceTerminationOutcomeV2,
    ) -> None:
        self.outcome = outcome
        self.requests: list[AttachmentResourceTerminationRequestV2] = []

    async def execute_with_resources(self, request, grant):
        raise AssertionError(f"execution is not expected: {request}, {grant}")

    async def terminate_resources(
        self,
        request: AttachmentResourceTerminationRequestV2,
    ) -> AttachmentResourceTerminationResultV2:
        self.requests.append(request)
        return AttachmentResourceTerminationResultV2.create(
            termination_request_ref=(attachment_resource_termination_request_ref(request)),
            outcome=self.outcome,
            completed_at=NOW,
        )


def _prepared(
    store: JobStore,
    *,
    job_id: str,
    pool_policy=None,
    process_capacity: int = 1,
):
    spec = _job_spec(job_id=job_id, idempotency_key=f"create-{job_id}")
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
    resources = ResourceControlService(store)
    if pool_policy is None:
        pool_policy = resources.bind_pool_policy(
            resource_domain_ref=_ref("resource-domain", "local"),
            capacity=NonModelResourceVectorV2(
                processes=process_capacity,
                renderers=1,
                network_requests=2,
                storage_bytes=4096,
            ),
            audit=_audit(),
            idempotency_key="pool-local",
        )
    job_policy = resources.bind_job_policy(
        graph=graph,
        pool_policy=pool_policy,
        audit=_audit(),
        idempotency_key=f"job-policy-{job_id}",
    )
    return (
        resources,
        graph,
        unit,
        readiness,
        control_policy,
        job_policy,
        pool_policy,
    )


def _demand(
    resources,
    graph,
    unit,
    job_policy,
    *,
    process_capacity: int = 1,
    process_budget: int = 1,
    storage_bytes: int = 1024,
):
    return resources.create_demand(
        graph=graph,
        work_unit=unit,
        job_policy=job_policy,
        execution_profile_ref=_ref("resource-execution-profile", "runtime"),
        attempt=1,
        capacity_units=NonModelResourceVectorV2(
            processes=process_capacity,
            renderers=0,
            network_requests=0,
            storage_bytes=storage_bytes,
        ),
        budget_allowance=NonModelResourceVectorV2(
            processes=process_budget,
            renderers=0,
            network_requests=0,
            storage_bytes=storage_bytes,
        ),
        termination_required=process_capacity > 0,
        audit=_audit(),
    )


def test_admission_atomically_creates_lease_stage_and_reservation(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    resources, graph, unit, readiness, control_policy, job_policy, pool = _prepared(
        store,
        job_id="job://r6-04/admitted",
    )
    demand = _demand(resources, graph, unit, job_policy)
    holder = _holder("resource")

    result = resources.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control_policy,
        job_policy=job_policy,
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-04",
    )
    replay = resources.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        control_policy=control_policy,
        job_policy=job_policy,
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-r6-04",
    )

    assert replay == result
    assert result.decision.outcome is ResourceAdmissionOutcomeV2.ADMITTED
    assert result.lease is not None
    assert result.reservation is not None
    assert result.lease.stage_run_ref is not None
    assert store.get_stage_run(result.lease.stage_run_ref.object_id).status is StageRunStatus.RUNNING
    assert resources.get_pool_head(pool.resource_pool_policy_id).active_capacity.processes == 1
    assert resources.get_job_head(job_policy.job_resource_policy_id).reserved_budget.processes == 1
    assert (
        resources.get_reservation_head(result.reservation.work_resource_reservation_id).state
        is WorkResourceReservationStateV2.ACTIVE
    )
    grant = resources.to_facade_grant(
        result.reservation,
        operation_ref=_facade_ref(
            _ref(
                "attachment-execution-request",
                "resource",
                version="v2",
            )
        ),
    )
    assert grant.fencing_token == result.lease.fencing_token
    assert grant.storage_byte_limit == demand.budget_allowance.storage_bytes

    with pytest.raises(WorkControlPolicyError, match="resource-aware"):
        WorkControlService(store).acquire(
            graph=graph,
            work_unit=unit,
            readiness_snapshot=readiness,
            policy=control_policy,
            holder_ref=holder,
            retry_decision=None,
            audit=_audit(),
            idempotency_key="legacy-bypass",
        )


def test_shared_pool_capacity_wait_creates_no_lease_or_stage_run(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    first = _prepared(store, job_id="job://r6-04/first")
    second = _prepared(
        store,
        job_id="job://r6-04/second",
        pool_policy=first[-1],
    )
    first_demand = _demand(first[0], first[1], first[2], first[5])
    second_demand = _demand(second[0], second[1], second[2], second[5])
    first[0].acquire(
        graph=first[1],
        work_unit=first[2],
        readiness_snapshot=first[3],
        control_policy=first[4],
        job_policy=first[5],
        demand=first_demand,
        holder_ref=_holder("first"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-first",
    )

    waiting = second[0].acquire(
        graph=second[1],
        work_unit=second[2],
        readiness_snapshot=second[3],
        control_policy=second[4],
        job_policy=second[5],
        demand=second_demand,
        holder_ref=_holder("second"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="wait-second",
    )

    assert waiting.decision.outcome is ResourceAdmissionOutcomeV2.WAITING_CAPACITY
    assert waiting.lease is None
    assert waiting.reservation is None
    assert store.list_stage_runs(job_id=second[1].job_id) == ()


def test_capacity_wait_can_be_readmitted_after_pool_head_changes(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    first = _prepared(store, job_id="job://r6-04/readmit-owner")
    second = _prepared(
        store,
        job_id="job://r6-04/readmit-waiter",
        pool_policy=first[-1],
    )
    first_demand = _demand(first[0], first[1], first[2], first[5])
    second_demand = _demand(second[0], second[1], second[2], second[5])
    first_holder = _holder("readmit-owner")
    owner = first[0].acquire(
        graph=first[1],
        work_unit=first[2],
        readiness_snapshot=first[3],
        control_policy=first[4],
        job_policy=first[5],
        demand=first_demand,
        holder_ref=first_holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="readmit-owner-admit",
    )
    assert owner.lease is not None and owner.reservation is not None

    waiting_head = second[0].get_pool_head(first[-1].resource_pool_policy_id)
    waiting_key = f"readmit-waiter:{waiting_head.row_version}"
    waiting = second[0].acquire(
        graph=second[1],
        work_unit=second[2],
        readiness_snapshot=second[3],
        control_policy=second[4],
        job_policy=second[5],
        demand=second_demand,
        holder_ref=_holder("readmit-waiter"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key=waiting_key,
    )
    replay = second[0].acquire(
        graph=second[1],
        work_unit=second[2],
        readiness_snapshot=second[3],
        control_policy=second[4],
        job_policy=second[5],
        demand=second_demand,
        holder_ref=_holder("readmit-waiter"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key=waiting_key,
    )
    assert waiting == replay
    assert waiting.decision.outcome is ResourceAdmissionOutcomeV2.WAITING_CAPACITY
    assert waiting.lease is None and waiting.reservation is None

    first[0].complete_stage(
        lease=owner.lease,
        control_policy=first[4],
        reservation=owner.reservation,
        holder_ref=first_holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        usage=ResourceUsageV2(
            observed=NonModelResourceVectorV2(
                processes=1,
                renderers=0,
                network_requests=0,
                storage_bytes=1024,
            )
        ),
        audit=_audit(),
        idempotency_key="readmit-owner-complete",
    )

    changed_head = second[0].get_pool_head(first[-1].resource_pool_policy_id)
    assert changed_head.row_version > waiting_head.row_version
    admitted = second[0].acquire(
        graph=second[1],
        work_unit=second[2],
        readiness_snapshot=second[3],
        control_policy=second[4],
        job_policy=second[5],
        demand=second_demand,
        holder_ref=_holder("readmit-waiter"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key=f"readmit-waiter:{changed_head.row_version}",
    )

    assert admitted.decision.outcome is ResourceAdmissionOutcomeV2.ADMITTED
    assert admitted.lease is not None
    assert admitted.reservation is not None


def test_resource_denial_does_not_bypass_cancelled_job_admission(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    first = _prepared(store, job_id="job://r6-04/capacity-owner")
    cancelled = _prepared(
        store,
        job_id="job://r6-04/cancelled-waiter",
        pool_policy=first[-1],
    )
    first_demand = _demand(first[0], first[1], first[2], first[5])
    cancelled_demand = _demand(
        cancelled[0],
        cancelled[1],
        cancelled[2],
        cancelled[5],
    )
    first[0].acquire(
        graph=first[1],
        work_unit=first[2],
        readiness_snapshot=first[3],
        control_policy=first[4],
        job_policy=first[5],
        demand=first_demand,
        holder_ref=_holder("capacity-owner"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-capacity-owner",
    )
    cancelled[0].cancel_job(
        graph=cancelled[1],
        control_policy=cancelled[4],
        job_policy=cancelled[5],
        reason_code="cancel-before-admission",
        audit=_audit(),
        idempotency_key="cancel-before-admission",
    )
    with sqlite3.connect(store.path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM resource_admission_decisions").fetchone()[0]

    with pytest.raises(WorkControlPolicyError, match="cancelled"):
        cancelled[0].acquire(
            graph=cancelled[1],
            work_unit=cancelled[2],
            readiness_snapshot=cancelled[3],
            control_policy=cancelled[4],
            job_policy=cancelled[5],
            demand=cancelled_demand,
            holder_ref=_holder("cancelled-waiter"),
            retry_decision=None,
            audit=_audit(),
            idempotency_key="deny-cancelled-waiter",
        )

    with sqlite3.connect(store.path) as connection:
        after = connection.execute("SELECT COUNT(*) FROM resource_admission_decisions").fetchone()[0]
    assert after == before


def test_budget_exhaustion_is_terminal_without_attempt(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/exhausted")
    demand = _demand(
        prepared[0],
        prepared[1],
        prepared[2],
        prepared[5],
        process_budget=3,
    )

    exhausted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=_holder("exhausted"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="exhaust-budget",
    )

    assert exhausted.decision.outcome is ResourceAdmissionOutcomeV2.BUDGET_EXHAUSTED
    assert exhausted.lease is None
    assert store.list_stage_runs(job_id=prepared[1].job_id) == ()
    with pytest.raises(
        ResourceControlPolicyError,
        match="terminal admission",
    ):
        prepared[0].acquire(
            graph=prepared[1],
            work_unit=prepared[2],
            readiness_snapshot=prepared[3],
            control_policy=prepared[4],
            job_policy=prepared[5],
            demand=demand,
            holder_ref=_holder("exhausted"),
            retry_decision=None,
            audit=_audit(),
            idempotency_key="exhaust-budget-again",
        )
    downstream = next(
        unit
        for unit in prepared[1].work_units
        if unit.depends_on_work_unit_refs
        and unit.depends_on_work_unit_refs[0].object_id == prepared[2].resolved_work_unit_id
    )
    blocked = WorkReadinessEvaluator().evaluate(
        graph=prepared[1],
        work_unit=downstream,
        stage_completions=(),
        item_records=(),
        resource_admission_decisions=(exhausted.decision,),
        work_resource_demands=(demand,),
        audit=_audit(),
    )
    assert blocked.readiness is WorkReadinessV2.BLOCKED_DEPENDENCY
    store.record_work_readiness(
        blocked,
        idempotency_key="resource-exhaustion-readiness",
    )


def test_completion_charges_observed_usage_and_releases_capacity(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/complete")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    holder = _holder("complete")
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-complete",
    )
    assert admitted.lease is not None and admitted.reservation is not None

    completed = prepared[0].complete_stage(
        lease=admitted.lease,
        control_policy=prepared[4],
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        usage=ResourceUsageV2(
            observed=NonModelResourceVectorV2(
                processes=1,
                renderers=0,
                network_requests=0,
                storage_bytes=128,
            )
        ),
        audit=_audit(),
        idempotency_key="complete-resources",
    )
    replay = prepared[0].complete_stage(
        lease=admitted.lease,
        control_policy=prepared[4],
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        usage=ResourceUsageV2(
            observed=NonModelResourceVectorV2(
                processes=1,
                renderers=0,
                network_requests=0,
                storage_bytes=128,
            )
        ),
        audit=_audit(),
        idempotency_key="complete-resources",
    )

    assert replay == completed
    with pytest.raises(IdempotencyConflictError, match="different request"):
        prepared[0].complete_stage(
            lease=admitted.lease,
            control_policy=prepared[4],
            reservation=admitted.reservation,
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(_ref("stage-output", "changed", version="record/v1"),),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            usage=ResourceUsageV2(
                observed=NonModelResourceVectorV2(
                    processes=1,
                    renderers=0,
                    network_requests=0,
                    storage_bytes=128,
                )
            ),
            audit=_audit(),
            idempotency_key="complete-resources",
        )
    assert completed.resource_event.event_kind is WorkResourceEventKindV2.COMPLETED_RELEASED
    pool_head = prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id)
    job_head = prepared[0].get_job_head(prepared[5].job_resource_policy_id)
    assert pool_head.active_capacity.processes == 0
    assert pool_head.active_capacity.storage_bytes == 128
    assert job_head.reserved_budget == NonModelResourceVectorV2.zero()
    assert job_head.consumed_budget.processes == 1
    assert job_head.consumed_budget.storage_bytes == 128


def test_expiry_holds_process_capacity_until_termination_acknowledgement(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-04/expire")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=_holder("expire"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-expire",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    clock.advance(30)

    expired = prepared[0].expire(
        lease=admitted.lease,
        control_policy=prepared[4],
        reservation=admitted.reservation,
        audit=_audit(),
        idempotency_key="expire-resources",
    )

    assert expired.termination_request is not None
    assert expired.resource_event.event_kind is (WorkResourceEventKindV2.RELEASE_PENDING_TERMINATION)
    assert prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id).active_capacity.processes == 1
    assert (
        prepared[0].get_reservation_head(admitted.reservation.work_resource_reservation_id).state
        is WorkResourceReservationStateV2.RELEASE_PENDING_TERMINATION
    )

    terminated = prepared[0].record_termination(
        request=expired.termination_request,
        facade_result=_facade_termination_result(
            expired.termination_request,
            WorkResourceTerminationOutcomeV2.TERMINATED,
        ),
        audit=_audit(),
        idempotency_key="terminate-expired",
    )
    termination_replay = prepared[0].record_termination(
        request=expired.termination_request,
        facade_result=_facade_termination_result(
            expired.termination_request,
            WorkResourceTerminationOutcomeV2.TERMINATED,
        ),
        audit=_audit(),
        idempotency_key="terminate-expired",
    )
    assert termination_replay == terminated
    assert terminated.resource_event.event_kind is (WorkResourceEventKindV2.TERMINATED_RELEASED)
    assert terminated.resource_event.resource_termination_result_ref == (
        work_resource_termination_result_v2_ref(terminated.termination_result)
    )
    pool_head = prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id)
    assert pool_head.active_capacity.processes == 0
    assert pool_head.active_capacity.storage_bytes == 1024


def test_non_process_expiry_releases_capacity_without_termination(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-04/network-expire")
    demand = prepared[0].create_demand(
        graph=prepared[1],
        work_unit=prepared[2],
        job_policy=prepared[5],
        execution_profile_ref=_ref(
            "resource-execution-profile",
            "network",
        ),
        attempt=1,
        capacity_units=NonModelResourceVectorV2(
            processes=0,
            renderers=0,
            network_requests=1,
            storage_bytes=0,
        ),
        budget_allowance=NonModelResourceVectorV2(
            processes=0,
            renderers=0,
            network_requests=1,
            storage_bytes=0,
        ),
        termination_required=False,
        audit=_audit(),
    )
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=_holder("network-expire"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-network-expire",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    clock.advance(30)

    expired = prepared[0].expire(
        lease=admitted.lease,
        control_policy=prepared[4],
        reservation=admitted.reservation,
        audit=_audit(),
        idempotency_key="expire-network-resource",
    )

    assert expired.termination_request is None
    assert expired.resource_event.event_kind is (WorkResourceEventKindV2.COMPLETED_RELEASED)
    assert (
        prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id).active_capacity.network_requests == 0
    )


def test_job_cancellation_creates_one_termination_request_and_replays(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/cancel")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=_holder("cancel"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-cancel",
    )
    assert admitted.reservation is not None

    cancelled = prepared[0].cancel_job(
        graph=prepared[1],
        control_policy=prepared[4],
        job_policy=prepared[5],
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-job-resources",
    )
    replay = prepared[0].cancel_job(
        graph=prepared[1],
        control_policy=prepared[4],
        job_policy=prepared[5],
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-job-resources",
    )

    assert replay == cancelled
    assert len(cancelled.termination_requests) == 1
    assert len(cancelled.resource_events) == 1
    assert cancelled.resource_events[0].event_kind is (WorkResourceEventKindV2.RELEASE_PENDING_TERMINATION)


@pytest.mark.asyncio
async def test_resource_service_executes_and_binds_facade_termination(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-04/facade-termination")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=_holder("facade-termination"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-facade-termination",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    clock.advance(30)
    expired = prepared[0].expire(
        lease=admitted.lease,
        control_policy=prepared[4],
        reservation=admitted.reservation,
        audit=_audit(),
        idempotency_key="expire-facade-termination",
    )
    assert expired.termination_request is not None
    facade = _TerminationFacade(AttachmentResourceTerminationOutcomeV2.TERMINATED)

    terminated = await prepared[0].terminate(
        request=expired.termination_request,
        facade=facade,
        audit=_audit(),
        idempotency_key="record-facade-termination",
    )

    assert len(facade.requests) == 1
    assert terminated.termination_result.facade_termination_result_ref.object_type == (
        "attachment-resource-termination-result"
    )
    assert terminated.resource_event.resource_termination_result_ref == (
        work_resource_termination_result_v2_ref(terminated.termination_result)
    )


def test_shared_pool_race_admits_exactly_one_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    first = _prepared(store, job_id="job://r6-04/race-first")
    second = _prepared(
        store,
        job_id="job://r6-04/race-second",
        pool_policy=first[-1],
    )
    first_demand = _demand(first[0], first[1], first[2], first[5])
    second_demand = _demand(second[0], second[1], second[2], second[5])
    barrier = Barrier(2)

    def admit(prepared, demand, suffix):
        barrier.wait(timeout=5)
        return prepared[0].acquire(
            graph=prepared[1],
            work_unit=prepared[2],
            readiness_snapshot=prepared[3],
            control_policy=prepared[4],
            job_policy=prepared[5],
            demand=demand,
            holder_ref=_holder(suffix),
            retry_decision=None,
            audit=_audit(),
            idempotency_key=f"race-{suffix}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            future.result()
            for future in (
                executor.submit(admit, first, first_demand, "first"),
                executor.submit(admit, second, second_demand, "second"),
            )
        )

    assert {result.decision.outcome for result in results} == {
        ResourceAdmissionOutcomeV2.ADMITTED,
        ResourceAdmissionOutcomeV2.WAITING_CAPACITY,
    }
    assert sum(result.lease is not None for result in results) == 1
    assert (
        len(store.list_stage_runs(job_id=first[1].job_id))
        + len(store.list_stage_runs(job_id=second[1].job_id))
        == 1
    )


def test_pool_head_rebuild_uses_one_read_snapshot_during_writer_commit(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/snapshot-reader")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    holder = _holder("snapshot-reader")
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="snapshot-reader-admit",
    )
    assert admitted.lease is not None and admitted.reservation is not None

    with store._read_snapshot() as connection:
        before = prepared[0]._get_pool_head(
            connection,
            prepared[-1].resource_pool_policy_id,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                prepared[0].complete_stage,
                lease=admitted.lease,
                control_policy=prepared[4],
                reservation=admitted.reservation,
                holder_ref=holder,
                expected_lease_version=0,
                status=StageRunStatus.SUCCEEDED,
                output_refs=(),
                failure=None,
                checkpoint_ref=None,
                metrics_ref=None,
                usage=ResourceUsageV2(
                    observed=NonModelResourceVectorV2(
                        processes=1,
                        renderers=0,
                        network_requests=0,
                        storage_bytes=1024,
                    )
                ),
                audit=_audit(),
                idempotency_key="snapshot-reader-complete",
            ).result(timeout=10)
        observed = prepared[0]._get_pool_head(
            connection,
            prepared[-1].resource_pool_policy_id,
        )

    assert observed == before
    current = prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id)
    assert current.row_version > before.row_version
    assert current.active_capacity.processes == 0


def test_pool_head_corruption_is_rejected_by_immutable_rebuild(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/corrupt")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=_holder("corrupt"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-corrupt",
    )
    head = prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id)
    corrupted = head.model_copy(update={"active_capacity": NonModelResourceVectorV2.zero()})
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE resource_pool_heads
            SET record_json = ?
            WHERE resource_pool_policy_id = ?
            """,
            (
                store._record_json(corrupted),
                prepared[-1].resource_pool_policy_id,
            ),
        )

    with pytest.raises(ImmutableResultError, match="rebuild"):
        prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id)


def test_admission_outbox_failure_rolls_back_lease_and_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/rollback")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])

    def fail_outbox(*args, **kwargs):
        del args, kwargs
        raise sqlite3.OperationalError("injected resource outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        prepared[0].acquire(
            graph=prepared[1],
            work_unit=prepared[2],
            readiness_snapshot=prepared[3],
            control_policy=prepared[4],
            job_policy=prepared[5],
            demand=demand,
            holder_ref=_holder("rollback"),
            retry_decision=None,
            audit=_audit(),
            idempotency_key="admit-rollback",
        )

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM resource_admission_decisions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM work_resource_reservations").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM work_leases").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM stage_runs").fetchone()[0] == 0


def test_policy_binding_replays_and_conflicts(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/policy-replay")
    resources = prepared[0]
    pool = prepared[-1]
    job_policy = prepared[5]

    replayed_pool = resources.bind_pool_policy(
        resource_domain_ref=pool.resource_domain_ref,
        capacity=pool.capacity,
        audit=_audit(),
        idempotency_key="pool-local",
    )
    replayed_job = resources.bind_job_policy(
        graph=prepared[1],
        pool_policy=pool,
        audit=_audit(),
        idempotency_key="job-policy-job://r6-04/policy-replay",
    )

    assert replayed_pool == pool
    assert replayed_job == job_policy
    with pytest.raises(IdempotencyConflictError, match="different request"):
        resources.bind_pool_policy(
            resource_domain_ref=pool.resource_domain_ref,
            capacity=pool.capacity.model_copy(
                update={"network_requests": pool.capacity.network_requests + 1}
            ),
            audit=_audit(),
            idempotency_key="pool-local",
        )


def test_usage_overrun_and_legacy_completion_leave_reservation_active(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/overrun")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    holder = _holder("overrun")
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-overrun",
    )
    assert admitted.lease is not None and admitted.reservation is not None

    with pytest.raises(ResourceControlPolicyError, match="exceeds"):
        prepared[0].complete_stage(
            lease=admitted.lease,
            control_policy=prepared[4],
            reservation=admitted.reservation,
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            usage=ResourceUsageV2(
                observed=demand.budget_allowance.model_copy(
                    update={"storage_bytes": demand.budget_allowance.storage_bytes + 1}
                )
            ),
            audit=_audit(),
            idempotency_key="complete-overrun",
        )
    with pytest.raises(WorkControlPolicyError, match="resource-aware"):
        WorkControlService(store).complete_stage(
            lease=admitted.lease,
            policy=prepared[4],
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key="legacy-complete-overrun",
        )
    assert (
        prepared[0].get_reservation_head(admitted.reservation.work_resource_reservation_id).state
        is WorkResourceReservationStateV2.ACTIVE
    )


def test_failed_termination_quarantines_capacity(tmp_path: Path) -> None:
    clock = _Clock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    prepared = _prepared(store, job_id="job://r6-04/quarantine")
    demand = _demand(prepared[0], prepared[1], prepared[2], prepared[5])
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=prepared[2],
        readiness_snapshot=prepared[3],
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=_holder("quarantine"),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="admit-quarantine",
    )
    assert admitted.lease is not None and admitted.reservation is not None
    clock.advance(30)
    expired = prepared[0].expire(
        lease=admitted.lease,
        control_policy=prepared[4],
        reservation=admitted.reservation,
        audit=_audit(),
        idempotency_key="expire-quarantine",
    )
    assert expired.termination_request is not None

    failed = prepared[0].record_termination(
        request=expired.termination_request,
        facade_result=_facade_termination_result(
            expired.termination_request,
            WorkResourceTerminationOutcomeV2.FAILED,
        ),
        audit=_audit(),
        idempotency_key="fail-termination",
    )

    assert failed.resource_event.event_kind is WorkResourceEventKindV2.QUARANTINED
    assert (
        prepared[0].get_reservation_head(admitted.reservation.work_resource_reservation_id).state
        is WorkResourceReservationStateV2.QUARANTINED
    )
    assert prepared[0].get_pool_head(prepared[-1].resource_pool_policy_id).active_capacity.processes == 1


def test_resource_admission_gold_is_executable(tmp_path: Path) -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    for case in payload["cases"]:
        store = JobStore(tmp_path / f"{case['case_id']}.sqlite3")
        prepared = _prepared(
            store,
            job_id=f"job://r6-04/gold/{case['case_id']}",
        )
        demand = _demand(
            prepared[0],
            prepared[1],
            prepared[2],
            prepared[5],
            process_capacity=case["process_capacity"],
            process_budget=case["process_budget"],
            storage_bytes=case["storage_bytes"],
        )
        result = prepared[0].acquire(
            graph=prepared[1],
            work_unit=prepared[2],
            readiness_snapshot=prepared[3],
            control_policy=prepared[4],
            job_policy=prepared[5],
            demand=demand,
            holder_ref=_holder(case["case_id"]),
            retry_decision=None,
            audit=_audit(),
            idempotency_key=f"gold-{case['case_id']}",
        )

        assert result.decision.outcome.value == case["expected_outcome"]


@pytest.mark.asyncio
async def test_stage_run_free_artifact_group_releases_resource_reservation(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "factory.sqlite3")
    prepared = _prepared(store, job_id="job://r6-04/artifact")
    parent = next(unit for unit in prepared[1].work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
    )
    planning = _planning_result(execution_plan)
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=prepared[1],
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=planning,
        audit=_audit(),
    )
    store.create_artifact_group_fanout(
        fanout,
        idempotency_key="resource-artifact-fanout",
    )
    group = fanout.group_work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=prepared[1],
        work_unit=group,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        audit=_audit(),
    )
    store.record_work_readiness(
        readiness,
        idempotency_key="resource-artifact-ready",
    )
    demand = prepared[0].create_demand(
        graph=prepared[1],
        work_unit=group,
        job_policy=prepared[5],
        execution_profile_ref=_ref("resource-execution-profile", "provider"),
        attempt=1,
        capacity_units=NonModelResourceVectorV2.zero(),
        budget_allowance=NonModelResourceVectorV2.zero(),
        termination_required=False,
        audit=_audit(),
    )
    holder = _holder("artifact-resource")
    admitted = prepared[0].acquire(
        graph=prepared[1],
        work_unit=group,
        readiness_snapshot=readiness,
        control_policy=prepared[4],
        job_policy=prepared[5],
        demand=demand,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="resource-artifact-admit",
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
    )

    completed = prepared[0].complete_artifact_group(
        lease=admitted.lease,
        control_policy=prepared[4],
        reservation=admitted.reservation,
        holder_ref=holder,
        expected_lease_version=0,
        event_kind=WorkLeaseEventKindV2.SUCCEEDED,
        result_refs=result_refs,
        failure=None,
        usage=ResourceUsageV2(observed=NonModelResourceVectorV2.zero()),
        audit=_audit(),
        idempotency_key="resource-artifact-complete",
    )

    assert completed.lease_event.event_kind is WorkLeaseEventKindV2.SUCCEEDED
    assert completed.resource_event.event_kind is (WorkResourceEventKindV2.COMPLETED_RELEASED)
    assert store.list_stage_runs(job_id=prepared[1].job_id) == ()
